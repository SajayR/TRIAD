# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from transformers import (
    HubertModel, 
    AutoProcessor, 
    AutoTokenizer, 
    AutoModel
)
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from PIL import Image
#################################################################
#                   Audio Embedder
#################################################################
class AudioEmbedder(nn.Module):
    """
    Pre-trained HuBERT to extract audio features from raw audio (16kHz).
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, hubert_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")  
        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.projection = nn.Linear(self.hubert.config.hidden_size, embedding_dim)
        
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio_input: (B, T) raw audio waveform at 16kHz
            
        Returns:
            audio_feats: (B, Na, D) 
                B = batch size
                Na = number of audio tokens (T/320 for Hubert)
                D = embedding_dim
        """
        if len(audio_input.shape) == 3:  # shape: [B, 1, T]
            audio_input = audio_input.squeeze(0)  # squeeze first dim to get [B, T]
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        ).input_values.squeeze(0)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        
        hubert_output = self.hubert(inputs).last_hidden_state  # (B, T', hidden_size)
        
        audio_feats = self.projection(hubert_output)  # (B, T', D)
        
        return audio_feats


#################################################################
#                   Text Embedder
#################################################################
class TextEmbedder(nn.Module):
    """
    pre-trained BERT-like model to extract text features.
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        print("Using text model: ", model_name)
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, text_list):
        """
        Args:
            text_list: List[str], batch of text inputs
            
        Returns:
            text_feats: (B, Nt, D)
            attention_mask: (B, Nt)
        """
        inputs = self.tokenizer(
            text_list, 
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=128,
            return_tensors="pt"
        )
        device = next(self.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = self.encoder(**inputs)  # (B, Nt, hidden_size)
        hidden_states = outputs.last_hidden_state
        text_feats = self.projection(hidden_states)  # (B, Nt, D)
        
        return text_feats, inputs["attention_mask"]


#################################################################
#                   Visual Embedder
#################################################################
class ViTEmbedder(nn.Module):
    """
    DINOv2to extract patch embeddings from an image.
    Then projects to a common dimension with a linear layer.
    """
    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=512, dropout_prob=0.1):
        super().__init__()
        self.model = torch.hub.load(model_name, arch)
        print("Using DINOv2 model: ", arch)
        self.projection = nn.Linear(self.model.embed_dim, embedding_dim)
        self.dropout = nn.Dropout(p=dropout_prob)

        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """
        #print(f"x shape: {x.shape}")
        if len(x.shape) == 5:  # shape: [1, 1, 3, 224, 224]
            x = x.squeeze(0)  # get [1, 3, 224, 224]
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        patches = self.model.get_intermediate_layers(x, n=1)[0]  
        feats = self.projection(patches)
        feats = self.dropout(feats)
        
        return feats


#################################################################
#                   Unified MultiModalModel
#################################################################
class MultiModalModel(nn.Module):
    def __init__(
        self, 
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1
    ):
        super().__init__()

        self.audio_embedder = AudioEmbedder(embedding_dim=512, hubert_name=audio_model_name)
        self.text_embedder  = TextEmbedder(embedding_dim=512, model_name=text_model_name)
        self.visual_embedder = ViTEmbedder(arch='dinov2_vitb14',
                                           embedding_dim=512,
                                           dropout_prob=visual_dropout_prob)

        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.patch_sparsity_threshold = patch_sparsity_threshold
        self.patch_sparsity_weight = patch_sparsity_weight

    ######################################################
    #               Shared Utilities
    ######################################################
    def compute_similarity_matrix(self, feats1, feats2):
        """
        Generic token-level dot-product similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """
        # feats1 = F.normalize(feats1, dim=-1)
        # feats2 = F.normalize(feats2, dim=-1)
        sim = torch.bmm(feats1, feats2.transpose(1, 2))
        return sim / self.temperature

    ######################################################
    #               AUDIO-VISUAL PATH
    ######################################################
    def compute_all_similarities_av(self, audio_feats, visual_feats):
        """
        Cross-batch approach: compute pairwise similarities for all 
        (audio_i, visual_j) in the batch.
        
        audio_feats: (B, Na, D)
        visual_feats: (B, Nv, D)
        
        Returns:
            clip_sims: (B, B)  for the aggregated similarity
            token_sims: (B, B, Na, Nv) raw token-level sims
        """
        B = audio_feats.shape[0]
        # Expand to shape (B, B, Na, D) and (B, B, Nv, D)
        af = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        # dot product => (B, B, Na, Nv)
        token_sims = torch.matmul(af, vf.transpose(2, 3)) / self.temperature
        # max over visual dimension => (B, B, Na)
        max_sims = torch.max(token_sims, dim=3)[0]
        # mean over audio dimension => (B, B)
        clip_sims = torch.mean(max_sims, dim=2)
        return clip_sims, token_sims

    def compute_regularization_losses_av(self, token_sims):
        """
        Based on the old AudioVisualModel, includes:
          1. Non-negative pressure (we clamp negative sims in [-20, 0])
          2. Temperature constraints (optional if you're using a trainable temperature)
        """
        # 1) Non-negative pressure
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims ** 4)

        # 2) Temperature calibration
        #    force it between [1,4] or sumn man idk.
        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) 
                               - torch.log(self.temperature), min=0) ** 4
        temp_high = torch.clamp(torch.log(self.temperature) 
                                - torch.log(torch.tensor(4.0, device=token_sims.device)), min=0) ** 4
        l_cal = temp_low + temp_high
        reg_loss = (8.0 * l_cal + 0.15 * l_nonneg)
        return reg_loss

    def compute_contrastive_loss_av(self, clip_sims, token_sims):
        """
        InfoNCE-style cross-entropy for audio<->visual + regularizations.
        """
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)

        log_prob_a2v = F.log_softmax(clip_sims, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(B), labels]

        log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(B), labels]

        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        reg_loss = self.compute_regularization_losses_av(token_sims)
        return contrastive_loss + reg_loss

    def forward_audio_visual(self, frames, audio):
        """
        audio: (B, T) raw waveform
        frames: (B, 3, 224, 224)
        
        If training: returns scalar loss
        If eval: returns token_sims
        """
        #print(audio.shape)
        visual_feats = self.visual_embedder(frames)      # (B, Nv, D)
        audio_feats = self.audio_embedder(audio)         # (B, Na, D)
        if self.training:
            clip_sims, token_sims = self.compute_all_similarities_av(audio_feats, visual_feats)
            return self.compute_contrastive_loss_av(clip_sims, token_sims)
        else:
            # shape => (B, Na, Nv)
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats)
            return token_sims


    ######################################################
    #               TEXT-VISUAL PATH
    ######################################################
    def compute_all_similarities_tv(self, text_feats, visual_feats, attention_mask):
        """
        cross-batch approach: (text_i, visual_j)
        text_feats:   (B, Nt, D)
        visual_feats: (B, Nv, D)
        attention_mask: (B, Nt)
        
        Returns:
            clip_sims: (B, B)
            token_sims: (B, B, Nt, Nv)
        """
        B = text_feats.shape[0]

        tf = text_feats.unsqueeze(1).expand(-1, B, -1, -1)  # (B, B, Nt, D)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1) # (B, B, Nv, D)
        # token-level similarity => (B, B, Nt, Nv)
        token_sims = torch.matmul(tf, vf.transpose(2, 3)) / self.temperature
        # max over visual dimension => (B, B, Nt)
        max_sims = torch.max(token_sims, dim=3)[0]
        # we need masked mean over Nt
        # attn_mask_expanded => (B, 1, Nt) => (B, B, Nt)
        mask = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        masked_sum = (max_sims * mask).sum(dim=2)  # (B, B)
        valid_tokens = mask.sum(dim=2).clamp(min=1e-7) 
        clip_sims = masked_sum / valid_tokens

        return clip_sims, token_sims

    def compute_regularization_losses_tv(self, token_sims):
        """
        1) negative sims near zero
        2) patch usage sparsity on positive pairs
        """
        # (B, B, Nt, Nv)
        B = token_sims.shape[0]
        # 1) negative clamp
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims**2)
        # 2) patch usage sparsity (for the diagonal pairs only)
        positive_sims = []
        for i in range(B):
            # shape (Nt, Nv) from token_sims[i, i]
            positive_sims.append(token_sims[i, i])
        if len(positive_sims) == 0:
            return 0.15 * l_nonneg
        positive_sims = torch.stack(positive_sims, dim=0)  # (B, Nt, Nv)
        # softmax over patches => (B, Nt, Nv)
        patch_probs = F.softmax(positive_sims, dim=-1)
        # fraction usage per patch => sum over Nt, then / Nt => (B, Nv)
        patch_fraction = patch_probs.sum(dim=1) / patch_probs.shape[1]
        # penalize if fraction > threshold
        excess = F.relu(patch_fraction - self.patch_sparsity_threshold)  # (B, Nv)
        loss_sparsity = (excess ** 2).mean()
        reg_loss = 0.15 * l_nonneg + self.patch_sparsity_weight * loss_sparsity
        return reg_loss

    def compute_contrastive_loss_tv(self, clip_sims, token_sims):
        """
        Standard cross-entropy for text<->visual plus the reg losses.
        """
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)

        # text->visual
        log_prob_t2v = F.log_softmax(clip_sims, dim=1)
        losses_t2v = -log_prob_t2v[torch.arange(B), labels]

        # visual->text
        log_prob_v2t = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2t = -log_prob_v2t[torch.arange(B), labels]

        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2
        reg_loss = self.compute_regularization_losses_tv(token_sims)

        total_loss = contrastive_loss + reg_loss
        return total_loss

    def forward_text_visual(self, frames, text_list):
        """
        frames: (B, 3, 224, 224)
        text_list: list of strings length B

        If training: return scalar contrastive loss
        else: return (sim_matrix, attention_mask)
        """
        visual_feats = self.visual_embedder(frames)               # (B, Nv, D)
        text_feats, attention_mask = self.text_embedder(text_list) # (B, Nt, D), (B, Nt)

        if self.training:
            clip_sims, token_sims = self.compute_all_similarities_tv(text_feats, visual_feats, attention_mask)
            return self.compute_contrastive_loss_tv(clip_sims, token_sims)
        else:
            sim_matrix = self.compute_similarity_matrix(text_feats, visual_feats)  # (B, Nt, Nv)
            return sim_matrix, attention_mask

    def forward(self, frames=None, audio=None, text_list=None):
        assert frames is not None or audio is not None or text_list is not None, "At least one modality must be provided"
        # we need to conver the image into the correct format and shit
        assert frames is not str, "Frames should be a path to an image"
        if frames is not None:
            image = Image.open(frames).convert('RGB')
            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
            ])
            frames = transform(image)
        embeddings = {}
        if frames is not None:
            embeddings['visual_feats'] = self.visual_embedder(frames)
        if audio is not None:
            embeddings['audio_feats'] = self.audio_embedder(audio)
        if text_list is not None:
            embeddings['text_feats'], _ = self.text_embedder(text_list)

        # if two or more modalities are present, we compute the similarity matrix 
        if frames is not None and text_list is not None:
            embeddings['vis_text_sim_matrix'] = self.compute_similarity_matrix(embeddings['text_feats'], embeddings['visual_feats'])
        if audio is not None and frames is not None:
            embeddings['vis_audio_sim_matrix'] = self.compute_similarity_matrix(embeddings['audio_feats'], embeddings['visual_feats'])
        if audio is not None and text_list is not None:
            embeddings['text_audio_sim_matrix'] = self.compute_similarity_matrix(embeddings['text_feats'], embeddings['audio_feats'])
        return embeddings

#################################################################
#                        Quick Test
#################################################################
if __name__ == "__main__":
    print("Testing MultiModalModel with random inputs...")
    model = MultiModalModel(
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="answerdotai/ModernBERT-base",
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1
    )
    batch_size = 2
    dummy_frames = torch.randn(batch_size, 3, 224, 224)      # image frames
    dummy_audio  = torch.randn(batch_size, 16000)            # 1 sec of 16kHz
    dummy_texts  = ["a man riding a bicycle", "a cat on a bed"]
    model.train()
    av_loss = model.forward_audio_visual(dummy_frames, dummy_audio)
    print(f"Audio-Visual loss: {av_loss.item():.4f}")
    tv_loss = model.forward_text_visual(dummy_frames, dummy_texts)
    print(f"Text-Visual loss: {tv_loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        av_sims = model.forward_audio_visual(dummy_frames, dummy_audio)
        print(f"Audio-Visual similarities shape: {av_sims.shape}")  
        # expected => (B, Na, Nv)
        tv_sims, tv_mask = model.forward_text_visual(dummy_frames, dummy_texts)
        print(f"Text-Visual similarities shape: {tv_sims.shape}, mask: {tv_mask.shape}")
        # expected => (B, Nt, Nv), (B, Nt)
    
    print("MultiModalModel test completed.")
