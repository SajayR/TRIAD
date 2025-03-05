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
        Cross-batch approach with Dynamic Relevance Pooling: only tokens with
        statistically significant matches contribute to the final similarity.
        
        audio_feats: (B, Na, D)
        visual_feats: (B, Nv, D)
        
        Returns:
            clip_sims: (B, B) aggregated similarity with relevance pooling
            token_sims: (B, B, Na, Nv) raw token-level similarities
        """
        B = audio_feats.shape[0]
        # Expand to shape (B, B, Na, D) and (B, B, Nv, D)
        af = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        
        # Compute token-level similarities: (B, B, Na, Nv)
        token_sims = torch.matmul(af, vf.transpose(2, 3)) / self.temperature
        
        # Max over visual dimension => (B, B, Na)
        max_sims = torch.max(token_sims, dim=3)[0]
        
        # Compute z-scores to find statistically significant tokens
        mean_sim = max_sims.mean(dim=2, keepdim=True)  # (B, B, 1)
        std_sim = torch.std(max_sims, dim=2, keepdim=True).clamp(min=1e-5)  # (B, B, 1)
        z_scores = (max_sims - mean_sim) / std_sim  # (B, B, Na)
        
        # Create relevance mask: only above-average tokens contribute
        relevance_mask = (z_scores > 0).float()  # (B, B, Na)
        
        # Compute weighted average using the relevance mask
        weighted_sum = (max_sims * relevance_mask).sum(dim=2)  # (B, B)
        token_count = relevance_mask.sum(dim=2).clamp(min=1)  # (B, B)
        clip_sims = weighted_sum / token_count  # (B, B)
        
        # Compute statistics for monitoring
        stats = {
            'av_relevant_tokens_pct': relevance_mask.mean().item() * 100,
            'av_z_score_min': z_scores.min().item(),
            'av_z_score_max': z_scores.max().item(),
            'av_z_score_mean': z_scores.mean().item(),
            # Focus on diagonal (positive pairs)
            'av_pos_relevant_pct': torch.stack([relevance_mask[i,i].mean() for i in range(B)]).mean().item() * 100
        }
        
        return clip_sims, token_sims, stats

    def compute_temporal_smoothness_loss(self, token_sims):
        """
        Computes temporal smoothness loss for audio-visual attention maps.
        
        Args:
            token_sims: Tensor of shape (B, B, Na, Nv)
            
        Returns:
            Scalar temporal smoothness loss
        """
        B = token_sims.shape[0]
        diagonal_sims = torch.stack([token_sims[i, i] for i in range(B)])  # Shape: (B, Na, Nv)
        temporal_diffs = diagonal_sims[:, 1:] - diagonal_sims[:, :-1]  # Shape: (B, Na-1, Nv)
        smoothness_loss = torch.mean(temporal_diffs ** 2)
        return smoothness_loss

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

        l_smooth = self.compute_temporal_smoothness_loss(token_sims)
        reg_loss = (8.0 * l_cal + 0.15 * l_nonneg)# + 0.1 * l_smooth)
        return reg_loss

    def compute_contrastive_loss_av(self, clip_sims, token_sims, stats):
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
        return contrastive_loss + reg_loss, stats

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
            clip_sims, token_sims, stats = self.compute_all_similarities_av(audio_feats, visual_feats)
            return self.compute_contrastive_loss_av(clip_sims, token_sims, stats)
        else:
            # shape => (B, Na, Nv)
            token_sims = self.compute_similarity_matrix(audio_feats, visual_feats)
            return token_sims


    ######################################################
    #               TEXT-VISUAL PATH
    ######################################################
    def compute_all_similarities_tv(self, text_feats, visual_feats, attention_mask):
        """
        Cross-batch approach with Dynamic Relevance Pooling for text-visual.
        Combines attention mask with statistical relevance.
        
        text_feats:   (B, Nt, D)
        visual_feats: (B, Nv, D)
        attention_mask: (B, Nt)
        
        Returns:
            clip_sims: (B, B)
            token_sims: (B, B, Nt, Nv)
        """
        B = text_feats.shape[0]

        tf = text_feats.unsqueeze(1).expand(-1, B, -1, -1)  # (B, B, Nt, D)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)  # (B, B, Nv, D)
        
        # Token-level similarities
        token_sims = torch.matmul(tf, vf.transpose(2, 3)) / self.temperature  # (B, B, Nt, Nv)
        
        # Max over visual dimension
        max_sims = torch.max(token_sims, dim=3)[0]  # (B, B, Nt)
        
        # Expand attention mask to match shape
        attn_mask = attention_mask.unsqueeze(1).float().expand(-1, B, -1)  # (B, B, Nt)
        
        # Compute z-scores (only for valid tokens)
        masked_max_sims = max_sims * attn_mask
        sum_sims = masked_max_sims.sum(dim=2, keepdim=True)
        valid_count = attn_mask.sum(dim=2, keepdim=True).clamp(min=1e-7)
        mean_sim = sum_sims / valid_count
        
        squared_diff = ((masked_max_sims - mean_sim) * attn_mask) ** 2
        variance = squared_diff.sum(dim=2, keepdim=True) / valid_count.clamp(min=1)
        std_sim = torch.sqrt(variance).clamp(min=1e-5)
        
        z_scores = (max_sims - mean_sim) / std_sim
        
        # Combined mask: token must be valid (attention_mask) AND statistically relevant (z_score > 0)
        relevance_mask = attn_mask * (z_scores > 0).float()
        
        # Ensure we have at least one relevant token per pair
        has_relevant = relevance_mask.sum(dim=2) > 0
        fallback_mask = ~has_relevant.unsqueeze(2) * attn_mask
        final_mask = relevance_mask + fallback_mask
        
        # Compute weighted average
        weighted_sum = (max_sims * final_mask).sum(dim=2)
        token_count = final_mask.sum(dim=2).clamp(min=1)
        clip_sims = weighted_sum / token_count
        
        # Compute statistics for monitoring
        # Focusing only on valid tokens (where attention_mask is 1)
        valid_tokens = attn_mask.sum().item()
        relevance_tokens = relevance_mask.sum().item()
        
        stats = {
            'tv_relevant_tokens_pct': (relevance_tokens / valid_tokens) * 100 if valid_tokens > 0 else 0,
            'tv_z_score_min': z_scores[attn_mask > 0].min().item() if torch.any(attn_mask > 0) else 0,
            'tv_z_score_max': z_scores[attn_mask > 0].max().item() if torch.any(attn_mask > 0) else 0,
            'tv_z_score_mean': (z_scores * attn_mask).sum().item() / valid_tokens if valid_tokens > 0 else 0,
            # Focus on diagonal (positive pairs)
            'tv_pos_relevant_pct': torch.stack([(relevance_mask[i,i] * attn_mask[i,i]).sum() / 
                                            attn_mask[i,i].sum().clamp(min=1) for i in range(B)]).mean().item() * 100
        }
        
        return clip_sims, token_sims, stats

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
        reg_loss = 8.0 * l_nonneg# + self.patch_sparsity_weight * loss_sparsity
        return reg_loss

    def compute_contrastive_loss_tv(self, clip_sims, token_sims, stats):
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
        return total_loss, stats

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
            clip_sims, token_sims, stats = self.compute_all_similarities_tv(text_feats, visual_feats, attention_mask)
            return self.compute_contrastive_loss_tv(clip_sims, token_sims, stats)
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
#################################################################
#                        Quick Test
#################################################################
#################################################################
#                        Quick Test
#################################################################
if __name__ == "__main__":
    print("Testing MultiModalModel with Dynamic Relevance Pooling...")
    model = MultiModalModel(
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1
    )
    
    # Test batch with varied content
    batch_size = 4
    dummy_frames = torch.randn(batch_size, 3, 224, 224)
    dummy_audio = torch.randn(batch_size, 16000)
    dummy_texts = [
        "a man riding a bicycle near a mountain",
        "a cat sleeping on a comfortable bed",
        "the sunset over the ocean horizon",
        "birds flying in the clear blue sky"
    ]
    
    # 1. Basic forward pass test
    print("\n=== Testing forward passes ===")
    model.train()
    av_loss = model.forward_audio_visual(dummy_frames, dummy_audio)
    tv_loss = model.forward_text_visual(dummy_frames, dummy_texts)
    print(f"Audio-Visual loss: {av_loss.item():.4f}")
    print(f"Text-Visual loss: {tv_loss.item():.4f}")
    
    # 2. Gradient flow test
    print("\n=== Testing gradient flow ===")
    model.zero_grad()
    combined_loss = av_loss + tv_loss
    combined_loss.backward()
    
    # Check gradients on parameters, not modules
    print("\nChecking gradients on key parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                print(f"WARNING: {name} has no gradient!")
            else:
                has_nan = torch.isnan(param.grad).any().item()
                max_grad = param.grad.abs().max().item()
                if "projection" in name or name == "temperature":
                    print(f"{name}: grad exists=True, has NaN={has_nan}, max={max_grad:.5f}")
    
    # 3. Inspect intermediate values
    print("\n=== Inspecting intermediate values ===")
    model.eval()
    with torch.no_grad():
        # Audio-visual path
        visual_feats = model.visual_embedder(dummy_frames)
        audio_feats = model.audio_embedder(dummy_audio)
        
        # Get raw token similarities
        token_sims_av = torch.matmul(audio_feats.unsqueeze(1).expand(-1, batch_size, -1, -1), 
                              visual_feats.unsqueeze(0).expand(batch_size, -1, -1, -1).transpose(2, 3)) / model.temperature
        
        # Get max similarities
        max_sims_av = torch.max(token_sims_av, dim=3)[0]
        
        # Calculate z-scores
        mean_sim_av = max_sims_av.mean(dim=2, keepdim=True)
        std_sim_av = torch.std(max_sims_av, dim=2, keepdim=True).clamp(min=1e-5)
        z_scores_av = (max_sims_av - mean_sim_av) / std_sim_av
        
        # Check relevance mask
        relevance_mask_av = (z_scores_av > 0).float()
        relevant_tokens_pct = relevance_mask_av.mean(dim=2).mean().item() * 100
        
        print(f"Audio path: {relevant_tokens_pct:.1f}% of tokens are considered relevant")
        print(f"Z-score min: {z_scores_av.min().item():.2f}, max: {z_scores_av.max().item():.2f}")
        
        # Text-visual path
        text_feats, attention_mask = model.text_embedder(dummy_texts)
        
        # Run a forward pass but extract intermediate results
        clip_sims, token_sims_tv = model.compute_all_similarities_tv(text_feats, visual_feats, attention_mask)
        
        print(f"Text-visual clip similarities shape: {clip_sims.shape}")
        print(f"Diagonal similarities (positive pairs): {clip_sims.diag()}")
        
    # 4. Compare with original pooling
    print("\n=== Comparing with original pooling ===")
    # Original audio-visual pooling (max-mean)
    original_clip_sims_av = torch.mean(max_sims_av, dim=2)
    
    # Get diagonal values (positive pairs)
    original_diag_av = original_clip_sims_av.diag()
    new_diag_av = clip_sims.diag()
    
    print(f"Audio-Visual original diag: {original_diag_av}")
    print(f"Text-Visual with relevance diag: {new_diag_av}")
    print(f"Difference: {(new_diag_av - original_diag_av)}")
    
    print("\nMultiModalModel test completed.")