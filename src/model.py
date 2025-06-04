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
import math
warnings.filterwarnings("ignore")
import torchvision.transforms as transforms
from PIL import Image
from torch.cuda.amp import autocast

from peft import (
    LoraConfig, 
    get_peft_model,
    TaskType,
)
class AudioEmbedder(nn.Module):
    """
    Pre-trained HuBERT to extract audio features from raw audio (16kHz).
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, hubert_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")  
        self.hubert = HubertModel.from_pretrained(hubert_name)

        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
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
        if len(audio_input.shape) == 3:
            audio_input = audio_input.squeeze(0)
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True
        ).input_values.squeeze(0)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        
        hubert_output = self.hubert(inputs).last_hidden_state
        
        audio_feats = self.projection2(self.layer_norm(self.projection1(hubert_output)))
        
        return audio_feats

class TextEmbedder(nn.Module):
    """
    pre-trained BERT-like model to extract text features.
    Projects them down to a desired embedding dimension.
    """
    def __init__(self, embedding_dim=512, model_name="answerdotai/ModernBERT-base"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection1 = nn.Linear(self.encoder.config.hidden_size, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        print("Using text model: ", model_name)
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
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

        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state
        text_feats = self.projection2(self.layer_norm(self.projection1(hidden_states)))
        
        return text_feats, inputs["attention_mask"]

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
        self.projection1 = nn.Linear(self.model.embed_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)

        self.patch_dropout_rate = dropout_prob
        self.patch_dropout = self.patch_dropout
        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def patch_dropout(self, x, drop_rate):
        """
        Actually removes patch embeddings during training
        Args:
            x: patch embeddings of shape (B, N, D) where N is number of patches
            drop_rate: probability of dropping a patch
        """
        if not self.training or drop_rate == 0:
            return x
            
        B, N, D = x.shape
        dtype = x.dtype
            

        keep_mask = torch.bernoulli(
            torch.ones(B, N, device=x.device, dtype=dtype) * (1 - drop_rate)
        ).bool()
        

        output_tensors = []
        

        for i in range(B):

            kept_tokens = x[i][keep_mask[i]]
            output_tensors.append(kept_tokens)
        

        max_len = max(tensor.size(0) for tensor in output_tensors)
        padded_outputs = []
        
        for tensor in output_tensors:
            if tensor.size(0) < max_len:
                padding = torch.zeros(max_len - tensor.size(0), D, dtype=dtype, device=x.device)
                padded_outputs.append(torch.cat([tensor, padding], dim=0))
            else:
                padded_outputs.append(tensor)
        

        x = torch.stack(padded_outputs, dim=0)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """

        if len(x.shape) == 5:
            x = x.squeeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        patches = self.model.get_intermediate_layers(x, n=1)[0]  
        
        feats = self.projection2(self.layer_norm(self.projection1(patches)))

        feats = self.patch_dropout(feats, self.patch_dropout_rate)

        return feats

class ViTLoRAEmbedder(nn.Module):
    """
    DINOv2 with LoRA adapters for parameter-efficient fine-tuning.
    Applies LoRA to both attention and MLP layers of the transformer.
    Projects output embeddings to a common dimension with a linear layer.
    """
    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=512, dropout_prob=0.1, lora_rank=8, lora_alpha=16):
        super().__init__()
        

        self.model = torch.hub.load(model_name, arch)
        print(f"Using DINOv2 model with LoRA adapters: {arch}")
    
        

        for param in self.model.parameters():
            param.requires_grad = False
            

        lora_target_modules = [

            "attn.qkv",
            "attn.proj",

        ]
        

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=0.0,
            fan_in_fan_out=True,
            bias="none",
            modules_to_save=None
        )
        

        self.model = get_peft_model(self.model, lora_config)
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"ViTLoRAEmbedder - Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}% of total)")

        self.projection1 = nn.Linear(self.model.embed_dim, 512)
        self.layer_norm = nn.LayerNorm(512)
        self.projection2 = nn.Linear(512, embedding_dim)
        self.patch_dropout_rate = dropout_prob
        self.patch_dropout = self.patch_dropout
        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.model.base_model.parameters():
            param.requires_grad = False
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True

    def patch_dropout(self, x, drop_rate):
        """
        Actually removes patch embeddings during training
        Args:
            x: patch embeddings of shape (B, N, D) where N is number of patches
            drop_rate: probability of dropping a patch
        """
        if not self.training or drop_rate == 0:
            return x
            
        B, N, D = x.shape
        dtype = x.dtype
            

        keep_mask = torch.bernoulli(
            torch.ones(B, N, device=x.device, dtype=dtype) * (1 - drop_rate)
        ).bool()
        

        output_tensors = []
        

        for i in range(B):

            kept_tokens = x[i][keep_mask[i]]
            output_tensors.append(kept_tokens)
        

        max_len = max(tensor.size(0) for tensor in output_tensors)
        padded_outputs = []
        
        for tensor in output_tensors:
            if tensor.size(0) < max_len:
                padding = torch.zeros(max_len - tensor.size(0), D, dtype=dtype, device=x.device)
                padded_outputs.append(torch.cat([tensor, padding], dim=0))
            else:
                padded_outputs.append(tensor)
        

        x = torch.stack(padded_outputs, dim=0)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W), e.g. (B,3,224,224) image batch
        Returns:
            visual_feats: (B, Nv, D)
                Nv = number of visual tokens
                D  = embedding_dim
        """
        if len(x.shape) == 5:
            x = x.squeeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
            

        patches = self.model.get_intermediate_layers(x, n=1)[0]
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        feats = self.patch_dropout(feats, self.patch_dropout_rate)
        
        return feats

class MultiModalModel(nn.Module):
    def __init__(
        self, 
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=1.2,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1,
        use_amp=True
    ):
        super().__init__()

        self.audio_embedder = AudioEmbedder(embedding_dim=512, hubert_name=audio_model_name)
        self.text_embedder  = TextEmbedder(embedding_dim=512, model_name=text_model_name)
        self.visual_embedder = ViTLoRAEmbedder(arch='dinov2_vitb14_reg', embedding_dim=512, dropout_prob=visual_dropout_prob)

        self.temperature = nn.Parameter(torch.tensor(temperature))

        self.patch_sparsity_threshold = patch_sparsity_threshold
        self.patch_sparsity_weight = patch_sparsity_weight
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16

    def compute_similarity_matrix(self, feats1, feats2):
        """
        Generic token-level dot-product similarity between feats1 and feats2.
        feats1: (B, N1, D)
        feats2: (B, N2, D)
        Returns sim: (B, N1, N2)
        """ 

        feats1 = F.normalize(feats1, dim=-1)
        feats2 = F.normalize(feats2, dim=-1)

        with torch.cuda.amp.autocast(enabled=False):
            sim = torch.bmm(feats1, feats2.transpose(1, 2))
            return sim * self.temperature

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

        af = audio_feats.unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)

        token_sims = torch.matmul(af, vf.transpose(2, 3)) * self.temperature

        max_sims = torch.max(token_sims, dim=3)[0]

        clip_sims = torch.mean(max_sims, dim=2)
        return clip_sims, token_sims

    def compute_temporal_smoothness_loss(self, token_sims):
        """
        Computes temporal smoothness loss for audio-visual attention maps.
        
        Args:
            token_sims: Tensor of shape (B, B, Na, Nv)
            
        Returns:
            Scalar temporal smoothness loss
        """
        B = token_sims.shape[0]
        diagonal_sims = torch.stack([token_sims[i, i] for i in range(B)])
        temporal_diffs = diagonal_sims[:, 1:] - diagonal_sims[:, :-1]
        smoothness_loss = torch.mean(temporal_diffs ** 2)
        return smoothness_loss

    def compute_regularization_losses_av(self, token_sims):
        """
        Based on the old AudioVisualModel, includes:
          1. Non-negative pressure (we clamp negative sims in [-20, 0])
          2. Temperature constraints (optional if you're using a trainable temperature)
        """

        neg_sims = torch.clamp(token_sims, min=-60, max=0)
        l_nonneg = torch.mean(neg_sims ** 2)

        temp_low = torch.clamp(torch.log(torch.tensor(1.0, device=token_sims.device)) 
                               - torch.log(self.temperature), min=0) ** 2
        temp_high = torch.clamp(torch.log(self.temperature) 
                                - torch.log(torch.tensor(2.0, device=token_sims.device)), min=0) ** 2
        l_cal = temp_low

        l_smooth = self.compute_temporal_smoothness_loss(token_sims)
        reg_loss = (20 * l_cal + 0.15 * l_nonneg + 0.01 * l_smooth)
        return reg_loss, 0.01*l_smooth

    def compute_contrastive_loss_av(self, clip_sims, token_sims):
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)
        

        pos_sims = torch.diagonal(clip_sims)
        

        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(0)
        neg_sims = clip_sims[mask]
        

        pos_sim_mean = pos_sims.mean().item()
        pos_sim_std = pos_sims.std().item()
        neg_sim_mean = neg_sims.mean().item()
        neg_sim_std = neg_sims.std().item()
        hardest_negative = neg_sims.max().item()
        

        separation = pos_sim_mean - neg_sim_mean
        

        log_prob_a2v = F.log_softmax(clip_sims, dim=1)
        losses_a2v = -log_prob_a2v[torch.arange(B), labels]

        log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2a = -log_prob_v2a[torch.arange(B), labels]

        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2
        reg_loss, l_smooth = self.compute_regularization_losses_av(token_sims)
        

        similarity_stats = {
            "av_pos_sim_mean": pos_sim_mean,
            "av_pos_sim_std": pos_sim_std,
            "av_neg_sim_mean": neg_sim_mean,
            "av_neg_sim_std": neg_sim_std,
            "av_separation": separation,
            "av_hardest_negative": hardest_negative
        }
        
        return contrastive_loss + reg_loss, contrastive_loss, reg_loss, l_smooth, similarity_stats

    def forward_audio_visual(self, frames, audio):
        """
        audio: (B, T) raw waveform
        frames: (B, 3, 224, 224)
        
        If training: returns scalar loss
        If eval: returns token_sims
        """

        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            visual_feats = self.visual_embedder(frames)
            audio_feats = self.audio_embedder(audio)
        with torch.cuda.amp.autocast(enabled=False):
            clip_sims, token_sims = self.compute_all_similarities_av(audio_feats, visual_feats)
            return self.compute_contrastive_loss_av(clip_sims, token_sims)

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
        tf = text_feats.unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)

        token_sims = torch.matmul(tf, vf.transpose(2, 3)) * self.temperature

        max_sims = torch.max(token_sims, dim=3)[0]

        mask = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        masked_sum = (max_sims * mask).sum(dim=2)
        valid_tokens = mask.sum(dim=2).clamp(min=1e-7) 
        clip_sims = masked_sum / valid_tokens

        return clip_sims, token_sims

    def compute_regularization_losses_tv(self, token_sims):
        """
        1) negative sims near zero
        2) patch usage sparsity on positive pairs
        """

        B = token_sims.shape[0]

        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims**2)

        positive_sims = []
        for i in range(B):

            positive_sims.append(token_sims[i, i])
        if len(positive_sims) == 0:
            return 0.15 * l_nonneg
        positive_sims = torch.stack(positive_sims, dim=0)

        patch_probs = F.softmax(positive_sims, dim=-1)

        patch_fraction = patch_probs.sum(dim=1) / patch_probs.shape[1]

        excess = F.relu(patch_fraction - self.patch_sparsity_threshold)
        loss_sparsity = (excess ** 2).mean()
        reg_loss = 0.15 * l_nonneg + self.patch_sparsity_weight * loss_sparsity
        return reg_loss

    def compute_contrastive_loss_tv(self, clip_sims, token_sims):
        """
        Standard cross-entropy for text<->visual plus the reg losses,
        now with similarity statistics tracking.
        """
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)
        

        pos_sims = torch.diagonal(clip_sims)
        

        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(0)
        neg_sims = clip_sims[mask]
        

        pos_sim_mean = pos_sims.mean().item()
        pos_sim_std = pos_sims.std().item()
        neg_sim_mean = neg_sims.mean().item()
        neg_sim_std = neg_sims.std().item()
        hardest_negative = neg_sims.max().item()
        

        separation = pos_sim_mean - neg_sim_mean
        


        log_prob_t2v = F.log_softmax(clip_sims, dim=1)
        losses_t2v = -log_prob_t2v[torch.arange(B), labels]

        log_prob_v2t = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2t = -log_prob_v2t[torch.arange(B), labels]

        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2
        reg_loss = self.compute_regularization_losses_tv(token_sims)

        total_loss = contrastive_loss + reg_loss
        

        similarity_stats = {
            "tv_pos_sim_mean": pos_sim_mean,
            "tv_pos_sim_std": pos_sim_std,
            "tv_neg_sim_mean": neg_sim_mean,
            "tv_neg_sim_std": neg_sim_std,
            "tv_separation": separation,
            "tv_hardest_negative": hardest_negative
        }
        
        return total_loss, similarity_stats

    def forward_text_visual(self, frames, text_list):
        """
        frames: (B, 3, 224, 224)
        text_list: list of strings length B

        If training: return scalar contrastive loss
        else: return (sim_matrix, attention_mask)
        """
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):   
            visual_feats = self.visual_embedder(frames)
            text_feats, attention_mask = self.text_embedder(text_list)
        with torch.cuda.amp.autocast(enabled=False):
            clip_sims, token_sims = self.compute_all_similarities_tv(text_feats, visual_feats, attention_mask)
            return self.compute_contrastive_loss_tv(clip_sims, token_sims)

    def forward(self, frames=None, audio=None, text_list=None):
        assert frames is not None or audio is not None or text_list is not None, "At least one modality must be provided"

        assert frames is not str, "Frames Cross-modal retrieval using 1000 evaluation videos from the PlacesAudio and AudioSet validation datasets. DenseAV dramatically outperforms all approaches tested in all metrics. Most notably, the state-of-the-art image retrieval foundation model, ImageBind, is incapable of recognizing speech. We note that the ImageBind authors do not publish retraining code, so we evaluate their largest pretrained model. Models with a * indicate that they have been previously reported in the literature. Other numbers are calculated by using pretrained models when available or from training with the author’s official training scripts.should be a path to an image"
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

        if frames is not None and text_list is not None:
            embeddings['vis_text_sim_matrix'] = self.compute_similarity_matrix(embeddings['text_feats'], embeddings['visual_feats'])
        if audio is not None and frames is not None:
            embeddings['vis_audio_sim_matrix'] = self.compute_similarity_matrix(embeddings['audio_feats'], embeddings['visual_feats'])
        if audio is not None and text_list is not None:
            embeddings['text_audio_sim_matrix'] = self.compute_similarity_matrix(embeddings['text_feats'], embeddings['audio_feats'])
        return embeddings

if __name__ == "__main__":
    print("Testing MultiModalModel with random inputs...")
    model = MultiModalModel(
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.2
    )
    batch_size = 2
    dummy_frames = torch.randn(batch_size, 3, 224, 224)
    dummy_audio  = torch.randn(batch_size, 16000)
    dummy_texts  = ["a man riding a bicycle", "a cat on a bed"]
    model.train()
    av_loss, _, _, _, _ = model.forward_audio_visual(dummy_frames, dummy_audio)
    print(f"Audio-Visual loss: {av_loss.item():.4f}")
    tv_loss, _ = model.forward_text_visual(dummy_frames, dummy_texts)
    print(f"Text-Visual loss: {tv_loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        av_sims = model.forward_audio_visual(dummy_frames, dummy_audio)
        print(f"Audio-Visual similarities shape: {av_sims.shape}")  

        tv_sims, tv_mask = model.forward_text_visual(dummy_frames, dummy_texts)
        print(f"Text-Visual similarities shape: {tv_sims.shape}, mask: {tv_mask.shape}")

    
    print("MultiModalModel test completed.")#
