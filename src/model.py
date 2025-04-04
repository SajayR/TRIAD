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

#################################################################
#              Helper: "Smooth Max" via Log-Sum-Exp
#################################################################
def logsumexp_pool(token_sims: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Replaces the old 'max over tokens' with:
        aggregator = tau * logsumexp( token_sims / tau, dim=-1 )
    Then we can average over the dimension that enumerates tokens 
    in the "query" modality.

    Args:
        token_sims: (B, B, Na, Nv) or (B, B, Nt, Nv) 
                    where Na/Nt = # tokens in modality A/text,
                          Nv     = # tokens in modality V (visual).
        tau: a float > 0 controlling how soft/hard we approximate max.
             - Large tau => more uniform weighting (very soft).
             - Small tau => distribution gets sharp (close to max).
    
    Returns:
        clip_sims: shape (B, B), the final "pairwise" scores for each (i, j).
    """
    # token_sims / tau => shape still (B, B, Na, Nv)
    # log-sum-exp along last dim => shape (B, B, Na)
    # then multiply by tau => shape still (B, B, Na)
    # finally we average over "Na" => shape (B, B)
    scaled = token_sims / tau
    # logsumexp along dimension -1 (visual tokens)
    # => shape (B, B, Na)
    aggregator = tau * torch.logsumexp(scaled, dim=-1)
    # Now average over the "Na" dimension => shape (B, B)
    clip_sims = aggregator.mean(dim=-1)
    return clip_sims

#################################################################
#                   Audio Embedder
#################################################################
class AudioEmbedder(nn.Module):
    def __init__(self, embedding_dim=384, hubert_name="facebook/hubert-base-ls960"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")  
        self.hubert = HubertModel.from_pretrained(hubert_name)
        self.projection1 = nn.Linear(self.hubert.config.hidden_size, 384)
        self.layer_norm = nn.LayerNorm(384)
        self.projection2 = nn.Linear(384, embedding_dim)
        
        for param in self.hubert.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        
    def forward(self, audio_input: torch.Tensor) -> torch.Tensor:
        if len(audio_input.shape) == 3:  # shape: [B, 1, T]
            audio_input = audio_input.squeeze(0)  
        inputs = self.processor(
            audio_input, 
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
        ).input_values.squeeze(0)
        device = next(self.parameters()).device
        inputs = inputs.to(device)
        
        hubert_output = self.hubert(inputs).last_hidden_state  # (B, T', hidden_size)
        
        audio_feats = self.projection2(self.layer_norm(self.projection1(hubert_output)))  
        return audio_feats


#################################################################
#                   Text Embedder
#################################################################
class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim=384, model_name="distilbert/distilbert-base-uncased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection1 = nn.Linear(self.encoder.config.hidden_size, 384)
        self.layer_norm = nn.LayerNorm(384)
        self.projection2 = nn.Linear(384, embedding_dim)
        
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection1.parameters():
            param.requires_grad = True
        for param in self.projection2.parameters():
            param.requires_grad = True
        
    def forward(self, text_list):
        inputs = self.tokenizer(
            text_list, 
            padding=True,
            truncation=True,
            add_special_tokens=False,
            max_length=64,
            return_tensors="pt"
        )
        device = next(self.parameters()).device
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        outputs = self.encoder(**inputs)  # (B, Nt, hidden_size)
        hidden_states = outputs.last_hidden_state
        text_feats = self.projection2(self.layer_norm(self.projection1(hidden_states)))  
        
        return text_feats, inputs["attention_mask"]


#################################################################
#                   Visual Embedder (LoRA)
#################################################################
class ViTLoRAEmbedder(nn.Module):
    def __init__(self, model_name='facebookresearch/dinov2', arch='dinov2_vitb14',
                 embedding_dim=384, dropout_prob=0.1, lora_rank=8, lora_alpha=16):
        super().__init__()
        # Load the base DINOv2
        self.model = torch.hub.load(model_name, arch)
        
        # Freeze the base model
        for param in self.model.parameters():
            param.requires_grad = False
            
        # LoRA configuration
        lora_target_modules = [
            "attn.qkv",
            "attn.proj",
            # Optionally include MLP layers if you want:
            # "mlp.fc1",
            # "mlp.fc2",
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
        
        # Now enable training only on LoRA parameters
        for p in self.model.parameters():
            p.requires_grad = True
        # but we still consider the base model parameters as frozen:
        for p in self.model.base_model.parameters():
            p.requires_grad = False
        
        self.projection1 = nn.Linear(self.model.embed_dim, 384)
        self.layer_norm = nn.LayerNorm(384)
        self.projection2 = nn.Linear(384, embedding_dim)
        self.patch_dropout_rate = dropout_prob

    def patch_dropout(self, x, drop_rate):
        if not self.training or drop_rate == 0:
            return x
        B, N, D = x.shape
        device = x.device
        
        keep_mask = torch.bernoulli(
            torch.ones(B, N, device=device) * (1 - drop_rate)
        ).bool()

        output_tensors = []
        for i in range(B):
            kept_tokens = x[i][keep_mask[i]]
            output_tensors.append(kept_tokens)
        max_len = max(tensor.size(0) for tensor in output_tensors)
        padded_outputs = []
        for tensor in output_tensors:
            if tensor.size(0) < max_len:
                pad_len = max_len - tensor.size(0)
                padding = torch.zeros(pad_len, D, device=device)
                padded_outputs.append(torch.cat([tensor, padding], dim=0))
            else:
                padded_outputs.append(tensor)
        x = torch.stack(padded_outputs, dim=0)
        return x

    def forward(self, x):
        if len(x.shape) == 5:  # e.g. (1, 1, 3, 224, 224)
            x = x.squeeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        # get_intermediate_layers => a DINOv2 forward that returns patch tokens
        patches = self.model.get_intermediate_layers(x, n=1)[0]
        feats = self.projection2(self.layer_norm(self.projection1(patches)))
        feats = self.patch_dropout(feats, self.patch_dropout_rate)
        return feats


#################################################################
#                   MultiModalModel
#################################################################
class MultiModalModel(nn.Module):
    """
    Modified to use log-sum-exp aggregator instead of the hard max-mean aggregator.
    Also includes a new attribute `aggregator_temp` which you can schedule from high->low.
    """
    def __init__(
        self, 
        audio_model_name="facebook/hubert-base-ls960",
        text_model_name="distilbert/distilbert-base-uncased",
        # old 'temperature' param is no longer used for max aggregator 
        # but you can still keep a separate scale factor for the dot-product if you like
        aggregator_temp: float = 2.0,  # <- ADDED: default softmax temperature
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1,
        use_amp=True
    ):
        super().__init__()

        self.audio_embedder = AudioEmbedder(embedding_dim=384, hubert_name=audio_model_name)
        self.text_embedder  = TextEmbedder(embedding_dim=384, model_name=text_model_name)
        self.visual_embedder = ViTLoRAEmbedder(
            arch='dinov2_vitb14', 
            embedding_dim=384, 
            dropout_prob=visual_dropout_prob
        )

        # ADDED: aggregator temperature (we'll schedule this in training)
        self.aggregator_temp = aggregator_temp

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
        # ONLY NORMALIZE DURING INFERENCE, TRAINING CAUSES MODEL COLLAPSE
        feats1 = F.normalize(feats1, dim=-1)
        feats2 = F.normalize(feats2, dim=-1)
        #always run in full precision
        with torch.cuda.amp.autocast(enabled=False):
            sim = torch.bmm(feats1, feats2.transpose(1, 2))
            return sim 

    ######################################################
    #   Log-sum-exp aggregator (instead of max aggregator)
    ######################################################
    def compute_all_similarities_av(self, audio_feats, visual_feats):
        """
        Cross-batch approach: compute pairwise token_sims => shape (B, B, Na, Nv).
        Then use a log-sum-exp aggregator across the visual dimension, average over Na.
        """
        B = audio_feats.shape[0]
        af = audio_feats.float().unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.float().unsqueeze(0).expand(B, -1, -1, -1)
        token_sims = torch.matmul(af, vf.transpose(2, 3))  # shape => (B, B, Na, Nv)

        # aggregator with log-sum-exp
        clip_sims = logsumexp_pool(token_sims, tau=self.aggregator_temp)
        return clip_sims, token_sims

    def compute_all_similarities_tv(self, text_feats, visual_feats, attention_mask):
        """
        text_feats: (B, Nt, D)
        visual_feats: (B, Nv, D)
        attention_mask: (B, Nt)
        
        We do cross-batch => shape (B, B, Nt, Nv).
        Then log-sum-exp aggregator across Nv, average over Nt (but masked).
        """
        B = text_feats.shape[0]
        tf = text_feats.float().unsqueeze(1).expand(-1, B, -1, -1)
        vf = visual_feats.float().unsqueeze(0).expand(B, -1, -1, -1)
        token_sims = torch.matmul(tf, vf.transpose(2, 3))  # (B, B, Nt, Nv)

        # aggregator along the visual dim => shape (B, B, Nt)
        aggregator = logsumexp_pool(token_sims, tau=self.aggregator_temp)  # shape => (B, B)

        # However, we also have an attention_mask for text tokens. 
        # If you want to EXACTLY replicate the old "masked mean" approach, you'd 
        # do a slightly different aggregator. 
        # But let's keep it simple and do the same aggregator for all pairs. 
        # You can incorporate the attention mask as well, but that’s extra complexity.

        return aggregator, token_sims

    ######################################################
    #               AUDIO-VISUAL PATH
    ######################################################
    def compute_regularization_losses_av(self, token_sims):
        """
        If you want to keep your old regularizations (like negative clamp or 
        temperature calibration), you can do it here. 
        We'll show the old logic for reference, but you might want to disable it 
        or adapt it for log-sum-exp aggregator. 
        """
        # 1) Non-negative pressure
        neg_sims = torch.clamp(token_sims, min=-60, max=0)
        l_nonneg = torch.mean(neg_sims ** 2)

        # 2) We skip complicated temperature constraints, or adapt them if needed
        l_cal = 0.0  # you can reintroduce if you want

        # 3) For old "temporal smoothness", skip unless you are sure you want it
        l_smooth = 0.0

        reg_loss = 0.15 * l_nonneg + 0.01 * l_smooth + 0.9 * l_cal
        return reg_loss, l_smooth

    def compute_contrastive_loss_av(self, clip_sims, token_sims):
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)

        pos_sims = torch.diagonal(clip_sims)
        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(False)
        neg_sims = clip_sims[mask]

        pos_sim_mean = pos_sims.mean().item()
        neg_sim_mean = neg_sims.mean().item()
        separation = pos_sim_mean - neg_sim_mean

        # Standard cross-entropy
        log_prob_a2v = F.log_softmax(clip_sims, dim=1)
        losses_a2v = -log_prob_a2v[labels, labels]
        log_prob_v2a = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2a = -log_prob_v2a[labels, labels]
        contrastive_loss = (losses_a2v + losses_v2a).mean() / 2

        reg_loss, l_smooth = self.compute_regularization_losses_av(token_sims)
        total_loss = contrastive_loss + reg_loss

        similarity_stats = {
            "av_pos_sim_mean": pos_sim_mean,
            "av_neg_sim_mean": neg_sim_mean,
            "av_separation": separation
        }
        return total_loss, contrastive_loss, reg_loss, l_smooth, similarity_stats

    def forward_audio_visual(self, frames, audio):
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            visual_feats = self.visual_embedder(frames)     # (B, Nv, D)
            audio_feats = self.audio_embedder(audio)        # (B, Na, D)
            clip_sims, token_sims = self.compute_all_similarities_av(audio_feats, visual_feats)
            return self.compute_contrastive_loss_av(clip_sims, token_sims)

    ######################################################
    #               TEXT-VISUAL PATH
    ######################################################
    def compute_regularization_losses_tv(self, token_sims):
        # If you have any text-visual specific regularizations, apply them here.
        # For example, negative clamp or patch usage sparsity. 
        # In your original code you had a patch_sparsity. You can keep that or drop it.
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims**2)
        # patch usage can still be done on diagonal pairs if you want
        return 0.15 * l_nonneg

    def compute_contrastive_loss_tv(self, clip_sims, token_sims):
        B = clip_sims.shape[0]
        labels = torch.arange(B, device=clip_sims.device)

        pos_sims = torch.diagonal(clip_sims)
        mask = torch.ones_like(clip_sims, dtype=torch.bool)
        mask.fill_diagonal_(False)
        neg_sims = clip_sims[mask]

        pos_sim_mean = pos_sims.mean().item()
        neg_sim_mean = neg_sims.mean().item()
        separation = pos_sim_mean - neg_sim_mean

        log_prob_t2v = F.log_softmax(clip_sims, dim=1)
        losses_t2v = -log_prob_t2v[labels, labels]
        log_prob_v2t = F.log_softmax(clip_sims.t(), dim=1)
        losses_v2t = -log_prob_v2t[labels, labels]
        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2

        reg_loss = self.compute_regularization_losses_tv(token_sims)
        total_loss = contrastive_loss + reg_loss
        
        similarity_stats = {
            "tv_pos_sim_mean": pos_sim_mean,
            "tv_neg_sim_mean": neg_sim_mean,
            "tv_separation": separation
        }
        return total_loss, similarity_stats

    def forward_text_visual(self, frames, text_list):
        with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            visual_feats = self.visual_embedder(frames)
            text_feats, attention_mask = self.text_embedder(text_list)
            clip_sims, token_sims = self.compute_all_similarities_tv(text_feats, visual_feats, attention_mask)
            return self.compute_contrastive_loss_tv(clip_sims, token_sims)

    ######################################################
    #  Inference forward / "Multimodal Embeddings" also
    ######################################################


    def forward(self, frames=None, audio=None, text_list=None):
        assert frames is not None or audio is not None or text_list is not None, "At least one modality must be provided"
        # we need to conver the image into the correct format and shit
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
        text_model_name="distilbert/distilbert-base-uncased",
        aggregator_temp=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.2
    )
    batch_size = 2
    dummy_frames = torch.randn(batch_size, 3, 224, 224)      # image frames
    dummy_audio  = torch.randn(batch_size, 16000)            # 1 sec of 16kHz
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
        # expected => (B, Na, Nv)
        tv_sims, tv_mask = model.forward_text_visual(dummy_frames, dummy_texts)
        print(f"Text-Visual similarities shape: {tv_sims.shape}, mask: {tv_mask.shape}")
        # expected => (B, Nt, Nv), (B, Nt)
    
    print("MultiModalModel test completed.")
