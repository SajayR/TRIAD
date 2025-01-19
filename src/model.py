import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class TextEmbedder(nn.Module):
    def __init__(self, embedding_dim=512, model_name="roberta-base"):
        super().__init__()
        # Example: using BERT instead of Roberta
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base") #bert-base-uncased
        self.encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
        self.projection = nn.Linear(self.encoder.config.hidden_size, embedding_dim)
        
        # Keep all parameters trainable
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.projection.parameters():
            param.requires_grad = True
        
    def forward(self, text):
        """
        Args:
            text: List[str] batch of text inputs
            
        Returns:
            features: (B, Nt, D) where:
                B is batch size
                Nt is number of text tokens
                D is embedding_dim
            attention_mask: (B, Nt)
        """
        inputs = self.tokenizer(
            text, 
            padding=True,
            truncation=True,
            max_length=128,  # Better parallelization with power of 2
            add_special_tokens=False,  # No CLS/SEP tokens
            return_tensors="pt"
        )
        # Move tokenizer outputs to same device as model
        inputs = {k: v.to(next(self.encoder.parameters()).device) for k, v in inputs.items()}
        
        # Encoded hidden states
        outputs = self.encoder(**inputs)
        hidden_states = outputs.last_hidden_state  # (B, Nt, hidden_size)
        features = self.projection(hidden_states)  # (B, Nt, embedding_dim)
        
        # Return both features and attention mask so we can do masked operations
        return features, inputs["attention_mask"]


class ViTEmbedder(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True, dropout_prob=0.1):
        super().__init__()
        # Example: Using DINOv2 from Facebook's hub
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.projection = nn.Linear(self.model.embed_dim, 512)

        # Visual dropout
        self.dropout = nn.Dropout(p=dropout_prob)
        
        for param in self.model.parameters():
            param.requires_grad = True
            
    def forward(self, x):
        """
        Args:
            x: (batch_size, channels, height, width)
        Returns:
            patch_embeddings: (batch_size, num_patches, embedding_dim)
        """
        # Get only the first intermediate layer
        x = self.model.get_intermediate_layers(x, n=1)[0]
        
        # Project and drop out
        x = self.projection(x)
        x = self.dropout(x)
        return x


class TextVisualModel(nn.Module):
    def __init__(
        self, 
        temperature=2.0, 
        patch_sparsity_threshold=0.3, 
        patch_sparsity_weight=0.1, 
        visual_dropout_prob=0.1
    ):
        super().__init__()
        
        # Insert the dropout probability in ViTEmbedder
        self.visual_embedder = ViTEmbedder(dropout_prob=visual_dropout_prob)
        self.text_embedder = TextEmbedder()
        
        # Sparsity hyperparams
        self.patch_sparsity_threshold = patch_sparsity_threshold
        self.patch_sparsity_weight = patch_sparsity_weight

        # If you want to keep a trainable temperature, uncomment below:
        # self.temperature = nn.Parameter(torch.tensor(temperature))
        # else you can keep it as a constant:
        #self.temperature = temperature

    def compute_similarity_matrix(self, text_feats, visual_feats):
        """
        Compute pairwise cosine similarities between text and visual tokens
        
        Args:
            text_feats: (B, Nt, D)
            visual_feats: (B, Nv, D)
        
        Returns:
            similarity_matrix: (B, Nt, Nv)
        """
        text_feats = F.normalize(text_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        similarity = torch.bmm(text_feats, visual_feats.transpose(1, 2))  # (B, Nt, Nv)
        return similarity
    
    def aggregate_token_similarities(self, similarity_matrix, attention_mask):
        """
        After computing similarity_matrix = (B, Nt, Nv), we:
          1) Take the max over visual dimension => (B, Nt).
          2) Compute the mean over text tokens, *excluding padding* based on attention_mask.
        """
        # (B, Nt)
        max_similarities = torch.max(similarity_matrix, dim=2)[0]
        
        # Convert mask to float for multiplication
        mask_f = attention_mask.float()
        
        # Sum only valid tokens
        masked_sum = (max_similarities * mask_f).sum(dim=1)
        # Count how many valid tokens
        valid_tokens = mask_f.sum(dim=1).clamp(min=1e-7)
        
        # Final masked mean
        clip_similarity = masked_sum / valid_tokens
        return clip_similarity
    
    def compute_all_similarities(self, text_feats, visual_feats, attention_mask):
        """
        Compute similarities between all pairs of text and visual features in the batch
        for contrastive learning. 
        Returns:
          - clip_sims: (B, B) aggregated similarity for each pair in the batch
          - token_sims: (B, B, Nt, Nv) raw token-level similarities
        We do a masked mean along the text tokens dimension to get clip_sims.
        """
        B = text_feats.shape[0]
        
        # Expand so we can compute cross similarities across the batch
        # text_feats -> (B, 1, Nt, D) -> (B, B, Nt, D)
        text_feats = text_feats.unsqueeze(1).expand(-1, B, -1, -1)
        # visual_feats -> (1, B, Nv, D) -> (B, B, Nv, D)
        visual_feats = visual_feats.unsqueeze(0).expand(B, -1, -1, -1)
        
        text_feats = F.normalize(text_feats, dim=-1)
        visual_feats = F.normalize(visual_feats, dim=-1)
        
        # token_sims: (B, B, Nt, Nv)
        token_sims = torch.matmul(text_feats, visual_feats.transpose(2, 3)) 
        
        # Take the max over visual tokens => (B, B, Nt)
        max_sims = torch.max(token_sims, dim=3)[0]
        
        # Now we need a masked mean over Nt, but note attention_mask is shape (B, Nt).
        # We must expand it along dimension 1 (because of cross-batch).
        # attn_mask_expanded -> (B, 1, Nt) -> (B, B, Nt)
        attn_mask_expanded = attention_mask.unsqueeze(1).float().expand(-1, B, -1)
        
        masked_sum = (max_sims * attn_mask_expanded).sum(dim=2)  # (B, B)
        valid_tokens = attn_mask_expanded.sum(dim=2).clamp(min=1e-7)  # (B, B)
        
        # Final clip_sims => (B, B)
        clip_sims = masked_sum / valid_tokens
        
        return clip_sims, token_sims

    def compute_contrastive_loss(self, clip_similarities, token_sims):
        """
        Standard cross-entropy contrastive loss across text->vision and vision->text,
        plus regularization.
        """
        batch_size = clip_similarities.shape[0]
        labels = torch.arange(batch_size).to(clip_similarities.device)
        
        log_prob_t2v = F.log_softmax(clip_similarities, dim=1)
        losses_t2v = -log_prob_t2v[torch.arange(batch_size), labels]
        
        log_prob_v2t = F.log_softmax(clip_similarities.t(), dim=1)
        losses_v2t = -log_prob_v2t[torch.arange(batch_size), labels]
        
        contrastive_loss = (losses_t2v + losses_v2t).mean() / 2

        # Add our regularization terms
        reg_loss = self.compute_regularization_losses(clip_similarities, token_sims)
        
        total_loss = contrastive_loss + reg_loss
        return total_loss
    
    def compute_regularization_losses(self, clip_sims, token_sims):
        """
        1) Force negative sims near zero (existing approach).
        2) Patch usage sparsity: if a patch is used by > threshold fraction of tokens, penalize it.
        """
        # --- (1) Negative sims near zero ---
        # We only penalize negative values in token_sims (any of them across the batch).
        neg_sims = torch.clamp(token_sims, min=-20, max=0)
        l_nonneg = torch.mean(neg_sims ** 2)
        
        # --- (2) Patch usage sparsity ---
        # We'll compute soft usage for the "correct" pairs only (diagonal of token_sims).
        # token_sims has shape (B, B, Nt, Nv).
        B = token_sims.shape[0]
        positive_sims = []
        for i in range(B):
            # shape (Nt, Nv)
            positive_sims.append(token_sims[i, i])
        # Stack => (B, Nt, Nv)
        positive_sims = torch.stack(positive_sims, dim=0)

        # Now take a softmax over the patches dimension => (B, Nt, Nv)
        patch_probs = F.softmax(positive_sims, dim=-1)

        # patch_fraction[b, v] = fraction of text tokens that (softly) pick patch v
        # shape => (B, Nv)
        patch_fraction = patch_probs.sum(dim=1) / patch_probs.shape[1]

        # For each patch, if fraction > threshold, penalize (we do a ReLU).
        # shape => (B, Nv)
        excess = F.relu(patch_fraction - self.patch_sparsity_threshold)

        # We can do a squared penalty or linear. Example: squared:
        loss_sparsity = (excess ** 2).mean()

        # Weighted sum of negative sims penalty + patch usage penalty
        reg_loss = 0.15 * l_nonneg + self.patch_sparsity_weight * loss_sparsity
        return reg_loss
        
    def forward(self, frames, texts):
        """
        Forward pass computing embeddings, similarities and loss
        
        Args:
            frames: (B, C, H, W) batch of images
            texts: List[str] batch of text inputs
            
        Returns:
            If training: scalar contrastive loss
            If eval: (B, Nt, Nv) raw token similarity matrix OR aggregated similarity
        """
        # Get embeddings
        visual_feats = self.visual_embedder(frames)             # (B, Nv, D)
        text_feats, attention_mask = self.text_embedder(texts)  # (B, Nt, D), (B, Nt)
        
        if self.training:
            # Compute cross-batch similarities and do standard contrastive
            clip_sims, token_sims = self.compute_all_similarities(
                text_feats, visual_feats, attention_mask
            )
            return self.compute_contrastive_loss(clip_sims, token_sims)
        else:
            # Inference path: compute pairwise sim for each item and do masked mean
            sim_matrix = self.compute_similarity_matrix(text_feats, visual_feats) # (B, Nt, Nv)
            return sim_matrix, attention_mask


if __name__ == "__main__":
    # Example usage
    model = TextVisualModel(
        temperature=2.0,
        patch_sparsity_threshold=0.3,
        patch_sparsity_weight=0.1,
        visual_dropout_prob=0.1
    )
    
    batch_size = 4
    frames = torch.randn(batch_size, 3, 224, 224)
    texts = ["a dog running", "cat sleeping", "bird flying", "fish swimming"]
    
    # Training mode
    model.train()
    loss = model(frames, texts)
    print(f"Training loss: {loss.item()}")
    
    # Inference mode
    model.eval()
    similarities, attention_mask = model(frames, texts)
    print(f"Inference similarities shape: {similarities.shape}")
    print(f"Similarity values: {similarities}")
