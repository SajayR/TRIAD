import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

class TextVisualizer:
    def __init__(self, patch_size=14, image_size=224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size
        
        # Same colormap as audio - transparent -> blue -> red -> yellow
        colors = [
            (0,0,0,0),     # Transparent for low attention
            (0,0,1,0.5),   # Blue for medium-low
            (1,0,0,0.7),   # Red for medium-high  
            (1,1,0,1)      # Yellow for high attention
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)

    def get_token_attention_maps(self, model, frame, text):
        """Get attention maps for each text token"""
        model.eval()
        frame = frame.unsqueeze(0)  # Add batch dimension [1, C, H, W]
        with torch.no_grad():
            visual_feats = model.visual_embedder(frame)    # (1, Nv, D)
            text_feats, attention_mask = model.text_embedder([text])  # (1, Nt, D), (1, Nt)
            
            # Get raw similarity matrix and convert to attention maps
            similarity, mask = model(frame, [text])
            similarity = similarity.squeeze(0)  # (Nt, Nv)
            mask = mask.squeeze(0)  # (Nt,)
            
            # Only keep attention maps for real tokens (not padding)
            valid_maps = similarity[mask.bool()]
            attention_maps = self.patches_to_heatmaps(valid_maps)
            
            # Get the actual tokens for visualization
            tokens = model.text_embedder.tokenizer.tokenize(text)
            
        return attention_maps, tokens
    
    def patches_to_heatmaps(self, patch_attention):
        """Convert patch-level attention to pixel-level heatmaps"""
        Nt, Nv = patch_attention.shape
        # Square attention values to increase contrast
        patches = patch_attention.reshape(Nt, self.num_patches, self.num_patches)
        patches = patches ** 2
        
        # Upsample to image size
        heatmaps = F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)
        
        return heatmaps
    
    def create_overlay_frame(self, frame: np.ndarray, heatmap: np.ndarray, alpha=0.5):
        """Create a single frame with heatmap overlay"""
        # Normalize heatmap to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 2)  # Square for contrast
        
        # Apply colormap
        heatmap_colored = self.cmap(heatmap)
        heatmap_bgr = (heatmap_colored[...,:3] * 255).astype(np.uint8)
        
        # Blend with original frame
        overlay = ((1-alpha) * frame + alpha * heatmap_bgr).astype(np.uint8)
        return overlay
    
    def plot_token_attentions(self, model, frame, text, output_path=None):
        """
        Plot attention maps for each token in the text
        
        Args:
            model: The text-visual model
            frame: ImageNet normalized frame tensor [C, H, W]
            text: String of text to analyze
            output_path: Optional path to save visualization
        """
        # Get attention maps and tokens
        attention_maps, tokens = self.get_token_attention_maps(model, frame, text)
        num_tokens = len(tokens)
        
        # Convert frame to numpy and denormalize
        frame_np = frame.permute(1,2,0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        frame_np = frame_np * std + mean
        frame_np = np.clip(frame_np * 255, 0, 255).astype(np.uint8)
        
        # Create subplot grid
        rows = (num_tokens + 3) // 4  # 4 tokens per row
        cols = min(4, num_tokens)
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Plot each token's attention map
        for ax, token, heatmap in zip(axes, tokens, attention_maps):
            overlay = self.create_overlay_frame(frame_np, heatmap.cpu().numpy())
            ax.imshow(overlay)
            ax.set_title(f"Token: {token}")
            ax.axis('off')
            
        # Hide unused subplots
        for ax in axes[num_tokens:]:
            ax.axis('off')
            
        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

if __name__ == "__main__":
    from model import TextVisualModel
    model = TextVisualModel().eval()
    visualizer = TextVisualizer()
    
    # Create test image (white frame)
    frame = torch.ones(3, 224, 224)  # [C, H, W] - batch dim added in get_token_attention_maps
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    frame = (frame - mean) / std
    
    # Test with some text
    text = "a black cat sitting on a white fence"
    visualizer.plot_token_attentions(
        model, frame, text,
        output_path='text_attention_viz.png'
    )
    print("Done! Check text_attention_viz.png")