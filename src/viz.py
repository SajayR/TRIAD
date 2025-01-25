# viz.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import ffmpeg  # Ensure ffmpeg-python is installed for audio+video output


#########################################################
#                   AUDIO VISUALIZER
#########################################################
class AudioVisualizer:
    """
    Creates audio-visual attention overlays (or videos) given a model
    that outputs token-level similarity between audio and visual features.
    """
    def __init__(self, patch_size=14, image_size=224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size

        # Transparent -> blue -> red -> yellow
        colors = [
            (0, 0, 0, 0),
            (0, 0, 1, 0.5),
            (1, 0, 0, 0.7),
            (1, 1, 0, 1.0)
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)

    def _validate_inputs(self, frame, audio):
        """Check that inputs are in the expected range."""
        frame_min, frame_max = frame.min().item(), frame.max().item()
        assert -3 <= frame_min <= 3, f"Frame min {frame_min} outside normalized range"
        assert -3 <= frame_max <= 3, f"Frame max {frame_max} outside normalized range"
        audio_min, audio_max = audio.min().item(), audio.max().item()
        assert -2 <= audio_min <= 2, f"Audio min {audio_min} outside typical range"
        assert -2 <= audio_max <= 2, f"Audio max {audio_max} outside typical range"

    def get_attention_maps(self, model, frame, audio):
        """
        Given a model (e.g. MultiModalModel) and a single image+audio,
        get the token-level similarity => shape: (Na, Nv).
        Then convert to a set of heatmaps of shape: (Na, H, W).
        """
        model.eval()
        with torch.no_grad():
            # shape => (1, 3, H, W) and (1, T)
            frame = frame.unsqueeze(0)
            audio = audio.unsqueeze(0)
            visual_feats = model.visual_embedder(frame)   # (1, Nv, D)
            audio_feats = model.audio_embedder(audio)     # (1, Na, D)

            # token-sims => (1, Na, Nv)
            sim = model.compute_similarity_matrix(audio_feats, visual_feats).squeeze(0)
            # convert patches to heatmaps
            attention_maps = self.patches_to_heatmaps(sim)

        return attention_maps  # shape: (Na, H, W)

    def patches_to_heatmaps(self, patch_attention):
        """Upsample patch-level attention (Na, Nv) into pixel-level heatmaps (Na, H, W)."""
        Na, Nv = patch_attention.shape
        patches = patch_attention.reshape(Na, self.num_patches, self.num_patches)
        # Square the attention for better contrast
        patches = patches ** 2

        # Upsample to 224x224
        heatmaps = F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        return heatmaps  # shape: (Na, H, W)

    def create_overlay_frame(self, frame_np: np.ndarray, heatmap: np.ndarray, alpha=0.5):
        """Overlay a heatmap onto an RGB frame."""
        # Normalize the heatmap to [0,1]
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 2)

        heatmap_colored = self.cmap(heatmap)[..., :3]  # drop alpha, shape => (H, W, 3)
        heatmap_bgr = (heatmap_colored * 255).astype(np.uint8)

        overlay = ((1 - alpha) * frame_np + alpha * heatmap_bgr).astype(np.uint8)
        return overlay

    def make_attention_video(self, model, frame, audio, output_path, fps=50):
        """
        Generate an attention video for a single (frame, audio).
        In the final MP4, each frame corresponds to one audio token's attention over the image.
        Audio is optionally muxed in if you wish to do so (requires the original video or wave).
        For minimal code, we won't do advanced A/V muxing from raw wave here unless you prefer.
        """
        self._validate_inputs(frame, audio)
        # Denormalize the image to [0..255]
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        frame_np = (frame_np * std + mean) * 255
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        # Get attention maps => (Na, H, W)
        attention_maps = self.get_attention_maps(model, frame, audio)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_video_path = str(output_path.with_suffix('.temp.mp4'))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (self.image_size, self.image_size))

        for heatmap in attention_maps:
            # shape => (H, W)
            overlay = self.create_overlay_frame(frame_np, heatmap.cpu().numpy())
            # Convert RGB to BGR for OpenCV
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            writer.write(overlay_bgr)

        writer.release()

        # Optional: if you want to add real audio from a wave or from some source,
        # you'd need to do something like ffmpeg merging. Skipping it for brevity.
        Path(temp_video_path).rename(output_path)

        print(f"Saved attention video to {output_path}")


#########################################################
#                   TEXT VISUALIZER
#########################################################
class TextVisualizer:
    """
    Creates text-visual overlays for image + token-level text attention,
    using a model that outputs similarity (B, Nt, Nv).
    """
    def __init__(self, patch_size=14, image_size=224):
        self.patch_size = patch_size
        self.image_size = image_size
        self.num_patches = image_size // patch_size

        # Transparent -> blue -> red -> yellow
        colors = [
            (0, 0, 0, 0),
            (0, 0, 1, 0.5),
            (1, 0, 0, 0.7),
            (1, 1, 0, 1.0)
        ]
        self.cmap = LinearSegmentedColormap.from_list('custom', colors)

    def get_token_attention_maps(self, model, frame, text):
        """
        For a single (image, text), compute (Nt, Nv) similarity, then upsample each patch.
        Returns attention_maps => (Nt, H, W), plus the token list.
        """
        model.eval()
        with torch.no_grad():
            # shape => (1, 3, H, W) and text => single string in a list
            frame_batched = frame.unsqueeze(0)
            text_list = [text]

            visual_feats = model.visual_embedder(frame_batched)        # (1, Nv, D)
            text_feats, text_mask = model.text_embedder(text_list)     # (1, Nt, D), (1, Nt)

            sim_matrix = model.compute_similarity_matrix(text_feats, visual_feats)  # (1, Nt, Nv)
            sim_matrix = sim_matrix.squeeze(0)  # (Nt, Nv)
            valid_tokens_count = text_mask.sum().item()  # how many tokens are real (non-padding)

            # We'll only keep the valid rows from sim_matrix => shape (valid_tokens_count, Nv)
            sim_matrix = sim_matrix[:valid_tokens_count]

            # Convert to heatmaps
            attention_maps = self._patches_to_heatmaps(sim_matrix)
            tokens = model.text_embedder.tokenizer.tokenize(text)

        return attention_maps, tokens[:valid_tokens_count]

    def _patches_to_heatmaps(self, patch_attention):
        """
        patch_attention => (Nt, Nv)
        upsample => (Nt, H, W)
        """
        Nt, Nv = patch_attention.shape
        patches = patch_attention.reshape(Nt, self.num_patches, self.num_patches)
        # Square for contrast
        patches = patches ** 2

        heatmaps = F.interpolate(
            patches.unsqueeze(1),
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(1)

        return heatmaps  # (Nt, H, W)

    def _create_overlay_frame(self, frame_np, heatmap, alpha=0.5):
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = np.power(heatmap, 2)
        heatmap_colored = self.cmap(heatmap)[..., :3]
        heatmap_bgr = (heatmap_colored * 255).astype(np.uint8)
        overlay = ((1 - alpha) * frame_np + alpha * heatmap_bgr).astype(np.uint8)
        return overlay

    def plot_token_attentions(self, model, frame, text, output_path=None):
        """
        Create subplots of each token's overlay on the image.
        """
        attention_maps, tokens = self.get_token_attention_maps(model, frame, text)
        Nt = attention_maps.shape[0]

        # Denormalize the image
        frame_np = frame.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        frame_np = (frame_np * std + mean) * 255
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

        # Layout: up to 4 columns
        cols = min(4, Nt)
        rows = (Nt + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx in range(Nt):
            heatmap = attention_maps[idx].cpu().numpy()
            overlay = self._create_overlay_frame(frame_np, heatmap)
            ax = axes[idx]
            ax.imshow(overlay)
            ax.set_title(f"Token: {tokens[idx]}")
            ax.axis("off")

        for ax in axes[Nt:]:
            ax.axis("off")

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path)
            plt.close()
            print(f"Saved token attention figure to {output_path}")
        else:
            plt.show()


#########################################################
#                   QUICK TEST (MAIN)
#########################################################
def _quick_audio_visual_test():
    from model import MultiModalModel
    model = MultiModalModel()
    model.eval()

    # Create dummy data
    frame = torch.ones(3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    frame = (frame - mean) / std

    #audio = torch.randn(16331)  # 1 sec at 16kHz
    t = torch.linspace(0, 2*torch.pi, 16331)
    audio = torch.sin(2 * torch.pi * 440 * t) # [16331]
    
    viz = AudioVisualizer()

    out_file = "test_audio_attention.mp4"
    viz.make_attention_video(model, frame, audio, out_file, fps=5)


def _quick_text_visual_test():
    from model import MultiModalModel
    model = MultiModalModel()
    model.eval()

    frame = torch.ones(3, 224, 224)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    frame = (frame - mean) / std

    text = "a dog playing in the park"
    viz = TextVisualizer()

    # Just display or save to file
    out_file = "test_text_attention.png"
    viz.plot_token_attentions(model, frame, text, output_path=out_file)


if __name__ == "__main__":
    print("Running quick tests for AudioVisualizer and TextVisualizer...")
    _quick_audio_visual_test()
    _quick_text_visual_test()
    print("All tests complete! Check the output files for visualizations.")
