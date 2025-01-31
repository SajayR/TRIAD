import os
import gc
import math
import wandb
import torch
import logging
import warnings
import numpy as np
import matplotlib.pyplot as plt
import random

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from dataset import (
    AudioVisualDataset, collate_fn,
    LocalCaptionDataset
)
from model import MultiModalModel
from viz import AudioVisualizer, TextVisualizer

warnings.filterwarnings("ignore")


###########################################
#         Collate for the Text Dataset
###########################################
def collate_text_fn(batch):
    """
    Simple collate function for the LocalCaptionDataset.
    Returns:
    {
       'images': (B, 3, 224, 224),
       'captions': list[str] of length B
    }
    """
    images, captions = zip(*batch)  # from the dataset's __getitem__
    images = torch.stack(images)    # shape => (B, 3, 224, 224)
    return {
        'images': images,
        'captions': list(captions)
    }


###########################################
#         Trainer Class
###########################################
class MultiModalTrainer:
    """
    Trainer that sums AudioVisual and TextVisual losses in the same step.
    Uses two separate dataloaders: one for Audio-Visual, one for Text-Visual.

    Now uses a *step-based* unfreezing schedule instead of an epoch-based schedule:
    - If current_step >= unfreeze_audio_step => set LR for audio encoder to the normal LR, else 0.0
    - If current_step >= unfreeze_text_step  => set LR for text encoder to the normal LR, else 0.0
    - If current_step >= unfreeze_vit_step   => set LR for ViT to the normal LR, else 0.0
    """

    def __init__(
        self,
        audio_visual_data_root: str,
        text_dataset_path: str,
        output_dir: str = "./outputs_multimodal",
        batch_size_av: int = 16,
        batch_size_tv: int = 32,
        num_epochs: int = 20,
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        # Step-based unfreeze thresholds (instead of epoch-based)
        unfreeze_audio_step: int = 2000,
        unfreeze_text_step: int = 5000,
        unfreeze_vit_step: int = 8000,
        use_wandb: bool = True,
        project_name: str = "UnifiedMultiModal",
        vis_every: int = 1000,
        num_vis_samples_av: int = 10,
        num_vis_samples_tv: int = 10,
        save_every_steps: int = 10000,
        num_workers: int = 4,
        device: str = "cuda",
        force_new_training: bool = False
    ):
        """
        Args:
            audio_visual_data_root: path to video dataset (mp4 files)
            text_dataset_path:      path to images & text files (LocalCaptionDataset)
            output_dir:             where to save logs/checkpoints
            batch_size_av:          batch size for AudioVisual loader
            batch_size_tv:          batch size for TextVisual loader
            num_epochs:             total number of epochs
            learning_rate:          base LR for 'unfrozen' groups
            gradient_accumulation_steps: how many steps to accumulate before optimizer step
            unfreeze_audio_step, unfreeze_text_step, unfreeze_vit_step:
                after these step counts, set LR from 0.0 to the normal LR for each encoder
            ...
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.vis_every = vis_every
        self.save_every_steps = save_every_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Store main config in a dict for logging or potential checkpoint usage
        self.config = {
            "batch_size_av": batch_size_av,
            "batch_size_tv": batch_size_tv,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "unfreeze_audio_step": unfreeze_audio_step,
            "unfreeze_text_step": unfreeze_text_step,
            "unfreeze_vit_step": unfreeze_vit_step,
            "vis_every": vis_every,
            "save_every_steps": save_every_steps
        }

        # Logging setup
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # -----------------------------------------------------
        #  1) Datasets / Dataloaders
        # -----------------------------------------------------
        # a) AudioVisual dataset
        self.av_dataset = AudioVisualDataset(
            data_root=audio_visual_data_root,
            sample_fps=20
        )
        self.av_dataloader = DataLoader(
            self.av_dataset,
            batch_size=batch_size_av,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True,
            collate_fn=collate_fn,
            prefetch_factor=4
        )

        # b) Text-Visual dataset
        self.tv_dataset = LocalCaptionDataset(text_dataset_path)
        self.tv_dataloader = DataLoader(
            self.tv_dataset,
            batch_size=batch_size_tv,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_text_fn,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
            prefetch_factor=8
        )

        # Iterators will be (re)created each epoch
        self.av_iter = None
        self.tv_iter = None

        # -----------------------------------------------------
        #  2) Model
        # -----------------------------------------------------
        self.model = MultiModalModel(
            audio_model_name="facebook/hubert-base-ls960",
            text_model_name="answerdotai/ModernBERT-base",
            temperature=2.0,
            patch_sparsity_threshold=0.3,
            patch_sparsity_weight=0.1,
            visual_dropout_prob=0.1
        ).to(self.device)

        # -----------------------------------------------------
        #  3) Separate param groups for unfreeze logic
        # -----------------------------------------------------
        audio_params = []
        text_params  = []
        vit_params   = []
        others_params= []
        for name, param in self.model.named_parameters():
            if "audio_embedder.hubert" in name:
                audio_params.append(param)
            elif "text_embedder.encoder" in name:
                text_params.append(param)
            elif "visual_embedder.model" in name:
                vit_params.append(param)
            else:
                others_params.append(param)

        # Freeze them at start: set LR=0. We'll unfreeze in _adjust_lr_for_frozen_params
        self.optimizer = torch.optim.AdamW([
            {"params": others_params, "lr": learning_rate},   # group 0 => 'others' always active
            {"params": audio_params,  "lr": 0.0},             # group 1 => 'audio'
            {"params": text_params,   "lr": 0.0},             # group 2 => 'text'
            {"params": vit_params,    "lr": 0.0},             # group 3 => 'vision'
        ])

        # Scheduler: single OneCycle for the total steps
        total_steps_per_epoch = max(len(self.av_dataloader), len(self.tv_dataloader))
        self.steps_per_epoch = total_steps_per_epoch
        self.total_updates = self.steps_per_epoch * num_epochs // self.gradient_accumulation_steps
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=self.total_updates,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        # -----------------------------------------------------
        #  4) State tracking & optional resume
        # -----------------------------------------------------
        self.start_epoch = 0
        self.global_step = 0
        self.current_batch_idx = 0   # offset inside the epoch
        self.best_loss = float('inf')

        # Optionally restore from latest checkpoint if found
        if self.use_wandb and not force_new_training:
            ckpt = self.find_latest_checkpoint()
            if ckpt:
                self.load_checkpoint(ckpt)
            else:
                wandb.init(project=self.project_name, config=self.config)
        elif self.use_wandb and force_new_training:
            wandb.init(project=self.project_name, config=self.config)
        if self.use_wandb and wandb.run is None:
            wandb.init(project=self.project_name, config=self.config)

        # Visualization utilities
        self.audio_viz = AudioVisualizer()
        self.text_viz  = TextVisualizer()
        self.vis_samples_av = self._get_av_vis_samples(num_vis_samples_av)
        self.vis_samples_tv = self._get_tv_vis_samples(num_vis_samples_tv)

        self.logger.info("Initialized MultiModalTrainer")

    ###########################################
    #     Checkpointing Helpers
    ###########################################
    def find_latest_checkpoint(self):
        ckpts = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        if not ckpts:
            return None

        def _parse_ckpt_name(p):
            """
            e.g. filename: 'checkpoint_epoch3_step1500.pt'
            parse out epoch=3, step=1500 as (3, 1500)
            """
            name = p.stem
            ep_str = name.split('epoch')[1].split('_')[0]  # '3'
            st_str = name.split('step')[1]                 # '1500'
            return (int(ep_str), int(st_str))

        return max(ckpts, key=_parse_ckpt_name)

    def save_checkpoint(self, epoch, step):
        """
        Saves everything needed to resume from the same batch and random states.
        """
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"

        # Save random states so we preserve shuffle order on resume
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda":  torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }

        ckpt = {
            "epoch": epoch,
            "step": step,
            "current_batch_idx": self.current_batch_idx,
            "rng_state": rng_state,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
            "vis_samples_av": self.vis_samples_av,
            "vis_samples_tv": self.vis_samples_tv
        }
        torch.save(ckpt, ckpt_path)
        self.logger.info(f"Saved checkpoint: {ckpt_path}")

    def load_checkpoint(self, ckpt_path):
        self.logger.info(f"Loading checkpoint from {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ck["model_state_dict"])
        self.optimizer.load_state_dict(ck["optimizer_state_dict"])
        self.scheduler.load_state_dict(ck["scheduler_state_dict"])
        self.start_epoch = ck["epoch"]
        self.global_step = ck["step"]
        self.current_batch_idx = ck.get("current_batch_idx", 0)
        self.best_loss = ck["best_loss"]

        # restore random states if present
        rng_state = ck.get("rng_state", None)
        if rng_state:
            torch.set_rng_state(rng_state["torch"])
            for i, s in enumerate(rng_state["cuda"]):
                torch.cuda.set_rng_state(s, device=i)
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])

        self.vis_samples_av = ck["vis_samples_av"]
        self.vis_samples_tv = ck["vis_samples_tv"]

        # Make sure to adjust LR for any steps that might already require unfreezing
        self._adjust_lr_for_frozen_params(self.global_step)

        self.logger.info(
            f"Checkpoint loaded (epoch={self.start_epoch}, step={self.global_step}, "
            f"batch_offset={self.current_batch_idx})"
        )

    ###########################################
    #  Step-Based Unfreeze schedule
    ###########################################
    def _adjust_lr_for_frozen_params(self, current_step: int):
        """
        If current_step >= unfreeze_audio_step => set param group1's LR to self.config['learning_rate'], else 0.0
        If current_step >= unfreeze_text_step  => set param group2's LR to self.config['learning_rate'], else 0.0
        If current_step >= unfreeze_vit_step   => set param group3's LR to self.config['learning_rate'], else 0.0

        param_groups:
         group0 => others
         group1 => audio
         group2 => text
         group3 => vision
        """
        lr = self.config['learning_rate']
        # Audio
        if current_step >= self.config['unfreeze_audio_step']:
            self.optimizer.param_groups[1]['lr'] = lr
        else:
            self.optimizer.param_groups[1]['lr'] = 0.0

        # Text
        if current_step >= self.config['unfreeze_text_step']:
            self.optimizer.param_groups[2]['lr'] = lr
        else:
            self.optimizer.param_groups[2]['lr'] = 0.0

        # Vision
        if current_step >= self.config['unfreeze_vit_step']:
            self.optimizer.param_groups[3]['lr'] = lr
        else:
            self.optimizer.param_groups[3]['lr'] = 0.0

    ###########################################
    #     Visualization logic
    ###########################################
    def _get_av_vis_samples(self, n_samples=2):
        """
        Pull a small batch from av_dataloader for visualization.
        We'll store them on CPU as well.
        """
        batch = next(iter(self.av_dataloader))
        idxs = range(min(n_samples, len(batch['frame'])))

        frames = batch['frame'][list(idxs)].cpu()
        audio  = batch['audio'][list(idxs)].cpu()
        video_paths = [batch['video_paths'][j] for j in idxs]

        return {
            "frames": frames,
            "audio": audio,
            "video_paths": video_paths
        }

    def _get_tv_vis_samples(self, n_samples=2):
        """
        Pull a small batch from the tv_dataloader for visualization.
        Store them on CPU as well.
        """
        batch = next(iter(self.tv_dataloader))
        images = batch['images']
        captions = batch['captions']
        idxs = range(min(n_samples, len(images)))
        images = images[list(idxs)].cpu()
        texts = [captions[i] for i in idxs]
        return {
            "images": images,
            "texts": texts
        }

    def visualize_samples(self, epoch):
        """
        We'll create a snapshot for a few audio+visual pairs and text+visual pairs,
        storing them or logging to wandb if enabled.
        """
        self.model.eval()

        # 1) Audio-Visual
        with torch.no_grad():
            for i in range(len(self.vis_samples_av["frames"])):
                frame = self.vis_samples_av["frames"][i].to(self.device)
                audio = self.vis_samples_av["audio"][i].to(self.device)
                video_path = self.vis_samples_av["video_paths"][i]

                out_file_img = self.output_dir / f"av_snapshot_epoch{epoch}_sample{i}.png"
                self.audio_viz.plot_audio_token_attentions(
                    model=self.model,
                    frame=frame,
                    audio=audio,
                    output_path=str(out_file_img),
                    num_tokens_to_show=5
                )
                out_file_vid = self.output_dir / f"av_viz_epoch{epoch}_sample{i}.mp4"
                self.audio_viz.make_attention_video(
                    model=self.model,
                    frame=frame,
                    audio=audio,
                    output_path=out_file_vid,
                    video_path=video_path,
                    fps=50
                )
                if self.use_wandb:
                    wandb.log({
                        f"av_attention_sample_{i}": wandb.Image(str(out_file_img)),
                        "epoch": epoch,
                        "global_step": self.global_step
                    })

        # 2) Text-Visual
        with torch.no_grad():
            for i in range(len(self.vis_samples_tv["images"])):
                frame = self.vis_samples_tv["images"][i].to(self.device)
                text = self.vis_samples_tv["texts"][i]
                out_file_img = self.output_dir / f"tv_viz_epoch{epoch}_sample{i}.png"
                self.text_viz.plot_token_attentions(
                    model=self.model,
                    frame=frame,
                    text=text,
                    output_path=str(out_file_img)
                )
                if self.use_wandb:
                    wandb.log({
                        f"tv_attention_sample_{i}": wandb.Image(str(out_file_img), caption=text),
                        "epoch": epoch,
                        "global_step": self.global_step
                    })

        self.model.train()
        plt.close('all')
        gc.collect()

    ###########################################
    #           Main Training Loop
    ###########################################
    def train(self):
        """
        The main training loop, run for num_epochs.
        On each step:
         1) get a batch from the AV dataset
         2) get a batch from the TV dataset
         3) forward => compute av_loss + tv_loss
         4) backward & step
         5) log & checkpointing / visualization as needed

        Step-based unfreezing is applied at the beginning of each step.
        """
        total_epochs = self.config['num_epochs']
        accumulation_counter = 0

        for epoch in range(self.start_epoch, total_epochs):
            self.logger.info(f"Epoch {epoch} starting")

            # Fresh iterators
            self.av_iter = iter(self.av_dataloader)
            self.tv_iter = iter(self.tv_dataloader)
            max_steps = max(len(self.av_dataloader), len(self.tv_dataloader))
            pbar = tqdm(range(max_steps), desc=f"Epoch {epoch}")
            epoch_losses = []

            # If we're resuming mid-epoch, skip the first self.current_batch_idx steps
            for _ in range(self.current_batch_idx):
                try:
                    next(self.av_iter)
                except StopIteration:
                    self.av_iter = iter(self.av_dataloader)
                    next(self.av_iter)

                try:
                    next(self.tv_iter)
                except StopIteration:
                    self.tv_iter = iter(self.tv_dataloader)
                    next(self.tv_iter)

            for batch_idx in pbar:
                if batch_idx < self.current_batch_idx:
                    # still "catching up" from the skip
                    continue

                # Step-based unfreeze check:
                self._adjust_lr_for_frozen_params(self.global_step)

                # 1) AudioVisual batch
                try:
                    av_batch = next(self.av_iter)
                except StopIteration:
                    self.av_iter = iter(self.av_dataloader)
                    av_batch = next(self.av_iter)

                frames_av = av_batch['frame'].to(self.device)
                audio_av  = av_batch['audio'].to(self.device)

                # 2) Text-Visual batch
                try:
                    tv_batch = next(self.tv_iter)
                except StopIteration:
                    self.tv_iter = iter(self.tv_dataloader)
                    tv_batch = next(self.tv_iter)

                frames_tv = tv_batch['images'].to(self.device)
                texts_tv  = tv_batch['captions']

                # 3) Forward => compute losses
                self.model.train()
                loss_av = self.model.forward_audio_visual(frames_av, audio_av)
                loss_tv = self.model.forward_text_visual(frames_tv, texts_tv)
                loss_total = loss_av + loss_tv

                # 4) Backward & step (with gradient accumulation)
                loss_scaled = loss_total / self.gradient_accumulation_steps
                loss_scaled.backward()
                accumulation_counter += 1

                if accumulation_counter % self.gradient_accumulation_steps == 0:
                    clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                # Logging
                loss_val = loss_total.item()
                epoch_losses.append(loss_val)
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                if self.use_wandb:
                    wandb_dict = {
                        "train_loss": loss_val,
                        "loss_av": loss_av.item(),
                        "loss_tv": loss_tv.item(),
                        "epoch": epoch,
                        "global_step": self.global_step,
                    }
                    for i, pg in enumerate(self.optimizer.param_groups):
                        wandb_dict[f"lr_group{i}"] = pg["lr"]
                    wandb.log(wandb_dict)

                

                del frames_av, audio_av, frames_tv, texts_tv, loss_total
                torch.cuda.empty_cache()

                # Housekeeping
                if self.global_step % 500 == 0:
                    gc.collect()

                # Visualization
                if self.global_step % self.vis_every == 0:
                    self.visualize_samples(epoch)

                # Save checkpoint periodically
                if self.global_step % self.save_every_steps == 0:
                    self.current_batch_idx = batch_idx + 1
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1

            # End of epoch
            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f"Epoch {epoch} completed. Average loss={epoch_loss:.4f}")

            # Reset batch offset at end of an epoch
            self.current_batch_idx = 0

            # Save final checkpoint for the epoch
            self.save_checkpoint(epoch, self.global_step)

        self.logger.info("Training complete!")


###########################################
#        Main Script
###########################################
if __name__ == "__main__":
    print("Starting multi-modal training example...")

    # Example usage:
    trainer = MultiModalTrainer(
        audio_visual_data_root="/home/cis/GodSet",
        text_dataset_path="/home/cis/cc3m",
        output_dir="./outputs_multimodal_cc3m",
        batch_size_av=16,
        batch_size_tv=16,
        num_epochs=10,
        learning_rate=1e-4,
        use_wandb=True,
        force_new_training=True,
        vis_every=5000,
        save_every_steps=5000,
        num_workers=8,
        device="cuda",
        gradient_accumulation_steps=3,
        unfreeze_audio_step=70000,
        unfreeze_text_step=70000,
        unfreeze_vit_step=70000,
        project_name="Triad-cc3m-godset_fudge",
        num_vis_samples_av=15,
        num_vis_samples_tv=15,
    )

    trainer.train()
