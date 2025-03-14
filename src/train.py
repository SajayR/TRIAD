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
from PIL import Image
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
    Trainer that sums AudioVisual and TextVisual losses in the same step,
    but uses multiple optimizers & schedulers:
      - One param group for "others" (always unfrozen from step=0).
      - One param group for audio (unfreeze at unfreeze_audio_step).
      - One param group for text  (unfreeze at unfreeze_text_step).
      - One param group for vision(unfreeze at unfreeze_vit_step).

    Each param group has a separate OneCycleLR that starts its warmup
    the moment that group is unfrozen.
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
        force_new_training: bool = False,
        use_amp: bool = True
    ):
        """
        Args:
            audio_visual_data_root: path to video dataset (mp4 files)
            text_dataset_path:      path to images & text files (LocalCaptionDataset)
            output_dir:             where to save logs/checkpoints
            ...
            unfreeze_audio_step, unfreeze_text_step, unfreeze_vit_step:
                after these step counts, set requires_grad=True for each module
                and start their OneCycleLR from step=0 to (total_updates - unfreeze_step).
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.vis_every = vis_every
        self.save_every_steps = save_every_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
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
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.use_amp = use_amp
        # -----------------------------------------------------
        #  1) Datasets / Dataloaders
        # -----------------------------------------------------
        print("Loading AudioVisualDataset...")
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
            prefetch_factor=3
        )
        print("AudioVisualDataset loaded")
        print("Loading LocalCaptionDataset...")
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
        print("LocalCaptionDataset loaded")
        self.av_iter = None
        self.tv_iter = None

        # -----------------------------------------------------
        #  2) Model
        # -----------------------------------------------------
        self.model = MultiModalModel(
            audio_model_name="facebook/hubert-base-ls960",
            # text_model_name="answerdotai/ModernBERT-base",
            text_model_name="distilbert/distilbert-base-uncased",
            temperature=2.0,
            patch_sparsity_threshold=1.5,
            patch_sparsity_weight=0.5,
            visual_dropout_prob=0.05,
            use_amp=use_amp
        ).to(self.device)
        #enabling gradient checkpointing
        #self.model.audio_embedder.hubert.gradient_checkpointing_enable()
        #self.model.text_embedder.encoder.gradient_checkpointing_enable()
        #1self.model.visual_embedder.model.gradient_checkpointing = True

        # -----------------------------------------------------
        #  3) Separate param groups => separate optimizers
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

        # (a) Optimizer for "others" (train from start)
        self.opt_others = torch.optim.AdamW(
            others_params, lr=learning_rate
        )
        # (b) Audio optimizer (frozen at start)
        self.opt_audio = torch.optim.AdamW(
            audio_params, lr=learning_rate
        )
        # (c) Text optimizer
        self.opt_text = torch.optim.AdamW(
            text_params, lr=learning_rate
        )
        # (d) Vision optimizer
        self.opt_vit = torch.optim.AdamW(
            vit_params, lr=learning_rate
        )

        # We'll freeze audio/text/vit from step=0:
        for p in audio_params:
            p.requires_grad = False
        for p in text_params:
            p.requires_grad = False
        for p in vit_params:
            p.requires_grad = False

        # -----------------------------------------------------
        #  4) Multiple OneCycle schedulers
        # -----------------------------------------------------
        total_steps_per_epoch = max(len(self.av_dataloader), len(self.tv_dataloader))
        self.steps_per_epoch = total_steps_per_epoch
        self.total_updates = (self.steps_per_epoch * num_epochs) // self.gradient_accumulation_steps

        # We want "others" to run from 0 to self.total_updates
        self.sched_others = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_others,
            max_lr=learning_rate,
            total_steps=self.total_updates,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        # Audio starts at unfreeze_audio_step => full cycle from there to end
        audio_cycle = max(1, self.total_updates - unfreeze_audio_step)
        self.sched_audio = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_audio,
            max_lr=learning_rate,
            total_steps=audio_cycle,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        # Text
        text_cycle = max(1, self.total_updates - unfreeze_text_step)
        self.sched_text = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_text,
            max_lr=learning_rate,
            total_steps=text_cycle,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        # Vision
        vit_cycle = max(1, self.total_updates - unfreeze_vit_step)
        self.sched_vit = torch.optim.lr_scheduler.OneCycleLR(
            self.opt_vit,
            max_lr=learning_rate,
            total_steps=vit_cycle,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )

        # Local step counters for each scheduler
        self.step_others = 0
        self.step_audio  = 0
        self.step_text   = 0
        self.step_vit    = 0

        # -----------------------------------------------------
        #  5) State tracking & optional resume
        # -----------------------------------------------------
        self.start_epoch = 0
        self.global_step = 0
        self.current_batch_idx = 0
        self.best_loss = float('inf')
        
        print("Force new training: ", force_new_training)
        print("Use wandb: ", self.use_wandb)
        if not force_new_training:
            print("Loading checkpoint")
            ckpt = self.find_latest_checkpoint()
            print("Checkpoint found: ", ckpt)
            if ckpt:
                self.load_checkpoint(ckpt)
                print("Checkpoint loaded")
            elif self.use_wandb:
                print("No checkpoint found")
        elif self.use_wandb and force_new_training:
            wandb.init(project=self.project_name, name="Triad-webow", config=self.config)
        if self.use_wandb and wandb.run is None:
            wandb.init(project=self.project_name, name="Triad-webow", config=self.config)

        # Visualization
        self.audio_viz = AudioVisualizer()
        self.text_viz  = TextVisualizer()
        self.vis_samples_av = self._get_av_vis_samples(num_vis_samples_av)
        self.vis_samples_tv = self._get_tv_vis_samples(num_vis_samples_tv)

    

        print("Compiling model")
        self.model = torch.compile(self.model, mode="max-autotune")

        

        self.logger.info("Initialized MultiModalTrainer with multiple schedulers.")


    ###########################################
    #     Checkpointing Helpers
    ###########################################
    def find_latest_checkpoint(self):
        ckpts = list(self.output_dir.glob("checkpoint_epoch*_step*.pt"))
        if not ckpts:
            return None

        def _parse_ckpt_name(p):
            """
            e.g. 'checkpoint_epoch3_step1500.pt' => (3,1500)
            """
            name = p.stem
            ep_str = name.split('epoch')[1].split('_')[0]  # '3'
            st_str = name.split('step')[1]                 # '1500'
            return (int(ep_str), int(st_str))

        return max(ckpts, key=_parse_ckpt_name)

    def save_checkpoint(self, epoch, step):
        ckpt_path = self.output_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
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
            "current_segment": self.av_dataset.current_segment,
            "rng_state": rng_state,
            "model_state_dict": self.model._orig_mod.state_dict(),
            "opt_others_state": self.opt_others.state_dict(),
            "opt_audio_state":  self.opt_audio.state_dict(),
            "opt_text_state":   self.opt_text.state_dict(),
            "opt_vit_state":    self.opt_vit.state_dict(),
            "sched_others_state": self.sched_others.state_dict(),
            "sched_audio_state":  self.sched_audio.state_dict(),
            "sched_text_state":   self.sched_text.state_dict(),
            "sched_vit_state":    self.sched_vit.state_dict(),
            "sched_step_others": self.step_others,
            "sched_step_audio":  self.step_audio,
            "sched_step_text":   self.step_text,
            "sched_step_vit":    self.step_vit,
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
        #adding cross compatibility with both checkpoints with orig_mod and without
        if "_orig_mod." in list(ck["model_state_dict"].keys())[0]:
            print("Checkpoint with _orig_mod. found")
            state_dict = ck["model_state_dict"]
            clean_state_dict = {}
            clean_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    clean_key = key[len('_orig_mod.'):]
                    clean_state_dict[clean_key] = value
                else:
                    clean_state_dict[key] = value
            self.model.load_state_dict(clean_state_dict)
        else:
            self.model.load_state_dict(ck["model_state_dict"])
        self.opt_others.load_state_dict(ck["opt_others_state"])
        self.opt_audio.load_state_dict(ck["opt_audio_state"])
        self.opt_text.load_state_dict(ck["opt_text_state"])
        self.opt_vit.load_state_dict(ck["opt_vit_state"])
        self.sched_others.load_state_dict(ck["sched_others_state"])
        self.sched_audio.load_state_dict(ck["sched_audio_state"])
        self.sched_text.load_state_dict(ck["sched_text_state"])
        self.sched_vit.load_state_dict(ck["sched_vit_state"])
        self.step_others = ck.get("sched_step_others", 0)
        self.step_audio  = ck.get("sched_step_audio", 0)
        self.step_text   = ck.get("sched_step_text", 0)
        self.step_vit    = ck.get("sched_step_vit", 0)
        self.start_epoch = ck["epoch"]
        self.global_step = ck["step"]
        self.current_batch_idx = ck.get("current_batch_idx", 0)
        self.best_loss = ck["best_loss"]
        self.av_dataset.current_segment = ck["current_segment"]

        rng_state = ck.get("rng_state", None)
        if rng_state is not None:
            # torch RNG
            torch_state = rng_state["torch"]
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state.cpu())

            # cuda RNG
            for i, cuda_state in enumerate(rng_state["cuda"]):
                if not isinstance(cuda_state, torch.Tensor):
                    cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                torch.cuda.set_rng_state(cuda_state.cpu(), device=i)

            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])

        self.vis_samples_av = ck["vis_samples_av"]
        self.vis_samples_tv = ck["vis_samples_tv"]
        self._update_frozen_params(self.global_step)
        del ck

        self.logger.info(
            f"Checkpoint loaded (epoch={self.start_epoch}, step={self.global_step}, "
            f"batch_offset={self.current_batch_idx})"
        )


    ###########################################
    #  Freeze/Unfreeze logic
    ###########################################
    def _update_frozen_params(self, current_step: int):
        """
        Switch requires_grad to True if current_step >= unfreeze_*_step,
        else keep them False. This ensures no momentum buildup in optimizers
        for modules that are still frozen.
        """
        # Audio
        audio_module = self.model.audio_embedder.hubert
        if current_step < self.config['unfreeze_audio_step']:
            for p in audio_module.parameters():
                p.requires_grad = False
        else:
            for p in audio_module.parameters():
                p.requires_grad = True
        # Text
        text_module = self.model.text_embedder.encoder
        if current_step < self.config['unfreeze_text_step']:
            for p in text_module.parameters():
                p.requires_grad = False
        else:
            for p in text_module.parameters():
                p.requires_grad = True

        # Vision
        vision_module = self.model.visual_embedder.model
        if current_step < self.config['unfreeze_vit_step']:
            for p in vision_module.parameters():
                p.requires_grad = False
        else:
            for p in vision_module.parameters():
                p.requires_grad = True


    ###########################################
    #     Visualization logic
    ###########################################
    def _get_av_vis_samples(self, n_samples=2):
        """Get clean samples for visualization without augmentation"""
        batch_dataloader = DataLoader(
            self.av_dataset, 
            batch_size=n_samples, 
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn
        )
        batch = next(iter(batch_dataloader))
        
        # Get the raw samples again but without augmentation
        frames = []
        audio = []
        video_paths = []
        
        for i, video_path in enumerate(batch['video_paths']):
            if i >= n_samples:
                break
                
            sample = self.av_dataset.__getitem__(
                self.av_dataset.video_files.index(Path(video_path)), 
                apply_augmentation=False  # Important: no augmentation for viz
            )
            
            frames.append(sample['video_frames'])
            audio.append(sample['audio'])
            video_paths.append(sample['video_path'])
        
        # Use the same padding logic as in the main collate_fn
        max_audio_len = max(a.shape[0] for a in audio)
        audio_padded = torch.zeros(len(audio), max_audio_len)
        for i, a in enumerate(audio):
            audio_len = a.shape[0]
            audio_padded[i, :audio_len] = a
        
        return {
            "frames": torch.stack(frames),
            "audio": audio_padded,  # Use padded audio
            "video_paths": video_paths
        }

    def _get_tv_vis_samples(self, n_samples=2):
        batch = next(iter(self.tv_dataloader))
        # Instead of using batch images directly, get the original files
        # and apply clean_transform
        images = []
        captions = []
        for i in range(min(n_samples, len(batch['images']))):
            img_path = self.tv_dataset.image_files[i]
            img = Image.open(img_path).convert('RGB')
            images.append(self.tv_dataset.clean_transform(img))
            txt_path = img_path.with_suffix('.txt')
            with open(txt_path, 'r') as f:
                captions.append(f.read().strip())
        
        return {
            "images": torch.stack(images),
            "texts": captions
        }

    def visualize_samples(self, epoch):
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
        accumulation_counter = 0
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            self.logger.info(f"Epoch {epoch} starting")
            if self.current_batch_idx == 0 or self.start_epoch == 2:  # Fresh epoch
                print("Switching segment")
                self.av_dataset.switch_segment()
            self.av_iter = iter(self.av_dataloader)
            self.tv_iter = iter(self.tv_dataloader)

            # If resuming in the middle of an epoch
            print(f"Resuming from batch {self.current_batch_idx}")
            for _ in tqdm(range(self.current_batch_idx), desc="Resuming from checkpoint"):
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

            max_steps = max(len(self.av_dataloader), len(self.tv_dataloader))
            pbar = tqdm(range(max_steps-self.current_batch_idx), desc=f"Epoch {epoch}")
            epoch_losses = []

            for batch_idx in pbar:
                self._update_frozen_params(self.global_step)
                try:
                    av_batch = next(self.av_iter)
                except StopIteration:
                    print("StopIteration in av_iter")
                    self.av_iter = iter(self.av_dataloader)
                    av_batch = next(self.av_iter)
                frames_av = av_batch['frame'].to(self.device)
                audio_av  = av_batch['audio'].to(self.device)
                try:
                    tv_batch = next(self.tv_iter)
                except StopIteration:
                    print("StopIteration in tv_iter")
                    self.tv_iter = iter(self.tv_dataloader)
                    tv_batch = next(self.tv_iter)
                frames_tv = tv_batch['images'].to(self.device)
                texts_tv  = tv_batch['captions']
                self.model.train()
                loss_av, av_contrastive, av_reg, av_smooth, av_sim_stats = self.model.forward_audio_visual(frames_av, audio_av)
                loss_tv, tv_sim_stats = self.model.forward_text_visual(frames_tv, texts_tv)
                loss_total = loss_av + loss_tv
                loss_scaled = loss_total / self.gradient_accumulation_steps
                loss_scaled.backward()
                accumulation_counter += 1

                if (accumulation_counter % self.gradient_accumulation_steps) == 0:
                    clip_grad_norm_(self.model.parameters(), 1.0)

                    # ---- Step each optimizer *if* unfrozen. ----
                    # "Others" is always unfrozen
                    self.opt_others.step()
                    self.opt_others.zero_grad()
                    if self.step_others < self.total_updates:
                        self.sched_others.step()
                        self.step_others += 1

                    # Audio if current_step >= unfreeze_audio_step
                    if self.global_step >= self.config['unfreeze_audio_step']:
                        self.opt_audio.step()
                        self.opt_audio.zero_grad()
                        # Step scheduler if we have not exhausted steps
                        if self.step_audio < (self.total_updates - self.config['unfreeze_audio_step']):
                            self.sched_audio.step()
                            self.step_audio += 1
                    else:
                        self.opt_audio.zero_grad()

                    # Text if current_step >= unfreeze_text_step
                    if self.global_step >= self.config['unfreeze_text_step']:
                        self.opt_text.step()
                        self.opt_text.zero_grad()
                        if self.step_text < (self.total_updates - self.config['unfreeze_text_step']):
                            self.sched_text.step()
                            self.step_text += 1
                    else:
                        self.opt_text.zero_grad()

                    # Vision if current_step >= unfreeze_vit_step
                    if self.global_step >= self.config['unfreeze_vit_step']:
                        self.opt_vit.step()
                        self.opt_vit.zero_grad()
                        if self.step_vit < (self.total_updates - self.config['unfreeze_vit_step']):
                            self.sched_vit.step()
                            self.step_vit += 1
                    else:
                        self.opt_vit.zero_grad()
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
                        "lr_others": self.opt_others.param_groups[0]['lr'],
                        "lr_audio":  self.opt_audio.param_groups[0]['lr'],
                        "lr_text":   self.opt_text.param_groups[0]['lr'],
                        "lr_vit":    self.opt_vit.param_groups[0]['lr'],
                        "av_contrastive_loss": av_contrastive.item(),
                        "av_reg_loss": av_reg.item(),
                        "av_smooth_loss": av_smooth.item(),
                        "temperature": self.model.temperature.item(),
                    }

                    wandb_dict.update(av_sim_stats)
                    wandb_dict.update(tv_sim_stats)
                    wandb.log(wandb_dict)

                del frames_av, audio_av, frames_tv, texts_tv, loss_total
                torch.cuda.empty_cache()
                if self.global_step % 500 == 0:
                    gc.collect()
                if (self.global_step > 0) and (self.global_step % self.vis_every == 0):
                    self.visualize_samples(epoch)
                if (self.global_step > 0) and (self.global_step % self.save_every_steps == 0):
                    self.current_batch_idx = batch_idx + 1
                    self.save_checkpoint(epoch, self.global_step)

                self.global_step += 1
            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f"Epoch {epoch} completed. Average loss={epoch_loss:.4f}")
            self.current_batch_idx = 0
            self.save_checkpoint(epoch, self.global_step)

        self.logger.info("Training complete!")


###########################################
#        Main Script
###########################################
if __name__ == "__main__":
    print("Starting multi-modal training example with multiple schedulers...")

    trainer = MultiModalTrainer(
        audio_visual_data_root="/home/cis/GodSet",
        text_dataset_path="/home/cis/cc3m-ironic",
        output_dir="./outputs-webow",
        batch_size_av=22,
        batch_size_tv=22,
        num_epochs=10,  
        learning_rate=1e-4,
        use_wandb=True,
        force_new_training=False,
        vis_every=5000,
        save_every_steps=5000,
        num_workers=8,
        device="cuda",
        gradient_accumulation_steps=3,
        unfreeze_audio_step=5000,
        unfreeze_text_step=5000,
        unfreeze_vit_step=5000,
        project_name="Triad",
        num_vis_samples_av=24,
        num_vis_samples_tv=24,
        use_amp=True
    )

    trainer.train()
