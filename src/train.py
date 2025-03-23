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
from torchvision import transforms

from dataset import (
    AudioVisualDataset, collate_fn,
    LocalCaptionDataset, FlatAudioVisualDataset
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
        use_amp: bool = True,
        # New validation parameters
        audio_visual_val_data_root: str = None,
        text_dataset_val_path: str = None,
        validation_frequency: int = 10000,  # Run validation every N epochs
        # New staged training parameters
        av_focus_epochs: int = 3,
        tv_warmup_epochs: int = 1,
        weighted_joint_epochs: int = 2,
        av_weight_start: float = 0.8,
        av_weight_end: float = 0.5,
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
            audio_visual_val_data_root: path to validation video dataset
            text_dataset_val_path: path to validation images & text files
            validation_frequency: run validation every N epochs
            av_focus_epochs: number of epochs to focus only on audio-visual training
            tv_warmup_epochs: number of epochs to focus only on text-visual training
            weighted_joint_epochs: number of epochs to gradually shift from AV to balanced
            av_weight_start: starting weight for AV loss in weighted joint phase
            av_weight_end: ending weight for AV loss in weighted joint phase
        """
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.use_wandb = use_wandb
        self.project_name = project_name
        self.vis_every = vis_every
        self.save_every_steps = save_every_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.validation_frequency = validation_frequency
        
        # Store the new staged training parameters
        self.av_focus_epochs = av_focus_epochs
        self.tv_warmup_epochs = tv_warmup_epochs
        self.weighted_joint_epochs = weighted_joint_epochs 
        self.av_weight_start = av_weight_start
        self.av_weight_end = av_weight_end
        
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
            "save_every_steps": save_every_steps,
            "validation_frequency": validation_frequency,
            "av_focus_epochs": av_focus_epochs,
            "tv_warmup_epochs": tv_warmup_epochs,
            "weighted_joint_epochs": weighted_joint_epochs,
            "av_weight_start": av_weight_start,
            "av_weight_end": av_weight_end
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
            prefetch_factor=3,
            drop_last=True
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
            prefetch_factor=8,
            drop_last=True
        )
        print("LocalCaptionDataset loaded")
        self.av_iter = None
        self.tv_iter = None

        # -----------------------------------------------------
        #  1.1) Validation Datasets / Dataloaders (New)
        # -----------------------------------------------------
        self.val_av_dataset = None
        self.val_tv_dataset = None
        self.val_av_dataloader = None
        self.val_tv_dataloader = None
        
        if audio_visual_val_data_root:
            print("Loading Validation AudioVisualDataset...")
            self.val_av_dataset = FlatAudioVisualDataset(
                data_root=audio_visual_val_data_root,
                sample_fps=20
            )
            self.val_av_dataloader = DataLoader(
                self.val_av_dataset,
                batch_size=batch_size_av,
                shuffle=False,  # No shuffling for validation
                num_workers=num_workers,
                persistent_workers=(num_workers > 0),
                pin_memory=True,
                collate_fn=collate_fn,
                prefetch_factor=3,
                drop_last=True
            )
            print("Validation AudioVisualDataset loaded")
            
        if text_dataset_val_path:
            print("Loading Validation LocalCaptionDataset...")
            # Use clean transform without augmentation for validation
            val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            self.val_tv_dataset = LocalCaptionDataset(
                text_dataset_val_path, 
                transform=val_transform
            )
            self.val_tv_dataloader = DataLoader(
                self.val_tv_dataset,
                batch_size=batch_size_tv,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_text_fn,
                pin_memory=True,
                persistent_workers=(num_workers > 0),
                prefetch_factor=8,
                drop_last=True
            )
            print("Validation LocalCaptionDataset loaded")

        # -----------------------------------------------------
        #  2) Model
        # -----------------------------------------------------
        self.model = MultiModalModel(
            audio_model_name="facebook/hubert-base-ls960",
            # text_model_name="answerdotai/ModernBERT-base",
            text_model_name="distilbert/distilbert-base-uncased",
            temperature=1.2,
            patch_sparsity_threshold=0.80,
            patch_sparsity_weight=0.01,
            visual_dropout_prob=0.2,
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
            max_lr=learning_rate*0.75,
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
            max_lr=learning_rate*0.75,
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
            max_lr=learning_rate*0.3,
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
            wandb.init(project=self.project_name, name="Triad", config=self.config)
        if self.use_wandb and wandb.run is None:
            wandb.init(project=self.project_name, name="Triad", config=self.config)

        # Visualization
        self.audio_viz = AudioVisualizer()
        self.text_viz  = TextVisualizer()
        # Get visualization samples (preferably from validation sets if available)
        self.vis_samples_av = self._get_av_vis_samples(num_vis_samples_av, use_val=bool(self.val_av_dataset))
        self.vis_samples_tv = self._get_tv_vis_samples(num_vis_samples_tv, use_val=bool(self.val_tv_dataset))

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

    def save_checkpoint(self, epoch, step, is_best=False):
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
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(ckpt, best_path)
            self.logger.info(f"Saved best model checkpoint: {best_path}")

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

        # Check for phase configuration differences
        ckpt_config = ck.get("config", {})
        ckpt_av_focus = ckpt_config.get("av_focus_epochs", self.av_focus_epochs)
        ckpt_tv_warmup = ckpt_config.get("tv_warmup_epochs", self.tv_warmup_epochs)
        ckpt_weighted_joint = ckpt_config.get("weighted_joint_epochs", self.weighted_joint_epochs)
        
        if (ckpt_av_focus != self.av_focus_epochs or 
            ckpt_tv_warmup != self.tv_warmup_epochs or
            ckpt_weighted_joint != self.weighted_joint_epochs):
            self.logger.warning(
                f"WARNING: Checkpoint has different training phases! "
                f"Checkpoint: AV focus={ckpt_av_focus}, TV warmup={ckpt_tv_warmup}, "
                f"Weighted joint={ckpt_weighted_joint} | "
                f"Current: AV focus={self.av_focus_epochs}, TV warmup={self.tv_warmup_epochs}, "
                f"Weighted joint={self.weighted_joint_epochs}"
            )
            # Determine if we need to adjust the current epoch to maintain phase consistency
            total_phases_ckpt = ckpt_av_focus + ckpt_tv_warmup + ckpt_weighted_joint
            total_phases_current = self.av_focus_epochs + self.tv_warmup_epochs + self.weighted_joint_epochs
            
            if total_phases_ckpt != total_phases_current:
                self.logger.warning(
                    f"Total phase duration differs: checkpoint={total_phases_ckpt}, current={total_phases_current}. "
                    f"This may affect training continuity."
                )

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
        self.best_loss = ck["best_loss"]
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
    def _get_av_vis_samples(self, n_samples=2, use_val=False):
        """Get clean samples for visualization without augmentation"""
        # Use validation dataset if requested and available
        dataset = self.val_av_dataset if use_val and self.val_av_dataset else self.av_dataset
        
        batch_dataloader = DataLoader(
            dataset, 
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
                
            sample = dataset.__getitem__(
                dataset.video_files.index(Path(video_path)), 
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

    def _get_tv_vis_samples(self, n_samples=2, use_val=False):
        """Get clean samples for visualization without augmentation"""
        # Use validation dataset if requested and available
        dataset = self.val_tv_dataset if use_val and self.val_tv_dataset else self.tv_dataset
        
        # Use dataset's clean transform for consistent visualization
        clean_transform = dataset.clean_transform
        
        # Get a batch from the dataset
        batch_dataloader = DataLoader(
            dataset,
            batch_size=n_samples,
            shuffle=True,
            collate_fn=collate_text_fn
        )
        batch = next(iter(batch_dataloader))
        
        # Get original images with clean transform
        images = []
        captions = []
        for i in range(min(n_samples, len(batch['images']))):
            idx = i  # In this simple case, we just use the index in the batch
            if hasattr(dataset, 'image_files'):  # LocalCaptionDataset
                img_path = dataset.image_files[idx]
                img = Image.open(img_path).convert('RGB')
                images.append(clean_transform(img))
                
                txt_path = img_path.with_suffix('.txt')
                with open(txt_path, 'r') as f:
                    captions.append(f.read().strip())
            else:
                # Fallback to batch images if we can't reconstruct
                images.append(batch['images'][i])
                captions.append(batch['captions'][i])
        
        return {
            "images": torch.stack(images),
            "texts": captions
        }

    def visualize_samples(self, epoch):
        """Generate visualizations based on current training phase."""
        # Determine current phase
        if epoch < self.av_focus_epochs:
            phase = "av_focus"
        elif epoch < self.av_focus_epochs + self.tv_warmup_epochs:
            phase = "tv_warmup"
        elif epoch < self.av_focus_epochs + self.tv_warmup_epochs + self.weighted_joint_epochs:
            phase = "weighted_joint"
        else:
            phase = "full_joint"
            
        self.logger.info(f"Generating visualizations for phase: {phase}")
        self.model.eval()

        # 1) Audio-Visual - only visualize if not in tv_warmup phase
        if phase != "tv_warmup":
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
                            "global_step": self.global_step,
                            "visualization_phase": phase
                        })
        else:
            self.logger.info("Skipping A/V visualization in TV warmup phase")

        # 2) Text-Visual - only visualize if not in av_focus phase
        if phase != "av_focus":
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
                            "global_step": self.global_step,
                            "visualization_phase": phase
                        })
        else:
            self.logger.info("Skipping T/V visualization in AV focus phase")

        self.model.train()
        plt.close('all')
        gc.collect()

    ###########################################
    #           Validation Method (New)
    ###########################################
    def validate(self, phase=None):
        """
        Evaluate model on validation datasets based on current training phase.
        Returns the overall validation loss if validation datasets are available.
        """
        if phase is None:
            # Determine phase based on current epoch if not provided
            if self.start_epoch < self.av_focus_epochs:
                phase = "av_focus"
            elif self.start_epoch < self.av_focus_epochs + self.tv_warmup_epochs:
                phase = "tv_warmup"
            elif self.start_epoch < self.av_focus_epochs + self.tv_warmup_epochs + self.weighted_joint_epochs:
                phase = "weighted_joint"
            else:
                phase = "full_joint"
                
        if not self.val_av_dataloader and not self.val_tv_dataloader:
            self.logger.info("No validation datasets provided. Skipping validation.")
            return None, None, None
            
        self.model.eval()
        av_losses = []
        tv_losses = []
        av_contrastive_losses = []
        av_reg_losses = []
        av_smooth_losses = []
        av_sim_stats_list = []
        tv_sim_stats_list = []
        
        # Only validate A/V if not in text-visual warmup phase
        if (phase != "tv_warmup") and self.val_av_dataloader:
            with torch.no_grad():
                for av_batch in tqdm(self.val_av_dataloader, desc="Validating A/V"):
                    frames_av = av_batch['frame'].to(self.device)
                    audio_av = av_batch['audio'].to(self.device)
                    loss_av, av_contrastive, av_reg, av_smooth, av_sim_stats = self.model.forward_audio_visual(frames_av, audio_av)
                    av_losses.append(loss_av.item())
                    av_contrastive_losses.append(av_contrastive.item())
                    av_reg_losses.append(av_reg.item())
                    av_smooth_losses.append(av_smooth.item())
                    av_sim_stats_list.append(av_sim_stats)
        
        # Only validate T/V if not in audio-visual focus phase
        if (phase != "av_focus") and self.val_tv_dataloader:
            with torch.no_grad():
                for tv_batch in tqdm(self.val_tv_dataloader, desc="Validating T/V"):
                    frames_tv = tv_batch['images'].to(self.device)
                    texts_tv = tv_batch['captions']
                    loss_tv, tv_sim_stats = self.model.forward_text_visual(frames_tv, texts_tv)
                    tv_losses.append(loss_tv.item())
                    tv_sim_stats_list.append(tv_sim_stats)
        
        # Compute average metrics
        avg_av_loss = np.mean(av_losses) if av_losses else None
        avg_tv_loss = np.mean(tv_losses) if tv_losses else None
        avg_av_contrastive = np.mean(av_contrastive_losses) if av_contrastive_losses else None
        avg_av_reg = np.mean(av_reg_losses) if av_reg_losses else None
        avg_av_smooth = np.mean(av_smooth_losses) if av_smooth_losses else None
        
        # Calculate total validation loss based on current phase
        val_total_loss = None
        if phase == "av_focus" and avg_av_loss is not None:
            val_total_loss = avg_av_loss
        elif phase == "tv_warmup" and avg_tv_loss is not None:
            val_total_loss = avg_tv_loss
        elif phase == "weighted_joint" and avg_av_loss is not None and avg_tv_loss is not None:
            # Use same weighting as training
            current_epoch = self.start_epoch  # Use actual current epoch
            progress = min(1.0, max(0.0, (current_epoch - (self.av_focus_epochs + self.tv_warmup_epochs)) / self.weighted_joint_epochs))
            av_weight = self.av_weight_start - progress * (self.av_weight_start - self.av_weight_end)
            tv_weight = 1.0 - av_weight
            val_total_loss = av_weight * avg_av_loss + tv_weight * avg_tv_loss
        elif avg_av_loss is not None and avg_tv_loss is not None:
            val_total_loss = avg_av_loss + avg_tv_loss
        elif avg_av_loss is not None:
            val_total_loss = avg_av_loss
        elif avg_tv_loss is not None:
            val_total_loss = avg_tv_loss
        
        # Average similarity stats across batches
        avg_av_sim_stats = {}
        if av_sim_stats_list:
            for key in av_sim_stats_list[0].keys():
                avg_av_sim_stats[f"val_{key}"] = np.mean([stats[key] for stats in av_sim_stats_list if key in stats])
        
        avg_tv_sim_stats = {}
        if tv_sim_stats_list:
            for key in tv_sim_stats_list[0].keys():
                avg_tv_sim_stats[f"val_{key}"] = np.mean([stats[key] for stats in tv_sim_stats_list if key in stats])
        
        # Log validation metrics
        if self.use_wandb:
            wandb_dict = {
                "epoch": self.start_epoch,
                "global_step": self.global_step,
                "validation_phase": phase
            }
            
            if avg_av_loss is not None:
                wandb_dict["val_loss_av"] = avg_av_loss
                wandb_dict["val_av_contrastive_loss"] = avg_av_contrastive
                wandb_dict["val_av_reg_loss"] = avg_av_reg
                wandb_dict["val_av_smooth_loss"] = avg_av_smooth
            
            if avg_tv_loss is not None:
                wandb_dict["val_loss_tv"] = avg_tv_loss
                
            if val_total_loss is not None:
                wandb_dict["val_loss_total"] = val_total_loss
                
            # Include weighted values during weighted phase
            if phase == "weighted_joint":
                wandb_dict["val_av_weight"] = av_weight
                wandb_dict["val_tv_weight"] = tv_weight
                
            wandb_dict.update(avg_av_sim_stats)
            wandb_dict.update(avg_tv_sim_stats)
            
            wandb.log(wandb_dict)
        
        self.model.train()
        
        return avg_av_loss, avg_tv_loss, val_total_loss

    ###########################################
    #           Main Training Loop
    ###########################################
    def train(self):
        accumulation_counter = 0
        
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            # Determine training phase based on current epoch
            if epoch < self.av_focus_epochs:
                phase = "av_focus"
                self.logger.info(f"Epoch {epoch} - Phase: Audio-Visual Focus")
            elif epoch < self.av_focus_epochs + self.tv_warmup_epochs:
                phase = "tv_warmup"
                self.logger.info(f"Epoch {epoch} - Phase: Text-Visual Warmup")
            elif epoch < self.av_focus_epochs + self.tv_warmup_epochs + self.weighted_joint_epochs:
                phase = "weighted_joint"
                # Calculate dynamic weights
                progress = (epoch - (self.av_focus_epochs + self.tv_warmup_epochs)) / self.weighted_joint_epochs
                av_weight = self.av_weight_start - progress * (self.av_weight_start - self.av_weight_end)
                tv_weight = 1.0 - av_weight
                self.logger.info(f"Epoch {epoch} - Phase: Weighted Joint (AV: {av_weight:.2f}, TV: {tv_weight:.2f})")
            else:
                phase = "full_joint"
                self.logger.info(f"Epoch {epoch} - Phase: Full Joint Training")
                
            # Log phase transition clearly
            if epoch == 0 or self.start_epoch == epoch:
                self.logger.info(f"STARTING PHASE: {phase.upper()}")
            elif epoch == self.av_focus_epochs:
                self.logger.info(f"PHASE TRANSITION: AV FOCUS → TV WARMUP")
            elif epoch == self.av_focus_epochs + self.tv_warmup_epochs:
                self.logger.info(f"PHASE TRANSITION: TV WARMUP → WEIGHTED JOINT")
            elif epoch == self.av_focus_epochs + self.tv_warmup_epochs + self.weighted_joint_epochs:
                self.logger.info(f"PHASE TRANSITION: WEIGHTED JOINT → FULL JOINT")
                
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
            pbar = tqdm(range(max_steps-self.current_batch_idx), desc=f"Epoch {epoch} ({phase})")
            epoch_losses = []

            for batch_idx in pbar:
                self._update_frozen_params(self.global_step)
                
                # Process Audio-Visual data (except in tv_warmup phase)
                av_loss = av_contrastive = av_reg = av_smooth = None
                av_sim_stats = {}
                if phase != "tv_warmup":
                    try:
                        av_batch = next(self.av_iter)
                    except StopIteration:
                        print("StopIteration in av_iter")
                        self.av_iter = iter(self.av_dataloader)
                        av_batch = next(self.av_iter)
                    frames_av = av_batch['frame'].to(self.device)
                    audio_av = av_batch['audio'].to(self.device)
                    
                    av_loss, av_contrastive, av_reg, av_smooth, av_sim_stats = self.model.forward_audio_visual(frames_av, audio_av)
                
                # Process Text-Visual data (except in av_focus phase)
                tv_loss = None
                tv_sim_stats = {}
                if phase != "av_focus":
                    try:
                        tv_batch = next(self.tv_iter)
                    except StopIteration:
                        print("StopIteration in tv_iter")
                        self.tv_iter = iter(self.tv_dataloader)
                        tv_batch = next(self.tv_iter)
                    frames_tv = tv_batch['images'].to(self.device)
                    texts_tv = tv_batch['captions']
                    
                    tv_loss, tv_sim_stats = self.model.forward_text_visual(frames_tv, texts_tv)
                
                # Calculate total loss based on current phase
                if phase == "av_focus":
                    loss_total = av_loss
                    tv_loss = torch.tensor(0.0, device=self.device)  # Dummy for logging
                elif phase == "tv_warmup":
                    loss_total = tv_loss
                    av_loss = av_contrastive = av_reg = av_smooth = torch.tensor(0.0, device=self.device)  # Dummy for logging
                elif phase == "weighted_joint":
                    progress = (epoch - (self.av_focus_epochs + self.tv_warmup_epochs)) / self.weighted_joint_epochs
                    av_weight = self.av_weight_start - progress * (self.av_weight_start - self.av_weight_end)
                    tv_weight = 1.0 - av_weight
                    loss_total = av_weight * av_loss + tv_weight * tv_loss
                else:  # full_joint
                    loss_total = av_loss + tv_loss
                
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
                pbar.set_postfix({"loss": f"{loss_val:.4f}", "phase": phase})

                if self.use_wandb:
                    wandb_dict = {
                        "train_loss": loss_val,
                        "loss_av": av_loss.item() if av_loss is not None else 0.0,
                        "loss_tv": tv_loss.item() if tv_loss is not None else 0.0,
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "training_phase": phase,
                        "lr_others": self.opt_others.param_groups[0]['lr'],
                        "lr_audio": self.opt_audio.param_groups[0]['lr'],
                        "lr_text": self.opt_text.param_groups[0]['lr'],
                        "lr_vit": self.opt_vit.param_groups[0]['lr'],
                        "temperature": self.model.temperature.item(),
                    }

                    # Add phase-specific metrics
                    if phase != "tv_warmup":
                        wandb_dict.update({
                            "av_contrastive_loss": av_contrastive.item(),
                            "av_reg_loss": av_reg.item(),
                            "av_smooth_loss": av_smooth.item(),
                        })
                        wandb_dict.update(av_sim_stats)
                    
                    if phase != "av_focus":
                        wandb_dict.update(tv_sim_stats)
                    
                    # Add weights during weighted joint phase
                    if phase == "weighted_joint":
                        wandb_dict.update({
                            "av_weight": av_weight,
                            "tv_weight": tv_weight,
                        })
                    
                    wandb.log(wandb_dict)

                # Memory cleanup and management
                memory_cleanup = []
                if phase != "tv_warmup" and 'frames_av' in locals():
                    memory_cleanup.extend(['frames_av', 'audio_av'])
                if phase != "av_focus" and 'frames_tv' in locals():
                    memory_cleanup.extend(['frames_tv', 'texts_tv'])
                    
                for var in memory_cleanup:
                    if var in locals():
                        del locals()[var]
                del loss_total
                torch.cuda.empty_cache()
                
                if self.global_step % 500 == 0:
                    gc.collect()
                if (self.global_step > 0) and (self.global_step % self.vis_every == 0):
                    self.visualize_samples(epoch)
                if (self.global_step > 0) and (self.global_step % self.save_every_steps == 0):
                    self.current_batch_idx = batch_idx + 1
                    self.save_checkpoint(epoch, self.global_step)
                if self.global_step > 0 and self.global_step % self.validation_frequency == 0:
                    val_av_loss, val_tv_loss, val_total_loss = self.validate(phase=phase)
                    print(f"Validation_Audio_Visual: {val_av_loss:.4f if val_av_loss else 'N/A'}, "
                          f"Validation_Text_Visual: {val_tv_loss:.4f if val_tv_loss else 'N/A'}, "
                          f"Validation_Total: {val_total_loss:.4f if val_total_loss else 'N/A'}")

                self.global_step += 1
                
            # End of epoch
            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f"Epoch {epoch} completed. Average loss={epoch_loss:.4f}")
            
            # Run validation at the end of each epoch if the frequency criteria is met
            if (self.val_av_dataloader or self.val_tv_dataloader):
                self.logger.info(f"Running validation after epoch {epoch}...")
                val_av_loss, val_tv_loss, val_total_loss = self.validate(phase=phase)
                if val_total_loss is not None:
                    self.logger.info(f"Validation loss after epoch {epoch}: {val_total_loss:.4f}")
                    
                    # Save best model based on validation loss
                    if val_total_loss < self.best_loss:
                        self.best_loss = val_total_loss
                        self.save_checkpoint(epoch, self.global_step, is_best=True)
                        self.logger.info(f"New best model saved with val_loss: {val_total_loss:.4f}")
            
            # Reset batch index for the next epoch
            self.current_batch_idx = 0
            self.save_checkpoint(epoch, self.global_step)

        self.logger.info("Training complete!")


###########################################
#        Main Script
###########################################
if __name__ == "__main__":
    print("Starting multi-modal training with staged approach...")

    trainer = MultiModalTrainer(
        audio_visual_data_root="/home/cis/GodSet",
        text_dataset_path="/home/cis/cc3m-ironic",
        audio_visual_val_data_root="/home/cis/UnGodSet", 
        text_dataset_val_path="/home/cis/cc3m-ironic-val",  
        output_dir="./outputs-staged-training",
        batch_size_av=22,
        batch_size_tv=22,
        num_epochs=12,  
        learning_rate=1e-4,
        use_wandb=True,
        force_new_training=False,
        vis_every=10000,
        save_every_steps=10000,
        num_workers=10,
        device="cuda",
        gradient_accumulation_steps=3,
        unfreeze_audio_step=5000,
        unfreeze_text_step=5000,
        unfreeze_vit_step=5000,
        project_name="TriadNerd-Staged",
        num_vis_samples_av=24,
        num_vis_samples_tv=24,
        use_amp=True,
        validation_frequency=10000,
        av_focus_epochs=3,
        tv_warmup_epochs=1,
        weighted_joint_epochs=2,
        av_weight_start=0.8,
        av_weight_end=0.5,
    )

    trainer.train()
