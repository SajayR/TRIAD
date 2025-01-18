import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb
from model import TextVisualModel
from dataset import LivisDataset
from viz import TextVisualizer
import numpy as np
import matplotlib.pyplot as plt
import gc
import warnings
import datasets
warnings.filterwarnings("ignore")

def collate_fn(batch):
    """Simple collate function for images and text"""
    images, captions, short_captions = zip(*batch)
    images = torch.stack(images)
    return {
        'images': images,
        'captions': short_captions  # Using short captions as specified
    }

class TextVisualTrainer:
    def __init__(
        self,
        hf_dataset,
        output_dir: str,
        batch_size: int = 64,
        num_epochs: int = 100,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        vis_every: int = 1000,
        num_vis_samples: int = 15,
        device: str = 'cuda',
        use_wandb: bool = True,
        force_new_training: bool = False,
        gradient_accumulation_steps: int = 1,
        unfreeze_text_epoch: int = 5,
        unfreeze_vit_epoch: int = 10,
        save_every_steps: int = 3000
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize model and move to device
        self.model = TextVisualModel().to(device)
        
        # Training config
        self.config = {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'num_workers': num_workers,
            'vis_every': vis_every,
            'num_vis_samples': num_vis_samples,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'unfreeze_text_epoch': unfreeze_text_epoch,
            'unfreeze_vit_epoch': unfreeze_vit_epoch,
            'save_every_steps': save_every_steps
        }
        
        # Setup logging
        logging.basicConfig(
            filename=str(self.output_dir / 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Initialize training state
        self.start_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        # Setup dataset and dataloader
        self.dataset = LivisDataset(hf_dataset)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=6
        )

        # Initially freeze backbone models
        for param in self.model.visual_embedder.model.parameters():
            param.requires_grad = False
        for param in self.model.text_embedder.encoder.parameters():
            param.requires_grad = False

        # Setup parameter groups
        projection_params = []
        temperature_params = []
        text_params = []
        vit_params = []

        for name, param in self.model.named_parameters():
            if "text_embedder.encoder" in name:
                text_params.append(param)
            elif "visual_embedder.model" in name:
                vit_params.append(param)
            elif "projection" in name:
                projection_params.append(param)
            elif "temperature" in name:
                temperature_params.append(param)
            else:
                projection_params.append(param)

        # Setup optimizers
        self.optimizer_projection = torch.optim.AdamW([
            {'params': projection_params, 'lr': learning_rate},
            {'params': temperature_params, 'lr': learning_rate}
        ])
        self.optimizer_text = torch.optim.AdamW(
            [{'params': text_params, 'lr': learning_rate * 0.1}]
        )
        self.optimizer_vit = torch.optim.AdamW(
            [{'params': vit_params, 'lr': learning_rate * 0.1}]
        )

        # Setup schedulers
        steps_per_epoch = len(self.dataloader)
        total_steps = steps_per_epoch * num_epochs

        self.scheduler_projection = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_projection,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        
        self.scheduler_text = None
        self.scheduler_vit = None

        # Setup visualization
        self.visualizer = TextVisualizer()
        self.vis_samples = self._get_visualization_samples()

        # Initialize wandb if needed
        if use_wandb and not force_new_training:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                wandb.init(project="TextVisualAlignment", config=self.config)

    def find_latest_checkpoint(self):
        """Find the latest checkpoint in output directory"""
        checkpoints = list(self.output_dir.glob('checkpoint_epoch*.pt'))
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda x: int(x.stem.split('epoch')[1].split('_')[0]))

    def save_checkpoint(self, epoch: int, step: int):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / f'checkpoint_epoch{epoch}_step{step}.pt'
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_projection_state_dict': self.optimizer_projection.state_dict(),
            'optimizer_text_state_dict': self.optimizer_text.state_dict(),
            'optimizer_vit_state_dict': self.optimizer_vit.state_dict(),
            'scheduler_projection_state_dict': self.scheduler_projection.state_dict(),
            'scheduler_text_state_dict': self.scheduler_text.state_dict() if self.scheduler_text else None,
            'scheduler_vit_state_dict': self.scheduler_vit.state_dict() if self.scheduler_vit else None,
            'best_loss': self.best_loss,
            'config': self.config,
            'vis_samples': self.vis_samples
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f'Saved checkpoint to {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer_projection.load_state_dict(checkpoint['optimizer_projection_state_dict'])
        self.optimizer_text.load_state_dict(checkpoint['optimizer_text_state_dict'])
        self.optimizer_vit.load_state_dict(checkpoint['optimizer_vit_state_dict'])
        
        if checkpoint['scheduler_projection_state_dict']:
            self.scheduler_projection.load_state_dict(checkpoint['scheduler_projection_state_dict'])
        
        self.start_epoch = checkpoint['epoch']
        self.global_step = checkpoint['step']
        self.best_loss = checkpoint['best_loss']
        self.vis_samples = checkpoint['vis_samples']
        
        self._set_freeze_state(self.start_epoch)

    def _set_freeze_state(self, current_epoch: int):
        """Update which parameters are frozen based on current epoch"""
        if current_epoch >= self.config['unfreeze_text_epoch']:
            for param in self.model.text_embedder.encoder.parameters():
                param.requires_grad = True
            if self.scheduler_text is None:
                steps_remaining = (self.config['num_epochs'] - current_epoch) * len(self.dataloader)
                self.scheduler_text = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer_text,
                    max_lr=self.config['learning_rate'] * 0.1,
                    total_steps=steps_remaining,
                    pct_start=0.1,
                    div_factor=10,
                    final_div_factor=1e4,
                    anneal_strategy='cos'
                )

        if current_epoch >= self.config['unfreeze_vit_epoch']:
            for param in self.model.visual_embedder.model.parameters():
                param.requires_grad = True
            if self.scheduler_vit is None:
                steps_remaining = (self.config['num_epochs'] - current_epoch) * len(self.dataloader)
                self.scheduler_vit = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer_vit,
                    max_lr=self.config['learning_rate'] * 0.1,
                    total_steps=steps_remaining,
                    pct_start=0.1,
                    div_factor=10,
                    final_div_factor=1e4,
                    anneal_strategy='cos'
                )

    def _get_visualization_samples(self):
        """Get a fixed set of samples for visualization"""
        batch = next(iter(self.dataloader))
        indices = torch.randperm(len(batch['images']))[:self.config['num_vis_samples']]
        return {
            'images': batch['images'][indices].to(self.device),
            'captions': [batch['captions'][i] for i in indices]
        }

    def train(self):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.config['num_epochs']):
            self._set_freeze_state(epoch)
            self.model.train()
            epoch_losses = []
            
            # Print trainable layers
            print(f"\nEpoch {epoch} - Training the following layers:")
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    print(f"  {name}")
            
            pbar = tqdm(self.dataloader, desc=f'Epoch {epoch}')
            accumulation_counter = 0

            for batch in pbar:
                self.model.train()
                images = batch['images'].to(self.device)
                captions = batch['captions']
                
                loss = self.model(images, captions)
                
                if loss.item() > 10:
                    print(f"Skipping batch with loss: {loss.item():.4f}")
                    continue
                
                # Gradient accumulation
                loss = loss / self.config['gradient_accumulation_steps']
                loss.backward()
                
                accumulation_counter += 1
                
                if accumulation_counter % self.config['gradient_accumulation_steps'] == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    
                    # Step optimizers
                    self.optimizer_projection.step()
                    self.scheduler_projection.step()
                    
                    if epoch >= self.config['unfreeze_text_epoch']:
                        self.optimizer_text.step()
                        self.scheduler_text.step()
                    
                    if epoch >= self.config['unfreeze_vit_epoch']:
                        self.optimizer_vit.step()
                        self.scheduler_vit.step()
                    
                    # Zero gradients
                    self.optimizer_projection.zero_grad()
                    self.optimizer_text.zero_grad()
                    self.optimizer_vit.zero_grad()
                
                # Logging
                loss_value = loss.item() * self.config['gradient_accumulation_steps']
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                if self.use_wandb:
                    log_dict = {
                        "train_loss": loss_value,
                        "temperature": self.model.temperature.item(),
                        "projection_lr": self.optimizer_projection.param_groups[0]['lr'],
                        "epoch": epoch,
                        "step": self.global_step
                    }
                    
                    if epoch >= self.config['unfreeze_text_epoch']:
                        log_dict["text_lr"] = self.optimizer_text.param_groups[0]['lr']
                    if epoch >= self.config['unfreeze_vit_epoch']:
                        log_dict["vit_lr"] = self.optimizer_vit.param_groups[0]['lr']
                    
                    wandb.log(log_dict)
                
                # Cleanup
                del images, loss
                torch.cuda.empty_cache()
                
                # Periodic operations
                if self.global_step % 500 == 0:
                    gc.collect()
                
                if self.global_step % self.config['vis_every'] == 0:
                    with torch.no_grad():
                        # Log each sample separately to wandb
                        for i in range(len(self.vis_samples['images'])):
                            self.visualizer.plot_token_attentions(
                                self.model,
                                self.vis_samples['images'][i],
                                self.vis_samples['captions'][i],
                                output_path=self.output_dir / f'viz_epoch{epoch}_step{self.global_step}_sample{i}.png'
                            )
                            if self.use_wandb:
                                # Open the saved image and log to wandb
                                wandb.log({
                                    f"token_attention_sample_{i}": wandb.Image(
                                        str(self.output_dir / f'viz_epoch{epoch}_step{self.global_step}_sample{i}.png'),
                                        caption=self.vis_samples['captions'][i]
                                    ),
                                    "epoch": epoch,
                                    "step": self.global_step
                                })
                            plt.close('all')
                        
                    gc.collect()
                
                if self.global_step % self.config['save_every_steps'] == 0:
                    self.save_checkpoint(epoch, self.global_step)
                
                self.global_step += 1
            
            # End of epoch
            epoch_loss = np.mean(epoch_losses)
            self.logger.info(f'Epoch {epoch} - Loss: {epoch_loss:.4f}')
            
            if self.use_wandb:
                wandb.log({
                    'epoch_loss': epoch_loss,
                    'epoch': epoch
                })
            
            self.save_checkpoint(epoch, self.global_step)
        
        print("Training completed!")

if __name__ == "__main__":
    # Load dataset
    dset = datasets.load_dataset("/home/cis/heyo/DenseRead/livis")
    
    trainer = TextVisualTrainer(
        hf_dataset=dset,
        output_dir='./outputs',
        batch_size=64,
        num_epochs=30,
        learning_rate=1e-4,
        use_wandb=True,
        num_vis_samples=15,
        gradient_accumulation_steps=1,
        vis_every=1000,
        num_workers=12,
        force_new_training=False,
        unfreeze_text_epoch=1,  # Start fine-tuning text encoder after 5 epochs
        unfreeze_vit_epoch=3,  # Start fine-tuning ViT after 10 epochs
        save_every_steps=3000
    )
    
    trainer.train()