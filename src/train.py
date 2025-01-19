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
import math
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

        # 🔴 We have TWO counters now:
        self.global_step = 0   # increments EVERY batch
        self.global_update = 0 # increments EVERY optimizer step

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

        # Freeze backbone models initially
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

        # 🔴 Calculate the REAL total number of optimizer updates (not mini‐batches):
        steps_per_epoch = len(self.dataloader)
        self.updates_per_epoch = math.ceil(steps_per_epoch / gradient_accumulation_steps)
        self.total_updates = self.updates_per_epoch * num_epochs

        # Setup optimizers
        self.optimizer_projection = torch.optim.AdamW([
            {'params': projection_params, 'lr': learning_rate},
            #{'params': temperature_params, 'lr': learning_rate}
        ])
        
        # If text/ViT get 0.1*LR, you might consider lowering that further e.g. 0.01 if still exploding
        self.optimizer_text = torch.optim.AdamW(
            [{'params': text_params, 'lr': learning_rate * 0.1}]
        )
        self.optimizer_vit = torch.optim.AdamW(
            [{'params': vit_params, 'lr': learning_rate * 0.1}]
        )

        # 🔴 Setup the projection's OneCycle scheduler with total number of UPDATES:
        self.scheduler_projection = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer_projection,
            max_lr=learning_rate,
            total_steps=self.total_updates,
            pct_start=0.1,
            div_factor=10,
            final_div_factor=1e4,
            anneal_strategy='cos'
        )
        
        self.scheduler_text = None
        self.scheduler_vit = None

        # Visualization
        self.visualizer = TextVisualizer()
        self.vis_samples = self._get_visualization_samples()

        # Initialize wandb if needed
        if use_wandb and not force_new_training:
            checkpoint_path = self.find_latest_checkpoint()
            if checkpoint_path:
                self.load_checkpoint(checkpoint_path)
            else:
                wandb.init(project="TextVisualAlignment", config=self.config)
        if use_wandb and wandb.run is None:
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
            'step': step,  # batch-wise step
            'update': self.global_update,  # how many times we stepped optim
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
        self.global_step = checkpoint['step']   # batch-level
        self.global_update = checkpoint.get('update', 0)  # how many times we've stepped optim
        self.best_loss = checkpoint['best_loss']
        self.vis_samples = checkpoint['vis_samples']
        
        # Make sure we get the right freeze state after loading
        self._set_freeze_state(self.start_epoch)

        # Now if text/vit schedulers were in the checkpoint, load them
        if checkpoint.get('scheduler_text_state_dict'):
            if self.scheduler_text is not None:
                self.scheduler_text.load_state_dict(checkpoint['scheduler_text_state_dict'])
        if checkpoint.get('scheduler_vit_state_dict'):
            if self.scheduler_vit is not None:
                self.scheduler_vit.load_state_dict(checkpoint['scheduler_vit_state_dict'])

    def _set_freeze_state(self, current_epoch: int):
        """Update which parameters are frozen based on current epoch"""
        # If it's time to unfreeze text
        if current_epoch >= self.config['unfreeze_text_epoch']:
            for param in self.model.text_embedder.encoder.parameters():
                param.requires_grad = True
            
            if self.scheduler_text is None:
                # 🔴 We still want to schedule across remaining *updates*, not batches.
                #    So the remaining updates is total_updates - self.global_update
                steps_remaining = self.total_updates - self.global_update
                # Potentially reduce the text LR further if it still explodes
                text_max_lr = self.config['learning_rate'] * 0.1

                self.scheduler_text = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer_text,
                    max_lr=text_max_lr,
                    total_steps=steps_remaining,
                    pct_start=0.3,
                    div_factor=1000,
                    final_div_factor=1e3,
                    anneal_strategy='cos'
                )

        # If it's time to unfreeze the visual encoder
        if current_epoch >= self.config['unfreeze_vit_epoch']:
            for param in self.model.visual_embedder.model.parameters():
                param.requires_grad = True

            if self.scheduler_vit is None:
                steps_remaining = self.total_updates - self.global_update
                vit_max_lr = self.config['learning_rate'] * 0.2

                self.scheduler_vit = torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer_vit,
                    max_lr=vit_max_lr,
                    total_steps=steps_remaining,
                    pct_start=0.2,
                    div_factor=100,
                    final_div_factor=1e3,
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
                
                # forward pass
                loss = self.model(images, captions)

                if loss.item()>15:
                    print(f"Skipping batch with loss {loss.item()}")
                    continue
                #check if loss is nan
                if torch.isnan(loss):
                    print(f"Skipping batch with nan loss")
                    continue
                #check if loss is inf
                if torch.isinf(loss):
                    print(f"Skipping batch with inf loss")
                    continue
                
                # gradient accumulation
                loss = loss / self.config['gradient_accumulation_steps']
                loss.backward()
                
                accumulation_counter += 1
                
                # only step optimizers + schedulers after the desired # of accumulation steps
                if accumulation_counter % self.config['gradient_accumulation_steps'] == 0:
                    # Clip gradients to help prevent exploding
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    
                    # 🔴 Step each optimizer *and* increment self.global_update 
                    self.optimizer_projection.step()
                    self.optimizer_projection.zero_grad()
                    
                    self.scheduler_projection.step()  # step projection scheduler
                    if epoch >= self.config['unfreeze_text_epoch']:
                        self.optimizer_text.step()
                        self.optimizer_text.zero_grad()
                        if self.scheduler_text is not None:
                            self.scheduler_text.step()

                    if epoch >= self.config['unfreeze_vit_epoch']:
                        self.optimizer_vit.step()
                        self.optimizer_vit.zero_grad()
                        if self.scheduler_vit is not None:
                            self.scheduler_vit.step()

                    # 🔴 increment the number of optimizer updates
                    self.global_update += 1
                
                # Logging
                loss_value = loss.item() * self.config['gradient_accumulation_steps']
                epoch_losses.append(loss_value)
                pbar.set_postfix({'loss': f'{loss_value:.4f}'})
                
                if self.use_wandb:
                    log_dict = {
                        "train_loss": loss_value,
                        #"temperature": self.model.temperature.item(),
                        # show LR from projection (param group 0)
                        "projection_lr": self.optimizer_projection.param_groups[0]['lr'],
                        "epoch": epoch,
                        "global_batch_step": self.global_step,
                        "global_update_step": self.global_update,
                    }
                    
                    if epoch >= self.config['unfreeze_text_epoch']:
                        log_dict["text_lr"] = self.optimizer_text.param_groups[0]['lr']
                    if epoch >= self.config['unfreeze_vit_epoch']:
                        log_dict["vit_lr"] = self.optimizer_vit.param_groups[0]['lr']
                    
                    wandb.log(log_dict)
                
                # increment the number of *batches* processed
                self.global_step += 1
                
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
        batch_size=40,
        num_epochs=30,
        learning_rate=1e-4,
        use_wandb=True,
        num_vis_samples=15,
        gradient_accumulation_steps=2,
        vis_every=1000,
        num_workers=12,
        force_new_training=True,
        unfreeze_text_epoch=1,
        unfreeze_vit_epoch=5,
        save_every_steps=3000
    )
    
    trainer.train()
