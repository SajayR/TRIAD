import os
import warnings
import multiprocessing
from pathlib import Path
from urllib.parse import urlparse
import av
import datasets
import numpy as np
import random
import torch
import torch.nn as nn
import torchaudio.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms
from typing import Dict, List

warnings.filterwarnings("ignore")

try:
    multiprocessing.set_start_method('fork', force=True)
except:
    multiprocessing.set_start_method('spawn', force=True)
import gc
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

class LocalCaptionDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform or transforms.Compose([
            # These augmentations work even with 224×224 images
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Small translations
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Random erasing for robustness
            
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        
        # Clean transform for visualization
        self.clean_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])

        self.image_files = []
        for subdir in self.root_dir.iterdir():
            if subdir.is_dir():
                self.image_files.extend(list(subdir.glob("*.jpg")))
        print(f"Found {len(self.image_files)} images in {self.root_dir}")
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        txt_path = img_path.with_suffix('.txt')
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            with open(txt_path, 'r') as f:
                caption = f.read().strip()
                
            return image, caption
            
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will show the full stack trace
            return torch.zeros((3, 224, 224)), ""


def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    container = None
    try:
        container = av.open(str(video_path))
        audio = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
        
        samples = []
        for frame in container.decode(audio):
            frame.pts = None
            resampled = resampler.resample(frame)
            if resampled:  
                frame = resampled[0]
                samples.append(frame.to_ndarray().reshape(-1))

        if not samples:  
            return torch.zeros(16331)

        samples = torch.tensor(np.concatenate(samples))
        samples = samples.float() / 32768.0  
        return samples
    except Exception as e:
        print(f"Failed to load audio from {video_path}: {str(e)}")
        return torch.zeros(16331)
    finally:
        if container:
            container.close()


def load_and_preprocess_video(video_path: str, sample_fps: int, apply_augmentation=True) -> torch.Tensor:
    """Load only one random frame from the 1s video using PyAV, resize, and normalize."""
    container = av.open(video_path)
    video_stream = container.streams.video[0]
    original_fps = float(video_stream.average_rate)
    video_duration = 1.0
    num_original_frames = int(round(original_fps * video_duration))
    desired_frame_count = int(video_duration * sample_fps)  # equals sample_fps
    frame_indices = np.linspace(0, num_original_frames - 1, desired_frame_count, dtype=int)
    chosen_index = frame_indices[np.random.randint(0, desired_frame_count)]
    chosen_time_seconds = chosen_index / original_fps
    chosen_pts = int(chosen_time_seconds / video_stream.time_base)

    container.seek(chosen_pts, stream=video_stream, any_frame=False, backward=True)
    closest_frame = None
    min_pts_diff = float('inf')
    for frame in container.decode(video_stream):
        pts_diff = abs(frame.pts - chosen_pts)
        if pts_diff < min_pts_diff:
            min_pts_diff = pts_diff
            closest_frame = frame
        # gone too far past our target, stop
        if frame.pts > chosen_pts + original_fps/10:  # 1/10th second overshoot
            break
    
    container.close()
    if closest_frame is None:
        raise ValueError(f"Failed to find appropriate frame for index {chosen_index}")
    decoded_frame = closest_frame.to_rgb().to_ndarray()
    frame_tensor = torch.from_numpy(decoded_frame).permute(2, 0, 1).float() / 255.0
    
    # Resize first to 256x256 to allow for random cropping
    frame_tensor = torch.nn.functional.interpolate(
        frame_tensor.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    if apply_augmentation:
        # Apply random augmentations
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(frame_tensor, output_size=(224, 224))
        frame_tensor = frame_tensor[:, i:i+h, j:j+w]
        
        # Random horizontal flip
        if random.random() > 0.5:
            frame_tensor = torch.flip(frame_tensor, [2])
            
        # Color jitter (manually applying since we're using tensors)
        if random.random() > 0.5:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            
            # Apply brightness
            frame_tensor = frame_tensor * brightness
            
            # Apply contrast
            mean = torch.mean(frame_tensor, dim=[1, 2], keepdim=True)
            frame_tensor = (frame_tensor - mean) * contrast + mean
            
            # Apply saturation (simplified)
            grayscale = torch.mean(frame_tensor, dim=0, keepdim=True).repeat(3, 1, 1)
            frame_tensor = frame_tensor * saturation + grayscale * (1 - saturation)
    else:
        # Simple center crop for visualization
        frame_tensor = transforms.functional.center_crop(frame_tensor, (224, 224))
    
    # Clamp values to be in valid range [0, 1]
    frame_tensor = torch.clamp(frame_tensor, 0, 1)
    
    # Apply normalization
    frame_tensor = (frame_tensor - IMAGENET_MEAN) / IMAGENET_STD
    return frame_tensor

class VideoBatchSampler(Sampler):  #point is to sample videos with different vid_nums in a batch
    def __init__(self, vid_nums: List[int], batch_size: int):
        self.vid_nums = np.array(vid_nums)
        self.batch_size = batch_size
        self.total_samples = len(vid_nums)

    def __iter__(self):
        all_indices = list(range(self.total_samples))
        random.shuffle(all_indices)
        
        current_batch = []
        used_vids = set()
        
        for idx in all_indices:
            vid = self.vid_nums[idx]
            if vid not in used_vids:
                current_batch.append(idx)
                used_vids.add(vid)
                if len(current_batch) == self.batch_size:
                    yield current_batch
                    current_batch = []
                    used_vids = set()
        
        if current_batch:
            yield current_batch
    
    def __len__(self):
        return self.total_samples // self.batch_size

class AudioVisualDataset(Dataset):
    def __init__(self, data_root: str, sample_fps: int = 20):
        self.data_root = Path(data_root)
        self.sample_fps = sample_fps
        self.segment_folders = sorted([d for d in self.data_root.iterdir() if d.is_dir()], 
                                    key=lambda x: int(x.name.split('_')[1]))
        self.segment_to_videos = {}
        for folder in self.segment_folders:
            segment_num = int(folder.name.split('_')[1])
            self.segment_to_videos[segment_num] = sorted(list(folder.glob("*.mp4")))
        self.current_segment = int((self.segment_folders[0].name).split('_')[1])
        self.video_files = self.segment_to_videos[self.current_segment]

    def switch_segment(self):
        """Randomly switch to a different segment"""
        available_segments = list(self.segment_to_videos.keys())
        available_segments.remove(self.current_segment)
        if available_segments:
            self.current_segment = random.choice(available_segments)
            self.video_files = self.segment_to_videos[self.current_segment]
            print(f"Switching segment to {self.current_segment}")
            
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx, apply_augmentation=True):
        """Get dataset item with option to apply augmentation"""
        video_path = self.video_files[idx]
        try: 
            audio = extract_audio_from_video(video_path)
        except Exception as e:
            print(f"Error processing {video_path} audio: {str(e)}")
            audio = torch.zeros(16331)
            
        try:
            video_frame = load_and_preprocess_video(str(video_path), self.sample_fps, apply_augmentation=apply_augmentation)
        except Exception as e:
            print(f"Error processing {video_path} video frame: {str(e)}")
            video_frame = torch.zeros(3, 224, 224)
            
        return {
            'video_path': str(video_path),
            'video_frames': video_frame, 
            'audio': audio,
            #'vid_num': int(video_path.stem.split('_')[0]),
            #'segment_num': self.current_segment
        }
        

def collate_fn(batch):
    video_tokens = torch.stack([item['video_frames'] for item in batch])
    max_audio_len = max(item['audio'].shape[0] for item in batch)
    audio_padded = torch.zeros(len(batch), max_audio_len)
    for i, item in enumerate(batch):
        audio_len = item['audio'].shape[0]
        audio_padded[i, :audio_len] = item['audio']
    
    return {
        'frame': video_tokens,
        'audio': audio_padded,
        #'vid_nums': [item['vid_num'] for item in batch],
        #'segment_nums': [item['segment_num'] for item in batch],
        'video_paths': [str(item['video_path']) for item in batch]
    }


if __name__ == "__main__":
    print("Testing LocalCaptionDataset...")
    dataset = LocalCaptionDataset("/home/cis/cc3m")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print("\nTesting batch loading...")
    for batch_idx, (images, captions) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Image batch shape: {images.shape}")  # Should be [4, 3, 224, 224]
        print(f"Sample caption: {captions[0]}")
        break

    print("Testing AudioVisualDataset with segmented structure...")
    
    dataset = AudioVisualDataset(
        data_root="/home/cis/GodSet",
        sample_fps=20
    )
    dataloader = DataLoader(
        dataset,
        batch_size=18,
        shuffle=True,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=3
    )
    for batch_idx, batch in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Current segment: {dataset.current_segment}")
        print(f"Frame shape: {batch['frame'].shape}")
        print(f"Audio shape: {batch['audio'].shape}")
        print(f"Sample video path: {batch['video_paths'][0]}")
        if batch_idx % 3 == 0:
            dataset.switch_segment()
            
        