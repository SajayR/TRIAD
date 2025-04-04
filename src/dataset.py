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
import torchaudio
warnings.filterwarnings("ignore")
from torchcodec.decoders import VideoDecoder
import random
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
            # Geometric transformations (on PIL image)
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Small translations
            # Convert to tensor (0-1 range)
            transforms.ToTensor(),
            # Color transformations (on tensor)
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            # Normalize with ImageNet stats
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
            #transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Optional: Random erasing at the end
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
    try:
        waveform, sample_rate = torchaudio.load(video_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        return waveform[0]
    except Exception as e:
        print(f"Failed to load audio with torchaudio from {video_path}: {str(e)}")
        return torch.zeros(16331)



def load_and_preprocess_video(video_path: str, sample_fps: int, apply_augmentation=True) -> torch.Tensor:
    decoder = VideoDecoder(source=video_path)
    num_total_frames = decoder.metadata.num_frames
    frame_index = random.randint(0, num_total_frames - 1)
    frame = decoder[frame_index]
    frame = frame.float() / 255.0
    #print(f"Frame tensor shape: {frame.shape}")
    frame_tensor = torch.nn.functional.interpolate(
            frame.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
            ).squeeze(0)
    
    # frame tensor at this point is (3, 224, 224)
    
    if apply_augmentation:
        # Horizontal flip with 50% probability
        if random.random() < 0.5:
            frame_tensor = frame_tensor.flip(dims=[2])
        
        # Color jitter
        if random.random() < 0.8:
            # Brightness adjustment
            brightness_factor = random.uniform(0.6, 1.4)
            frame_tensor = frame_tensor * brightness_factor
            
            # Contrast adjustment
            if random.random() < 0.5:
                contrast_factor = random.uniform(0.6, 1.4)
                mean = torch.mean(frame_tensor, dim=[1, 2], keepdim=True)
                frame_tensor = (frame_tensor - mean) * contrast_factor + mean
            
            # Saturation adjustment (assuming RGB input)
            if random.random() < 0.5:
                saturation_factor = random.uniform(0.6, 1.4)
                gray = frame_tensor.mean(dim=0, keepdim=True)
                gray = gray.expand_as(frame_tensor)
                frame_tensor = frame_tensor * saturation_factor + gray * (1 - saturation_factor)

    else:
        frame_tensor = frame_tensor
        
    # Ensure values stay in [0, 1] range
    frame_tensor = torch.clamp(frame_tensor, 0, 1)
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
        error_occurred = False
        
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
        

# Add this class to train.py before the MultiModalTrainer class
class FlatAudioVisualDataset(Dataset):
    """
    Version of AudioVisualDataset that works with a flat directory structure
    (no segment subdirectories)
    """
    def __init__(self, data_root: str, sample_fps: int = 20):
        self.data_root = Path(data_root)
        self.sample_fps = sample_fps
        # Directly gather all mp4 files in the root directory
        self.video_files = sorted(list(self.data_root.glob("*.mp4")))
        if not self.video_files:
            raise ValueError(f"No MP4 files found in {data_root}")
            
        print(f"Found {len(self.video_files)} videos in flat directory {data_root}")
        
        # No segments in this version
        self.current_segment = 0
        
    def switch_segment(self):
        # No-op for flat dataset
        pass
    
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
        'video_paths': [str(item['video_path']) for item in batch]
    }


if __name__ == "__main__":
    '''print("Testing LocalCaptionDataset...")
    dataset = LocalCaptionDataset("/home/cis/cc3m")
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    print("\nTesting batch loading...")
    for batch_idx, (images, captions) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Image batch shape: {images.shape}")  # Should be [4, 3, 224, 224]
        print(f"Sample caption: {captions[0]}")
        break'''
    import tqdm
    print("Testing AudioVisualDataset with segmented structure...")
    
    dataset = AudioVisualDataset(
        data_root="/home/cis/GodSet",
        sample_fps=20
    )
    dataloader = DataLoader(
        dataset,
        batch_size=18,
        shuffle=True,
        num_workers=12,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=3
    )
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        #print(f"\nBatch {batch_idx + 1}")
        #print(f"Current segment: {dataset.current_segment}")
        #print(f"Frame shape: {batch['frame'].shape}")
        #print(f"Audio shape: {batch['audio'].shape}")
        #print(f"Sample video path: {batch['video_paths'][0]}")
        #break
        #if batch_idx % 3 == 0:
            #dataset.switch_segment()
        pass
    
    print("Dataset processing complete. Errors have been logged to segmented_error_videos.txt")

    print("Testing FlatAudioVisualDataset...")
    dataset = FlatAudioVisualDataset(
        data_root="/home/cis/UnGodSet",
        sample_fps=20
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=12, collate_fn=collate_fn)
    
    print("Processing dataset...")
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader)):
        pass
        #print(f"\nBatch {batch_idx + 1}")
        #print(f"Frame shape: {batch['frame'].shape}")
        #print(f"Audio shape: {batch['audio'].shape}")
        #print(f"Sample video path: {batch['video_paths'][0]}")
    
    print("Dataset processing complete. Errors have been logged to error_videos.txt")
    # No need to write from the list anymore as errors are logged directly during processing
        
            
        