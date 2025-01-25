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

# use fork for potentially faster dataloader start
try:
    multiprocessing.set_start_method('fork', force=True)
except:
    multiprocessing.set_start_method('spawn', force=True)
import gc
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

class LivisDataset(Dataset):
    def __init__(self, hf_dataset, split='train', transform=None):
        """
        Args:
            hf_dataset: HuggingFace dataset
            split: Split to use (only 'train' for now)
            transform: Optional transform to be applied on images
        """
        self.data = hf_dataset[split]
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
        self.img_dir = Path("/home/cis/heyo/DenseRead/images")
        
    def _get_local_path(self, url):
        """Convert URL to local file path"""
        # Extract ID and split from URL
        path = urlparse(url).path
        img_id = os.path.splitext(os.path.basename(path))[0]
        split = "val" if "val2017" in path else "train" if "train2017" in path else "unknown"
        
        return self.img_dir / f"{split}_{img_id}.jpg"
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = self._get_local_path(item['url'])
        
        # Check if image exists locally
        if not img_path.exists():
            raise FileNotFoundError(f"Image {img_path} not found! Make sure to download images first.")
            
        # Load and transform image
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a dummy tensor and empty captions if image loading fails
            return torch.zeros((3, 224, 224)), "", ""
            
        return image, item['caption'], item['short_caption']


def extract_audio_from_video(video_path: Path) -> torch.Tensor:
    """Extract entire 1s audio from video."""
    try:
        container = av.open(str(video_path))
        audio = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)
        
        samples = []
        for frame in container.decode(audio):
            frame.pts = None
            frame = resampler.resample(frame)[0]
            samples.append(frame.to_ndarray().reshape(-1))
        container.close()

        samples = torch.tensor(np.concatenate(samples))
        samples = samples.float() / 32768.0  
        return samples
    except:
        print(f"Failed to load audio from {video_path}")
        return torch.zeros(16331)
    finally:
        if container:
            container.close()
        #gc.collect()
        #torch.cuda.empty_cache()

def load_and_preprocess_video(video_path: str, sample_fps: int) -> torch.Tensor:
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
    frame_tensor = torch.nn.functional.interpolate(
        frame_tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
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
        self.video_files = sorted(list(self.data_root.glob("*.mp4")))
        
        self.vid_to_files = {}
        for file in self.video_files:
            vid_num = int(file.stem.split('_')[0])
            if vid_num not in self.vid_to_files:
                self.vid_to_files[vid_num] = []
            self.vid_to_files[vid_num].append(file)
            
        self.vid_nums = [int(f.stem.split('_')[0]) for f in self.video_files]
        print("Max Video Number: ", max(self.vid_nums))

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
    
        video_path = self.video_files[idx]
        try:
            audio = extract_audio_from_video(video_path)
            video_frame = load_and_preprocess_video(str(video_path), self.sample_fps)
            return {
                'video_path': str(video_path),
                'video_frames': video_frame, 
                'audio': audio,
                'vid_num': int(video_path.stem.split('_')[0]),
                'segment_num': int(video_path.stem.split('_')[1]),
            }
        except Exception as e:
            print(f"Error processing {self.video_files[idx]}: {str(e)}")
            return {
                'video_path': str(self.video_files[idx]),
                'video_frames': torch.zeros(3, 224, 224),
                'audio': torch.zeros(16331),
                'vid_num': -1,
                'segment_num': -1
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
        'vid_nums': [item['vid_num'] for item in batch],
        'segment_nums': [item['segment_num'] for item in batch],
        'video_paths': [str(item['video_path']) for item in batch]
    }


if __name__ == "__main__":
    print("Testing LivisDataset...")
    
    # Load dataset
    dset = datasets.load_dataset("/home/cis/heyo/DenseRead/livis")
    
    # Create dataset instance
    dataset = LivisDataset(dset)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
    
    # Test a batch
    print("\nTesting batch loading...")
    for batch_idx, (images, captions, short_captions) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Image batch shape: {images.shape}") # [batch_size, 3, 224, 224]
        print(f"Sample caption: {captions[0][:100]}...") 
        print(f"Sample short caption: {short_captions[0][:100]}...")
        
        # Test just one batch
        break

    print("Testing AudioVisualDataset...")

    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = AudioVisualDataset(
        data_root="/home/cis/VGGSound_Splits",
        sample_fps=20
    )

    batch_sampler = VideoBatchSampler(
        vid_nums=dataset.vid_nums,
        batch_size=2
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2
    )

    for batch in dataloader:
        print(batch['frame'].shape) # [batch_size, 3, 224, 224]
        print(batch['audio'].shape) # [batch_size, 16331]
        print(batch['vid_nums']) # [batch_size]
        print(batch['segment_nums']) # [batch_size]
        print(batch['video_paths']) # [batch_size]
        break
        
