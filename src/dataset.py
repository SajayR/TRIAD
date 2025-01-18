import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
import datasets
from urllib.parse import urlparse
import os

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
        print(f"Image batch shape: {images.shape}")
        print(f"Sample caption: {captions[0][:100]}...")
        print(f"Sample short caption: {short_captions[0][:100]}...")
        
        # Test just one batch
        break