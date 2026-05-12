import torch
import torchvision.transforms as transforms
from torchvision import datasets
from pathlib import Path

def normalise(train_path: Path):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    ds = datasets.ImageFolder(
        root=train_path,
        transform=transforms.ToTensor()  # Convert to tensor for calculation
    )
    count = 0
    for idx, (image, label) in enumerate(ds):
        # image is already a tensor from ToTensor() transform
        mean += image.mean(dim=[1, 2])
        std += image.std(dim=[1, 2])
        count += 1
        
        # Progress indicator
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(ds)} images...")
    
    return tuple(mean.div(count).tolist()), tuple(std.div(count).tolist())

