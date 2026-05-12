from clearml import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import data.normalize as normalize

def load_dataset(dataset_name, batch_size, path = None):
    if path is None:
        dataset = Dataset.get(
            dataset_project="Small Group Project - Team 2",
            dataset_name=dataset_name
        )
        path = dataset.get_local_copy()
        print("Dataset path:", path)
    
    train_path = path + "/train"
    mean, std = normalize.normalise(Path(train_path))
    
    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_augmented = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    train_dataset_mvp = datasets.ImageFolder(f"{path}/train", transform=tf_augmented)
    train_dataset_clean = datasets.ImageFolder(f"{path}/train", transform=tf)  # Clean version for evaluation
    val_dataset_mvp   = datasets.ImageFolder(f"{path}/val", transform=tf)
    test_dataset_mvp  = datasets.ImageFolder(f"{path}/test", transform=tf)
        
    train_dataloader_mvp = DataLoader(train_dataset_mvp, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    train_dataloader_clean = DataLoader(train_dataset_clean, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    val_dataloader_mvp = DataLoader(val_dataset_mvp, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    test_dataloader_mvp = DataLoader(test_dataset_mvp, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=2, persistent_workers=True)
    
    return train_dataloader_mvp, train_dataloader_clean, val_dataloader_mvp, test_dataloader_mvp, mean, std