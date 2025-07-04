import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image


class DualViewDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform_rgb=None, transform_gray=None):
        self.dataset = datasets.ImageFolder(root=root_dir)
        self.transform_rgb = transform_rgb
        self.transform_gray = transform_gray

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        image_rgb = Image.open(path).convert('RGB')
        image_gray = Image.open(path).convert('L')

        if self.transform_rgb:
            image_rgb = self.transform_rgb(image_rgb)
        if self.transform_gray:
            image_gray = self.transform_gray(image_gray)

        return image_gray, image_rgb, label


def get_transforms():
    """Get transforms for training and validation"""
    transform_rgb_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.RandomAffine(15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    transform_gray_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(15, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    transform_rgb_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    transform_gray_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    return transform_rgb_train, transform_gray_train, transform_rgb_val, transform_gray_val


def create_dataloaders(train_dir, val_dir, batch_size, num_workers=2):
    """Create train and validation dataloaders"""
    transform_rgb_train, transform_gray_train, transform_rgb_val, transform_gray_val = get_transforms()
    
    train_dataset = DualViewDataset(train_dir, transform_rgb_train, transform_gray_train)
    val_dataset = DualViewDataset(val_dir, transform_rgb_val, transform_gray_val)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return train_loader, val_loader


def create_test_dataloader(test_dir, batch_size, num_workers=2):
    """Create test dataloader"""
    _, _, transform_rgb_val, transform_gray_val = get_transforms()
    
    test_dataset = DualViewDataset(test_dir, transform_rgb_val, transform_gray_val)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )

    return test_loader