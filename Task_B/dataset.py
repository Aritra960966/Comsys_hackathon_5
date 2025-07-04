"""
Dataset module for TaskB Face Recognition
"""

import os
import random
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class FacePairDataset(Dataset):
    def __init__(self, root, transform, is_training=True, balance_ratio=1.5):
        self.transform = transform
        self.root = root
        self.is_training = is_training
        self.balance_ratio = balance_ratio
        self.entities = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.data = []
        self.weights = []

        print(f"Found {len(self.entities)} entities")

        positive_count = 0
        negative_count = 0

        for entity in tqdm(self.entities, desc="Loading dataset"):
            entity_path = os.path.join(root, entity)
            distortion_path = os.path.join(entity_path, "distortion")

            if not os.path.exists(distortion_path):
                continue

            originals = glob.glob(os.path.join(entity_path, "*.jpg"))
            distorted = glob.glob(os.path.join(distortion_path, "*.jpg"))

            for orig in originals:
                same_dist = [d for d in distorted if
                           os.path.basename(d).startswith(os.path.splitext(os.path.basename(orig))[0])]
                for dimg in same_dist:
                    self.data.append((orig, dimg, 1))
                    positive_count += 1

                for _ in range(int(self.balance_ratio)):
                    neg_entity = random.choice([e for e in self.entities if e != entity])
                    neg_dist_path = os.path.join(root, neg_entity, "distortion")
                    if os.path.exists(neg_dist_path):
                        neg_dist = glob.glob(os.path.join(neg_dist_path, "*.jpg"))
                        if neg_dist:
                            self.data.append((orig, random.choice(neg_dist), 0))
                            negative_count += 1

        total_pos = positive_count
        total_neg = negative_count

        for _, _, label in self.data:
            if label == 1:
                self.weights.append(1.0 / total_pos)
            else:
                self.weights.append(1.0 / total_neg)

        print(f"Dataset created: {positive_count} positive, {negative_count} negative pairs")
        print(f"Class balance ratio: {negative_count/positive_count:.2f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path, comp_path, label = self.data[idx]

        try:
            anchor = Image.open(anchor_path).convert("RGB")
            comp = Image.open(comp_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images: {e}")
            anchor = Image.new('RGB', (224, 224), color='black')
            comp = Image.new('RGB', (224, 224), color='black')
            label = 0

        if self.transform:
            anchor = self.transform(anchor)
            comp = self.transform(comp)

        return anchor, comp, torch.tensor(label, dtype=torch.float32)