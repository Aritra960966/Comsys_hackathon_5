import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import glob
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# CONFIG
CONFIG = {
    'EMBEDDING_DIM': 256,
    'DROPOUT_RATE': 0.3,
    'BATCH_SIZE': 32,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'THRESHOLD': 0.5
}

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class FacePairDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.root = root
        self.entities = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.data = []
        for entity in self.entities:
            entity_path = os.path.join(root, entity)
            distortion_path = os.path.join(entity_path, "distortion")
            if not os.path.exists(distortion_path):
                continue
            originals = glob.glob(os.path.join(entity_path, "*.jpg"))
            distorted = glob.glob(os.path.join(distortion_path, "*.jpg"))
            for orig in originals:
                same_dist = [d for d in distorted if os.path.basename(d).startswith(os.path.splitext(os.path.basename(orig))[0])]
                for dimg in same_dist:
                    self.data.append((orig, dimg, 1))
                for _ in range(2):
                    neg_entity = random.choice([e for e in self.entities if e != entity])
                    neg_dist_path = os.path.join(root, neg_entity, "distortion")
                    if os.path.exists(neg_dist_path):
                        neg_dist = glob.glob(os.path.join(neg_dist_path, "*.jpg"))
                        if neg_dist:
                            self.data.append((orig, random.choice(neg_dist), 0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_path, comp_path, label = self.data[idx]
        anchor = Image.open(anchor_path).convert("RGB")
        comp = Image.open(comp_path).convert("RGB")
        if self.transform:
            anchor = self.transform(anchor)
            comp = self.transform(comp)
        return anchor, comp, torch.tensor(label, dtype=torch.float32)

class FaceEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, dropout_rate=0.3):
        super().__init__()

        backbone = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone_out_channels = 512

        self.feature_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.distortion_sim = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout_rate)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight is not None:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.features(x)
        feat_orig = self.feature_conv(feat)
        feat_aug = self.distortion_sim(feat_orig)
        fusion_input = torch.cat([feat_orig.unsqueeze(1), feat_aug.unsqueeze(1)], dim=1)
        fusion_input = fusion_input.view(x.size(0), 512, feat_orig.size(2), feat_orig.size(3))
        att = self.attention_mlp(fusion_input)
        alpha = att[:, 0:1]
        beta = att[:, 1:2]
        fused_feat = alpha * feat_orig + beta * feat_aug
        pooled = self.global_pool(fused_feat).flatten(1)
        embedding = self.embedding(pooled)
        return F.normalize(embedding, p=2, dim=1)

class AlternativeFaceEmbedder(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        super().__init__()
        backbone = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.embedding(x)
        return F.normalize(x, p=2, dim=1)


def detect_backbone(model_path):
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any('spatial_attention' in k or 'channel_attention' in k for k in state_dict):
        return 'alternative'
    if any('conv3' in k for k in state_dict):
        return 'resnet50'
    return 'resnet18'

def safe_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    compatible_dict = {
        k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape
    }
    if not compatible_dict:
        raise RuntimeError("No compatible weights found in the state_dict.")
    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict)

def load_model(model_path, device):
    backbone = detect_backbone(model_path)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    try:
        if backbone == 'alternative':
            model = AlternativeFaceEmbedder(CONFIG['EMBEDDING_DIM'], CONFIG['DROPOUT_RATE']).to(device)
        else:
            model = FaceEmbedder(CONFIG['EMBEDDING_DIM'], CONFIG['DROPOUT_RATE']).to(device)
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print("Strict load failed:", str(e))
        print("Trying safe partial load with AlternativeFaceEmbedder...")
        model = AlternativeFaceEmbedder(CONFIG['EMBEDDING_DIM'], CONFIG['DROPOUT_RATE']).to(device)
        safe_load_state_dict(model, state_dict)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, threshold=0.5):
    all_labels, sim_preds = [], []
    with torch.no_grad():
        for img1, img2, labels in tqdm(dataloader):
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            emb1 = model(img1)
            emb2 = model(img2)
            sims = F.cosine_similarity(emb1, emb2)
            preds = (sims > threshold).float()
            all_labels.append(labels)
            sim_preds.append(preds)
    y_true = torch.cat(all_labels).cpu().numpy()
    y_pred = torch.cat(sim_preds).cpu().numpy()
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='/content/dataset/Comys_Hackathon5/Task_B/test')
    parser.add_argument('--model_dir', default=script_dir)
    args = parser.parse_args()

    dataset = FacePairDataset(args.test_dir, val_transform)
    dataloader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'])

    results = {}
    for model_name in ['taskB_best_similarity_model.pth', 'taskB_best_distance_model.pth']:
        model_path = os.path.join(args.model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue
        model = load_model(model_path, CONFIG['DEVICE'])
        metrics = evaluate_model(model, dataloader, CONFIG['DEVICE'], CONFIG['THRESHOLD'])
        results[model_name] = metrics

    df = pd.DataFrame(results).T
    df.to_csv('taskB_test_results.csv')
    print("Saved results to taskB_test_results.csv")

    best_model = max(results.items(), key=lambda x: x[1]['macro_f1'])
    print("Test metrices")
    for metric, value in best_model[1].items():
        print(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main()
