"""
Model module for TaskB Face Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class NoiseLayer(nn.Module):
    def __init__(self, std=0.02):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training:
            return x + torch.randn_like(x) * self.std
        return x


class FaceEmbedder(nn.Module):
    def __init__(self, embedding_dim=256, dropout_rate=0.3):
        super().__init__()

        # Base feature extractor
        backbone = models.resnet18(weights='DEFAULT')
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.backbone_out_channels = 512

        # Shallow transform for original and distorted
        self.feature_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        # Internal distortion simulation (1x1 conv)
        self.distortion_sim = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Learnable attention for multi-view fusion
        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 2, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )

        # Global pooling and embedding head
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
        feat = self.features(x)                # [B, 512, H, W]
        feat_orig = self.feature_conv(feat)    # [B, 256, H, W]

        # Simulate distortion
        feat_aug = self.distortion_sim(feat_orig)

        # Concatenate and compute attention weights
        fusion_input = torch.cat([feat_orig.unsqueeze(1), feat_aug.unsqueeze(1)], dim=1)  # [B, 2, 256, H, W]
        fusion_input = fusion_input.view(x.size(0), 512, feat_orig.size(2), feat_orig.size(3))  # [B, 512, H, W]

        att = self.attention_mlp(fusion_input)  # [B, 2, 1, 1]
        alpha = att[:, 0:1]                     # attention for orig
        beta = att[:, 1:2]                      # attention for aug

        fused_feat = alpha * feat_orig + beta * feat_aug

        pooled = self.global_pool(fused_feat).flatten(1)
        embedding = self.embedding(pooled)
        return F.normalize(embedding, p=2, dim=1)
    
class FaceEmbedder(nn.Module):
    """Updated model architecture to handle both ResNet18 and ResNet50 based checkpoints"""
    def __init__(self, embedding_dim=256, dropout_rate=0.3, backbone_type='resnet18'):
        super().__init__()
        
        self.backbone_type = backbone_type
        
        # Base feature extractor - flexible backbone
        if backbone_type == 'resnet18':
            backbone = models.resnet18(weights='DEFAULT')
            self.backbone_out_channels = 512
        elif backbone_type == 'resnet50':
            backbone = models.resnet50(weights='DEFAULT')
            self.backbone_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Use features (everything except final FC layer)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # Adaptive feature processing
        self.feature_conv = nn.Sequential(
            nn.Conv2d(self.backbone_out_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )
        
        # Internal distortion simulation
        self.distortion_sim = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256 * 2, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # Global pooling and embedding
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
        
        # Simulate distortion
        feat_aug = self.distortion_sim(feat_orig)
        
        # Attention-based fusion
        fusion_input = torch.cat([feat_orig, feat_aug], dim=1)
        att = self.attention_mlp(fusion_input)
        alpha = att[:, 0:1]
        beta = att[:, 1:2]
        
        fused_feat = alpha * feat_orig + beta * feat_aug
        
        pooled = self.global_pool(fused_feat).flatten(1)
        embedding = self.embedding(pooled)
        return F.normalize(embedding, p=2, dim=1)


class AlternativeFaceEmbedder(nn.Module):
    """Alternative architecture for models with different structure"""
    def __init__(self, embedding_dim=256, dropout_rate=0.3):
        super().__init__()
        
        # ResNet50 backbone
        backbone = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Attention mechanisms
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 2048),
            nn.Sigmoid()
        )
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
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
        feat = self.backbone(x)
        
        # Apply attention
        spatial_att = self.spatial_attention(feat)
        channel_att = self.channel_attention(feat).unsqueeze(-1).unsqueeze(-1)
        
        attended_feat = feat * spatial_att * channel_att
        
        embedding = self.embedding_head(attended_feat)
        return F.normalize(embedding, p=2, dim=1)




class TripletMarginLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = F.pairwise_distance(anchor, positive)
        distance_negative = F.pairwise_distance(anchor, negative)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()


class HybridLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.7):
        super().__init__()
        self.margin = margin
        self.alpha = alpha

    def forward(self, emb1, emb2, labels):
        distances = F.pairwise_distance(emb1, emb2)
        contrastive = torch.mean(
            labels * distances**2 +
            (1 - labels) * F.relu(self.margin - distances)**2
        )

        cos_sim = F.cosine_similarity(emb1, emb2)
        cosine_loss = torch.mean(
            labels * (1 - cos_sim) +
            (1 - labels) * F.relu(cos_sim + 0.1)
        )

        return self.alpha * contrastive + (1 - self.alpha) * cosine_loss