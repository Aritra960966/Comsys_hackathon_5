import torch
import torch.nn as nn
import torchvision.models as models


class AMR_CD_Model(nn.Module):
    def __init__(self):
        super().__init__()

        # Gray branch - ResNet18
        self.resnet_gray = models.resnet18(pretrained=True)
        self.resnet_gray.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_gray.fc = nn.Identity()

        # RGB branch - EfficientNet-B0
        self.effnet_rgb = models.efficientnet_b0(pretrained=True)
        self.effnet_rgb.classifier = nn.Identity()

        # Feature projection layers
        self.gray_conv = nn.Conv2d(512, 256, kernel_size=1)
        self.rgb_conv = nn.Conv2d(1280, 256, kernel_size=1)

        # Auxiliary classification heads
        self.gray_head = nn.Linear(256, 2)
        self.rgb_head = nn.Linear(256, 2)

        # Attention mechanism for feature fusion
        self.attn_fc = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x_gray, x_rgb):
        B = x_gray.size(0)

        # Extract features from gray branch
        F_gray = self.resnet_gray(x_gray).view(B, 512, 1, 1)
        F_gray = self.gray_conv(F_gray).view(B, 256)

        # Extract features from RGB branch
        F_rgb = self.effnet_rgb(x_rgb).view(B, 1280, 1, 1)
        F_rgb = self.rgb_conv(F_rgb).view(B, 256)

        # Auxiliary predictions
        logits_gray = self.gray_head(F_gray)
        logits_rgb = self.rgb_head(F_rgb)

        # Feature fusion with attention
        concat = torch.cat([F_gray, F_rgb], dim=1)  # Shape: [B, 512]
        attn = self.attn_fc(concat)                 # Shape: [B, 256]
        F_fused = attn * F_gray + (1 - attn) * F_rgb

        # Final prediction
        out = self.classifier(F_fused)
        return out, logits_gray, logits_rgb


def create_model(device):
    """Create and initialize the model"""
    model = AMR_CD_Model().to(device)
    return model


def load_model(model_path, device):
    """Load a saved model"""
    model = create_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model