"""
ml_models/cnn/architecture.py
==============================
Custom CNN for Ambulance vs Non-Ambulance classification.
Architecture: 5 conv blocks + dropout + FC layers.

Input  : (B, 3, 224, 224)
Output : (B, 2)  — [non_ambulance, ambulance]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class AmbulanceCNN(nn.Module):
    """
    Lightweight CNN ambulance classifier.
    ~4M parameters, fast inference.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(3, 32),           # 224 → 112
            ConvBlock(32, 64),          # 112 → 56
            ConvBlock(64, 128),         # 56  → 28
            ConvBlock(128, 256),        # 28  → 14
            ConvBlock(256, 256, pool=False),  # 14 → 14
        )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # → (B, 256, 4, 4)
        self.flatten = nn.Flatten()               # → (B, 4096)

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return self.classifier(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    model = AmbulanceCNN()
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (4, 2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
