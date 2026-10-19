"""Original MLPHead from scripts/train_tissue_phikon.py; inference only."""
import torch.nn as nn

class MLPHead(nn.Module):
    """Simple MLP classification head."""
    def __init__(self, feat_dim, num_classes, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
