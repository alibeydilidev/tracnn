import torch.nn as nn

class CNNBackbone(nn.Module):
    def __init__(self, input_channels, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, output_dim, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)  # [B, C, H, W] → [B, output_dim, 1, 1]
        return x.view(x.size(0), -1)  # [B, output_dim]