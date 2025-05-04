import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x, target):
        log_probs = self.log_softmax(x)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class TraCNNDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(3, 64, 64), seq_len=20, feat_dim=64, num_classes=5):
        super().__init__()
        self.num_samples = num_samples
        self.img_size = img_size
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes

        self.labels = torch.randint(0, num_classes, (num_samples,))
        # Generate sequence features with structured patterns
        self.sequences = torch.zeros(num_samples, seq_len, feat_dim)
        for i in range(num_samples):
            label = self.labels[i].item()
            t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(1)
            t += torch.randn_like(t) * 0.1
            if label == 0:
                pattern = torch.sin(t)
            elif label == 1:
                pattern = torch.cos(t)
            elif label == 2:
                pattern = torch.linspace(0, 1, seq_len).unsqueeze(1)
            elif label == 3:
                pattern = torch.linspace(1, 0, seq_len).unsqueeze(1)
            else:
                pattern = torch.sign(torch.sin(3 * t))  # square wave style

            noise = torch.randn(seq_len, feat_dim) * 0.7
            self.sequences[i] = pattern @ torch.ones(1, feat_dim) + noise

        self.images = torch.zeros(num_samples, *img_size)
        c, h, w = img_size
        for i in range(num_samples):
            label = self.labels[i].item()
            img = torch.zeros(img_size)
            if label == 0:
                # horizontal bar in the center
                bar_height = h // 8
                start = (h - bar_height) // 2
                img[:, start:start+bar_height, :] = 1.0
            elif label == 1:
                # vertical bar in the center
                bar_width = w // 8
                start = (w - bar_width) // 2
                img[:, :, start:start+bar_width] = 1.0
            elif label == 2:
                # diagonal line from top-left to bottom-right
                for idx in range(min(h, w)):
                    img[:, idx, idx] = 1.0
            elif label == 3:
                # center square
                sq_size = min(h, w) // 4
                start_h = (h - sq_size) // 2
                start_w = (w - sq_size) // 2
                img[:, start_h:start_h+sq_size, start_w:start_w+sq_size] = 1.0
            else:
                # checkerboard pattern
                tile_size = 4
                for y in range(0, h, tile_size):
                    for x in range(0, w, tile_size):
                        if ((x // tile_size) + (y // tile_size)) % 2 == 0:
                            img[:, y:y+tile_size, x:x+tile_size] = 1.0
            img += torch.randn_like(img) * 0.3
            img = img.clamp(0, 1)
            self.images[i] = img

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'sequence': self.sequences[idx],
            'label': self.labels[idx]
        }