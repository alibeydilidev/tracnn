import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn
from models.cnn_backbone import CNNBackbone
from models.transformer_module import TransformerModule
from models.fusion_layer import FusionLayer


class TraCNN(nn.Module):
    def __init__(self, input_channels, trans_dim, fused_dim, num_classes):
        super().__init__()
        self.cnn = CNNBackbone(input_channels, output_dim=trans_dim)
        self.transformer = TransformerModule(input_dim=trans_dim)
        self.fusion = FusionLayer(cnn_dim=trans_dim, trans_dim=trans_dim, fused_dim=fused_dim)
        self.classifier = nn.Linear(fused_dim, num_classes)

    def forward(self, img_seq, feat_seq):  
        # img_seq: [B, C, H, W]
        # feat_seq: [B, T, D]
        cnn_out = self.cnn(img_seq)
        trans_out = self.transformer(feat_seq)
        fused = self.fusion(cnn_out, trans_out)
        return self.classifier(fused)