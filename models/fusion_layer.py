import torch
import torch.nn as nn

class FusionLayer(nn.Module):
    def __init__(self, cnn_dim, trans_dim, fused_dim):
        super().__init__()
        self.linear_cnn = nn.Linear(cnn_dim, fused_dim)
        self.linear_trans = nn.Linear(trans_dim, fused_dim)
        self.fuser = nn.Sequential(
            nn.ReLU(),
            nn.Linear(fused_dim * 2, fused_dim)
        )

    def forward(self, cnn_feat, trans_feat):
        cnn_proj = self.linear_cnn(cnn_feat)
        trans_proj = self.linear_trans(trans_feat.mean(dim=1))  # average over time

        gate_input = torch.cat([cnn_proj, trans_proj], dim=-1)
        gate = torch.sigmoid(self.fuser[1](gate_input))

        weighted_cnn = cnn_proj * gate
        weighted_trans = trans_proj * (1 - gate)

        fusion_input = torch.cat([weighted_cnn, weighted_trans], dim=-1)
        fused_output = self.fuser(torch.relu(fusion_input))

        return fused_output