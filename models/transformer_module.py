import torch
import torch.nn as nn


class TransformerModule(nn.Module):
    class TransformerEncoderLayerWithAttention(nn.Module):
        def __init__(self, input_dim, num_heads=2):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
            self.dropout1 = nn.Dropout(0.1)
            self.dropout2 = nn.Dropout(0.1)
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, input_dim * 4),
                nn.ReLU(),
                nn.Linear(input_dim * 4, input_dim),
            )
            self.attn_weights = None

        def forward(self, src):
            attn_output, attn_weights = self.self_attn(src, src, src, need_weights=True, average_attn_weights=False)
            self.attn_weights = attn_weights.detach()
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

            ff_output = self.mlp(src)
            src = src + self.dropout2(ff_output)
            src = self.norm2(src)

            return src

    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerModule.TransformerEncoderLayerWithAttention(input_dim, num_heads=2)
            for _ in range(num_layers)
        ])
        self.positional_encoding = nn.Parameter(torch.zeros(1, 512, input_dim))  # max length 512
        nn.init.uniform_(self.positional_encoding, a=-0.1, b=0.1)

    def forward(self, x):  # [B, T, D]
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        for layer in self.layers:
            x = layer(x)
        return x

    def get_attention_rollout(self):
        attn_maps = [layer.attn_weights for layer in self.layers if layer.attn_weights is not None]
        if not attn_maps:
            return None

        # Head-mean first: [B, H, T, T] → [B, T, T]
        attn_maps = [attn.mean(dim=1) for attn in attn_maps]

        rollout = attn_maps[0]
        for attn in attn_maps[1:]:
            rollout = torch.bmm(rollout, attn)

        # Average over batch
        rollout_mean = rollout.mean(dim=0)  # [T, T]
        return rollout_mean