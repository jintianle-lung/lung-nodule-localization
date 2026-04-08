import torch
import torch.nn as nn


class ImplicitTemporalChangeEncoder(nn.Module):
    """Learn a temporal-change surrogate from feature sequences without explicit delta input."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden_dim = max(int(out_dim), int(in_dim) // 2, 16)
        self.net = nn.Sequential(
            nn.Conv1d(int(in_dim), int(hidden_dim), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Conv1d(int(hidden_dim), int(out_dim), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(int(out_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        x = seq.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)
