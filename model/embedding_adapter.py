import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from config.model_args import DEVICE

class EmbeddingAdapter(nn.Module):
    """
    轻量adapter, query和triplet同时使用
    """

    def __init__(self, input_dim: int = 1024, output_dim: int = 1024, bottleneck: int = 64, dropout: float = 0.1,
                 init_scale: float = 1.0, device: str or torch.device = DEVICE):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, output_dim, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.device = device

        self.to(device)

        # 初始化参数
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up(self.act(self.down(self.ln(x))))
        return self.dropout(h)
