
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class Scorer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bias=True):
        super().__init__()
        self.lin1 = nn.Linear(input_size * 2, hidden_size, bias=bias)
        self.lin2 = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, qtr_emb, message_flow_emb):
        input_emb = torch.cat((qtr_emb, message_flow_emb), dim=-1)
        x = self.lin1(input_emb)
        return self.lin2(x)

    def save(self, save_path: str, overwrite: bool = False, extra_info: Optional[Dict] = None):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists")
        state = {"model_state_dict": self.state_dict()}
        if extra_info:
            state.update(extra_info)
        torch.save(state, save_path)


