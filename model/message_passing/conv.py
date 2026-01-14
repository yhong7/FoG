import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import math



class GATAggregator(nn.Module):
    def __init__(self, dim, alpha=0.2):
        super().__init__()
        self.attn = nn.Linear(dim, 1, bias=False)
        self.lin = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Linear(dim, dim)
        )
        self.leakyrelu = nn.LeakyReLU(alpha, inplace=False)

    def forward(
        self,
        messages: torch.Tensor,       # (B, Dim)
        src_node_idx: torch.Tensor,  # (B,)
        tgt_node_idx: torch.Tensor,  # (B,)
        num_entities: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        device = messages.device
        B, D = messages.shape

        src_node_idx = src_node_idx.to(device)
        tgt_node_idx = tgt_node_idx.to(device)

        # 注意力打分 (B,)
        message_passing_feature = self.lin(messages)
        e = self.leakyrelu(self.attn(message_passing_feature)).squeeze(-1)

        # softmax 按 target 节点分组
        exp_e = torch.exp(e - e.max())  # 数值稳定
        denom = torch.zeros(num_entities, device=device).index_add_(0, tgt_node_idx, exp_e)
        alpha = exp_e / (denom[tgt_node_idx] + 1e-9)  # (B,)

        # 加权聚合
        out = torch.zeros(num_entities, D, device=device, dtype=messages.dtype)
        out.index_add_(0, tgt_node_idx, messages * alpha.unsqueeze(-1))  # (num_entities, D)

        return out, alpha



class GraphMessagePassing(nn.Module):
    """
    主模块：把消息通过 MLP 编码后，按实体索引做聚合。
    """
    def __init__(
        self,
        query_dim: int = 1024,
        dim: int = 256,
        num_layers: int = 4,   # self-att 层数
        num_heads: int = 8,             # multi-head 数量, 需整除 dim
    ):
        super().__init__()
        # self.msg_encoder = MessageMLP(in_dim, hidden_dim, out_dim)
        self.query_dim_align = nn.Linear(query_dim, dim)
        self.lin = nn.Linear(dim, dim)  # 用于把 qtr_emb 映射到和 pre_flow_emb 同一空间

        # 多层 self-att，对三元组之间做交互
        self.num_self_att_layers = num_layers
        self.self_att_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # 用于 query 与三元组做 cross attention 风格的打分
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)

    def forward(
        self,
        query_emb: torch.Tensor,
        qtr_emb: torch.Tensor,
        pre_flow_emb: torch.Tensor,
        source_node_idx: torch.Tensor,
        target_node_idx: torch.Tensor,
        num_entities: int,
    ) -> torch.Tensor:
        """
        参数
        - x_embed: (num_entities, 1024)
        - qtr_emb:   (B, 1024)  批量消息
        - begin_node_idx: (B,), 上一跳聚和完的信息，存放在节点的位置
        - target_node_idx: (B,)       每条消息的目标实体 id，范围 [0, num_entities-1]
        - query_emb: (1024,)         query 的向量表征

        返回
        - entity_repr: (num_entities, 1024)  聚合后的实体表示
        - logits:      (B,)                  每条三元组与 query 的相关性 logit
        """
        query_emb = self.query_dim_align(query_emb)
        device = qtr_emb.device
        B, D = qtr_emb.shape

        # ===== 1) 原始三元组表征 =====
        # msgs = self.msg_encoder(qtr_emb) + x_embed[source_node_idx]  # (B, 1024)
        # 这里把 pre_flow_emb 看成“上一跳/结构信息”，lin(qtr_emb) 看成“当前 query-三元组内容”表征
        triple_raw = pre_flow_emb + self.lin(qtr_emb)  # (B, D)

        # ===== 2) Self-Attention：三元组之间做交互 =====
        # 把所有三元组当成一个序列：batch_size = 1, seq_len = B
        x = triple_raw.unsqueeze(0)  # (1, B, D)
        for attn in self.self_att_layers:
            x, _ = attn(x, x, x)     # self-att
        triple_ctx = x.squeeze(0)    # (B, D)，self-att 后的三元组表征

        # ===== 3) Cross-Attention 风格打分：query 当作 Q，三元组当作 K =====
        # query_emb 期望是 (D,)
        if query_emb.dim() == 2 and query_emb.size(0) == 1:
            query_emb = query_emb.squeeze(0)  # (D,)

        assert query_emb.dim() == 1 and query_emb.size(0) == D, \
            f"query_emb 期望形状为 ({D},) 或 (1, {D})，实际为 {tuple(query_emb.shape)}"

        # 线性映射到 Q / K 空间
        q = self.q_proj(query_emb)          # (D,)
        k = self.k_proj(triple_ctx)         # (B, D)

        # scaled dot-product，得到每条三元组一个 logit
        # logits[i] = <q, k_i> / sqrt(D)
        logits = (k * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(D)  # (B,)

        # ===== 4) 聚合到实体上 =====
        # 使用 self-att 后的三元组表示 triple_ctx 聚合到目标实体
        target_node_idx = target_node_idx.to(device)
        entity_repr = torch.zeros(num_entities, D, device=device, dtype=triple_ctx.dtype)
        entity_repr.index_add_(0, target_node_idx, triple_ctx)  # (num_entities, D)

        return entity_repr

    def save(self, save_path: str, overwrite: bool = False, extra_info: dict = None):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists")
        state = {"model_state_dict": self.state_dict(),}
        if extra_info:
            state.update(extra_info)
        torch.save(state, save_path)


