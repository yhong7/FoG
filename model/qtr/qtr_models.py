import math
import os
import time

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from model.embedding_adapter import EmbeddingAdapter
from config.model_args import DEVICE


class FiLM(nn.Module):
    def __init__(self, d_in, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, hidden_size * 2), nn.GELU(), nn.Linear(hidden_size * 2, hidden_size * 2)
        )

    def forward(self, q, T):
        gamma_beta = self.mlp(q)  #
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(1) * T + beta.unsqueeze(1)


class AttnBlock(nn.Module):
    def __init__(self, hidden_size, n_heads=4, dropout=0.1, cross=False):
        super().__init__()
        self.cross = cross
        self.mha = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, kv=None):
        if self.cross:
            q = self.ln1(x)
            k = v = self.ln1(kv)
            y, _ = self.mha(q, k, v, need_weights=False)
            x = x + self.drop(y)
        else:
            y, _ = self.mha(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
            x = x + self.drop(y)

        z = self.ffn(self.ln2(x))
        x = x + self.drop(z)
        return x



class QTRFeatureMapBody(nn.Module):
    """
    主模型，两阶段训练：
    1. 预训练(带头)
    2. strip_classifier() 后当特征提取器 + 外接头/对比学习
    """

    def __init__(self,
                 embed_dim: int = 1024,
                 hidden_size: int = 128,
                 output_dim: int = 1024,
                 n_heads: int = 4,
                 order_sensitive: bool = False,
                 dropout: float = 0.1,
                 adapter_bottleneck: int = 64,
                 train_mode: Literal["adapter_only", "normal", "freeze"] = 'normal',
                 device: str | torch.device = DEVICE,
                 ):
        """
        :param embed_dim: 输入维度
        :param hidden_size: 模型 dim
        :param n_heads: multihead num
        :param order_sensitive: 三元组是否有序, 为 True 时加入可学习位置编码
        :param dropout: 0-1
        :param adapter_bottleneck: Adapter 的瓶颈维度
        :param train_mode: 初始化时的冻结策略（str）
        """
        super().__init__()
        self.order_sensitive = order_sensitive
        self.device = device

        # 共享Adapter（对 q 与 T 的最后一维）
        self.adapter = EmbeddingAdapter(
            input_dim=embed_dim,
            bottleneck=adapter_bottleneck,
            dropout=dropout,
            output_dim=hidden_size,
            init_scale=1.0,
            device=self.device
        )

        # 投影到骨干空间
        self.q_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.entity_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.relation_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))

        if order_sensitive:
            self.pos = nn.Parameter(torch.randn(1, 3, hidden_size, device=self.device) * 0.02)

        self.film = FiLM(hidden_size, hidden_size)
        self.cross = AttnBlock(hidden_size, n_heads, dropout, cross=True)
        self.sab = AttnBlock(hidden_size, n_heads, dropout, cross=False)

        self.pool_q = nn.Linear(hidden_size, hidden_size)
        self.pool_t = nn.Linear(hidden_size, hidden_size)
        self.pool_v = nn.Linear(hidden_size, 1)

        # self.ntn = NTN(hidden_size, k=ntn_k, low_rank=ntn_rank)
        self.lin_out = nn.Sequential(
            nn.LayerNorm(4 * hidden_size),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout)
        )

        self.to(self.device)

        # 初始化训练策略
        if train_mode == 'adapter_only':
            self.freeze_backbone()
        elif train_mode == 'normal':
            self.unfreeze()
        elif train_mode == 'freeze':
            self.freeze()
        else:
            raise ValueError("train_mode must be 'adapter_only' , 'normal' or 'freeze'")

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, q, T, use_adapter:bool=True):
        """
        :param q: [B, embed_dim],
        :param T: [B, 3, embed_dim]
        """
        q = q.to(self.device)
        T = T.to(self.device)
        B = q.size(0)
        if use_adapter:
            q = self.adapter(q)  # [B,embed_dim]
            T = self.adapter(T)  # [B,3,embed_dim]

        qh = self.q_proj(q)  # [B,hidden_size]
        Th = torch.cat(
            [self.entity_proj(T[:, 0:1, :]), self.relation_proj(T[:, 1:2, :]), self.entity_proj(T[:, 2:3, :])],
            dim=1)  # [B,3,hidden_size]
        if self.order_sensitive:
            Th = Th + self.pos.expand(B, -1, -1)

        Tc = self.film(qh, Th)
        q_tok = qh.unsqueeze(1)
        q_ctx = self.cross(q_tok, kv=Tc).squeeze(1)  # [B,hidden_size]
        H = self.sab(Tc)  # [B,3,hidden_size]

        scores = self.pool_v(
            torch.tanh(self.pool_q(q_ctx).unsqueeze(1) + self.pool_t(H))
        ).squeeze(-1)
        attn = scores.softmax(dim=1)
        s = (attn.unsqueeze(-1) * H).sum(dim=1)

        inter = torch.cat([s, q_ctx, s * q_ctx, (s - q_ctx).abs()], dim=-1)  # [B,4*hidden_size]
        feat = self.lin_out(inter)

        # return feat
        return {
            "features": feat,  # [B, output_dim]
            "pooled": s,  # [B, hidden]，triplet
            "q_ctx": q_ctx,  # [B, hidden]，query
            "attn": attn  # [B, 3]
        }

    def save(self, save_path: str, overwrite: bool = False, extra_info=None):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists")
        state = {"model_state_dict": self.state_dict(),}
        if extra_info:
            state.update(extra_info)
        torch.save(state, save_path)



class QTRFeatureMapBodyAtt(nn.Module):
    """
    主模型，两阶段训练：
    1. 预训练(带头)
    2. strip_classifier() 后当特征提取器 + 外接头/对比学习
    """
    def __init__(self,
                 embed_dim: int = 1024,
                 hidden_size: int = 128,
                 output_dim: int = 1024,
                 n_heads: int = 4,
                 num_layers: int = 2,
                 order_sensitive: bool = False,
                 dropout: float = 0.1,
                 adapter_bottleneck: int = 64,
                 train_mode: Literal["adapter_only", "normal", "freeze"] = 'normal',
                 device: str | torch.device = "cuda",
                 ):
        """
        :param embed_dim: 输入维度
        :param hidden_size: 模型 dim
        :param n_heads: multihead num
        :param order_sensitive: 三元组是否有序, 为 True 时加入可学习位置编码
        :param dropout: 0-1
        :param adapter_bottleneck: Adapter 的瓶颈维度
        :param train_mode: 初始化时的冻结策略（str）
        """
        super().__init__()
        self.order_sensitive = order_sensitive
        self.device = device
        assert hidden_size % n_heads == 0, "hidden_size 必须能被 n_heads 整除"
        self.num_layers = num_layers

        # 共享Adapter（对 q 与 T 的最后一维）
        self.adapter = EmbeddingAdapter(
            input_dim=embed_dim,
            bottleneck=adapter_bottleneck,
            dropout=dropout,
            output_dim=hidden_size,
            init_scale=1.0,
            device=self.device
        )

        # 投影到骨干空间
        self.q_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.entity_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))
        self.relation_proj = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.LayerNorm(hidden_size))

        if order_sensitive:
            self.pos = nn.Parameter(torch.randn(1, 3, hidden_size, device=self.device) * 0.02)

        # ===== 纯官方库：Self-Att（Encoder） =====
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4*hidden_size,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # ===== 纯官方库：Cross-Att（取权重） =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.cross_ln_q = nn.LayerNorm(hidden_size)
        self.cross_ln_kv = nn.LayerNorm(hidden_size)
        self.cross_ffn = nn.Sequential(
            nn.Linear(hidden_size, 4*hidden_size), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4*hidden_size, hidden_size)
        )
        self.cross_dropout = nn.Dropout(dropout)
        self.cross_ln_out = nn.LayerNorm(hidden_size)

        # 输出头（保持原有 four-way 交互拼接）
        self.lin_out = nn.Sequential(
            nn.LayerNorm(4 * hidden_size),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
            nn.Dropout(dropout)
        )

        self.to(self.device)

        # 初始化训练策略
        if train_mode == 'adapter_only':
            self.freeze_backbone()
        elif train_mode == 'normal':
            self.unfreeze()
        elif train_mode == 'freeze':
            self.freeze()
        else:
            raise ValueError("train_mode must be 'adapter_only' , 'normal' or 'freeze'")

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.adapter.parameters():
            p.requires_grad = True

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, q, T, use_adapter:bool=True):
        """
        :param q: [B, embed_dim],
        :param T: [B, 3, embed_dim]
        """
        q = q.to(self.device)
        T = T.to(self.device)
        B = q.size(0)

        # 1) 适配到 hidden_size
        if use_adapter:
            q = self.adapter(q)      # [B, hidden]
            T = self.adapter(T)      # [B, 3, hidden]

        # 2) 基础投影
        qh = self.q_proj(q)  # [B, hidden]
        Th = torch.cat(
            [self.entity_proj(T[:, 0:1, :]), self.relation_proj(T[:, 1:2, :]), self.entity_proj(T[:, 2:3, :])],
            dim=1
        )  # [B, 3, hidden]
        if self.order_sensitive:
            Th = Th + self.pos.expand(B, -1, -1)

        # 3) Self-Att over triples（官方 Encoder）
        H = self.encoder(Th)  # [B, 3, hidden]

        # 4) Cross-Att：Q=qh，KV=H（拿到注意力）
        q_tok = qh.unsqueeze(1)  # [B,1,H]
        qn = self.cross_ln_q(q_tok)
        kvn = self.cross_ln_kv(H)
        q_ctx, attn_w = self.cross_attn(
            qn, kvn, kvn, need_weights=True, average_attn_weights=True
        )  # q_ctx:[B,1,H], attn_w:[B,1,3]
        # 残差 + FFN（官方形态）
        q_ctx = q_tok + self.cross_dropout(q_ctx)
        q_ctx = q_ctx + self.cross_dropout(self.cross_ffn(self.cross_ln_out(q_ctx)))
        q_ctx = q_ctx.squeeze(1)      # [B,H]

        # 5) 用 cross-att 权重做三元组加权池化
        attn = attn_w.squeeze(1)      # [B,3]
        s = torch.bmm(attn.unsqueeze(1), H).squeeze(1)   # [B,H]

        # 6) 交互拼接 + 输出
        inter = torch.cat([s, q_ctx, s * q_ctx, (s - q_ctx).abs()], dim=-1)  # [B,4H]
        feat = self.lin_out(inter)  # [B, output_dim]

        return {
            "features": feat,  # [B, output_dim]
            "pooled": s,       # [B, hidden]
            "q_ctx": q_ctx,    # [B, hidden]
            "attn": attn       # [B, 3]
        }

    def save(self, save_path: str, overwrite: bool = False, extra_info=None):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists")
        state = {"model_state_dict": self.state_dict(),}
        if extra_info:
            state.update(extra_info)
        torch.save(state, save_path)

class QTRClassifierHead(nn.Module):
    """
    分类头：1024 -> 1 (输出 logit)，并提供 sigmoid 概率
    """

    def __init__(self, in_dim: int = 1024, device: str | torch.device = DEVICE):
        super().__init__()
        self.device = device
        # 结构保持与你原 classifier 类似，只是固定输入 1024
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1),
        )
        # 模型整体迁移到 device
        self.to(self.device)

    def forward(self, x):
        """
        x: [B, 1024] -> return (prob, logit)
        """
        x = x.to(self.device)
        logit = self.net(x).squeeze(-1)
        prob = torch.sigmoid(logit)
        return prob, logit

    def save(self, save_path: str, overwrite: bool = False, extra_info: dict=None):
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not overwrite and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists")
        state = {"model_state_dict": self.state_dict(),}
        if extra_info:
            state.update(extra_info)
        torch.save(state, save_path)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True


class QTRPipeline(nn.Module):
    def __init__(self, body: QTRFeatureMapBody, head: QTRClassifierHead, device: str or torch.device = DEVICE):
        super().__init__()
        self.body = body.to(device)
        self.head = head.to(device)
        self.device = device

    def forward(self, q, T):
        q = q.to(self.device)
        T = T.to(self.device)
        body_out = self.body(q, T)  # dict
        feat = body_out["features"]
        prob, logit = self.head(feat)
        return prob, logit

    def save(self, save_dir: str, overwrite: bool = False, extra_info: dict = None):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.body.save(os.path.join(save_dir, "qtr_body_pretrain.pt"), overwrite=overwrite, extra_info=extra_info)
        self.head.save(os.path.join(save_dir, "qtr_head_pretrain.pt"), overwrite=overwrite, extra_info=extra_info)



if __name__ == '__main__':
    # 1. pretrain
    model = QTRPipeline(
        embed_dim=1024, hidden_size=128, output_dim=1024,
        n_heads=4,
        order_sensitive=True,
        adapter_bottleneck=64,
        train_mode='normal'  # 预训练时全参训练
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    n = 10
    q_embed = torch.randn(n, 1024)
    triplet_embed = torch.randn(n, 3, 1024)
    labels = torch.tensor([0 for _ in range(n // 2)] + [1 for _ in range(n // 2)])

    for _ in range(100):
        prob, logit = model.forward(q_embed, triplet_embed)  # q:[B,1024] T:[B,3,1024]
        loss = F.binary_cross_entropy_with_logits(logit, labels.float())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)

    torch.save(model.body.state_dict(), "checkpoints/qtr_body_pretrain.pt")
    torch.save(model.head.state_dict(), "checkpoints/qtr_head_pretrain.pt")
