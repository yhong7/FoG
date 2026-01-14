import os
import json
import random
import asyncio
from typing import Tuple
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from config.singleton import KG_VOCAB

from model.qtr.qtr_dataloader_builder import build_mixed_dataloader
from model.qtr.qtr_models import QTRPipeline
from config.model_args import QTR_PRETRAIN_DATALOADER_BUILDING_ARGS, \
    QTR_PRETRAIN_OPTIMIZER_ARGS, SEED
from config.singleton import QTR_BODY_PRETRAIN, QTR_HEAD_PRETRAIN, DEFAULT_EMBEDDER

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
async def validate(model: torch.nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
    """
    对验证集计算：
      - 平均交叉熵损失（组内 softmax，正样本 index=0）
      - 组内 top-1 精度（argmax 是否为 0）
    返回: (avg_loss, top1_acc)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_batches = 0
    total_groups = 0

    for queries, triples, labels, group_ptr in val_loader:
        # 取向量
        q_embed = await DEFAULT_EMBEDDER.embed_with_tensor(queries)  # [N,D] 各
        triple_embed = await DEFAULT_EMBEDDER.embed_with_tensor(triples)  # [N,3,D]

        # 前向
        _, logit = model(q_embed, triple_embed)  # [N]

        # 组内 reshape
        B = group_ptr.numel() - 1  # 组数
        group_size = int((group_ptr[1] - group_ptr[0]).item())  # = 1+K
        logit = logit.view(B, group_size)  # [B, 1+K]

        # 组内交叉熵（正样本都在 index=0）
        target = torch.zeros(B, dtype=torch.long, device=logit.device)  # [B]
        loss = F.cross_entropy(logit, target)

        # top-1 精度（预测 index 是否为 0）
        pred = logit.argmax(dim=1)  # [B]
        correct = (pred == 0).sum().item()

        total_loss += loss.item()
        total_correct += correct
        total_batches += 1
        total_groups += B

    avg_loss = total_loss / max(total_batches, 1)
    top1_acc = total_correct / max(total_groups, 1)
    return avg_loss, top1_acc


# =====================
# 训练逻辑
# =====================
async def train():
    # 1) 读取数据并划分训练/验证
    with open("files/generated_triplet_question_dataset.jsonl", "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    # 3) 模型与优化器
    model = QTRPipeline(body=QTR_BODY_PRETRAIN, head=QTR_HEAD_PRETRAIN)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        **QTR_PRETRAIN_OPTIMIZER_ARGS
    )

    # 4) 训练循环 + 验证 + 保存最佳模型
    best_val_loss = float("inf")
    save_dir = "checkpoints/qtr"
    os.makedirs(save_dir, exist_ok=True)
    best_path = os.path.join(save_dir, "best.pt")
    last_path = os.path.join(save_dir, "last.pt")

    max_epochs = 100
    log_every = 20  # 每多少个训练 batch 打印一次 loss

    global_step = 0

    # 2) 构建 DataLoader（训练/验证可共享同一构造函数）
    train_loader, val_loader = build_mixed_dataloader(
        raw_dataset=raw_data,
        kg_vocab=KG_VOCAB,
        **QTR_PRETRAIN_DATALOADER_BUILDING_ARGS
    )

    for epoch in range(1, max_epochs + 1):
        QTR_BODY_PRETRAIN.train()
        QTR_HEAD_PRETRAIN.train()

        for i, (queries, triples, labels, group_ptr) in enumerate(train_loader, start=1):
            # ---- 取向量
            q_embed = torch.tensor(await DEFAULT_EMBEDDER.embed_cached(queries))  # [N,D] 各
            triple_embed = torch.tensor(await DEFAULT_EMBEDDER.embed_cached(triples))  # [N,3,D]

            # ---- 前向
            _, logit = model(q_embed, triple_embed)
            # ---- 组内 reshape -> [B, 1+K]
            B = group_ptr.numel() - 1
            group_size = int((group_ptr[1] - group_ptr[0]).item())  # = 1+K
            logit = logit.view(B, group_size)

            # ---- 交叉熵（正样本 index=0）
            target = torch.zeros(B, dtype=torch.long, device=logit.device)
            loss = F.cross_entropy(logit, target)

            # ---- 反向与优化
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            # print(loss)

            global_step += 1
            if i % log_every == 0:
                print(f"[Epoch {epoch} | Step {global_step}] train_loss = {loss.item():.4f}")

        # ---- 每个 epoch 结束后做一次验证
        val_loss, val_acc = await validate(model, val_loader)
        print(f"[Epoch {epoch}] VAL  loss = {val_loss:.4f} | top1_acc = {val_acc:.4f}")

        # ---- 保存最佳
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save(
                save_dir="checkpoints/qtr/best",
                overwrite=True,
                extra_info={
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "seed": SEED,
                },
            )
            print(f"  ↳ New best saved to: {best_path}")
    else:
        model.save(
            save_dir="checkpoints/qtr/last",
            overwrite=True,
            extra_info={
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "seed": SEED,
            },
        )
        print(f"  ↳ New best saved to: {last_path}")

    print("Training finished.")


if __name__ == "__main__":
    asyncio.run(train())
