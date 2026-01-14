import asyncio
from typing import Dict, List, Sequence, Tuple, Optional, Any, Literal
import torch
from torch import nn
from config.model_args import DEVICE
from config.singleton import (
    CONV, SCORER,
    QTR_BODY,
)
from model.message_passing.conv import GraphMessagePassing
from model.message_passing.scorer import Scorer
from model.message_passing.message_passing_subgraph_provider import MessagePassingSubgraphProvider
from loguru import logger
# from config.demo_kg_dataset import DEMO_KG_DATASET
from model.schemas import KGDataset, MessagePassingFlow
import wandb
from model.message_passing.utils import update_qtr_emb
import os


class MessagePassingEpochTrainer:
    """
    单个 epoch 的“问答驱动 KG 消息传递 + QTR 训练”执行器。
    - 支持多 hop 子图顺序训练（例如 [2, 1]）。
    - 对每个 hop：
        1) 取子图三元组并做子图嵌入（仅对子图涉及的三元组 embedding）
        2) 依据已知答案三元组在原图中的 idx 构造标签，并负采样
        3) QTR 打分 -> BCEWithLogitsLoss（可按类不均衡调 pos_weight）
        4) 反传优化（梯度裁剪）
        5) 消息传递（用 QTR_BODY_PRETRAIN 产生的 qtr_emb 驱动 conv），更新 kg.x
    """
    def __init__(
        self,
        kg: KGDataset,
        conv: GraphMessagePassing,
        qtr_body = None,
        scorer: Optional[Scorer] = None ,
        device: torch.device = DEVICE,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        grad_clip: float = 1.0,
        neg_ratio: int = 5,
        pos_weight_ratio: float = 1.0
    ) -> None:
        self.kg = kg
        self.conv = conv.to(device)
        self.qtr_body = qtr_body.to(device)
        self.scorer = scorer.to(device)
        self.device = torch.device(device)
        self.grad_clip = grad_clip
        self.neg_ratio = neg_ratio
        self.pos_weight_ratio = pos_weight_ratio

        if hasattr(kg, "answer_triplet_index"):
            self.answer_triplet_index = kg.answer_triplet_index

        # 优化器初始化
        self.params = (list(self.qtr_body.parameters())
                        + list(self.conv.parameters())
                        + list(self.scorer.parameters())
        )
        self.optim = torch.optim.Adam(
            self.params,
            lr=lr, weight_decay=weight_decay
        )

        self.criterion = nn.BCEWithLogitsLoss(reduction="mean")  # pos_weight 训练时动态设置

        # 将 KG 主存放到目标设备
        # self.kg.to(self.device)
        # if hasattr(self.kg, "x"):
        #     self.kg.x = self.kg.x.to(self.device)
        # if hasattr(self.kg, "edge_attr"):
        #     self.kg.edge_attr = self.kg.edge_attr.to(self.device)
        # if hasattr(self.kg, "edge_index"):
        #     self.kg.edge_index = self.kg.edge_index.to(self.device)
        # if hasattr(self.kg, "edge_type"):
        #     self.kg.edge_type = self.kg.edge_type.to(self.device)
        # if hasattr(self.kg, "message_flow_emb"):
        #     self.kg.message_flow_emb = self.kg.message_flow_emb.to(self.device)



        # 子图提供器
        self.subgraph_provider =  MessagePassingSubgraphProvider(
            kg=self.kg,
            directed=False, expand_threshold=0.0, quality_threshold=0.0,
        )

    async def validate(
            self,
            hops: Sequence[int],
            threshold: float = 0.5,
            sample_neg_ratio: Optional[int] = None,  # None=用全部负例；否则按比例采样
            log_prefix: str = "val",
    ) -> Dict[int, Dict[str, float]]:
        """
        对给定 hops 依次做评估。
        - 默认使用子图中的全部负样本；可通过 sample_neg_ratio 控制负采样加速。
        - 指标：loss/acc/precision/recall/f1 以及 TP/FP/TN/FN 计数。
        返回：{hop: {"loss": x, "acc": y, "precision": p, "recall": r, "f1": f,
                     "TP": tp, "FP": fp, "TN": tn, "FN": fn}}
        """
        metrics: Dict[int, Dict[str, float]] = {}

        # 早退：无答案三元组
        answer_triple_idx_tensor = torch.as_tensor(
            self.answer_triplet_index, dtype=torch.long
        )
        if answer_triple_idx_tensor.numel() == 0:
            return metrics

        for hop in hops:
            # 1) 收集子图（与 train 保持相同入口）
            with torch.no_grad():
                flow = self.subgraph_provider.collect_between(hop_from=hop, hop_to=hop - 1)

                # 2) 子图嵌入（沿用 train 的 _embed_subgraph/特征取法）
                triplet_idx = flow.triplet_index
                src = flow.from_entity_index
                dst = flow.to_entity_index
                flow_emb = self.kg.message_flow_emb[src]  # [E_sub, h]

                # 3) 构造标签并按需要负采样（评估集）
                pos_mask = torch.isin(triplet_idx, answer_triple_idx_tensor)  # [E_sub]
                num_pos_total = int(pos_mask.sum().item())
                if num_pos_total == 0:
                    print(f"[{log_prefix}] hop={hop}: 子图无正样本，跳过验证。")
                    continue

                pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
                neg_all = (~pos_mask).nonzero(as_tuple=False).squeeze(-1)
                if (sample_neg_ratio is None) or (sample_neg_ratio <= 0):
                    neg_idx = neg_all
                else:
                    k_neg = min(len(neg_all), max(1, len(pos_idx) * sample_neg_ratio))
                    if k_neg < len(neg_all):
                        perm = torch.randperm(len(neg_all))[:k_neg]
                        neg_idx = neg_all[perm]
                    else:
                        neg_idx = neg_all

                eval_idx = torch.cat([pos_idx, neg_idx], dim=0)
                if eval_idx.numel() == 0:
                    print(f"[{log_prefix}] hop={hop}: 无可评估样本。")
                    continue

                y_eval = pos_mask.float()[eval_idx].to(self.device)
                global_idx = triplet_idx[eval_idx]
                eval_qtr_emb = self.kg.qtr_emb[global_idx]
                eval_flow_emb = flow_emb[eval_idx]

                # 4) 前向（与 train 的打分器一致）
                logits = self.scorer(eval_qtr_emb, eval_flow_emb).reshape(-1)

                # 5) 损失与指标口径对齐 train：复用 _step_optim 的“正例过采样 + 无权重 BCE”
                num_pos_eval = int((y_eval > 0.5).sum().item())
                num_neg_eval = int(y_eval.numel() - num_pos_eval)
                step_out = self._step_optim(
                    logits=logits,
                    y_train=y_eval,
                    num_pos=num_pos_eval,
                    num_neg=num_neg_eval,
                )
                loss = step_out["loss"].detach()

                # 6) 指标（_step_optim 已在原始评估集口径上计算了混淆矩阵相关量）
                acc = float(step_out["acc"])
                tp, tn = int(step_out["tp"]), int(step_out["tn"])
                fp, fn = int(step_out["fp"]), int(step_out["fn"])
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

                log_dict = {
                    "hop": hop,
                    "loss": round(float(loss.item()), 4),
                    "acc": round(float(acc), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "TP": tp,
                    "FP": fp,
                    "TN": tn,
                    "FN": fn,
                }
                print(
                    f"[{log_prefix}] "
                    + " | ".join([f"{k}={v}" for k, v in log_dict.items()])
                )

                metrics[hop] = {
                    "loss": log_dict["loss"],
                    "acc": log_dict["acc"],
                    "precision": log_dict["precision"],
                    "recall": log_dict["recall"],
                    "f1": log_dict["f1"],
                    "TP": log_dict["TP"],
                    "FP": log_dict["FP"],
                    "TN": log_dict["TN"],
                    "FN": log_dict["FN"],
                }

        return metrics

    async def train_epoch(
        self,
        hops: Sequence[int],
        log_prefix: str = "train",
    ) -> Dict[int, Dict[str, float]]:
        """
        对给定 hops 依次执行一个 epoch 的训练与消息传递更新。
        返回每个 hop 的指标字典：{hop: {"loss": x, "acc": y, "P": p, "N": n}}
        """
        answer_triple_idx_tensor = torch.as_tensor(self.answer_triplet_index, dtype=torch.long)
        metrics: Dict[int, Dict[str, float]] = {}

        if len(answer_triple_idx_tensor) == 0:
            return metrics

        # 查询向量
        # q_emb = await DEFAULT_EMBEDDER.embed_with_tensor(self.kg.query)

        losses = []
        for hop in hops:
            loss, log_dict = await self._train_one_hop(
                hop=hop,
                answer_triple_idx=answer_triple_idx_tensor,
                # q_emb=q_emb,
                log_prefix=log_prefix,
            )
            if log_dict is not None:
                metrics[hop] = log_dict
                losses.append(loss)

        self.optim.zero_grad(set_to_none=True)
        sum(losses).backward()
        if self.grad_clip is not None and self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.params, max_norm=self.grad_clip)
        self.optim.step()

        for i, (hop, m) in enumerate(metrics.items()):
            m["loss"] = float(losses[i].detach().item())

        return metrics


    def _build_labels_and_sample(
        self,
        triplet_idx: torch.Tensor,
        answer_triple_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        基于“已知答案三元组在原图中的 idx”构造标签，并进行负采样。
        返回：
            train_idx: [P+N]
            y_train: [P+N] (float)
            stats: (num_pos, num_neg_sampled)
        """
        neg_ratio = self.neg_ratio

        pos_mask = torch.isin(triplet_idx, answer_triple_idx)  # [K]
        y = pos_mask.float()  # [K]
        num_pos = int(pos_mask.sum().item())
        if num_pos == 0:
            return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), (0, 0)

        pos_idx_local = pos_mask.nonzero(as_tuple=False).squeeze(-1)  # [P]
        neg_idx_local = (~pos_mask).nonzero(as_tuple=False).squeeze(-1)  # [N]

        k_neg = min(len(neg_idx_local), max(1, len(pos_idx_local) * neg_ratio))
        if k_neg < len(neg_idx_local):
            perm = torch.randperm(len(neg_idx_local))[:k_neg]
            neg_idx_local = neg_idx_local[perm]

        train_idx = torch.cat([pos_idx_local, neg_idx_local], dim=0)
        y_train = y[train_idx].to(self.device)
        return train_idx, y_train, (len(pos_idx_local), len(neg_idx_local))

    def _step_optim(
            self,
            logits: torch.Tensor,
            y_train: torch.Tensor,
            num_pos: int,
            num_neg: int,
    ) -> Dict[str, Any]:

        # 1) 计算“期望倍数”（仅用于指导过采样），不再用于 pos_weight
        pos_weight_val = max(1.0, (num_neg / max(1, num_pos))) if num_pos > 0 else 1.0
        pos_weight_val *= self.pos_weight_ratio

        device = self.device if hasattr(self, "device") else logits.device

        # 2)
        with torch.no_grad():
            y_bin = y_train.view(-1).to(device=device)
            pos_mask = (y_bin > 0.5)
            neg_mask = ~pos_mask
            pos_idx = torch.nonzero(pos_mask, as_tuple=True)[0]
            neg_idx = torch.nonzero(neg_mask, as_tuple=True)[0]

            if pos_idx.numel() == 0:
                resample_indices = torch.arange(y_bin.size(0), device=device)
            else:
                max_pos_oversample = getattr(self, "max_pos_oversample", 10)
                int_rep = int(min(max(pos_weight_val, 1.0), float(max_pos_oversample)))
                int_rep = max(1, int_rep)
                frac_rep = float(pos_weight_val - int_rep)

                pos_idx_rep = pos_idx.repeat_interleave(int_rep)
                if frac_rep > 1e-8:
                    extra_mask = (torch.rand(pos_idx.size(0), device=device) < frac_rep)
                    if extra_mask.any():
                        pos_idx_rep = torch.cat([pos_idx_rep, pos_idx[extra_mask]], dim=0)

                resample_indices = torch.cat([neg_idx, pos_idx_rep], dim=0)

            if resample_indices.numel() > 1:
                perm = torch.randperm(resample_indices.numel(), device=device)
                resample_indices = resample_indices[perm]

        # 3)
        logits_rs = logits.index_select(0, resample_indices)
        y_rs = y_train.index_select(0, resample_indices)

        # 4) 无权重的 BCE
        loss_fn = nn.BCEWithLogitsLoss(reduction="mean").to(device=self.device)

        loss = loss_fn(logits_rs, y_rs)

        # 5) 指标仍基于原始 batch（更客观）
        with torch.no_grad():
            prob = torch.sigmoid(logits)
            preds = (prob >= 0.5).float()
            acc = (preds == y_train).float().mean().item()
            tp = ((preds == 1) & (y_train == 1)).sum().item()
            tn = ((preds == 0) & (y_train == 0)).sum().item()
            fp = ((preds == 1) & (y_train == 0)).sum().item()
            fn = ((preds == 0) & (y_train == 1)).sum().item()

        return {
            'loss': loss,
            'acc': acc,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
        }

    async def _train_one_hop(
        self,
        hop: int,
        answer_triple_idx: torch.Tensor,
        # q_emb: torch.Tensor,
        log_prefix: str = "train",
    ) -> Optional[Tuple["torch.Tensor", Dict[str, Any]]]:
        """
        单个 hop 的完整流程：收集/嵌入 -> 采样/打分/损失 -> 反传 -> 消息传递
        """
        # 收集子图
        flow = self.subgraph_provider.collect_between(hop_from=hop, hop_to=hop - 1)
        triplet_idx = flow.triplet_index

        # 标签构造与负采样
        train_idx, y_train, (num_pos, num_neg) = self._build_labels_and_sample(triplet_idx, answer_triple_idx)
        if train_idx.numel() == 0:
            print(f"[{log_prefix}] hop={hop}: 子图无正样本，跳过训练。")
            # 仍然做一次消息传递以更新表示
            # 不做
            return (None, None)

        # format
        triplet_idx = flow.triplet_index
        src = flow.from_entity_index
        dst = flow.to_entity_index

        flow_emb = self.kg.message_flow_emb[src]  # [E_sub, h]

        global_idx = triplet_idx[train_idx]
        train_qtr_emb = self.kg.qtr_emb[global_idx]
        train_flow_emb = flow_emb[train_idx]  # [E_train, h]

        logits = self.scorer(train_qtr_emb, train_flow_emb).reshape(-1)
        metric_dict = self._step_optim(logits, y_train, num_pos=num_pos, num_neg=num_neg)
        loss = metric_dict['loss']

        # 指标
        acc = metric_dict['acc']
        tp = metric_dict['tp']
        tn = metric_dict['tn']
        fp = metric_dict['fp']
        fn = metric_dict['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        log_dict = {
            "hop": hop,
            "acc": round(acc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        }
        print(f"[{log_prefix}] " + " | ".join([f"{k}={v}" for k, v in log_dict.items()]))


        qtr_emb = self.kg.qtr_emb[triplet_idx]
        flow_emb_new = self.conv(self.kg.query_emb, qtr_emb, flow_emb, src, dst, self.kg.num_entities)
        self.kg.message_flow_emb = flow_emb_new  # 覆盖更新(因为基于环带的消息传递不会有一个节点的重复更新）

        return loss, log_dict


# ------------------------ 使用示例 ------------------------ #

async def train(ds_type: Literal["webqsp", "cwq"]):
    from model.build_freebase_dataset.data_iterator import graph_dataset_iterator

    device = DEVICE
    async def train_epoch(kg):
        await update_qtr_emb(kg)
        trainer = MessagePassingEpochTrainer(
            kg=kg,
            conv=conv,
            qtr_body=qtr_body,
            scorer=scorer,
            device=DEVICE,
            lr=1e-5,
            weight_decay=1e-4,
            grad_clip=1.0,
            neg_ratio=5,
            pos_weight_ratio=1.0
        )
        # 单个 epoch：先 2-hop 再 1-hop
        metrics = await trainer.train_epoch(
            hops=[4, 3, 2, 1],
            log_prefix="train",
        )
        print("epoch metrics:", metrics)
        return metrics

    async def validate_epoch(kg):
        await update_qtr_emb(kg)
        validator = MessagePassingEpochTrainer(
            kg=kg,
            conv=conv,
            qtr_body=qtr_body,
            scorer=scorer,
            device=DEVICE,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            neg_ratio=5,
            pos_weight_ratio=1.2
        )
        # 例：答案三元组在“原图”中的索引（外部传入，不再硬编码）
        # 单个 epoch：先 2-hop 再 1-hop
        metrics = await validator.validate(
            hops=[4, 3, 2, 1],
            log_prefix="validate",
        )
        print("epoch metrics:", metrics)
        return metrics

    qtr_body = QTR_BODY.to(device)
    conv = CONV.to(device)
    scorer = SCORER.to(device)
    # qtr_body.freeze_backbone()

    best_re = 0
    best_pr = 0

    for epoch in range(1000):
        for kg_idx, kg in enumerate(graph_dataset_iterator(ds_type, mode="train")):
            # 预计算qtr
            # from config.demo_kg_dataset import DEMO_KG_DATASET
            # from copy import deepcopy
            # kg = deepcopy(DEMO_KG_DATASET)


            if not kg.answer_triplet_index:
                logger.warning(f"graph: {kg.graph_id}, 无标签")
                print(kg.answer_triplet_index)
                # continue
            # if kg_idx == 5:
            #     break
            # kg = DEMO_KG_DATASET
            else:
                await train_epoch(kg)
                # try:
                #     await train_epoch(kg)
                # except Exception as e:
                #     logger.error(f"train exception: {e}")
                #     continue

            # 每20个图进一次验证集
            if kg_idx % 20 == 0:
                logger.info(f"graph_{kg_idx} validate: ")
                tp, fp, tn, fn = 0, 0, 0, 0
                total_loss = 0
                for val_kg_idx, val_kg in enumerate(graph_dataset_iterator(ds_type, mode="validate")):
                    # print(val_kg_idx)
                    if val_kg_idx in [i for i in range(20)]:
                        if not val_kg.answer_triplet_index:
                            logger.warning(f"graph_{val_kg.graph_id}")
                            pass
                        else:
                            try:
                                metrics = await validate_epoch(val_kg)
                                for hop, dic in metrics.items():
                                    tp += dic['TP']
                                    fp += dic['FP']
                                    tn += dic['TN']
                                    fn += dic['FN']
                                    total_loss += dic['loss']
                            except Exception as e:
                                logger.error(f"validate exception: {e}")
                                continue
                    if val_kg_idx > 20:
                        break
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

                # ===== 新增：满足阈值则额外保存一次 =====
                if (recall > 0.8) and (precision > 0.05):
                    extra_dir_name = f"re{recall:.2f}_pr{precision:.2f}_epoch{epoch}_graph{kg_idx}"
                    # 四类模型分别保存到同名子目录（与原有目录结构一致）
                    for sub, obj, fname in [
                        ("conv", conv, "conv.pt"),
                        ("qtr_body", qtr_body, "qtr_body.pt"),
                        ("scorer", scorer, "scorer.pt")
                    ]:
                        save_dir = os.path.join("./checkpoints", sub, extra_dir_name)
                        os.makedirs(save_dir, exist_ok=True)
                        obj.save(os.path.join(save_dir, fname), overwrite=True)
                    logger.info(f"[EXTRA SAVE] 模型已额外保存到 {extra_dir_name}")

            if kg_idx % 200 == 0:
                conv.save(f"./checkpoints/conv/graph_{epoch}_{kg_idx}/conv.pt", overwrite=True)
                qtr_body.save(f"./checkpoints/qtr_body/graph_{epoch}_{kg_idx}/qtr_body.pt", overwrite=True)
                scorer.save(f"./checkpoints/scorer/graph_{epoch}_{kg_idx}/scorer.pt", overwrite=True)

if __name__ == "__main__":

    asyncio.run(train())
