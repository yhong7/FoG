import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, Subset

# ============== 工具：三元组去重（只对 triple 去重） ==============
def _uniq_triples_and_index(raw_dataset: List[Dict]) -> Tuple[List[Tuple[str,str,str]], Dict[Tuple[str,str,str], int]]:
    triples_uniq: List[Tuple[str,str,str]] = []
    triple_to_idx: Dict[Tuple[str,str,str], int] = {}

    for ex in raw_dataset:
        h, r, t = str(ex["triple"][0]), str(ex["triple"][1]), str(ex["triple"][2])
        key = (h, r, t)
        if key not in triple_to_idx:
            triple_to_idx[key] = len(triples_uniq)
            triples_uniq.append(key)

    return triples_uniq, triple_to_idx

# ============== 最小样本结构（样本仍按原条目，不去重） ==============
@dataclass
class _Example:
    query: str
    pos_uidx: int   # 指向“去重后的三元组表”的索引

class _QueryPosDataset(Dataset):
    def __init__(self, raw_dataset: List[Dict], triple_to_uidx: Dict[Tuple[str,str,str], int]):
        self.items: List[_Example] = []
        for ex in raw_dataset:
            q = str(ex["query"])
            key = (str(ex["triple"][0]), str(ex["triple"][1]), str(ex["triple"][2]))
            self.items.append(_Example(query=q, pos_uidx=triple_to_uidx[key]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ex = self.items[idx]
        return ex.query, ex.pos_uidx

# ============== Hard/Soft 池与采样器 ==============
class _HardPool:
    """
    head，rel，tail中至少有一个与给定三元组相同
    """
    def __init__(self, triple_ids: List[Tuple[int,int,int]], max_per_pos: Optional[int], seed: int):
        self.triple_ids = triple_ids
        self.max_per_pos = max_per_pos
        self.rng = random.Random(seed)
        self._pool: Dict[int, List[int]] = {}

    def build(self):
        by_h, by_r, by_t = {}, {}, {}
        for i, (h,r,t) in enumerate(self.triple_ids):
            by_h.setdefault(h, []).append(i)
            by_r.setdefault(r, []).append(i)
            by_t.setdefault(t, []).append(i)
        pool = {}
        for i, (h,r,t) in enumerate(self.triple_ids):
            cand = set(by_h.get(h, [])) | set(by_r.get(r, [])) | set(by_t.get(t, []))
            cand.discard(i)  # 排除自身
            cands = list(cand); self.rng.shuffle(cands)
            if self.max_per_pos is not None:
                cands = cands[:self.max_per_pos]
            pool[i] = cands
        self._pool = pool

    def candidates(self, i: int) -> List[int]:
        return self._pool.get(i, [])

class _SoftPool:
    """
    避开 HardPool 已经取过的，从其它三元组任意取
    """
    def __init__(self, num_triples: int, hard_pool: _HardPool, seed: int):
        self.n = num_triples
        self.hard = hard_pool
        self.rng = random.Random(seed)

    def sample(self, pos_idx: int, k: int) -> List[int]:
        if self.n <= 1: return []
        hard_set = set(self.hard.candidates(pos_idx))
        negs, tried, limit = [], 0, k*20
        while len(negs) < k and tried < limit:
            j = self.rng.randrange(self.n); tried += 1
            if j == pos_idx or j in hard_set or j in negs:
                continue
            negs.append(j)

        # 若还不够，放宽仅避开自身
        while len(negs) < k:
            j = self.rng.randrange(self.n)
            if j != pos_idx and j not in negs:
                negs.append(j)
        return negs

class _NegativeSampler:
    def __init__(self, hard: _HardPool, soft: _SoftPool, p_hard: float, seed: int):
        assert 0.0 <= p_hard <= 1.0
        self.hard, self.soft, self.p = hard, soft, p_hard
        self.rng = random.Random(seed)

    def sample(self, pos_idx: int, k: int) -> List[int]:
        k_h = int(round(self.p * k))
        k_s = k - k_h
        hard_cands = list(self.hard.candidates(pos_idx))
        self.rng.shuffle(hard_cands)
        hard_pick = hard_cands[:k_h]
        deficit = k_h - len(hard_pick)
        k_s += max(0, deficit)
        soft_pick = self.soft.sample(pos_idx, k_s)
        chosen = set(hard_pick) | set(soft_pick)

        # 再从 hard 里补齐
        while len(chosen) < k and hard_cands:
            j = hard_cands.pop()
            if j != pos_idx and j not in chosen:
                chosen.add(j)
        out = list(chosen)

        if len(out) > k:
            self.rng.shuffle(out); out = out[:k]
        return out


def build_mixed_dataloader(
    raw_dataset: List[Dict],
    kg_vocab,
    batch_size: int = 32,
    k_neg: int = 3,
    p_hard: float = 0.7,
    hard_max_per_pos: Optional[int] = 128,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: int = 2025,
    test_ratio: float = 1,
    test_shuffle: bool = False,
):
    """
    返回：train_loader, test_loader
    每个 loader 的 batch：
      queries: List[str]，长度 = B*(1+K)
      triples: List[(h,r,t)]（来自“去重后”的三元组表），同长
      labels : torch.FloatTensor，同长（1 为正样本）
      group_ptr: LongTensor，长度 B+1，指向每个 (1+K) 组的起止边界
    """
    # 允许 0 和 1
    assert 0.0 <= test_ratio <= 1.0, "test_ratio 必须在 [0,1] 之间"

    # 1) 对 triple 去重（样本本身不去重）
    triples_uniq, triple_to_uidx = _uniq_triples_and_index(raw_dataset)

    # 2) 数据集：每条样本指向它的“正样本 uniq_triple 索引”
    ds = _QueryPosDataset(raw_dataset, triple_to_uidx)

    # 3) 用 kg_vocab 对“去重后的三元组表”编码（仅用于构池，不做 embedding）
    heads = [h for (h,_,_) in triples_uniq]
    rels  = [r for (_,r,_) in triples_uniq]
    tails = [t for (_,_,t) in triples_uniq]
    hid = kg_vocab.encode_entity(heads)
    rid = kg_vocab.encode_relation(rels)
    tid = kg_vocab.encode_entity(tails)
    triple_ids = list(zip(hid, rid, tid))

    # 4) 构建 Hard/Soft 池与采样器（共享给 train/test，以保持输出格式一致）
    hard = _HardPool(triple_ids, max_per_pos=hard_max_per_pos, seed=seed); hard.build()
    soft = _SoftPool(num_triples=len(triples_uniq), hard_pool=hard, seed=seed)
    neg_sampler = _NegativeSampler(hard, soft, p_hard=p_hard, seed=seed)

    # 5) collate：直接返回 (queries, triples, labels, group_ptr)
    def _collate(batch: List[Tuple[str, int]]):
        B = len(batch)
        one_plus_k = 1 + k_neg
        queries: List[str] = []
        triples: List[Tuple[str, str, str]] = []
        labels = []
        group_ptr = torch.arange(0, B * one_plus_k + 1, step=one_plus_k, dtype=torch.long)

        for q, pos_uidx in batch:
            negs = neg_sampler.sample(pos_uidx, k_neg)
            cand = [pos_uidx] + negs
            queries.extend([q] * one_plus_k)
            triples.extend([triples_uniq[i] for i in cand])
            labels.extend([1] + [0] * k_neg)

        labels = torch.tensor(labels, dtype=torch.float32)

        # 每一个三元组转换成list，方便embedding
        triples = [list(t) for t in triples]
        return queries, triples, labels, group_ptr

    # 6) 随机性控制
    def _worker_init_fn(worker_id: int):
        base = seed + worker_id
        random.seed(base)
        torch.manual_seed(base)

    g_train = torch.Generator()
    g_test  = torch.Generator()
    if seed is not None:
        g_train.manual_seed(seed)
        g_test.manual_seed(seed + 1)

    # 7) 划分 train/test（确定性；支持边界值）
    n = len(ds)
    indices = list(range(n))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)

    if test_ratio == 0.0:
        train_idx = indices
        test_idx  = []
    elif test_ratio == 1.0:
        train_idx = []
        test_idx  = indices
    else:
        cut = max(1, int(round(n * (1 - test_ratio))))
        cut = min(cut, n - 1)  # 保证两边至少有 1 条
        train_idx = indices[:cut]
        test_idx  = indices[cut:]

    ds_train = Subset(ds, train_idx)
    ds_test  = Subset(ds, test_idx)

    # 空数据集时禁用 shuffle，避免 RandomSampler 在 len==0 时报错
    shuffle_train = shuffle and len(ds_train) > 0
    shuffle_test  = test_shuffle and len(ds_test) > 0

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        generator=g_train,
        pin_memory=False,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=batch_size,
        shuffle=shuffle_test,
        num_workers=num_workers,
        drop_last=False,             # 测试集一般不丢最后一批
        collate_fn=_collate,
        worker_init_fn=_worker_init_fn,
        generator=g_test,
        pin_memory=False,
    )

    return train_loader, test_loader



if __name__ == "__main__":
    import json
    from config.singleton import KG_VOCAB

    with open("../../files/generated_triplet_question_dataset.jsonl", "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f]

    train_loader, test_loader = build_mixed_dataloader(
        raw_dataset=raw_data,  # List[{"query": ..., "triple": ...}]
        kg_vocab=KG_VOCAB,
        k_neg=3,
        p_hard=0.3,        # 难分类样本（hrt至少一个与pos样本一致）
        batch_size=32,
        test_ratio=0.1,     # 10% 作为测试集
        shuffle=True,
        test_shuffle=False, # 测试集默认不打乱（可改 True）
    )

    for split_name, loader in [("train", train_loader), ("test", test_loader)]:
        for queries, triples, labels, group_ptr in loader:
            # queries   : List[str]，长度 = B*(1+K)
            # triples   : List[(h,r,t)]，同长
            # labels    : torch.FloatTensor，同长
            # group_ptr : LongTensor，长度 B+1，每组 query 的起止索引范围
            pass
