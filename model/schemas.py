from model.vocab.knowledge_graph_vocab import KnowledgeGraphVocab
from pydantic import BaseModel, computed_field, ConfigDict
from typing import Optional, List, Any, Tuple, Union
import torch

import networkx as nx
from typing import List, Sequence, Set, Dict
from collections import defaultdict
from config.model_args import DEVICE
from loguru import logger
import torch.nn as nn
from config.model_args import QTR_BODY_ATT_ARGS, CONV_ATT_ARGS

def collect_answer_triple_ids(
    sources: Sequence[str],
    tails: Sequence[str],
    triples: Sequence[Sequence[str]],
    bidirectional: bool = True  # 默认双向建边：对每个 (h,r,t) 同时建 h->t 与 t->h
) -> List[int]:
    """
    返回命中“答案路径”的三元组下标（不再返回全量 mask）。
    规则：
      - 建有向图；若 bidirectional=True，则每条三元组的索引同时挂到正反两条边上；
      - 对每个 (source, tail) 取所有最短路径；
      - 将路径上的边所关联到的 triple 索引收集起来并去重；
      - 返回排序后的下标列表（长度 <= len(triples)）。
    """
    # 1) 建图：每条边维护其命中的 triple 索引列表
    G = nx.DiGraph()
    for idx, (h, r, t) in enumerate(triples):
        # forward
        if G.has_edge(h, t):
            G[h][t].setdefault("triple_ids", []).append(idx)
        else:
            G.add_edge(h, t, triple_ids=[idx])
        # backward (可选)
        if bidirectional:
            if G.has_edge(t, h):
                G[t][h].setdefault("triple_ids", []).append(idx)
            else:
                G.add_edge(t, h, triple_ids=[idx])

    def _all_shortest_paths_safe(g: nx.DiGraph, s: str, t: str):
        try:
            return list(nx.all_shortest_paths(g, s, t))
        except Exception:
            return []

    # 2) 搜最短路并收集 triple 下标
    positive_ids: Set[int] = set()
    for s in sources:
        for t in tails:
            paths = _all_shortest_paths_safe(G, s, t)
            if not paths:
                continue
            # 这些 path 都是 s->t 的最短路径
            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    for tid in G[u][v].get("triple_ids", []):
                        positive_ids.add(tid)

    # 3) 返回排序后的下标列表
    return sorted(positive_ids)


def one_hop_positive_with_constrains(
        sources: Sequence[str],
        constrains: Sequence[str],
        triples: Sequence[Sequence[str]],
        bidirectional: bool = True
) -> List[int]:
    """
    在仅一跳内判断 sources 是否与 constrains 直接相邻。
    命中的边（连接 source 与 constraint 的三元组）对应的 triple 索引加入 positive_ids。

    参数
    ----
    sources:     源实体列表
    constrains:  约束实体列表（也是实体名）
    triples:     三元组列表，每项为 [head, relation, tail]
    bidirectional: 若为 True，则 (h,t) 与 (t,h) 均视作一跳邻接（默认 True）

    返回
    ----
    positive_ids: List[int]，满足条件的一跳边在 triples 中的索引（去重、升序）
    """
    if not constrains:
        return []
    source_set = set(sources)
    constrain_set = set(constrains)
    positive = set()

    for idx, (h, r, t) in enumerate(triples):
        # source -> constraint
        if h in source_set and t in constrain_set:
            positive.add(idx)
            continue
        # constraint -> source (若视为双向一跳)
        if bidirectional and t in source_set and h in constrain_set:
            positive.add(idx)

    return sorted(positive)


class KGDataset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # 支持torch

    # 必填field
    device: Union[str, torch.device] = DEVICE
    triplet_text: List[List[str]]
    query: str
    source_entity: List[str]
    answer_entity: Optional[List[str]] = None

    constrain_entity: Optional[List[str]] = None
    answer_triplet_index: Optional[List[int]] = None

    graph_id: Optional[str] = None

    # 基础属性 (自动计算)
    edge_index: Optional[torch.Tensor] = None  # [2, num_relations]
    edge_type: Optional[torch.Tensor] = None  # [num_relations] 是映射到 id 的 edge_type
    num_entities: Optional[int] = None
    num_relations_types: Optional[int] = None
    num_triplets: Optional[int] = None

    # 大模型部分
    sub_query: Optional[List[str]] = None
    memory: Optional[List[List[str]]] = None

    # 图推理部分
    # 所有向量全初始化为0，实际使用才会实时计算并缓存
    query_emb: Optional[torch.Tensor] = None  # [1024]
    x: Optional[torch.Tensor] = None  # [N, 1024]: 节点的embedding
    edge_attr: Optional[torch.Tensor] = None  # [E, 1024]: 边的embedding
    qtr_emb: Optional[torch.Tensor] = None  # [E, h]: idx是edge_index的每一个三元组的emb
    qtr_score: Optional[torch.Tensor] = None  # [E, 1]: qtr_head出来的分数
    edge_mask: Optional[torch.BoolTensor] = None  # [E]
    source_entity_index: Optional[torch.Tensor] = None
    message_flow_emb: Optional[torch.Tensor] = None  # [N, h]: 消息汇聚到某一个实体的embedding，和实体本身的embedding独立开来

    edge_score: Optional[torch.Tensor] = None  # [E], 三元组的打分

    # 词表(每一个图实例间都独立)
    vocab: KnowledgeGraphVocab = KnowledgeGraphVocab()

    def model_post_init(self, __context):
        # 基本信息
        self.edge_index, self.edge_type = self._edge_index_and_type()
        self.source_entity_index = torch.tensor(self.vocab.encode_entity(self.source_entity), dtype=torch.long)

        self.num_entities = self.vocab.num_entities
        self.num_relations_types = self.vocab.num_relations
        self.num_triplets = len(self.triplet_text)


        # 初始化图计算的向量
        self.query_emb = torch.full((1024, ), float('nan'), device=self.device, dtype=torch.float32)
        self.x = torch.full((self.num_entities, 1024), float('nan'), device=self.device, dtype=torch.float32)
        self.edge_attr = torch.full((self.num_relations_types, 1024), float('nan'), device=self.device, dtype=torch.float32)
        self.qtr_emb = torch.full((self.num_triplets, QTR_BODY_ATT_ARGS['output_dim']), float('nan'), device=self.device, dtype=torch.float32)
        self.qtr_score = torch.full((self.num_triplets, ), float('nan'), device=self.device, dtype=torch.float32)
        self.message_flow_emb = torch.zeros((self.num_entities, CONV_ATT_ARGS['dim']), device=self.device, dtype=torch.float32)
        # self.message_flow_emb = nn.Parameter(
        #     torch.empty(self.num_entities, 64, device=self.device, dtype=torch.float32)
        # )
        # torch.nn.init.normal_(self.message_flow_emb, mean=0.0, std=0.02)
        self.edge_score = torch.full((self.num_triplets, ), float('nan'), dtype=torch.float32)


        if not self.sub_query:
            self.sub_query = [self.query]

        # # 计算答案三元组标签
        # self.answer_triplet_index = collect_answer_triple_ids(self.source_entity, self.answer_entity, self.triplet_text)
        # subgraph_e = list(set([x for i in self.answer_triplet_index for x in (self.triplet_text[i][0], self.triplet_text[i][2])]))
        # self.answer_triplet_index += one_hop_positive_with_constrains(subgraph_e, self.constrain_entity, self.triplet_text)
        # self.answer_triplet_index = list(set(self.answer_triplet_index))  # 去重


    def _edge_index_and_type(self) -> Tuple[torch.Tensor, torch.Tensor]:
        triplet_idx = self.vocab.encode_triple(self.triplet_text)
        if not triplet_idx:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_type = torch.empty((0,), dtype=torch.long)
            return edge_index, edge_type

        heads, rels, tails = zip(*triplet_idx)  # tuples of len=E
        edge_index = torch.tensor([heads, tails], dtype=torch.long)  # (2, E)
        edge_type = torch.tensor(rels, dtype=torch.long)  # (E,)
        return edge_index, edge_type

    def to(self, device: Union[str, torch.device]):
        """
        将所有 tensor 移到指定 device，类似 tensor 的 to
        """
        tensor_fields = [
            "edge_index",
            "edge_type",
            "x",
            "edge_attr",
            "qtr_emb",
            "edge_mask",
            "source_entity_index",
            "message_flow_emb",
        ]

        for name in tensor_fields:
            t = getattr(self, name)
            if isinstance(t, torch.Tensor):
                setattr(self, name, t.to(device))

        return self

    def __add__(self, other: "KGDataset") -> "KGDataset":
        if not isinstance(other, KGDataset):
            raise TypeError(f"KGDataset can only add KGDataset, got {type(other)}")
        if self.query != other.query \
                or self.source_entity != other.source_entity \
                or self.answer_entity != other.answer_entity:
            raise ValueError("KGDataset + KGDataset 失败：query/source_entity/answer_entity 不一致")

        # 右侧含 emb -> 仅记录错误日志，不使用其 emb
        def _tensor_has_value(t):
            if not isinstance(t, torch.Tensor) or t.numel() == 0:
                return False
            if t.dtype.is_floating_point:
                return not torch.isnan(t).all().item()
            return True

        if any([
            _tensor_has_value(getattr(other, "x", None)),
            _tensor_has_value(getattr(other, "edge_attr", None)),
            _tensor_has_value(getattr(other, "qtr_emb", None)),
            _tensor_has_value(getattr(other, "message_flow_emb", None)),
            _tensor_has_value(getattr(other, "edge_score", None)),
        ]):
            logger.log("ERROR", "KGDataset + KGDataset：右操作数包含已计算的 embedding，将被忽略。")

        # 仅做浅拷贝（不 deepcopy、不 clone），随后所有“张量字段”都用新张量整体重绑，避免对旧张量做原地写
        new = self.model_copy(deep=False)

        # 合并三元组：保留左侧顺序，右侧去重追加
        def _to_tuple_list(tt):
            return [tuple(x) for x in tt] if tt else []

        left_tris = _to_tuple_list(self.triplet_text)
        right_tris = _to_tuple_list(other.triplet_text)
        exist = set(left_tris)
        appended = [t for t in right_tris if t not in exist]
        merged_tuple = left_tris + appended
        new.triplet_text = [list(t) for t in merged_tuple]

        E_old = len(left_tris)
        N_old = self.num_entities
        R_old = self.num_relations_types

        E_new = len(merged_tuple)

        # 复用左侧 vocab 进行编码（此步会扩 vocab）
        new.edge_index, new.edge_type = new._edge_index_and_type()

        # 基本统计（“新”的尺寸）
        new.num_entities = new.vocab.num_entities
        new.num_relations_types = new.vocab.num_relations
        new.num_triplets = E_new  # <<< 别漏这个

        new.source_entity_index = torch.tensor(
            new.vocab.encode_entity(new.source_entity), dtype=torch.long,
            device=self.source_entity_index.device if isinstance(self.source_entity_index, torch.Tensor) else None
        )

        # ---------- 下面均为“新张量整体重绑”，不对旧张量做原地写入 ----------

        # x: [N, 1024] 扩容（旧切片直接参与 cat，梯度可回传；新补全用 NaN）
        if isinstance(self.x, torch.Tensor):
            D_x = self.x.size(1)
            if new.num_entities > N_old:
                pad = torch.full((new.num_entities - N_old, D_x), float('nan'),
                                 dtype=self.x.dtype, device=self.x.device)
                new.x = torch.cat([self.x, pad], dim=0)
            else:
                new.x = self.x
        else:
            # 左侧还未初始化 x，则构造全 NaN
            new.x = torch.full((new.num_entities, 1024), float('nan'),
                               dtype=torch.float32, device=new.device)

        # edge_attr: [R, 1024] 扩容
        if isinstance(self.edge_attr, torch.Tensor):
            D_e = self.edge_attr.size(1)
            if new.num_relations_types > R_old:
                pad = torch.full((new.num_relations_types - R_old, D_e), float('nan'),
                                 dtype=self.edge_attr.dtype, device=self.edge_attr.device)
                new.edge_attr = torch.cat([self.edge_attr, pad], dim=0)
            else:
                new.edge_attr = self.edge_attr
        else:
            new.edge_attr = torch.full((new.num_relations_types, 1024), float('nan'),
                                       dtype=torch.float32, device=new.device)

        # qtr_emb: [E, h] 与 qtr_score: [E] —— 只扩到新边数，旧行保持；新增行填 NaN
        h = self.qtr_emb.size(1)
        if E_new > E_old:
            qtr_emb_pad = torch.full((E_new - E_old, h), float('nan'),
                             dtype=self.qtr_emb.dtype, device=self.qtr_emb.device)
            qtr_score_pad = torch.full((E_new - E_old, ), float('nan'),
                             dtype=self.qtr_score.dtype, device=self.qtr_score.device)
            new.qtr_emb = torch.cat([self.qtr_emb, qtr_emb_pad], dim=0)
            new.qtr_score = torch.cat([self.qtr_score, qtr_score_pad], dim=0)
        else:
            new.qtr_emb = self.qtr_emb
            new.qtr_score = self.qtr_score

        # message_flow_emb: [N, h2] 扩容
        if isinstance(self.message_flow_emb, torch.Tensor) and self.message_flow_emb.dim() == 2:
            h2 = self.message_flow_emb.size(1)
            if new.num_entities > N_old:
                pad = torch.full((new.num_entities - N_old, h2), float('nan'),
                                 dtype=self.message_flow_emb.dtype, device=self.message_flow_emb.device)
                new.message_flow_emb = torch.cat([self.message_flow_emb, pad], dim=0)
            else:
                new.message_flow_emb = self.message_flow_emb
        else:
            new.message_flow_emb = torch.full((new.num_entities, 64), float('nan'),
                                              dtype=torch.float32, device=new.device)

        # edge_score: [E] 旧值保留，新增为 NaN
        if isinstance(self.edge_score, torch.Tensor):
            if E_new > E_old:
                pad = torch.full((E_new - E_old,), float('nan'),
                                 dtype=self.edge_score.dtype, device=self.edge_score.device)
                new.edge_score = torch.cat([self.edge_score, pad], dim=0)
            else:
                new.edge_score = self.edge_score
        else:
            new.edge_score = torch.full((E_new,), float('nan'),
                                        dtype=torch.float32, device=new.device)

        # edge_mask: [E] 旧值保留，新增 False
        if isinstance(self.edge_mask, torch.Tensor):
            if E_new > E_old:
                pad = torch.zeros((E_new - E_old,), dtype=torch.bool, device=self.edge_mask.device)
                new.edge_mask = torch.cat([self.edge_mask, pad], dim=0)
            else:
                new.edge_mask = self.edge_mask
        else:
            # 保持 None（按你之前的设定）；如需默认全 False，也可改为 torch.zeros(E_new, ...)
            new.edge_mask = None

        # 设备一致（仅重绑字段；不做就地 .to_）
        new = new.to(new.device)
        return new

class MessagePassingFlow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ori_triplet_text: List[List[str]]  # 原始的完整的字符串三元组

    triplet_index: torch.Tensor
    from_entity_index: torch.Tensor
    to_entity_index: torch.Tensor

    # 自动计算
    edge_mask: Optional[torch.Tensor] = None
    subgraph_triplet_text: Optional[List[List[str]]] = None

    def model_post_init(self, __context):
        self.subgraph_triplet_text = [self.ori_triplet_text[i] for i in self.triplet_index]
        self.edge_mask = torch.zeros(len(self.ori_triplet_text), dtype=torch.bool)
        self.edge_mask[self.triplet_index] = True

    def __add__(self, other: "MessagePassingFlow") -> "MessagePassingFlow":
        if not isinstance(other, MessagePassingFlow):
            return NotImplemented

        # 1) 语义一致性：必须共享同一套原始三元组
        if self.ori_triplet_text != other.ori_triplet_text:
            raise ValueError("ori_triplet_text 不一致，无法做并集。")

        # 2) 稳定并集：先保留左侧顺序，再追加右侧未出现的
        left = self.triplet_index.view(-1).tolist()
        right = other.triplet_index.view(-1).tolist()

        seen = set()
        union_idx_list = []
        for x in left + right:
            if x not in seen:
                union_idx_list.append(x)
                seen.add(x)

        # 3) 构造 index -> (from, to) 的映射，并校验重复边一致性
        def build_map(flow: "MessagePassingFlow"):
            ti = flow.triplet_index.view(-1).tolist()
            fr = flow.from_entity_index.view(-1).tolist()
            to = flow.to_entity_index.view(-1).tolist()
            return {ti[i]: (fr[i], to[i]) for i in range(len(ti))}

        map_self = build_map(self)
        map_other = build_map(other)

        new_from_list, new_to_list = [], []

        for idx in union_idx_list:
            if idx in map_self and idx in map_other:
                f1, t1 = map_self[idx]
                f2, t2 = map_other[idx]
                if (f1, t1) != (f2, t2):
                    raise ValueError(
                        f"并集冲突：triplet_index={idx} 的 (from,to) 不一致："
                        f"{(f1, t1)} vs {(f2, t2)}"
                    )
                f, t = f1, t1
            elif idx in map_self:
                f, t = map_self[idx]
            elif idx in map_other:
                f, t = map_other[idx]
            else:
                # 理论上不会发生，因为 union 来源于两侧之一
                raise RuntimeError(f"找不到 triplet_index={idx} 的 (from,to) 信息。")

            new_from_list.append(f)
            new_to_list.append(t)

        # 4) 设备/类型对齐：沿用左操作数设置
        device = self.triplet_index.device
        new_triplet_index = torch.tensor(union_idx_list, dtype=torch.long, device=device)
        new_from = torch.tensor(new_from_list, dtype=self.from_entity_index.dtype, device=self.from_entity_index.device)
        new_to = torch.tensor(new_to_list, dtype=self.to_entity_index.dtype, device=self.to_entity_index.device)

        # 5) 返回新实例（会自动触发 model_post_init 重算 mask / 子图文本）
        return MessagePassingFlow(
            ori_triplet_text=self.ori_triplet_text,
            triplet_index=new_triplet_index,
            from_entity_index=new_from,
            to_entity_index=new_to,
        )


if __name__ == '__main__':
    # 假设这是全图的三元组文本（索引从0开始）
    ori_triplet_text = [
        ["A", "rel", "B"],  # idx 0
        ["B", "rel", "C"],  # idx 1
        ["C", "rel", "D"],  # idx 2
        ["A", "rel", "D"],  # idx 3
        ["D", "rel", "E"],  # idx 4
    ]

    # flow_a：包含边 0、2
    flow_a = MessagePassingFlow(
        ori_triplet_text=ori_triplet_text,
        triplet_index=torch.tensor([0, 2]),
        from_entity_index=torch.tensor([0, 2]),
        to_entity_index=torch.tensor([1, 3]),
    )

    # flow_b：包含边 2、3（与 flow_a 在索引2上重叠，且 from/to 一致）
    flow_b = MessagePassingFlow(
        ori_triplet_text=ori_triplet_text,
        triplet_index=torch.tensor([2, 3]),
        from_entity_index=torch.tensor([2, 3]),
        to_entity_index=torch.tensor([3, 0]),
    )

    # 并集（稳定顺序）：应得到 [0, 2, 3]
    flow_u = flow_a + flow_b

    print("=== 成功并集示例 ===")
    print("并集 triplet_index:", flow_u.triplet_index.tolist())  # 期望: [0, 2, 3]
    print("并集 edge_mask(True的位置):",
          torch.nonzero(flow_u.edge_mask, as_tuple=False).view(-1).tolist())  # 期望: [0, 2, 3]
    print("并集 subgraph_triplet_text:", flow_u.subgraph_triplet_text)
    # 期望:
    # [
    #   ["A","rel","B"],  # idx 0
    #   ["C","rel","D"],  # idx 2
    #   ["A","rel","D"],  # idx 3
    # ]

    # 冲突测试：同一个 triplet_index=2，但 from/to 不一致
    flow_c = MessagePassingFlow(
        ori_triplet_text=ori_triplet_text,
        triplet_index=torch.tensor([2]),
        from_entity_index=torch.tensor([9]),  # 故意与 flow_a/flow_b 不同
        to_entity_index=torch.tensor([9]),
    )

    print("\n=== 冲突测试（预期抛 ValueError） ===")
    try:
        _ = flow_a + flow_c
    except ValueError as e:
        print("捕获到异常:", e)

    # 你也可以用“按位或”来做并集
