from typing import Tuple, List, Optional, Union, Any
import torch
from model.message_passing.subgraph_hop_selector import SubgraphHopSelector
from model.schemas import KGDataset, MessagePassingFlow
from model.message_passing.utils import bfs_by_threshold
from loguru import logger


BoolMask = torch.Tensor  # dtype=bool, shape: [E]
EdgeIds = torch.Tensor   # dtype=long

class MessagePassingSubgraphProvider:
    """
    消息传递子图收集器（仅正向）。
    三个接口：
      - collect(h1, h2): 精确收集 h1 -> h2
      - collect_to(h2):  收集 (h1 -> h2) + (h2 -> h2)
      - collect_within(max_hop): 收集小于 max_hop 的所有边（不含 max_hop->max_hop 与 0->0）
      - append_block_idx: 把这些边mask掉


    规则：
      - 同 hop (h1==h2) 使用高质量掩码 quality_mask + 双向展开
      - 跨 hop (h1>h2) 使用可行掩码 feasible_mask + 定向 (h1->h2)

    新增：
      - 追加掩码 append_mask：通过 append_mask() 逐次“或”并入内部 mask；
        在任何输出前都会与该 mask 做“与”操作以筛边。
    """

    def __init__(
        self,
        kg: KGDataset,
        *,
        directed: bool = False,
        expand_threshold: float = 0.0,
        quality_threshold: float = 0.0,
    ):
        self.kg = kg
        self.directed = directed
        self.expand_threshold = expand_threshold
        self.quality_threshold = quality_threshold

        self.triplet_text = kg.triplet_text
        self.vocab = kg.vocab
        # self.source_entity = kg.source_entity
        self.source_entity = kg.vocab.decode_entity(kg.source_entity_index.tolist())
        self.sub_query = kg.sub_query
        self.edge_idx = kg.edge_index  # [2, E]
        self.num_edges = self.edge_idx.size(1)


        # 阶段一：阈值筛选
        self._feasible_edge_mask: Optional[torch.Tensor] = None  # bool [E]
        self._quality_edge_masks: Optional[torch.Tensor] = None  # bool [E]

        # hop 选择器
        self._hop_selector: Optional[SubgraphHopSelector] = None

        # 追加掩码（累积 OR）
        self._append_mask: Optional[BoolMask] = None  # bool [E]

        # 预计算
        logger.info("start precomputing")
        self._run_threshold_selection()
        self._init_hop_selector()

    # def _apply_append_mask(self, mask: BoolMask) -> BoolMask:
    #     """将内部的追加掩码与当前 mask 做 AND；若未设置则原样返回。"""
    #     if self._append_mask is None:
    #         return mask
    #     return mask & self._append_mask

    def _run_threshold_selection(self):
        res = bfs_by_threshold(
            kg = self.kg,
            max_hops=4,  # 不限制，由 hop_selector 的距离掩码实际约束
            expand_threshold=self.expand_threshold,
            quality_threshold=self.quality_threshold,
            directed=self.directed,
        )
        self._feasible_edge_mask = res['feasible']['edge_masks']
        self._quality_edge_masks = res['quality']['edge_masks']

    def _init_hop_selector(self):
        self._hop_selector = SubgraphHopSelector(
            edge_index=self.edge_idx,
            num_nodes=self.kg.num_entities,
            directed=self.directed,
        )
        source_entity_index = self.vocab.encode_entity(self.source_entity)
        self._hop_selector.compute_distances(source_entity_index)

    # def _apply_quality_rule(self, base_mask: BoolMask, h1: int, h2: int) -> BoolMask:
    #     """同 hop 用 quality；跨 hop 用 feasible。"""
    #     assert self._feasible_edge_mask is not None and self._quality_edge_masks is not None
    #     if h1 == h2:
    #         mask = base_mask & self._quality_edge_masks
    #     else:
    #         mask = base_mask & self._feasible_edge_mask
    #     # 叠加“追加掩码”
    #     mask = self._apply_append_mask(mask)
    #     return mask

    # def _assemble_from_mask_samehop(self, mask: BoolMask):
    #     """同 hop：双向展开。"""
    #     if mask.sum() == 0:
    #         empty = torch.tensor([], dtype=torch.long)
    #         return empty, empty, empty
    #     idx = mask.nonzero(as_tuple=True)[0]            # [N]
    #     src_dst = self.edge_idx[:, mask]                # [2, N]
    #     triplet_idx = idx.repeat(2)                     # [2N]
    #     sources = torch.cat([src_dst[1], src_dst[0]], dim=0)  # [2N]
    #     targets = torch.cat([src_dst[0], src_dst[1]], dim=0)  # [2N]
    #
    #     return triplet_idx, sources, targets

    # def _assemble_from_mask_crosshop(self, h1: int, h2: int, base_mask: BoolMask) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """跨 hop：定向 h1->h2。"""
    #     directional_mask, src_idx, dst_idx = self._hop_selector.edge_mask_by_distance(begin_hop=h1, end_hop=h2)
    #     mask = base_mask & directional_mask
    #     mask = self._apply_append_mask(mask)
    #     if mask.sum() == 0:
    #         empty = torch.tensor([], dtype=torch.long)
    #         return empty, empty, empty
    #     triplet_idx = mask.nonzero(as_tuple=True)[0]
    #     sources = src_idx[mask]
    #     targets = dst_idx[mask]
    #     return triplet_idx, sources, targets

    @staticmethod
    def _cat_many(parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx_list, s_list, t_list = [], [], []
        for idx, s, t in parts:
            if idx.numel():
                idx_list.append(idx)
                s_list.append(s)
                t_list.append(t)
        if not idx_list:
            empty = torch.tensor([], dtype=torch.long)
            return empty, empty, empty
        return torch.cat(idx_list, dim=0), torch.cat(s_list, dim=0), torch.cat(t_list, dim=0)

    def _build_flow(
            self,
            triplet_idx: torch.Tensor,
            sources: torch.Tensor,
            targets: torch.Tensor,
    ) -> MessagePassingFlow:
        """
        将 (triplet_idx, sources, targets) 转成 MessagePassingFlow。
        - ori_triplet_text: [[text], ...] 形状对齐 triplet_idx
        - 其余字段保留为原始的 1D long Tensor
        """
        # 规范为 1D long
        triplet_idx = triplet_idx.reshape(-1).to(dtype=torch.long)
        sources = sources.reshape(-1).to(dtype=torch.long)
        targets = targets.reshape(-1).to(dtype=torch.long)

        assert triplet_idx.numel() == sources.numel() == targets.numel(), \
            "triplet_idx / sources / targets 长度必须一致"

        return MessagePassingFlow(
            ori_triplet_text=self.triplet_text,
            triplet_index=triplet_idx,
            from_entity_index=sources,
            to_entity_index=targets,
        )

    # 去重
    @staticmethod
    def deduplicate(ref: torch.Tensor, *args: torch.Tensor):
        if ref.ndim != 1:
            raise ValueError("ref 必须是一维张量 (N,)")
        n = ref.size(0)
        for i, a in enumerate(args):
            if a.size(0) != n:
                raise ValueError(f"*args 第 {i} 个张量长度与 ref 不一致")
        if n == 0:
            return (ref, *args)

        # unique 的顺序与 sorted=False 时“首次出现顺序”一致
        _, inv = torch.unique(ref, sorted=False, return_inverse=True)
        # 对每个类别求最小位置（即首次出现位置）
        first_pos = torch.full((inv.max().item() + 1,), n, dtype=torch.long)
        first_pos.scatter_reduce_(0, inv, torch.arange(n), reduce='amin', include_self=True)

        # 按首次出现的位置排序，保持原顺序
        keep_idx = first_pos.sort().values

        new_ref = ref.index_select(0, keep_idx)
        new_args = [a.index_select(0, keep_idx) for a in args]
        return (new_ref, *new_args)


    def _collect_once(self, h1: int, h2: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """仅返回三元组索引与两端点，用于批量组合。"""
        assert h1 >= 0 and h2 >= 0, "hop 必须为非负整数"
        base_mask, _, _ = self._hop_selector.edge_mask_by_distance(begin_hop=h1, end_hop=h2)
        mask = base_mask & self._feasible_edge_mask
        # mask = self._apply_append_mask(mask)

        if self._append_mask is not None:
            mask = mask & self._append_mask
        if h1 == h2:
            """同 hop：双向展开。"""
            if mask.sum() == 0:
                empty = torch.tensor([], dtype=torch.long)
                return empty, empty, empty
            idx = mask.nonzero(as_tuple=True)[0]  # [N]
            src_dst = self.edge_idx[:, mask]  # [2, N]
            triplet_idx = idx.repeat(2)  # [2N]
            sources = torch.cat([src_dst[1], src_dst[0]], dim=0)  # [2N]
            targets = torch.cat([src_dst[0], src_dst[1]], dim=0)  # [2N]
            return triplet_idx, sources, targets
            # return self._assemble_from_mask_samehop(mask)
        else:
            """跨 hop：定向 h1->h2。"""
            directional_mask, src_idx, dst_idx = self._hop_selector.edge_mask_by_distance(begin_hop=h1, end_hop=h2)
            mask = mask & directional_mask
            if mask.sum() == 0:
                empty = torch.tensor([], dtype=torch.long)
                return empty, empty, empty
            triplet_idx = mask.nonzero(as_tuple=True)[0]
            sources = src_idx[mask]
            targets = dst_idx[mask]
            return triplet_idx, sources, targets
            # return self._assemble_from_mask_crosshop(h1, h2, mask)

    # -------------------- 公共接口：精确收集 --------------------
    def collect(self, h1: int, h2: int) -> "MessagePassingFlow":
        """精确收集 h1 -> h2，并格式化为 MessagePassingFlow。"""
        triplet_idx, sources, targets = self._collect_once(h1, h2)
        triplet_idx, sources, targets = self.deduplicate(triplet_idx, sources, targets)
        return self._build_flow(triplet_idx, sources, targets)

    def collect_between(self, hop_from: int, hop_to: int) -> "MessagePassingFlow":
        """
        收集从 hop_from 到 hop_to 的边块（闭区间限制层号），并格式化为 MessagePassingFlow：
          - 排除 hop_from->hop_from
          - 包含 hop_from->hop_from-1
          - 对 k = hop_from-1 ... hop_to：
              * 加入 k->k
              * 若 k > hop_to 再加入 k->k-1
          - 不包含 hop_to->hop_to-1

        说明：
          - 同层使用质量掩码并双向展开；跨层使用可行掩码并按定向 (h1->h2)
        """
        assert hop_from >= 1, "hop_from 必须 >= 1"
        assert hop_to >= 0, "hop_to 必须 >= 0"
        assert hop_from >= hop_to, "要求 hop_from >= hop_to"

        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # 顶层向内：hop_from -> hop_from-1（排除 hop_from->hop_from）
        parts.append(self._collect_once(hop_from, hop_from - 1))

        # 对每层 k，加入同层与（若 k>hop_to）向内
        for k in range(hop_from - 1, hop_to - 1, -1):
            parts.append(self._collect_once(k, k))  # k -> k
            if k > hop_to:
                parts.append(self._collect_once(k, k - 1))  # k -> k-1

        triplet_idx, sources, targets = self._cat_many(parts)
        triplet_idx, sources, targets = self.deduplicate(triplet_idx, sources, targets)
        return self._build_flow(triplet_idx, sources, targets)

    def collect_within(self, max_hop: int) -> "MessagePassingFlow":
        """
        收集小于 max_hop 的所有边（排除 max_hop->max_hop 与 0->0），并格式化为 MessagePassingFlow。
        例如 max_hop=3：3->2, 2->2, 2->1, 1->1, 1->0
        """
        assert max_hop >= 1, "max_hop 必须 >= 1"

        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # 顶层向内：max_hop -> max_hop-1
        parts.append(self._collect_once(max_hop, max_hop - 1))

        # 逐层：k->k 与 k->k-1
        for k in range(max_hop - 1, 0, -1):
            parts.append(self._collect_once(k, k))  # 同 hop
            parts.append(self._collect_once(k, k - 1))  # 向内

        triplet_idx, sources, targets = self._cat_many(parts)
        triplet_idx, sources, targets = self.deduplicate(triplet_idx, sources, targets)
        return self._build_flow(triplet_idx, sources, targets)


def _test():
    # ===== 1) 构造更充足的 KG =====
    triplets = [
        ["Alice",   "likes",       "Bob"],          # e0  高分（主路径）
        ["Bob",     "friend_of",   "Charlie"],      # e1  高分（主路径）
        ["Charlie", "works_at",    "InceptionCorp"],# e2  中高分（分支）
        ["Bob",     "works_at",    "InceptionCorp"],# e3  中分（分支）
        ["Alice",   "lives_in",    "Paris"],        # e4  中分（分支）
        ["Paris",   "located_in",  "France"],       # e5  高分（分支/两跳）
        ["Charlie", "knows",       "Alice"],        # e6  回环（Charl -> Alice）
        ["Dave",    "likes",       "Alice"],        # e7  指向起点的边（入边）
        ["Eve",     "likes",       "Frank"],        # e8  离散子图（与起点无关）
        ["Frank",   "knows",       "Grace"],        # e9  离散子图内延伸
        ["Grace",   "knows",       "Heidi"],        # e10 离散子图内延伸
        ["Noise",   "random_link", "Junk"],         # e11  低分噪声
    ]

    query = "Who does Alice connect to (directly or within 3 hops)?"
    source_entity = ["Alice"]
    answer_entity = ["Charlie"]  # 仅占位

    kg = KGDataset(
        device="cuda",
        triplet_text=triplets,
        query=query,
        source_entity=source_entity,
        answer_entity=answer_entity,
    )

    # ===== 2) 预置边分数（与 triplets 对齐）=====
    # 设计：expand_threshold=0.60, quality_threshold=0.80
    # - 主路径 e0/e1 设为高分（≥0.85），确保 BFS 可达并进入“质量”子图
    # - 分支 e2=0.78（可行但非高质）、e3=0.65（可行）、e4=0.70（可行）、e5=0.88（高质）
    # - 回环 e6=0.83（高质，使得回到 Alice）
    # - 入边 e7=0.62（可行；无向模式下可帮助扩展，定向模式下不影响从 Alice 出发的正向可达）
    # - 离散子图 e8/e9/e10 即便高分，也因不可达不会出现在诱导子图里
    # - 噪声 e11=0.10（低分）
    scores = [
        0.90,  # e0
        0.87,  # e1
        0.78,  # e2
        0.65,  # e3
        0.70,  # e4
        0.88,  # e5
        0.83,  # e6
        0.62,  # e7
        0.92,  # e8  离散
        0.85,  # e9  离散
        0.81,  # e10 离散
        0.10,  # e11 噪声
    ]
    E = len(triplets)
    device = torch.device(kg.device) if hasattr(kg, "device") else torch.device("cpu")
    kg.qtr_score = torch.tensor(scores, dtype=torch.float32, device=device)
    assert kg.qtr_score.shape[0] == E

    # ===== 3) 无向模式：更易形成可达诱导子图 =====
    provider_ud = MessagePassingSubgraphProvider(
        kg=kg,
        directed=False,
        expand_threshold=0.60,
        quality_threshold=0.80,
    )
    flow_ud = provider_ud.collect_between(3, 0)

    print("\n===== UNDIRECTED (within 3 hops) =====")
    print("triplet_index:", flow_ud.triplet_index.tolist())
    print("from_entity_index:", flow_ud.from_entity_index.tolist())
    print("to_entity_index:", flow_ud.to_entity_index.tolist())

    # # 断言：主路径一定可达
    ud_idx = set(flow_ud.triplet_index.tolist())
    assert 0 in ud_idx and 1 in ud_idx, "无向模式下，主路径 e0/e1 应被收集"
    # e4/e5 分支：Alice->Paris->France，其中 e4=0.70（可行），e5=0.88（高质）
    # 因为是“within 3 hops”，且扩展阈值 0.60，无向模式通常能把分支也纳入诱导子图
    assert 4 in ud_idx and 5 in ud_idx, "应包含分支 e4/e5"
    # 回环 e6（0.83）为高质，可能通过 Charlie 再回到 Alice
    assert 6 in ud_idx, "应包含回环 e6"
    # e7（Dave->Alice）在无向模式下，作为 Alice 的“反向邻居”也可能出现
    assert 7 in ud_idx, "无向模式下应包含入边 e7"
    # 离散组件 e8/e9/e10 即便高分，由于与 Alice 不连通，不应出现
    assert not ({8, 9, 10} & ud_idx), "离散子图不应出现在诱导子图中"
    # 低分 e11 不应影响任何可达扩展
    assert 11 not in ud_idx, "低分噪声 e11 不应出现"

    # 掩码检查（可选观察）
    if provider_ud._quality_edge_masks is not None and provider_ud._feasible_edge_mask is not None:
        q_mask = provider_ud._quality_edge_masks.detach().cpu().numpy().astype(bool).tolist()
        f_mask = provider_ud._feasible_edge_mask.detach().cpu().numpy().astype(bool).tolist()
        print("quality_edge_masks:", q_mask)
        print("feasible_edge_mask:", f_mask)
        # 质量应包含分数≥0.80：e0,e1,e5,e6（以及离散 e8,e9,e10 但它们不可达不会影响诱导）
        # 可行应包含分数≥0.60：e0,e1,e2,e3,e4,e5,e6,e7


    # 友好打印：把实体 id 解码成名字（若 vocab 提供 decode_entity）
    def _pretty_print(flow, tag):
        print(f"\n(as names) {tag}")
        try:
            src_names = kg.vocab.decode_entity(flow.from_entity_index.tolist())
            dst_names = kg.vocab.decode_entity(flow.to_entity_index.tolist())
            for idx, s, t in zip(flow.triplet_index.tolist(), src_names, dst_names):
                print(f"[e{idx}] {s} -> {t}")
        except Exception:
            pass

    _pretty_print(flow_ud, "UNDIRECTED")

    print("\nAll assertions passed for the rich test.")
