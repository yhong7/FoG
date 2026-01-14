from collections import deque
from typing import Iterable, Sequence, Union, Optional, List
import torch


class SubgraphHopSelector:
    """
    在给定图上进行多源 BFS，并基于最近一次 BFS 的距离结果做筛选/查询。

    参数
    ----
    edge_index: torch.LongTensor，[2, E]（CPU），有向边 (u->v)
    num_nodes: int，节点数，节点 id ∈ [0, num_nodes)
    directed: bool，若 False 则按无向图处理（为每条 (u->v) 额外加入 (v->u)）

    属性
    ----
    last_dist: torch.LongTensor，[num_nodes]，最近一次 BFS 计算得到的距离；在调用 compute() 之前为 None
    """
    def __init__(self,
                 edge_index: Union[List[List[int]], torch.Tensor],
                 num_nodes: int,
                 directed: bool = False) -> None:
        if not isinstance(edge_index, torch.Tensor):
            edge_index = torch.tensor(edge_index)

        if edge_index.shape[0] != 2:
            raise ValueError("edge_index 形状应为 [2, E]")

        self.edge_index = edge_index
        self.num_nodes = int(num_nodes)
        self.directed = bool(directed)
        self._adj: Optional[List[List[int]]] = None  # 邻接表（lazy 构建）

        # cache
        self.last_dist: Optional[torch.LongTensor] = None

    # ---------- 基础：邻接构建与 BFS ----------
    def _build_adj(self) -> None:
        if self._adj is not None:
            return
        src = self.edge_index[0].tolist()
        dst = self.edge_index[1].tolist()
        adj: List[List[int]] = [[] for _ in range(self.num_nodes)]
        for u, v in zip(src, dst):
            adj[u].append(v)
            if not self.directed:
                adj[v].append(u)
        self._adj = adj

    def compute_distances(self, start_idx: Union[int, Sequence[int], Iterable[int], torch.Tensor]) -> torch.LongTensor:
        """
        多源 BFS，计算并缓存每个点到任一 start 的最短跳数；不可达为 -1。
        返回值也会存到 self.last_dist。
        """
        # 统一起点
        if isinstance(start_idx, torch.Tensor):
            starts = start_idx.detach().view(-1).tolist()
        elif isinstance(start_idx, int):
            starts = [start_idx]
        else:
            starts = list(start_idx)

        if len(starts) == 0:
            self.last_dist = torch.full((self.num_nodes,), -1, dtype=torch.long)
            return self.last_dist

        # 去重 + 合法性检查
        starts = sorted(set(int(s) for s in starts))
        for s in starts:
            if not (0 <= s < self.num_nodes):
                raise ValueError(f"start_idx 包含非法节点编号 {s}，应在 [0, {self.num_nodes}) 内。")

        # 准备邻接
        self._build_adj()
        assert self._adj is not None

        # BFS
        dist = [-1] * self.num_nodes
        q = deque()
        for s in starts:
            dist[s] = 0
            q.append(s)

        while q:
            u = q.popleft()
            du = dist[u]
            for v in self._adj[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(v)

        self.last_dist = torch.tensor(dist, dtype=torch.long)
        return self.last_dist

    # ---------- 工具：距离集合归一化 ----------
    @staticmethod
    def _normalize_distance_selector(sel: Union[int, Sequence[int], Iterable[int]]) -> torch.Tensor:
        """
        把一个 int 或一组 int 归一为去重、升序的 LongTensor（允许包含 -1）。
        """
        if isinstance(sel, int):
            return torch.tensor([sel], dtype=torch.long)
        return torch.tensor(sorted(set(int(x) for x in sel)), dtype=torch.long)

    # ---------- 方法 1（改）：按（begin_hop -> end_hop）筛边 ----------
    @torch.no_grad()
    def edge_mask_by_distance(
            self,
            begin_hop: int,
            end_hop: int
    ) -> torch.BoolTensor:
        """
        直接在 edge_index 上判断哪些边 (u->v) 满足：
            last_dist[u] == begin_hop 且 last_dist[v] == end_hop
        返回
            mask, source_node_idx, target_node_idx
        """
        if self.last_dist is None:
            raise RuntimeError("请先调用 compute(start_idx) 计算距离。")

        src = self.edge_index[0]
        dst = self.edge_index[1]
        head_dist = self.last_dist[src]  # [E]
        tail_dist = self.last_dist[dst]  # [E]
        if self.directed:
            mask = (head_dist == int(begin_hop)) & (tail_dist == int(end_hop))
            source_node_idx = src
            target_node_idx = dst
        else:
            mask1 = (head_dist == int(begin_hop)) & (tail_dist == int(end_hop))
            mask2 = (tail_dist == int(begin_hop)) & (head_dist == int(end_hop))
            mask = mask1 | mask2
            source_node_idx = torch.where(mask1, src,
                                          torch.where(mask2, dst, -1),)
            target_node_idx = torch.where(mask1, dst,
                                          torch.where(mask2, src, -1),)
        return mask, source_node_idx, target_node_idx

    # ---------- 方法 2：取“target_hop ∈ S”的所有实体（只返回 mask） ----------
    @torch.no_grad()
    def entity_mask_by_distance(
        self,
        target_hop: Union[int, Sequence[int], Iterable[int]]
    ) -> torch.BoolTensor:
        """
        返回与所有实体对齐的掩码，表示哪些实体满足 last_dist ∈ target_hop。
        """
        if self.last_dist is None:
            raise RuntimeError("请先调用 compute(start_idx) 计算距离。")

        dist_sel = self._normalize_distance_selector(target_hop)
        mask = torch.isin(self.last_dist, dist_sel)
        return mask

    @torch.no_grad()
    def entity_idx_by_distance(
        self,
        distance: Union[int, Sequence[int], Iterable[int]]
    ) -> torch.LongTensor:
        return torch.where(self.entity_mask_by_distance(distance))[0]

    @torch.no_grad()
    def edge_idx_by_distance(
            self,
            begin_hop: int,
            end_hop: int
    ) -> torch.BoolTensor:
        return torch.where(self.edge_mask_by_distance(begin_hop, end_hop))[0]



if __name__ == "__main__":
    # 假设有 num_nodes=6，边如下（无向）
    edge_index = torch.tensor([[0,1,2,2,3,4],
                               [1,2,3,4,4,5]], dtype=torch.long)  # [2, E]
    gd = SubgraphHopSelector(edge_index=edge_index, num_nodes=8, directed=False)

    # 从起点 {0, 5} 计算 BFS 距离
    dist = gd.compute_distances(start_idx=[0, 5])   # e.g. tensor([0,1,2,2,3,0])

    # 取“第 2 跳”的所有实体 mask
    ents_mask = gd.entity_mask_by_distance(2)  # BoolTensor[N]
    ents_idx = gd.entity_idx_by_distance(2)  # BoolTensor[N]

    # 直接在 edge_index 上找：头距离=1 且 尾距离=2 的边（只返回与 E 对齐的 mask）
    edge_mask = gd.edge_mask_by_distance(begin_hop=1, end_hop=0)  # BoolTensor[E]
    edge_idx = gd.edge_idx_by_distance(begin_hop=2, end_hop=2)  # BoolTensor[E]

    print(edge_mask)
    print(edge_idx)