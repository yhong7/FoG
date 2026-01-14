from model.schemas import KGDataset
from config.singleton import DEFAULT_EMBEDDER
from typing import List, Tuple, Dict, Optional, Union
from collections import defaultdict, deque
import torch
import aiohttp
import asyncio
from typing import List
from config.settings import CONFIG

FB_NS = CONFIG["freebase"]["name_space"]
DEFAULT_SPARQL_URL = CONFIG["freebase"]["sparql_url"]


def bfs_by_threshold(
    kg: KGDataset,
    *,
    max_hops: int,
    expand_threshold: float,
    quality_threshold: float,
    directed: bool = False,
) -> dict:
    """
    仅使用预计算的 kg.qtr_score[E] 做筛选与 BFS 扩展；返回 dict 而非 dataclass。
    返回结构：
    {
        "feasible": {
            "edge_indices": List[int],
            "entity_ids": List[int],
            "edge_masks": torch.BoolTensor,   # 形状 [E]
            "entity_masks": torch.BoolTensor, # 形状 [num_entities]
        },
        "quality": { ... 同上 ... },
        "edge_scores": List[float],          # 来自 kg.qtr_score
    }
    """
    edge_index: torch.Tensor = kg.edge_index          # [2, E]
    edge_type: Optional[torch.Tensor] = kg.edge_type
    num_entities: int = kg.num_entities
    vocab = kg.vocab
    start_entities_raw: Union[str, List[str]] = kg.source_entity
    qtr_score: torch.Tensor = kg.qtr_score            # [E], float
    E = int(edge_index.size(1))

    # --- 起点标准化 ---
    if isinstance(start_entities_raw, str):
        start_entities = [start_entities_raw]
    else:
        start_entities = [s for s in (start_entities_raw or []) if isinstance(s, str) and s.strip()]

    def _empty_result(entity_ids: List[int]) -> dict:
        edge_mask = torch.zeros(E, dtype=torch.bool, device=edge_index.device)
        ent_mask = torch.zeros(num_entities, dtype=torch.bool, device=edge_index.device)
        if entity_ids:
            ent_mask[torch.as_tensor(sorted(set(entity_ids)), dtype=torch.long, device=edge_index.device)] = True
        return {
            "feasible": {
                "edge_indices": [],
                "entity_ids": sorted(set(entity_ids)),
                "edge_masks": edge_mask.clone(),
                "entity_masks": ent_mask.clone(),
            },
            "quality": {
                "edge_indices": [],
                "entity_ids": sorted(set(entity_ids)),
                "edge_masks": edge_mask.clone(),
                "entity_masks": ent_mask.clone(),
            },
            "edge_scores": qtr_score.detach().to("cpu").tolist() if isinstance(qtr_score, torch.Tensor) else list(qtr_score),
        }

    if not start_entities:
        return _empty_result(entity_ids=[])

    # --- 起点编码 ---
    start_ids_all = vocab.encode_entity(start_entities)
    start_ids = [int(eid) for eid in start_ids_all if isinstance(eid, (int,))]
    if not start_ids:
        return _empty_result(entity_ids=[])

    # --- 建正向邻接 ---
    full_adj: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for eidx in range(E):
        sid = int(edge_index[0, eidx]); oid = int(edge_index[1, eidx])
        rid = int(edge_type[eidx]) if edge_type is not None else -1
        full_adj[sid].append((eidx, rid, oid))

    # --- 无向所需反向邻接 ---
    rev_adj: Optional[Dict[int, List[Tuple[int, int, int]]]] = None
    if not directed:
        rev_adj = defaultdict(list)
        for sid, lst in full_adj.items():
            for (eidx, rid, oid) in lst:
                rev_adj[oid].append((eidx, rid, sid))

    def _neighbors(u: int):
        for it in full_adj.get(u, []):
            yield it
        if (not directed) and (rev_adj is not None):
            for it in rev_adj.get(u, []):
                yield it

    # --- BFS：用 expand_threshold 控制“新节点扩展” ---
    dist: Dict[int, int] = {}
    q = deque()
    for s_id in start_ids:
        dist[s_id] = 0
        q.append(s_id)

    hop = 0
    while q and hop < max_hops:
        layer = list(q); q.clear()
        # 收集候选边（只针对尚未访问的目标节点）
        cand_pairs: List[Tuple[int, int, int]] = []  # (u, eidx, v)
        cand_eidx: List[int] = []
        for u in layer:
            d = dist[u]
            if d == max_hops:
                continue
            for eidx, rid, v in _neighbors(u):
                if v in dist:
                    continue
                cand_pairs.append((u, eidx, v))
                cand_eidx.append(eidx)

        if not cand_eidx:
            break

        # 边分数直接来自 qtr_score；为避免重复索引，先去重再查分
        cand_eidx = list(dict.fromkeys(cand_eidx))
        pmap = {e: float(qtr_score[e]) for e in cand_eidx}

        for u, eidx, v in cand_pairs:
            if pmap.get(eidx, 0.0) >= float(expand_threshold):
                dist[v] = dist[u] + 1
                q.append(v)

        hop += 1

    nodes_in = set(dist.keys())
    if not nodes_in:
        return _empty_result(entity_ids=start_ids)

    # --- k-hop 诱导子图边全集 ---
    induced_edges: List[int] = []
    for sid, lst in full_adj.items():
        if sid not in nodes_in:
            continue
        for (eidx, rid, oid) in lst:
            if oid in nodes_in:
                induced_edges.append(eidx)
    induced_edges = list(dict.fromkeys(induced_edges))

    only_entities = sorted(nodes_in)
    if not induced_edges:
        return _empty_result(entity_ids=only_entities)

    # --- 两阈值切分 ---
    feasible_edges = [e for e in induced_edges if float(qtr_score[e]) >= float(expand_threshold)]
    quality_edges  = [e for e in induced_edges if float(qtr_score[e]) >= float(quality_threshold)]

    # --- 由边反推实体 + 掩码 ---
    def _entities_from_edges(edges: List[int]) -> List[int]:
        # if not edges:
        #     return only_entities
        ent = set()
        for eidx in edges:
            sid = int(edge_index[0, eidx])
            oid = int(edge_index[1, eidx])
            ent.add(sid); ent.add(oid)
        return sorted(ent)

    feasible_ents = _entities_from_edges(feasible_edges)
    quality_ents  = _entities_from_edges(quality_edges)

    device = edge_index.device
    feasible_edge_masks = torch.zeros(E, dtype=torch.bool, device=device)
    if feasible_edges:
        feasible_edge_masks[torch.as_tensor(feasible_edges, dtype=torch.long, device=device)] = True
    feasible_entity_masks = torch.zeros(num_entities, dtype=torch.bool, device=device)
    if feasible_ents:
        feasible_entity_masks[torch.as_tensor(feasible_ents, dtype=torch.long, device=device)] = True

    quality_edge_masks = torch.zeros(E, dtype=torch.bool, device=device)
    if quality_edges:
        quality_edge_masks[torch.as_tensor(quality_edges, dtype=torch.long, device=device)] = True
    quality_entity_masks = torch.zeros(num_entities, dtype=torch.bool, device=device)
    if quality_ents:
        quality_entity_masks[torch.as_tensor(quality_ents, dtype=torch.long, device=device)] = True

    return {
        "feasible": {
            "edge_indices": feasible_edges,
            "entity_ids": feasible_ents,
            "edge_masks": feasible_edge_masks,
            "entity_masks": feasible_entity_masks,
        },
        "quality": {
            "edge_indices": quality_edges,
            "entity_ids": quality_ents,
            "edge_masks": quality_edge_masks,
            "entity_masks": quality_entity_masks,
        },
    }

async def update_qtr_emb(kg: KGDataset, inference_mode: bool = False):
    from config.singleton import QTR_BODY, SCORER

    qtr_body = QTR_BODY
    scorer = SCORER

    q_emb = await DEFAULT_EMBEDDER.embed_with_tensor(kg.query)
    kg.query_emb = q_emb

    entity_list = kg.vocab.entity_list
    relation_list = kg.vocab.relation_list

    # 对kg更新x
    device = kg.device

    # if not isinstance(kg.x, torch.Tensor) or kg.x.size(0) != N or kg.x.size(1) != D:
    #     kg.x = torch.full((N, D), float('nan'), device=device, dtype=torch.float32)

    # 仅收集整行全 NaN 的实体，批量 embed 一次
    ent_need_mask = torch.isnan(kg.x).all(dim=1)
    if ent_need_mask.any():
        ent_idx = ent_need_mask.nonzero(as_tuple=False).squeeze(-1)
        ent_idx_list = ent_idx.tolist()
        B_ENT = 1024  # 实体 embedding 的 batch size，可根据显存调整

        for start in range(0, len(ent_idx_list), B_ENT):
            end = start + B_ENT
            sub_idx = ent_idx_list[start:end]
            ent_names = [entity_list[i] for i in sub_idx]
            ent_emb = await DEFAULT_EMBEDDER.embed_with_tensor(ent_names)  # 期望 [K, 1024]
            # if not isinstance(ent_emb, torch.Tensor):
            #     ent_emb = torch.tensor(ent_emb, dtype=torch.float32)
            ent_emb = ent_emb.to(device=device, dtype=torch.float32)
            kg.x[ent_idx[start:end]] = ent_emb

    # 对kg更新edge_attr
    R = kg.vocab.num_relations

    # 仅收集整行全 NaN 的关系，批量 embed 一次
    rel_need_mask = torch.isnan(kg.edge_attr).all(dim=1)
    if rel_need_mask.any():
        rel_idx = rel_need_mask.nonzero(as_tuple=False).squeeze(-1)
        rel_idx_list = rel_idx.tolist()
        B_REL = 1024  # 关系 embedding 的 batch size

        for start in range(0, len(rel_idx_list), B_REL):
            end = start + B_REL
            sub_idx = rel_idx_list[start:end]
            rel_names = [relation_list[i] for i in sub_idx]
            rel_emb = await DEFAULT_EMBEDDER.embed_with_tensor(rel_names)  # 期望 [K, 1024]
            # if not isinstance(rel_emb, torch.Tensor):
            #     rel_emb = torch.tensor(rel_emb, dtype=torch.float32)
            rel_emb = rel_emb.to(device=device, dtype=torch.float32)
            kg.edge_attr[rel_idx[start:end]] = rel_emb

    # 根据x、edge_attr和edge_index、edge_type，计算qtr_emb，并更新kg
    heads = kg.edge_index[0].to(device)
    tails = kg.edge_index[1].to(device)
    rels  = kg.edge_type.to(device)
    E = heads.numel()

    # 决定哪些边需要计算 qtr_emb（整行全 NaN）
    compute_all = True
    idx_to_compute = None

    # 已有 [E, h]，只算 NaN 行
    mask_nan_edges = torch.isnan(kg.qtr_emb).all(dim=1)
    if mask_nan_edges.any():
        idx_to_compute = mask_nan_edges.nonzero(as_tuple=False).squeeze(-1)
        compute_all = False
    else:
        # 没有需要补的，直接返回
        return kg.qtr_emb

    # edge 计算也按 batch 分块
    idx_list = idx_to_compute.tolist()
    B_EDGE = 4096  # 可以根据显存大小调节这个 batch size

    if inference_mode:
        # 纯推理场景：不开梯度、节省显存
        with torch.inference_mode():
            for start in range(0, len(idx_list), B_EDGE):
                end = start + B_EDGE
                sub_idx_tensor = idx_to_compute[start:end]

                # 仅为需要的边构建 T_emb 并计算
                h_sel = heads.index_select(0, sub_idx_tensor)
                t_sel = tails.index_select(0, sub_idx_tensor)
                r_sel = rels.index_select(0, sub_idx_tensor)

                x_h = kg.x.index_select(0, h_sel)          # [K, 1024]
                x_t = kg.x.index_select(0, t_sel)          # [K, 1024]
                r_e = kg.edge_attr.index_select(0, r_sel)  # [K, 1024]
                T_emb = torch.stack([x_h, r_e, x_t], dim=1)  # [K, 3, 1024]

                q_emb_batch = q_emb.expand(T_emb.shape[0], -1)
                qtr_part = qtr_body(q=q_emb_batch, T=T_emb)  # [K, h]
                qtr_emb_new = qtr_part['features']
                qtr_logit = scorer(qtr_emb_new, torch.zeros_like(qtr_emb_new))
                qtr_score = torch.sigmoid(qtr_logit)

                # 写回：如果已有 [E, h]，仅填 NaN 行；否则创建新张量并仅填需要的行
                kg.qtr_emb[sub_idx_tensor] = qtr_emb_new
                for i, ori_idx in enumerate(sub_idx_tensor):
                    kg.qtr_score[ori_idx] = qtr_score[i][0]

                # 可选：显式删除临时变量，帮助释放 Python 引用
                del h_sel, t_sel, r_sel, x_h, x_t, r_e, T_emb, q_emb_batch, qtr_part, qtr_emb_new, qtr_logit, qtr_score

                # 这句不是必须的，但有时能缓解碎片导致的 OOM
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    else:
        # 需要梯度或你不想关 inference_mode 的情况
        for start in range(0, len(idx_list), B_EDGE):
            end = start + B_EDGE
            sub_idx_tensor = idx_to_compute[start:end]

            # 仅为需要的边构建 T_emb 并计算
            h_sel = heads.index_select(0, sub_idx_tensor)
            t_sel = tails.index_select(0, sub_idx_tensor)
            r_sel = rels.index_select(0, sub_idx_tensor)

            x_h = kg.x.index_select(0, h_sel)          # [K, 1024]
            x_t = kg.x.index_select(0, t_sel)          # [K, 1024]
            r_e = kg.edge_attr.index_select(0, r_sel)  # [K, 1024]
            T_emb = torch.stack([x_h, r_e, x_t], dim=1)  # [K, 3, 1024]

            q_emb_batch = q_emb.expand(T_emb.shape[0], -1)
            qtr_part = qtr_body(q=q_emb_batch, T=T_emb)  # [K, h]
            qtr_emb_new = qtr_part['features']
            qtr_logit = scorer(qtr_emb_new, torch.zeros_like(qtr_emb_new))
            qtr_score = torch.sigmoid(qtr_logit)

            # 写回：如果已有 [E, h]，仅填 NaN 行；否则创建新张量并仅填需要的行
            kg.qtr_emb[sub_idx_tensor] = qtr_emb_new
            for i, ori_idx in enumerate(sub_idx_tensor):
                kg.qtr_score[ori_idx] = qtr_score[i][0]

            # 可选：显式删除临时变量，帮助释放 Python 引用
            del h_sel, t_sel, r_sel, x_h, x_t, r_e, T_emb, q_emb_batch, qtr_part, qtr_emb_new, qtr_logit, qtr_score

            if device.type == "cuda":
                torch.cuda.empty_cache()

    return kg.qtr_emb

async def fetch_one_hop_triplets(
    entity_id: str,
    endpoint: str = DEFAULT_SPARQL_URL,
    timeout: int = 60,
) -> List[List[str]]:
    """
    给定一个 Freebase 实例 id（如 'm.0160w'），返回其一跳范围内的所有三元组（包含入边和出边），
    返回格式为 List[List[str]]，内部元素为 [s, p, o]，均为去掉 NS 前缀的短 id。
    """
    NS = FB_NS
    P_BLACKLIST = {NS + "type.object.type", NS + "type.type.instance"}

    # ---- 工具逻辑完全内联，不调用外部函数 ----

    # 1. 统一成 IRI
    if entity_id.startswith("http"):
        center_iri = entity_id
    else:
        center_iri = NS + entity_id

    # 2. 构造 SPARQL（出边）
    out_q = f"""PREFIX ns:<{NS}>
SELECT ?s ?p ?o WHERE {{
  <{center_iri}> ?p ?o .
  BIND(<{center_iri}> AS ?s)
  FILTER(isIRI(?o))
  FILTER( STRSTARTS(STR(?o), "{NS}m.") || STRSTARTS(STR(?o), "{NS}en.") )
  FILTER(STRSTARTS(STR(?p), "{NS}"))
  FILTER(?p != <{NS}type.object.type> && ?p != <{NS}type.type.instance>)
}}"""

    # 3. 构造 SPARQL（入边）
    in_q = f"""PREFIX ns:<{NS}>
SELECT ?s ?p ?o WHERE {{
  ?s ?p <{center_iri}> .
  BIND(<{center_iri}> AS ?o)
  FILTER(isIRI(?s))
  FILTER( STRSTARTS(STR(?s), "{NS}m.") || STRSTARTS(STR(?s), "{NS}en.") )
  FILTER(STRSTARTS(STR(?p), "{NS}"))
  FILTER(?p != <{NS}type.object.type> && ?p != <{NS}type.type.instance>)
}}"""

    headers = {"Accept": "application/sparql-results+json"}
    tries = 2
    backoff = 1.2

    async with aiohttp.ClientSession() as session:
        # --- 请求 out_q ---
        out_rows = []
        for k in range(tries + 1):
            try:
                async with session.get(
                    endpoint,
                    params={"query": out_q, "format": "application/sparql-results+json"},
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    out_rows = data.get("results", {}).get("bindings", [])
                    break
            except Exception:
                if k == tries:
                    out_rows = []
                    break
                await asyncio.sleep(backoff * (k + 1))

        # --- 请求 in_q ---
        in_rows = []
        for k in range(tries + 1):
            try:
                async with session.get(
                    endpoint,
                    params={"query": in_q, "format": "application/sparql-results+json"},
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    in_rows = data.get("results", {}).get("bindings", [])
                    break
            except Exception:
                if k == tries:
                    in_rows = []
                    break
                await asyncio.sleep(backoff * (k + 1))

    # 4. 聚合 + 过滤，只保留实例-实例三元组且去重
    triples = set()

    def _is_instance_iri(u: str) -> bool:
        return u.startswith(NS + "m.") or u.startswith(NS + "en.")

    for b in out_rows:
        s = b["s"]["value"]
        p = b["p"]["value"]
        o = b["o"]["value"]
        if p in P_BLACKLIST:
            continue
        if not (_is_instance_iri(s) and _is_instance_iri(o)):
            continue
        triples.add((s, p, o))

    for b in in_rows:
        s = b["s"]["value"]
        p = b["p"]["value"]
        o = b["o"]["value"]
        if p in P_BLACKLIST:
            continue
        if not (_is_instance_iri(s) and _is_instance_iri(o)):
            continue
        triples.add((s, p, o))

    # 5. 去掉 NS 前缀，转成 List[List[str]]
    result: List[List[str]] = []
    for s, p, o in triples:
        if s.startswith(NS):
            s_short = s[len(NS):]
        else:
            s_short = s
        if p.startswith(NS):
            p_short = p[len(NS):]
        else:
            p_short = p
        if o.startswith(NS):
            o_short = o[len(NS):]
        else:
            o_short = o
        result.append([s_short, p_short, o_short])

    return result





def _update_qtr_emb_test():
    import asyncio

    # ---------- 测试数据（正常合并） ----------
    left_triplets = [
        ["Alice", "likes",    "Bob"],
        ["Bob",   "friend_of","Charlie"],
    ]

    query = "Who does Alice connect to?"
    source_entity = ["Alice"]
    answer_entity = ["Charlie"]

    # 左侧：视为“已计算过部分向量”的实例
    ds_left = KGDataset(
        device="cuda",
        triplet_text=left_triplets,
        query=query,
        source_entity=source_entity,
        answer_entity=answer_entity,
    )

    asyncio.run(update_qtr_emb(ds_left))
    print(ds_left)

    # 追加部分
    right_triplets = [
        ["Charlie", "works_at", "Acme"],  # 新增的三元组（将引入新实体、新关系）
        ["Alice",   "likes",    "Bob"],   # 重复，不应再次加入
    ]
    ds_right = KGDataset(
        device="cuda",
        triplet_text=right_triplets,
        query=query,
        source_entity=source_entity,
        answer_entity=answer_entity,
    )

    # 执行加法
    merged = ds_left + ds_right
    asyncio.run(update_qtr_emb(merged))
    print(merged)




def _bfs_by_threshold_test():
    import asyncio
    from model.message_passing.utils import update_qtr_emb
    from model.schemas import KGDataset

    # ---------- 测试数据（正常合并） ----------
    left_triplets = [
        ["Alice", "likes", "Bob"],
        ["Bob", "friend_of", "Charlie"],
    ]

    query = "Who does Alice connect to?"
    source_entity = ["Alice"]
    answer_entity = ["Charlie"]

    # 左侧：视为“已计算过部分向量”的实例
    ds_left = KGDataset(
        device="cuda",
        triplet_text=left_triplets,
        query=query,
        source_entity=source_entity,
        answer_entity=answer_entity,
    )

    asyncio.run(update_qtr_emb(ds_left))
    print(ds_left)

    # 追加部分
    right_triplets = [
        ["Charlie", "works_at", "Acme"],  # 新增的三元组（将引入新实体、新关系）
        ["Alice", "likes", "Bob"],  # 重复，不应再次加入
    ]
    ds_right = KGDataset(
        device="cuda",
        triplet_text=right_triplets,
        query=query,
        source_entity=source_entity,
        answer_entity=answer_entity,
    )

    # 执行加法
    merged = ds_left + ds_right
    asyncio.run(update_qtr_emb(merged))
    print(merged)

    res = bfs_by_threshold(kg=merged, max_hops=2, expand_threshold=0, quality_threshold=0, )
    return res

if __name__ == "__main__":
    res = _test()
    print(res)

