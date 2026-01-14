import aiohttp
import random
import asyncio
from typing import List
from config.settings import CONFIG

NS = CONFIG["freebase"]["name_space"]
DEFAULT_SPARQL_URL = CONFIG["freebase"]["sparql_url"]

P_BLACKLIST = {NS + "type.object.type", NS + "type.type.instance"}

def to_uri(t): return t if t.startswith("http") else NS + t
def from_uri(u): return u[len(NS):] if u.startswith(NS) else u
def is_instance_iri(u): return u.startswith(NS + "m.") or u.startswith(NS + "en.")

def _build_outgoing_batch_q(entities, per_dir_cap):
    branches = []
    for e in entities:
        branches.append(f"""
  {{
    SELECT (<{e}> AS ?s) ?p ?o WHERE {{
      <{e}> ?p ?o .
      FILTER(isIRI(?o))
      FILTER( STRSTARTS(STR(?o), "{NS}m.") || STRSTARTS(STR(?o), "{NS}en.") )
      FILTER(STRSTARTS(STR(?p), "{NS}"))
      FILTER(?p != <{NS}type.object.type> && ?p != <{NS}type.type.instance>)
    }} ORDER BY RAND() LIMIT {per_dir_cap}
  }}""")
    return f"PREFIX ns:<{NS}>\nSELECT ?s ?p ?o WHERE {{\n" + " UNION ".join(branches) + "\n}"

def _build_incoming_batch_q(entities, per_dir_cap):
    branches = []
    for e in entities:
        branches.append(f"""
  {{
    SELECT ?s ?p (<{e}> AS ?o) WHERE {{
      ?s ?p <{e}> .
      FILTER(isIRI(?s))
      FILTER( STRSTARTS(STR(?s), "{NS}m.") || STRSTARTS(STR(?s), "{NS}en.") )
      FILTER(STRSTARTS(STR(?p), "{NS}"))
      FILTER(?p != <{NS}type.object.type> && ?p != <{NS}type.type.instance>)
    }} ORDER BY RAND() LIMIT {per_dir_cap}
  }}""")
    return f"PREFIX ns:<{NS}>\nSELECT ?s ?p ?o WHERE {{\n" + " UNION ".join(branches) + "\n}"

async def _fetch(session, endpoint, query, timeout=60, tries=2):
    params = {"query": query, "format": "application/sparql-results+json"}
    headers = {"Accept": "application/sparql-results+json"}
    backoff = 1.2
    for k in range(tries + 1):
        try:
            async with session.get(endpoint, params=params, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data.get("results", {}).get("bindings", [])
        except Exception:
            if k == tries:
                return []
            await asyncio.sleep(backoff * (k + 1))

async def expand_freebase_parallel(
    gold: List[List[str]],
    endpoint=DEFAULT_SPARQL_URL,
    hops=2,
    limit_per_hop=5,
    fetch_factor=1,
    max_triples_per_hop=None  # 新增：每一跳最多新增多少条，超出即停止并返回
):
    # gold -> 起点（仅实例）
    gold_uri = {(to_uri(s), to_uri(p), to_uri(o)) for s,p,o in gold}
    start_entities = {to_uri(s) for s,_,_ in gold if is_instance_iri(to_uri(s))} | \
                     {to_uri(o) for _,_,o in gold if is_instance_iri(to_uri(o))}

    def is_gold_equiv(s,p,o): return (s,p,o) in gold_uri or (o,p,s) in gold_uri

    expanded = set()
    visited, frontier = set(), set(start_entities)

    async with aiohttp.ClientSession() as session:
        for _ in range(hops):
            curr = [e for e in frontier if e not in visited and is_instance_iri(e)]
            if not curr:
                break
            random.shuffle(curr)         # 随机打散锚点，避免顺序偏置
            for e in curr: visited.add(e)

            per_dir_cap = max(1, limit_per_hop * fetch_factor)  # 入/出各多抓些供抽样
            out_q = _build_outgoing_batch_q(curr, per_dir_cap)
            in_q  = _build_incoming_batch_q(curr,  per_dir_cap)

            out_rows, in_rows = await asyncio.gather(
                _fetch(session, endpoint, out_q),
                _fetch(session, endpoint, in_q)
            )

            # 聚合到锚点
            by_anchor = {e: [] for e in curr}
            for b in out_rows:
                s,p,o = b["s"]["value"], b["p"]["value"], b["o"]["value"]
                if s in by_anchor: by_anchor[s].append((s,p,o))
            for b in in_rows:
                s,p,o = b["s"]["value"], b["p"]["value"], b["o"]["value"]
                if o in by_anchor: by_anchor[o].append((s,p,o))

            next_frontier = set()
            hop_added = 0  # 本跳新增计数

            for anchor in curr:
                # 候选（入+出合并）→ 预过滤
                candidates = by_anchor.get(anchor, [])
                pruned = []
                for (s,p,o) in candidates:
                    if p in P_BLACKLIST: continue
                    if is_gold_equiv(s,p,o): continue
                    if (s,p,o) in expanded or (o,p,s) in expanded: continue
                    if not (is_instance_iri(s) and is_instance_iri(o)): continue
                    pruned.append((s,p,o))

                # 随机选 ≤ limit_per_hop
                if len(pruned) > limit_per_hop:
                    chosen = random.sample(pruned, limit_per_hop)
                else:
                    random.shuffle(pruned)
                    chosen = pruned

                # 写入，同时检查“本跳总上限”
                for (s,p,o) in chosen:
                    if max_triples_per_hop is not None and hop_added >= max_triples_per_hop:
                        # 直接结束整个过程并返回当前结果
                        return [[from_uri(s1), from_uri(p1), from_uri(o1)] for s1,p1,o1 in expanded]
                    expanded.add((s,p,o))
                    hop_added += 1
                    neighbor = o if s == anchor else s
                    if is_instance_iri(neighbor):
                        next_frontier.add(neighbor)

                # 加入后再次上限判断（防止刚好超限）
                if max_triples_per_hop is not None and hop_added >= max_triples_per_hop:
                    return [[from_uri(s1), from_uri(p1), from_uri(o1)] for s1,p1,o1 in expanded]

            frontier = next_frontier

    # 返回“扩展部分”；若要 gold∪扩展，改为 (gold_uri | expanded)
    return [[from_uri(s), from_uri(p), from_uri(o)] for s,p,o in expanded]




if __name__ == '__main__':
    import asyncio
    gold = [['m.03st9j', 'location.location.containedby', 'm.0160w']]
    print(asyncio.run(expand_freebase_parallel(gold, hops=2, limit_per_hop=5)))
