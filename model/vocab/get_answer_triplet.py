import re
import time
import requests
from typing import List, Tuple, Iterable
from textwrap import dedent

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
UA = "LCQuADQuery/1.2 (yh; mailto:yhong7307@gmail.com)"

# ---------------- 核心工具函数 ----------------
def to_standard_wdqs(original_query: str) -> str:
    """
    将原始 SPARQL 查询转换为返回 Wikidata QID 的标准查询。
    自动检测目标变量（?obj、?x、?answer等）。
    """
    q = original_query.strip()

    # 提取 WHERE 块
    where_match = re.search(r"WHERE\s*\{(.*)\}\s*$", q, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        where_body = where_match.group(1).strip()
    else:
        where_body = q.strip("{} \n\t")

    # 自动检测变量名：优先取 SELECT 中第一个变量
    select_var_match = re.search(r"SELECT\s+DISTINCT\s+(\?\w+)", q, flags=re.IGNORECASE)
    if not select_var_match:
        select_var_match = re.search(r"SELECT\s+(\?\w+)", q, flags=re.IGNORECASE)

    if select_var_match:
        target_var = select_var_match.group(1)
    else:
        # 从 WHERE 里找第一个 ?变量
        where_var_match = re.search(r"(\?\w+)", where_body)
        target_var = where_var_match.group(1) if where_var_match else "?obj"

    # 统一前缀与标准选择
    prefixes = dedent("""\
        PREFIX wd:  <http://www.wikidata.org/entity/>
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX p:   <http://www.wikidata.org/prop/>
        PREFIX ps:  <http://www.wikidata.org/prop/statement/>
        PREFIX pq:  <http://www.wikidata.org/prop/qualifier/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX bd:  <http://www.bigdata.com/rdf#>
    """)

    select_part = (
        f'SELECT DISTINCT (STRAFTER(STR({target_var}), "http://www.wikidata.org/entity/") AS ?id)'
    )

    return dedent(f"""\
        {prefixes}
        {select_part}
        WHERE {{
          {where_body}
        }}
    """).strip()

def _norm_token(tok: str) -> str:
    """去掉 wdt:/wd:/ps:/pq:/p:/rdfs:/xsd: 前缀，只保留 Q/P 或原样字面量。"""
    if not tok:
        return tok
    tok = tok.strip().strip("<>")
    m = re.match(r"^https?://www\.wikidata\.org/entity/(Q\d+)$", tok)
    if m:
        return m.group(1)
    return re.sub(r"^(wdt:|wd:|ps:|pq:|p:|rdfs:|xsd:)", "", tok)

def _extract_where_body(query: str) -> str:
    """抽取 WHERE { ... } 中的内容"""
    q = query.replace("\\n", "\n").replace("\r\n", "\n")
    m = re.search(r"WHERE\s*\{(.*)\}", q, flags=re.S | re.I)
    return m.group(1) if m else ""

def _extract_triples(where_body: str) -> List[Tuple[str, str, str]]:
    """按句点分割并抽取三元组"""
    # 去注释
    lines = []
    for raw in where_body.split("\n"):
        line = re.sub(r"#.*$", "", raw).strip()
        if line:
            lines.append(line)
    body = " ".join(lines)
    stmts = re.split(r'\s*\.\s*(?=(?:[^"]*"[^"]*")*[^"]*$)', body)
    triples = []
    for s in stmts:
        if re.match(r"^(FILTER|BIND|OPTIONAL|MINUS|SERVICE|VALUES|GRAPH|UNION)\b", s, re.I):
            continue
        parts = s.strip(" }").split()
        if len(parts) >= 3:
            triples.append(tuple(parts[:3]))
    return triples

def _fetch_bindings(endpoint: str, where_body: str, var: str,
                    page_size: int, max_results: int, pause_sec: float) -> List[str]:
    """分页查询 SELECT DISTINCT ?var"""
    results, offset = [], 0
    session = requests.Session()
    headers = {"Accept": "application/sparql-results+json", "User-Agent": UA}
    key = var.lstrip("?")

    while len(results) < max_results:
        sq = f"SELECT DISTINCT {var} WHERE {{ {where_body} }} LIMIT {page_size} OFFSET {offset}"
        resp = session.get(endpoint, params={"query": sq}, headers=headers, timeout=30)
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        if not bindings:
            break
        for b in bindings:
            if key in b:
                results.append(b[key]["value"])
                if len(results) >= max_results:
                    break
        offset += page_size
        if pause_sec > 0:
            time.sleep(pause_sec)
    return results

# ---------------- 主逻辑：合并为单一子图 ----------------

def build_graph(query: str,
                endpoint: str = WIKIDATA_ENDPOINT,
                var: str = "?x0",
                page_size: int = 500,
                max_results_per_query: int = 2000,
                pause_sec: float = 0.25) -> List[Tuple[str, str, str]]:
    """
    执行多条 SPARQL 查询，将不同绑定下的三元组合并为一个子图。
    返回 list[tuple[str,str,str]]。
    """
    graph: List[Tuple[str, str, str]] = []

    # query = query
    where = _extract_where_body(query)

    triples = _extract_triples(where)
    bindings = _fetch_bindings(endpoint, where, var, page_size, max_results_per_query, pause_sec)
    bindings_short = [_norm_token(b) for b in bindings]
    for b in bindings_short:
        for s, p, o in triples:
            s2 = _norm_token(s.replace(var, b))
            p2 = _norm_token(p.replace(var, b))
            o2 = _norm_token(o.replace(var, b))
            graph.append((s2, p2, o2))
    return graph



from typing import List, Tuple, Iterable, Set, Dict


Triple = Tuple[str, str, str]  # (Q, P, Q)

def _run_sparql(query: str, retry: int = 3):
    headers = {"Accept": "application/sparql-results+json", "User-Agent": UA}
    for i in range(retry):
        r = requests.get(WIKIDATA_ENDPOINT, params={"query": query, "format": "json"},
                         headers=headers, timeout=60)
        if r.status_code == 200:
            return r.json()["results"]["bindings"]
        time.sleep(1.2 * (i + 1))
    r.raise_for_status()

def _uri_tail(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]

def _fetch_undirected_one_hop(node_q: str, raw_limit: int = 30) -> List[Triple]:
    """
    从单个节点出发抓取“无向一跳”：同时包含
      - 出边：node_q --(wdt:P)--> nbr
      - 入边：nbr --(wdt:P)--> node_q
    仅保留实体邻居 (Q*)，并返回标准三元组 (subject, Pid, object)。
    为了后续 Python 侧过滤/限额，这里先多取一些（raw_limit）。
    """
    query = f"""
    PREFIX wd:  <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wikibase: <http://wikiba.se/ontology#>
    SELECT ?s ?prop ?o WHERE {{
      {{
        # 出边：node_q -> nbr
        BIND(wd:{node_q} AS ?s)
        wd:{node_q} ?p ?o .
        FILTER(STRSTARTS(STR(?p), STR(wdt:))) .
        FILTER(STRSTARTS(STR(?o), STR(wd:))) .
        ?p ^wikibase:directClaim ?prop .
      }}
      UNION
      {{
        # 入边：nbr -> node_q
        BIND(wd:{node_q} AS ?o)
        ?s ?p wd:{node_q} .
        FILTER(STRSTARTS(STR(?p), STR(wdt:))) .
        FILTER(STRSTARTS(STR(?s), STR(wd:))) .
        ?p ^wikibase:directClaim ?prop .
      }}
    }}
    LIMIT {raw_limit}
    """
    rows = _run_sparql(query)
    triples: List[Triple] = []
    for r in rows:
        s = _uri_tail(r["s"]["value"])
        p = _uri_tail(r["prop"]["value"])
        o = _uri_tail(r["o"]["value"])
        triples.append((s, p, o))
    return triples

def generate_train_graph(
    gold_triples: Iterable[Triple],
    per_entity_per_hop_limit: int = 5,
    include_gold: bool = True,
    outward_only: bool = True,
) -> List[Triple]:
    """
    按“子图为整体”做两跳扩展（无向）：
      - hop1 前沿 = gold 子图全部节点
      - hop2 前沿 = hop1 产生的新节点的去重集合
    规则：
      - 每个“出发实体 × 每一跳”最多 per_entity_per_hop_limit 条（不含 gold 本身）
      - 仅保留实体邻居 (Q*)
      - outward_only=True 时，扩展时不回到 gold 节点（始终“向外”）
      - include_gold=True 时，把 gold 自身也并入输出
    返回：按 ('Qxxx','Pyyy','Qzzz') 的三元组列表（去重）
    """
    gold_triples = [tuple(t) for t in gold_triples]
    gold_set: Set[Triple] = set(gold_triples)
    # 子图节点（整体前沿）
    gold_nodes: Set[str] = set([s for s, _, _ in gold_triples] + [o for _, _, o in gold_triples])

    # —— hop 1：从所有 gold 节点统一出发（无向一跳），每节点限额 —— #
    hop1_triples_all: List[Triple] = []
    for q in sorted(gold_nodes):
        candidates = _fetch_undirected_one_hop(q, raw_limit=60)  # 多取些，便于过滤
        filtered = []
        for (s, p, o) in candidates:
            t = (s, p, o)
            # 过滤 gold 自身三元组
            if t in gold_set:
                continue
            # outward_only: 不要把 hop 结果仍局限在 gold 子图内部
            if outward_only and (s in gold_nodes and o in gold_nodes):
                continue
            filtered.append(t)
            if len(filtered) >= per_entity_per_hop_limit:
                break
        hop1_triples_all.extend(filtered)

    # hop1 产生的新节点集合（不属于 gold 的节点）
    hop1_new_nodes: Set[str] = set()
    for s, _, o in hop1_triples_all:
        if s not in gold_nodes:
            hop1_new_nodes.add(s)
        if o not in gold_nodes:
            hop1_new_nodes.add(o)

    # —— hop 2：从 hop1 新节点统一出发（无向一跳），每节点限额 —— #
    hop2_triples_all: List[Triple] = []
    for q in sorted(hop1_new_nodes):
        candidates = _fetch_undirected_one_hop(q, raw_limit=60)
        filtered = []
        for (s, p, o) in candidates:
            t = (s, p, o)
            # 仍旧不包含 gold 本身
            if t in gold_set:
                continue
            # 如果只向外，则避免回到 gold 节点（允许 hop2 节点之间互连）
            if outward_only and (s in gold_nodes or o in gold_nodes):
                continue
            filtered.append(t)
            if len(filtered) >= per_entity_per_hop_limit:
                break
        hop2_triples_all.extend(filtered)

    # —— 汇总去重输出 —— #
    result: Set[Triple] = set()
    if include_gold:
        result.update(gold_triples)
    result.update(hop1_triples_all)
    result.update(hop2_triples_all)

    # 统一转成 ('Q','P','Q') 的字符串三元组列表（稳定排序便于复现）
    return sorted((str(s), str(p), str(o)) for (s, p, o) in result)

def first_wikidata_entity_id(sparql: str) -> str | None:
    """
    从 SPARQL 字符串中提取第一个 Wikidata 实体（wd:Q...），
    返回不带前缀的 'Q...'，若找不到则返回 None。
    """
    # 匹配 wd:Q后跟数字，并确保是独立 token（避免匹配到更长的串）
    m = re.search(r'\bwd:Q\d+\b', sparql)
    if not m:
        return None
    return m.group(0).split(':', 1)[1]  # 去掉 'wd:' 前缀


# ---------------- 示例 ----------------
if __name__ == "__main__":
    sample_queries = [
        r'SELECT ?answer WHERE { wd:Q1356316 wdt:P156 ?X . ?X wdt:P1346 ?answer}'    ]

    sparql = r"select distinct ?obj where { wd:Q188920 wdt:P2813 ?obj . ?obj wdt:P31 wd:Q1002697 } "
    gold_graph = build_graph(sparql, page_size=100, max_results_per_query=300)
    train_graph = generate_train_graph(gold_graph)
    begin_entity = first_wikidata_entity_id(sparql)

    print(f"Total triples: {len(train_graph)}")
    print(f"Total triples: {gold_graph}")
    print(begin_entity)
    for t in train_graph:
        print(t)
