from typing import Iterable, List, Tuple, Dict, Set
import aiohttp
import re
from config.settings import CONFIG

FB_NS = CONFIG["freebase"]["name_space"]
DEFAULT_SPARQL_URL = CONFIG["freebase"]["sparql_url"]
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")  # 兜底：纯数字文本

def _iri_to_fb_compact(value: str) -> str:
    return value[len(FB_NS):] if value.startswith(FB_NS) else value

def _parse_binding(node: Dict) -> str:
    t = node.get("type")
    v = node.get("value", "")
    if t == "uri":
        return _iri_to_fb_compact(v)
    # 字面量：原样返回（SPARQL 已把非 en 的描述过滤掉）
    return v

def _batched(iterable: Iterable[str], n: int) -> Iterable[List[str]]:
    batch = []
    for x in iterable:
        x = x.strip()
        if x.startswith(FB_NS):
            x = _iri_to_fb_compact(x)
        if x.startswith("/m/"):
            x = "m." + x[3:]
        batch.append(x)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch

def _values(eids: Iterable[str]) -> str:
    items = " ".join(f"fb:{eid}" for eid in eids)
    return f"VALUES ?e {{ {items} }}"

async def fetch_freebase_triples_en_no_description_async(
    endpoints: Iterable[str],
    sparql_url: str = DEFAULT_SPARQL_URL,
    batch_size: int = 50,
    limit_per_batch: int = 20000,
    timeout: int = 60,
) -> List[List[str]]:
    """
    只保留：
      - 对象为实体 m.*；或
      - 对象为字面量，且 (lang=en 的字符串) 或 (数值类型/纯数字文本)
    并排除谓词 common.topic.description 与 /key/ 命名空间。
    （异步版：逐批顺序请求，行为与同步版一致）
    """
    pref = (
        "PREFIX fb:  <http://rdf.freebase.com/ns/>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n"
    )
    out_set: Set[Tuple[str, str, str]] = set()

    timeout_cfg = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
        for chunk in _batched(endpoints, batch_size):
            q = (
                pref
                + "SELECT ?s ?p ?o WHERE {\n"
                + f"  {_values(chunk)}\n"
                + "  {\n"
                + "    ?e ?p ?o . BIND(?e AS ?s)\n"
                + "  }\n"
                + "  UNION\n"
                + "  {\n"
                + "    ?s ?p ?e . BIND(?e AS ?o)\n"
                + "  }\n"
                # 谓词约束：在 FB 命名空间；排除 /key/；排除 common.topic.description
                + '  FILTER( STRSTARTS(STR(?p), "http://rdf.freebase.com/ns/") )\n'
                + '  FILTER( !CONTAINS(STR(?p), "/key/") )\n'
                + '  FILTER( STR(?p) != "http://rdf.freebase.com/ns/common.topic.description" )\n'
                # 主语&宾语：若是 IRI，限定为 m.*
                + '  FILTER( STRSTARTS(STR(?s), "http://rdf.freebase.com/ns/m.") )\n'
                + '  FILTER( ( isIRI(?o) && STRSTARTS(STR(?o), "http://rdf.freebase.com/ns/m.") )\n'
                # 字面量分支：只收 en 或 数值
                + '       || ( isLiteral(?o) && (\n'
                + '              langMatches(lang(?o), "en")\n'
                + '           || datatype(?o) IN (xsd:integer, xsd:decimal, xsd:double, xsd:float)\n'
                + "           ) ) )\n"
                + "}\n"
                + f"LIMIT {int(limit_per_batch)}"
            )

            params = {"query": q, "format": "application/sparql-results+json"}
            async with session.get(sparql_url, params=params) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)

            for b in data.get("results", {}).get("bindings", []):
                s = _parse_binding(b["s"])
                p = _parse_binding(b["p"])
                o = _parse_binding(b["o"])

                # 兜底：如果服务端把数字当成无类型字符串返回，这里再判一次纯数字；也过滤掉形如 "/wikipedia/..." 的值
                if not o.startswith("m."):
                    o_str = o.strip()
                    if not o_str:
                        continue
                    if o_str.startswith("/"):
                        continue
                    if not _NUMERIC_RE.match(o_str):  # 非纯数字 → 只可能是 en 文本（描述已在 SPARQL排掉）
                        # 在查询里已剔除 description，所以留下来的 en 文本多半是 name/alias 等
                        # 如果你连 en 文本也不想要，直接 continue
                        continue

                out_set.add((s, p, o))

    return [list(t) for t in out_set]

async def get_add_triplets_freebase(
    endpoints: Iterable[str],
    excluded_entities: Iterable[str] = None,
    sparql_url: str = DEFAULT_SPARQL_URL,
    batch_size: int = 50,
    limit_per_batch: int = 20000,
    timeout: int = 60,
) -> List[List[str]]:
    triples = await fetch_freebase_triples_en_no_description_async(
        endpoints=endpoints,
        sparql_url=sparql_url,
        batch_size=batch_size,
        limit_per_batch=limit_per_batch,
        timeout=timeout,
    )

    ep = set()
    for e in endpoints:
        e = e.strip()
        if e.startswith(FB_NS):
            e = _iri_to_fb_compact(e)
        if e.startswith("/m/"):
            e = "m." + e[3:]
        ep.add(e)

    banned = set()
    if excluded_entities is not None:
        for b in excluded_entities:
            b = b.strip()
            if b.startswith(FB_NS):
                b = _iri_to_fb_compact(b)
            if b.startswith("/m/"):
                b = "m." + b[3:]
            banned.add(b)

    out: List[List[str]] = []
    for s, p, o in triples:
        if not (s in ep or o in ep):
            continue
        if (s in banned) or (o.startswith("m.") and o in banned):
            continue
        out.append([s, p, o])
    return out


async def _test():
    import time
    begin = time.time()
    triples = await get_add_triplets_freebase(
        endpoints=["m.02qswpp" for _ in range(100)],
        excluded_entities=["m.badentity1", "m.badentity2"],  # 可选
        sparql_url=DEFAULT_SPARQL_URL,  # 默认就是这个
    )
    print(time.time() - begin)
    return triples

if __name__ == "__main__":
    import asyncio
    res = asyncio.run(_test())
    print(len(res))
