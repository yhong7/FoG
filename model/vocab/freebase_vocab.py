from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Tuple, Union, Optional
import re
import aiohttp
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.async_runner import AsyncRunner
from config.settings import CONFIG

FB_NS = CONFIG["freebase"]["name_space"]
DEFAULT_SPARQL_URL = CONFIG["freebase"]["sparql_url"]

# ====== 复用：向量化工具 ======
def _is_iterable_but_not_str(x):
    try:
        from collections.abc import Iterable as _It
        return isinstance(x, _It) and not isinstance(x, (str, bytes))
    except Exception:
        return False

def _is_single_triple(x):
    return isinstance(x, (list, tuple)) and len(x) == 3 and not any(isinstance(e, (list, tuple)) for e in x)

def vectorize_scalar(fn):
    async def wrapper(self, x, *args, **kwargs):
        if _is_iterable_but_not_str(x):
            return [await fn(self, xi, *args, **kwargs) for xi in x]
        return await fn(self, x, *args, **kwargs)
    return wrapper

def vectorize_triple(fn, max_concurrency=20):
    async def wrapper(self, x, *args, **kwargs):
        if _is_single_triple(x):
            return await fn(self, x, *args, **kwargs)
        if _is_iterable_but_not_str(x):
            xs = list(x)
            runner = AsyncRunner(fn, unpack=True, max_concurrency=max_concurrency, max_retries = 3, **kwargs)
            tasks = [(self, tri, *args) for tri in xs]
            res = await runner.run(tasks)
            return res
            # return [await fn(self, tri, *args, **kwargs) for tri in xs]
        raise TypeError(f"vectorize_triple 期望 (m.xxx, rel, m.xxx) 或其序列，收到类型：{type(x)}")
    return wrapper

# ====== Freebase 基础判别 ======
MID_RE = re.compile(r"^(?:m|g)\.[A-Za-z0-9_]+$")
PRED_RE  = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+$")  # people.person.sibling_s

def is_mid(x: str) -> bool: return bool(MID_RE.match(x))
def is_pred(x: str) -> bool: return bool(PRED_RE.match(x))

def fb_curie(local: str) -> str:
    """把 m.06w2sn5 或 people.person.sibling_s 转为 fb: 前缀 CURIE。"""
    return f"fb:{local}"

# ============================== 持久化（新增，仅缓存相关） ==============================
# - 不改动任何原有注释/业务逻辑
# - 实例化时自动读取缓存；新增缓存达到阈值自动保存
import os, json, time

def _now_ts() -> float:
    return time.time()

# ===========================================================================

@dataclass
class FreebaseVocab:
    """
    Freebase 轻量版词表/解码器（基于 RDF/SPARQL）。
    - 实体：m.xxx
    - 关系：people.person.sibling_s 这类点分形式
    - 三元组：(mid, predicate, mid)
    """
    endpoint: str = DEFAULT_SPARQL_URL
    user_agent: str = "freebase-vocab/1.0"
    timeout: float = 1000
    decode_type: Literal["ori", "label", "label_with_description"] = "label"
    langs: str = "en"   # 语言优先级

    # ---------- 持久化参数/状态（新增） ----------
    autosave_freq: int = 1000                                 # 每新增多少条缓存触发一次保存
    autosave_path: Optional[str] = "files/freebase_vocab.json"  # 保存路径；None 关闭持久化
    _autosave_counter: int = field(default=0, init=False, repr=False)
    _meta_cache: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)  # id -> meta
    _last_loaded_path: Optional[str] = field(default=None, init=False, repr=False)
    _last_saved_ts: float = field(default=0.0, init=False, repr=False)
    _name2id: Dict[str, str] = field(default_factory=dict, init=False, repr=False)  # name -> id

    # ---------- HTTP/SPARQL ----------

    def _norm(self, s: str) -> str:
        return (s or "").strip().lower()

    async def sparql(self, query: str) -> Dict:
        headers = {"User-Agent": self.user_agent}
        params = {"query": query, "format": "json"}
        async with aiohttp.ClientSession() as session:
            async with session.get(self.endpoint, headers=headers, params=params, timeout=self.timeout) as r:
                r.raise_for_status()
                return await r.json(content_type=None)

    async def sparql_with_triplet_res(self, query: str) -> List[List[str]]:
        FB = "http://rdf.freebase.com/ns/"
        # IRI三元组；字面量行会被忽略
        iri_pat = re.compile(r'^\s*<([^>]*)>\s+<([^>]*)>\s+<([^>]*)>\s*\.\s*$')
        headers = {"User-Agent": self.user_agent, "Accept": "application/n-triples"}
        params = {"query": query}
        triples: List[List[str]] = []
        async with aiohttp.ClientSession() as session:
            async with session.get(self.endpoint, headers=headers, params=params, timeout=self.timeout) as r:
                r.raise_for_status()
                for line in (await r.text()).splitlines():
                    m = iri_pat.match(line)
                    if not m:
                        continue
                    s, p, o = m.groups()
                    if s.startswith(FB) and p.startswith(FB) and o.startswith(FB):
                        triples.append([s[len(FB):], p[len(FB):], o[len(FB):]])
        return triples

    def _ingest_meta_to_name2id(self, _id: str, meta: Dict[str, Any]) -> None:
        """从一条已缓存的 meta（含 label/aliases）灌入 name→id 映射。"""
        if not isinstance(meta, dict):
            return
        label = self._norm(meta.get("label", ""))
        if label:
            self._name2id[label] = _id
        for a in meta.get("aliases", []) or []:
            na = self._norm(a)
            if na:
                self._name2id[na] = _id

    # ---------- 语言优先序列 ----------
    def _lang_chain(self) -> List[str]:
        # "zh-hans,zh,en" -> ["zh-hans", "zh", "en"]
        xs = [self._norm(x) for x in self.langs.split(",") if x.strip()]
        # 同时加入首段的“语言基”如 zh-hans -> zh
        extra = []
        for x in xs:
            base = x.split("-")[0]
            if base not in xs and base not in extra:
                extra.append(base)
        return xs + extra

    # ---------- 统一标签提取 ----------
    async def _fetch_label_desc_alias(self, local_id: str) -> Dict[str, Any]:
        """
        对实体或属性（local_id 为 m.xxx 或 people.person.prop）取 name/desc/alias。
        Freebase RDF 常见三类文字属性：
          - rdfs:label
          - fb:type.object.name
          - fb:common.topic.description
          - fb:common.topic.alias
        """
        # 命中持久化缓存则直接返回（新增）
        cached = self._meta_cache.get(local_id)
        if cached:
            return cached

        q = f"""
        PREFIX fb:   <http://rdf.freebase.com/ns/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

        SELECT ?name ?desc ?alias
        WHERE {{
          OPTIONAL {{
            {fb_curie(local_id)} rdfs:label ?name .
          }}
          OPTIONAL {{
            {fb_curie(local_id)} fb:type.object.name ?name .
          }}
          OPTIONAL {{
            {fb_curie(local_id)} fb:common.topic.description ?desc .
          }}
          OPTIONAL {{
            {fb_curie(local_id)} fb:common.topic.alias ?alias .
          }}
        }}
        """
        data = await self.sparql(q)
        rows = data.get("results", {}).get("bindings", [])

        # 收集所有语言变体
        names   = []
        descs   = []
        aliases = []
        for b in rows:
            if "name" in b and "value" in b["name"]:
                names.append((b["name"]["value"], b["name"].get("xml:lang", "").lower()))
            if "desc" in b and "value" in b["desc"]:
                descs.append((b["desc"]["value"], b["desc"].get("xml:lang", "").lower()))
            if "alias" in b and "value" in b["alias"]:
                aliases.append((b["alias"]["value"], b["alias"].get("xml:lang", "").lower()))

        # 按语言优先级挑选最佳 name/desc，并收集 alias（同语种去重）
        def _pick_best(cands: List[Tuple[str, str]]) -> str:
            if not cands: return ""
            pref = self._lang_chain()
            for lg in pref:
                for v, lang in cands:
                    # lang 可能为 "en" / "zh" / "zh-hans" 等
                    if lang == lg or (lg and lang.startswith(lg.split("-")[0])):
                        return v
            return cands[0][0]  # 退回第一条

        best_name = _pick_best(names)
        best_desc = _pick_best(descs)
        # alias：保留与 best_name 不同的、按语言优先做去重
        alias_texts = []
        seen = set()
        for v, lg in aliases:
            key = (v, lg)
            if v and v != best_name and key not in seen:
                seen.add(key)
                alias_texts.append(v)

        meta = {
            "id": local_id,
            "label": best_name,
            "description": best_desc,
            "aliases": sorted(list({a for a in alias_texts if a}))
        }

        # 写入持久化缓存并计数（新增）
        if local_id not in self._meta_cache:
            self._meta_cache[local_id] = meta
            self._bump_autosave(1)

        return meta

    # ---------- encode：不做在线搜索，仅做格式校验 ----------
    @vectorize_scalar
    async def encode_entity(self, token: str) -> str:
        if not token:
            logger.warning("encode_entity: 空实体")
            return token
        if is_mid(token):
            return token

        key = self._norm(token)
        ent_id = self._name2id.get(key)
        if ent_id:
            return ent_id

        logger.warning(f"encode_entity: 未找到自然语言实体映射：{token}")
        return token

    # ---------- decode ----------
    @vectorize_scalar
    async def decode_entity(self, mid: str) -> Any:
        if not is_mid(mid):
            logger.warning(f"期望 m.xxx/g.xxx，收到：{mid}")
            return mid
        meta = await self._fetch_label_desc_alias(mid)
        if self.decode_type == "label":
            ret = meta["label"] or meta["id"]
        elif self.decode_type == "label_with_description":
            if meta["label"] and meta["description"]:
                ret = f'{meta["label"]}: {meta["description"]}'
            else:
                ret = meta["label"] or meta["id"]
        elif self.decode_type == "ori":
            ret = meta
        else:
            raise NotImplementedError
        if isinstance(ret, str):
            self._name2id[self._norm(ret)] = mid
        return ret

    # ---------- 三元组 ----------
    @vectorize_scalar
    async def encode_entity(self, entity: str) -> str:
        if not entity:
            logger.warning("encode_entity: 空实体")
            return entity
        if is_mid(entity):
            return entity

        key = self._norm(entity)
        ent_id = self._name2id.get(key)
        if ent_id:
            return ent_id

        logger.warning(f"encode_entity: 未找到自然语言实体映射：{entity}")
        return entity

    @vectorize_triple
    async def decode_triple(self, tri_ids: Tuple[str, str, str]) -> List[Any]:
        h, r, t = tri_ids
        return [await self.decode_entity(h), r, await self.decode_entity(t)]

    # async def get_triples_by_ends(
    #     self,
    #     ends: List[str],
    #     excludes: Optional[List[str]] = None,
    # ) -> List[List[str]]:
    #     """
    #     提取所有以 ends 中任一实体为 头或尾 的三元组，且三元组中不包含 excludes 里的任一实体。
    #     - ends:    ["m.1234", "m.abcd" ...]
    #     - excludes:["m.5678", ...]  （可为空或 None）
    #
    #     返回：List[[head_mid, predicate, tail_mid], ...]，均为 Freebase 本地 id（不含前缀）
    #     """
    #
    #     # ---- 参数清洗与校验 ----
    #     ends = [e.strip() for e in (ends or []) if e and e.strip()]
    #     if not ends:
    #         return []
    #     excludes = [e.strip() for e in (excludes or []) if e and e.strip()]
    #
    #     # 仅接受合法的 Freebase MID
    #     valid_ends = [e for e in ends if is_mid(e)]
    #     if not valid_ends:
    #         raise KeyError(f"ends 需为 Freebase MID（m.xxx），收到：{ends}")
    #     valid_excludes = [e for e in excludes if is_mid(e)]
    #
    #     # 组装 SPARQL 片段
    #     def _vals(xs: List[str]) -> str:
    #         # 转为 fb:CURIE 形式，例如 fb:m.1234
    #         return " ".join(f"{fb_curie(x)}" for x in xs)
    #
    #     ends_vals = _vals(valid_ends)
    #     excludes_vals = _vals(valid_excludes) if valid_excludes else ""
    #
    #     # excludes 过滤子句（为空则不加）
    #     excludes_filter = ""
    #     if excludes_vals:
    #         # 只限制头尾（?s 与 ?o），谓词不需要排除
    #         excludes_filter = f"""
    #           FILTER( !(?s IN ({excludes_vals})) && !(?o IN ({excludes_vals})) )
    #         """
    #
    #     # 用 CONSTRUCT 生成 N-Triples，便于复用 sparql_with_triplet_res
    #     # 只取 FB 命名空间下的三元组，避免字面量等（sparql_with_triplet_res 也会再次过滤）
    #     query = f"""
    #     PREFIX fb: <http://rdf.freebase.com/ns/>
    #
    #     CONSTRUCT {{
    #       ?s ?p ?o .
    #     }}
    #     WHERE {{
    #       VALUES ?seed {{ {ends_vals} }}
    #       ?s ?p ?o .
    #       FILTER( (?s = ?seed) || (?o = ?seed) )
    #       FILTER( STRSTARTS(STR(?s), "http://rdf.freebase.com/ns/")
    #            && STRSTARTS(STR(?p), "http://rdf.freebase.com/ns/")
    #            && STRSTARTS(STR(?o), "http://rdf.freebase.com/ns/") )
    #       {excludes_filter}
    #     }}
    #     """
    #
    #     return await self.sparql_with_triplet_res(query)

    # ============================== 持久化方法（新增） ==============================
    def _bump_autosave(self, newly_added: int = 0) -> None:
        """达到阈值则触发保存（仅持久化相关，不改业务逻辑）。"""
        if not self.autosave_path or newly_added <= 0 or self.autosave_freq <= 0:
            return
        self._autosave_counter += newly_added
        if self._autosave_counter >= self.autosave_freq:
            try:
                self.to_json(self.autosave_path)
                logger.info(f"[autosave] saved to {self.autosave_path} (+{self._autosave_counter})")
            except Exception as e:
                logger.warning(f"[autosave] failed: {e}")
            finally:
                self._autosave_counter = 0

    def to_json(self, path: str) -> None:
        obj = {
            "langs": self.langs,
            "decode_type": self.decode_type,
            "meta_cache": self._meta_cache,
            "saved_at": _now_ts(),
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        self._last_saved_ts = _now_ts()

    @classmethod
    def from_json(cls, path: str) -> "FreebaseVocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = cls(
            langs=obj.get("langs", "en"),
            decode_type=obj.get("decode_type", "label"),
        )
        v._meta_cache = obj.get("meta_cache", {}) or {}
        v._last_loaded_path = path

        # 基于持久化内容构建 name→id
        v._name2id.clear()
        for _id, meta in (v._meta_cache or {}).items():
            v._ingest_meta_to_name2id(_id, meta)
        return v

    def __post_init__(self):
        """实例化后自动读取缓存（若 autosave_path 存在）。"""
        if self.autosave_path and os.path.exists(self.autosave_path):
            try:
                with open(self.autosave_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, dict):
                    cache = obj.get("meta_cache", {})
                    if isinstance(cache, dict):
                        self._meta_cache.update(cache)
                        self._last_loaded_path = self.autosave_path
                        logger.info(f"[autosave] loaded {len(cache)} entries from {self.autosave_path}")
            except Exception as e:
                logger.warning(f"[autosave] load failed: {e}")

        # 基于 _meta_cache 做一次全量反向更新
        self._name2id.clear()
        for _id, meta in (self._meta_cache or {}).items():
            self._ingest_meta_to_name2id(_id, meta)

# ====== 示例 ======
if __name__ == "__main__":
    import asyncio

    async def demo():
        fb = FreebaseVocab(
            endpoint=DEFAULT_SPARQL_URL,
            decode_type="label",
            langs="en"
        )
        mid_h = "m.06w2sn5"
        mid_t = "m.0gxnnwp"
        pred  = "people.person.sibling_s"

        # # decode（从本地 Virtuoso 取名与描述）
        print("decode_entity(h):", await fb.decode_entity(mid_h))
        print("encode_entity: ", await fb.encode_entity("Justin Bieber"))
        #
        # 三元组
        tri = (mid_h, pred, mid_t)
        print("decode_triple:", await fb.decode_triple(tri))

        print("="*100)
        # print(await fb.get_triples_by_ends(["m.0ny57", "m.02qswpp"]))

    asyncio.run(demo())
