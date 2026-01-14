from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Literal, Any, Union
from collections import defaultdict
import time
import json
import re
import aiohttp
from tenacity import retry, stop_after_attempt, wait_fixed
from utils.async_runner import AsyncRunner
from loguru import logger

# ====== 复用你文件里的“向量化”风格 ======
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


def vectorize_triple(fn):
    async def wrapper(self, x, *args, **kwargs):
        if _is_single_triple(x):
            return await fn(self, x, *args, **kwargs)
        if _is_iterable_but_not_str(x):
            xs = list(x)
            return [await fn(self, tri, *args, **kwargs) for tri in xs]
        raise TypeError(f"vectorize_triple 期望 (h, r, t) 或其序列，收到类型：{type(x)}")
    return wrapper

# ====== 工具 ======
ID_RE_ENTITY = re.compile(r"^Q\d+$")
ID_RE_PROP    = re.compile(r"^P\d+$")
ID_RE_REVERSE = re.compile(r"^R(\d+)$")

def is_qid(x: str) -> bool: return bool(ID_RE_ENTITY.match(x))
def is_pid(x: str) -> bool: return bool(ID_RE_PROP.match(x))
def is_rid(x: str) -> bool: return bool(ID_RE_REVERSE.match(x))
def rid_to_pid(rid: str) -> str:
    m = ID_RE_REVERSE.match(rid)
    if not m: raise ValueError(f"不是 Rxxx：{rid}")
    return "P" + m.group(1)

def chunk(lst, n=50):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@dataclass
class WikidataVocab:
    """
    与 KnowledgeGraphVocab 的方法名/向量化风格保持一致：
      - encode_entity / decode_entity
      - encode_relation / decode_relation
      - encode_triple / decode_triple
    其中
      - encode_*  ：label -> Wikidata ID（若已是 ID 则直返）
      - decode_*  ：ID -> {id,label,description,aliases}
    """
    langs: str = "en"                  # 例如 "zh-hans,zh,en"
    wdqs:  str = "https://query.wikidata.org/sparql"
    user_agent: str = "wd-vocab/1.0"
    decode_type: Literal["ori", "label", "label_with_description"] = "label_with_description"  # 返回的字符串格式为 “label: desc”
    batch_size: int = 50
    max_concurrency: int = 5
    max_retries: int = 4
    backoff: float = 1.5
    timeout: float = 10

    # —— 自动持久化超参数/状态 ——
    autosave_freq: int = 200                 # 超参数：每新增多少条缓存触发持久化
    autosave_path: Optional[str] = "files/wikidata_vocab.json"        # 保存路径；为 None 则不启用自动持久化
    _autosave_counter: int = field(default=0, init=False, repr=False)  # 计数器

    # 缓存
    _entity_meta: Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)  # Qid -> meta
    _prop_meta:   Dict[str, Dict[str, Any]] = field(default_factory=dict, init=False, repr=False)  # Pid -> meta
    _label2qid:   Dict[str, str] = field(default_factory=dict, init=False, repr=False)  # 归一 label -> Qid
    _label2pid:   Dict[str, str] = field(default_factory=dict, init=False, repr=False)  # 归一 label -> Pid

    # ---------- 规范化 ----------
    def _norm_label(self, x: str) -> str:
        return x.strip().casefold()

    # ---------- HTTP/SPARQL ----------
    async def sparql(self, query: str) -> Dict:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    self.wdqs,
                    params={"query": query, "format": "json"},
                    headers={"User-Agent": self.user_agent},
                    timeout=self.timeout,
            ) as r:
                r.raise_for_status()
                return await r.json(content_type=None)

    @retry(
        stop=stop_after_attempt(3),  # 最多重试3次
        wait=wait_fixed(2),  # 每次重试之间等待2秒
    )
    async def _fetch_meta_ids(self, qids: Iterable[str] | None = None,
                        pids: Iterable[str] | None = None) -> None:
        """
        接受任意可迭代的 qids/pids/rids；pids 可包含 Rxxx，会统一映射到底层 Pxxx。
        自动去重并跳过已在缓存中的 ID。
        """
        # --- 归一化实体 ---
        q_raw = [str(q) for q in (qids or []) if q]
        q_norm: List[str] = []
        for q in q_raw:
            if is_qid(q) and q not in q_norm:
                q_norm.append(q)

        # --- 归一化属性（含 Rxxx→Pxxx）---
        p_raw = [str(p) for p in (pids or []) if p]
        p_norm: List[str] = []
        for p in p_raw:
            if is_pid(p):
                pid = p
            elif is_rid(p):
                pid = rid_to_pid(p)  # R27 -> P27
            else:
                continue
            if pid not in p_norm:
                p_norm.append(pid)

        # 只取还没抓过的
        q_to_fetch = [q for q in q_norm if q not in self._entity_meta]
        p_to_fetch = [p for p in p_norm if p not in self._prop_meta]
        if not q_to_fetch and not p_to_fetch:
            return


        # --- 实体 ---
        added = 0  # 统计本次真正写入缓存的条数
        async def autosave_entities(pack):
            nonlocal added
            # logger.info("发送一次 wikidata 请求")
            values = " ".join(f"wd:{qid}" for qid in pack)
            q = f"""
            SELECT ?id ?idLabel ?idDescription ?idAltLabel WHERE {{
              VALUES ?id {{ {values} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.langs}". }}
            }}
            """
            data = await self.sparql(q)
            data = data.get("results", {}).get("bindings", [])

            temp = defaultdict(lambda: {"id": "", "label": "", "description": "", "aliases": []})
            for b in data:
                iid = b["id"]["value"].split("/")[-1]
                if not temp[iid]["id"]:
                    temp[iid]["id"] = iid
                if "idLabel" in b:
                    temp[iid]["label"] = b["idLabel"]["value"]
                if "idDescription" in b:
                    temp[iid]["description"] = b["idDescription"]["value"]
                if "idAltLabel" in b:
                    temp[iid]["aliases"].append(b["idAltLabel"]["value"])
            for iid, meta in temp.items():
                if iid not in self._entity_meta:  # 确保只统计“新写入”
                    added += 1
                meta["aliases"] = sorted(list({a for a in meta["aliases"] if a}))
                self._entity_meta[iid] = meta
                if meta["label"]:
                    self._label2qid.setdefault(self._norm_label(meta["label"]), iid)
                for al in meta["aliases"]:
                    self._label2qid.setdefault(self._norm_label(al), iid)
            self._bump_autosave(added)

        # --- 属性 ---
        async def autosave_relations(pack):
            nonlocal added
            values = " ".join(f"wd:{pid}" for pid in pack)
            q = f"""
            SELECT ?pid ?pidLabel ?pidDescription ?pidAltLabel WHERE {{
              VALUES ?pid {{ {values} }}
              SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.langs}". }}
            }}
            """
            data = await self.sparql(q)
            data = data.get("results", {}).get("bindings", [])
            temp = defaultdict(lambda: {"id": "", "label": "", "description": "", "aliases": []})
            for b in data:
                iid = b["pid"]["value"].split("/")[-1]
                if not temp[iid]["id"]:
                    temp[iid]["id"] = iid
                if "pidLabel" in b:
                    temp[iid]["label"] = b["pidLabel"]["value"]
                if "pidDescription" in b:
                    temp[iid]["description"] = b["pidDescription"]["value"]
                if "pidAltLabel" in b:
                    temp[iid]["aliases"].append(b["pidAltLabel"]["value"])
            for iid, meta in temp.items():
                if iid not in self._prop_meta:  # 确保只统计“新写入”
                    added += 1
                meta["aliases"] = sorted(list({a for a in meta["aliases"] if a}))
                self._prop_meta[iid] = meta
                if meta["label"]:
                    self._label2pid.setdefault(self._norm_label(meta["label"]), iid)
                for al in meta["aliases"]:
                    self._label2pid.setdefault(self._norm_label(al), iid)
            self._bump_autosave(added)

        # 异步跑
        entity_tasks = [pack for pack in chunk(q_to_fetch, self.batch_size)]
        runner = AsyncRunner(autosave_entities, max_concurrency=self.max_concurrency)
        await runner.run(entity_tasks)
        relation_tasks = [pack for pack in chunk(p_to_fetch, self.batch_size)]
        runner = AsyncRunner(autosave_relations, max_concurrency=self.max_concurrency)
        await runner.run(relation_tasks)

        # ---------- seed/持久化 ----------
    def _bump_autosave(self, newly_added: int = 0) -> None:
        """根据新增条数累加计数，达到阈值则 to_json 持久化并清零计数。"""
        if newly_added <= 0:
            return
        self._autosave_counter += newly_added
        if (
                self.autosave_path
                and self.autosave_freq > 0
                and self._autosave_counter >= self.autosave_freq
        ):
            try:
                self.to_json(self.autosave_path)
                logger.info(f"[autosave] saved to {self.autosave_path} (added {self._autosave_counter})")
            except Exception as exc:
                logger.warning(f"[autosave] failed: {exc}")
            finally:
                self._autosave_counter = 0

    async def build_from_ids(self, ids: Iterable[str]) -> "WikidataVocab":
        """手动进行id持久化，自适应 Q/P/R"""
        xs  = [str(x) for x in ids if x]
        bad = [s for s in xs if not (is_qid(s) or is_pid(s) or is_rid(s))]
        if bad:
            raise KeyError(f"[build_from_ids] 非法 ID：{bad}（仅支持 Qxxx/Pxxx/Rxxx）")
        q = {s for s in xs if is_qid(s)}
        p = {rid_to_pid(s) if is_rid(s) else s for s in xs if is_pid(s) or is_rid(s)}
        if q or p:
            await self._fetch_meta_ids(qids=q, pids=p)
        return self

    async def build_from_triplets(self, triplets: Union[
        Tuple[str, str, str], Iterable[Tuple[str, str, str]], Iterable[List[Any]]]):
        """手动进行三元组持久化"""
        ts = [triplets] if _is_single_triple(triplets) else list(triplets)
        q = {x for h, r, t in ts for x in (h, t) if is_qid(x)}
        p = {rid_to_pid(r) if is_rid(r) else r for _, r, _ in ts if is_pid(r) or is_rid(r)}
        await self._fetch_meta_ids(qids=q, pids=p)


    def to_json(self, path: str) -> None:
        obj = {
            "langs": self.langs,
            "entity_meta": self._entity_meta,
            "prop_meta": self._prop_meta,
            "label2qid": self._label2qid,
            "label2pid": self._label2pid,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "WikidataVocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        v = cls(langs=obj.get("langs", "en"))
        v._entity_meta = obj.get("entity_meta", {})
        v._prop_meta   = obj.get("prop_meta", {})
        v._label2qid   = obj.get("label2qid", {})
        v._label2pid   = obj.get("label2pid", {})
        return v

    # ---------- encode（label -> ID；若已是 ID 原样返回） ----------
    @vectorize_scalar
    async def encode_entity(self, token: Union[str, List[str]]) -> Union[str, List[str]]:
        if not token:
            raise KeyError("空实体")
        if is_qid(token):
            return token
        key = self._norm_label(token)
        if key in self._label2qid:
            return self._label2qid[key]
        # 兜底：在线查找（先按 label 去搜 entity，再缓存）
        # 出于精度，这里只支持“已 seed 的 QID 或别名”；若需要“任意 label 在线查找”，可以扩展一个 search 接口
        raise KeyError(f"未知实体标签（未在缓存/种子中）：{token}")

    @vectorize_scalar
    async def encode_relation(self, token: Union[str, List[str]]) -> Union[str, List[str]]:
        if not token:
            raise KeyError("空关系")
        if is_pid(token) or is_rid(token):
            return token
        key = self._norm_label(token)
        if key in self._label2pid:
            return self._label2pid[key]
        raise KeyError(f"未知关系标签（未在缓存/种子中）：{token}")

    # ---------- decode（ID -> 富元信息 dict） ----------
    @vectorize_scalar
    async def decode_entity(self, qid: Union[str, List[str]]) -> Union[Any, List[Any]]:
        # if not is_qid(qid):
        #     raise KeyError(f"期望 Qxxx，收到：{qid}")
        if qid not in self._entity_meta:
            await self._fetch_meta_ids(qids=[qid])
        meta = self._entity_meta.get(qid)
        if not meta:
            raise KeyError(f"无法在 Wikidata 上找到实体：{qid}")

        if self.decode_type == "label":
            return meta["label"]
        elif self.decode_type == "label_with_description":
            return meta["label"] + ": " + meta["description"]
        elif self.decode_type == "ori":
            return meta
        raise NotImplementedError
        # return {"id": qid, "label": meta.get("label",""), "description": meta.get("description",""), "aliases": meta.get("aliases",[])}

    @vectorize_scalar
    async def decode_relation(self, rid_or_pid: Union[str, List[str]]) -> Union[Any, List[Any]]:
        # 统一拿到 meta
        if is_rid(rid_or_pid):
            pid = rid_to_pid(rid_or_pid)
            meta = await self._decode_pid_with_id_override(pid, rid_or_pid)
        else:
            if not is_pid(rid_or_pid):
                raise KeyError(f"期望 Pxxx/Rxxx，收到：{rid_or_pid}")
            meta = await self._decode_pid_with_id_override(rid_or_pid, rid_or_pid)

        # 再按 decode_type 返回
        if self.decode_type == "label":
            return meta["label"]
        elif self.decode_type == "label_with_description":
            return meta["label"] + ": " + meta["description"]
        elif self.decode_type == "ori":
            return meta
        raise NotImplementedError

    async def _decode_pid_with_id_override(self, pid: str, return_id: str) -> Dict[str, Any]:
        """
        pid: 必须是 Pxxx（底层查询用）
        return_id: 输出中保留的 id，可为 Pxxx 或 Rxxx（当 decode R 时用于“原样返回 Rxxx”）
        """
        if pid not in self._prop_meta:
            await self._fetch_meta_ids(pids=[pid])  # 已支持 Iterable & R->P 归一化
        meta = self._prop_meta.get(pid)
        if not meta:
            raise KeyError(f"无法在 Wikidata 上找到属性：{pid}")
        return {
            "id": return_id,  # 保留调用方传入的 id（可能是 Rxxx）
            "label": meta.get("label", ""),
            "description": meta.get("description", ""),
            "aliases": meta.get("aliases", []),
        }

    # ---------- 三元组 ----------
    @vectorize_triple
    async def encode_triple(self, tri: Tuple[str, str, str]) -> List[str]:
        h, r, t = tri
        return [await self.encode_entity(h), await self.encode_relation(r), await self.encode_entity(t)]

    @vectorize_triple
    async def decode_triple(self, tri_ids: Tuple[str, str, str]) -> List[Dict[str,Any]]:  # 返回列表而不是tuple方便embedding
        hid, rid, tid = tri_ids
        if bool(ID_RE_REVERSE.match(rid)):
            return [await self.decode_entity(tid), await self.decode_relation(rid), await self.decode_entity(hid)]
        else:
            return [await self.decode_entity(hid), await self.decode_relation(rid), await self.decode_entity(tid)]

    # ---------- 单元 decode（对三元组中的“一个元素”做解码；独立查询，不用缓存） ----------
    async def _sparql_labels_for_ids(self, prefixed_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        独立查询：对一批带前缀（如 'wd:Q42', 'wd:P19', 'wd:L35248-S1'）的 IRI 做 label/description/aliases 解析。
        返回映射：原始前缀 IRI -> {id,label,description,aliases}
        """
        if not prefixed_ids:
            return {}
        values = " ".join(prefixed_ids)
        q = f"""
        SELECT ?id ?idLabel ?idDescription ?idAltLabel WHERE {{
          VALUES ?id {{ {values} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.langs}". }}
        }}
        """
        data = await self.sparql(q)
        rows = data.get("results", {}).get("bindings", [])
        temp = defaultdict(lambda: {"id": "", "label": "", "description": "", "aliases": []})
        for b in rows:
            iid_full = b["id"]["value"]  # 例如 https://www.wikidata.org/entity/Q42
            # 还原成调用方传入的前缀 IRI（wd:/wds:/wdt:/ps:），用末尾 local id 对齐
            local = iid_full.split("/")[-1]  # Q42 / P19 / L35248-S1 ...
            # 猜测调用方 key：优先 wd: 前缀；否则就直接用 wd:local 作为 key
            # 这里做一个“兜底”映射：尽量找到与 local 对应的原始条目
            candidates = [x for x in prefixed_ids if x.split(":")[1] == local]
            key = candidates[0] if candidates else f"wd:{local}"

            if not temp[key]["id"]:
                temp[key]["id"] = key
            if "idLabel" in b:
                temp[key]["label"] = b["idLabel"]["value"]
            if "idDescription" in b:
                temp[key]["description"] = b["idDescription"]["value"]
            if "idAltLabel" in b:
                temp[key]["aliases"].append(b["idAltLabel"]["value"])
        for k, v in temp.items():
            v["aliases"] = sorted(list({a for a in v["aliases"] if a}))
        return temp

    async def _sparql_statement_brief(self, wds_id: str) -> Optional[Dict[str, Any]]:
        """
        从语句节点 (?st) 出发，通过 ps:* 反推属性；主语可选。
        """
        q = f"""
        PREFIX wd:  <http://www.wikidata.org/entity/>
        PREFIX wds: <http://www.wikidata.org/entity/statement/>
        PREFIX p:   <http://www.wikidata.org/prop/>
        PREFIX ps:  <http://www.wikidata.org/prop/statement/>
        PREFIX wikibase: <http://wikiba.se/ontology#>

        SELECT ?item ?prop ?itemLabel ?propLabel ?itemDescription ?propDescription WHERE {{
          BIND({wds_id} AS ?st)

          ?st ?ps ?obj .
          FILTER(STRSTARTS(STR(?ps), STR(ps:)))
          ?prop wikibase:statementProperty ?ps .

          OPTIONAL {{
            ?item ?p ?st .
            ?p wikibase:statementProperty ?prop .
          }}

          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.langs}". }}
        }}
        LIMIT 1
        """
        data = await self.sparql(q)
        rows = data.get("results", {}).get("bindings", [])
        if not rows:
            return None

        b = rows[0]
        subj_local = b.get("item", {}).get("value", "").split("/")[-1] if "item" in b else None
        prop_local = b["prop"]["value"].split("/")[-1]

        ids = [f"wd:{prop_local}"] + ([f"wd:{subj_local}"] if subj_local else [])
        mapped = await self._sparql_labels_for_ids(ids)

        subject = None
        if subj_local:
            subject = mapped.get(f"wd:{subj_local}", {
                "id": f"wd:{subj_local}",
                "label": b.get("itemLabel", {}).get("value", ""),
                "description": b.get("itemDescription", {}).get("value", ""),
                "aliases": []
            })
        prop = mapped.get(f"wd:{prop_local}", {
            "id": f"wd:{prop_local}",
            "label": b.get("propLabel", {}).get("value", ""),
            "description": b.get("propDescription", {}).get("value", ""),
            "aliases": []
        })
        return {"id": wds_id, "subject": subject, "property": prop}

    async def _sparql_lexeme_sense(self, sense_id: str) -> Optional[Dict[str, Any]]:
        """
        解析 wd:Lxxxx-Sn 义项：返回 lexeme id、lemma、gloss 列表（含语言）。
        """
        q = f"""
        PREFIX wd:  <http://www.wikidata.org/entity/>
        PREFIX wikibase: <http://wikiba.se/ontology#>
        PREFIX ontolex: <http://www.w3.org/ns/lemon/ontolex#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

        SELECT ?lexeme ?lemma ?gloss ?glossLang WHERE {{
          BIND({sense_id} AS ?sense)
          ?lexeme ontolex:sense ?sense .
          ?lexeme wikibase:lemma ?lemma .
          ?sense  skos:definition ?gloss .
          BIND(LANG(?gloss) AS ?glossLang)
        }}
        """
        data = await self.sparql(q)
        rows = data.get("results", {}).get("bindings", [])
        if not rows:
            return None

        lexeme_iri = rows[0]["lexeme"]["value"]
        lexeme_id = "wd:" + lexeme_iri.split("/")[-1]
        lemma = rows[0]["lemma"]["value"]
        glosses = [{"lang": r["glossLang"]["value"], "text": r["gloss"]["value"]} for r in rows]

        return {"id": sense_id, "lexeme": {"id": lexeme_id, "lemma": lemma}, "glosses": glosses}

    @vectorize_scalar
    async def decode(self, token: str) -> Any:
        if token is None or (isinstance(token, str) and token.strip() == ""):
            raise KeyError("空 token")

        # 字面量
        if isinstance(token, (int, float)) or (isinstance(token, str) and ":" not in token):
            return {"type": "literal", "value": token} if self.decode_type == "ori" else str(token)

        t = token.strip()
        low = t.lower()

        # ---- wds: 声明节点 ----
        if low.startswith("wds:"):
            info = await self._sparql_statement_brief(t)
            if not info:
                return {"type": "statement", "id": t} if self.decode_type == "ori" else t
            subj = info["subject"]  # 可能为 None
            prop = info["property"]

            if self.decode_type == "ori":
                return {"type": "statement", "id": t, "subject": subj, "property": prop}

            subj_label = (subj or {}).get("label", "")
            prop_label = (prop or {}).get("label", "")
            if self.decode_type == "label":
                if subj_label and prop_label:
                    return f"{subj_label} — {prop_label}"
                return subj_label or prop_label or t
            else:  # label_with_description
                s_left = subj_label
                s_desc = (subj or {}).get("description", "") if subj else ""
                p_left = prop_label
                p_desc = (prop or {}).get("description", "") if prop else ""
                left = f"{s_left}: {s_desc}" if s_left and s_desc else (s_left or "")
                right = f"{p_left}: {p_desc}" if p_left and p_desc else (p_left or "")
                text = " — ".join([x for x in [left, right] if x]) or t
                return text

        # ---- 谓词语法糖 -> 属性 ----
        if low.startswith(("p:", "ps:", "wdt:")):
            m = re.match(r"^(?:p|ps|wdt):(?P<p>P\d+)$", t, re.IGNORECASE)
            if not m:
                return {"type": "property_like", "id": t} if self.decode_type == "ori" else t
            p_full = f"wd:{m.group('p').upper()}"
            meta = await self._sparql_labels_for_ids([p_full])
            meta = meta.get(p_full, {"id": p_full, "label": "", "description": "", "aliases": []})
            if self.decode_type == "ori":
                return {"type": "property", **meta}
            lab, desc = meta.get("label", ""), meta.get("description", "")
            return (lab + (f": {desc}" if (self.decode_type == "label_with_description" and desc) else "")) or meta.get(
                "id")

        # ---- wd: 前缀 ----
        if low.startswith("wd:"):
            local = t.split(":")[1]

            # (a) 义项 L…-S…
            if re.match(r"^L\d+-S\d+$", local, re.IGNORECASE):
                info = await self._sparql_lexeme_sense(t)
                if not info:
                    return {"type": "lexeme_sense", "id": t} if self.decode_type == "ori" else t

                if self.decode_type == "ori":
                    return {"type": "lexeme_sense", **info}

                lemma = info["lexeme"]["lemma"]
                # 选择与 self.langs 中首选语言匹配的 gloss；否则退回第一条
                pref_langs = [x.strip() for x in self.langs.split(",") if x.strip()]
                gloss = None
                for lang in pref_langs:
                    gloss = next((g["text"] for g in info["glosses"] if
                                  g["lang"].lower().startswith(lang.split("-")[0].lower())), None)
                    if gloss:
                        break
                if not gloss and info["glosses"]:
                    gloss = info["glosses"][0]["text"]

                if self.decode_type == "label":
                    return f"{lemma} — {gloss}" if gloss else lemma
                else:  # label_with_description
                    return f"{lemma}: {gloss}" if gloss else lemma

            # (b) 其他 Q/P/L（非 sense）：走通用标签查询
            meta = await self._sparql_labels_for_ids([t])
            meta = meta.get(t, {"id": t, "label": "", "description": "", "aliases": []})
            if self.decode_type == "ori":
                typ = "entity"
                if local.startswith("P"):
                    typ = "property"
                elif local.startswith("L"):
                    typ = "lexeme"
                return {"type": typ, **meta}
            lab, desc = meta.get("label", ""), meta.get("description", "")
            return (lab + (f": {desc}" if (self.decode_type == "label_with_description" and desc) else "")) or meta.get(
                "id")

        # ---- 其他未知前缀 ----
        return {"type": "unknown", "id": t} if self.decode_type == "ori" else t


if __name__ == "__main__":

    async def main_case():
        # 1) 先用你数据里出现的 ID 进行 seed（避免 encode(label) 找不到）
        q_ids = ["Q126399", "Q42"]        # e.g. Warner Bros., Douglas Adams
        p_ids = ["P19", "R27"]            # 出生地，R27= P27 的逆 (country of citizenship) 仅示例
        vocab = await WikidataVocab(langs="en").build_from_ids(q_ids + p_ids)
        await vocab.build_from_triplets(("Q126399", "P19", "Q126399"))
        # 2) encode：label -> ID（若传 ID 原样返回）
        # （注意：label->id 只支持“已 seed”的 label/别名；扩展搜索可在需要时添加）
        try:
            print("encode_entity('Douglas Adams') ->", await vocab.encode_entity("Douglas Adams"))
        except KeyError as e:
            print("[warn]", e)

        # 3) decode：ID -> {id,label,description,aliases}
        print("decode_entity('Q126399') ->", await vocab.decode_entity(["Q126399"]))
        print("decode_relation('P19')   ->", await vocab.decode_relation(["P19"]))
        print("decode_relation('R27')   ->", await vocab.decode_relation("R27"))

        # 4) 三元组
        print("build triplet begin")
        vocab = WikidataVocab(langs="en")
        await vocab.build_from_triplets([["Q126399", "P19", "Q126399"]])
        time.sleep(1)
        print("triplet begin")
        tri_ids = ("Q126399", "P19", "Q126399")
        print("decode_triple(tri_ids) ->", await vocab.decode_triple(tri_ids))
        print("triplet end")

    import asyncio
    asyncio.run(main_case())