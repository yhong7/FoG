from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterable, Optional, Sequence, Literal, Union
from collections import Counter
import json, os
from functools import wraps
from collections.abc import Iterable
from loguru import logger
from functools import wraps
from collections.abc import Iterable

# --------- 通用判断：把字符串视为标量 ---------
def _is_iterable_but_not_str(x):
    return isinstance(x, Iterable) and not isinstance(x, (str, bytes))

def _is_single_triple(x):
    # 单个三元组：list/tuple 且长度为3，且元素不是 list/tuple（避免 [[...], ... , ...]）
    return isinstance(x, (list, tuple)) and len(x) == 3 and not any(isinstance(e, (list, tuple)) for e in x)

# --------- 装饰器1：标量/序列自动向量化（针对 entity_to_id 等标量函数） ---------
def vectorize_scalar(fn):
    """
    输入是标量 -> 返回标量
    输入是可迭代(list/tuple/generator/np array等，非字符串) -> 返回 list，元素为逐个调用 fn 的结果
    """
    @wraps(fn)
    def wrapper(self, x, *args, **kwargs):
        if _is_iterable_but_not_str(x):
            return [fn(self, xi, *args, **kwargs) for xi in x]
        return fn(self, x, *args, **kwargs)
    return wrapper

# --------- 装饰器2：三元组自动向量化（针对 encode_triple / decode_triple） ---------
def vectorize_triple(fn):
    """
    (h, r, t) 或 (hid, rid, tid) 的单个三元组 -> 单个结果
    三元组序列 -> 列表结果
    """
    @wraps(fn)
    def wrapper(self, x, *args, **kwargs):
        if _is_single_triple(x):
            return fn(self, x, *args, **kwargs)
        if _is_iterable_but_not_str(x):
            xs = list(x)
            for tri in xs:
                if not _is_single_triple(tri):
                    raise TypeError(f"vectorize_triple 期望三元组或其序列，发现非法项：{tri!r}")
            return [fn(self, tri, *args, **kwargs) for tri in xs]
        raise TypeError(f"vectorize_triple 期望 (h, r, t) 或其序列，收到类型：{type(x)}")
    return wrapper


@dataclass
class KnowledgeGraphVocab:
    """
    知识图谱词表
    - 两个独立命名空间：entities / relations
    - 从三元组构建；或按类型单独添加
    - 编码/解码：(h, r, t) <-> (hid, rid, tid)
    - 冻结/裁剪/保存/加载

    oov_policy:
      - "error": 未登录项会抛出 KeyError
      - "add":   未冻结时遇到新项自动加入
    """

    lowercase: bool = True
    oov_policy: Literal["error", "add"] = "add"

    # 频次与上限（可分别配置）
    min_freq_entity: int = 1
    min_freq_relation: int = 1
    max_entities: Optional[int] = None
    max_relations: Optional[int] = None

    # --- 内部状态（实体） ---
    _e_itos: List[str] = field(default_factory=list, init=False, repr=False)  # id -> entity
    _e_stoi: Dict[str, int] = field(default_factory=dict, init=False, repr=False)  # entity -> id
    _e_counter: Counter = field(default_factory=Counter, init=False, repr=False)
    _e_frozen: bool = field(default=False, init=False, repr=False)

    # --- 内部状态（关系） ---
    _r_itos: List[str] = field(default_factory=list, init=False, repr=False)  # id -> relation
    _r_stoi: Dict[str, int] = field(default_factory=dict, init=False, repr=False)  # relation -> id
    _r_counter: Counter = field(default_factory=Counter, init=False, repr=False)
    _r_frozen: bool = field(default=False, init=False, repr=False)

    # -------------------- 基础 --------------------
    def _norm(self, s: str) -> str:
        return s.lower() if self.lowercase else s

    @property
    def num_entities(self) -> int:
        return len(self._e_itos)

    @property
    def num_relations(self) -> int:
        return len(self._r_itos)

    @property
    def entity_list(self) -> List[str]:
        return self._e_itos

    @property
    def relation_list(self) -> List[str]:
        return self._r_itos

    # -------------------- 添加 --------------------
    def add(self, token: str, kind: Literal["entity", "relation"]) -> int:
        if kind == "entity":
            return self.add_entity(token)
        else:
            return self.add_relation(token)

    def add_entity(self, entity: str) -> int:
        self._ensure_not_frozen("entity")
        e = self._norm(entity)
        if e in self._e_stoi:
            return self._e_stoi[e]
        idx = len(self._e_itos)
        self._e_itos.append(entity)
        self._e_stoi[e] = idx
        return idx

    def add_relation(self, relation: str) -> int:
        self._ensure_not_frozen("relation")
        r = self._norm(relation)
        if r in self._r_stoi:
            return self._r_stoi[r]
        idx = len(self._r_itos)
        self._r_itos.append(relation)
        self._r_stoi[r] = idx
        return idx

    # -------------------- 构建/更新 --------------------
    def build_from_triples(
        self,
        triples: Iterable[Tuple[str, str, str]],
        *,
        min_freq_entity: Optional[int] = None,
        min_freq_relation: Optional[int] = None,
        max_entities: Optional[int] = None,
        max_relations: Optional[int] = None,
        force_recreate: bool = False,
    ) -> None:
        """
        基于 (head, relation, tail) 三元组构建或增量更新词表。
        """
        if force_recreate:
            # 重建（清空再建）
            self._e_itos, self._e_stoi, self._e_counter = [], {}, Counter()
            self._r_itos, self._r_stoi, self._r_counter = [], {}, Counter()

        mf_e = self.min_freq_entity if min_freq_entity is None else min_freq_entity
        mf_r = self.min_freq_relation if min_freq_relation is None else min_freq_relation
        mx_e = self.max_entities if max_entities is None else max_entities
        mx_r = self.max_relations if max_relations is None else max_relations

        e_counter, r_counter = Counter(), Counter()
        for h, r, t in triples:
            e_counter.update([self._norm(h), self._norm(t)])
            r_counter.update([self._norm(r)])

        # 记录计数，便于后续 trim
        self._e_counter.update(e_counter)
        self._r_counter.update(r_counter)

        # 依据频次选取候选并稳定排序
        e_cand = [(tok, c) for tok, c in e_counter.items() if c >= mf_e]
        r_cand = [(tok, c) for tok, c in r_counter.items() if c >= mf_r]
        e_cand.sort(key=lambda x: (-x[1], x[0]))
        r_cand.sort(key=lambda x: (-x[1], x[0]))
        if mx_e is not None:
            e_cand = e_cand[:mx_e]
        if mx_r is not None:
            r_cand = r_cand[:mx_r]

        # 写入
        for tok, _ in e_cand:
            if tok not in self._e_stoi:
                self._e_stoi[tok] = len(self._e_itos)
                self._e_itos.append(tok)
        for tok, _ in r_cand:
            if tok not in self._r_stoi:
                self._r_stoi[tok] = len(self._r_itos)
                self._r_itos.append(tok)

    # -------------------- 编码/解码 --------------------
    @vectorize_scalar
    def encode_entity(self, entity:Union[str, List[str]]) -> Union[int, List[int]]:
        key = self._norm(entity)
        if key in self._e_stoi:
            return self._e_stoi[key]
        if self.oov_policy == "add" and not self._e_frozen:
            return self.add_entity(entity)
        raise KeyError(f"未知实体：{entity}")

    @vectorize_scalar
    def encode_relation(self, relation:Union[str, List[str]]) -> Union[int, List[int]]:
        key = self._norm(relation)
        if key in self._r_stoi:
            return self._r_stoi[key]
        if self.oov_policy == "add" and not self._r_frozen:
            return self.add_relation(relation)
        raise KeyError(f"未知关系：{relation}")

    @vectorize_scalar
    def decode_entity(self, idx:Union[int, List[int]]) -> Union[str, List[str]]:
        if idx < 0 or idx >= len(self._e_itos):
            raise IndexError(f"entity id {idx} 越界 [0, {len(self._e_itos) - 1}]")
        return self._e_itos[idx]

    @vectorize_scalar
    def decode_relation(self, idx:Union[int, List[int]]) -> Union[str, List[str]]:
        if idx < 0 or idx >= len(self._r_itos):
            raise IndexError(f"relation id {idx} 越界 [0, {len(self._r_itos) - 1}]")
        return self._r_itos[idx]

    @vectorize_triple
    def encode_triple(self, triple: Union[List[str], List[List[str]]]) -> Union[List[int], List[List[int]]]:
        h, r, t = triple
        return [self.encode_entity(h), self.encode_relation(r), self.encode_entity(t)]

    @vectorize_triple
    def decode_triple(self, id_triple: Union[List[int], List[List[int]]]) -> Union[List[str], List[List[str]]]:
        hid, rid, tid = id_triple
        return [self.decode_entity(hid), self.decode_relation(rid), self.decode_entity(tid)]

    # -------------------- 冻结/裁剪 --------------------
    def freeze(self, kind: Optional[Literal["entity", "relation"]] = None) -> None:
        if kind in (None, "entity"):
            self._e_frozen = True
        if kind in (None, "relation"):
            self._r_frozen = True

    def unfreeze(self, kind: Optional[Literal["entity", "relation"]] = None) -> None:
        if kind in (None, "entity"):
            self._e_frozen = False
        if kind in (None, "relation"):
            self._r_frozen = False

    def _ensure_not_frozen(self, kind: Literal["entity", "relation"]):
        if kind == "entity" and self._e_frozen:
            raise RuntimeError("实体词表已冻结，无法修改。")
        if kind == "relation" and self._r_frozen:
            raise RuntimeError("关系词表已冻结，无法修改。")

    def trim(
        self,
        *,
        min_freq_entity: Optional[int] = None,
        min_freq_relation: Optional[int] = None,
        max_entities: Optional[int] = None,
        max_relations: Optional[int] = None,
    ) -> None:
        """按频次/数量对当前词表裁剪（会重新分配 id）。"""
        self._ensure_not_frozen("entity")
        self._ensure_not_frozen("relation")

        mf_e = self.min_freq_entity if min_freq_entity is None else min_freq_entity
        mf_r = self.min_freq_relation if min_freq_relation is None else min_freq_relation
        mx_e = self.max_entities if max_entities is None else max_entities
        mx_r = self.max_relations if max_relations is None else max_relations

        # --- 裁剪实体 ---
        e_counts = {tok: self._e_counter.get(tok, 1) for tok in self._e_itos}
        e_cand = [(tok, c) for tok, c in e_counts.items() if c >= mf_e]
        e_cand.sort(key=lambda x: (-x[1], x[0]))
        if mx_e is not None:
            e_cand = e_cand[:mx_e]
        new_e_itos, new_e_stoi = [], {}
        for tok, _ in e_cand:
            new_e_stoi[tok] = len(new_e_itos)
            new_e_itos.append(tok)
        self._e_itos, self._e_stoi = new_e_itos, new_e_stoi

        # --- 裁剪关系 ---
        r_counts = {tok: self._r_counter.get(tok, 1) for tok in self._r_itos}
        r_cand = [(tok, c) for tok, c in r_counts.items() if c >= mf_r]
        r_cand.sort(key=lambda x: (-x[1], x[0]))
        if mx_r is not None:
            r_cand = r_cand[:mx_r]
        new_r_itos, new_r_stoi = [], {}
        for tok, _ in r_cand:
            new_r_stoi[tok] = len(new_r_itos)
            new_r_itos.append(tok)
        self._r_itos, self._r_stoi = new_r_itos, new_r_stoi

    # -------------------- 导入/导出 --------------------
    def to_json(self, path: str) -> None:
        obj = {
            "lowercase": self.lowercase,
            "oov_policy": self.oov_policy,
            "min_freq_entity": self.min_freq_entity,
            "min_freq_relation": self.min_freq_relation,
            "max_entities": self.max_entities,
            "max_relations": self.max_relations,
            "entities": self._e_itos,   # 顺序即 id
            "relations": self._r_itos,  # 顺序即 id
            "frozen": {"entity": self._e_frozen, "relation": self._r_frozen},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "KnowledgeGraphVocab":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        vocab = cls(
            lowercase=obj.get("lowercase", False),
            oov_policy=obj.get("oov_policy", "error"),
            min_freq_entity=obj.get("min_freq_entity", 1),
            min_freq_relation=obj.get("min_freq_relation", 1),
            max_entities=obj.get("max_entities", None),
            max_relations=obj.get("max_relations", None),
        )
        # 直接使用保存顺序恢复映射
        vocab._e_itos = list(obj.get("entities", []))
        vocab._r_itos = list(obj.get("relations", []))
        vocab._e_stoi = {vocab._norm(t): i for i, t in enumerate(vocab._e_itos)}
        vocab._r_stoi = {vocab._norm(t): i for i, t in enumerate(vocab._r_itos)}
        frz = obj.get("frozen", {})
        vocab._e_frozen = bool(frz.get("entity", False))
        vocab._r_frozen = bool(frz.get("relation", False))

        logger.info(f"[Vocab] Vocab loaded successfully → path={path}")
        return vocab

    # -------------------- 导出映射/信息 --------------------
    def entity_dict(self) -> Dict[str, int]:
        return {tok: i for i, tok in enumerate(self._e_itos)}

    def relation_dict(self) -> Dict[str, int]:
        return {tok: i for i, tok in enumerate(self._r_itos)}

    def info(self) -> str:
        return (f"KGVocab(entities={self.num_entities}, relations={self.num_relations}, "
                f"lowercase={self.lowercase}, oov_policy='{self.oov_policy}', "
                f"frozen_e={self._e_frozen}, frozen_r={self._r_frozen})")


if __name__ == "__main__":
    triples = [
        ("Paris", "capital_of", "France"),
        ("France", "located_in", "Europe"),
        ("Louvre", "located_in", "Paris"),
    ]

    v = KnowledgeGraphVocab(lowercase=True, oov_policy="add")
    v.build_from_triples(triples)
    print(v.info())  # -> 实体/关系数量

    # encode / decode
    v.encode_entity("Paris")  # -> int
    v.encode_entity(["Paris", "France"])  # -> List[int]

    v.encode_triple(("Paris", "capital_of", "France"))  # -> (hid, rid, tid)
    v.encode_triple([
        ("Paris", "capital_of", "France"),
        ("Louvre", "located_in", "Paris"),
    ])  # -> List[(hid, rid, tid)]
    v.decode_triple([(1, 0, 2), (2, 1, 3)])  # -> List[(h, r, t)]

    # 增量添加单个条目（明确类型）
    v.add("Berlin", kind="entity")
    print(v.info())  # -> 实体/关系数量

    # 冻结，训练期防止 id 漂移
    v.freeze()  # 或 v.freeze("entity") / v.freeze("relation")
    # v.add("NewEntity", "entity")  # 会报错
    # v.encode_triple(("NewEntity", "NewRelation", "NewEntity2"))  # 会报错

    # 保存/加载
    v.to_json("files/kg_vocab.json")
    v2 = KnowledgeGraphVocab.from_json("files/kg_vocab.json")
    print(v2.encode_triple(("Paris", "capital_of", "France")))
    print(v2.info())  # -> 实体/关系数量
