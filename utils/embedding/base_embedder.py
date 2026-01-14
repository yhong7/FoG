import numpy as np


from abc import ABC, abstractmethod
from typing import List, Union, Any, Dict, Tuple, Optional
import asyncio
import hashlib
import importlib
import os
import pickle
import tempfile

import torch
import faiss

from ..async_runner import AsyncRunner
from utils.flatten_and_restore_list import flatten_and_restore_list
from config.model_args import DEVICE

from pydantic import BaseModel


def batch_texts(texts: List[str], batch_size: int) -> List[List[str]]:
    return [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]


class ConnectionConfig(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    model_name: str
    embedding_args: Optional[Dict[str, Any]] = None

    model_config = {
        "protected_namespaces": ()
    }

class FaissTextCache:
    """
    文本精确匹配缓存 + 懒加载 FAISS + 周期性持久化（默认每新增 persist_every 个向量落盘一次）。

    - 读缓存（命中）只依赖本地字典，不导入 faiss；
    - add()/读索引时才导入 faiss；
    - 持久化两份文件：index（FAISS 原生）+ meta（pickle：id 映射、原始向量等）；
    - 原子写入：先写临时文件，再 rename。
    """

    def __init__(
        self,
        persist_dir: Optional[str],
        persist_every: int = 32,
        metric: str = "ip",
        load_if_exists: bool = True,
    ):
        """
        :param persist_dir: 持久化目录（None 则不持久化）
        :param persist_every: 每新增多少条向量自动持久化一次
        :param metric: "ip"（内积/余弦）或 "l2"
        :param load_if_exists: 若目录存在持久化文件则自动恢复
        """
        assert metric in ("ip", "l2")
        self.metric = metric

        # 懒加载组件
        self.index: Optional[Any] = None        # faiss.IndexIDMap2，按需创建/加载
        self._dim: Optional[int] = None         # 向量维度（用于一致性检查）

        # 轻量本地映射（读缓存无需 faiss）
        self.idmap: Dict[str, int] = {}         # text_hash -> id
        self.id2vec: Dict[int, np.ndarray] = {} # id -> 向量（float32，未归一化）
        self.id2text: Dict[int, str] = {}       # 便于调试
        self._next_id: int = 0

        # 持久化配置
        self.persist_dir = persist_dir
        self.persist_every = max(1, persist_every)
        self._added_since_persist = 0

        self._lock = asyncio.Lock()

        # 启动恢复（仅元数据就足够满足纯命中读）
        if self.persist_dir and load_if_exists:
            self._load_meta_if_available()

    # -------------------- 工具函数 --------------------
    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _to_float32(arr: Union[List[float], List[List[float]], np.ndarray]) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 1:
            a = a[None, :]
        return a

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return mat / norms

    def _index_path(self) -> Optional[str]:
        if not self.persist_dir:
            return None
        return os.path.join(self.persist_dir, "faiss.index")

    def _meta_path(self) -> Optional[str]:
        if not self.persist_dir:
            return None
        return os.path.join(self.persist_dir, "faiss_meta.pkl")

    def _load_meta_if_available(self):
        """仅加载元数据 meta，不加载 FAISS 索引；这样纯命中读不依赖 faiss。"""
        meta_path = self._meta_path()
        if not meta_path or not os.path.isfile(meta_path):
            return
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.idmap = meta.get("idmap", {})
            self.id2text = meta.get("id2text", {})
            raw_id2vec = meta.get("id2vec", {})
            self.id2vec = {int(k): np.asarray(v, dtype=np.float32) for k, v in raw_id2vec.items()}
            self._next_id = int(meta.get("_next_id", 0))
            self.metric = meta.get("metric", self.metric)
            self._dim = int(meta["dim"]) if "dim" in meta else (next(iter(self.id2vec.values())).shape[0] if self.id2vec else None)
        except Exception:
            # 遇到损坏数据时忽略恢复，避免影响运行
            pass

    def _ensure_index(self, expected_dim: Optional[int] = None):
        """按需加载/创建索引（此处才导入 faiss）。"""
        if self.index is not None:
            if expected_dim is not None and self._dim is not None and self._dim != expected_dim:
                raise ValueError(f"FAISS 索引维度({self._dim})与新增向量维度({expected_dim})不一致。")
            return

        # 优先从磁盘读已有索引
        index_path = self._index_path()
        if index_path and os.path.isfile(index_path):
            self.index = faiss.read_index(index_path)
            try:
                self._dim = int(self.index.d)
            except Exception:
                self._dim = expected_dim
            return

        # 否则根据期望维度新建
        if expected_dim is None:
            if self._dim is None:
                if self.id2vec:
                    self._dim = next(iter(self.id2vec.values())).shape[0]
                else:
                    raise ValueError("无法确定索引维度：请在第一次写入时提供向量以确定维度。")
        else:
            self._dim = expected_dim

        base = faiss.IndexFlatIP(self._dim) if self.metric == "ip" else faiss.IndexFlatL2(self._dim)
        self.index = faiss.IndexIDMap2(base)

    # -------------------- 核心接口 --------------------
    async def get_hits_and_misses(self, texts: List[str]) -> Tuple[List[Optional[np.ndarray]], List[int], List[str]]:
        """
        返回：
          - hits: 与 texts 等长的列表，命中处为 np.ndarray(1, dim)，未命中为 None
          - miss_positions: 未命中的下标
          - miss_texts: 未命中的文本
        """
        hits: List[Optional[np.ndarray]] = []
        miss_positions: List[int] = []
        miss_texts: List[str] = []

        for i, t in enumerate(texts):
            key = self._hash_text(t)
            if key in self.idmap:
                _id = self.idmap[key]
                vec = self.id2vec.get(_id)
                if vec is not None:
                    hits.append(vec[None, :])
                else:
                    hits.append(None)
                    miss_positions.append(i)
                    miss_texts.append(t)
            else:
                hits.append(None)
                miss_positions.append(i)
                miss_texts.append(t)
        return hits, miss_positions, miss_texts

    async def add(self, texts: List[str], embeds: List[List[float]]):
        """
        写入 FAISS + 本地字典；根据 persist_every 进行周期性持久化。
        首次写入时才会导入 faiss & 创建/加载索引。
        具备“已存在则跳过”的保护，避免重复写入。
        """
        vecs_all = self._to_float32(embeds)  # (n, dim)

        # 过滤掉已经存在于缓存中的文本（并保持顺序）
        filtered_texts: List[str] = []
        filtered_vecs: List[np.ndarray] = []
        for t, v in zip(texts, vecs_all):
            key = self._hash_text(t)
            if key in self.idmap:
                continue
            filtered_texts.append(t)
            filtered_vecs.append(v)

        if not filtered_texts:
            return  # 全部已存在，无需任何写入

        vecs = np.vstack(filtered_vecs)  # (m, dim)
        n, dim = vecs.shape

        async with self._lock:
            self._ensure_index(expected_dim=dim)

            add_vecs = self._l2_normalize(vecs) if self.metric == "ip" else vecs
            ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)

            # 更新本地字典（未归一化向量）
            for i, t in enumerate(filtered_texts):
                key = self._hash_text(t)
                _id = int(ids[i])
                self.idmap[key] = _id
                self.id2vec[_id] = vecs[i]
                self.id2text[_id] = t

            # 写入 FAISS
            self.index.add_with_ids(add_vecs, ids)  # type: ignore

            # 计数并推进游标
            self._next_id += n
            self._added_since_persist += n

            # 周期性持久化
            if self.persist_dir and self._added_since_persist >= self.persist_every:
                self._persist_locked()
                self._added_since_persist = 0

    async def get_by_texts(self, texts: List[str]) -> List[np.ndarray]:
        """假定都已存在缓存；返回 [(dim,), ...]"""
        out: List[np.ndarray] = []
        for t in texts:
            key = self._hash_text(t)
            _id = self.idmap[key]
            out.append(self.id2vec[_id])
        return out

    # -------------------- 持久化 --------------------
    def persist(self):
        """
        主动持久化（线程/协程外部调用）。如需显式保存，可在外层调用。
        """
        if not self.persist_dir:
            return
        async def _persist_async():
            async with self._lock:
                self._persist_locked()
        try:
            loop = asyncio.get_running_loop()
            return loop.create_task(_persist_async())
        except RuntimeError:
            asyncio.run(_persist_async())

    def _persist_locked(self):
        """要求外部已持有 self._lock。"""
        os.makedirs(self.persist_dir, exist_ok=True)  # type: ignore
        meta_path = self._meta_path()
        index_path = self._index_path()

        # 1) 写 meta
        meta = {
            "idmap": self.idmap,
            "id2vec": self.id2vec,
            "id2text": self.id2text,
            "_next_id": self._next_id,
            "metric": self.metric,
            "dim": self._dim,
        }
        with tempfile.NamedTemporaryFile(dir=self.persist_dir, delete=False) as tf:  # type: ignore
            pickle.dump(meta, tf)
            tmp_meta = tf.name
        os.replace(tmp_meta, meta_path)  # type: ignore

        # 2) 写 index（若已初始化）
        if self.index is not None and index_path:
            with tempfile.NamedTemporaryFile(dir=self.persist_dir, delete=False) as tf:  # type: ignore
                faiss.write_index(self.index, tf.name)
                tmp_index = tf.name
            os.replace(tmp_index, index_path)  # type: ignore

    def close(self):
        """关闭前可调用，确保将内存落盘。"""
        if self.persist_dir:
            self._persist_locked()


class BaseEmbedder(ABC):
    def __init__(
            self,
            batch_size: int = 10,
            max_concurrency: int = 5,
            max_retries: int = 0,
            max_rate: int = None,
            if_tqdm: bool = True,
            faiss_metric: str = "ip",
            cache_persist_dir: Optional[str] = None,
            cache_persist_every: int = 512,
            **kwargs
    ):
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.if_tqdm = if_tqdm
        self.max_rate = max_rate

        self.model_name = "unknown_embedder_name"

        # 缓存/持久化策略（此处不创建 cache；首次 embed_cached 再创建，避免初始化即触发）
        self.faiss_metric = faiss_metric
        self.cache_persist_dir = cache_persist_dir
        self.cache_persist_every = cache_persist_every

        self._cache: Optional[FaissTextCache] = None  # 首次 embed_cached 才创建

    def _ensure_cache(self):
        if self._cache is None:
            self._cache = FaissTextCache(
                metric=self.faiss_metric,
                persist_dir=self.cache_persist_dir,
                persist_every=self.cache_persist_every,
                load_if_exists=True,
            )

    @abstractmethod
    async def _embed(self, texts: List[str]) -> List[List[float]]:
        """要求直接输出二维的向量列表"""
        pass

    @flatten_and_restore_list(flatten_param_name='texts')
    async def embed(self, texts: Union[str, List[Any]]) -> List[Any]:
        """
        :param texts: str 或任意嵌套格式的 List
        :return: embedding 或原格式的 embedding 列表
        """
        if isinstance(texts, str):
            embedding_lst = await self.embed([texts])
            return embedding_lst[0]

        batches = batch_texts(texts, self.batch_size)  # 打成batch
        runner = AsyncRunner(
            self._embed,
            max_concurrency=self.max_concurrency,
            if_tqdm=self.if_tqdm,
            max_retries=self.max_retries,
            max_rate=self.max_rate,
            func_desc=f"{self.model_name} embedding"
        )
        embeddings = await runner.run(batches)
        flatten_embeddings = [item for sublist in embeddings for item in sublist]  # 从batch还原

        return flatten_embeddings

    @flatten_and_restore_list(flatten_param_name='texts')
    async def embed_cached(self, texts: Union[str, List[Any]]) -> List[Any]:
        if isinstance(texts, str):
            res = await self.embed_cached([texts])
            return res[0]

        # texts = ["hidden entity" if t.startswith(("g.", "h.")) else t for t in texts]

        # 首次使用缓存时再创建缓存对象（不会导入 faiss）
        self._ensure_cache()

        # 1) 先查缓存，定位 miss
        hits, miss_pos, miss_texts = await self._cache.get_hits_and_misses(texts)  # type: ignore

        # 2) 对 miss 去重，仅对唯一文本做一次 _embed
        if miss_texts:
            unique_miss_texts: List[str] = []
            positions_by_text: Dict[str, List[int]] = {}
            for pos, t in zip(miss_pos, miss_texts):
                if t not in positions_by_text:
                    positions_by_text[t] = [pos]
                    unique_miss_texts.append(t)
                else:
                    positions_by_text[t].append(pos)

            # 只对 unique_miss_texts 调用一次 _embed
            runner = AsyncRunner(
                self._embed,
                max_concurrency=self.max_concurrency,
                if_tqdm=self.if_tqdm,
                max_retries=self.max_retries,
                max_rate=self.max_rate,
                func_desc=f"{self.model_name} embedding (cache miss)"
            )
            miss_batches = batch_texts(unique_miss_texts, self.batch_size)
            miss_embeds_batches = await runner.run(miss_batches)
            unique_miss_embeds: List[List[float]] = [e for b in miss_embeds_batches for e in b]

            # 3) 写入缓存（内部已做“已存在则跳过”的保护）
            await self._cache.add(unique_miss_texts, unique_miss_embeds)  # type: ignore

            # 4) 将每个唯一文本的向量回填到其所有位置
            for t, emb in zip(unique_miss_texts, unique_miss_embeds):
                vec = np.asarray(emb, dtype=np.float32)[None, :]
                for pos in positions_by_text[t]:
                    hits[pos] = vec

        # 5) 按输入顺序输出
        out: List[List[float]] = [h.squeeze(0).tolist() for h in hits]  # type: ignore

        return out

    async def embed_with_tensor(self, texts: Union[str, List[Any]], device: str = DEVICE, cached: bool = True) -> torch.Tensor:

        if cached:
            out = await self.embed_cached(texts)
        else:
            out = await self.embed(texts)
        out = torch.tensor(out, dtype=torch.float32)
        if device:
            out = out.to(device)
        return out


    def flush_cache(self):
        if self._cache:
            self._cache.persist()

    def close(self):
        if self._cache:
            self._cache.close()


    def __call__(self, texts: List[str]) -> Union[List[float], np.ndarray, torch.Tensor]:
        # 快捷输出
        return asyncio.run(self.embed(texts))
