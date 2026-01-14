import faiss
import numpy as np
import os
import json
from typing import List, Dict, Union, Any, Tuple, Literal, Callable, Coroutine, Optional
from loguru import logger
import asyncio

class FaissVectorStore:
    # Define an internal key for storing the actual content used for vectorization
    _VECTOR_SOURCE_CONTENT_KEY = "__vector_source_content__"

    @classmethod
    async def create(cls,
                     vector_store_root_dir: str,
                     embedder: Callable[[List[str]], Coroutine],
                     faiss_index_type: str = "Flat",
                     persist_every_n: Optional[int] = None
                     ):
        """
        :param embedder: 一个async的embedding函数，输入[texts], 输出[embeddings]
        :param faiss_index_type: faiss中index方式，默认为Flat
        """
        # 异步获取一个嵌入向量以确定维度
        embedding = await embedder(["length check"])
        embedding_dim = len(embedding[0])
        return cls(vector_store_root_dir, embedder, embedding_dim, faiss_index_type, persist_every_n)

    def __init__(
        self,
        vector_store_root_dir: str,
        embedder: Callable[[List[str]], Coroutine],
        embedding_dim: int,
        faiss_index_type: str = "Flat",
        persist_every_n: Optional[int] = None):
        """
        Initializes the FAISS vector store.

        Args:
            vector_store_root_dir (str): The root directory for storing all files.
            embedder: async embedding fn(texts)->embeddings
            embedding_dim (int): Dimension of embeddings.
            faiss_index_type (str): The type of FAISS index, e.g., "Flat" (exact search). Currently only "Flat" is supported.
        """
        self.embedding_dim = embedding_dim
        self.vector_store_root_dir = vector_store_root_dir
        self.embedder = embedder

        # Internal file paths
        self.index_file_path = os.path.join(self.vector_store_root_dir, "faiss_index.bin")
        self.payload_file_path = os.path.join(self.vector_store_root_dir, "payload_map.json")
        # Store config for consistent loading
        self.config_file_path = os.path.join(self.vector_store_root_dir, "store_config.json")

        # 自动持久化参数
        self.persist_every_n = persist_every_n
        self._since_last_persist = 0

        # payload_map: Stores FAISS ID -> full payload (dict)
        self.payload_map: Dict[int, Dict[str, Any]] = {}
        # content_to_index_map: Maps vector source content -> FAISS ID (用于去重)
        self.content_to_index_map: Dict[str, int] = {}

        # Ensure the root directory exists
        os.makedirs(self.vector_store_root_dir, exist_ok=True)

        self.faiss_index = None
        self._initialize_faiss_index(faiss_index_type)
        self._load_from_disk_if_exists()



    def _initialize_faiss_index(self, faiss_index_type: str):
        """Initializes the FAISS index."""
        if faiss_index_type == "Flat":
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)  # Using L2 target_hop for exact search
        else:
            raise ValueError(f"Unsupported FAISS index type: {faiss_index_type}. Currently only 'Flat' is supported.")

    def _add_entry(self, payload: Dict[str, Any], vector: np.ndarray, vector_key_value: str):
        """
        Adds a new entry (including vector and full payload) to the vector store.
        Automatically adds an internal field to the payload for the vector source content.

        Args:
            payload (Dict[str, Any]): The complete metadata payload.
            vector (np.ndarray): The corresponding vector.
            vector_key_value (str): The content from the payload used to generate the vector.
        """
        vector = vector.astype('float32').reshape(1, -1)

        current_faiss_id = self.faiss_index.ntotal
        self.faiss_index.add(vector)

        # Add the internal field for vector source content
        payload[self._VECTOR_SOURCE_CONTENT_KEY] = vector_key_value

        self.payload_map[current_faiss_id] = payload
        self.content_to_index_map[str(vector_key_value)] = current_faiss_id

        # 自动持久化
        if self.persist_every_n is not None:
            self._since_last_persist += 1
            if self._since_last_persist >= self.persist_every_n:
                self.persist_to_disk()
                self._since_last_persist = 0

    async def get_or_create_embeddings(self, payloads: List[Dict[str, Any]], vector_key_name: str) -> List[List[float]]:
        """
        并发安全 & 本批次去重 的版本。
        - 同批次将相同内容合并（避免重复 embed）
        - 写入前在锁内做二次检查，避免重复插入
        """
        # 懒初始化一个全局写入锁，保护 _add_entry 与 content_to_index_map 的一致性
        if not hasattr(self, "_index_lock"):
            self._index_lock = asyncio.Lock()

        final_embeddings: List[Union[List[float], None]] = [None] * len(payloads)

        # 统计：key -> 该内容在本批次出现的所有原始下标列表
        positions_by_key: Dict[str, List[int]] = {}
        # 需要重算但库里已有（用于非 IndexFlat、或你希望保持现有行为重算的情况）
        need_embed_existing: List[str] = []
        # 需要真正新插入（当前库中不存在）
        need_embed_new: List[str] = []
        # 新插入时用哪个 payload 作为代表（第一个看到的那份即可）
        representative_payload: Dict[str, Dict[str, Any]] = {}

        # 先扫描一遍，命中缓存的就直接填；未命中的做本批次去重
        for i, p in enumerate(payloads):
            val = p.get(vector_key_name)
            if val is None:
                print(f"Warning: Payload at index {i} does not contain '{vector_key_name}'. Skipping.")
                continue

            key_str = str(val)

            # 记录该 key 在本批次所有位置
            positions_by_key.setdefault(key_str, []).append(i)

            if key_str in self.content_to_index_map:
                faiss_id = self.content_to_index_map[key_str]
                if isinstance(self.faiss_index, faiss.IndexFlat):
                    # 命中缓存且支持重构，直接回填到所有该 key 的位置
                    vec = self.faiss_index.reconstruct(faiss_id).tolist()
                    for pos in positions_by_key[key_str]:
                        final_embeddings[pos] = vec
                else:
                    # 命中缓存但不便重构：保持原行为——稍后重算返回，但不写入
                    # 只加入一次（批内去重）
                    if key_str not in need_embed_existing:
                        need_embed_existing.append(key_str)
            else:
                # 库里没有：批内去重后准备 embed + 插入
                if key_str not in need_embed_new:
                    need_embed_new.append(key_str)
                    representative_payload[key_str] = p  # 选第一份作为插入使用的元数据

        # 需要进行 embed 的 keys（包括：库里已有但要重算返回的、以及全新要插入的）
        keys_to_embed: List[str] = need_embed_existing + need_embed_new

        if keys_to_embed:
            # 做一次批量 embed（按 keys_to_embed 的顺序）
            new_vectors = await self.embedder(keys_to_embed)

            # 逐个 key 处理
            for idx, key_str in enumerate(keys_to_embed):
                embedding_vec = np.array(new_vectors[idx])

                if key_str in need_embed_new:
                    # 对于“新内容”，写入前加锁 + 二次检查，确保并发下只会插一次
                    async with self._index_lock:
                        if key_str not in self.content_to_index_map:
                            # 仍不存在 -> 确认插入
                            self._add_entry(representative_payload[key_str], embedding_vec, key_str)
                            stored_vec = embedding_vec.tolist()
                        else:
                            # 已被其他协程抢先插入
                            faiss_id = self.content_to_index_map[key_str]
                            if isinstance(self.faiss_index, faiss.IndexFlat):
                                stored_vec = self.faiss_index.reconstruct(faiss_id).tolist()
                            else:
                                # 无法/不便重构，就用刚算出的向量返回即可（不再写入）
                                stored_vec = embedding_vec.tolist()
                    # 回填所有对应位置
                    for pos in positions_by_key.get(key_str, []):
                        final_embeddings[pos] = stored_vec
                else:
                    # 对于“库里已有但为了返回而重算”的情形，仅回填，不写入
                    computed_vec = embedding_vec.tolist()
                    for pos in positions_by_key.get(key_str, []):
                        # 若之前已因同批次其他位置命中缓存而填过，这里不覆盖
                        if final_embeddings[pos] is None:
                            final_embeddings[pos] = computed_vec

        # 过滤掉 None（那些被跳过或不含目标字段的）
        return [emb for emb in final_embeddings if emb is not None]

    def filter_payloads_and_embeddings(self, query_criteria: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[List[float]]]:
        """
        Filters stored payloads based on specified criteria and returns matching payloads and their embeddings.

        Args:
            query_criteria (Dict[str, Any]): Dict of key-value pairs to match against payloads.

        Returns:
            Tuple[List[Dict[str, Any]], List[List[float]]]
        """
        matching_payloads: List[Dict[str, Any]] = []
        matching_embeddings: List[List[float]] = []

        for faiss_id, payload_data in self.payload_map.items():
            is_match = True
            for key, value in query_criteria.items():
                if key not in payload_data or payload_data[key] != value:
                    is_match = False
                    break

            if is_match:
                matching_payloads.append(payload_data)
                if isinstance(self.faiss_index, faiss.IndexFlat):
                    vec = self.faiss_index.reconstruct(faiss_id).tolist()
                    matching_embeddings.append(vec)
                else:
                    print(f"Warning: Cannot efficiently retrieve embedding for FAISS ID {faiss_id} with non-Flat index during filtering.")

        return matching_payloads, matching_embeddings

    def persist_to_disk(self):
        """
        Persists the FAISS index and all payload files to disk.
        Saves the store's configuration (without unique_id_field).
        """
        faiss.write_index(self.faiss_index, self.index_file_path)
        with open(self.payload_file_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in self.payload_map.items()}, f, ensure_ascii=False, indent=4)

        # Save configuration
        config_data = {
            "embedding_dim": self.embedding_dim,
            "faiss_index_type": "Flat",
            "persist_every_n": self.persist_every_n,
        }
        with open(self.config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Finish persisting FAISS index to {self.index_file_path} and payloads to {self.payload_file_path}")
        self._since_last_persist = 0


    def _load_from_disk_if_exists(self):
        """
        Loads the FAISS index and payload files from disk if files exist.
        Reconstructs content_to_index_map using '__vector_source_content__'.
        """
        if os.path.exists(self.index_file_path) and os.path.exists(self.payload_file_path) and os.path.exists(self.config_file_path):
            print(f"Loading FAISS index from {self.index_file_path} and payloads from {self.payload_file_path}...")

            # Load config
            with open(self.config_file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                if config_data.get("embedding_dim") != self.embedding_dim:
                    print(f"Warning: Loaded embedding_dim ({config_data.get('embedding_dim')}) differs from initialized ({self.embedding_dim}).")
                if "persist_every_n" in config_data and self.persist_every_n is None:
                    self.persist_every_n = config_data["persist_every_n"]

            self.faiss_index = faiss.read_index(self.index_file_path)
            with open(self.payload_file_path, 'r', encoding='utf-8') as f:
                loaded_payload_map = json.load(f)
                self.payload_map = {int(k): v for k, v in loaded_payload_map.items()}

            # Rebuild content_to_index_map from stored payloads
            self.content_to_index_map = {}
            for faiss_id, payload_data in self.payload_map.items():
                content_val = payload_data.get(self._VECTOR_SOURCE_CONTENT_KEY)
                if content_val is not None:
                    self.content_to_index_map[str(content_val)] = faiss_id
                else:
                    # 兼容旧数据：如果旧payload没有该字段，可选择忽略或回填
                    raise NotImplementedError
            self._since_last_persist = 0
            logger.info(f"[FAISS] Load from disk -> dir={self.vector_store_root_dir}")
        else:
            logger.info(f"[FAISS] Initialize from empty -> dir={self.vector_store_root_dir}")

async def main_case():
    from config.singleton import DEFAULT_EMBEDDER
    import tempfile

    # embedding函数 (async)
    default_embedding = DEFAULT_EMBEDDER.embed

    # 测试数据
    payloads_a = [
        {"id": "doc1", "title": "Apple Inc.",       "text_content": "Apple is a tech company known for iPhones."},
        {"id": "doc2", "title": "Banana Republic",  "text_content": "Banana is a type of fruit, yellow in color."},
        {"id": "doc3", "title": "Orange Juice",     "text_content": "Orange is a citrus fruit, rich in Vitamin C."},
        {"id": "doc4", "title": "Grape Vine",       "text_content": "Grape is a fruit that grows in clusters."},
    ]
    payloads_b = [
        {"id": "doc2",     "title": "Banana Fresh",     "text_content": "Banana is a type of fruit, yellow in color."},  # 重复内容
        {"id": "doc1",     "title": "Apple Store",      "text_content": "Apple is a tech company known for iPhones."},   # 重复内容
        {"id": "doc5",     "title": "Kiwi Fruit",       "text_content": "Kiwi is a small, oval fruit with fuzzy skin."}, # 新内容
        {"id": "doc3_new", "title": "Orange Marmalade", "text_content": "Orange is a citrus fruit, rich in Vitamin C."}, # 重复内容（新ID）
        {"id": "doc6",     "title": "Pineapple",        "text_content": "A tropical fruit."},                             # 新内容
    ]

    # 用临时目录
    store_dir = "./test"

    # 1) 初始化（新版无需 unique_id_field；必须传入根目录）
    vs = await FaissVectorStore.create(
        vector_store_root_dir=store_dir,
        embedder=default_embedding,
    )

    # 2) 首次写入（全部新“内容”）
    await vs.get_or_create_embeddings(payloads_a, vector_key_name="text_content")

    # 去重依据是“内容文本”而非ID；快速 sanity check（Apple 文本应存在）
    apple_text = "Apple is a tech company known for iPhones."
    assert apple_text in vs.content_to_index_map
    apple_faiss_id = vs.content_to_index_map[apple_text]
    assert vs.payload_map[apple_faiss_id]["title"] == "Apple Inc."

    # 3) 二次写入（含重复内容与新内容）
    await vs.get_or_create_embeddings(payloads_b, vector_key_name="text_content")

    # “内容级”去重：重复文本不应新增条目；仅新增 Kiwi 与 Pineapple 两个“新内容”
    # 首批 4 个内容 + 二批 2 个新内容 = 6
    assert vs.faiss_index.ntotal == 6
    assert len(vs.payload_map) == 6
    assert len(vs.content_to_index_map) == 6

    # 新ID若内容重复不会新增payload；确认不存在以 doc3_new 为id的payload
    all_ids = {p.get("id") for p in vs.payload_map.values()}
    assert "doc3_new" not in all_ids
    # 但应包含真正新增内容的ID
    assert "doc5" in all_ids and "doc6" in all_ids

    # 4) 持久化并重载实例
    vs.persist_to_disk()
    vs2 = await FaissVectorStore.create(
        vector_store_root_dir=store_dir,
        embedder=default_embedding,
    )

    # 5) 过滤检索（按任意字段组合）
    # by (title & id) — 原始 Orange
    p1, e1 = vs2.filter_payloads_and_embeddings({"title": "Orange Juice", "id": "doc3"})
    assert len(p1) == 1 and p1[0]["id"] == "doc3" and len(e1) == 1

    # by (text_content) — Banana 的文本只保留了一份（最初写入的 payload）
    banana_text = "Banana is a type of fruit, yellow in color."
    p2, e2 = vs2.filter_payloads_and_embeddings({"text_content": banana_text})
    assert len(p2) == 1 and p2[0]["id"] == "doc2" and len(e2) == 1

    # by (id & title) — 新增且唯一内容：Kiwi
    p3, e3 = vs2.filter_payloads_and_embeddings({"id": "doc5", "title": "Kiwi Fruit"})
    assert len(p3) == 1 and p3[0]["id"] == "doc5" and len(e3) == 1

    # miss
    p4, e4 = vs2.filter_payloads_and_embeddings({"title": "NonExistent Title"})
    assert len(p4) == 0 and len(e4) == 0

    print(
        "✅ content-dedupe case passed:",
        f"unique_contents={len(vs2.content_to_index_map)}",
        f"faiss_ntotal={vs2.faiss_index.ntotal}"
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main_case())