import threading
from typing import TYPE_CHECKING

from utils.embedding.embedder_factory import EmbedderFactory
from utils.chat_model_factory import ChatModelFactory, ChatModel
from model.vocab.knowledge_graph_vocab import KnowledgeGraphVocab
from utils.embedding.base_embedder import BaseEmbedder
import os

import torch
from config.model_args import (
 QTR_HEAD_ARGS,
QTR_HEAD_CHECKPOINT, QTR_BODY_CHECKPOINT,
    ADAPTER_ARGS,
SCORER_ARGS,
CONV_CHECKPOINT,
QTR_BODY_ATT_ARGS, CONV_ATT_ARGS,
SCORER_CHECKPOINT
)
from model.qtr.qtr_models import QTRClassifierHead, QTRFeatureMapBodyAtt
from model.embedding_adapter import EmbeddingAdapter
from model.message_passing.conv import GraphMessagePassing
from model.message_passing.scorer import Scorer
from loguru import logger


__all__ = ["DEFAULT_EMBEDDER",
           "DEFAULT_CHAT_MODEL",
           "KG_VOCAB",
           "QTR_BODY_PRETRAIN",
           "QTR_HEAD_PRETRAIN",
           "QTR_BODY",
           "QTR_HEAD",
           "EMBEDDING_ADAPTER",
           "CONV",
           "SCORER"
           ]
__lazy_cache = {}
__lazy_lock = threading.RLock()

def __dir__():
    return sorted(list(globals().keys()) + __all__)

def _cache_and_export(name, value):
    with __lazy_lock:
        __lazy_cache[name] = value
        globals()[name] = value
    return value


def __getattr__(name):
    # 已缓存则直接返回
    with __lazy_lock:
        if name in __lazy_cache:
            return __lazy_cache[name]


    if name == "DEFAULT_EMBEDDER":
        """
        common methods:
            await DEFAULT_EMBEDDER.embed(List[str]) -> embeddings: torch.float32
        """
        embedder = EmbedderFactory.create(
            "custom_embedder",
            max_concurrency=30,
            max_retries=1,
            max_rate=25,
            batch_size=10,
            cache_persist_every=512,
            cache_persist_dir = './faiss_cache/embedder',
        )
        return _cache_and_export(name, embedder)

    if name == "DEFAULT_CHAT_MODEL":
        chat_model = ChatModelFactory.create(
            llm_engine="custom_llm",
            # llm_engine="ali_qwen3",
            temperature=0.0,
            max_tokens=8192,
            stream=False,
            enable_thinking=False,
            timeout=3000,
        )
        return _cache_and_export(name, chat_model)

    if name == "KG_VOCAB":
        inst = KnowledgeGraphVocab()
        return _cache_and_export(name, inst)

    elif name == "QTR_BODY":
        qtr_body = QTRFeatureMapBodyAtt(**QTR_BODY_ATT_ARGS)
        if os.path.exists(QTR_BODY_CHECKPOINT):
            logger.info(f"loading QTR_BODY from {QTR_BODY_CHECKPOINT}")
            state = torch.load(QTR_BODY_CHECKPOINT)
            qtr_body.load_state_dict(state["model_state_dict"], strict=True)
        else:
            logger.info("initializing QTR_BODY")
        return _cache_and_export(name, qtr_body)

    elif name == "QTR_HEAD":
        qtr_head = QTRClassifierHead(**QTR_HEAD_ARGS)
        if os.path.exists(QTR_HEAD_CHECKPOINT):
            logger.info(f"loading QTR_HEAD from {QTR_HEAD_CHECKPOINT}")
            state = torch.load(QTR_HEAD_CHECKPOINT)
            qtr_head.load_state_dict(state["model_state_dict"], strict=True)
        else:
            logger.info("initializing QTR_HEAD")
        return _cache_and_export(name, qtr_head)

    elif name == "CONV":
        conv = GraphMessagePassing(**CONV_ATT_ARGS)
        if os.path.exists(CONV_CHECKPOINT):
            logger.info(f"loading CONV from {CONV_CHECKPOINT}")
            state = torch.load(CONV_CHECKPOINT)
            conv.load_state_dict(state["model_state_dict"], strict=True)
        else:
            logger.info("initializing CONV")

        return _cache_and_export(name, conv)

    elif name == "SCORER":
        scorer = Scorer(**SCORER_ARGS)
        if os.path.exists(SCORER_CHECKPOINT):
            logger.info(f"loading SCORER from {SCORER_CHECKPOINT}")
            state = torch.load(SCORER_CHECKPOINT)
            scorer.load_state_dict(state["model_state_dict"], strict=True)
        else:
            logger.info("initializing SCORER")
        return _cache_and_export(name, scorer)

    elif name == "EMBEDDING_ADAPTER":
        inst = EmbeddingAdapter(**ADAPTER_ARGS)
        return _cache_and_export(name, inst)

    raise AttributeError(f"module {__name__} has no attribute {name}")


from typing import Any

if TYPE_CHECKING:
    DEFAULT_EMBEDDER: BaseEmbedder
    DEFAULT_CHAT_MODEL: ChatModel
    KG_VOCAB: KnowledgeGraphVocab
    QTR_BODY_PRETRAIN: QTRFeatureMapBodyAtt
    QTR_HEAD_PRETRAIN: QTRClassifierHead
    EMBEDDING_ADAPTER: EmbeddingAdapter
    CONV: GraphMessagePassing
    SCORER: Scorer
    QTR_BODY: QTRFeatureMapBodyAtt
    QTR_HEAD: QTRClassifierHead


