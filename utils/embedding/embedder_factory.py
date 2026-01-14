from .base_embedder import BaseEmbedder
from .ali import Default_Embedder
from config.settings import CONFIG
from .base_embedder import ConnectionConfig


class EmbedderFactory:
    @staticmethod
    def create(
            embedding_engine: str,
            batch_size: int = 10,
            max_concurrency: int = 10,
            max_retries: int = 0,
            max_rate: int = None,
            if_tqdm = True,
            **kwargs
    ) -> BaseEmbedder:

        connect_config = CONFIG['embedding'].get(embedding_engine)
        connect_config = ConnectionConfig(**connect_config)

        CustomizedEmbedder = Default_Embedder

        return CustomizedEmbedder(
            connect_config=connect_config,
            batch_size=batch_size,
            max_concurrency=max_concurrency,
            max_retries=max_retries,
            max_rate=max_rate,
            if_tqdm=if_tqdm,
            **kwargs
        )

