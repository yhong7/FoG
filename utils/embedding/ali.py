from openai import AsyncOpenAI
from .base_embedder import BaseEmbedder
from config.settings import CONFIG
from typing import List, Literal
from .base_embedder import ConnectionConfig


class Default_Embedder(BaseEmbedder):
    def __init__(
        self,
        connect_config:ConnectionConfig,

        batch_size: int = 10,
        max_concurrency: int = 10,
        **kwargs,
    ):
        super().__init__(batch_size, max_concurrency, **kwargs)
        base_url = connect_config.base_url
        api_key = connect_config.api_key

        self.model_name = connect_config.model_name
        self.embedding_args = connect_config.embedding_args

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )


    async def _embed(self, texts: List[str]) -> List[List[float]]:
        response = await self.client.embeddings.create(
            model=self.model_name,
            input=texts,
            **self.embedding_args
        )
        return [item.embedding for item in response.data]