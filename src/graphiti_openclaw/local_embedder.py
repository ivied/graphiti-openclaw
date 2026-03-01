"""
Local embedding client for Graphiti using sentence-transformers.

No API keys needed — runs entirely on the local machine.
Uses all-MiniLM-L6-v2 (384 dims) by default — fast and good quality.
"""

import asyncio
from typing import Iterable

from sentence_transformers import SentenceTransformer

from graphiti_core.embedder.client import EmbedderClient


class LocalEmbedder(EmbedderClient):
    """
    Local embedder using sentence-transformers.
    
    Wraps SentenceTransformer model to implement Graphiti's EmbedderClient interface.
    Runs synchronous model.encode() in a thread pool to keep async compatibility.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None  # lazy load

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def create(
        self, input_data: str | list[str] | Iterable[int] | Iterable[Iterable[int]]
    ) -> list[float]:
        """Create embedding for a single input."""
        if isinstance(input_data, str):
            text = input_data
        elif isinstance(input_data, list) and input_data and isinstance(input_data[0], str):
            text = input_data[0]
        else:
            text = str(input_data)
        
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, normalize_embeddings=True)
        )
        return embedding.tolist()

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        """Create embeddings for a batch of inputs — more efficient than one-by-one."""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(input_data_list, normalize_embeddings=True)
        )
        return [emb.tolist() for emb in embeddings]
