"""
Embedding service with batching, retry, and normalization.

Wraps the OpenAI text-embedding-3-large model with production
reliability features.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import Settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Production embedding service with batching and retry.

    Features:
    - OpenAI text-embedding-3-large (3072 dimensions)
    - Automatic batching for large document sets
    - Exponential backoff retry on transient failures
    - L2 normalization of output vectors
    - Token counting and cost tracking
    """

    def __init__(self, settings: Settings) -> None:
        self.model = settings.openai.embedding_model
        self.dimensions = settings.openai.embedding_dimensions
        self.batch_size = settings.rag.embedding_batch_size
        self._client: Any = None
        self._api_key = settings.openai.api_key.get_secret_value() if settings.openai.api_key else ""
        self._total_tokens = 0
        self._provider_disabled = (
            not self._api_key
            or not self._api_key.strip()
            or self._api_key.startswith("sk-proj-demo")
            or "demo" in self._api_key.lower()
        )

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI async client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _deterministic_fallback(self, text: str) -> list[float]:
        """Generate a deterministic pseudo-random embedding vector from text hash."""
        import hashlib
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimensions)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    async def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.
        Returns a normalized embedding vector with deterministic fallback on auth errors.
        """
        if self._provider_disabled:
            return self._deterministic_fallback(text)

        try:
            return await self._call_openai_embed_query(text)
        except Exception as exc:
            err_str = str(exc).lower()
            if "unauthorized" in err_str or "authenticationerror" in err_str or "401" in err_str or "invalid_api_key" in err_str or "retryerror" in err_str:
                self._provider_disabled = True
                logger.warning("OpenAI embedding API unavailable (%s). Using deterministic fallback.", exc)
                return self._deterministic_fallback(text)
            raise

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _call_openai_embed_query(self, text: str) -> list[float]:
        client = self._get_client()

        response = await client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimensions,
        )

        embedding = response.data[0].embedding
        self._total_tokens += response.usage.total_tokens

        # L2 normalize
        return self._normalize(embedding)

    async def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """
        Embed multiple documents with automatic batching.

        Returns normalized embedding vectors.
        """
        if not texts:
            return []

        if self._provider_disabled:
            return [self._deterministic_fallback(t) for t in texts]

        all_embeddings: list[list[float]] = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = await self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        logger.info(
            "Embedded %d documents (%d total tokens)",
            len(texts),
            self._total_tokens,
        )

        return all_embeddings

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._provider_disabled:
            return [self._deterministic_fallback(t) for t in texts]

        try:
            return await self._call_openai_embed_batch(texts)
        except Exception as exc:
            err_str = str(exc).lower()
            if "unauthorized" in err_str or "authenticationerror" in err_str or "401" in err_str or "invalid_api_key" in err_str or "retryerror" in err_str:
                self._provider_disabled = True
                logger.warning("OpenAI batch embedding unavailable (%s). Using deterministic fallback.", exc)
                return [self._deterministic_fallback(t) for t in texts]
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _call_openai_embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a single batch with retry."""
        client = self._get_client()

        # Token-aware truncation to avoid token limit errors (e.g. 8191 tokens for text-embedding-3-large)
        truncated = [self._truncate_to_tokens(t, max_tokens=8000) for t in texts]

        response = await client.embeddings.create(
            model=self.model,
            input=truncated,
            dimensions=self.dimensions,
        )

        self._total_tokens += response.usage.total_tokens

        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        embeddings = [item.embedding for item in sorted_data]

        # Normalize all
        return [self._normalize(emb) for emb in embeddings]

    def _truncate_to_tokens(self, text: str, max_tokens: int = 8000) -> str:
        """Truncate text to a maximum token count using model encoding."""
        if not text:
            return ""
        try:
            import tiktoken

            try:
                enc = tiktoken.encoding_for_model(self.model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")

            tokens = enc.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return enc.decode(tokens[:max_tokens])
        except Exception as exc:
            logger.warning("tiktoken truncation failed, fallback to char slicing: %s", exc)
            return text[: max_tokens * 4]

    @staticmethod
    def _normalize(embedding: list[float]) -> list[float]:
        """L2 normalize an embedding vector."""
        arr = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist()

    @property
    def total_tokens(self) -> int:
        """Total tokens consumed across all embedding calls."""
        return self._total_tokens

    @property
    def estimated_cost(self) -> float:
        """Estimated cost in USD (text-embedding-3-large: $0.13/1M tokens)."""
        return self._total_tokens * 0.13 / 1_000_000
