"""
Service protocols (interfaces) for dependency inversion.

Using Python's ``Protocol`` (structural subtyping) so that any class
implementing the right methods is accepted — no explicit inheritance
required. This lets us swap implementations (e.g., pgvector ↔ Pinecone)
without changing consuming code.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Vector Store
# ---------------------------------------------------------------------------
@runtime_checkable
class VectorStoreProtocol(Protocol):
    """Interface for vector store backends."""

    async def initialize(self) -> None: ...

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]: ...

    async def hybrid_search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def delete_documents(self, ids: list[str]) -> int: ...

    async def get_document_count(self) -> int: ...


# ---------------------------------------------------------------------------
# Embedding Service
# ---------------------------------------------------------------------------
@runtime_checkable
class EmbeddingServiceProtocol(Protocol):
    """Interface for embedding generation."""

    dimensions: int

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def total_tokens(self) -> int: ...


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------
@runtime_checkable
class LLMServiceProtocol(Protocol):
    """Interface for LLM generation."""

    last_token_count: int

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> str: ...

    async def stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]: ...

    @property
    def total_tokens(self) -> int: ...


# ---------------------------------------------------------------------------
# Semantic Cache
# ---------------------------------------------------------------------------
@runtime_checkable
class SemanticCacheProtocol(Protocol):
    """Interface for semantic caching."""

    async def get(self, query: str) -> dict[str, Any] | None: ...

    async def set(self, query: str, response: dict[str, Any]) -> None: ...

    async def invalidate(self) -> None: ...

    async def get_stats(self) -> dict[str, Any]: ...
