"""
Abstract base class for vector store implementations.

Provides a unified interface so the RAG pipeline is agnostic to
whether pgvector, Pinecone, or another backend is used.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class VectorStoreBase(ABC):
    """Abstract interface for vector store backends."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize connections, create tables/indexes if needed."""
        ...

    @abstractmethod
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """
        Add documents with their embeddings to the store.

        Returns list of document IDs.
        """
        ...

    @abstractmethod
    async def hybrid_search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform hybrid (vector + keyword) search.

        Returns list of document dicts with 'id', 'content', 'metadata', 'score'.
        """
        ...

    @abstractmethod
    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID. Returns number deleted."""
        ...

    @abstractmethod
    async def get_parent_document(self, parent_id: str) -> dict[str, Any] | None:
        """Retrieve a parent document by its ID."""
        ...

    @abstractmethod
    async def get_document_count(self) -> int:
        """Return total number of documents in the store."""
        ...

    async def close(self) -> None:
        """Cleanup / close connections."""
        pass
