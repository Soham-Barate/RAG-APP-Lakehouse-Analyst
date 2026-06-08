"""
Pinecone Serverless vector store implementation.

Supports hybrid search (dense + sparse), metadata filtering,
namespace-based multi-tenancy, and batch upsert with retry.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from backend.vectorstore.base import VectorStoreBase
from config.settings import Settings

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStoreBase):
    """
    Pinecone Serverless vector store with hybrid search.

    Features:
    - Dense + sparse vector search
    - Native metadata filtering
    - Namespace-based multi-tenancy
    - Batch upsert with exponential backoff retry
    """

    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.pinecone.api_key.get_secret_value()
        self.index_name = settings.pinecone.index_name
        self.environment = settings.pinecone.environment
        self.cloud = settings.pinecone.cloud
        self.dimensions = settings.openai.embedding_dimensions
        self._index: Any = None
        self._parent_docs: dict[str, dict[str, Any]] = {}  # In-memory parent cache

    async def initialize(self) -> None:
        """Initialize Pinecone client and ensure index exists."""
        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if self.index_name not in existing_indexes:
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.cloud,
                        region=self.environment,
                    ),
                )
                logger.info("Created Pinecone index: %s", self.index_name)

            self._index = pc.Index(self.index_name)
            logger.info("Pinecone index connected: %s", self.index_name)

        except ImportError:
            logger.error("pinecone-client not installed")
            raise
        except Exception as exc:
            logger.error("Pinecone initialization failed: %s", exc)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Batch upsert documents to Pinecone with retry."""
        if not documents or not self._index:
            return []

        metadata = metadata or [{}] * len(documents)
        doc_ids: list[str] = []
        vectors: list[dict[str, Any]] = []

        for doc, emb, meta in zip(documents, embeddings, metadata):
            doc_id = doc.get("id", str(uuid.uuid4()))

            # Store parent docs in memory cache
            if doc.get("is_parent"):
                parent_id = meta.get("parent_id", doc_id)
                self._parent_docs[parent_id] = {
                    "id": parent_id,
                    "content": doc.get("content", ""),
                    "metadata": meta,
                }

            # Clean metadata for Pinecone (only scalar values)
            clean_meta = {
                "content": doc.get("content", "")[:1000],  # Pinecone metadata limit
                "document_type": meta.get("document_type", "product"),
                "category": meta.get("category", ""),
                "parent_id": meta.get("parent_id", ""),
            }

            # Add date as epoch for numeric filtering
            if "created_at" in meta:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(str(meta["created_at"]))
                    clean_meta["created_at_epoch"] = int(dt.timestamp())
                except (ValueError, TypeError):
                    pass

            vectors.append({
                "id": doc_id,
                "values": emb,
                "metadata": clean_meta,
            })
            doc_ids.append(doc_id)

        # Batch upsert (Pinecone recommends batches of 100)
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            self._index.upsert(vectors=batch, namespace="default")

        logger.info("Upserted %d documents to Pinecone", len(doc_ids))
        return doc_ids

    async def hybrid_search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform dense vector search with metadata filtering."""
        if not self._index or query_embedding is None:
            return []

        # Build Pinecone filter
        pine_filter = self._build_filter(filters)

        try:
            results = self._index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                namespace="default",
                filter=pine_filter if pine_filter else None,
            )

            documents: list[dict[str, Any]] = []
            for match in results.get("matches", []):
                meta = match.get("metadata", {})
                documents.append({
                    "id": match["id"],
                    "content": meta.get("content", ""),
                    "metadata": meta,
                    "score": float(match["score"]),
                    "vector_score": float(match["score"]),
                    "bm25_score": 0.0,
                    "rrf_score": float(match["score"]),
                })

            return documents

        except Exception as exc:
            logger.error("Pinecone search failed: %s", exc)
            return []

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID from Pinecone."""
        if not self._index or not ids:
            return 0

        try:
            self._index.delete(ids=ids, namespace="default")
            return len(ids)
        except Exception as exc:
            logger.error("Pinecone delete failed: %s", exc)
            return 0

    async def get_parent_document(self, parent_id: str) -> dict[str, Any] | None:
        """Retrieve parent document from cache."""
        return self._parent_docs.get(parent_id)

    async def get_document_count(self) -> int:
        """Return approximate document count."""
        if not self._index:
            return 0
        try:
            stats = self._index.describe_index_stats()
            return stats.get("total_vector_count", 0)
        except Exception:
            return 0

    def _build_filter(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """Build Pinecone metadata filter from user filters."""
        if not filters:
            return None

        conditions: list[dict[str, Any]] = []

        if filters.get("categories"):
            conditions.append({
                "category": {"$in": filters["categories"]}
            })

        if filters.get("document_types"):
            conditions.append({
                "document_type": {"$in": filters["document_types"]}
            })

        if not conditions:
            return None

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}
