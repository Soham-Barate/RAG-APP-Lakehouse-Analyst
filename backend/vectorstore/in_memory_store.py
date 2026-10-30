"""
In-memory vector store for local development and fallback mode.
Provides basic cosine similarity search and keyword matching without requiring PostgreSQL/pgvector.
"""

from __future__ import annotations

import math
from typing import Any

from backend.vectorstore.base import VectorStoreBase


class InMemoryVectorStore(VectorStoreBase):
    """Simple in-memory vector store for development fallback."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.parents: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        pass

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]],
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        ids = []
        for i, doc in enumerate(documents):
            doc_id = doc.get("id", f"inmem_{len(self.documents)}")
            emb = embeddings[i] if i < len(embeddings) else []
            meta = metadata[i] if metadata and i < len(metadata) else doc.get("metadata", {})
            entry = {
                "id": doc_id,
                "content": doc.get("content", ""),
                "embedding": emb,
                "metadata": meta,
            }
            self.documents.append(entry)
            ids.append(doc_id)
            if "parent_id" in meta:
                self.parents[meta["parent_id"]] = entry
        return ids

    async def hybrid_search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        results = []
        q_words = set(query_text.lower().split())

        for doc in self.documents:
            # Check filters
            if filters:
                doc_meta = doc.get("metadata", {})
                match = True
                for k, v in filters.items():
                    if doc_meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            score = 0.0
            # Lexical overlap
            c_words = set(doc["content"].lower().split())
            if q_words and c_words:
                overlap = len(q_words & c_words)
                score += (overlap / len(q_words)) * 0.4

            # Vector cosine similarity
            if query_embedding and doc.get("embedding"):
                doc_emb = doc["embedding"]
                dot = sum(a * b for a, b in zip(query_embedding, doc_emb))
                norm_q = math.sqrt(sum(a * a for a in query_embedding))
                norm_d = math.sqrt(sum(b * b for b in doc_emb))
                if norm_q > 0 and norm_d > 0:
                    sim = dot / (norm_q * norm_d)
                    score += max(0.0, sim) * 0.6

            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "metadata": doc["metadata"],
                "score": round(score, 4),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    async def delete_documents(self, ids: list[str]) -> int:
        initial = len(self.documents)
        id_set = set(ids)
        self.documents = [d for d in self.documents if d["id"] not in id_set]
        return initial - len(self.documents)

    async def get_parent_document(self, parent_id: str) -> dict[str, Any] | None:
        return self.parents.get(parent_id)

    async def get_document_count(self) -> int:
        return len(self.documents)

    async def close(self) -> None:
        self.documents.clear()
        self.parents.clear()

