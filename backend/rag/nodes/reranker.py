"""
Cross-Encoder Reranker Node.

Reranks retrieved documents using Cohere Rerank API for precise
relevance scoring, with fallback to similarity-based reranking.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks documents using a cross-encoder model (Cohere Rerank).

    Cross-encoders process query-document pairs jointly, producing
    much more accurate relevance scores than bi-encoder (embedding)
    similarity alone.

    Fallback: If Cohere is unavailable, falls back to the existing
    RRF scores from the retriever.
    """

    def __init__(
        self,
        cohere_api_key: str = "",
        rerank_model: str = "rerank-english-v3.0",
        top_n: int = 5,
    ) -> None:
        self.cohere_api_key = cohere_api_key
        self.rerank_model = rerank_model
        self.top_n = top_n
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazy-initialize the Cohere client."""
        if self._client is None and self.cohere_api_key:
            try:
                import cohere

                self._client = cohere.AsyncClientV2(api_key=self.cohere_api_key)
            except ImportError:
                logger.warning("cohere package not installed, using fallback reranking")
        return self._client

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Rerank retrieved documents and return top-N."""
        start = time.perf_counter()

        docs = state.get("retrieved_docs", [])
        query = state["query"]

        if not docs:
            return {
                "reranked_docs": [],
                "reranking_latency_ms": 0.0,
                "documents_after_rerank": 0,
            }

        try:
            reranked = await self._cohere_rerank(query, docs)
        except Exception as exc:
            logger.warning("Cohere rerank failed, using fallback: %s", exc)
            reranked = self._fallback_rerank(docs)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Reranking: %d -> %d documents (%.1fms)",
            len(docs),
            len(reranked),
            latency_ms,
        )

        return {
            "reranked_docs": reranked,
            "reranking_latency_ms": round(latency_ms, 2),
            "documents_after_rerank": len(reranked),
        }

    async def _cohere_rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rerank using Cohere Rerank API."""
        client = self._get_client()
        if not client:
            return self._fallback_rerank(docs)

        # Prepare documents for Cohere
        doc_texts = [d.get("content", "") for d in docs]

        response = await client.rerank(
            model=self.rerank_model,
            query=query,
            documents=doc_texts,
            top_n=self.top_n,
        )

        reranked: list[dict[str, Any]] = []
        for result in response.results:
            doc = docs[result.index].copy()
            doc["rerank_score"] = result.relevance_score
            reranked.append(doc)

        return reranked

    def _fallback_rerank(
        self,
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fallback: sort by existing RRF score and take top-N."""
        sorted_docs = sorted(
            docs,
            key=lambda d: d.get("rrf_score", d.get("score", 0.0)),
            reverse=True,
        )

        reranked: list[dict[str, Any]] = []
        for doc in sorted_docs[: self.top_n]:
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = doc.get("rrf_score", doc.get("score", 0.5))
            reranked.append(doc_copy)

        return reranked
