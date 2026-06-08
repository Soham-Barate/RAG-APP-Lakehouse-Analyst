"""
Retriever tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.rag.nodes.retriever import HybridRetriever


@pytest.fixture
def retriever(mock_vector_store, mock_embedding_service):
    return HybridRetriever(
        vector_store=mock_vector_store,
        embedding_service=mock_embedding_service,
        top_k=10,
    )


class TestHybridRetriever:
    """Tests for the hybrid retriever."""

    @pytest.mark.asyncio
    async def test_retrieves_documents(self, retriever):
        state: dict[str, Any] = {
            "query": "wireless earbuds",
            "rewritten_queries": ["wireless earbuds", "bluetooth earphones"],
            "hyde_document": "",
            "metadata_filters": {},
        }

        result = await retriever(state)
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) > 0
        assert result["documents_retrieved"] > 0
        assert result["retrieval_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_uses_hyde_document(self, retriever, mock_vector_store):
        state: dict[str, Any] = {
            "query": "test",
            "rewritten_queries": ["test"],
            "hyde_document": "A hypothetical answer about earbuds",
            "metadata_filters": {},
        }

        await retriever(state)
        # Should have called hybrid_search at least twice (query + hyde)
        assert mock_vector_store.hybrid_search.call_count >= 2

    def test_rrf_fusion(self, retriever):
        results = [
            {"id": "doc1", "content": "doc1", "score": 0.9},
            {"id": "doc2", "content": "doc2", "score": 0.8},
            {"id": "doc1", "content": "doc1", "score": 0.7},  # Duplicate
        ]

        fused = retriever._reciprocal_rank_fusion(results)
        # doc1 should rank higher because it appears in multiple results
        assert fused[0]["id"] == "doc1"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]

    @pytest.mark.asyncio
    async def test_deduplicates_results(self, retriever):
        state: dict[str, Any] = {
            "query": "test",
            "rewritten_queries": ["test", "test again"],
            "hyde_document": "",
            "metadata_filters": {},
        }

        result = await retriever(state)
        doc_ids = [d["id"] for d in result["retrieved_docs"]]
        assert len(doc_ids) == len(set(doc_ids))  # No duplicates
