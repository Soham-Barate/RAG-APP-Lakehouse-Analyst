"""
Retriever tests.
"""

from __future__ import annotations

from typing import Any

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
        """Verify RRF correctly fuses multiple independent query ranked lists."""
        list1 = [
            {"id": "doc1", "content": "doc1", "score": 0.9},
            {"id": "doc2", "content": "doc2", "score": 0.8},
        ]
        list2 = [
            {"id": "doc1", "content": "doc1", "score": 0.7},
            {"id": "doc3", "content": "doc3", "score": 0.6},
        ]

        fused = retriever._reciprocal_rank_fusion([list1, list2], k=60)
        # doc1 appears in both lists at rank 1: score = 1/61 + 1/61 = 2/61 ≈ 0.0327869
        # doc2 appears only in list1 at rank 2: score = 1/62 ≈ 0.016129
        # doc3 appears only in list2 at rank 2: score = 1/62 ≈ 0.016129
        assert fused[0]["id"] == "doc1"
        assert fused[0]["rrf_score"] > fused[1]["rrf_score"]
        expected_doc1 = (1.0 / 61) + (1.0 / 61)
        assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == expected_doc1
        assert pytest.approx(fused[1]["rrf_score"], rel=1e-5) == 1.0 / 62

    def test_rrf_deduplication_within_same_list(self, retriever):
        """Verify that duplicate documents in the SAME list do NOT inflate RRF scores."""
        list_with_dup = [
            {"id": "doc1", "content": "first appearance", "score": 0.9},
            {"id": "doc2", "content": "second doc", "score": 0.8},
            {"id": "doc1", "content": "duplicate appearance", "score": 0.5},
        ]
        fused = retriever._reciprocal_rank_fusion([list_with_dup], k=60)
        # doc1 should only be counted once (at rank 1), not twice
        assert fused[0]["id"] == "doc1"
        assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == 1.0 / 61
        assert pytest.approx(fused[1]["rrf_score"], rel=1e-5) == 1.0 / 62
        assert len(fused) == 2

    def test_rrf_rank_reversals_and_ties(self, retriever):
        """Verify that cumulative rank across lists correctly reverses single-list ranking."""
        list1 = [{"id": "winner_q1"}, {"id": "consensus_doc"}]
        list2 = [{"id": "winner_q2"}, {"id": "consensus_doc"}]
        list3 = [{"id": "consensus_doc"}, {"id": "winner_q3"}]

        fused = retriever._reciprocal_rank_fusion([list1, list2, list3], k=60)
        # consensus_doc: rank 2 in list1 (1/62) + rank 2 in list2 (1/62) + rank 1 in list3 (1/61) ≈ 0.04865
        # winner_q1: rank 1 in list1 (1/61) ≈ 0.01639
        assert fused[0]["id"] == "consensus_doc"
        expected_consensus = (1.0 / 62) + (1.0 / 62) + (1.0 / 61)
        assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == expected_consensus

    def test_rrf_independent_multi_query_ranks(self, retriever):
        # Two queries producing their own top results
        query1_results = [
            {"id": "doc_q1_top", "content": "query 1 top hit"},
            {"id": "shared_doc", "content": "appears in both"},
        ]
        query2_results = [
            {"id": "doc_q2_top", "content": "query 2 top hit"},
            {"id": "shared_doc", "content": "appears in both"},
        ]

        fused = retriever._reciprocal_rank_fusion([query1_results, query2_results], k=60)
        # Both top docs should have identical rank 1 score: 1/(60+1)
        # shared_doc appears at rank 2 in both: 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.032258
        # top docs: 1/61 ≈ 0.01639
        # So shared_doc should rank 1st overall!
        assert fused[0]["id"] == "shared_doc"
        expected_shared_score = (1.0 / 62) + (1.0 / 62)
        assert pytest.approx(fused[0]["rrf_score"], rel=1e-5) == expected_shared_score

        top_docs = {fused[1]["id"], fused[2]["id"]}
        assert top_docs == {"doc_q1_top", "doc_q2_top"}
        assert pytest.approx(fused[1]["rrf_score"], rel=1e-5) == 1.0 / 61
        assert pytest.approx(fused[2]["rrf_score"], rel=1e-5) == 1.0 / 61

    def test_rrf_empty_inputs(self, retriever):
        assert retriever._reciprocal_rank_fusion([]) == []
        assert retriever._reciprocal_rank_fusion([[]]) == []

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
