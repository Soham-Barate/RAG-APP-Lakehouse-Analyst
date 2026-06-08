"""
RAG pipeline node tests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.rag.nodes.grader import RelevanceGrader
from backend.rag.nodes.hyde import HyDEGenerator
from backend.rag.nodes.query_rewriter import QueryRewriter
from backend.rag.nodes.router import QueryRouter


class TestQueryRouter:
    """Tests for query routing."""

    @pytest.mark.asyncio
    async def test_routes_to_hybrid_by_default(self, mock_llm_service):
        router = QueryRouter(llm_service=mock_llm_service)
        state = {"query": "Tell me about electronics products"}

        result = await router(state)
        assert "route" in result
        assert result["route"] in ("sql", "vector", "hybrid")

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM down"))

        router = QueryRouter(llm_service=llm)
        result = await router({"query": "test"})
        assert result["route"] == "hybrid"  # Default fallback


class TestQueryRewriter:
    """Tests for query rewriting."""

    @pytest.mark.asyncio
    async def test_produces_multiple_queries(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "rewritten_queries": [
                    "electronics products top rated",
                    "best electronics customer reviews",
                ],
                "strategy": "expansion",
            })
        )

        rewriter = QueryRewriter(llm_service=mock_llm_service)
        result = await rewriter({"query": "best electronics", "iteration_count": 0})

        assert "rewritten_queries" in result
        assert len(result["rewritten_queries"]) >= 2

    @pytest.mark.asyncio
    async def test_includes_original_query(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "rewritten_queries": ["expanded query"],
                "strategy": "expansion",
            })
        )

        rewriter = QueryRewriter(llm_service=mock_llm_service)
        result = await rewriter({"query": "original query", "iteration_count": 0})

        assert "original query" in result["rewritten_queries"]


class TestHyDEGenerator:
    """Tests for HyDE generation."""

    @pytest.mark.asyncio
    async def test_generates_hypothetical_document(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value="The TechPro Earbuds feature active noise cancellation..."
        )

        hyde = HyDEGenerator(llm_service=mock_llm_service)
        result = await hyde({"query": "Tell me about TechPro Earbuds"})

        assert "hyde_document" in result
        assert len(result["hyde_document"]) > 0

    @pytest.mark.asyncio
    async def test_handles_failure_gracefully(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        hyde = HyDEGenerator(llm_service=llm)
        result = await hyde({"query": "test"})
        assert result["hyde_document"] == ""


class TestRelevanceGrader:
    """Tests for relevance grading."""

    @pytest.mark.asyncio
    async def test_proceeds_when_relevant(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "relevance": "relevant",
                "score": 0.85,
                "reasoning": "Directly answers the query",
            })
        )

        grader = RelevanceGrader(
            llm_service=mock_llm_service,
            relevance_threshold=0.4,
        )

        state: dict[str, Any] = {
            "query": "test",
            "reranked_docs": [
                {"content": "relevant doc", "rerank_score": 0.8},
            ],
            "iteration_count": 0,
        }

        result = await grader(state)
        assert result["should_retry"] is False

    @pytest.mark.asyncio
    async def test_retries_when_irrelevant(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "relevance": "irrelevant",
                "score": 0.1,
                "reasoning": "Not related",
            })
        )

        grader = RelevanceGrader(
            llm_service=mock_llm_service,
            relevance_threshold=0.4,
            max_iterations=3,
        )

        state: dict[str, Any] = {
            "query": "test",
            "reranked_docs": [
                {"content": "irrelevant doc", "rerank_score": 0.1},
            ],
            "iteration_count": 0,
        }

        result = await grader(state)
        assert result["should_retry"] is True
        assert result["iteration_count"] == 1
