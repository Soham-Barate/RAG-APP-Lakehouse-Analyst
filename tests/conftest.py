"""
Shared pytest fixtures for all test modules.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Mock LLM service that returns canned responses."""
    service = MagicMock()
    service.last_token_count = 100
    service.generate = AsyncMock(
        return_value='{"route": "hybrid", "reasoning": "general query"}'
    )
    service.stream = AsyncMock()
    return service


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    """Mock embedding service with consistent vectors."""
    service = MagicMock()
    service.dimensions = 3072
    service.embed_query = AsyncMock(return_value=[0.1] * 3072)
    service.embed_documents = AsyncMock(
        return_value=[[0.1] * 3072]
    )
    return service


@pytest.fixture
def mock_vector_store() -> MagicMock:
    """Mock vector store with canned search results."""
    store = MagicMock()
    store.hybrid_search = AsyncMock(
        return_value=[
            {
                "id": "doc_1",
                "content": "TechPro Wireless Earbuds — premium earbuds with ANC",
                "metadata": {
                    "document_type": "product",
                    "category": "Electronics",
                },
                "score": 0.85,
                "vector_score": 0.85,
                "bm25_score": 0.72,
                "rrf_score": 0.85,
            },
            {
                "id": "doc_2",
                "content": "Customer review: Great earbuds, excellent sound quality",
                "metadata": {
                    "document_type": "review",
                    "category": "Electronics",
                },
                "score": 0.75,
                "vector_score": 0.75,
                "bm25_score": 0.60,
                "rrf_score": 0.75,
            },
        ]
    )
    store.add_documents = AsyncMock(return_value=["doc_1", "doc_2"])
    store.delete_documents = AsyncMock(return_value=2)
    store.get_parent_document = AsyncMock(return_value=None)
    store.get_document_count = AsyncMock(return_value=100)
    store.initialize = AsyncMock()
    return store


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    redis = MagicMock()
    redis.ping = AsyncMock()
    redis.hgetall = AsyncMock(return_value={})
    redis.hset = AsyncMock()
    redis.hget = AsyncMock(return_value=None)
    redis.hlen = AsyncMock(return_value=0)
    redis.expire = AsyncMock()
    redis.delete = AsyncMock()
    redis.close = AsyncMock()
    redis.pipeline = MagicMock()
    return redis


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """Sample documents for testing."""
    return [
        {
            "id": "prod_001",
            "content": (
                "TechPro Wireless Earbuds — premium wireless earbuds with "
                "active noise cancellation, 30-hour battery life."
            ),
            "metadata": {
                "document_type": "product",
                "category": "Electronics",
                "price": 79.99,
            },
        },
        {
            "id": "rev_001",
            "content": (
                "Love these earbuds! Great sound quality and the ANC is "
                "excellent. Battery lasts as advertised."
            ),
            "metadata": {
                "document_type": "review",
                "category": "Electronics",
                "rating": 5,
            },
        },
        {
            "id": "tkt_001",
            "content": (
                "My earbuds stopped charging after 2 weeks. The left earbud "
                "doesn't connect anymore."
            ),
            "metadata": {
                "document_type": "ticket",
                "category": "Electronics",
                "priority": "high",
            },
        },
    ]


@pytest.fixture
def sample_rag_state() -> dict[str, Any]:
    """Sample RAG pipeline state for testing."""
    return {
        "query": "What are customers saying about the TechPro Earbuds?",
        "conversation_id": "test-conv-1",
        "metadata_filters": {},
        "max_results": 5,
        "messages": [],
        "iteration_count": 0,
        "route": "hybrid",
        "rewritten_queries": [
            "TechPro Earbuds customer reviews feedback",
            "wireless earbuds customer satisfaction sentiment",
        ],
        "hyde_document": "",
        "retrieved_docs": [],
        "reranked_docs": [],
        "compressed_context": "",
        "relevance_scores": [],
        "answer": "",
        "sources": [],
        "confidence_score": 0.0,
        "should_retry": False,
    }
