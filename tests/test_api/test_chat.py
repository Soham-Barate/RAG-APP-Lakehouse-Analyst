"""
API endpoint integration tests.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client with mocked services."""
    # Patch settings to avoid .env dependency
    with patch("config.settings.get_settings") as mock_settings:
        settings = MagicMock()
        settings.app_name = "Test RAG"
        settings.app_env.value = "development"
        settings.app_debug = True
        settings.is_debug = True
        settings.app_host = "0.0.0.0"
        settings.app_port = 8000
        settings.log_level = "WARNING"
        settings.log_format.value = "text"
        settings.api_key_header = "X-API-Key"
        settings.api_keys = ["test-key"]
        settings.vector_store_provider.value = "pgvector"
        settings.openai.api_key.get_secret_value.return_value = ""
        settings.openai.embedding_dimensions = 3072
        settings.groq.api_key.get_secret_value.return_value = ""
        settings.cohere.api_key.get_secret_value.return_value = ""
        settings.postgres.async_url = "postgresql+asyncpg://test@localhost/test"
        settings.pinecone.api_key.get_secret_value.return_value = ""
        settings.redis.url = "redis://localhost:6379/0"
        settings.redis.cache_ttl_seconds = 3600
        settings.rate_limit.requests_per_minute = 1000
        settings.rag.retrieval_top_k = 20
        settings.rag.rerank_top_n = 5
        settings.rag.relevance_threshold = 0.4
        settings.rag.confidence_floor = 0.3
        settings.rag.max_retrieval_iterations = 3
        settings.rag.semantic_cache_similarity_threshold = 0.95
        settings.rag.embedding_batch_size = 100
        settings.cohere.rerank_model = "rerank-english-v3.0"
        settings.observability.api_key.get_secret_value.return_value = ""
        settings.observability.project = "test"
        mock_settings.return_value = settings

        from backend.main import create_app

        app = create_app()
        yield TestClient(app, headers={"X-API-Key": "test-key"})


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_includes_services(self, client):
        resp = client.get("/api/v1/health")
        data = resp.json()
        assert "services" in data

    def test_readiness_probe(self, client):
        resp = client.get("/api/v1/ready")
        assert resp.status_code == 200


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    def test_chat_requires_auth(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "test"},
            headers={},  # No API key
        )
        assert resp.status_code == 401

    def test_chat_with_valid_key(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"query": "What products do you have?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "message_id" in data

    def test_chat_validates_query(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={"query": ""},
        )
        assert resp.status_code == 422  # Validation error

    def test_chat_with_filters(self, client):
        resp = client.post(
            "/api/v1/chat",
            json={
                "query": "electronics products",
                "filters": {"categories": ["Electronics"]},
            },
        )
        assert resp.status_code == 200


class TestFeedbackEndpoint:
    """Tests for the feedback endpoint."""

    def test_submit_feedback(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "message_id": "test-msg-1",
                "conversation_id": "test-conv-1",
                "rating": "thumbs_up",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "received"

    def test_feedback_with_comment(self, client):
        resp = client.post(
            "/api/v1/feedback",
            json={
                "message_id": "test-msg-2",
                "conversation_id": "test-conv-1",
                "rating": "thumbs_down",
                "comment": "Answer was incorrect",
                "is_hallucination": True,
            },
        )
        assert resp.status_code == 200
