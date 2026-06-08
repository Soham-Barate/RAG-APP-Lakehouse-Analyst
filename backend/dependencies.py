"""
FastAPI dependency injection providers.

Centralizes creation and lifecycle of shared services so that route
handlers receive fully-configured instances via ``Depends()``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import Depends, Request

from config.settings import Settings, VectorStoreProvider, get_settings


def get_vector_store(request: Request) -> Any:
    """Provide the active vector store from app state."""
    return getattr(request.app.state, "vector_store", None)


def get_embedding_service(request: Request) -> Any:
    """Provide the embedding service from app state."""
    return getattr(request.app.state, "embedding_service", None)


def get_llm_service(request: Request) -> Any:
    """Provide the LLM service from app state."""
    return getattr(request.app.state, "llm_service", None)


def get_semantic_cache(request: Request) -> Any:
    """Provide the semantic cache from app state."""
    return getattr(request.app.state, "semantic_cache", None)


def get_rag_graph(request: Request) -> Any:
    """Provide the compiled LangGraph RAG pipeline."""
    return getattr(request.app.state, "rag_graph", None)


def get_redis(request: Request) -> Any:
    """Provide the Redis client from app state."""
    return getattr(request.app.state, "redis", None)
