"""
FastAPI dependency providers.

Each function extracts a specific service from the ``ServiceContainer``
stored on ``app.state.container``.  Routes declare what they need via
``Depends(get_vector_store)`` etc., keeping route logic decoupled from
initialisation.
"""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request, status

from backend.core.container import ServiceContainer


# ---------------------------------------------------------------------------
# Container accessor
# ---------------------------------------------------------------------------
def get_container(request: Request) -> ServiceContainer:
    """
    Retrieve the service container from the application state.

    Every other dependency function below calls this first.
    """
    container: ServiceContainer | None = getattr(
        request.app.state, "container", None
    )
    if container is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "SERVICE_UNAVAILABLE",
                "message": "Service container not initialised. "
                           "The application may still be starting up.",
            },
        )
    return container


# ---------------------------------------------------------------------------
# Individual service extractors
# ---------------------------------------------------------------------------
def get_vector_store(
    container: ServiceContainer = Depends(get_container),
) -> Any:
    """Return the active vector store or 503."""
    if container.vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "VECTOR_STORE_UNAVAILABLE",
                "message": "Vector store is not initialised.",
            },
        )
    return container.vector_store


def get_embedding_service(
    container: ServiceContainer = Depends(get_container),
) -> Any:
    """Return the embedding service or 503."""
    if container.embedding_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "EMBEDDING_SERVICE_UNAVAILABLE",
                "message": "Embedding service is not initialised.",
            },
        )
    return container.embedding_service


def get_llm_service(
    container: ServiceContainer = Depends(get_container),
) -> Any:
    """Return the LLM service or 503."""
    if container.llm_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "LLM_SERVICE_UNAVAILABLE",
                "message": "LLM service is not initialised.",
            },
        )
    return container.llm_service


def get_rag_pipeline(
    container: ServiceContainer = Depends(get_container),
) -> Any | None:
    """Return the compiled RAG graph (may be None if not ready)."""
    return container.rag_graph


def get_semantic_cache(
    container: ServiceContainer = Depends(get_container),
) -> Any | None:
    """Return the semantic cache (may be None if Redis is down)."""
    return container.semantic_cache


def get_redis(
    container: ServiceContainer = Depends(get_container),
) -> Any | None:
    """Return the Redis client (may be None)."""
    return container.redis
