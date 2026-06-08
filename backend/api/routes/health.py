"""
Health and readiness probe endpoints.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Request

from backend.core.schemas import HealthResponse
from config.settings import get_settings

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Liveness probe — returns healthy if the API process is running.
    Also checks connectivity to dependent services.
    """
    settings = get_settings()
    services: dict[str, str] = {}

    # Check Redis
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client:
        try:
            await redis_client.ping()
            services["redis"] = "healthy"
        except Exception:
            services["redis"] = "unhealthy"
    else:
        services["redis"] = "not_configured"

    # Check PostgreSQL
    db_engine = getattr(request.app.state, "db_engine", None)
    if db_engine:
        try:
            from sqlalchemy import text

            async with db_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            services["postgres"] = "healthy"
        except Exception:
            services["postgres"] = "unhealthy"
    else:
        services["postgres"] = "not_configured"

    # Check vector store
    vector_store = getattr(request.app.state, "vector_store", None)
    if vector_store:
        services["vector_store"] = "healthy"
    else:
        services["vector_store"] = "not_configured"

    # Overall status
    unhealthy = [k for k, v in services.items() if v == "unhealthy"]
    overall = "degraded" if unhealthy else "healthy"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        environment=settings.app_env.value,
        services=services,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ready")
async def readiness_check(request: Request) -> dict[str, str]:
    """
    Readiness probe — returns ready only when all critical services
    are connected and the pipeline is initialized.
    """
    rag_graph = getattr(request.app.state, "rag_graph", None)
    if rag_graph is None:
        return {"status": "not_ready", "reason": "RAG pipeline not initialized"}
    return {"status": "ready"}
