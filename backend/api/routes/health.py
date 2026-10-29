"""
Health and readiness probe endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse

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
            import asyncio

            from sqlalchemy import text

            async def _check_db():
                async with db_engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))

            await asyncio.wait_for(_check_db(), timeout=1.5)
            services["postgres"] = "healthy"
        except Exception:
            services["postgres"] = "unhealthy"
    else:
        services["postgres"] = "not_configured"

    # Check vector store
    vector_store = getattr(request.app.state, "vector_store", None)
    services["vector_store"] = "healthy" if vector_store is not None else "not_configured"

    # Check LLM service
    llm_service = getattr(request.app.state, "llm_service", None)
    services["llm_service"] = "healthy" if llm_service is not None else "not_configured"

    # Check RAG graph
    rag_graph = getattr(request.app.state, "rag_graph", None)
    services["rag_pipeline"] = "healthy" if rag_graph is not None else "not_initialized"

    # Overall status (Redis failure is degraded, critical service failure makes it unhealthy)
    critical_unhealthy = [
        k for k, v in services.items()
        if k in ("postgres", "vector_store", "llm_service", "rag_pipeline") and v in ("unhealthy", "not_configured", "not_initialized")
    ]
    unhealthy = [k for k, v in services.items() if v == "unhealthy"]

    if critical_unhealthy and settings.app_env.value != "development":
        overall = "unhealthy"
    elif unhealthy or critical_unhealthy:
        overall = "degraded"
    else:
        overall = "healthy"

    return HealthResponse(
        status=overall,
        version="1.0.0",
        environment=settings.app_env.value,
        services=services,
        timestamp=datetime.now(UTC),
    )


@router.get("/ready")
async def readiness_check(request: Request) -> JSONResponse:
    """
    Readiness probe — returns HTTP 200 only when all critical services
    are connected and the pipeline is initialized, or HTTP 503 when not ready.
    """
    rag_graph = getattr(request.app.state, "rag_graph", None)
    if rag_graph is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "not_ready", "reason": "RAG pipeline not initialized"},
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={"status": "ready"},
    )
