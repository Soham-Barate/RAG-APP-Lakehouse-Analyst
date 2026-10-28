"""
FastAPI application factory with lifespan management.

Initializes all services (database pools, Redis, vector stores, LLM clients,
the RAG pipeline) on startup and tears them down gracefully on shutdown.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.middleware.logging_mw import LoggingMiddleware
from backend.api.routes import chat, feedback, health, ingest
from backend.core.constants import API_V1_PREFIX
from backend.core.exceptions import RAGException
from config.settings import LogFormat, get_settings


# ---------------------------------------------------------------------------
# Structured logging configuration
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    settings = get_settings()
    log_level = getattr(logging, settings.log_level, logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if settings.log_format == LogFormat.JSON
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manages the application lifecycle via the ``ServiceContainer``.

    Startup:  Build the container (connects DBs, Redis, LLMs, RAG graph).
    Shutdown: Tear everything down gracefully.
    """
    settings = get_settings()
    logger = structlog.get_logger("lifespan")

    await logger.ainfo("starting_application", env=settings.app_env.value)

    from backend.core.container import ServiceContainer

    container = await ServiceContainer.create(settings)
    app.state.container = container

    # Keep legacy attributes for backward compatibility during migration
    app.state.redis = container.redis
    app.state.db_engine = container.db_engine
    app.state.embedding_service = container.embedding_service
    app.state.vector_store = container.vector_store
    app.state.llm_service = container.llm_service
    app.state.semantic_cache = container.semantic_cache
    app.state.rag_graph = container.rag_graph

    await logger.ainfo("application_started")

    yield  # ---- Application is running ----

    # ----- Shutdown -----
    await logger.ainfo("shutting_down")
    await container.close()
    await logger.ainfo("shutdown_complete")


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    _configure_logging()
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Production-grade E-commerce RAG Knowledge Assistant. "
            "Answers questions about products, orders, customers, sales trends, "
            "returns, and inventory using advanced retrieval-augmented generation."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.is_debug else None,
        redoc_url="/redoc" if settings.is_debug else None,
    )

    # ----- Middleware (order matters: outermost first) -----
    app.add_middleware(LoggingMiddleware)
    cors_origins = (
        ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:3000", "http://localhost:8000"]
        if settings.is_debug
        else ["http://localhost:8501"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----- Global exception handler -----
    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
            headers={"X-Request-ID": getattr(request.state, "request_id", "unknown")},
        )

    # ----- Routes -----
    app.include_router(health.router, prefix=API_V1_PREFIX)
    app.include_router(chat.router, prefix=API_V1_PREFIX)
    app.include_router(ingest.router, prefix=API_V1_PREFIX)
    app.include_router(feedback.router, prefix=API_V1_PREFIX)

    # New production routes
    from backend.api.routes import analytics, conversations, upload
    app.include_router(analytics.router)
    app.include_router(conversations.router)
    app.include_router(upload.router)

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
app = create_app()

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "backend.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.is_debug,
        log_level=settings.log_level.lower(),
        workers=1 if settings.is_debug else 4,
    )
