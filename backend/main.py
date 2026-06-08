"""
FastAPI application factory with lifespan management.

Initializes all services (database pools, Redis, vector stores, LLM clients,
the RAG pipeline) on startup and tears them down gracefully on shutdown.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.middleware.logging_mw import LoggingMiddleware
from backend.api.routes import chat, feedback, health, ingest
from backend.core.constants import API_V1_PREFIX
from backend.core.exceptions import RAGException
from config.settings import Environment, LogFormat, VectorStoreProvider, get_settings


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
        wrapper_class=structlog.stdlib.AsyncBoundLogger,
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
    Manages the application lifecycle:
    - Startup: connect to databases, Redis, initialize vector stores,
      LLM clients, embedding service, RAG pipeline.
    - Shutdown: close connections gracefully.
    """
    settings = get_settings()
    logger = structlog.get_logger("lifespan")

    await logger.ainfo("starting_application", env=settings.app_env.value)

    # ----- Redis -----
    try:
        import redis.asyncio as aioredis

        redis_client = aioredis.from_url(
            settings.redis.url,
            decode_responses=False,
        )
        await redis_client.ping()
        app.state.redis = redis_client
        await logger.ainfo("redis_connected")
    except Exception as exc:
        await logger.awarning("redis_connection_failed", error=str(exc))
        app.state.redis = None

    # ----- PostgreSQL engine -----
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            settings.postgres.async_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=settings.is_debug,
        )
        app.state.db_engine = engine
        await logger.ainfo("postgres_engine_created")
    except Exception as exc:
        await logger.awarning("postgres_connection_failed", error=str(exc))
        app.state.db_engine = None

    # ----- Embedding service -----
    try:
        from backend.embeddings.embedding_service import EmbeddingService

        embedding_service = EmbeddingService(settings=settings)
        app.state.embedding_service = embedding_service
        await logger.ainfo("embedding_service_initialized")
    except Exception as exc:
        await logger.awarning("embedding_service_failed", error=str(exc))
        app.state.embedding_service = None

    # ----- Vector store -----
    try:
        provider = settings.vector_store_provider
        if provider in (VectorStoreProvider.PGVECTOR, VectorStoreProvider.BOTH):
            from backend.vectorstore.pgvector_store import PgVectorStore

            pg_store = PgVectorStore(
                engine=app.state.db_engine,
                embedding_service=app.state.embedding_service,
                settings=settings,
            )
            await pg_store.initialize()
            app.state.vector_store = pg_store
            await logger.ainfo("pgvector_store_initialized")

        if provider in (VectorStoreProvider.PINECONE, VectorStoreProvider.BOTH):
            from backend.vectorstore.pinecone_store import PineconeVectorStore

            pine_store = PineconeVectorStore(settings=settings)
            await pine_store.initialize()
            if provider == VectorStoreProvider.PINECONE:
                app.state.vector_store = pine_store
            else:
                app.state.pinecone_store = pine_store
            await logger.ainfo("pinecone_store_initialized")

    except Exception as exc:
        await logger.awarning("vector_store_init_failed", error=str(exc))
        app.state.vector_store = None

    # ----- LLM service -----
    try:
        from backend.services.llm_service import LLMService

        llm_service = LLMService(settings=settings)
        app.state.llm_service = llm_service
        await logger.ainfo("llm_service_initialized")
    except Exception as exc:
        await logger.awarning("llm_service_failed", error=str(exc))
        app.state.llm_service = None

    # ----- Semantic cache -----
    try:
        if app.state.redis and app.state.embedding_service:
            from backend.rag.cache.semantic_cache import SemanticCache

            semantic_cache = SemanticCache(
                redis_client=app.state.redis,
                embedding_service=app.state.embedding_service,
                similarity_threshold=settings.rag.semantic_cache_similarity_threshold,
                ttl_seconds=settings.redis.cache_ttl_seconds,
            )
            app.state.semantic_cache = semantic_cache
            await logger.ainfo("semantic_cache_initialized")
        else:
            app.state.semantic_cache = None
    except Exception as exc:
        await logger.awarning("semantic_cache_failed", error=str(exc))
        app.state.semantic_cache = None

    # ----- RAG pipeline (LangGraph) -----
    try:
        if app.state.vector_store and app.state.llm_service:
            from backend.rag.graph import build_rag_graph

            rag_graph = build_rag_graph(
                vector_store=app.state.vector_store,
                llm_service=app.state.llm_service,
                embedding_service=app.state.embedding_service,
                settings=settings,
            )
            app.state.rag_graph = rag_graph
            await logger.ainfo("rag_pipeline_compiled")
        else:
            app.state.rag_graph = None
            await logger.awarning(
                "rag_pipeline_skipped",
                reason="Missing vector_store or llm_service",
            )
    except Exception as exc:
        await logger.awarning("rag_pipeline_failed", error=str(exc))
        app.state.rag_graph = None

    await logger.ainfo("application_started")

    yield  # ---- Application is running ----

    # ----- Shutdown -----
    await logger.ainfo("shutting_down")

    if getattr(app.state, "redis", None):
        await app.state.redis.close()

    if getattr(app.state, "db_engine", None):
        await app.state.db_engine.dispose()

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_debug else ["http://localhost:8501"],
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
