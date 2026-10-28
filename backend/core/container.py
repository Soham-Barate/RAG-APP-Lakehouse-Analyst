"""
Service container — centralised dependency registry.

Replaces scattered ``app.state.xxx`` access with a single, typed
container that is created once at startup and injected via FastAPI
``Depends()``.

Why a container?
  • Every service is accessed through a typed attribute, so IDEs and
    type-checkers catch misuse at write-time.
  • Unit tests can replace individual services with mocks by constructing
    a container with only the mocks they need.
  • The ``create()`` class-method documents the exact wiring order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

from config.settings import Environment, Settings, VectorStoreProvider, get_settings

logger = structlog.get_logger(__name__)

CRITICAL_SERVICES = {
    "db_engine",
    "embedding_service",
    "vector_store",
    "llm_service",
    "rag_graph",
}


@dataclass
class ServiceContainer:
    """
    Holds references to all initialised application services.

    Attributes are typed as ``Any | None`` so that individual services
    can be absent (e.g., Redis down, Pinecone not configured).
    """

    settings: Settings
    redis: Any = None
    db_engine: Any = None
    embedding_service: Any = None
    vector_store: Any = None
    pinecone_store: Any = None
    llm_service: Any = None
    semantic_cache: Any = None
    rag_graph: Any = None

    # ---------------------------------------------------------------------------
    # Factory
    # ---------------------------------------------------------------------------
    @classmethod
    async def create(cls, settings: Settings | None = None) -> ServiceContainer:
        """
        Construct and wire all services in dependency order.

        This is the **single place** where the whole object graph is
        assembled.  The lifespan handler calls this once at startup.
        """
        settings = settings or get_settings()
        container = cls(settings=settings)

        await container._init_redis()
        await container._init_database()
        await container._init_embeddings()
        await container._init_vector_store()
        await container._init_llm()
        await container._init_semantic_cache()
        await container._init_rag_pipeline()

        await logger.ainfo("service_container_ready")
        return container

    # ---------------------------------------------------------------------------
    # Private initialisers (called in order by create())
    # ---------------------------------------------------------------------------
    async def _init_redis(self) -> None:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(
                self.settings.redis.url, decode_responses=False
            )
            await client.ping()
            self.redis = client
            await logger.ainfo("redis_connected")
        except Exception as exc:
            await logger.awarning("redis_unavailable", error=str(exc))

    async def _init_database(self) -> None:
        try:
            from sqlalchemy.ext.asyncio import create_async_engine

            self.db_engine = create_async_engine(
                self.settings.postgres.async_url,
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
                echo=self.settings.is_debug,
            )
            await logger.ainfo("postgres_engine_created")
        except Exception as exc:
            await logger.awarning("postgres_unavailable", error=str(exc))
            if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                raise RuntimeError(f"Critical service 'db_engine' failed to initialize: {exc}") from exc

    async def _init_embeddings(self) -> None:
        try:
            from backend.embeddings.embedding_service import EmbeddingService

            self.embedding_service = EmbeddingService(settings=self.settings)
            await logger.ainfo("embedding_service_ready")
        except Exception as exc:
            await logger.awarning("embedding_service_failed", error=str(exc))
            if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                raise RuntimeError(f"Critical service 'embedding_service' failed to initialize: {exc}") from exc

    async def _init_vector_store(self) -> None:
        try:
            provider = self.settings.vector_store_provider
            if provider in (VectorStoreProvider.PGVECTOR, VectorStoreProvider.BOTH):
                from backend.vectorstore.pgvector_store import PgVectorStore

                pg = PgVectorStore(
                    engine=self.db_engine,
                    embedding_service=self.embedding_service,
                    settings=self.settings,
                )
                await pg.initialize()
                self.vector_store = pg
                await logger.ainfo("pgvector_store_ready")

            if provider in (VectorStoreProvider.PINECONE, VectorStoreProvider.BOTH):
                from backend.vectorstore.pinecone_store import PineconeVectorStore

                pine = PineconeVectorStore(settings=self.settings)
                await pine.initialize()
                if provider == VectorStoreProvider.PINECONE:
                    self.vector_store = pine
                else:
                    self.pinecone_store = pine
                await logger.ainfo("pinecone_store_ready")
        except Exception as exc:
            await logger.awarning("vector_store_failed", error=str(exc))
            if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                raise RuntimeError(f"Critical service 'vector_store' failed to initialize: {exc}") from exc

    async def _init_llm(self) -> None:
        try:
            from backend.services.llm_service import LLMService

            self.llm_service = LLMService(settings=self.settings)
            await logger.ainfo("llm_service_ready")
        except Exception as exc:
            await logger.awarning("llm_service_failed", error=str(exc))
            if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                raise RuntimeError(f"Critical service 'llm_service' failed to initialize: {exc}") from exc

    async def _init_semantic_cache(self) -> None:
        try:
            if self.redis and self.embedding_service:
                from backend.rag.cache.semantic_cache import SemanticCache

                self.semantic_cache = SemanticCache(
                    redis_client=self.redis,
                    embedding_service=self.embedding_service,
                    similarity_threshold=self.settings.rag.semantic_cache_similarity_threshold,
                    ttl_seconds=self.settings.redis.cache_ttl_seconds,
                )
                await logger.ainfo("semantic_cache_ready")
        except Exception as exc:
            await logger.awarning("semantic_cache_failed", error=str(exc))

    async def _init_rag_pipeline(self) -> None:
        try:
            if self.vector_store and self.llm_service:
                from backend.rag.graph import build_rag_graph

                self.rag_graph = build_rag_graph(
                    vector_store=self.vector_store,
                    llm_service=self.llm_service,
                    embedding_service=self.embedding_service,
                    settings=self.settings,
                )
                await logger.ainfo("rag_pipeline_compiled")
            else:
                await logger.awarning(
                    "rag_pipeline_skipped",
                    reason="Missing vector_store or llm_service",
                )
                if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                    raise RuntimeError("Critical service 'rag_graph' could not be initialized due to missing dependencies.")
        except Exception as exc:
            await logger.awarning("rag_pipeline_failed", error=str(exc))
            if self.settings.app_env in (Environment.PRODUCTION, Environment.STAGING):
                raise RuntimeError(f"Critical service 'rag_graph' failed to initialize: {exc}") from exc

    # ---------------------------------------------------------------------------
    # Teardown
    # ---------------------------------------------------------------------------
    async def close(self) -> None:
        """Gracefully shut down all services."""
        if self.redis:
            await self.redis.close()
        if self.db_engine:
            await self.db_engine.dispose()
        await logger.ainfo("service_container_closed")
