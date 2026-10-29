"""
Centralized configuration via Pydantic v2 Settings.

Loads from environment variables / .env file with full validation,
defaults, and nested configuration groups for each subsystem.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class VectorStoreProvider(str, Enum):
    PGVECTOR = "pgvector"
    PINECONE = "pinecone"
    BOTH = "both"


class LogFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------
class OpenAISettings(BaseSettings):
    """OpenAI provider settings."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Primary LLM model")
    embedding_model: str = Field(
        default="text-embedding-3-large", description="Embedding model"
    )
    embedding_dimensions: int = Field(
        default=3072, description="Embedding vector dimensions"
    )


class GroqSettings(BaseSettings):
    """Groq fallback LLM settings."""

    model_config = SettingsConfigDict(env_prefix="GROQ_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="Groq API key")
    model: str = Field(
        default="llama-3.1-70b-versatile", description="Groq model name"
    )


class CohereSettings(BaseSettings):
    """Cohere reranker settings."""

    model_config = SettingsConfigDict(env_prefix="COHERE_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="Cohere API key")
    rerank_model: str = Field(
        default="rerank-english-v3.0", description="Reranking model"
    )


class PostgresSettings(BaseSettings):
    """PostgreSQL + pgvector settings."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = Field(default="localhost")
    port: int = Field(default=5432)
    db: str = Field(default="rag_ecommerce")
    user: str = Field(default="rag_user")
    password: SecretStr = Field(default=SecretStr("change-me-in-production"))

    @property
    def async_url(self) -> str:
        """Async SQLAlchemy connection URL."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def sync_url(self) -> str:
        """Sync connection URL for migrations / scripts."""
        pwd = self.password.get_secret_value()
        return (
            f"postgresql://{self.user}:{pwd}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class PineconeSettings(BaseSettings):
    """Pinecone Serverless settings."""

    model_config = SettingsConfigDict(env_prefix="PINECONE_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="Pinecone API key")
    index_name: str = Field(default="ecommerce-rag")
    environment: str = Field(default="us-east-1")
    cloud: str = Field(default="aws")


class RedisSettings(BaseSettings):
    """Redis cache settings."""

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    url: str = Field(default="redis://localhost:6379/0")
    cache_ttl_seconds: int = Field(default=3600, description="Cache TTL in seconds")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="RATE_LIMIT_", env_file=".env", extra="ignore")

    requests_per_minute: int = Field(default=60)
    tokens_per_minute: int = Field(default=100_000)
    fail_closed: bool = Field(
        default=False,
        description="Whether to block traffic when Redis is unavailable (recommended true in production)",
    )


class RAGSettings(BaseSettings):
    """RAG pipeline parameters."""

    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    retrieval_top_k: int = Field(
        default=20, description="Initial retrieval candidate count"
    )
    rerank_top_n: int = Field(
        default=5, description="Final reranked document count"
    )
    chunk_size_parent: int = Field(
        default=2000, description="Parent chunk size in tokens"
    )
    chunk_size_child: int = Field(
        default=256, description="Child chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=64, description="Overlap between child chunks"
    )
    max_retrieval_iterations: int = Field(
        default=3, description="Max self-correction loops"
    )
    relevance_threshold: float = Field(
        default=0.4, description="Minimum relevance to proceed"
    )
    confidence_floor: float = Field(
        default=0.3, description="Below this → add disclaimer"
    )
    semantic_cache_similarity_threshold: float = Field(
        default=0.95, description="Cosine sim threshold for cache hit"
    )
    embedding_batch_size: int = Field(
        default=100, description="Batch size for embedding calls"
    )


class ObservabilitySettings(BaseSettings):
    """Tracing and monitoring settings."""

    model_config = SettingsConfigDict(env_prefix="LANGSMITH_", env_file=".env", extra="ignore")

    api_key: SecretStr = Field(default=SecretStr(""), description="LangSmith API key")
    project: str = Field(default="ecommerce-rag-prod")
    tracing_v2: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Root settings
# ---------------------------------------------------------------------------
class Settings(BaseSettings):
    """
    Root application settings.

    Loads automatically from .env file at project root.
    Access via ``get_settings()`` singleton.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    app_name: str = Field(default="E-commerce RAG Assistant")
    app_env: Environment = Field(default=Environment.DEVELOPMENT)
    app_debug: bool = Field(default=False)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")
    log_format: LogFormat = Field(default=LogFormat.JSON)

    # Security
    api_key_header: str = Field(default="X-API-Key")
    api_keys: list[str] = Field(default_factory=lambda: ["dev-key-1"])
    jwt_secret_key: SecretStr = Field(
        default=SecretStr("change-this-to-a-random-secret")
    )
    jwt_algorithm: str = Field(default="HS256")
    jwt_expiration_minutes: int = Field(default=60)
    allowed_file_types: list[str] = Field(
        default_factory=lambda: [
            ".pdf", ".docx", ".xlsx", ".csv", ".txt",
            ".json", ".jsonl", ".md", ".markdown", ".parquet"
        ]
    )
    max_file_size_mb: int = Field(default=50)
    api_key_roles: dict[str, str] = Field(
        default_factory=lambda: {"dev-key-1": "admin"}
    )

    # Vector store selection
    vector_store_provider: VectorStoreProvider = Field(
        default=VectorStoreProvider.PGVECTOR
    )

    # Frontend
    streamlit_port: int = Field(default=8501)
    backend_url: str = Field(default="http://localhost:8000")

    # Sub-configurations (initialized manually — not env-nested)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    cohere: CohereSettings = Field(default_factory=CohereSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    pinecone: PineconeSettings = Field(default_factory=PineconeSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings
    )

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v  # type: ignore[return-value]

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, v: Any) -> str:
        if isinstance(v, str):
            return v.upper()
        return v  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_production_secrets(self) -> Settings:
        """Warn if running production without real secrets."""
        if self.app_env == Environment.PRODUCTION:
            if self.jwt_secret_key.get_secret_value() == "change-this-to-a-random-secret":
                raise ValueError(
                    "JWT_SECRET_KEY must be changed for production deployments"
                )
        return self

    @property
    def is_debug(self) -> bool:
        return self.app_debug or self.app_env == Environment.DEVELOPMENT


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for application settings."""
    return Settings()
