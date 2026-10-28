"""
Observability: LangSmith tracing + Prometheus metrics + structured logging.
"""

from __future__ import annotations

import logging

from prometheus_client import Counter, Gauge, Histogram

from config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus Metrics
# ---------------------------------------------------------------------------
REQUEST_COUNT = Counter(
    "ecommerce_rag_requests_total",
    "Total RAG requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "ecommerce_rag_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

TOKENS_USED = Counter(
    "ecommerce_rag_tokens_total",
    "Total LLM tokens consumed",
    ["provider", "operation"],
)

RETRIEVAL_QUALITY = Histogram(
    "ecommerce_rag_retrieval_relevance",
    "Average retrieval relevance score",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

CACHE_HITS = Counter(
    "ecommerce_rag_cache_hits_total",
    "Semantic cache hits",
)

CACHE_MISSES = Counter(
    "ecommerce_rag_cache_misses_total",
    "Semantic cache misses",
)

PIPELINE_ITERATIONS = Histogram(
    "ecommerce_rag_pipeline_iterations",
    "Number of retrieval iterations per request",
    buckets=(1, 2, 3, 4, 5),
)

ACTIVE_REQUESTS = Gauge(
    "ecommerce_rag_active_requests",
    "Currently processing requests",
)

DOCUMENT_COUNT = Gauge(
    "ecommerce_rag_document_count",
    "Total documents in vector store",
)


# ---------------------------------------------------------------------------
# LangSmith integration
# ---------------------------------------------------------------------------
class ObservabilityService:
    """Centralized observability for tracing, metrics, and logging."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._langsmith_enabled = False
        self._setup_langsmith()

    def _setup_langsmith(self) -> None:
        """Initialize LangSmith tracing if configured."""
        api_key = self.settings.observability.api_key.get_secret_value()
        if api_key:
            try:
                import os

                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = api_key
                os.environ["LANGCHAIN_PROJECT"] = self.settings.observability.project
                self._langsmith_enabled = True
                logger.info(
                    "LangSmith tracing enabled: project=%s",
                    self.settings.observability.project,
                )
            except Exception as exc:
                logger.warning("LangSmith setup failed: %s", exc)

    def record_request(
        self,
        endpoint: str,
        status: str,
        latency_seconds: float,
        tokens: int = 0,
        cache_hit: bool = False,
        relevance_score: float = 0.0,
        iterations: int = 1,
    ) -> None:
        """Record metrics for a completed request."""
        REQUEST_COUNT.labels(endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency_seconds)

        if tokens:
            TOKENS_USED.labels(provider="openai", operation="chat").inc(tokens)

        if cache_hit:
            CACHE_HITS.inc()
        else:
            CACHE_MISSES.inc()

        if relevance_score > 0:
            RETRIEVAL_QUALITY.observe(relevance_score)

        PIPELINE_ITERATIONS.observe(iterations)

    @staticmethod
    def increment_active() -> None:
        ACTIVE_REQUESTS.inc()

    @staticmethod
    def decrement_active() -> None:
        ACTIVE_REQUESTS.dec()

    @staticmethod
    def set_document_count(count: int) -> None:
        DOCUMENT_COUNT.set(count)

    @property
    def is_langsmith_enabled(self) -> bool:
        return self._langsmith_enabled
