"""
Custom exception hierarchy for the RAG application.

Every exception carries a machine-readable ``error_code`` and maps to an
HTTP status code so the FastAPI global handler can translate them cleanly.
"""

from __future__ import annotations

from typing import Any


class RAGException(Exception):
    """Base exception for the RAG application."""

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"

    def __init__(
        self,
        message: str = "An internal error occurred",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Retrieval errors
# ---------------------------------------------------------------------------
class RetrievalError(RAGException):
    status_code = 502
    error_code = "RETRIEVAL_ERROR"

    def __init__(self, message: str = "Failed to retrieve documents", **kw: Any):
        super().__init__(message, **kw)


class VectorStoreConnectionError(RetrievalError):
    error_code = "VECTOR_STORE_CONNECTION_ERROR"

    def __init__(self, message: str = "Cannot connect to vector store", **kw: Any):
        super().__init__(message, **kw)


class EmbeddingError(RetrievalError):
    error_code = "EMBEDDING_ERROR"

    def __init__(self, message: str = "Embedding generation failed", **kw: Any):
        super().__init__(message, **kw)


# ---------------------------------------------------------------------------
# LLM errors
# ---------------------------------------------------------------------------
class LLMError(RAGException):
    status_code = 502
    error_code = "LLM_ERROR"

    def __init__(self, message: str = "LLM request failed", **kw: Any):
        super().__init__(message, **kw)


class LLMRateLimitError(LLMError):
    status_code = 429
    error_code = "LLM_RATE_LIMIT"

    def __init__(self, message: str = "LLM rate limit exceeded", **kw: Any):
        super().__init__(message, **kw)


class LLMTimeoutError(LLMError):
    status_code = 504
    error_code = "LLM_TIMEOUT"

    def __init__(self, message: str = "LLM request timed out", **kw: Any):
        super().__init__(message, **kw)


# ---------------------------------------------------------------------------
# Auth / rate-limit errors
# ---------------------------------------------------------------------------
class AuthenticationError(RAGException):
    status_code = 401
    error_code = "AUTHENTICATION_ERROR"

    def __init__(self, message: str = "Authentication failed", **kw: Any):
        super().__init__(message, **kw)


class RateLimitError(RAGException):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(self, message: str = "Rate limit exceeded", **kw: Any):
        super().__init__(message, **kw)


# ---------------------------------------------------------------------------
# Guardrail errors
# ---------------------------------------------------------------------------
class GuardrailError(RAGException):
    status_code = 400
    error_code = "GUARDRAIL_VIOLATION"

    def __init__(self, message: str = "Request blocked by guardrails", **kw: Any):
        super().__init__(message, **kw)


class PIIDetectedError(GuardrailError):
    error_code = "PII_DETECTED"

    def __init__(self, message: str = "PII detected in request", **kw: Any):
        super().__init__(message, **kw)


class ToxicContentError(GuardrailError):
    error_code = "TOXIC_CONTENT"

    def __init__(self, message: str = "Toxic content detected", **kw: Any):
        super().__init__(message, **kw)


# ---------------------------------------------------------------------------
# Ingestion errors
# ---------------------------------------------------------------------------
class IngestionError(RAGException):
    status_code = 500
    error_code = "INGESTION_ERROR"

    def __init__(self, message: str = "Data ingestion failed", **kw: Any):
        super().__init__(message, **kw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
class ValidationError(RAGException):
    status_code = 422
    error_code = "VALIDATION_ERROR"

    def __init__(self, message: str = "Request validation failed", **kw: Any):
        super().__init__(message, **kw)
