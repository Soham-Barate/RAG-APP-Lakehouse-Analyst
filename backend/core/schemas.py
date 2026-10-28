"""
Pydantic v2 request / response schemas for all API endpoints.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DocumentType(str, Enum):
    PRODUCT = "product"
    ORDER = "order"
    REVIEW = "review"
    TICKET = "ticket"
    CUSTOMER = "customer"
    KNOWLEDGE_BASE = "knowledge_base"


class FeedbackRating(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"


class IngestSourceType(str, Enum):
    ICEBERG = "iceberg"
    FILE = "file"
    SYNTHETIC = "synthetic"


# ---------------------------------------------------------------------------
# Shared / nested models
# ---------------------------------------------------------------------------
class MetadataFilter(BaseModel):
    """Metadata filters applied to retrieval."""

    date_from: datetime | None = Field(default=None, description="Start date filter")
    date_to: datetime | None = Field(default=None, description="End date filter")
    categories: list[str] | None = Field(
        default=None, description="Filter by product categories"
    )
    departments: list[str] | None = Field(
        default=None, description="Filter by departments"
    )
    document_types: list[DocumentType] | None = Field(
        default=None, description="Filter by document type"
    )
    min_rating: float | None = Field(
        default=None, ge=0.0, le=5.0, description="Minimum rating filter"
    )


class SourceDocument(BaseModel):
    """A retrieved source document included in the response."""

    document_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Relevant content excerpt")
    document_type: DocumentType = Field(description="Type of source document")
    relevance_score: float = Field(
        ge=0.0, le=1.0, description="Relevance score from reranker"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )


class PipelineMetrics(BaseModel):
    """Performance metrics for the RAG pipeline execution."""

    total_latency_ms: float = Field(description="Total end-to-end latency")
    retrieval_latency_ms: float = Field(default=0.0)
    reranking_latency_ms: float = Field(default=0.0)
    generation_latency_ms: float = Field(default=0.0)
    tokens_used: int = Field(default=0, description="Total LLM tokens consumed")
    retrieval_iterations: int = Field(
        default=1, description="Number of retrieval loops"
    )
    cache_hit: bool = Field(default=False, description="Whether cache was used")
    documents_retrieved: int = Field(default=0)
    documents_after_rerank: int = Field(default=0)


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Incoming chat query from the user."""

    query: str = Field(
        ..., min_length=1, max_length=4000, description="User's question"
    )
    conversation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation thread ID",
    )
    filters: MetadataFilter = Field(
        default_factory=MetadataFilter,
        description="Optional metadata filters",
    )
    max_results: int = Field(
        default=5, ge=1, le=20, description="Max source documents to return"
    )
    include_sources: bool = Field(
        default=True, description="Whether to include source documents"
    )
    stream: bool = Field(
        default=False, description="Enable streaming response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the top-selling electronics products this quarter?",
                "filters": {"categories": ["Electronics"], "date_from": "2026-01-01T00:00:00"},
                "max_results": 5,
            }
        }


class ChatResponse(BaseModel):
    """Complete chat response with answer, sources, and metrics."""

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier",
    )
    conversation_id: str = Field(description="Conversation thread ID")
    answer: str = Field(description="Generated answer")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in the answer"
    )
    sources: list[SourceDocument] = Field(
        default_factory=list, description="Retrieved source documents"
    )
    sql_query: str | None = Field(
        default=None, description="Executed SQL query if structured lakehouse path was used"
    )
    sql_results: list[dict[str, Any]] | None = Field(
        default=None, description="Structured query results if analytical path was used"
    )
    route: str | None = Field(
        default=None, description="Pipeline route chosen: sql, vector, or hybrid"
    )
    metrics: PipelineMetrics = Field(description="Pipeline performance metrics")
    disclaimer: str | None = Field(
        default=None, description="Added when confidence is low"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StreamChunk(BaseModel):
    """A single chunk in a streaming response (SSE)."""

    message_id: str
    chunk: str = Field(description="Text chunk")
    done: bool = Field(default=False, description="True if this is the final chunk")
    sources: list[SourceDocument] | None = Field(
        default=None, description="Sent with the final chunk only"
    )
    metrics: PipelineMetrics | None = Field(
        default=None, description="Sent with the final chunk only"
    )


# ---------------------------------------------------------------------------
# Feedback
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    """User feedback on a specific response."""

    message_id: str = Field(description="ID of the message being rated")
    conversation_id: str = Field(description="Conversation thread ID")
    rating: FeedbackRating = Field(description="Thumbs up or down")
    comment: str | None = Field(
        default=None, max_length=2000, description="Optional comment"
    )
    is_hallucination: bool = Field(
        default=False, description="User reports factual inaccuracy"
    )


class FeedbackResponse(BaseModel):
    """Acknowledgement of received feedback."""

    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="received")


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
class IngestRequest(BaseModel):
    """Request to ingest data into the vector store."""

    source_type: IngestSourceType = Field(description="Type of data source")
    source_path: str | None = Field(
        default=None, description="File path or table name"
    )
    table_name: str | None = Field(
        default=None, description="Iceberg table name"
    )
    document_type: DocumentType | None = Field(
        default=None, description="Type of documents being ingested"
    )
    batch_size: int = Field(
        default=100, ge=1, le=10000, description="Ingestion batch size"
    )


class IngestResponse(BaseModel):
    """Result of an ingestion operation."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: str = Field(default="started")
    documents_processed: int = Field(default=0)
    chunks_created: int = Field(default=0)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = Field(default=0.0)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    services: dict[str, str] = Field(
        default_factory=dict,
        description="Status of dependent services",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------
class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    details: dict[str, Any] = Field(default_factory=dict)
    request_id: str | None = Field(default=None)
