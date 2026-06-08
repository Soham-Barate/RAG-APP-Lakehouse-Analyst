"""
RAG pipeline state definition.

This TypedDict defines the shared mutable state that flows through every
node in the LangGraph StateGraph. Each node reads what it needs and
writes its outputs back.
"""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage

from backend.core.schemas import SourceDocument


class RAGState(TypedDict, total=False):
    """
    Complete state for the agentic RAG pipeline.

    Fields marked as required are populated at the start.
    Optional fields are set by individual nodes.
    """

    # ----- Input (set by the API layer) -----
    query: str                            # Original user query
    conversation_id: str                  # Conversation thread ID
    metadata_filters: dict[str, Any]      # Filters from the user
    max_results: int                      # Max sources to return
    messages: list[BaseMessage]           # Conversation history
    stream: bool                          # Whether to stream

    # ----- Query processing -----
    route: str                            # "sql" | "vector" | "hybrid"
    rewritten_queries: list[str]          # Expanded / decomposed queries
    hyde_document: str                    # Hypothetical answer document

    # ----- Retrieval -----
    retrieved_docs: list[dict[str, Any]]  # Raw retrieved document dicts
    reranked_docs: list[dict[str, Any]]   # After cross-encoder reranking
    compressed_context: str               # Compressed context string
    relevance_scores: list[float]         # Per-document relevance

    # ----- Generation -----
    answer: str                           # Final generated answer
    sources: list[SourceDocument]         # Citation sources
    confidence_score: float               # Overall confidence (0–1)
    disclaimer: str | None                # Low-confidence disclaimer

    # ----- Control flow -----
    iteration_count: int                  # Retrieval loop counter
    should_retry: bool                    # Grader decision flag

    # ----- Guardrails -----
    guardrail_flags: dict[str, Any]       # Safety check results
    pii_redacted_query: str               # Input after PII redaction

    # ----- Metrics (accumulated across nodes) -----
    retrieval_latency_ms: float
    reranking_latency_ms: float
    generation_latency_ms: float
    tokens_used: int
    documents_retrieved: int
    documents_after_rerank: int
