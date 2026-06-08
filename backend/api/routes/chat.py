"""
Chat endpoints — the primary user-facing API.

Provides both synchronous and Server-Sent Events (SSE) streaming
chat interfaces. Both check the semantic cache first, then invoke
the LangGraph RAG pipeline.
"""

from __future__ import annotations

import json
import time
import uuid

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from backend.api.middleware.auth import verify_api_key
from backend.api.middleware.rate_limiter import check_rate_limit
from backend.core.schemas import (
    ChatRequest,
    ChatResponse,
    PipelineMetrics,
    StreamChunk,
)
from config.settings import get_settings

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    api_key: str = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit),
) -> ChatResponse:
    """
    Synchronous chat endpoint.

    Flow:
    1. Check semantic cache for a similar previous query
    2. If miss → invoke LangGraph RAG pipeline
    3. Cache the result
    4. Return complete response with sources and metrics
    """
    settings = get_settings()
    start = time.perf_counter()
    message_id = str(uuid.uuid4())

    await logger.ainfo(
        "chat_request",
        conversation_id=body.conversation_id,
        query_length=len(body.query),
        has_filters=body.filters is not None,
    )

    # ----- 1. Check semantic cache -----
    cache = getattr(request.app.state, "semantic_cache", None)
    if cache:
        cached = await cache.get(body.query)
        if cached:
            latency_ms = (time.perf_counter() - start) * 1000
            await logger.ainfo("cache_hit", conversation_id=body.conversation_id)
            cached["metrics"]["cache_hit"] = True
            cached["metrics"]["total_latency_ms"] = round(latency_ms, 2)
            return ChatResponse(
                message_id=message_id,
                conversation_id=body.conversation_id,
                **cached,
            )

    # ----- 2. Invoke RAG pipeline -----
    rag_graph = getattr(request.app.state, "rag_graph", None)
    if rag_graph is None:
        # Graceful fallback when pipeline isn't initialized
        latency_ms = (time.perf_counter() - start) * 1000
        return ChatResponse(
            message_id=message_id,
            conversation_id=body.conversation_id,
            answer=(
                "The RAG pipeline is not yet initialized. "
                "Please run data ingestion first or check service health."
            ),
            confidence_score=0.0,
            sources=[],
            metrics=PipelineMetrics(
                total_latency_ms=round(latency_ms, 2),
                cache_hit=False,
            ),
            disclaimer="System not fully initialized.",
        )

    # Build pipeline input state
    pipeline_input = {
        "query": body.query,
        "conversation_id": body.conversation_id,
        "metadata_filters": body.filters.model_dump(exclude_none=True),
        "max_results": body.max_results,
        "messages": [],
        "iteration_count": 0,
    }

    try:
        result = await rag_graph.ainvoke(pipeline_input)
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        await logger.aerror(
            "pipeline_error",
            conversation_id=body.conversation_id,
            error=str(exc),
        )
        return ChatResponse(
            message_id=message_id,
            conversation_id=body.conversation_id,
            answer="I encountered an error processing your request. Please try again.",
            confidence_score=0.0,
            sources=[],
            metrics=PipelineMetrics(total_latency_ms=round(latency_ms, 2)),
            disclaimer="An error occurred during processing.",
        )

    latency_ms = (time.perf_counter() - start) * 1000

    # Build response
    response = ChatResponse(
        message_id=message_id,
        conversation_id=body.conversation_id,
        answer=result.get("answer", "No answer generated."),
        confidence_score=result.get("confidence_score", 0.0),
        sources=result.get("sources", []),
        metrics=PipelineMetrics(
            total_latency_ms=round(latency_ms, 2),
            retrieval_latency_ms=result.get("retrieval_latency_ms", 0.0),
            reranking_latency_ms=result.get("reranking_latency_ms", 0.0),
            generation_latency_ms=result.get("generation_latency_ms", 0.0),
            tokens_used=result.get("tokens_used", 0),
            retrieval_iterations=result.get("iteration_count", 1),
            cache_hit=False,
            documents_retrieved=result.get("documents_retrieved", 0),
            documents_after_rerank=result.get("documents_after_rerank", 0),
        ),
        disclaimer=result.get("disclaimer"),
    )

    # ----- 3. Cache the result -----
    if cache and response.confidence_score > 0.3:
        await cache.set(
            body.query,
            {
                "answer": response.answer,
                "confidence_score": response.confidence_score,
                "sources": [s.model_dump() for s in response.sources],
                "metrics": response.metrics.model_dump(),
            },
        )

    await logger.ainfo(
        "chat_response",
        conversation_id=body.conversation_id,
        confidence=response.confidence_score,
        sources_count=len(response.sources),
        latency_ms=round(latency_ms, 2),
        tokens_used=response.metrics.tokens_used,
    )

    return response


@router.post("/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    api_key: str = Depends(verify_api_key),
    _rate: None = Depends(check_rate_limit),
) -> StreamingResponse:
    """
    SSE streaming chat endpoint.

    Streams response tokens as Server-Sent Events. The final event
    includes sources and pipeline metrics.
    """
    message_id = str(uuid.uuid4())

    async def event_generator():  # type: ignore[return]
        start = time.perf_counter()

        rag_graph = getattr(request.app.state, "rag_graph", None)
        if rag_graph is None:
            chunk = StreamChunk(
                message_id=message_id,
                chunk="RAG pipeline not initialized. Please run ingestion first.",
                done=True,
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
            return

        pipeline_input = {
            "query": body.query,
            "conversation_id": body.conversation_id,
            "metadata_filters": body.filters.model_dump(exclude_none=True),
            "max_results": body.max_results,
            "messages": [],
            "iteration_count": 0,
            "stream": True,
        }

        try:
            # Use astream_events for token-level streaming
            collected_answer = ""
            final_result = {}

            async for event in rag_graph.astream_events(
                pipeline_input, version="v2"
            ):
                kind = event.get("event", "")

                if kind == "on_chat_model_stream":
                    token = event.get("data", {}).get("chunk", "")
                    if hasattr(token, "content") and token.content:
                        collected_answer += token.content
                        chunk = StreamChunk(
                            message_id=message_id,
                            chunk=token.content,
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                elif kind == "on_chain_end":
                    output = event.get("data", {}).get("output", {})
                    if isinstance(output, dict) and "answer" in output:
                        final_result = output

            # Send final chunk with sources and metrics
            latency_ms = (time.perf_counter() - start) * 1000
            final_chunk = StreamChunk(
                message_id=message_id,
                chunk="",
                done=True,
                sources=final_result.get("sources", []),
                metrics=PipelineMetrics(
                    total_latency_ms=round(latency_ms, 2),
                    tokens_used=final_result.get("tokens_used", 0),
                    retrieval_iterations=final_result.get("iteration_count", 1),
                    cache_hit=False,
                ),
            )
            yield f"data: {final_chunk.model_dump_json()}\n\n"

        except Exception as exc:
            error_chunk = StreamChunk(
                message_id=message_id,
                chunk=f"Error: {str(exc)}",
                done=True,
            )
            yield f"data: {error_chunk.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Message-ID": message_id,
        },
    )
