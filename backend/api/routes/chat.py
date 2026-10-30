"""
Chat endpoints — the primary user-facing API.

Provides both synchronous and Server-Sent Events (SSE) streaming
chat interfaces. Both check the semantic cache first, then invoke
the LangGraph RAG pipeline.
"""

from __future__ import annotations

import time
import uuid

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from backend.api.middleware.auth import verify_api_key
from backend.api.middleware.rate_limiter import check_rate_limit
from backend.core.schemas import (
    ChatRequest,
    ChatResponse,
    PipelineMetrics,
    StreamChunk,
)

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
    start = time.perf_counter()
    message_id = str(uuid.uuid4())

    from backend.services.observability import (
        ACTIVE_REQUESTS,
        CACHE_HITS,
        CACHE_MISSES,
        REQUEST_COUNT,
        REQUEST_LATENCY,
        TOKENS_USED,
    )

    ACTIVE_REQUESTS.inc()

    await logger.ainfo(
        "chat_request",
        conversation_id=body.conversation_id,
        query_length=len(body.query),
        has_filters=body.filters is not None,
    )

    try:
        # ----- 1. Check semantic cache -----
        cache = getattr(request.app.state, "semantic_cache", None)
        if cache:
            filters_dict = body.filters.model_dump(exclude_none=True)
            cached = await cache.get(body.query, filters=filters_dict)
            if cached:
                latency_ms = (time.perf_counter() - start) * 1000
                await logger.ainfo("cache_hit", conversation_id=body.conversation_id)
                CACHE_HITS.inc()
                REQUEST_COUNT.labels(endpoint="/chat", status="200").inc()
                REQUEST_LATENCY.labels(endpoint="/chat").observe(latency_ms / 1000.0)
                cached["metrics"]["cache_hit"] = True
                cached["metrics"]["total_latency_ms"] = round(latency_ms, 2)
                return ChatResponse(
                    message_id=message_id,
                    conversation_id=body.conversation_id,
                    **cached,
                )
            else:
                CACHE_MISSES.inc()

        # ----- 2. Invoke RAG pipeline -----
        rag_graph = getattr(request.app.state, "rag_graph", None)
        if rag_graph is None:
            await logger.awarning("rag_pipeline_uninitialized", conversation_id=body.conversation_id)
            REQUEST_COUNT.labels(endpoint="/chat", status="503").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="The RAG pipeline is not yet initialized. Please run data ingestion first or check service health.",
            )

        # Build pipeline input state
        filters_dict = body.filters.model_dump(exclude_none=True)
        if body.dataset_id:
            filters_dict["dataset_id"] = body.dataset_id

        pipeline_input = {
            "query": body.query,
            "conversation_id": body.conversation_id,
            "dataset_id": body.dataset_id,
            "metadata_filters": filters_dict,
            "max_results": body.max_results,
            "messages": [],
            "iteration_count": 0,
            "fast_mode": True,
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
            REQUEST_COUNT.labels(endpoint="/chat", status="500").inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred while executing the RAG pipeline. Please retry.",
            )

        latency_ms = (time.perf_counter() - start) * 1000

        # Build response
        response = ChatResponse(
            message_id=message_id,
            conversation_id=body.conversation_id,
            answer=result.get("answer", "No answer generated."),
            confidence_score=result.get("confidence_score", 0.0),
            sources=result.get("sources", []),
            sql_query=result.get("sql_query"),
            sql_results=result.get("sql_results"),
            route=result.get("route"),
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

        # Record Prometheus telemetry metrics
        REQUEST_COUNT.labels(endpoint="/chat", status="200").inc()
        REQUEST_LATENCY.labels(endpoint="/chat").observe(latency_ms / 1000.0)
        tokens_count = response.metrics.tokens_used or 0
        if tokens_count > 0:
            TOKENS_USED.labels(provider="groq", operation="chat").inc(tokens_count)

        # ----- 3. Cache the result -----
        if cache and response.confidence_score > 0.3:
            filters_dict = body.filters.model_dump(exclude_none=True)
            await cache.set(
                body.query,
                {
                    "answer": response.answer,
                    "confidence_score": response.confidence_score,
                    "sources": [s.model_dump() for s in response.sources],
                    "metrics": response.metrics.model_dump(),
                },
                filters=filters_dict,
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
    finally:
        ACTIVE_REQUESTS.dec()


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
