"""
Response Generator Node.

Produces the final answer using compressed context, conversation
history, and the structured generation prompt with citations.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.core.schemas import DocumentType, SourceDocument
from backend.rag.prompts.templates import GENERATOR_PROMPT, SYSTEM_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """
    Final response generation with structured output.

    Produces an answer with inline citations, confidence score, and
    source document references. Supports both synchronous and streaming
    modes.
    """

    def __init__(
        self,
        llm_service: Any,
        confidence_floor: float = 0.3,
    ) -> None:
        self.llm_service = llm_service
        self.confidence_floor = confidence_floor

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Generate the final response."""
        start = time.perf_counter()

        query = state["query"]
        context = state.get("compressed_context", "No context available.")
        docs = state.get("reranked_docs", [])
        relevance_scores = state.get("relevance_scores", [])
        history = self._format_history(state.get("messages", []))

        # Build the generation prompt
        prompt = GENERATOR_PROMPT.format(
            context=context,
            history=history,
            query=query,
        )

        try:
            answer = await self.llm_service.generate(
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=2048,
            )

            # Estimate confidence from relevance scores and answer quality
            confidence = self._estimate_confidence(relevance_scores, answer)

            # Build source documents
            sources = self._build_sources(docs, relevance_scores)

            # Add disclaimer for low confidence
            disclaimer = None
            if confidence < self.confidence_floor:
                disclaimer = (
                    "⚠️ I have low confidence in this answer. The retrieved "
                    "context may not fully address your question. Please verify "
                    "the information independently."
                )

            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Generated response: %d chars, confidence=%.2f, sources=%d (%.1fms)",
                len(answer),
                confidence,
                len(sources),
                latency_ms,
            )

            return {
                "answer": answer.strip(),
                "confidence_score": confidence,
                "sources": sources,
                "disclaimer": disclaimer,
                "generation_latency_ms": round(latency_ms, 2),
                "tokens_used": self.llm_service.last_token_count,
            }

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.error("Generation failed: %s", exc)
            return {
                "answer": (
                    "I apologize, but I encountered an error generating a response. "
                    "Please try rephrasing your question or try again later."
                ),
                "confidence_score": 0.0,
                "sources": [],
                "disclaimer": "An error occurred during response generation.",
                "generation_latency_ms": round(latency_ms, 2),
                "tokens_used": 0,
            }

    def _estimate_confidence(
        self,
        relevance_scores: list[float],
        answer: str,
    ) -> float:
        """
        Estimate response confidence from multiple signals.

        Factors:
        - Average document relevance score
        - Whether the answer contains hedging language
        - Answer length (very short = low confidence)
        """
        if not relevance_scores:
            return 0.2

        avg_relevance = sum(relevance_scores) / len(relevance_scores)

        # Hedging penalty
        hedging_phrases = [
            "i'm not sure", "i don't have", "no information",
            "cannot find", "not enough data", "unclear",
        ]
        hedging_penalty = sum(
            0.05 for phrase in hedging_phrases if phrase in answer.lower()
        )

        # Length bonus (very short answers are suspicious)
        length_factor = min(1.0, len(answer) / 200)

        confidence = (
            avg_relevance * 0.6
            + length_factor * 0.2
            + 0.2  # base confidence
            - hedging_penalty
        )

        return max(0.0, min(1.0, confidence))

    def _build_sources(
        self,
        docs: list[dict[str, Any]],
        relevance_scores: list[float],
    ) -> list[SourceDocument]:
        """Build SourceDocument objects from retrieved documents."""
        sources: list[SourceDocument] = []

        for i, doc in enumerate(docs):
            metadata = doc.get("metadata", {})
            doc_type_str = metadata.get("document_type", "product")

            try:
                doc_type = DocumentType(doc_type_str)
            except ValueError:
                doc_type = DocumentType.PRODUCT

            relevance = (
                relevance_scores[i] if i < len(relevance_scores)
                else doc.get("rerank_score", 0.5)
            )

            sources.append(
                SourceDocument(
                    document_id=doc.get("id", f"doc_{i}"),
                    content=doc.get("content", "")[:500],  # Truncate for response
                    document_type=doc_type,
                    relevance_score=min(1.0, max(0.0, relevance)),
                    metadata={
                        k: v for k, v in metadata.items()
                        if k not in ("embedding", "vector")
                    },
                )
            )

        return sources

    def _format_history(self, messages: list[Any]) -> str:
        """Format conversation history for the prompt."""
        if not messages:
            return "No previous conversation."

        parts: list[str] = []
        for msg in messages[-6:]:  # Last 3 turns
            role = getattr(msg, "type", "unknown")
            content = getattr(msg, "content", str(msg))
            parts.append(f"{role}: {content}")

        return "\n".join(parts)
