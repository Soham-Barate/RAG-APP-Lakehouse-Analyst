"""
Response Generator Node.

Produces the final answer using compressed context, conversation
history, and the structured generation prompt with citations.
"""

from __future__ import annotations

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
        if not docs and state.get("sql_doc"):
            docs = [state["sql_doc"]]
            relevance_scores = [1.0]

        history = self._format_history(state.get("messages", []))

        dataset_id = state.get("dataset_id") or "demo_lakehouse"
        dataset_label = (
            "E-Commerce Lakehouse (Demo: products, orders, customers)"
            if dataset_id == "demo_lakehouse"
            else f"User Uploaded Dataset '{dataset_id}'"
        )
        context_with_scope = (
            f"### Active Query Target\n"
            f"- Active Dataset: **{dataset_label}**\n"
            f"- Dataset Identifier: `{dataset_id}`\n\n"
            f"{context}"
        )

        # Build the generation prompt
        prompt = GENERATOR_PROMPT.format(
            context=context_with_scope,
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

            # Quality Safeguard: prevent "code block only" or empty responses
            cleaned_answer = answer.strip()
            sql_results = state.get("sql_results", [])
            sql_error = state.get("sql_error")

            is_bare_code_block = (
                (cleaned_answer.startswith("```") and cleaned_answer.endswith("```") and cleaned_answer.count("```") == 2)
                or len(cleaned_answer) < 30
            )

            if is_bare_code_block:
                if sql_results and len(sql_results) > 0:
                    import pandas as pd
                    sample_df = pd.DataFrame(sql_results[:10])
                    sample_table = sample_df.to_markdown(index=False)
                    first_col = sample_df.columns[0]
                    cleaned_answer = (
                        f"### Executive Analysis Summary\n\n"
                        f"Analysis executed successfully against the active dataset ({len(sql_results):,} matching records found):\n\n"
                        f"{sample_table}\n\n"
                        f"**Key Analytical Findings:**\n"
                        f"- Total records evaluated: **{len(sql_results):,}**\n"
                        f"- Primary dimension: **{first_col}**\n"
                        f"- Would you like to break this down by another dimension or filter by date range?"
                    )
                elif sql_error:
                    cleaned_answer = (
                        f"### Structured Query Notice\n\n"
                        f"The requested analytical query could not be executed against the active dataset schema:\n"
                        f"> `{sql_error}`\n\n"
                        f"**Suggestions:**\n"
                        f"- Please verify the target column names in the active dataset schema.\n"
                        f"- For example, you can query by `category`, `price`, `rating`, `total_amount`, or `status`."
                    )
                elif not sql_results and state.get("sql_query"):
                    cleaned_answer = (
                        f"### Query Completed (Zero Records Matched)\n\n"
                        f"The analytical query executed successfully, but zero records matched your specific filter criteria in the current dataset.\n\n"
                        f"**Suggestions:**\n"
                        f"- Try widening your date range or adjusting filter parameters.\n"
                        f"- Ask a broader question such as *'Show top categories by order volume'*."
                    )
            answer = cleaned_answer

            out_dict = {
                "answer": answer,
                "confidence_score": confidence,
                "sources": sources,
                "disclaimer": disclaimer,
                "generation_latency_ms": round(latency_ms, 2),
                "tokens_used": self.llm_service.last_token_count,
            }
            if "sql_query" in state:
                out_dict["sql_query"] = state["sql_query"]
            if "sql_results" in state:
                out_dict["sql_results"] = state["sql_results"]
            if "sql_error" in state:
                out_dict["sql_error"] = state["sql_error"]

            return out_dict

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
