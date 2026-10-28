"""
Contextual Compressor Node.

Extracts only query-relevant portions from retrieved documents,
removes redundancy, and produces a dense context string within
the token budget.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.rag.prompts.templates import COMPRESSOR_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    LLM-based contextual compression.

    Takes reranked documents and extracts only the portions that are
    directly relevant to the user's query. This reduces noise, removes
    redundancy across documents, and keeps the generation context
    within budget.
    """

    def __init__(
        self,
        llm_service: Any,
        max_context_tokens: int = 4000,
    ) -> None:
        self.llm_service = llm_service
        self.max_context_tokens = max_context_tokens

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Compress reranked documents into a focused context string."""
        docs = state.get("reranked_docs", [])
        query = state["query"]
        sql_context = state.get("sql_context")

        if not docs:
            if sql_context:
                return {"compressed_context": sql_context}
            return {"compressed_context": "No relevant documents found."}

        # Format documents for the compressor prompt
        docs_text = self._format_documents(docs)

        try:
            prompt = COMPRESSOR_PROMPT.format(
                query=query,
                documents=docs_text,
                max_tokens=self.max_context_tokens,
            )

            compressed = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.max_context_tokens,
            )

            logger.info(
                "Context compressed: %d chars -> %d chars (%.0f%% reduction)",
                len(docs_text),
                len(compressed),
                (1 - len(compressed) / max(len(docs_text), 1)) * 100,
            )

            final_text = compressed.strip()
            if sql_context:
                final_text = f"{sql_context}\n\n### Qualitative Product & Knowledge Context\n{final_text}"

            return {"compressed_context": final_text}

        except Exception as exc:
            logger.warning("Compression failed, using raw context: %s", exc)
            # Fallback: concatenate document contents directly
            fallback = self._create_fallback_context(docs)
            if sql_context:
                fallback = f"{sql_context}\n\n### Qualitative Product & Knowledge Context\n{fallback}"
            return {"compressed_context": fallback}

    def _format_documents(self, docs: list[dict[str, Any]]) -> str:
        """Format documents for the compressor prompt."""
        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            doc_type = metadata.get("document_type", "unknown")
            score = doc.get("rerank_score", doc.get("rrf_score", 0.0))

            parts.append(
                f"[Source {i}] (Type: {doc_type}, Relevance: {score:.2f})\n{content}"
            )

        return "\n\n---\n\n".join(parts)

    def _create_fallback_context(
        self,
        docs: list[dict[str, Any]],
        max_chars: int = 8000,
    ) -> str:
        """Create a simple concatenated context as fallback."""
        parts: list[str] = []
        total_chars = 0

        for i, doc in enumerate(docs, 1):
            content = doc.get("content", "")
            entry = f"[Source {i}] {content}"

            if total_chars + len(entry) > max_chars:
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n\n".join(parts)
