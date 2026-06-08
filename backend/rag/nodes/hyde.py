"""
HyDE (Hypothetical Document Embedding) Node.

Generates a hypothetical answer document and embeds it alongside
the real query to improve retrieval recall on abstract queries.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.rag.prompts.templates import HYDE_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class HyDEGenerator:
    """
    Generates a hypothetical document that would perfectly answer the query.

    The embedding of this document is used alongside the original query
    embedding during retrieval to capture semantic nuances that the
    raw query might miss.
    """

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Generate a hypothetical document for the query."""
        query = state["query"]

        try:
            prompt = HYDE_PROMPT.format(query=query)
            hyde_doc = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.7,  # Higher temp for creative generation
                max_tokens=300,
            )

            logger.info(
                "HyDE document generated: %d chars",
                len(hyde_doc),
            )

            return {"hyde_document": hyde_doc.strip()}

        except Exception as exc:
            logger.warning("HyDE generation failed, skipping: %s", exc)
            # If HyDE fails, the retriever will still work with rewritten queries
            return {"hyde_document": ""}
