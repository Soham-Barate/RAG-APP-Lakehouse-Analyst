"""
Query Rewriter / Decomposer Node.

Expands, decomposes, or generalizes the user query to improve
retrieval recall.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.rag.prompts.templates import QUERY_REWRITER_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Rewrites the user query into multiple optimized search queries.

    Strategies:
    - Expansion: add synonyms and related terms
    - Decomposition: break multi-part questions into sub-queries
    - Step-back: generate broader queries for abstract topics
    """

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Rewrite the query and store alternatives in state."""
        query = state["query"]
        iteration = state.get("iteration_count", 0)

        # On retry iterations, add context about why we're retrying
        if iteration > 0:
            query = (
                f"[Retry #{iteration}: Previous retrieval was insufficient. "
                f"Try a different angle.] {query}"
            )

        try:
            prompt = QUERY_REWRITER_PROMPT.format(query=query)
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=300,
                json_mode=True,
            )

            result = json.loads(response)
            rewritten = result.get("rewritten_queries", [query])

            # Always include the original query
            if query not in rewritten:
                rewritten.insert(0, state["query"])

            # Limit to 4 queries to control cost
            rewritten = rewritten[:4]

            logger.info(
                "Query rewritten: %d variants (strategy: %s)",
                len(rewritten),
                result.get("strategy", "unknown"),
            )

            return {"rewritten_queries": rewritten}

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Rewriter parse error, using original: %s", exc)
            return {"rewritten_queries": [state["query"]]}
        except Exception as exc:
            logger.error("Rewriter failed, using original: %s", exc)
            return {"rewritten_queries": [state["query"]]}
