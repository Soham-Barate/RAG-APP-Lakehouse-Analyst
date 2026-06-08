"""
Query Router Node.

Classifies the user query into sql / vector / hybrid to determine
the optimal retrieval strategy.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.rag.prompts.templates import ROUTER_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class QueryRouter:
    """LLM-based query router that classifies queries for optimal retrieval."""

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Classify the query and set the route in state."""
        query = state["query"]

        try:
            prompt = ROUTER_PROMPT.format(query=query)
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=100,
                json_mode=True,
            )

            result = json.loads(response)
            route = result.get("route", "hybrid")

            if route not in ("sql", "vector", "hybrid"):
                route = "hybrid"

            logger.info(
                "Query routed: %s -> %s (reason: %s)",
                query[:50],
                route,
                result.get("reasoning", "N/A"),
            )

            return {"route": route}

        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Router parse error, defaulting to hybrid: %s", exc)
            return {"route": "hybrid"}
        except Exception as exc:
            logger.error("Router failed, defaulting to hybrid: %s", exc)
            return {"route": "hybrid"}
