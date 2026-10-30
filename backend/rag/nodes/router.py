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
        q_lower = query.lower().strip()

        # Deterministic fast-path: identity, dataset metadata, and greetings do not require SQL
        meta_phrases = (
            "which dataset", "what dataset", "name it", "dataset name",
            "active dataset", "which database", "what database", "current database",
            "who are you", "what can you do", "hello", "hi", "help"
        )
        if any(p in q_lower for p in meta_phrases) and not any(w in q_lower for w in ("row", "rows", "select", "table", "data", "count", "product", "order")):
            logger.info("Query matched metadata fast-path: %s -> vector", query[:50])
            return {"route": "vector"}

        # Deterministic SQL fast-path: raw row inspection, limits, frequency, aggregates, table queries
        import re
        sql_keywords = (
            "first five", "first 5", "first 10", "first 20", "top 5", "top 10", "top 20",
            "give me rows", "first rows", "sample rows", "preview rows", "show rows",
            "first few rows", "sample records", "show records", "first records",
            "occurred", "occur", "occurrence", "frequency", "most frequent", "most common",
            "most ordered", "best selling", "top selling", "highest price", "lowest price",
            "highest rated", "lowest rated", "total revenue", "total orders", "total spent",
            "average price", "avg price", "average order", "avg order", "how many orders",
            "how many products", "how many customers", "order count", "customer count",
            "product count", "orders table", "products table", "customers table",
            "breakdown by", "group by", "order by"
        )
        if any(ind in q_lower for ind in sql_keywords) or re.search(r"\b(first|top|sample|give me|show)\s+(\d+|five|ten|few)\s+(rows?|records?)", q_lower) or re.search(r"\b\d+\s+rows?\b", q_lower):
            logger.info("Query matched SQL analytical fast-path: %s -> sql", query[:50])
            return {"route": "sql"}

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
