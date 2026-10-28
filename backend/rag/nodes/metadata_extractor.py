"""
Automatic Metadata Extractor Node.

Analyses the user query to automatically extract metadata filters
(categories, date ranges, product names, document types) so users
don't need to manually set filters via the UI.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.rag.state import RAGState

logger = logging.getLogger(__name__)

METADATA_EXTRACTION_PROMPT = """Analyze the following user query and extract any implicit metadata filters that would help narrow the search.

User query: {query}

Extract the following if mentioned or implied:
1. **categories**: Product categories (e.g., Electronics, Clothing, Home & Garden)
2. **document_types**: Type of data being asked about (product, order, review, ticket, customer)
3. **date_from / date_to**: Any date ranges mentioned (ISO format)
4. **departments**: Business departments mentioned

If a filter is not mentioned or implied, set it to null.

Respond with ONLY a JSON object:
{{"categories": ["cat1"] | null, "document_types": ["type1"] | null, "date_from": "YYYY-MM-DD" | null, "date_to": "YYYY-MM-DD" | null, "departments": ["dept"] | null, "reasoning": "brief explanation"}}"""


class MetadataExtractor:
    """
    Auto-extracts metadata filters from natural language queries.

    Instead of requiring users to manually select "Electronics" from
    a dropdown, this node detects "What are the best electronics?"
    and auto-populates the category filter.
    """

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Extract metadata filters from the query."""
        query = state["query"]
        existing_filters = state.get("metadata_filters", {})

        # Skip if user already provided filters
        has_user_filters = any(
            v for k, v in existing_filters.items()
            if k != "reasoning" and v
        )
        if has_user_filters:
            return {}

        try:
            prompt = METADATA_EXTRACTION_PROMPT.format(query=query)
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=200,
                json_mode=True,
            )

            result = json.loads(response)

            # Merge extracted filters with existing (user filters take priority)
            merged: dict[str, Any] = {}
            for key in ("categories", "document_types", "date_from", "date_to", "departments"):
                user_val = existing_filters.get(key)
                extracted_val = result.get(key)
                merged[key] = user_val if user_val else extracted_val

            # Remove None values
            merged = {k: v for k, v in merged.items() if v}

            if merged:
                logger.info(
                    "Auto-extracted filters: %s (reason: %s)",
                    list(merged.keys()),
                    result.get("reasoning", "N/A"),
                )

            return {"metadata_filters": merged} if merged else {}

        except Exception as exc:
            logger.warning("Metadata extraction failed, skipping: %s", exc)
            return {}
