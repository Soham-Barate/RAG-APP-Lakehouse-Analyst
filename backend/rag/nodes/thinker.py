"""
Chain-of-Thought Reasoning Node.

Generates an internal reasoning trace *before* final answer generation.
The trace walks through retrieved context step-by-step, identifies key
facts, resolves contradictions, and outlines the answer strategy.

The reasoning is stored in state for debugging/evaluation but is NOT
shown to the end user — it feeds into the generator as hidden context.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.rag.prompts.templates import CHAIN_OF_THOUGHT_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class ChainOfThoughtReasoner:
    """
    Internal reasoning engine that thinks before answering.

    By generating an explicit reasoning trace, the LLM is forced to:
    1. Enumerate the facts from each source
    2. Identify what's relevant vs. noise
    3. Spot contradictions between sources
    4. Plan the answer structure

    This dramatically reduces hallucinations and improves coherence.
    """

    def __init__(self, llm_service: Any) -> None:
        self.llm_service = llm_service

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Generate internal reasoning trace from compressed context."""
        start = time.perf_counter()

        query = state["query"]
        context = state.get("compressed_context", "No context available.")

        try:
            prompt = CHAIN_OF_THOUGHT_PROMPT.format(
                query=query,
                context=context,
            )

            reasoning = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024,
            )

            latency_ms = (time.perf_counter() - start) * 1000

            logger.info(
                "Chain-of-thought: %d chars reasoning (%.1fms)",
                len(reasoning),
                latency_ms,
            )

            return {"reasoning_trace": reasoning.strip()}

        except Exception as exc:
            logger.warning("Chain-of-thought failed, skipping: %s", exc)
            return {"reasoning_trace": ""}
