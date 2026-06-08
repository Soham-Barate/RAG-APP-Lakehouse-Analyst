"""
Relevance Grader Node.

Evaluates whether the retrieved (and reranked) documents contain
enough relevant information to generate a good answer, or whether
the pipeline should retry with rewritten queries.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from backend.rag.prompts.templates import GRADER_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class RelevanceGrader:
    """
    LLM-based relevance grader with threshold-based decision making.

    Grades each document as relevant/partially_relevant/irrelevant,
    computes an aggregate relevance score, and decides whether to
    proceed to generation or retry retrieval.
    """

    def __init__(
        self,
        llm_service: Any,
        relevance_threshold: float = 0.4,
        max_iterations: int = 3,
    ) -> None:
        self.llm_service = llm_service
        self.relevance_threshold = relevance_threshold
        self.max_iterations = max_iterations

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Grade documents and decide whether to proceed or retry."""
        docs = state.get("reranked_docs", [])
        query = state["query"]
        iteration = state.get("iteration_count", 0)

        if not docs:
            return {
                "relevance_scores": [],
                "should_retry": iteration < self.max_iterations,
                "iteration_count": iteration + 1,
            }

        scores: list[float] = []
        grades: list[str] = []

        # Grade each document
        for doc in docs:
            score, grade = await self._grade_document(query, doc)
            scores.append(score)
            grades.append(grade)

        # Compute aggregate relevance
        avg_score = sum(scores) / len(scores) if scores else 0.0
        relevant_count = sum(1 for g in grades if g == "relevant")
        relevant_ratio = relevant_count / len(grades) if grades else 0.0

        # Decision: retry if relevance is below threshold and iterations remain
        should_retry = (
            avg_score < self.relevance_threshold
            and relevant_ratio < 0.4
            and iteration < self.max_iterations
        )

        logger.info(
            "Grading: avg_score=%.2f, relevant=%d/%d (%.0f%%), "
            "should_retry=%s, iteration=%d",
            avg_score,
            relevant_count,
            len(grades),
            relevant_ratio * 100,
            should_retry,
            iteration,
        )

        return {
            "relevance_scores": scores,
            "should_retry": should_retry,
            "iteration_count": iteration + 1,
        }

    async def _grade_document(
        self,
        query: str,
        doc: dict[str, Any],
    ) -> tuple[float, str]:
        """Grade a single document for relevance."""
        content = doc.get("content", "")[:2000]  # Limit to prevent token overflows

        try:
            prompt = GRADER_PROMPT.format(query=query, document=content)
            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=150,
                json_mode=True,
            )

            result = json.loads(response)
            score = float(result.get("score", 0.5))
            grade = result.get("relevance", "partially_relevant")

            return score, grade

        except Exception as exc:
            logger.warning("Grading failed for document, using default: %s", exc)
            # Use the reranker score as a proxy
            rerank_score = doc.get("rerank_score", 0.5)
            grade = "relevant" if rerank_score > 0.7 else "partially_relevant"
            return rerank_score, grade
