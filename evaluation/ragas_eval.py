"""
RAGAS evaluation suite for the RAG pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def run_ragas_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run RAGAS evaluation on a set of QA pairs.

    Metrics:
    - Faithfulness: Are claims in the answer grounded in context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved docs actually relevant?
    - Context Recall: Is the retrieval complete?

    Returns a dict with metric scores.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        # Build evaluation dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }

        if ground_truths:
            data["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data)

        # Select metrics
        metrics = [faithfulness, answer_relevancy, context_precision]
        if ground_truths:
            metrics.append(context_recall)

        # Run evaluation
        result = evaluate(dataset=dataset, metrics=metrics)

        scores = {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
        }

        if ground_truths:
            scores["context_recall"] = float(result.get("context_recall", 0.0))

        # Compute overall score
        scores["overall"] = sum(scores.values()) / len(scores)

        logger.info("RAGAS evaluation complete: %s", scores)
        return scores

    except ImportError:
        logger.warning("RAGAS not installed. Install with: pip install ragas")
        return {"error": "RAGAS not installed"}
    except Exception as exc:
        logger.error("RAGAS evaluation failed: %s", exc)
        return {"error": str(exc)}


def format_ragas_report(scores: dict[str, Any]) -> str:
    """Format RAGAS scores as a readable report."""
    if "error" in scores:
        return f"Evaluation Error: {scores['error']}"

    lines = [
        "=" * 50,
        "RAGAS Evaluation Report",
        "=" * 50,
        "",
        f"  Faithfulness:      {scores.get('faithfulness', 0):.4f}",
        f"  Answer Relevancy:  {scores.get('answer_relevancy', 0):.4f}",
        f"  Context Precision: {scores.get('context_precision', 0):.4f}",
    ]

    if "context_recall" in scores:
        lines.append(f"  Context Recall:    {scores['context_recall']:.4f}")

    lines.extend([
        "",
        "-" * 50,
        f"  Overall Score:     {scores.get('overall', 0):.4f}",
        "=" * 50,
    ])

    return "\n".join(lines)
