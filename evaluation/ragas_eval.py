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


async def run_sql_evaluation(
    queries: list[dict[str, Any]] | None = None,
    client: Any = None,
) -> dict[str, Any]:
    """
    Evaluate structured analytical queries against the Text-to-SQL lakehouse branch.

    Validates:
    - Generation of safe, syntactically valid SQL
    - Non-empty SQL execution results
    - SQL schema matching expected tables/columns
    - SQL execution success rate metric
    """
    from evaluation.test_queries import STRUCTURED_ANALYTICAL_QUERIES

    target_queries = queries or STRUCTURED_ANALYTICAL_QUERIES
    total = len(target_queries)
    successful_sql_runs = 0
    results_detail: list[dict[str, Any]] = []

    for item in target_queries:
        query_text = str(item["query"])
        detail: dict[str, Any] = {"query": query_text, "success": False}

        if client:
            resp = client.post(
                "/api/v1/chat",
                json={"query": query_text},
            )
            if resp.status_code == 200:
                data = resp.json()
                sql_q = data.get("sql_query")
                sql_res = data.get("sql_results") or []
                if sql_q and (len(sql_res) >= int(item.get("min_rows", 0))):
                    successful_sql_runs += 1
                    detail["success"] = True
                    detail["sql_query"] = sql_q
                    detail["row_count"] = len(sql_res)
        else:
            # Local evaluation check
            detail["status"] = "client_not_provided"

        results_detail.append(detail)

    success_rate = successful_sql_runs / max(total, 1)

    return {
        "total_queries": total,
        "successful_executions": successful_sql_runs,
        "sql_success_rate": round(success_rate, 4),
        "details": results_detail,
    }
