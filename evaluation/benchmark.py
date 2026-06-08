"""
End-to-end benchmark runner.

Runs the full test suite against the live RAG pipeline, collects
RAGAS + custom metrics, and generates a comprehensive report.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

from evaluation.custom_metrics import CustomMetrics
from evaluation.ragas_eval import format_ragas_report, run_ragas_evaluation
from evaluation.test_queries import GOLDEN_QUERIES, QUICK_TEST_QUERIES

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """End-to-end RAG pipeline benchmark."""

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        api_key: str = "dev-key-1",
    ) -> None:
        self.backend_url = backend_url
        self.api_key = api_key
        self.metrics = CustomMetrics()
        self.results: list[dict[str, Any]] = []

    async def run(
        self,
        queries: list[dict[str, object]] | None = None,
        quick: bool = False,
    ) -> dict[str, Any]:
        """
        Run the benchmark suite.

        Args:
            queries: Custom query set (defaults to golden queries)
            quick: Use quick test subset (10 queries)

        Returns:
            Complete benchmark report
        """
        if queries is None:
            queries = QUICK_TEST_QUERIES if quick else GOLDEN_QUERIES

        logger.info("Starting benchmark with %d queries", len(queries))
        start = time.time()

        questions: list[str] = []
        answers: list[str] = []
        contexts: list[list[str]] = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, q in enumerate(queries):
                query = str(q["query"])
                logger.info(
                    "Query %d/%d: %s", i + 1, len(queries), query[:60]
                )

                result = await self._run_single(client, query)
                self.results.append(result)

                # Collect for RAGAS
                questions.append(query)
                answers.append(result.get("answer", ""))
                source_texts = [
                    s.get("content", "") for s in result.get("sources", [])
                ]
                contexts.append(source_texts if source_texts else ["No context"])

                # Record custom metrics
                metrics = result.get("metrics", {})
                self.metrics.record(
                    latency_ms=metrics.get("total_latency_ms", 0),
                    confidence=result.get("confidence_score", 0),
                    tokens=metrics.get("tokens_used", 0),
                    cache_hit=metrics.get("cache_hit", False),
                )

        total_duration = time.time() - start

        # Run RAGAS evaluation
        logger.info("Running RAGAS evaluation...")
        ragas_scores = await run_ragas_evaluation(
            questions=questions,
            answers=answers,
            contexts=contexts,
        )

        # Compile report
        report = {
            "benchmark_duration_seconds": round(total_duration, 2),
            "total_queries": len(queries),
            "ragas_scores": ragas_scores,
            "custom_metrics": self.metrics.compute_report(),
            "per_query_results": self.results,
        }

        # Print report
        self._print_report(report)

        # Save report
        report_path = Path("evaluation") / "benchmark_report.json"
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Report saved to %s", report_path)

        return report

    async def _run_single(
        self,
        client: httpx.AsyncClient,
        query: str,
    ) -> dict[str, Any]:
        """Run a single query against the backend."""
        try:
            resp = await client.post(
                f"{self.backend_url}/api/v1/chat",
                json={"query": query, "max_results": 5},
                headers={"X-API-Key": self.api_key},
            )

            if resp.status_code == 200:
                return resp.json()
            else:
                return {
                    "answer": f"Error: {resp.status_code}",
                    "confidence_score": 0.0,
                    "sources": [],
                    "metrics": {},
                }
        except Exception as exc:
            return {
                "answer": f"Error: {exc}",
                "confidence_score": 0.0,
                "sources": [],
                "metrics": {},
            }

    def _print_report(self, report: dict[str, Any]) -> None:
        """Print the benchmark report to console."""
        print("\n" + "=" * 60)
        print("  E-COMMERCE RAG BENCHMARK REPORT")
        print("=" * 60)
        print(f"\n  Duration: {report['benchmark_duration_seconds']:.1f}s")
        print(f"  Queries:  {report['total_queries']}")
        print()

        # RAGAS scores
        print(format_ragas_report(report["ragas_scores"]))
        print()

        # Custom metrics
        print(self.metrics.format_report())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    """Run benchmark from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Benchmark Runner")
    parser.add_argument("--quick", action="store_true", help="Quick test (10 queries)")
    parser.add_argument("--url", default="http://localhost:8000", help="Backend URL")
    parser.add_argument("--api-key", default="dev-key-1", help="API key")
    args = parser.parse_args()

    runner = BenchmarkRunner(backend_url=args.url, api_key=args.api_key)
    await runner.run(quick=args.quick)


if __name__ == "__main__":
    asyncio.run(main())
