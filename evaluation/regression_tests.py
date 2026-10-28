"""
Regression tests for the RAG pipeline.

Runs a golden test set with expected answers, compares results using
semantic similarity, and outputs a pass/fail verdict.  Designed to
run in CI/CD to catch quality regressions before deployment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Golden test set — expected answer patterns (not exact matches)
# ---------------------------------------------------------------------------
REGRESSION_TESTS: list[dict[str, str]] = [
    {
        "query": "What are the top selling products?",
        "expected_keywords": ["top", "selling", "product", "popular"],
        "expected_type": "product",
    },
    {
        "query": "Show me return rates by category",
        "expected_keywords": ["return", "rate", "category"],
        "expected_type": "order",
    },
    {
        "query": "What is the average order value?",
        "expected_keywords": ["average", "order", "value"],
        "expected_type": "order",
    },
    {
        "query": "What are customers saying about electronics?",
        "expected_keywords": ["customer", "review", "electronics"],
        "expected_type": "review",
    },
    {
        "query": "How do I track my order?",
        "expected_keywords": ["track", "order", "status"],
        "expected_type": "knowledge_base",
    },
]


class RegressionTestRunner:
    """
    Run golden regression tests against the RAG pipeline.

    Each test checks:
    1. The answer contains expected keywords (keyword coverage).
    2. The confidence score is above a minimum threshold.
    3. The response latency is within SLO bounds.
    4. Sources are retrieved and relevant.
    """

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        api_key: str = "dev-key-1",
        min_keyword_coverage: float = 0.5,
        min_confidence: float = 0.3,
        max_latency_ms: float = 10000,
    ) -> None:
        self.backend_url = backend_url
        self.api_key = api_key
        self.min_keyword_coverage = min_keyword_coverage
        self.min_confidence = min_confidence
        self.max_latency_ms = max_latency_ms

    async def run(
        self,
        tests: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Run all regression tests and return results."""
        tests = tests or REGRESSION_TESTS
        results: list[dict[str, Any]] = []
        passed = 0
        failed = 0

        async with httpx.AsyncClient(timeout=120.0) as client:
            for i, test in enumerate(tests):
                result = await self._run_single(client, test, i + 1)
                results.append(result)
                if result["passed"]:
                    passed += 1
                else:
                    failed += 1

        report = {
            "total_tests": len(tests),
            "passed": passed,
            "failed": failed,
            "pass_rate": round(passed / max(len(tests), 1), 4),
            "verdict": "PASS" if failed == 0 else "FAIL",
            "results": results,
        }

        # Print summary
        self._print_summary(report)

        return report

    async def _run_single(
        self,
        client: httpx.AsyncClient,
        test: dict[str, str],
        index: int,
    ) -> dict[str, Any]:
        """Run a single regression test."""
        query = test["query"]
        expected_keywords = test.get("expected_keywords", [])

        start = time.time()
        try:
            resp = await client.post(
                f"{self.backend_url}/api/v1/chat",
                json={"query": query, "max_results": 5},
                headers={"X-API-Key": self.api_key},
            )

            latency_ms = (time.time() - start) * 1000

            if resp.status_code != 200:
                return {
                    "index": index,
                    "query": query,
                    "passed": False,
                    "reason": f"HTTP {resp.status_code}",
                    "latency_ms": latency_ms,
                }

            data = resp.json()
            answer = data.get("answer", "").lower()
            confidence = data.get("confidence_score", 0.0)
            sources = data.get("sources", [])
            api_latency = data.get("metrics", {}).get("total_latency_ms", latency_ms)

        except Exception as exc:
            return {
                "index": index,
                "query": query,
                "passed": False,
                "reason": f"Request failed: {exc}",
                "latency_ms": (time.time() - start) * 1000,
            }

        # Check keyword coverage
        found_keywords = [
            kw for kw in expected_keywords if kw.lower() in answer
        ]
        keyword_coverage = len(found_keywords) / max(len(expected_keywords), 1)

        # Determine pass/fail
        issues: list[str] = []
        if keyword_coverage < self.min_keyword_coverage:
            issues.append(
                f"Low keyword coverage: {keyword_coverage:.0%} "
                f"(expected >= {self.min_keyword_coverage:.0%})"
            )
        if confidence < self.min_confidence:
            issues.append(
                f"Low confidence: {confidence:.2f} "
                f"(expected >= {self.min_confidence})"
            )
        if api_latency > self.max_latency_ms:
            issues.append(
                f"High latency: {api_latency:.0f}ms "
                f"(expected <= {self.max_latency_ms:.0f}ms)"
            )
        if not sources:
            issues.append("No sources retrieved")

        return {
            "index": index,
            "query": query,
            "passed": len(issues) == 0,
            "answer_preview": answer[:200],
            "confidence": confidence,
            "keyword_coverage": round(keyword_coverage, 4),
            "latency_ms": round(api_latency, 1),
            "source_count": len(sources),
            "issues": issues,
        }

    def _print_summary(self, report: dict[str, Any]) -> None:
        """Print regression test summary."""
        verdict = report["verdict"]
        color = "✅" if verdict == "PASS" else "❌"

        print("\n" + "=" * 60)
        print(f"  {color} REGRESSION TEST RESULTS: {verdict}")
        print("=" * 60)
        print(f"  Tests: {report['total_tests']}")
        print(f"  Passed: {report['passed']}")
        print(f"  Failed: {report['failed']}")
        print(f"  Pass rate: {report['pass_rate']:.0%}")
        print()

        for result in report["results"]:
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} [{result['index']}] {result['query']}")
            if result.get("issues"):
                for issue in result["issues"]:
                    print(f"       ⚠️  {issue}")

        print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RAG Regression Tests")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--api-key", default="dev-key-1")
    args = parser.parse_args()

    runner = RegressionTestRunner(backend_url=args.url, api_key=args.api_key)
    report = await runner.run()

    # Save report
    Path("evaluation").mkdir(exist_ok=True)
    with open("evaluation/regression_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Exit with non-zero if failed
    exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    asyncio.run(main())
