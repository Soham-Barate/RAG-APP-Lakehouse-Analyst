"""
Automated quality gates for CI/CD.

Defines minimum thresholds for deployment readiness.
Returns a clear PASS/FAIL verdict with detailed breakdowns.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deployment quality thresholds
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = {
    "faithfulness_min": 0.80,
    "answer_relevancy_min": 0.70,
    "context_precision_min": 0.65,
    "p95_latency_max_ms": 3000,
    "p99_latency_max_ms": 5000,
    "hallucination_rate_max": 0.05,       # 5%
    "mean_confidence_min": 0.50,
    "cache_hit_rate_min": 0.10,            # 10% after warmup
    "regression_pass_rate_min": 0.90,      # 90% of regression tests pass
}


class QualityGate:
    """
    Automated quality gate checker.

    Compares benchmark results against deployment thresholds and
    returns a pass/fail verdict with detailed reasoning.
    """

    def __init__(
        self,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

    def check(
        self,
        benchmark_report: dict[str, Any],
        regression_report: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Run all quality gates against benchmark results.

        Args:
            benchmark_report: Output from BenchmarkRunner.run()
            regression_report: Output from RegressionTestRunner.run()

        Returns:
            Gate results with overall PASS/FAIL verdict.
        """
        checks: list[dict[str, Any]] = []

        # ----- RAGAS scores -----
        ragas = benchmark_report.get("ragas_scores", {})

        checks.append(self._check(
            "Faithfulness",
            ragas.get("faithfulness", 0),
            self.thresholds["faithfulness_min"],
            ">=",
        ))
        checks.append(self._check(
            "Answer Relevancy",
            ragas.get("answer_relevancy", 0),
            self.thresholds["answer_relevancy_min"],
            ">=",
        ))
        checks.append(self._check(
            "Context Precision",
            ragas.get("context_precision", 0),
            self.thresholds["context_precision_min"],
            ">=",
        ))

        # ----- Custom metrics -----
        custom = benchmark_report.get("custom_metrics", {})
        latency = custom.get("latency", {})
        cache = custom.get("cache", {})
        confidence = custom.get("confidence", {})

        checks.append(self._check(
            "P95 Latency",
            latency.get("p95_ms", 0),
            self.thresholds["p95_latency_max_ms"],
            "<=",
            unit="ms",
        ))
        checks.append(self._check(
            "P99 Latency",
            latency.get("p99_ms", 0),
            self.thresholds["p99_latency_max_ms"],
            "<=",
            unit="ms",
        ))
        checks.append(self._check(
            "Hallucination Rate",
            custom.get("hallucination_rate", 0),
            self.thresholds["hallucination_rate_max"],
            "<=",
        ))
        checks.append(self._check(
            "Mean Confidence",
            confidence.get("mean", 0),
            self.thresholds["mean_confidence_min"],
            ">=",
        ))
        checks.append(self._check(
            "Cache Hit Rate",
            cache.get("hit_rate", 0),
            self.thresholds["cache_hit_rate_min"],
            ">=",
        ))

        # ----- Regression tests -----
        if regression_report:
            checks.append(self._check(
                "Regression Pass Rate",
                regression_report.get("pass_rate", 0),
                self.thresholds["regression_pass_rate_min"],
                ">=",
            ))

        # ----- Overall verdict -----
        all_passed = all(c["passed"] for c in checks)
        critical_failed = [c for c in checks if not c["passed"]]

        result = {
            "verdict": "PASS" if all_passed else "FAIL",
            "checks_total": len(checks),
            "checks_passed": sum(1 for c in checks if c["passed"]),
            "checks_failed": len(critical_failed),
            "checks": checks,
            "failed_checks": critical_failed,
        }

        # Print summary
        self._print(result)

        return result

    @staticmethod
    def _check(
        name: str,
        actual: float,
        threshold: float,
        direction: str = ">=",
        unit: str = "",
    ) -> dict[str, Any]:
        """Check a single metric against its threshold."""
        if direction == ">=":
            passed = actual >= threshold
        else:
            passed = actual <= threshold

        unit_str = f" {unit}" if unit else ""
        return {
            "name": name,
            "passed": passed,
            "actual": round(actual, 4),
            "threshold": threshold,
            "direction": direction,
            "detail": (
                f"{'✅' if passed else '❌'} {name}: "
                f"{actual:.4f}{unit_str} ({direction} {threshold}{unit_str})"
            ),
        }

    @staticmethod
    def _print(result: dict[str, Any]) -> None:
        """Print quality gate summary."""
        v = result["verdict"]
        icon = "✅" if v == "PASS" else "❌"
        print("\n" + "=" * 60)
        print(f"  {icon} QUALITY GATE: {v}")
        print("=" * 60)
        print(
            f"  Passed: {result['checks_passed']}/{result['checks_total']}"
        )
        print()
        for check in result["checks"]:
            print(f"  {check['detail']}")
        print("=" * 60)
