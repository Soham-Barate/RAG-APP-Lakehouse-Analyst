"""
Custom evaluation metrics beyond RAGAS.
"""

from __future__ import annotations

import statistics
from typing import Any


class CustomMetrics:
    """
    Custom metrics for production RAG monitoring.

    Tracks latency, retrieval precision, hallucination rate,
    cache efficiency, and token cost.
    """

    def __init__(self) -> None:
        self.latencies: list[float] = []
        self.retrieval_precisions: list[float] = []
        self.confidence_scores: list[float] = []
        self.tokens_used: list[int] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.total_requests: int = 0
        self.hallucination_flags: int = 0

    def record(
        self,
        latency_ms: float,
        confidence: float,
        tokens: int,
        cache_hit: bool,
        retrieval_precision: float = 0.0,
        is_hallucination: bool = False,
    ) -> None:
        """Record metrics for a single request."""
        self.latencies.append(latency_ms)
        self.confidence_scores.append(confidence)
        self.tokens_used.append(tokens)
        self.retrieval_precisions.append(retrieval_precision)
        self.total_requests += 1

        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        if is_hallucination:
            self.hallucination_flags += 1

    def compute_report(self) -> dict[str, Any]:
        """Compute all metrics and return a summary report."""
        if not self.latencies:
            return {"error": "No data recorded"}

        report = {
            "total_requests": self.total_requests,
            "latency": {
                "p50_ms": self._percentile(self.latencies, 50),
                "p95_ms": self._percentile(self.latencies, 95),
                "p99_ms": self._percentile(self.latencies, 99),
                "mean_ms": statistics.mean(self.latencies),
            },
            "confidence": {
                "mean": statistics.mean(self.confidence_scores),
                "min": min(self.confidence_scores),
                "max": max(self.confidence_scores),
            },
            "retrieval_precision": {
                "mean": statistics.mean(self.retrieval_precisions) if self.retrieval_precisions else 0,
            },
            "cache": {
                "hit_rate": self.cache_hits / max(self.total_requests, 1),
                "hits": self.cache_hits,
                "misses": self.cache_misses,
            },
            "tokens": {
                "total": sum(self.tokens_used),
                "mean_per_request": statistics.mean(self.tokens_used) if self.tokens_used else 0,
                "estimated_cost_usd": sum(self.tokens_used) * 10 / 1_000_000,
            },
            "hallucination_rate": self.hallucination_flags / max(self.total_requests, 1),
        }

        return report

    @staticmethod
    def _percentile(data: list[float], p: int) -> float:
        """Compute the p-th percentile."""
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d = k - f
        return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])

    def format_report(self) -> str:
        """Format the report as a readable string."""
        report = self.compute_report()
        if "error" in report:
            return report["error"]

        lines = [
            "=" * 55,
            "Custom Metrics Report",
            "=" * 55,
            f"  Total Requests: {report['total_requests']}",
            "",
            "  Latency:",
            f"    P50:  {report['latency']['p50_ms']:.0f}ms",
            f"    P95:  {report['latency']['p95_ms']:.0f}ms",
            f"    P99:  {report['latency']['p99_ms']:.0f}ms",
            f"    Mean: {report['latency']['mean_ms']:.0f}ms",
            "",
            "  Quality:",
            f"    Mean Confidence:        {report['confidence']['mean']:.4f}",
            f"    Retrieval Precision:    {report['retrieval_precision']['mean']:.4f}",
            f"    Hallucination Rate:     {report['hallucination_rate']:.4f}",
            "",
            "  Cache:",
            f"    Hit Rate:    {report['cache']['hit_rate']:.2%}",
            f"    Hits/Misses: {report['cache']['hits']}/{report['cache']['misses']}",
            "",
            "  Cost:",
            f"    Total Tokens: {report['tokens']['total']:,}",
            f"    Mean/Request: {report['tokens']['mean_per_request']:,.0f}",
            f"    Est. Cost:    ${report['tokens']['estimated_cost_usd']:.4f}",
            "=" * 55,
        ]

        return "\n".join(lines)
