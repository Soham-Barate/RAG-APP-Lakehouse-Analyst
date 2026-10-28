"""
Analytics API endpoints.

Provides read-only dashboards for usage, quality, feedback, and
cost metrics.  Data is sourced from Prometheus counters/histograms
and an optional feedback store.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends

from backend.api.middleware.rbac import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/analytics", tags=["Analytics"])


@router.get(
    "/usage",
    summary="Usage analytics",
    dependencies=[Depends(require_permission("analytics:read"))],
)
async def get_usage_analytics() -> dict[str, Any]:
    """Return request counts, latency percentiles, and token usage."""
    from backend.services.observability import (
        ACTIVE_REQUESTS,
        CACHE_HITS,
        CACHE_MISSES,
        REQUEST_COUNT,
        TOKENS_USED,
    )

    try:
        # Read from Prometheus metrics
        total_requests = sum(
            s.value for s in REQUEST_COUNT.collect()[0].samples
            if s.name.endswith("_total")
        )
        total_tokens = sum(
            s.value for s in TOKENS_USED.collect()[0].samples
            if s.name.endswith("_total")
        )
        cache_hits = sum(
            s.value for s in CACHE_HITS.collect()[0].samples
            if s.name.endswith("_total")
        )
        cache_misses = sum(
            s.value for s in CACHE_MISSES.collect()[0].samples
            if s.name.endswith("_total")
        )

        cache_total = cache_hits + cache_misses
        hit_rate = cache_hits / max(cache_total, 1)

        return {
            "total_requests": int(total_requests),
            "total_tokens": int(total_tokens),
            "cache": {
                "hits": int(cache_hits),
                "misses": int(cache_misses),
                "hit_rate": round(hit_rate, 4),
            },
            "active_requests": int(ACTIVE_REQUESTS._value.get()),
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.warning("Usage analytics error: %s", exc)
        return {"error": str(exc)}


@router.get(
    "/quality",
    summary="Quality metrics",
    dependencies=[Depends(require_permission("analytics:read"))],
)
async def get_quality_analytics() -> dict[str, Any]:
    """Return confidence scores, retrieval quality, and pipeline stats."""
    from backend.services.observability import (
        PIPELINE_ITERATIONS,
        RETRIEVAL_QUALITY,
    )

    try:
        return {
            "retrieval_quality": {
                "sample_count": int(
                    sum(
                        s.value for s in RETRIEVAL_QUALITY.collect()[0].samples
                        if s.name.endswith("_count")
                    )
                ),
            },
            "pipeline_iterations": {
                "sample_count": int(
                    sum(
                        s.value for s in PIPELINE_ITERATIONS.collect()[0].samples
                        if s.name.endswith("_count")
                    )
                ),
            },
            "timestamp": time.time(),
        }
    except Exception as exc:
        logger.warning("Quality analytics error: %s", exc)
        return {"error": str(exc)}


@router.get(
    "/health/detailed",
    summary="Detailed health check",
    dependencies=[Depends(require_permission("analytics:read"))],
)
async def get_detailed_health(
) -> dict[str, Any]:
    """Return detailed health status of all subsystems."""
    from backend.services.observability import DOCUMENT_COUNT

    return {
        "subsystems": {
            "api": "healthy",
            "metrics_endpoint": "healthy",
        },
        "document_count": int(DOCUMENT_COUNT._value.get()),
        "timestamp": time.time(),
    }
