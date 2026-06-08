"""
Structured request/response logging middleware.

Assigns a unique correlation ID to every request and logs method, path,
status, and latency in structured JSON format.
"""

from __future__ import annotations

import time
import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that:
    1. Assigns a unique ``X-Request-ID`` to every request
    2. Logs request start (method, path, client IP)
    3. Logs request completion (status code, latency)
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Bind request_id to structlog context for the duration of this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        # Store on request state for downstream access
        request.state.request_id = request_id

        start = time.perf_counter()

        await logger.ainfo(
            "request_started",
            method=request.method,
            path=str(request.url.path),
            query=str(request.url.query) if request.url.query else None,
            client_ip=request.client.host if request.client else None,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            await logger.aerror(
                "request_failed",
                method=request.method,
                path=str(request.url.path),
                latency_ms=round(latency_ms, 2),
                error=str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000

        await logger.ainfo(
            "request_completed",
            method=request.method,
            path=str(request.url.path),
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        # Attach request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response
