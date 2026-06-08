"""
Token-bucket rate limiter backed by Redis.

Limits are applied per API key.  When Redis is unavailable the limiter
degrades gracefully (allows all traffic) rather than blocking.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """
    Redis-backed token-bucket rate limiter.

    Each API key gets ``max_tokens`` tokens that refill at a constant rate.
    Each request consumes one token.
    """

    def __init__(
        self,
        redis_client: Any,
        max_tokens: int = 60,
        refill_seconds: int = 60,
        prefix: str = "rag:ratelimit:",
    ) -> None:
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self.prefix = prefix

    async def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if self.redis is None:
            return True  # graceful degradation

        bucket_key = f"{self.prefix}{key}"
        now = time.time()

        try:
            pipe = self.redis.pipeline(transaction=True)
            pipe.hgetall(bucket_key)
            results = await pipe.execute()
            bucket = results[0]

            if not bucket:
                # First request — initialize bucket
                pipe2 = self.redis.pipeline(transaction=True)
                pipe2.hset(
                    bucket_key,
                    mapping={
                        "tokens": str(self.max_tokens - 1),
                        "last_refill": str(now),
                    },
                )
                pipe2.expire(bucket_key, self.refill_seconds * 2)
                await pipe2.execute()
                return True

            tokens = float(bucket.get(b"tokens", bucket.get("tokens", self.max_tokens)))
            last_refill = float(
                bucket.get(b"last_refill", bucket.get("last_refill", now))
            )

            # Refill tokens based on elapsed time
            elapsed = now - last_refill
            refill_rate = self.max_tokens / self.refill_seconds
            tokens = min(self.max_tokens, tokens + elapsed * refill_rate)

            if tokens < 1:
                return False

            # Consume one token
            pipe3 = self.redis.pipeline(transaction=True)
            pipe3.hset(
                bucket_key,
                mapping={"tokens": str(tokens - 1), "last_refill": str(now)},
            )
            pipe3.expire(bucket_key, self.refill_seconds * 2)
            await pipe3.execute()
            return True

        except Exception:
            logger.warning("Rate limiter Redis error — allowing request", exc_info=True)
            return True

    async def get_remaining(self, key: str) -> int:
        """Return the approximate number of remaining tokens."""
        if self.redis is None:
            return self.max_tokens
        try:
            bucket_key = f"{self.prefix}{key}"
            tokens = await self.redis.hget(bucket_key, "tokens")
            return int(float(tokens)) if tokens else self.max_tokens
        except Exception:
            return self.max_tokens


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------
async def check_rate_limit(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency that enforces rate limits.

    Extracts the API key from the request state (set by auth middleware)
    and checks the token bucket.
    """
    redis_client = getattr(request.app.state, "redis", None)
    api_key = request.headers.get(settings.api_key_header, "anonymous")

    limiter = TokenBucketRateLimiter(
        redis_client=redis_client,
        max_tokens=settings.rate_limit.requests_per_minute,
        refill_seconds=60,
    )

    allowed = await limiter.check(api_key)
    if not allowed:
        remaining = await limiter.get_remaining(api_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": "Too many requests. Please retry later.",
                "remaining_tokens": remaining,
            },
            headers={"Retry-After": "60"},
        )
