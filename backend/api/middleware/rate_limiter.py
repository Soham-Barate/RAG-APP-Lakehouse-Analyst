"""
Token-bucket rate limiter backed by Redis using an atomic Lua script.

Limits are applied per API key. Under high concurrency, token evaluation
and deduction execute atomically in Redis to prevent race conditions.
Configurable fallback behavior supports development vs fail-closed production modes.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# Atomic Token Bucket Lua Script
RATE_LIMIT_LUA_SCRIPT = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

if not tokens or not last_refill then
    tokens = max_tokens
    last_refill = now
else
    local elapsed = math.max(0, now - last_refill)
    tokens = math.min(max_tokens, tokens + (elapsed * refill_rate))
    last_refill = now
end

if tokens < requested then
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
    redis.call('EXPIRE', key, ttl)
    return {0, math.floor(tokens)}
else
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tostring(tokens), 'last_refill', tostring(last_refill))
    redis.call('EXPIRE', key, ttl)
    return {1, math.floor(tokens)}
end
"""


class TokenBucketRateLimiter:
    """
    Redis-backed token-bucket rate limiter with atomic Lua script execution.
    """

    def __init__(
        self,
        redis_client: Any,
        max_tokens: int = 60,
        refill_seconds: int = 60,
        prefix: str = "rag:ratelimit:",
        fail_closed: bool = False,
    ) -> None:
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_seconds = refill_seconds
        self.prefix = prefix
        self.fail_closed = fail_closed
        self._last_remaining: int = max_tokens

    async def check(self, key: str) -> bool:
        """Return True if the request is allowed, False if rate-limited."""
        if self.redis is None:
            if self.fail_closed:
                logger.error("Rate limiter: Redis unavailable in fail-closed mode; rejecting request.")
                return False
            return True  # graceful degradation for dev

        bucket_key = f"{self.prefix}{key}"
        now = time.time()
        refill_rate = self.max_tokens / max(self.refill_seconds, 1)
        ttl = self.refill_seconds * 2

        try:
            # Execute atomic Lua script
            result = await self.redis.eval(
                RATE_LIMIT_LUA_SCRIPT,
                1,
                bucket_key,
                str(self.max_tokens),
                str(refill_rate),
                "1",
                str(now),
                str(ttl),
            )
            allowed, remaining = result[0], result[1]
            self._last_remaining = int(remaining)
            return bool(allowed == 1)

        except Exception as exc:
            if self.fail_closed:
                logger.error("Rate limiter Redis failure in fail-closed mode: %s", exc)
                return False
            logger.warning("Rate limiter Redis error — allowing request (fail-open mode): %s", exc)
            return True

    async def get_remaining(self, key: str) -> int:
        """Return the approximate number of remaining tokens."""
        if hasattr(self, "_last_remaining"):
            return self._last_remaining
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
    """FastAPI dependency that enforces rate limits."""
    redis_client = getattr(request.app.state, "redis", None)
    api_key = request.headers.get(settings.api_key_header, "anonymous")

    fail_closed_flag = getattr(settings.rate_limit, "fail_closed", False)
    fail_closed = bool(fail_closed_flag is True)

    limiter = TokenBucketRateLimiter(
        redis_client=redis_client,
        max_tokens=settings.rate_limit.requests_per_minute,
        refill_seconds=60,
        fail_closed=fail_closed,
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
