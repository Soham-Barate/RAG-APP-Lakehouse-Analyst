"""
Redis-backed semantic cache.

Stores query embeddings and their responses, enabling fast retrieval
of answers for semantically similar queries (cosine sim > threshold).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic caching layer for the RAG pipeline.

    Flow:
    1. Embed the incoming query
    2. Compare against cached query embeddings (cosine similarity)
    3. If similarity > threshold → return cached response (cache hit)
    4. If miss → pipeline runs, result is cached with TTL

    Uses Redis sorted sets + hashes for efficient storage and lookup.
    """

    def __init__(
        self,
        redis_client: Any,
        embedding_service: Any,
        similarity_threshold: float = 0.95,
        ttl_seconds: int = 3600,
        max_cache_entries: int = 10000,
    ) -> None:
        self.redis = redis_client
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_entries = max_cache_entries

        self._cache_key_embeddings = "rag:cache:embeddings"
        self._cache_key_responses = "rag:cache:responses"
        self._cache_key_index = "rag:cache:index"

    async def get(self, query: str) -> dict[str, Any] | None:
        """
        Look up a semantically similar cached response.

        Returns the cached response dict if a match is found
        above the similarity threshold, otherwise None.
        """
        if not self.redis:
            return None

        try:
            # Embed the query
            query_embedding = await self.embedding_service.embed_query(query)

            # Get all cached embeddings
            cached_entries = await self.redis.hgetall(self._cache_key_embeddings)

            if not cached_entries:
                return None

            best_key: str | None = None
            best_score: float = 0.0

            query_arr = np.array(query_embedding, dtype=np.float32)

            for cache_key, emb_bytes in cached_entries.items():
                cached_emb = np.frombuffer(
                    emb_bytes if isinstance(emb_bytes, bytes) else emb_bytes.encode(),
                    dtype=np.float32,
                )

                # Cosine similarity
                sim = float(np.dot(query_arr, cached_emb) / (
                    np.linalg.norm(query_arr) * np.linalg.norm(cached_emb) + 1e-8
                ))

                if sim > best_score:
                    best_score = sim
                    if isinstance(cache_key, bytes):
                        best_key = cache_key.decode()
                    else:
                        best_key = cache_key

            if best_key and best_score >= self.similarity_threshold:
                # Cache hit — retrieve the response
                response_bytes = await self.redis.hget(
                    self._cache_key_responses, best_key
                )
                if response_bytes:
                    if isinstance(response_bytes, bytes):
                        response_bytes = response_bytes.decode()
                    logger.info(
                        "Cache hit: similarity=%.4f, key=%s",
                        best_score, best_key[:20]
                    )
                    return json.loads(response_bytes)

            return None

        except Exception as exc:
            logger.warning("Semantic cache lookup failed: %s", exc)
            return None

    async def set(self, query: str, response: dict[str, Any]) -> None:
        """
        Cache a query-response pair.

        Stores the query embedding and the serialized response with TTL.
        """
        if not self.redis:
            return

        try:
            query_embedding = await self.embedding_service.embed_query(query)

            # Generate a cache key from the query hash
            import hashlib
            cache_key = hashlib.sha256(query.encode()).hexdigest()[:16]

            # Store embedding as bytes
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            await self.redis.hset(self._cache_key_embeddings, cache_key, emb_bytes)

            # Store response as JSON
            response_json = json.dumps(response, default=str)
            await self.redis.hset(self._cache_key_responses, cache_key, response_json)

            # Set TTL on the hash keys
            await self.redis.expire(self._cache_key_embeddings, self.ttl_seconds)
            await self.redis.expire(self._cache_key_responses, self.ttl_seconds)

            # Enforce max cache size (evict oldest if needed)
            cache_size = await self.redis.hlen(self._cache_key_embeddings)
            if cache_size > self.max_cache_entries:
                await self._evict_oldest()

            logger.debug("Cached response for key: %s", cache_key)

        except Exception as exc:
            logger.warning("Semantic cache set failed: %s", exc)

    async def invalidate(self) -> None:
        """Clear the entire cache (e.g., after data ingestion)."""
        if not self.redis:
            return

        try:
            await self.redis.delete(
                self._cache_key_embeddings,
                self._cache_key_responses,
                self._cache_key_index,
            )
            logger.info("Semantic cache invalidated")
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)

    async def get_stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        if not self.redis:
            return {"status": "disabled"}

        try:
            size = await self.redis.hlen(self._cache_key_embeddings)
            ttl = await self.redis.ttl(self._cache_key_embeddings)
            return {
                "status": "active",
                "entries": size,
                "ttl_remaining": ttl,
                "max_entries": self.max_cache_entries,
                "similarity_threshold": self.similarity_threshold,
            }
        except Exception:
            return {"status": "error"}

    async def _evict_oldest(self) -> None:
        """Evict entries to maintain max cache size."""
        try:
            # Simple strategy: remove random entries to get below limit
            entries = await self.redis.hkeys(self._cache_key_embeddings)
            if len(entries) > self.max_cache_entries:
                to_remove = entries[: len(entries) - self.max_cache_entries]
                for key in to_remove:
                    await self.redis.hdel(self._cache_key_embeddings, key)
                    await self.redis.hdel(self._cache_key_responses, key)
        except Exception as exc:
            logger.warning("Cache eviction failed: %s", exc)
