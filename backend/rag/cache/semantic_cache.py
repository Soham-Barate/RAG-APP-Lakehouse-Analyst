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
        max_entries_per_partition: int = 500,
    ) -> None:
        self.redis = redis_client
        self.embedding_service = embedding_service
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_cache_entries = max_cache_entries
        self.max_entries_per_partition = max_entries_per_partition

        self._cache_key_embeddings = "rag:cache:embeddings"
        self._cache_key_responses = "rag:cache:responses"
        self._cache_key_timestamps = "rag:cache:timestamps"
        self._cache_key_index = "rag:cache:index"

    @staticmethod
    def _normalize_filter_key(filters: dict[str, Any] | None) -> str:
        """Produce a deterministic, compact filter fingerprint to prevent cross-filter cache leakage."""
        if not filters:
            return "global"
        try:
            clean = {k: v for k, v in sorted(filters.items()) if v is not None}
            if not clean:
                return "global"
            import hashlib
            serialized = json.dumps(clean, sort_keys=True)
            return hashlib.sha256(serialized.encode()).hexdigest()[:8]
        except Exception:
            return "global"

    async def get(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Look up a semantically similar cached response within the same filter scope.

        Uses batch matrix cosine similarity for fast lookup performance.
        """
        if not self.redis:
            return None

        try:
            # First, clean expired entries
            await self._evict_expired()

            filter_scope = self._normalize_filter_key(filters)
            prefix = f"{filter_scope}:"

            # Get all cached embeddings in one call
            cached_entries = await self.redis.hgetall(self._cache_key_embeddings)
            if not cached_entries:
                return None

            # Filter entries matching this filter scope
            keys: list[str] = []
            emb_list: list[np.ndarray] = []

            for raw_key, emb_bytes in cached_entries.items():
                key_str = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
                if not key_str.startswith(prefix):
                    continue

                raw = emb_bytes if isinstance(emb_bytes, bytes) else emb_bytes.encode()
                emb_list.append(np.frombuffer(raw, dtype=np.float32).copy())
                keys.append(key_str)

            if not emb_list:
                return None

            # Embed query
            query_embedding = await self.embedding_service.embed_query(query)

            # Vectorised cosine similarity: matrix multiply
            query_arr = np.array(query_embedding, dtype=np.float32)
            matrix = np.stack(emb_list)  # (N, D)

            # Normalise
            query_norm = query_arr / (np.linalg.norm(query_arr) + 1e-8)
            row_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
            matrix_norm = matrix / row_norms

            similarities = matrix_norm @ query_norm  # (N,)

            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])

            if best_score >= self.similarity_threshold:
                best_key = keys[best_idx]
                response_bytes = await self.redis.hget(
                    self._cache_key_responses, best_key
                )
                if response_bytes:
                    if isinstance(response_bytes, bytes):
                        response_bytes = response_bytes.decode()

                    # Update timestamp on hit for LRU retention
                    try:
                        await self.redis.zadd(
                            self._cache_key_timestamps, {best_key: time.time()}
                        )
                    except Exception:
                        pass

                    logger.info(
                        "Cache hit: similarity=%.4f, scope=%s, key=%s",
                        best_score, filter_scope, best_key[:20],
                    )
                    return json.loads(response_bytes)

            return None

        except Exception as exc:
            logger.warning("Semantic cache lookup failed: %s", exc)
            return None

    async def set(
        self,
        query: str,
        response: dict[str, Any],
        filters: dict[str, Any] | None = None,
    ) -> None:
        """
        Cache a query-response pair scoped to the active filters.

        Stores the query embedding, serialized response, and eviction timestamp.
        """
        if not self.redis:
            return

        try:
            import hashlib

            filter_scope = self._normalize_filter_key(filters)
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            cache_key = f"{filter_scope}:{query_hash}"

            query_embedding = await self.embedding_service.embed_query(query)

            # Store embedding as bytes
            emb_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
            await self.redis.hset(self._cache_key_embeddings, cache_key, emb_bytes)

            # Store response as JSON
            response_json = json.dumps(response, default=str)
            await self.redis.hset(self._cache_key_responses, cache_key, response_json)

            # Track entry creation timestamp for per-item TTL / LRU eviction
            now = time.time()
            await self.redis.zadd(self._cache_key_timestamps, {cache_key: now})

            # Enforce max cache size and remove expired items
            await self._evict_expired()
            await self._evict_partition_excess(filter_scope)
            await self._evict_oldest()

            logger.debug("Cached response for scope %s, key: %s", filter_scope, cache_key)

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
                self._cache_key_timestamps,
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
            return {
                "status": "active",
                "entries": size,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_cache_entries,
                "max_entries_per_partition": self.max_entries_per_partition,
                "similarity_threshold": self.similarity_threshold,
            }
        except Exception:
            return {"status": "error"}

    async def _evict_expired(self) -> None:
        """Evict entries whose age exceeds ttl_seconds."""
        try:
            cutoff = time.time() - self.ttl_seconds
            expired_keys = await self.redis.zrangebyscore(
                self._cache_key_timestamps, 0, cutoff
            )
            if expired_keys:
                for k in expired_keys:
                    key_str = k.decode() if isinstance(k, bytes) else k
                    await self.redis.hdel(self._cache_key_embeddings, key_str)
                    await self.redis.hdel(self._cache_key_responses, key_str)
                    await self.redis.zrem(self._cache_key_timestamps, key_str)
                logger.debug("Evicted %d expired cache entries", len(expired_keys))
        except Exception as exc:
            logger.warning("Cache TTL eviction failed: %s", exc)

    async def _evict_partition_excess(self, filter_scope: str) -> None:
        """Evict oldest entries in a specific partition if it exceeds max_entries_per_partition."""
        if not self.max_entries_per_partition or self.max_entries_per_partition <= 0:
            return
        try:
            prefix = f"{filter_scope}:"
            all_keys = await self.redis.zrange(self._cache_key_timestamps, 0, -1)
            partition_keys = []
            for k in all_keys:
                key_str = k.decode() if isinstance(k, bytes) else k
                if key_str.startswith(prefix):
                    partition_keys.append(key_str)

            if len(partition_keys) > self.max_entries_per_partition:
                excess_count = len(partition_keys) - self.max_entries_per_partition
                to_evict = partition_keys[:excess_count]
                for key_str in to_evict:
                    await self.redis.hdel(self._cache_key_embeddings, key_str)
                    await self.redis.hdel(self._cache_key_responses, key_str)
                    await self.redis.zrem(self._cache_key_timestamps, key_str)
                logger.debug(
                    "Evicted %d entries exceeding partition limit (%d) for scope %s",
                    len(to_evict),
                    self.max_entries_per_partition,
                    filter_scope,
                )
        except Exception as exc:
            logger.warning("Cache partition eviction failed: %s", exc)

    async def _evict_oldest(self) -> None:
        """Evict oldest entries to maintain max cache size."""
        try:
            total = await self.redis.zcard(self._cache_key_timestamps)
            if total > self.max_cache_entries:
                excess = total - self.max_cache_entries
                oldest = await self.redis.zrange(self._cache_key_timestamps, 0, excess - 1)
                for k in oldest:
                    key_str = k.decode() if isinstance(k, bytes) else k
                    await self.redis.hdel(self._cache_key_embeddings, key_str)
                    await self.redis.hdel(self._cache_key_responses, key_str)
                    await self.redis.zrem(self._cache_key_timestamps, key_str)
                logger.debug("Evicted %d oldest cache entries", len(oldest))
        except Exception as exc:
            logger.warning("Cache LRU eviction failed: %s", exc)
