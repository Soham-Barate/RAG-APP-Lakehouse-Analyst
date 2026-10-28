"""
Unit tests for the Redis-backed SemanticCache.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest

from backend.rag.cache.semantic_cache import SemanticCache


class FakeAsyncRedis:
    """In-memory async Redis simulator for testing SemanticCache."""

    def __init__(self):
        self.hashes: dict[str, dict[str, Any]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def hgetall(self, name: str) -> dict[str, Any]:
        return dict(self.hashes.get(name, {}))

    async def hget(self, name: str, key: str) -> Any | None:
        return self.hashes.get(name, {}).get(key)

    async def hset(self, name: str, key: str, value: Any) -> int:
        if name not in self.hashes:
            self.hashes[name] = {}
        self.hashes[name][key] = value
        return 1

    async def hdel(self, name: str, *keys: str) -> int:
        removed = 0
        if name in self.hashes:
            for k in keys:
                if k in self.hashes[name]:
                    del self.hashes[name][k]
                    removed += 1
        return removed

    async def hlen(self, name: str) -> int:
        return len(self.hashes.get(name, {}))

    async def zadd(self, name: str, mapping: dict[str, float]) -> int:
        if name not in self.zsets:
            self.zsets[name] = {}
        self.zsets[name].update(mapping)
        return len(mapping)

    async def zrangebyscore(
        self, name: str, min_score: float, max_score: float
    ) -> list[str]:
        zset = self.zsets.get(name, {})
        return [k for k, score in zset.items() if min_score <= score <= max_score]

    async def zrange(self, name: str, start: int, end: int) -> list[str]:
        zset = self.zsets.get(name, {})
        sorted_keys = sorted(zset.keys(), key=lambda k: zset[k])
        if end == -1 or end >= len(sorted_keys):
            return sorted_keys[start:]
        return sorted_keys[start : end + 1]

    async def zcard(self, name: str) -> int:
        return len(self.zsets.get(name, {}))

    async def zrem(self, name: str, *keys: str) -> int:
        removed = 0
        if name in self.zsets:
            for k in keys:
                if k in self.zsets[name]:
                    del self.zsets[name][k]
                    removed += 1
        return removed

    async def delete(self, *names: str) -> int:
        count = 0
        for n in names:
            if n in self.hashes:
                del self.hashes[n]
                count += 1
            if n in self.zsets:
                del self.zsets[n]
                count += 1
        return count


class FakeEmbeddingService:
    """Deterministic embedding service mapping queries to fixed unit vectors."""

    async def embed_query(self, text: str) -> list[float]:
        # Simple hash-based deterministic 4-D embedding
        if "laptop" in text.lower():
            vec = [1.0, 0.0, 0.0, 0.0]
        elif "notebook" in text.lower():
            # Very close to laptop
            vec = [0.98, 0.1, 0.0, 0.0]
        elif "shoes" in text.lower():
            vec = [0.0, 1.0, 0.0, 0.0]
        elif "watch" in text.lower():
            vec = [0.0, 0.0, 1.0, 0.0]
        elif "camera" in text.lower():
            vec = [0.0, 0.0, 0.0, 1.0]
        else:
            vec = [0.5, 0.5, 0.5, 0.5]
        norm = np.linalg.norm(vec)
        return (np.array(vec) / norm).tolist()


@pytest.fixture
def fake_redis():
    return FakeAsyncRedis()


@pytest.fixture
def fake_emb():
    return FakeEmbeddingService()


@pytest.fixture
def cache(fake_redis, fake_emb):
    return SemanticCache(
        redis_client=fake_redis,
        embedding_service=fake_emb,
        similarity_threshold=0.90,
        ttl_seconds=3600,
        max_cache_entries=100,
    )


class TestSemanticCache:
    """Test suite for SemanticCache features and correctness."""

    @pytest.mark.asyncio
    async def test_set_and_exact_get(self, cache):
        query = "best laptop for programming"
        payload = {"answer": "The MacBook Pro 16 is great.", "confidence": 0.95}

        await cache.set(query, payload)
        result = await cache.get(query)

        assert result is not None
        assert result["answer"] == payload["answer"]

    @pytest.mark.asyncio
    async def test_semantic_similarity_hit(self, cache):
        await cache.set("best laptop computer", {"answer": "Dell XPS 15"})

        # "notebook computer" is mapped to vector very close to "laptop"
        result = await cache.get("best notebook computer")
        assert result is not None
        assert result["answer"] == "Dell XPS 15"

    @pytest.mark.asyncio
    async def test_semantic_dissimilarity_miss(self, cache):
        await cache.set("best laptop computer", {"answer": "Dell XPS 15"})

        # "shoes" is orthogonal to laptop
        result = await cache.get("running shoes for marathon")
        assert result is None

    @pytest.mark.asyncio
    async def test_filter_isolation_prevents_leakage(self, cache):
        query = "best products"
        filter_electronics = {"category": "Electronics"}
        filter_apparel = {"category": "Apparel"}

        await cache.set(query, {"answer": "Laptops & Phones"}, filters=filter_electronics)

        # Lookup with same query but DIFFERENT filter must MISS
        res_apparel = await cache.get(query, filters=filter_apparel)
        assert res_apparel is None

        # Lookup with same query and NO filter must MISS
        res_none = await cache.get(query, filters=None)
        assert res_none is None

        # Lookup with matching filter must HIT
        res_match = await cache.get(query, filters=filter_electronics)
        assert res_match is not None
        assert res_match["answer"] == "Laptops & Phones"

    @pytest.mark.asyncio
    async def test_item_ttl_expiration(self, fake_redis, fake_emb):
        # Cache with 1-second TTL
        short_cache = SemanticCache(
            redis_client=fake_redis,
            embedding_service=fake_emb,
            similarity_threshold=0.90,
            ttl_seconds=1,
        )

        await short_cache.set("laptop sale", {"answer": "10% off"})
        assert (await short_cache.get("laptop sale")) is not None

        # Manually fast-forward entry timestamp in sorted set to simulate expiration
        timestamps = fake_redis.zsets[short_cache._cache_key_timestamps]
        for k in timestamps:
            timestamps[k] -= 10.0  # 10 seconds ago

        # Eviction occurs on get
        expired_result = await short_cache.get("laptop sale")
        assert expired_result is None

    @pytest.mark.asyncio
    async def test_cache_invalidation(self, cache):
        await cache.set("laptop deals", {"answer": "Special deals"})
        assert (await cache.get("laptop deals")) is not None

        await cache.invalidate()
        assert (await cache.get("laptop deals")) is None

    @pytest.mark.asyncio
    async def test_partition_capacity_limit(self, fake_redis, fake_emb):
        """Verify partition limit evicts oldest within the same filter partition."""
        partition_cache = SemanticCache(
            redis_client=fake_redis,
            embedding_service=fake_emb,
            similarity_threshold=0.90,
            max_cache_entries=100,
            max_entries_per_partition=2,
        )

        filter_a = {"category": "Electronics"}
        filter_b = {"category": "Apparel"}

        # Insert 2 items in partition A
        await partition_cache.set("laptop deals", {"answer": "A1"}, filters=filter_a)
        await partition_cache.set("watch deals", {"answer": "A2"}, filters=filter_a)

        # Insert 1 item in partition B
        await partition_cache.set("shoes deals", {"answer": "B1"}, filters=filter_b)

        # Now insert 3rd item in partition A -> should evict the oldest in partition A ("laptop deals")
        await partition_cache.set("camera deals", {"answer": "A3"}, filters=filter_a)

        # Partition A should now only have watch deals and camera deals
        res_a1 = await partition_cache.get("laptop deals", filters=filter_a)
        res_a2 = await partition_cache.get("watch deals", filters=filter_a)
        res_a3 = await partition_cache.get("camera deals", filters=filter_a)

        assert res_a1 is None, "Oldest query 1 should have been evicted from partition A"
        assert res_a2 is not None
        assert res_a2["answer"] == "A2"
        assert res_a3 is not None
        assert res_a3["answer"] == "A3"

        # Partition B item should remain intact
        res_b1 = await partition_cache.get("shoes deals", filters=filter_b)
        assert res_b1 is not None
        assert res_b1["answer"] == "B1"

    @pytest.mark.asyncio
    async def test_global_capacity_eviction(self, fake_redis, fake_emb):
        """Verify global capacity limit evicts oldest items across partitions."""
        small_cache = SemanticCache(
            redis_client=fake_redis,
            embedding_service=fake_emb,
            similarity_threshold=0.90,
            max_cache_entries=2,
            max_entries_per_partition=10,
        )

        await small_cache.set("q1", {"answer": "1"})
        await small_cache.set("q2", {"answer": "2"})
        await small_cache.set("q3", {"answer": "3"})

        stats = await small_cache.get_stats()
        assert stats["entries"] <= 2
        assert stats["max_entries_per_partition"] == 10

    @pytest.mark.asyncio
    async def test_graceful_redis_failure(self, fake_emb):
        """SemanticCache should gracefully handle Redis exceptions without raising."""
        broken_redis = AsyncMock()
        broken_redis.hgetall.side_effect = ConnectionError("Redis connection refused")
        broken_redis.hset.side_effect = ConnectionError("Redis connection refused")

        fail_cache = SemanticCache(
            redis_client=broken_redis,
            embedding_service=fake_emb,
        )

        # Should not raise exception
        get_res = await fail_cache.get("any query")
        assert get_res is None

        # Set should not raise exception
        await fail_cache.set("any query", {"answer": "test"})

