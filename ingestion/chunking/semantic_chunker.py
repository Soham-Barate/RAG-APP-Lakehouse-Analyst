"""
Semantic chunking strategy.

Groups semantically related sentences together using embedding-based
similarity.  Unlike fixed-size chunking that blindly splits every N
tokens, semantic chunking ensures that topically coherent paragraphs
stay intact, producing higher-quality retrieval chunks.

Fallback: if sentence tokenization or embedding is unavailable, this
delegates to the standard ParentChildChunker.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class SemanticChunker:
    """
    Embedding-similarity-based document chunker.

    Algorithm:
    1. Split document into sentences.
    2. Embed each sentence (or use cached embeddings).
    3. Compute cosine similarity between adjacent sentences.
    4. Split wherever similarity drops below a dynamic threshold.
    5. Merge resulting groups into chunks of appropriate size.

    This produces chunks where each chunk is a semantically coherent
    block of text — ideal for retrieval.
    """

    def __init__(
        self,
        embedding_fn: Any = None,
        max_chunk_tokens: int = 512,
        min_chunk_tokens: int = 64,
        similarity_threshold: float | None = None,
        percentile_cutoff: int = 25,
    ) -> None:
        """
        Args:
            embedding_fn: Callable that takes a list of strings and returns
                          a list of embedding vectors.  If None, falls back
                          to character-based splitting.
            max_chunk_tokens: Maximum tokens per chunk (approximate).
            min_chunk_tokens: Minimum tokens per chunk (merge if smaller).
            similarity_threshold: Fixed similarity threshold for splitting.
                                  If None, uses dynamic threshold based on
                                  percentile_cutoff of all adjacent similarities.
            percentile_cutoff: When using dynamic threshold, split at
                               boundaries below this percentile of
                               similarity scores.
        """
        self.embedding_fn = embedding_fn
        self.max_chunk_tokens = max_chunk_tokens
        self.min_chunk_tokens = min_chunk_tokens
        self.similarity_threshold = similarity_threshold
        self.percentile_cutoff = percentile_cutoff

    async def chunk_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Chunk a document using semantic similarity between sentences.

        Returns a list of chunk dicts with id, content, and metadata.
        """
        if not content or not content.strip():
            return []

        metadata = metadata or {}
        sentences = self._split_sentences(content)

        if len(sentences) <= 1:
            chunk_id = self._generate_id(content)
            return [{
                "id": chunk_id,
                "content": content.strip(),
                "metadata": {
                    **metadata,
                    "chunk_type": "semantic",
                    "chunk_index": 0,
                    "total_chunks": 1,
                },
                "is_parent": False,
                "parent_id": chunk_id,
            }]

        # Try embedding-based semantic splitting
        if self.embedding_fn:
            try:
                groups = await self._semantic_split(sentences)
            except Exception as exc:
                logger.warning(
                    "Semantic splitting failed, falling back: %s", exc
                )
                groups = self._fallback_split(sentences)
        else:
            groups = self._fallback_split(sentences)

        # Enforce size limits
        final_groups = self._enforce_size_limits(groups)

        # Build chunk dicts
        chunks: list[dict[str, Any]] = []
        parent_id = self._generate_id(content[:200])

        for idx, group in enumerate(final_groups):
            chunk_text = " ".join(group).strip()
            if not chunk_text:
                continue

            chunk_id = self._generate_id(f"{parent_id}:{idx}:{chunk_text[:50]}")
            chunks.append({
                "id": chunk_id,
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "semantic",
                    "chunk_index": idx,
                    "total_chunks": len(final_groups),
                    "parent_id": parent_id,
                    "sentence_count": len(group),
                },
                "is_parent": False,
                "parent_id": parent_id,
            })

        logger.debug(
            "Semantic chunking: %d sentences → %d chunks",
            len(sentences),
            len(chunks),
        )

        return chunks

    async def _semantic_split(
        self, sentences: list[str]
    ) -> list[list[str]]:
        """Split sentences into groups based on embedding similarity."""
        # Embed all sentences
        embeddings = await self.embedding_fn(sentences)
        emb_array = np.array(embeddings)

        # Compute cosine similarity between adjacent sentences
        similarities: list[float] = []
        for i in range(len(emb_array) - 1):
            a = emb_array[i]
            b = emb_array[i + 1]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 0 and norm_b > 0:
                sim = float(np.dot(a, b) / (norm_a * norm_b))
            else:
                sim = 0.0
            similarities.append(sim)

        # Determine split threshold
        if self.similarity_threshold is not None:
            threshold = self.similarity_threshold
        else:
            threshold = float(np.percentile(similarities, self.percentile_cutoff))

        # Find split points (where similarity drops below threshold)
        split_points: list[int] = []
        for i, sim in enumerate(similarities):
            if sim < threshold:
                split_points.append(i + 1)

        # Build groups
        groups: list[list[str]] = []
        start = 0
        for split in split_points:
            groups.append(sentences[start:split])
            start = split
        groups.append(sentences[start:])

        return groups

    def _fallback_split(self, sentences: list[str]) -> list[list[str]]:
        """Fallback: group sentences by approximate token count."""
        groups: list[list[str]] = []
        current_group: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = len(sent) // 4  # Rough estimate
            if current_tokens + sent_tokens > self.max_chunk_tokens and current_group:
                groups.append(current_group)
                current_group = []
                current_tokens = 0
            current_group.append(sent)
            current_tokens += sent_tokens

        if current_group:
            groups.append(current_group)

        return groups

    def _enforce_size_limits(
        self, groups: list[list[str]]
    ) -> list[list[str]]:
        """Merge groups that are too small, split groups that are too large."""
        merged: list[list[str]] = []

        for group in groups:
            text = " ".join(group)
            token_est = len(text) // 4

            if token_est > self.max_chunk_tokens * 1.5:
                # Split large group in half
                mid = len(group) // 2
                merged.append(group[:mid])
                merged.append(group[mid:])
            elif merged and token_est < self.min_chunk_tokens:
                # Merge with previous group
                merged[-1].extend(group)
            else:
                merged.append(group)

        return merged

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        # Split on sentence-ending punctuation followed by whitespace
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if s.strip()]

    @staticmethod
    def _generate_id(content: str) -> str:
        """Generate a deterministic chunk ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
