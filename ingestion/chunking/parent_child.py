"""
Parent-child document chunking strategy.

Creates large parent chunks (2000 tokens) for full-context generation
and small child chunks (256 tokens) for precise retrieval. Maintains
parent-child relationships via IDs.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

import tiktoken

logger = logging.getLogger(__name__)


class ParentChildChunker:
    """
    Two-tier chunking strategy for optimal RAG performance.

    - Parent chunks (large): Used for generation context (full meaning)
    - Child chunks (small): Used for retrieval (precise matching)
    - Each child links to its parent via parent_id
    - Overlapping windows between child chunks prevent context loss

    This approach gives the best of both worlds: precise retrieval
    with small chunks + rich context from parent chunks during generation.
    """

    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 256,
        child_overlap: int = 64,
        encoding_name: str = "cl100k_base",
    ) -> None:
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception:
            self.encoding = None
            logger.warning(
                "tiktoken encoding %s unavailable, using character-based chunking",
                encoding_name,
            )

    def chunk_document(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Chunk a document into parent and child chunks.

        Returns a list of chunk dicts, each containing:
        - id: unique chunk ID
        - content: chunk text
        - metadata: inherited metadata + chunk-specific fields
        - is_parent: True for parent chunks
        - parent_id: ID of the parent chunk (for child chunks)
        """
        if not content or not content.strip():
            return []

        metadata = metadata or {}
        all_chunks: list[dict[str, Any]] = []

        # Step 1: Split into parent chunks
        parent_texts = self._split_into_chunks(
            content, self.parent_chunk_size, overlap=0
        )

        for p_idx, parent_text in enumerate(parent_texts):
            parent_id = self._generate_id(parent_text)

            # Create parent chunk
            parent_chunk = {
                "id": parent_id,
                "content": parent_text,
                "metadata": {
                    **metadata,
                    "chunk_type": "parent",
                    "chunk_index": p_idx,
                    "parent_id": parent_id,
                    "total_parents": len(parent_texts),
                },
                "is_parent": True,
                "parent_id": parent_id,
            }
            all_chunks.append(parent_chunk)

            # Step 2: Split parent into child chunks
            child_texts = self._split_into_chunks(
                parent_text,
                self.child_chunk_size,
                overlap=self.child_overlap,
            )

            for c_idx, child_text in enumerate(child_texts):
                child_id = self._generate_id(f"{parent_id}:{c_idx}:{child_text[:50]}")

                child_chunk = {
                    "id": child_id,
                    "content": child_text,
                    "metadata": {
                        **metadata,
                        "chunk_type": "child",
                        "chunk_index": c_idx,
                        "parent_id": parent_id,
                        "parent_chunk_index": p_idx,
                        "total_children": len(child_texts),
                    },
                    "is_parent": False,
                    "parent_id": parent_id,
                }
                all_chunks.append(child_chunk)

        logger.debug(
            "Chunked document: %d parents, %d total chunks",
            len(parent_texts),
            len(all_chunks),
        )

        return all_chunks

    def _split_into_chunks(
        self,
        text: str,
        chunk_size: int,
        overlap: int = 0,
    ) -> list[str]:
        """Split text into chunks by token count."""
        if self.encoding:
            return self._split_by_tokens(text, chunk_size, overlap)
        return self._split_by_chars(text, chunk_size * 4, overlap * 4)

    def _split_by_tokens(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Token-aware splitting with overlap."""
        tokens = self.encoding.encode(text)

        if len(tokens) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunks.append(chunk_text.strip())

            # Move forward by (chunk_size - overlap)
            step = chunk_size - overlap
            start += step

        return chunks

    def _split_by_chars(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        """Character-based splitting fallback."""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)
                if break_point > start + chunk_size // 2:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    @staticmethod
    def _generate_id(content: str) -> str:
        """Generate a deterministic ID from content hash."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]
