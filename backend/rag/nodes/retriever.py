"""
Hybrid Retriever Node.

Performs vector + BM25 retrieval, fuses results via Reciprocal Rank
Fusion (RRF), applies metadata pre-filtering, and resolves parent
documents for small-chunk hits.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.rag.state import RAGState

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Multi-query hybrid retrieval with parent-document resolution.

    Pipeline:
    1. Embed all query variants (original + rewritten + HyDE doc)
    2. Run vector + BM25 search for each query variant
    3. Apply metadata filters (date, category, department)
    4. Fuse results via Reciprocal Rank Fusion (RRF)
    5. Deduplicate and resolve parent documents
    """

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        top_k: int = 20,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Execute hybrid retrieval and return ranked documents."""
        start = time.perf_counter()

        queries = state.get("rewritten_queries", [state["query"]])
        hyde_doc = state.get("hyde_document", "")
        filters = state.get("metadata_filters", {})

        ranked_lists: list[list[dict[str, Any]]] = []
        raw_count = 0

        # ----- 1. Retrieve for each query variant -----
        for query_text in queries:
            try:
                results = await self.vector_store.hybrid_search(
                    query_text=query_text,
                    filters=filters,
                    top_k=self.top_k,
                )
                if results:
                    ranked_lists.append(results)
                    raw_count += len(results)
            except Exception as exc:
                logger.warning("Retrieval failed for query variant: %s", exc)

        # ----- 2. Retrieve using HyDE document embedding -----
        if hyde_doc:
            try:
                hyde_results = await self.vector_store.hybrid_search(
                    query_text=hyde_doc,
                    filters=filters,
                    top_k=self.top_k // 2,  # Fewer results for HyDE
                )
                if hyde_results:
                    ranked_lists.append(hyde_results)
                    raw_count += len(hyde_results)
            except Exception as exc:
                logger.warning("HyDE retrieval failed: %s", exc)

        # ----- 3. Reciprocal Rank Fusion (RRF) across independent ranked lists -----
        fused = self._reciprocal_rank_fusion(ranked_lists)

        # ----- 4. Deduplicate -----
        seen_ids: set[str] = set()
        unique_results: list[dict[str, Any]] = []
        for doc in fused:
            doc_id = doc.get("id", doc.get("content", "")[:100])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                unique_results.append(doc)

        # Limit to top_k
        unique_results = unique_results[: self.top_k]

        # ----- 5. Resolve parent documents -----
        resolved = await self._resolve_parents(unique_results)

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Retrieval: %d raw -> %d fused -> %d unique (%.1fms)",
            raw_count,
            len(fused),
            len(resolved),
            latency_ms,
        )

        return {
            "retrieved_docs": resolved,
            "retrieval_latency_ms": round(latency_ms, 2),
            "documents_retrieved": len(resolved),
        }

    def _reciprocal_rank_fusion(
        self,
        results: list[list[dict[str, Any]]] | list[dict[str, Any]],
        k: int = 60,
    ) -> list[dict[str, Any]]:
        """
        Reciprocal Rank Fusion — merges multiple independently ranked lists.

        Formula:
            RRF(d) = Σ_{r in R} 1 / (k + rank_r(d))
        where rank_r(d) is the 1-based rank of document d in ranked list r.

        Guarantees:
        - Each independent list contributes at most once per unique document.
        - Stable document IDs are used for deduplication.
        - Document metadata, scores, and contents are safely preserved.
        """
        if not results:
            return []

        doc_scores: dict[str, float] = {}
        doc_map: dict[str, dict[str, Any]] = {}

        # Accept either list of ranked lists or a single flat list (for backward compatibility)
        if isinstance(results[0], dict):
            ranked_lists: list[list[dict[str, Any]]] = [results]  # type: ignore[list-item]
        else:
            ranked_lists = results  # type: ignore[assignment]

        for r_list in ranked_lists:
            if not isinstance(r_list, list):
                continue
            seen_in_current_list: set[str] = set()
            for rank, doc in enumerate(r_list, start=1):
                if not isinstance(doc, dict):
                    continue
                doc_id = str(doc.get("id") or doc.get("content", "")[:100]).strip()
                if not doc_id:
                    continue

                # Ensure document only contributes once per independent list (at its best rank)
                if doc_id in seen_in_current_list:
                    continue
                seen_in_current_list.add(doc_id)

                rrf_score = 1.0 / (k + rank)

                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    doc_map[doc_id] = doc.copy()

        # Sort by RRF score descending, with deterministic tie-breaking by doc_id
        sorted_ids = sorted(
            doc_scores.keys(),
            key=lambda x: (doc_scores[x], x),
            reverse=True,
        )

        fused = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = doc_scores[doc_id]
            fused.append(doc)

        return fused

    async def _resolve_parents(
        self,
        docs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Resolve parent documents for child chunks.

        If a retrieved chunk is a child (small chunk), fetch its parent
        (large chunk) to provide fuller context in generation.
        """
        resolved: list[dict[str, Any]] = []

        for doc in docs:
            parent_id = doc.get("metadata", {}).get("parent_id")
            if parent_id and hasattr(self.vector_store, "get_parent_document"):
                try:
                    parent = await self.vector_store.get_parent_document(parent_id)
                    if parent:
                        # Keep child metadata but use parent content
                        parent_doc = doc.copy()
                        parent_doc["content"] = parent.get("content", doc["content"])
                        parent_doc["metadata"]["resolved_from_child"] = True
                        resolved.append(parent_doc)
                        continue
                except Exception:
                    pass  # Fall through to using the child chunk

            resolved.append(doc)

        return resolved
