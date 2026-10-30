"""
Dataset-scoped text indexer for user-uploaded datasets.
Extracts unstructured text columns and indexes them into pgvector with strict dataset isolation.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from backend.dataset.manager import DatasetManager, get_dataset_manager
from backend.vectorstore.base import VectorStoreBase

logger = logging.getLogger(__name__)


class DatasetTextIndexer:
    """Indexes text content from uploaded datasets into the vector store."""

    def __init__(
        self,
        vector_store: VectorStoreBase,
        embedding_service: Any,
        dataset_manager: DatasetManager | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.dataset_manager = dataset_manager or get_dataset_manager()

    async def index_dataset_text(self, dataset_id: str, max_rows: int = 500) -> int:
        """
        Scan dataset for unstructured text columns, generate embeddings,
        and index them into the vector store with metadata dataset_id.
        Returns the number of documents indexed.
        """
        meta = self.dataset_manager.get_dataset(dataset_id)
        if not meta or meta.is_demo:
            return 0

        profile = self.dataset_manager.get_profiling_report(dataset_id)
        if not profile:
            return 0

        # Identify text columns suitable for RAG
        text_cols = [col.name for col in profile.columns if col.is_text]
        if not text_cols:
            logger.info("Dataset %s has no text columns suitable for vector indexing.", dataset_id)
            return 0

        try:
            df = pd.read_parquet(meta.parquet_path)
        except Exception as e:
            logger.error("Failed reading parquet for dataset %s: %s", dataset_id, e)
            return 0

        documents_to_add: list[dict[str, Any]] = []
        texts_to_embed: list[str] = []

        # Sample rows up to max_rows
        sample_df = df.head(max_rows)

        for idx, row in sample_df.iterrows():
            # Build meaningful text representation from text columns
            text_parts: list[str] = []
            for col in text_cols:
                val = row.get(col)
                if pd.notnull(val) and str(val).strip():
                    text_parts.append(f"{col}: {str(val).strip()}")

            if not text_parts:
                continue

            full_text = "\n".join(text_parts)
            doc_id = f"{dataset_id}_r{idx}"

            row_metadata = {
                "dataset_id": dataset_id,
                "source_file": meta.filename,
                "dataset_name": meta.name,
                "row_index": int(idx),
                "document_type": "user_dataset",
            }

            # Add non-text identifying columns (like id, title, name) to metadata
            for col in df.columns:
                if col not in text_cols and ("id" in col or "name" in col or "title" in col):
                    val = row.get(col)
                    if pd.notnull(val):
                        row_metadata[col] = str(val)[:100]

            documents_to_add.append({
                "id": doc_id,
                "content": full_text,
                "metadata": row_metadata,
                "document_type": "user_dataset",
                "category": meta.name,
            })
            texts_to_embed.append(full_text)

        if not documents_to_add:
            return 0

        logger.info(
            "Embedding and indexing %d text documents for dataset %s (%s)",
            len(documents_to_add),
            dataset_id,
            meta.name,
        )

        try:
            embeddings = await self.embedding_service.embed_documents(texts_to_embed)
            await self.vector_store.add_documents(
                documents=documents_to_add,
                embeddings=embeddings,
                metadata=[d["metadata"] for d in documents_to_add],
            )
            # Mark metadata as indexed
            manifest = self.dataset_manager._read_manifest()
            if dataset_id in manifest:
                manifest[dataset_id]["has_text_indexed"] = True
                self.dataset_manager._write_manifest(manifest)
            return len(documents_to_add)
        except Exception as e:
            logger.error("Failed indexing text for dataset %s: %s", dataset_id, e)
            return 0
