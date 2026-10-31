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

    async def index_dataset_text(
        self,
        dataset_id: str,
        max_rows: int | None = None,
        batch_size: int = 250,
        progress_cb: Any | None = None,
    ) -> int:
        """
        Scan dataset for unstructured text columns, generate embeddings in streaming
        Arrow record batches, and index them into pgvector with strict dataset isolation.
        Memory footprint is strictly bounded to batch_size items.
        """
        import pyarrow.parquet as pq

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

        parquet_path = Path(meta.parquet_path)
        if not parquet_path.exists():
            logger.error("Parquet file not found for dataset %s: %s", dataset_id, parquet_path)
            return 0

        total_indexed = 0
        try:
            pq_file = pq.ParquetFile(parquet_path)
            total_file_rows = pq_file.metadata.num_rows
            target_rows = min(max_rows, total_file_rows) if max_rows is not None else total_file_rows

            # Read only relevant columns in bounded streaming batches
            needed_columns = list(set(text_cols + [col.name for col in profile.columns if any(k in col.name.lower() for k in ("id", "name", "title"))]))
            available_schema_names = set(pq_file.schema.names)
            projection = [c for c in needed_columns if c in available_schema_names]

            processed_rows = 0
            for batch in pq_file.iter_batches(batch_size=batch_size, columns=projection):
                if max_rows is not None and processed_rows >= target_rows:
                    break

                batch_pylist = batch.to_pylist()
                documents_to_add: list[dict[str, Any]] = []
                texts_to_embed: list[str] = []

                for row in batch_pylist:
                    if max_rows is not None and processed_rows >= target_rows:
                        break
                    processed_rows += 1

                    text_parts = [
                        f"{col}: {str(row[col]).strip()}"
                        for col in text_cols
                        if row.get(col) is not None and str(row[col]).strip()
                    ]
                    if not text_parts:
                        continue

                    full_text = "\n".join(text_parts)
                    doc_id = f"{dataset_id}_r{processed_rows}"

                    row_metadata = {
                        "dataset_id": dataset_id,
                        "source_file": meta.filename,
                        "dataset_name": meta.name,
                        "row_index": processed_rows,
                        "document_type": "user_dataset",
                    }
                    for k, v in row.items():
                        if k not in text_cols and v is not None:
                            row_metadata[k] = str(v)[:100]

                    documents_to_add.append({
                        "id": doc_id,
                        "content": full_text,
                        "metadata": row_metadata,
                        "document_type": "user_dataset",
                        "category": meta.name,
                    })
                    texts_to_embed.append(full_text)

                if documents_to_add and self.embedding_service and self.vector_store:
                    try:
                        embeddings = await self.embedding_service.embed_documents(texts_to_embed)
                        await self.vector_store.add_documents(
                            documents=documents_to_add,
                            embeddings=embeddings,
                            metadata=[d["metadata"] for d in documents_to_add],
                        )
                        total_indexed += len(documents_to_add)
                    except Exception as embed_err:
                        logger.warning("Batch embedding insertion failed: %s", embed_err)

                if progress_cb and target_rows > 0:
                    try:
                        progress_cb(processed_rows, target_rows)
                    except Exception:
                        pass

            # Mark metadata as indexed
            manifest = self.dataset_manager._read_manifest()
            if dataset_id in manifest:
                manifest[dataset_id]["has_text_indexed"] = True
                self.dataset_manager._write_manifest(manifest)

            logger.info("Indexed %d text records for dataset %s (%s).", total_indexed, dataset_id, meta.name)
            return total_indexed

        except Exception as e:
            logger.error("Failed streaming indexing for dataset %s: %s", dataset_id, e)
            return total_indexed
