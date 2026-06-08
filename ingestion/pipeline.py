"""
Main ingestion pipeline orchestrator.

Coordinates the full ingestion flow:
1. Connect to data source
2. Extract documents
3. Chunk using parent-child strategy
4. Generate embeddings
5. Upsert to vector store(s)
6. Invalidate semantic cache
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from ingestion.chunking.parent_child import ParentChildChunker
from ingestion.connectors.iceberg_connector import IcebergConnector
from ingestion.connectors.unstructured_loader import UnstructuredLoader

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end ingestion pipeline.

    Takes raw data from various sources, chunks it, generates embeddings,
    and upserts into the configured vector store(s).
    """

    def __init__(self, app_state: Any = None) -> None:
        self.app_state = app_state
        self.chunker = ParentChildChunker()
        self.iceberg = IcebergConnector()
        self.loader = UnstructuredLoader()

        # Get services from app state
        self.vector_store = getattr(app_state, "vector_store", None)
        self.embedding_service = getattr(app_state, "embedding_service", None)
        self.semantic_cache = getattr(app_state, "semantic_cache", None)

    async def run(
        self,
        source_type: str,
        source_path: str | None = None,
        table_name: str | None = None,
        document_type: str | None = None,
        batch_size: int = 100,
    ) -> dict[str, Any]:
        """
        Execute the full ingestion pipeline.

        Args:
            source_type: "iceberg", "file", or "synthetic"
            source_path: Path to data file/directory
            table_name: Iceberg table name
            document_type: Type of documents being ingested
            batch_size: Number of documents to process at a time

        Returns:
            Ingestion statistics
        """
        start = time.time()
        stats = {
            "documents_extracted": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "documents_upserted": 0,
            "errors": [],
            "duration_seconds": 0.0,
        }

        logger.info(
            "Starting ingestion: source_type=%s, path=%s",
            source_type, source_path,
        )

        # ----- Step 1: Extract documents -----
        try:
            raw_docs = await self._extract(
                source_type, source_path, table_name, document_type
            )
            stats["documents_extracted"] = len(raw_docs)
            logger.info("Extracted %d documents", len(raw_docs))
        except Exception as exc:
            error = f"Extraction failed: {exc}"
            stats["errors"].append(error)
            logger.error(error)
            return stats

        if not raw_docs:
            logger.warning("No documents extracted")
            return stats

        # ----- Step 2: Chunk documents -----
        all_chunks: list[dict[str, Any]] = []
        for doc in raw_docs:
            chunks = self.chunker.chunk_document(
                content=doc["content"],
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)

        stats["chunks_created"] = len(all_chunks)
        logger.info("Created %d chunks from %d documents", len(all_chunks), len(raw_docs))

        if not self.vector_store or not self.embedding_service:
            logger.warning(
                "Vector store or embedding service not available. "
                "Chunks created but not indexed."
            )
            stats["duration_seconds"] = round(time.time() - start, 2)
            return stats

        # ----- Step 3 & 4: Embed and upsert in batches -----
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]

            try:
                # Extract content for embedding (only child chunks)
                texts_to_embed = [
                    c["content"] for c in batch if not c.get("is_parent")
                ]
                child_chunks = [c for c in batch if not c.get("is_parent")]
                parent_chunks = [c for c in batch if c.get("is_parent")]

                # Generate embeddings for child chunks
                if texts_to_embed:
                    embeddings = await self.embedding_service.embed_documents(
                        texts_to_embed
                    )
                    stats["embeddings_generated"] += len(embeddings)

                    # Upsert child chunks with embeddings
                    metadata_list = [c.get("metadata", {}) for c in child_chunks]
                    await self.vector_store.add_documents(
                        documents=child_chunks,
                        embeddings=embeddings,
                        metadata=metadata_list,
                    )
                    stats["documents_upserted"] += len(child_chunks)

                # Upsert parent chunks (no embedding needed, stored for retrieval)
                if parent_chunks:
                    # Parents get a zero-vector (they're never searched, only fetched)
                    zero_embeddings = [
                        [0.0] * self.embedding_service.dimensions
                        for _ in parent_chunks
                    ]
                    parent_meta = [c.get("metadata", {}) for c in parent_chunks]
                    await self.vector_store.add_documents(
                        documents=parent_chunks,
                        embeddings=zero_embeddings,
                        metadata=parent_meta,
                    )

                logger.info(
                    "Batch %d-%d: %d chunks embedded and upserted",
                    i, i + len(batch), len(batch),
                )

            except Exception as exc:
                error = f"Batch {i} failed: {exc}"
                stats["errors"].append(error)
                logger.error(error)

        # ----- Step 5: Invalidate cache -----
        if self.semantic_cache:
            try:
                await self.semantic_cache.invalidate()
                logger.info("Semantic cache invalidated")
            except Exception as exc:
                logger.warning("Cache invalidation failed: %s", exc)

        stats["duration_seconds"] = round(time.time() - start, 2)

        logger.info(
            "Ingestion complete: %d docs -> %d chunks -> %d upserted (%.1fs)",
            stats["documents_extracted"],
            stats["chunks_created"],
            stats["documents_upserted"],
            stats["duration_seconds"],
        )

        return stats

    async def _extract(
        self,
        source_type: str,
        source_path: str | None,
        table_name: str | None,
        document_type: str | None,
    ) -> list[dict[str, Any]]:
        """Extract documents from the specified source."""
        all_docs: list[dict[str, Any]] = []

        if source_type == "iceberg":
            # Extract from Iceberg via DuckDB
            products = await self.iceberg.extract_products(source_path)
            orders = await self.iceberg.extract_orders(source_path)
            all_docs.extend(products)
            all_docs.extend(orders)

        elif source_type == "file":
            if source_path:
                path = Path(source_path)
                if path.is_dir():
                    all_docs = await self.loader.load_directory(
                        path, document_type
                    )
                else:
                    all_docs = await self.loader.load_file(path, document_type)

        elif source_type == "synthetic":
            from ingestion.synthetic.data_generator import SyntheticDataGenerator

            generator = SyntheticDataGenerator()
            data_path = generator.generate_all()
            all_docs = await self.loader.load_directory(data_path)

        return all_docs
