"""
pgvector (PostgreSQL) vector store implementation.

Full hybrid search with HNSW vector index + tsvector full-text search,
parent-child document support, metadata JSONB filtering, and RRF fusion.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from backend.vectorstore.base import VectorStoreBase
from config.settings import Settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SQL DDL for table creation
# ---------------------------------------------------------------------------
CREATE_TABLES_SQL = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Parent documents table (large chunks for full context)
CREATE TABLE IF NOT EXISTS parent_documents (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    metadata    JSONB DEFAULT '{{}}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Child documents table (small chunks for precise retrieval)
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    parent_id       TEXT REFERENCES parent_documents(id) ON DELETE CASCADE,
    content         TEXT NOT NULL,
    embedding       vector({dimensions}),
    content_tsv     tsvector,
    metadata        JSONB DEFAULT '{{}}',
    document_type   TEXT DEFAULT 'product',
    category        TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast vector similarity search
CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- GIN index for full-text search
CREATE INDEX IF NOT EXISTS idx_documents_content_tsv
    ON documents USING gin (content_tsv);

-- JSONB index for metadata filtering
CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING gin (metadata);

-- Index for document type and category filtering
CREATE INDEX IF NOT EXISTS idx_documents_type
    ON documents (document_type);

CREATE INDEX IF NOT EXISTS idx_documents_category
    ON documents (category);

-- Trigger to auto-update tsvector column
CREATE OR REPLACE FUNCTION update_tsv() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv := to_tsvector('english', COALESCE(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_update_tsv ON documents;
CREATE TRIGGER trg_update_tsv
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_tsv();
"""

# ---------------------------------------------------------------------------
# Hybrid search SQL with RRF fusion
# ---------------------------------------------------------------------------
HYBRID_SEARCH_SQL = """
WITH vector_results AS (
    SELECT
        id,
        parent_id,
        content,
        metadata,
        document_type,
        1 - (embedding <=> :query_embedding::vector) AS vector_score,
        ROW_NUMBER() OVER (ORDER BY embedding <=> :query_embedding::vector) AS vector_rank
    FROM documents
    WHERE 1=1
    {filter_clause}
    ORDER BY embedding <=> :query_embedding::vector
    LIMIT :candidate_limit
),
bm25_results AS (
    SELECT
        id,
        parent_id,
        content,
        metadata,
        document_type,
        ts_rank_cd(content_tsv, plainto_tsquery('english', :query_text)) AS bm25_score,
        ROW_NUMBER() OVER (
            ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', :query_text)) DESC
        ) AS bm25_rank
    FROM documents
    WHERE content_tsv @@ plainto_tsquery('english', :query_text)
    {filter_clause}
    LIMIT :candidate_limit
),
fused AS (
    SELECT
        COALESCE(v.id, b.id) AS id,
        COALESCE(v.parent_id, b.parent_id) AS parent_id,
        COALESCE(v.content, b.content) AS content,
        COALESCE(v.metadata, b.metadata) AS metadata,
        COALESCE(v.document_type, b.document_type) AS document_type,
        COALESCE(v.vector_score, 0) AS vector_score,
        COALESCE(b.bm25_score, 0) AS bm25_score,
        -- Reciprocal Rank Fusion
        COALESCE(1.0 / (60 + v.vector_rank), 0) +
        COALESCE(1.0 / (60 + b.bm25_rank), 0) AS rrf_score
    FROM vector_results v
    FULL OUTER JOIN bm25_results b ON v.id = b.id
)
SELECT *
FROM fused
ORDER BY rrf_score DESC
LIMIT :top_k;
"""


class PgVectorStore(VectorStoreBase):
    """
    PostgreSQL + pgvector hybrid vector store.

    Features:
    - HNSW index for fast approximate nearest neighbor search
    - tsvector + GIN index for BM25-style full-text search
    - RRF fusion of vector and keyword results in a single SQL query
    - Parent-child document support
    - JSONB metadata filtering
    """

    def __init__(
        self,
        engine: AsyncEngine,
        embedding_service: Any,
        settings: Settings,
    ) -> None:
        self.engine = engine
        self.embedding_service = embedding_service
        self.dimensions = settings.openai.embedding_dimensions

    async def initialize(self) -> None:
        """Create tables and indexes if they don't exist."""
        ddl = CREATE_TABLES_SQL.format(dimensions=self.dimensions)
        try:
            async with self.engine.begin() as conn:
                # Execute each statement separately
                for statement in ddl.split(";"):
                    statement = statement.strip()
                    if statement:
                        await conn.execute(text(statement))
            logger.info("pgvector tables and indexes initialized")
        except Exception as exc:
            logger.error("Failed to initialize pgvector tables: %s", exc)
            raise

    async def add_documents(
        self,
        documents: list[dict[str, Any]],
        embeddings: list[list[float]] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        """Add documents with embeddings to PostgreSQL."""
        if not documents:
            return []

        metadata = metadata or [{}] * len(documents)
        doc_ids: list[str] = []

        async with self.engine.begin() as conn:
            # First pass: Upsert all parent documents if present
            for doc, meta in zip(documents, metadata):
                if doc.get("is_parent"):
                    p_id = doc.get("id") or meta.get("parent_id")
                    if p_id:
                        content = doc.get("content", "")
                        await conn.execute(
                            text(
                                "INSERT INTO parent_documents (id, content, metadata) "
                                "VALUES (:id, :content, :metadata) "
                                "ON CONFLICT (id) DO UPDATE SET content = :content, metadata = :metadata"
                            ),
                            {
                                "id": p_id,
                                "content": content,
                                "metadata": json.dumps(meta),
                            },
                        )
                        doc_ids.append(p_id)

            # Second pass: Upsert child or regular documents into vector table (never parent chunks)
            if embeddings:
                non_parent_pairs = [(d, m) for d, m in zip(documents, metadata) if not d.get("is_parent")]
                if len(embeddings) == len(non_parent_pairs):
                    child_iterable = zip([d for d, _ in non_parent_pairs], embeddings, [m for _, m in non_parent_pairs])
                else:
                    child_iterable = zip(documents, embeddings, metadata)

                for doc, emb, meta in child_iterable:
                    if doc.get("is_parent"):
                        # Parent documents live strictly in parent_documents; never pollute documents
                        continue

                    doc_id = doc.get("id", str(uuid.uuid4()))
                    parent_id = meta.get("parent_id") or doc.get("parent_id")
                    content = doc.get("content", "")
                    doc_type = meta.get("document_type", "product")
                    category = meta.get("category", "")

                    # Insert child/regular document
                    embedding_str = f"[{','.join(str(x) for x in emb)}]"
                    await conn.execute(
                        text(
                            "INSERT INTO documents "
                            "(id, parent_id, content, embedding, metadata, document_type, category) "
                            "VALUES (:id, :parent_id, :content, :embedding::vector, "
                            ":metadata, :doc_type, :category) "
                            "ON CONFLICT (id) DO UPDATE SET "
                            "content = :content, embedding = :embedding::vector, "
                            "metadata = :metadata"
                        ),
                        {
                            "id": doc_id,
                            "parent_id": parent_id,
                            "content": content,
                            "embedding": embedding_str,
                            "metadata": json.dumps(meta),
                            "doc_type": doc_type,
                            "category": category,
                        },
                    )
                    doc_ids.append(doc_id)

        logger.info("Added %d documents to pgvector", len(doc_ids))
        return doc_ids

    async def hybrid_search(
        self,
        query_text: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid vector + BM25 search with RRF fusion."""
        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = await self.embedding_service.embed_query(query_text)

        # Build filter clause with parameterized values
        extra_params: dict[str, Any] = {}
        filter_clause = self._build_filter_clause(filters, extra_params)

        # Format embedding for PostgreSQL
        embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

        sql = HYBRID_SEARCH_SQL.format(filter_clause=filter_clause)

        query_params = {
            "query_embedding": embedding_str,
            "query_text": query_text,
            "top_k": top_k,
            "candidate_limit": top_k * 5,
            **extra_params,
        }

        async with self.engine.begin() as conn:
            result = await conn.execute(text(sql), query_params)
            rows = result.fetchall()

        documents: list[dict[str, Any]] = []
        for row in rows:
            meta = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
            documents.append({
                "id": row.id,
                "parent_id": row.parent_id,
                "content": row.content,
                "metadata": {**meta, "document_type": row.document_type},
                "score": float(row.rrf_score),
                "vector_score": float(row.vector_score),
                "bm25_score": float(row.bm25_score),
                "rrf_score": float(row.rrf_score),
            })

        return documents

    async def delete_documents(self, ids: list[str]) -> int:
        """Delete documents by ID."""
        if not ids:
            return 0

        async with self.engine.begin() as conn:
            result = await conn.execute(
                text("DELETE FROM documents WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
            return result.rowcount or 0

    async def get_parent_document(self, parent_id: str) -> dict[str, Any] | None:
        """Retrieve a parent document by ID."""
        async with self.engine.begin() as conn:
            result = await conn.execute(
                text("SELECT id, content, metadata FROM parent_documents WHERE id = :id"),
                {"id": parent_id},
            )
            row = result.fetchone()

        if row:
            meta = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
            return {
                "id": row.id,
                "content": row.content,
                "metadata": meta,
            }
        return None

    async def get_document_count(self) -> int:
        """Return total document count."""
        async with self.engine.begin() as conn:
            result = await conn.execute(text("SELECT COUNT(*) FROM documents"))
            row = result.fetchone()
            return row[0] if row else 0

    def _build_filter_clause(
        self,
        filters: dict[str, Any] | None,
        params: dict[str, Any],
    ) -> str:
        """
        Build SQL WHERE clause from metadata filters using parameterized
        queries to prevent SQL injection.

        Args:
            filters: User-provided filter dict.
            params: Mutable dict that will be populated with bind-parameter
                    values for use in ``text()`` queries.

        Returns:
            SQL fragment with ``:param`` placeholders (safe for ``text()``).
        """
        if not filters:
            return ""

        clauses: list[str] = []

        if filters.get("categories"):
            placeholders = []
            for idx, cat in enumerate(filters["categories"]):
                key = f"flt_cat_{idx}"
                params[key] = cat
                placeholders.append(f":{key}")
            clauses.append(f"AND category IN ({', '.join(placeholders)})")

        if filters.get("document_types"):
            placeholders = []
            for idx, dt in enumerate(filters["document_types"]):
                key = f"flt_dt_{idx}"
                params[key] = dt
                placeholders.append(f":{key}")
            clauses.append(f"AND document_type IN ({', '.join(placeholders)})")

        if filters.get("date_from"):
            params["flt_date_from"] = filters["date_from"]
            clauses.append("AND created_at >= :flt_date_from")

        if filters.get("date_to"):
            params["flt_date_to"] = filters["date_to"]
            clauses.append("AND created_at <= :flt_date_to")

        if filters.get("departments"):
            placeholders = []
            for idx, dept in enumerate(filters["departments"]):
                key = f"flt_dept_{idx}"
                params[key] = dept
                placeholders.append(f":{key}")
            clauses.append(
                f"AND metadata->>'department' IN ({', '.join(placeholders)})"
            )

        return " ".join(clauses)

    async def close(self) -> None:
        """Dispose engine connections."""
        if self.engine:
            await self.engine.dispose()
