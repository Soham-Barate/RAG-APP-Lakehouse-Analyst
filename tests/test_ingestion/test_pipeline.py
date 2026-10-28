"""
Ingestion pipeline tests.
"""

from __future__ import annotations

import pytest

from ingestion.chunking.parent_child import ParentChildChunker
from ingestion.synthetic.data_generator import SyntheticDataGenerator


class TestParentChildChunker:
    """Tests for the parent-child chunking strategy."""

    def test_chunks_short_document(self):
        chunker = ParentChildChunker(
            parent_chunk_size=100,
            child_chunk_size=30,
            child_overlap=10,
        )

        chunks = chunker.chunk_document(
            content="This is a short document.",
            metadata={"document_type": "product"},
        )

        assert len(chunks) >= 1
        # Should have at least one parent
        parents = [c for c in chunks if c["is_parent"]]
        assert len(parents) >= 1

    def test_chunks_long_document(self):
        chunker = ParentChildChunker(
            parent_chunk_size=50,
            child_chunk_size=20,
            child_overlap=5,
        )

        # Generate a longer document
        content = " ".join(["word"] * 500)
        chunks = chunker.chunk_document(content)

        parents = [c for c in chunks if c["is_parent"]]
        children = [c for c in chunks if not c["is_parent"]]

        assert len(parents) >= 2
        assert len(children) >= len(parents)

    def test_maintains_parent_child_links(self):
        chunker = ParentChildChunker(
            parent_chunk_size=100,
            child_chunk_size=30,
            child_overlap=5,
        )

        content = " ".join(["word"] * 300)
        chunks = chunker.chunk_document(content)

        parent_ids = {c["id"] for c in chunks if c["is_parent"]}
        children = [c for c in chunks if not c["is_parent"]]

        for child in children:
            assert child["parent_id"] in parent_ids

    def test_metadata_inheritance(self):
        chunker = ParentChildChunker()
        chunks = chunker.chunk_document(
            content="Test content for metadata",
            metadata={"document_type": "review", "category": "Electronics"},
        )

        for chunk in chunks:
            assert chunk["metadata"]["document_type"] == "review"
            assert chunk["metadata"]["category"] == "Electronics"

    def test_empty_content(self):
        chunker = ParentChildChunker()
        chunks = chunker.chunk_document(content="")
        assert chunks == []

    def test_deterministic_ids(self):
        chunker = ParentChildChunker()
        chunks1 = chunker.chunk_document(content="Test content")
        chunks2 = chunker.chunk_document(content="Test content")

        # Same content should produce same IDs
        for c1, c2 in zip(chunks1, chunks2):
            assert c1["id"] == c2["id"]


class TestSyntheticDataGenerator:
    """Tests for the synthetic data generator."""

    def test_generates_products(self):
        gen = SyntheticDataGenerator(output_dir="test_data")
        products = gen._generate_products(10)

        assert len(products) == 10
        for p in products:
            assert "product_id" in p
            assert "name" in p
            assert "category" in p
            assert "price" in p
            assert p["price"] > 0

    def test_generates_reviews_with_sentiment(self):
        gen = SyntheticDataGenerator()
        products = gen._generate_products(5)
        customers = gen._generate_customers(5)
        reviews = gen._generate_reviews(100, products, customers)

        assert len(reviews) == 100
        sentiments = {r["sentiment"] for r in reviews}
        assert "positive" in sentiments
        assert "negative" in sentiments

    def test_generates_support_tickets(self):
        gen = SyntheticDataGenerator()
        products = gen._generate_products(5)
        customers = gen._generate_customers(5)
        orders = gen._generate_orders(10, products, customers)
        tickets = gen._generate_tickets(20, products, orders)

        assert len(tickets) == 20
        for t in tickets:
            assert "ticket_id" in t
            assert "title" in t
            assert "body" in t
            assert "priority" in t


class TestParentChildIngestionRegression:
    """Regression tests verifying parent chunks are excluded from the searchable documents table."""

    @pytest.mark.asyncio
    async def test_parent_chunks_only_in_parent_documents_table(self):
        """Assert that is_parent chunks never enter the documents vector table."""
        from unittest.mock import AsyncMock, MagicMock

        from backend.vectorstore.pgvector_store import PgVectorStore

        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        executed_sqls = []

        async def capture_execute(statement, params=None):
            executed_sqls.append((str(statement), params))
            mock_res = MagicMock()
            mock_res.rowcount = 1
            return mock_res

        mock_conn.execute = capture_execute
        mock_engine.begin.return_value.__aenter__.return_value = mock_conn

        store = PgVectorStore(
            engine=mock_engine,
            embedding_service=MagicMock(),
            settings=MagicMock(openai=MagicMock(embedding_dimensions=3072)),
        )

        test_docs = [
            {"id": "parent_1", "content": "Large parent content", "is_parent": True},
            {"id": "child_1", "content": "Child chunk 1", "is_parent": False, "parent_id": "parent_1"},
        ]
        test_embeddings = [
            [0.1] * 3072,  # embedding for child_1
        ]
        test_meta = [
            {"parent_id": "parent_1"},
            {"parent_id": "parent_1", "document_type": "product"},
        ]

        await store.add_documents(test_docs, embeddings=test_embeddings, metadata=test_meta)

        # Check executed SQLs
        parent_inserts = [
            (sql, p) for sql, p in executed_sqls if "INSERT INTO parent_documents" in sql
        ]
        document_inserts = [
            (sql, p) for sql, p in executed_sqls if "INSERT INTO documents" in sql
        ]

        assert len(parent_inserts) == 1
        assert parent_inserts[0][1]["id"] == "parent_1"

        assert len(document_inserts) == 1
        assert document_inserts[0][1]["id"] == "child_1"
        assert document_inserts[0][1]["parent_id"] == "parent_1"

        # Explicitly verify parent_1 was NEVER inserted into documents table
        for _sql, p in document_inserts:
            assert p["id"] != "parent_1"
