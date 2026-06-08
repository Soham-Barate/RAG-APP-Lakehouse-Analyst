"""
Ingestion pipeline tests.
"""

from __future__ import annotations

from pathlib import Path

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
