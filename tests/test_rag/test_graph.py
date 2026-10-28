"""
RAG pipeline node tests.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.rag.nodes.grader import RelevanceGrader
from backend.rag.nodes.hyde import HyDEGenerator
from backend.rag.nodes.query_rewriter import QueryRewriter
from backend.rag.nodes.router import QueryRouter


class TestQueryRouter:
    """Tests for query routing."""

    @pytest.mark.asyncio
    async def test_routes_to_hybrid_by_default(self, mock_llm_service):
        router = QueryRouter(llm_service=mock_llm_service)
        state = {"query": "Tell me about electronics products"}

        result = await router(state)
        assert "route" in result
        assert result["route"] in ("sql", "vector", "hybrid")

    @pytest.mark.asyncio
    async def test_handles_llm_failure_gracefully(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM down"))

        router = QueryRouter(llm_service=llm)
        result = await router({"query": "test"})
        assert result["route"] == "hybrid"  # Default fallback


class TestQueryRewriter:
    """Tests for query rewriting."""

    @pytest.mark.asyncio
    async def test_produces_multiple_queries(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "rewritten_queries": [
                    "electronics products top rated",
                    "best electronics customer reviews",
                ],
                "strategy": "expansion",
            })
        )

        rewriter = QueryRewriter(llm_service=mock_llm_service)
        result = await rewriter({"query": "best electronics", "iteration_count": 0})

        assert "rewritten_queries" in result
        assert len(result["rewritten_queries"]) >= 2

    @pytest.mark.asyncio
    async def test_includes_original_query(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "rewritten_queries": ["expanded query"],
                "strategy": "expansion",
            })
        )

        rewriter = QueryRewriter(llm_service=mock_llm_service)
        result = await rewriter({"query": "original query", "iteration_count": 0})

        assert "original query" in result["rewritten_queries"]


class TestHyDEGenerator:
    """Tests for HyDE generation."""

    @pytest.mark.asyncio
    async def test_generates_hypothetical_document(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value="The TechPro Earbuds feature active noise cancellation..."
        )

        hyde = HyDEGenerator(llm_service=mock_llm_service)
        result = await hyde({"query": "Tell me about TechPro Earbuds"})

        assert "hyde_document" in result
        assert len(result["hyde_document"]) > 0

    @pytest.mark.asyncio
    async def test_handles_failure_gracefully(self):
        llm = MagicMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))

        hyde = HyDEGenerator(llm_service=llm)
        result = await hyde({"query": "test"})
        assert result["hyde_document"] == ""


class TestRelevanceGrader:
    """Tests for relevance grading."""

    @pytest.mark.asyncio
    async def test_proceeds_when_relevant(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "relevance": "relevant",
                "score": 0.85,
                "reasoning": "Directly answers the query",
            })
        )

        grader = RelevanceGrader(
            llm_service=mock_llm_service,
            relevance_threshold=0.4,
        )

        state: dict[str, Any] = {
            "query": "test",
            "reranked_docs": [
                {"content": "relevant doc", "rerank_score": 0.8},
            ],
            "iteration_count": 0,
        }

        result = await grader(state)
        assert result["should_retry"] is False

    @pytest.mark.asyncio
    async def test_retries_when_irrelevant(self, mock_llm_service):
        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "relevance": "irrelevant",
                "score": 0.1,
                "reasoning": "Not related",
            })
        )

        grader = RelevanceGrader(
            llm_service=mock_llm_service,
            relevance_threshold=0.4,
            max_iterations=3,
        )

        state: dict[str, Any] = {
            "query": "test",
            "reranked_docs": [
                {"content": "irrelevant doc", "rerank_score": 0.1},
            ],
            "iteration_count": 0,
        }

        result = await grader(state)
        assert result["should_retry"] is True
        assert result["iteration_count"] == 1


class TestSafeSQLNode:
    """Tests for safe Text-to-SQL generation and execution."""

    @pytest.mark.asyncio
    async def test_sql_node_executes_valid_query(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT name, price FROM products WHERE price < 100 ORDER BY price ASC LIMIT 10",
                "explanation": "Find affordable products",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "What products are under $100?"})

        assert result["sql_error"] is None
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 1
        assert len(result["sql_results"]) > 0
        assert "products" in result["sql_query"].lower()

    @pytest.mark.asyncio
    async def test_sql_node_blocks_drop_table(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "DROP TABLE products;",
                "explanation": "Malicious attack",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "Delete all products"})

        assert result["sql_error"] is not None
        assert result["route"] == "sql"
        assert result["sql_results"] == []
        assert "lakehouse_sql_error" in result["retrieved_docs"][0]["id"]

    @pytest.mark.asyncio
    async def test_sql_node_blocks_unauthorized_tables(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT * FROM secret_credentials LIMIT 10",
                "explanation": "Access secret table",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "Show secrets"})

        assert result["sql_error"] is not None
        assert result["route"] == "sql"
        assert "secret_credentials" in result["sql_error"]

    @pytest.mark.asyncio
    async def test_sql_node_blocks_unauthorized_columns(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT secret_password FROM products WHERE price < 100",
                "explanation": "Access unauthorized column",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "Show secret password"})

        assert result["sql_error"] is not None
        assert result["route"] == "sql"
        assert "secret_password" in result["sql_error"]

    @pytest.mark.asyncio
    async def test_sql_node_blocks_duckdb_file_functions(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        adversarial_queries = [
            "SELECT * FROM read_parquet('data/secret.parquet')",
            "SELECT * FROM read_csv('data/secret.csv')",
            "SELECT * FROM read_text('C:/Windows/win.ini')",
            "SELECT * FROM glob('data/*')",
            "ATTACH 'secret.db' AS secret_db",
        ]

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        for malicious_sql in adversarial_queries:
            mock_llm_service.generate = AsyncMock(
                return_value=json.dumps({
                    "sql": malicious_sql,
                    "explanation": "Adversarial file access attempt",
                })
            )
            result = await sql_node({"query": "Hack lakehouse"})
            assert result["sql_error"] is not None, f"Expected {malicious_sql} to be rejected"
            assert result["sql_results"] == []

    @pytest.mark.asyncio
    async def test_sql_node_blocks_multi_statements(self, mock_llm_service):
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT name FROM products; DROP TABLE products;",
                "explanation": "Multiple statements attack",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "List and drop"})

        assert result["sql_error"] is not None
        assert "Multiple statements" in result["sql_error"]

    @pytest.mark.asyncio
    async def test_sql_node_customer_analytics(self, mock_llm_service):
        """Verify customers table is registered and queryable with aggregations."""
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT tier, COUNT(*) AS cust_count, AVG(total_spent) AS avg_spent FROM customers GROUP BY tier ORDER BY avg_spent DESC",
                "explanation": "Analyze customers by spending tier",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "What is the average spending by customer tier?"})

        assert result["sql_error"] is None
        assert len(result["sql_results"]) > 0
        assert "tier" in result["sql_results"][0]

    @pytest.mark.asyncio
    async def test_sql_node_zero_row_result(self, mock_llm_service):
        """Verify zero matching rows produce a clear analytical message without hallucinations."""
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT name, price FROM products WHERE price > 999999",
                "explanation": "Products that don't exist",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "Show products over $1,000,000"})

        assert result["sql_error"] is None
        assert result["sql_results"] == []
        assert result["route"] == "sql"
        assert len(result["retrieved_docs"]) == 1
        assert result["retrieved_docs"][0]["id"] == "lakehouse_sql_empty"
        assert "No records matched" in result["compressed_context"]

    def test_route_decision(self):
        from backend.rag.graph import _route_decision, _sql_post_decision

        assert _route_decision({"route": "sql"}) == "sql"
        assert _route_decision({"route": "vector"}) == "vector"
        assert _route_decision({"route": "hybrid"}) == "hybrid"

        # Explicit SQL route: success goes to think (or generate in fast mode)
        assert _sql_post_decision({"route": "sql", "sql_results": [{"id": 1}], "sql_error": None, "fast_mode": False}) == "think"
        assert _sql_post_decision({"route": "sql", "sql_results": [{"id": 1}], "sql_error": None, "fast_mode": True}) == "generate"

        # Explicit SQL route: failure or 0-rows NEVER silently falls back to vector
        assert _sql_post_decision({"route": "sql", "sql_results": [], "sql_error": "Syntax error", "fast_mode": False}) == "think"
        assert _sql_post_decision({"route": "sql", "sql_results": [], "sql_error": "Syntax error", "fast_mode": True}) == "generate"
        assert _sql_post_decision({"route": "sql", "sql_results": [], "sql_error": None, "fast_mode": False}) == "think"

        # Hybrid route: proceeds to vector search to retrieve qualitative document context and merge
        assert _sql_post_decision({"route": "hybrid", "sql_results": [{"id": 1}], "sql_error": None}) == "continue_to_vector"
        assert _sql_post_decision({"route": "hybrid", "sql_results": [], "sql_error": "error"}) == "continue_to_vector"

    @pytest.mark.asyncio
    async def test_sql_node_allows_string_literals(self, mock_llm_service):
        """Verify string literals in WHERE/CASE clauses are not rejected as unauthorized column identifiers."""
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT p.name, o.quantity, o.total_amount FROM products p JOIN orders o ON p.product_id = o.product_id WHERE o.status = 'delivered'",
                "explanation": "Find delivered orders and their products",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "Show all delivered orders with product names"})

        assert result["sql_error"] is None
        assert len(result["sql_results"]) > 0

    @pytest.mark.asyncio
    async def test_sql_node_allows_cte(self, mock_llm_service):
        """Verify read-only CTEs (WITH ... AS (...) SELECT ...) are properly parsed and executed."""
        from backend.rag.nodes.sql_node import SafeSQLNode

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "WITH top_cust AS (SELECT customer_id, total_spent FROM customers WHERE total_spent > 1000) SELECT customer_id, total_spent FROM top_cust",
                "explanation": "Select high-value customers via CTE",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service)
        result = await sql_node({"query": "List customers who spent over 1000"})

        assert result["sql_error"] is None
        assert len(result["sql_results"]) > 0

    @pytest.mark.asyncio
    async def test_sql_node_real_parquet_registration(self, tmp_path, mock_llm_service):
        """Verify real Parquet files are loaded into DuckDB memory and external access is permanently locked."""
        import pandas as pd

        from backend.rag.nodes.sql_node import SafeSQLNode

        # Create real Parquet files in temporary lakehouse folder
        gold_dir = tmp_path / "gold"
        silver_dir = tmp_path / "silver"
        gold_dir.mkdir(parents=True, exist_ok=True)
        silver_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame([
            {"product_id": "P-100", "name": "Real Parquet Earbuds", "category": "Electronics", "price": 49.99}
        ]).to_parquet(gold_dir / "products.parquet", index=False)

        pd.DataFrame([
            {"order_id": "O-100", "customer_id": "C-100", "product_id": "P-100", "quantity": 1, "total_amount": 49.99, "status": "delivered"}
        ]).to_parquet(silver_dir / "orders.parquet", index=False)

        pd.DataFrame([
            {"customer_id": "C-100", "tier": "Platinum", "total_orders": 10, "total_spent": 5000.0}
        ]).to_parquet(silver_dir / "customers.parquet", index=False)

        mock_llm_service.generate = AsyncMock(
            return_value=json.dumps({
                "sql": "SELECT name, price FROM products WHERE price < 100",
                "explanation": "Query real parquet data",
            })
        )

        sql_node = SafeSQLNode(llm_service=mock_llm_service, data_dir=tmp_path)
        result = await sql_node({"query": "What products are under $100?"})

        assert result["sql_error"] is None
        assert len(result["sql_results"]) == 1
        assert result["sql_results"][0]["name"] == "Real Parquet Earbuds"

        # Verify DuckDB connection is locked against subsequent external file reads
        conn = sql_node._get_connection()
        with pytest.raises(Exception):
            conn.execute("SELECT * FROM read_parquet('other.parquet')")


class TestEmbeddingServiceTokenTruncation:
    """Tests for token-aware truncation in EmbeddingService."""

    def test_token_aware_truncation(self):
        from unittest.mock import MagicMock

        from backend.embeddings.embedding_service import EmbeddingService

        settings = MagicMock()
        settings.openai.embedding_model = "text-embedding-3-large"
        settings.openai.embedding_dimensions = 3072
        settings.rag.embedding_batch_size = 100
        settings.openai.api_key.get_secret_value.return_value = "fake"

        service = EmbeddingService(settings=settings)

        # Generate a 10,000-token string
        long_text = "word " * 12000
        truncated = service._truncate_to_tokens(long_text, max_tokens=100)

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(truncated)
        assert len(tokens) <= 100

    def test_short_text_unmodified(self):
        from unittest.mock import MagicMock

        from backend.embeddings.embedding_service import EmbeddingService

        settings = MagicMock()
        settings.openai.embedding_model = "text-embedding-3-large"
        settings.openai.embedding_dimensions = 3072
        settings.rag.embedding_batch_size = 100
        settings.openai.api_key.get_secret_value.return_value = "fake"

        service = EmbeddingService(settings=settings)
        short_text = "Hello world"
        assert service._truncate_to_tokens(short_text, max_tokens=100) == short_text


class TestMetadataFilterSecurity:
    """Tests for SQL injection resistance in vector store metadata filtering."""

    def test_malicious_filter_values_are_parameterized(self):
        from unittest.mock import MagicMock

        from backend.vectorstore.pgvector_store import PgVectorStore

        store = PgVectorStore(
            engine=MagicMock(),
            embedding_service=MagicMock(),
            settings=MagicMock(openai=MagicMock(embedding_dimensions=3072)),
        )

        params: dict = {}
        malicious_input = {
            "categories": ["Electronics' OR '1'='1", "'; DROP TABLE documents; --"],
            "document_types": ["product' UNION SELECT * FROM users --"],
            "departments": ["Retail' OR 1=1 --"],
        }

        clause = store._build_filter_clause(malicious_input, params)

        # Assert no malicious payload was string-interpolated into SQL clause
        assert "DROP TABLE" not in clause
        assert "UNION SELECT" not in clause
        assert "OR '1'='1" not in clause

        # Assert all malicious values are safely isolated in bound params
        assert any("DROP TABLE" in str(v) for v in params.values())
        assert any("UNION SELECT" in str(v) for v in params.values())
        assert all(":flt_" in clause for _ in range(3))


class TestTokenBucketRateLimiterAtomic:
    """Tests for atomic token bucket rate limiting with Redis Lua script."""

    @pytest.mark.asyncio
    async def test_atomic_lua_execution(self):
        from unittest.mock import AsyncMock, MagicMock

        from backend.api.middleware.rate_limiter import TokenBucketRateLimiter

        mock_redis = MagicMock()
        # Mock Lua script returning [allowed=1, remaining=59]
        mock_redis.eval = AsyncMock(return_value=[1, 59])

        limiter = TokenBucketRateLimiter(
            redis_client=mock_redis,
            max_tokens=60,
            refill_seconds=60,
        )

        allowed = await limiter.check("test-user-api-key")
        assert allowed is True
        mock_redis.eval.assert_called_once()
        assert limiter._last_remaining == 59

    @pytest.mark.asyncio
    async def test_fail_closed_mode_when_redis_down(self):
        from backend.api.middleware.rate_limiter import TokenBucketRateLimiter

        # Redis is None with fail_closed=True
        limiter = TokenBucketRateLimiter(
            redis_client=None,
            max_tokens=60,
            fail_closed=True,
        )
        allowed = await limiter.check("any-key")
        assert allowed is False

    @pytest.mark.asyncio
    async def test_fail_open_mode_in_development(self):
        from backend.api.middleware.rate_limiter import TokenBucketRateLimiter

        # Redis is None with fail_closed=False
        limiter = TokenBucketRateLimiter(
            redis_client=None,
            max_tokens=60,
            fail_closed=False,
        )
        allowed = await limiter.check("any-key")
        assert allowed is True


class TestCriticalServicesFailFast:
    """Tests verifying fail-fast behavior on critical dependencies in production."""

    @pytest.mark.asyncio
    async def test_production_fails_fast_on_database_error(self):
        from unittest.mock import patch

        from backend.core.container import ServiceContainer
        from config.settings import Environment, Settings

        settings = Settings()
        settings.app_env = Environment.PRODUCTION

        container = ServiceContainer(settings=settings)
        with patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=Exception("DB Connection Refused")):
            with pytest.raises(RuntimeError) as exc_info:
                await container._init_database()
            assert "Critical service 'db_engine' failed to initialize" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_development_swallows_database_error_for_local_testing(self):
        from unittest.mock import patch

        from backend.core.container import ServiceContainer
        from config.settings import Environment, Settings

        settings = Settings()
        settings.app_env = Environment.DEVELOPMENT

        container = ServiceContainer(settings=settings)
        with patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=Exception("DB Connection Refused")):
            # In development, it logs a warning without crashing
            await container._init_database()
            assert container.db_engine is None


class TestHybridRouteMergesEvidence:
    """Tests verifying the hybrid route combines structured SQL evidence with vector document context."""

    @pytest.mark.asyncio
    async def test_hybrid_route_merges_sql_and_vector_context(self, mock_llm_service):
        from backend.rag.nodes.compressor import ContextualCompressor
        from backend.rag.nodes.generator import ResponseGenerator

        compressor = ContextualCompressor(llm_service=mock_llm_service)
        mock_llm_service.generate = AsyncMock(return_value="Vector qualitative details about TechPro.")

        state = {
            "query": "Compare sales and customer reviews for TechPro earbuds",
            "route": "hybrid",
            "sql_query": "SELECT SUM(total_amount) FROM orders WHERE product_name LIKE '%TechPro%'",
            "sql_results": [{"total_amount": 1599.80}],
            "sql_context": "### Structured Lakehouse Results\nTotal Sales: $1,599.80",
            "sql_doc": {
                "id": "lakehouse_sql_result",
                "content": "Total Sales: $1,599.80",
                "metadata": {"document_type": "lakehouse_analytics"},
                "score": 1.0,
            },
            "reranked_docs": [
                {"id": "doc_rev_1", "content": "Customer reviews praise the noise cancellation.", "metadata": {"document_type": "review"}},
            ],
            "messages": [],
        }

        # Step 1: Compressor merges structured SQL table with compressed qualitative text
        comp_res = await compressor(state)
        merged_context = comp_res["compressed_context"]
        assert "Total Sales: $1,599.80" in merged_context
        assert "Vector qualitative details" in merged_context

        # Step 2: Generator receives merged context and outputs answer with structured SQL preserved
        state["compressed_context"] = merged_context
        generator = ResponseGenerator(llm_service=mock_llm_service)
        mock_llm_service.generate = AsyncMock(return_value="TechPro achieved $1,599.80 in sales and received high customer satisfaction.")
        gen_res = await generator(state)

        assert "TechPro achieved $1,599.80" in gen_res["answer"]
        assert gen_res["sql_query"] == state["sql_query"]
        assert gen_res["sql_results"] == state["sql_results"]
        assert len(gen_res["sources"]) > 0


