"""
Safe, schema-aware Text-to-SQL Node.

Translates structured/analytical questions into read-only SQL queries
executed against DuckDB (querying Iceberg/Parquet lakehouse tables).
Includes strict SQL validation, table/column allowlisting, and automatic
fallback to vector retrieval upon failure.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from backend.rag.prompts.templates import TEXT_TO_SQL_PROMPT
from backend.rag.state import RAGState

logger = logging.getLogger(__name__)

# Security Allowlist
ALLOWED_TABLES = {
    "products", "orders", "customers",
    "daily_revenue", "category_metrics", "customer_lifetime_value"
}

ALLOWED_COLUMNS = {
    "products": {
        "product_id", "name", "category", "description", "specifications", "price",
        "rating", "review_count", "sku", "in_stock", "department"
    },
    "orders": {
        "order_id", "customer_id", "product_id", "product_name", "quantity",
        "total_amount", "status", "order_date", "payment_method", "shipping_method", "item_count"
    },
    "customers": {
        "customer_id", "tier", "total_orders", "total_spent", "join_date", "preferred_category"
    },
    "daily_revenue": {
        "order_date", "status", "payment_method", "total_orders", "total_items", "daily_revenue", "avg_order_value"
    },
    "category_metrics": {
        "category", "product_count", "total_orders", "total_units_sold", "total_revenue", "avg_price", "avg_rating"
    },
    "customer_lifetime_value": {
        "customer_id", "tier", "preferred_category", "order_count", "lifetime_spent", "avg_order_value", "last_order_date"
    }
}
ALL_ALLOWED_COLUMNS = set().union(*ALLOWED_COLUMNS.values())

DISALLOWED_KEYWORDS = {
    # DDL & DML operations
    "insert", "update", "delete", "drop", "alter", "create", "truncate", "replace",
    "upsert", "merge", "grant", "revoke",
    # Administrative & session management
    "attach", "detach", "copy", "export", "import", "pragma", "call", "load", "install",
    "system", "shell", "set", "reset", "checkpoint", "vacuum", "transaction", "commit", "rollback",
    # DuckDB file-reading / table functions (external resource access)
    "read_parquet", "read_csv", "read_csv_auto", "read_json", "read_json_auto",
    "read_ndjson", "read_ndjson_auto", "read_text", "read_blob", "scan_parquet",
    "scan_csv", "parquet_scan", "csv_scan", "glob",
    # Remote network access
    "httpfs", "s3", "azure", "gcs",
    # DuckDB catalog metadata functions
    "duckdb_settings", "duckdb_tables", "duckdb_views", "duckdb_columns"
}

ALLOWED_SQL_KEYWORDS = {
    # Core DQL
    "select", "from", "where", "group", "by", "having", "order", "limit", "offset", "as", "distinct",
    # Joins
    "join", "left", "right", "inner", "outer", "full", "cross", "natural", "on", "using",
    # Logical & comparison
    "and", "or", "not", "is", "null", "in", "like", "ilike", "between", "similar", "to", "exists",
    "case", "when", "then", "else", "end", "asc", "desc", "true", "false", "all", "any", "some",
    # Set, CTE & Window
    "with", "recursive", "union", "intersect", "except", "over", "partition", "window",
    # Null ordering
    "nulls", "first", "last",
    # Intervals & time units
    "interval", "year", "years", "month", "months", "day", "days", "quarter", "quarters",
    "week", "weeks", "hour", "hours", "minute", "minutes", "second", "seconds",
    # Qualify & Filter
    "qualify", "filter",
    # Data type casts
    "varchar", "integer", "int", "bigint", "smallint", "tinyint", "double", "float",
    "decimal", "numeric", "boolean", "bool", "date", "timestamp", "timestamptz", "text"
}

ALLOWED_FUNCTIONS = {
    # Aggregations
    "count", "sum", "avg", "min", "max", "median", "mode",
    "quantile_cont", "quantile_disc", "stddev", "stddev_samp", "var_samp", "variance",
    "approx_count_distinct",
    # Math
    "round", "abs", "ceil", "ceiling", "floor", "power", "sqrt", "cbrt", "sign",
    "log", "ln", "log10", "exp",
    # Conditionals & Nulls
    "coalesce", "nullif", "ifnull", "nvl", "greatest", "least", "cast", "try_cast", "typeof",
    # Date & Time (DuckDB native)
    "date_trunc", "extract", "year", "month", "day", "quarter", "epoch", "epoch_ms",
    "strftime", "strptime", "to_timestamp", "make_date", "make_timestamp",
    "date_add", "date_sub", "datediff", "date_diff", "date_part",
    "now", "current_date", "current_timestamp", "current_time", "today",
    "dayofweek", "dayofyear", "week", "isodow", "isoyear", "age",
    # String manipulation
    "lower", "upper", "trim", "ltrim", "rtrim", "concat", "concat_ws", "length", "len",
    "substr", "substring", "replace", "split_part", "contains", "starts_with", "ends_with",
    "string_agg", "group_concat", "listagg", "array_agg",
    # Window functions
    "row_number", "rank", "dense_rank", "ntile", "lag", "lead", "first_value", "last_value"
}

ALLOWED_IDENTIFIERS = ALL_ALLOWED_COLUMNS | ALLOWED_TABLES | ALLOWED_SQL_KEYWORDS | ALLOWED_FUNCTIONS


class SQLValidationError(Exception):
    """Raised when generated SQL violates safety constraints."""
    pass


class SafeSQLNode:
    """
    Translates analytical queries into SQL and executes safely via DuckDB.
    """

    _active_instances: list["SafeSQLNode"] = []

    def __init__(
        self,
        llm_service: Any,
        data_dir: str | Path = "data",
    ) -> None:
        self.llm_service = llm_service
        self.data_dir = Path(data_dir)
        self._conn: Any = None
        SafeSQLNode._active_instances.append(self)

    @classmethod
    def reset_connection(cls) -> None:
        """Reset DuckDB connections across all active instances so new data is loaded."""
        for inst in cls._active_instances:
            if inst._conn is not None:
                try:
                    inst._conn.close()
                except Exception:
                    pass
                inst._conn = None
        logger.info("Reset DuckDB connections for fresh data ingestion.")

    def _get_connection(self) -> Any:
        """Initialize or return the DuckDB connection with zero-copy views and resource boundaries."""
        if self._conn is None:
            import duckdb
            import os

            max_memory = os.getenv("DUCKDB_MAX_MEMORY", "4GB")
            threads = min(os.cpu_count() or 4, 8)

            # Create in-memory connection with scalable resource boundaries
            self._conn = duckdb.connect(
                database=":memory:",
                config={
                    "max_memory": max_memory,
                    "threads": threads,
                },
            )
            # Register lakehouse tables from Parquet/JSON as zero-copy views
            self._register_views()

        return self._conn

    def _register_views(self) -> None:
        """Register products, orders, and customers tables in DuckDB as zero-copy views."""
        conn = self._conn
        gold_parquet = self.data_dir / "gold" / "products.parquet"
        silver_parquet = self.data_dir / "silver" / "orders.parquet"
        customers_parquet = self.data_dir / "silver" / "customers.parquet"

        gold_json = self.data_dir / "gold" / "products.json"
        silver_json = self.data_dir / "silver" / "orders.json"
        customers_json = self.data_dir / "silver" / "customers.json"

        # Register products as view
        if gold_parquet.exists():
            conn.execute(f"CREATE OR REPLACE VIEW products AS SELECT * FROM read_parquet('{gold_parquet.as_posix()}');")
        elif gold_json.exists():
            conn.execute(f"CREATE OR REPLACE VIEW products AS SELECT * FROM read_json_auto('{gold_json.as_posix()}');")
        else:
            conn.execute("""
                DROP VIEW IF EXISTS products;
                DROP TABLE IF EXISTS products;
                CREATE TABLE products (
                    product_id VARCHAR PRIMARY KEY, name VARCHAR, category VARCHAR, description VARCHAR,
                    specifications VARCHAR, price DOUBLE, rating DOUBLE, review_count INTEGER,
                    sku VARCHAR, in_stock BOOLEAN, department VARCHAR
                );
                INSERT INTO products VALUES
                ('PROD-00001', 'TechPro Wireless Earbuds', 'Electronics', 'Noise cancelling earbuds', 'Weight: 0.2kg', 79.99, 4.5, 120, 'SKU-001', true, 'Retail'),
                ('PROD-00002', 'SmartFit Watch Pro', 'Electronics', 'Fitness smartwatch', 'Weight: 0.1kg', 149.99, 4.7, 340, 'SKU-002', true, 'Retail'),
                ('PROD-00003', 'UltraClean Air Purifier', 'Home & Garden', 'HEPA filter purifier', 'Weight: 4.5kg', 199.99, 4.8, 89, 'SKU-003', true, 'Online Exclusive');
            """)

        # Register orders as view
        if silver_parquet.exists():
            conn.execute(f"CREATE OR REPLACE VIEW orders AS SELECT * FROM read_parquet('{silver_parquet.as_posix()}');")
        elif silver_json.exists():
            conn.execute(f"CREATE OR REPLACE VIEW orders AS SELECT * FROM read_json_auto('{silver_json.as_posix()}');")
        else:
            conn.execute("""
                DROP VIEW IF EXISTS orders;
                DROP TABLE IF EXISTS orders;
                CREATE TABLE orders (
                    order_id VARCHAR PRIMARY KEY, customer_id VARCHAR, product_id VARCHAR, product_name VARCHAR,
                    quantity INTEGER, total_amount DOUBLE, status VARCHAR, order_date VARCHAR,
                    payment_method VARCHAR, shipping_method VARCHAR, item_count INTEGER
                );
                INSERT INTO orders VALUES
                ('ORD-00001', 'CUST-001', 'PROD-00001', 'TechPro Wireless Earbuds', 2, 159.98, 'delivered', '2026-01-15T10:00:00', 'credit_card', 'standard', 2),
                ('ORD-00002', 'CUST-002', 'PROD-00001', 'TechPro Wireless Earbuds', 1, 79.99, 'delivered', '2026-02-10T14:30:00', 'paypal', 'express', 1),
                ('ORD-00003', 'CUST-003', 'PROD-00003', 'UltraClean Air Purifier', 1, 199.99, 'returned', '2026-03-01T09:15:00', 'credit_card', 'standard', 1);
            """)

        # Register customers as view
        if customers_parquet.exists():
            conn.execute(f"CREATE OR REPLACE VIEW customers AS SELECT * FROM read_parquet('{customers_parquet.as_posix()}');")
        elif customers_json.exists():
            conn.execute(f"CREATE OR REPLACE VIEW customers AS SELECT * FROM read_json_auto('{customers_json.as_posix()}');")
        else:
            conn.execute("""
                DROP VIEW IF EXISTS customers;
                DROP TABLE IF EXISTS customers;
                CREATE TABLE customers (
                    customer_id VARCHAR PRIMARY KEY, tier VARCHAR, total_orders INTEGER,
                    total_spent DOUBLE, join_date VARCHAR, preferred_category VARCHAR
                );
                INSERT INTO customers VALUES
                ('CUST-000001', 'Gold', 45, 3420.50, '2022-03-15T10:00:00', 'Electronics'),
                ('CUST-000002', 'Silver', 12, 850.00, '2023-06-20T14:30:00', 'Home & Garden'),
                ('CUST-000003', 'Platinum', 110, 15400.75, '2021-01-10T09:15:00', 'Electronics');
            """)

        # Register Gold Analytical Views
        daily_rev_pq = self.data_dir / "gold" / "daily_revenue.parquet"
        cat_metrics_pq = self.data_dir / "gold" / "category_metrics.parquet"
        clv_pq = self.data_dir / "gold" / "customer_lifetime_value.parquet"

        if daily_rev_pq.exists():
            conn.execute(f"CREATE OR REPLACE VIEW daily_revenue AS SELECT * FROM read_parquet('{daily_rev_pq.as_posix()}');")
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW daily_revenue AS
                SELECT CAST(order_date AS DATE) AS order_date, status, payment_method,
                       COUNT(1) AS total_orders, SUM(quantity) AS total_items,
                       ROUND(SUM(total_amount), 2) AS daily_revenue, ROUND(AVG(total_amount), 2) AS avg_order_value
                FROM orders GROUP BY 1, 2, 3;
            """)

        if cat_metrics_pq.exists():
            conn.execute(f"CREATE OR REPLACE VIEW category_metrics AS SELECT * FROM read_parquet('{cat_metrics_pq.as_posix()}');")
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW category_metrics AS
                SELECT p.category, COUNT(DISTINCT p.product_id) AS product_count,
                       COUNT(o.order_id) AS total_orders, COALESCE(SUM(o.quantity), 0) AS total_units_sold,
                       ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS total_revenue,
                       ROUND(AVG(p.price), 2) AS avg_price, ROUND(AVG(p.rating), 2) AS avg_rating
                FROM products p LEFT JOIN orders o ON p.product_id = o.product_id
                GROUP BY 1;
            """)

        if clv_pq.exists():
            conn.execute(f"CREATE OR REPLACE VIEW customer_lifetime_value AS SELECT * FROM read_parquet('{clv_pq.as_posix()}');")
        else:
            conn.execute("""
                CREATE OR REPLACE VIEW customer_lifetime_value AS
                SELECT c.customer_id, c.tier, c.preferred_category,
                       COUNT(o.order_id) AS order_count,
                       ROUND(COALESCE(SUM(o.total_amount), 0), 2) AS lifetime_spent,
                       ROUND(COALESCE(AVG(o.total_amount), 0), 2) AS avg_order_value,
                       MAX(o.order_date) AS last_order_date
                FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id
                GROUP BY 1, 2, 3;
            """)

        # Register any user-uploaded datasets into DuckDB as views
        try:
            from backend.dataset.manager import get_dataset_manager
            get_dataset_manager().register_all_in_duckdb(conn)
        except Exception as e:
            logger.warning("Could not register user datasets into DuckDB: %s", e)

    def validate_sql(
        self,
        sql: str,
        allowed_tables: set[str] | None = None,
        allowed_columns: set[str] | None = None,
    ) -> str:
        """
        Enforce strict read-only, table-allowlisted, and column-allowlisted execution security via SQL AST.
        """
        target_tables = set(allowed_tables) if allowed_tables is not None else ALLOWED_TABLES
        target_columns = set(allowed_columns) if allowed_columns is not None else ALL_ALLOWED_COLUMNS
        from backend.rag.nodes.sql_validator import SQLASTValidator
        return SQLASTValidator.validate(
            sql=sql,
            allowed_tables=target_tables,
            allowed_columns=target_columns,
            max_limit=50,
        )

    async def __call__(self, state: RAGState) -> dict[str, Any]:
        """Generate and execute SQL query for structured analytical questions."""
        start = time.perf_counter()
        query = state["query"]
        dataset_id = state.get("dataset_id") or "demo_lakehouse"
        raw_sql = ""

        try:
            if dataset_id != "demo_lakehouse":
                from backend.dataset.manager import get_dataset_manager
                mgr = get_dataset_manager()
                schema_prompt, primary_table, allowed_cols = mgr.get_schema_prompt_for_dataset(dataset_id)
                prompt = (
                    f"{schema_prompt}\n\n"
                    f"User Question: {query}\n\n"
                    f"Return JSON format:\n"
                    f'{{\n  "sql": "SELECT ...",\n  "explanation": "..."\n}}'
                )
                target_tables = {primary_table}
                target_columns = allowed_cols
            else:
                prompt = TEXT_TO_SQL_PROMPT.format(query=query)
                target_tables = ALLOWED_TABLES
                target_columns = ALL_ALLOWED_COLUMNS

            response = await self.llm_service.generate(
                prompt=prompt,
                temperature=0.0,
                max_tokens=300,
                json_mode=True,
            )

            parsed = json.loads(response)
            raw_sql = parsed.get("sql", "").strip()
            explanation = parsed.get("explanation", "Lakehouse analytical query")

            if not raw_sql:
                raise SQLValidationError("No SQL statement generated by LLM.")

            safe_sql = self.validate_sql(raw_sql, allowed_tables=target_tables, allowed_columns=target_columns)
            logger.info("Executing safe Text-to-SQL for dataset %s: %s", dataset_id, safe_sql)

            # Execute query in DuckDB with statement timeout and result capping
            conn = self._get_connection()
            # Ensure views are up-to-date
            try:
                from backend.dataset.manager import get_dataset_manager
                get_dataset_manager().register_all_in_duckdb(conn)
            except Exception:
                pass

            # Defense-in-depth: Set 10-second query timeout and cap fetched rows to 50
            cursor = conn.cursor()
            try:
                cursor.execute("SET statement_timeout = '10s';")
            except Exception:
                pass

            df = cursor.execute(safe_sql).fetchdf()
            records = df.to_dict("records")
            latency_ms = (time.perf_counter() - start) * 1000

            # Case A: Analytical query matched records
            if records:
                table_md = self._format_markdown_table(df)
                header_title = f"Structured Query Results ({dataset_id})" if dataset_id != "demo_lakehouse" else "Structured Lakehouse Query Results (DuckDB / Iceberg)"
                context = (
                    f"### {header_title}\n"
                    f"**Query**: `{safe_sql}`\n"
                    f"**Purpose**: {explanation}\n\n"
                    f"{table_md}\n"
                )

                synthetic_doc = {
                    "id": "lakehouse_sql_result",
                    "content": context,
                    "metadata": {
                        "document_type": "lakehouse_analytics",
                        "source_table": "duckdb_iceberg",
                        "sql": safe_sql,
                        "row_count": len(records),
                    },
                    "score": 1.0,
                    "rrf_score": 1.0,
                }

                return {
                    "sql_query": safe_sql,
                    "sql_results": records,
                    "sql_error": None,
                    "sql_context": context,
                    "sql_doc": synthetic_doc,
                    "retrieved_docs": [synthetic_doc],
                    "reranked_docs": [synthetic_doc],
                    "compressed_context": context,
                    "relevance_scores": [1.0],
                    "retrieval_latency_ms": round(latency_ms, 2),
                    "documents_retrieved": 1,
                    "documents_after_rerank": 1,
                    "should_retry": False,
                    "route": state.get("route", "sql"),
                }

            # Case B: Analytical query executed successfully but 0 rows matched criteria
            # Stay on analytical path with explicit zero-row message to prevent qualitative document hallucinations
            empty_context = (
                f"### Structured Lakehouse Query Results (DuckDB / Iceberg)\n"
                f"**Query**: `{safe_sql}`\n"
                f"**Purpose**: {explanation}\n\n"
                f"_No records matched the analytical criteria in the lakehouse dataset._\n"
            )

            synthetic_empty_doc = {
                "id": "lakehouse_sql_empty",
                "content": empty_context,
                "metadata": {
                    "document_type": "lakehouse_analytics",
                    "source_table": "duckdb_iceberg",
                    "sql": safe_sql,
                    "row_count": 0,
                },
                "score": 1.0,
                "rrf_score": 1.0,
            }

            return {
                "sql_query": safe_sql,
                "sql_results": [],
                "sql_error": None,
                "sql_context": empty_context,
                "sql_doc": synthetic_empty_doc,
                "retrieved_docs": [synthetic_empty_doc],
                "reranked_docs": [synthetic_empty_doc],
                "compressed_context": empty_context,
                "relevance_scores": [1.0],
                "retrieval_latency_ms": round(latency_ms, 2),
                "documents_retrieved": 1,
                "documents_after_rerank": 1,
                "should_retry": False,
                "route": state.get("route", "sql"),
            }

        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.warning("Text-to-SQL validation/execution failed: %s", exc)

            # Case C: Analytical query failed execution or validation
            # Return transparent notice to generator so it does not hallucinate numbers
            error_context = (
                f"### Structured Lakehouse Query Notice\n"
                f"The analytical query could not be executed: {exc}.\n"
                f"Please refine your query or specify valid lakehouse attributes (products, orders, customers)."
            )

            synthetic_error_doc = {
                "id": "lakehouse_sql_error",
                "content": error_context,
                "metadata": {
                    "document_type": "lakehouse_analytics",
                    "error": str(exc),
                    "row_count": 0,
                },
                "score": 1.0,
                "rrf_score": 1.0,
            }

            return {
                "sql_query": raw_sql,
                "sql_results": [],
                "sql_error": str(exc),
                "sql_context": error_context,
                "sql_doc": synthetic_error_doc,
                "retrieved_docs": [synthetic_error_doc],
                "reranked_docs": [synthetic_error_doc],
                "compressed_context": error_context,
                "relevance_scores": [1.0],
                "retrieval_latency_ms": round(latency_ms, 2),
                "documents_retrieved": 1,
                "documents_after_rerank": 1,
                "should_retry": False,
                "route": state.get("route", "sql"),
            }

    @staticmethod
    def _format_markdown_table(df: Any) -> str:
        """Format a pandas DataFrame as a clean Markdown table without external dependencies."""
        if df.empty:
            return "_No rows returned._"
        cols = [str(c) for c in df.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = []
        for _, row in df.iterrows():
            formatted_row = " | ".join(str(val).replace("\n", " ").replace("|", "\\|") for val in row.values)
            rows.append(f"| {formatted_row} |")
        return "\n".join([header, sep] + rows)

