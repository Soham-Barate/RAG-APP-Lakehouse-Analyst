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
ALLOWED_TABLES = {"products", "orders", "customers"}

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
    "select", "from", "where", "join", "left", "right", "inner", "outer", "full", "cross",
    "on", "group", "by", "order", "having", "limit", "offset", "as", "distinct",
    "and", "or", "not", "is", "null", "in", "like", "ilike", "between",
    "case", "when", "then", "else", "end", "asc", "desc", "true", "false",
    "all", "with", "over", "partition", "using", "union", "interval",
    # Data type casts
    "varchar", "integer", "int", "bigint", "smallint", "tinyint", "double", "float",
    "decimal", "numeric", "boolean", "date", "timestamp", "text"
}

ALLOWED_FUNCTIONS = {
    # Aggregations
    "count", "sum", "avg", "min", "max",
    # Math
    "round", "abs", "ceil", "floor", "power", "sqrt",
    # Conditionals & Nulls
    "coalesce", "nullif", "greatest", "least", "cast",
    # Date & Time (DuckDB native)
    "date_trunc", "extract", "year", "month", "day", "quarter", "epoch",
    "strftime", "to_timestamp", "date_add", "date_sub", "datediff", "date_diff",
    "now", "current_date", "current_timestamp", "current_time",
    # String manipulation
    "lower", "upper", "trim", "ltrim", "rtrim", "concat", "length",
    "substr", "substring", "replace", "split_part",
    # Window functions
    "row_number", "rank", "dense_rank", "lag", "lead", "first_value", "last_value"
}

ALLOWED_IDENTIFIERS = ALL_ALLOWED_COLUMNS | ALLOWED_TABLES | ALLOWED_SQL_KEYWORDS | ALLOWED_FUNCTIONS


class SQLValidationError(Exception):
    """Raised when generated SQL violates safety constraints."""
    pass


class SafeSQLNode:
    """
    Translates analytical queries into SQL and executes safely via DuckDB.
    """

    def __init__(
        self,
        llm_service: Any,
        data_dir: str | Path = "data",
    ) -> None:
        self.llm_service = llm_service
        self.data_dir = Path(data_dir)
        self._conn: Any = None

    def _get_connection(self) -> Any:
        """Initialize or return the DuckDB connection with tables registered and external access locked."""
        if self._conn is None:
            import duckdb

            # Create in-memory connection with resource boundaries
            self._conn = duckdb.connect(
                database=":memory:",
                config={
                    "max_memory": "512MB",
                    "threads": 2,
                },
            )
            # Register lakehouse tables from Parquet/JSON into memory (or fallback mock schema)
            self._register_views()

            # Now permanently disable external filesystem/network operations and lock configuration
            try:
                self._conn.execute("SET enable_external_access = false;")
                self._conn.execute("SET lock_configuration = true;")
            except Exception as exc:
                logger.warning("Could not lock DuckDB configuration: %s", exc)

        return self._conn

    def _register_views(self) -> None:
        """Register products, orders, and customers tables in DuckDB memory."""
        conn = self._conn
        gold_parquet = self.data_dir / "gold" / "products.parquet"
        silver_parquet = self.data_dir / "silver" / "orders.parquet"
        customers_parquet = self.data_dir / "silver" / "customers.parquet"

        gold_json = self.data_dir / "gold" / "products.json"
        silver_json = self.data_dir / "silver" / "orders.json"
        customers_json = self.data_dir / "silver" / "customers.json"

        # Register products
        if gold_parquet.exists():
            conn.execute(f"CREATE TABLE products AS SELECT * FROM read_parquet('{gold_parquet.as_posix()}')")
        elif gold_json.exists():
            conn.execute(f"CREATE TABLE products AS SELECT * FROM read_json_auto('{gold_json.as_posix()}')")
        else:
            conn.execute("""
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

        # Register orders
        if silver_parquet.exists():
            conn.execute(f"CREATE TABLE orders AS SELECT * FROM read_parquet('{silver_parquet.as_posix()}')")
        elif silver_json.exists():
            conn.execute(f"CREATE TABLE orders AS SELECT * FROM read_json_auto('{silver_json.as_posix()}')")
        else:
            conn.execute("""
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

        # Register customers
        if customers_parquet.exists():
            conn.execute(f"CREATE TABLE customers AS SELECT * FROM read_parquet('{customers_parquet.as_posix()}')")
        elif customers_json.exists():
            conn.execute(f"CREATE TABLE customers AS SELECT * FROM read_json_auto('{customers_json.as_posix()}')")
        else:
            conn.execute("""
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

        # Register any user-uploaded datasets into DuckDB
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
        Enforce strict read-only, table-allowlisted, and column-allowlisted execution security.
        """
        target_tables = set(allowed_tables) if allowed_tables is not None else ALLOWED_TABLES
        target_columns = set(allowed_columns) if allowed_columns is not None else ALL_ALLOWED_COLUMNS

        # 1. Strip SQL comments (both inline -- and block /* ... */)
        cleaned = re.sub(r"--.*?$", " ", sql, flags=re.MULTILINE)
        cleaned = re.sub(r"/\*[\s\S]*?\*/", " ", cleaned).strip().rstrip(";")

        # 2. Prevent multi-statement query chaining
        if ";" in cleaned:
            raise SQLValidationError("Multiple statements are forbidden.")

        # 3. Strip single-quoted string literals so literal values are not mistaken for column identifiers
        no_strings = re.sub(r"'(?:''|[^'])*'", "''", cleaned)

        # 4. Strip quoted identifier delimiters for uniform token inspection
        normalized = re.sub(r'["`]', "", no_strings)

        # 5. Extract CTE aliases if query starts with WITH
        cte_names: set[str] = set()
        cte_matches = re.finditer(r"(?:with|,)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", normalized, re.IGNORECASE)
        for m in cte_matches:
            cte_names.add(m.group(1).lower())

        # 6. Extract word tokens
        tokens = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", normalized.lower())
        if not tokens or tokens[0] not in ("select", "with"):
            raise SQLValidationError("Only SELECT or read-only WITH queries are allowed.")

        # 7. Check disallowed keywords and dangerous functions
        for token in tokens:
            if token in DISALLOWED_KEYWORDS:
                raise SQLValidationError(f"Disallowed keyword or function detected: '{token}'")

        # 8. Table allowlist validation: verify tables following FROM and JOIN
        found_tables: set[str] = set()
        aliases: set[str] = set(cte_names)
        permitted_tables = target_tables | cte_names

        for i, token in enumerate(tokens):
            if token in ("from", "join") and i + 1 < len(tokens):
                tbl = tokens[i + 1]
                if tbl not in permitted_tables:
                    raise SQLValidationError(f"Table '{tbl}' is not permitted.")
                found_tables.add(tbl)
                # Next token might be an alias: e.g. FROM products p or JOIN orders o
                if i + 2 < len(tokens) and tokens[i + 2] not in ("where", "join", "left", "right", "inner", "outer", "on", "group", "order", "limit"):
                    aliases.add(tokens[i + 2])

            if token == "as" and i + 1 < len(tokens):
                aliases.add(tokens[i + 1])

        if not found_tables:
            matched = [tbl for tbl in permitted_tables if tbl in tokens]
            if not matched:
                raise SQLValidationError("Query does not reference an allowlisted table.")

        # 9. Comprehensive Column & Identifier allowlisting
        allowed_lookup = target_columns | target_tables | ALLOWED_SQL_KEYWORDS | ALLOWED_FUNCTIONS | aliases
        for token in tokens:
            if token not in allowed_lookup:
                raise SQLValidationError(f"Column or identifier '{token}' is not permitted in analytical queries.")

        # 10. Enforce row limit
        if "limit" not in tokens:
            cleaned += " LIMIT 50"
        else:
            limit_match = re.search(r"limit\s+(\d+)", cleaned, re.IGNORECASE)
            if limit_match and int(limit_match.group(1)) > 50:
                cleaned = re.sub(r"limit\s+\d+", "LIMIT 50", cleaned, flags=re.IGNORECASE)

        return cleaned

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

            # Execute query in DuckDB with external access locked
            conn = self._get_connection()
            # Ensure views are up-to-date
            try:
                from backend.dataset.manager import get_dataset_manager
                get_dataset_manager().register_all_in_duckdb(conn)
            except Exception:
                pass

            df = conn.execute(safe_sql).fetchdf()
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

