"""
Iceberg lakehouse connector via DuckDB.

Connects to Apache Iceberg tables (Bronze/Silver/Gold layers)
and extracts structured data as documents for the RAG pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class IcebergConnector:
    """
    Iceberg / DuckDB data source connector.

    In local mode, uses DuckDB to query Parquet files organized
    in Bronze/Silver/Gold layers. In production, connects to an
    Iceberg catalog via pyiceberg.
    """

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self._conn: Any = None

    def _get_connection(self) -> Any:
        """Get or create a DuckDB connection."""
        if self._conn is None:
            import duckdb

            self._conn = duckdb.connect(database=":memory:")
            logger.info("DuckDB connection established")
        return self._conn

    async def extract_products(
        self, table_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Extract product data from the Gold layer."""
        conn = self._get_connection()
        path = table_path or str(self.data_dir / "gold" / "products.parquet")

        try:
            df = conn.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()

            documents: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                content = (
                    f"Product: {row.get('name', 'Unknown')}\n"
                    f"Category: {row.get('category', 'N/A')}\n"
                    f"Price: ${row.get('price', 0):.2f}\n"
                    f"Description: {row.get('description', 'N/A')}\n"
                    f"Specifications: {row.get('specifications', 'N/A')}\n"
                    f"Rating: {row.get('rating', 'N/A')}/5\n"
                    f"In Stock: {row.get('in_stock', True)}\n"
                    f"SKU: {row.get('sku', 'N/A')}"
                )

                documents.append({
                    "content": content,
                    "metadata": {
                        "document_type": "product",
                        "category": str(row.get("category", "")),
                        "product_id": str(row.get("product_id", "")),
                        "price": float(row.get("price", 0)),
                        "rating": float(row.get("rating", 0)),
                        "source_table": "products",
                    },
                })

            logger.info("Extracted %d products", len(documents))
            return documents

        except Exception as exc:
            logger.error("Product extraction failed: %s", exc)
            return []

    async def extract_orders(
        self, table_path: str | None = None
    ) -> list[dict[str, Any]]:
        """Extract order data from the Silver layer."""
        conn = self._get_connection()
        path = table_path or str(self.data_dir / "silver" / "orders.parquet")

        try:
            df = conn.execute(f"SELECT * FROM read_parquet('{path}')").fetchdf()

            documents: list[dict[str, Any]] = []
            for _, row in df.iterrows():
                content = (
                    f"Order #{row.get('order_id', 'N/A')}\n"
                    f"Customer: {row.get('customer_id', 'N/A')}\n"
                    f"Date: {row.get('order_date', 'N/A')}\n"
                    f"Status: {row.get('status', 'N/A')}\n"
                    f"Total: ${row.get('total_amount', 0):.2f}\n"
                    f"Items: {row.get('item_count', 0)}\n"
                    f"Payment Method: {row.get('payment_method', 'N/A')}\n"
                    f"Shipping Method: {row.get('shipping_method', 'N/A')}"
                )

                documents.append({
                    "content": content,
                    "metadata": {
                        "document_type": "order",
                        "order_id": str(row.get("order_id", "")),
                        "status": str(row.get("status", "")),
                        "total_amount": float(row.get("total_amount", 0)),
                        "source_table": "orders",
                    },
                })

            logger.info("Extracted %d orders", len(documents))
            return documents

        except Exception as exc:
            logger.error("Order extraction failed: %s", exc)
            return []

    async def run_sql(self, query: str) -> list[dict[str, Any]]:
        """Execute arbitrary SQL against DuckDB (for SQL-routed queries)."""
        conn = self._get_connection()
        try:
            result = conn.execute(query).fetchdf()
            return result.to_dict("records")
        except Exception as exc:
            logger.error("SQL execution failed: %s", exc)
            return []

    def close(self) -> None:
        """Close the DuckDB connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
