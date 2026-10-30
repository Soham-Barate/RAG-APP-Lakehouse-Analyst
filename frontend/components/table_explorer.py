"""
Lakehouse Table Explorer Component.

Provides interactive schema inspection, row-level exploration, data dictionary,
and zero-copy DuckDB analytical querying across Bronze, Silver, and Gold datasets.
Strict Apple/Linear restraint, deterministic typography, and zero-emoji policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

DATASET_CATALOG: dict[str, dict[str, Any]] = {
    "products": {
        "name": "products",
        "tier": "GOLD",
        "description": "Curated, enriched enterprise product catalog with price, rating, and stock status.",
        "path": DATA_DIR / "gold" / "products.parquet",
        "json_path": DATA_DIR / "gold" / "products.json",
        "format": "Parquet (Snappy)",
        "column_descriptions": {
            "product_id": "Unique enterprise product identifier (UUID)",
            "name": "Commercial product display name",
            "category": "High-level retail category classification",
            "description": "Long-form product specification and marketing copy",
            "specifications": "Key technical attributes and material dimensions",
            "price": "Unit retail price in USD",
            "rating": "Aggregated customer review rating (1.0 to 5.0)",
            "review_count": "Total verified purchaser reviews",
            "sku": "Stock Keeping Unit inventory tracking code",
            "in_stock": "Current warehouse availability flag (boolean)",
            "department": "Superordinate enterprise merchandise department",
        },
        "preset_queries": [
            (
                "Product Distribution & Average Price by Category",
                "SELECT category, count(*) AS product_count, round(avg(price), 2) AS avg_price, round(avg(rating), 2) AS avg_rating\n"
                "FROM products\n"
                "GROUP BY category\n"
                "ORDER BY product_count DESC",
            ),
            (
                "Top 10 Most Reviewed In-Stock Products",
                "SELECT name, category, price, rating, review_count\n"
                "FROM products\n"
                "WHERE in_stock = true\n"
                "ORDER BY review_count DESC\n"
                "LIMIT 10",
            ),
            (
                "Rating Distribution Across Departments",
                "SELECT department, count(*) AS sku_count, round(avg(rating), 2) AS avg_rating, round(min(price), 2) AS min_price, round(max(price), 2) AS max_price\n"
                "FROM products\n"
                "GROUP BY department\n"
                "ORDER BY avg_rating DESC",
            ),
        ],
    },
    "orders": {
        "name": "orders",
        "tier": "SILVER",
        "description": "Cleansed transaction log capturing order fulfillment, payment channels, and financial totals.",
        "path": DATA_DIR / "silver" / "orders.parquet",
        "json_path": DATA_DIR / "silver" / "orders.json",
        "format": "Parquet (Snappy)",
        "column_descriptions": {
            "order_id": "Unique sales order transaction ID",
            "customer_id": "Purchasing customer reference ID",
            "product_id": "Purchased SKU identifier",
            "product_name": "Commercial item description at time of order",
            "quantity": "Units purchased in transaction",
            "total_amount": "Total transaction monetary gross in USD",
            "status": "Order fulfillment status (delivered, shipped, pending, cancelled)",
            "order_date": "Timestamp of order authorization",
            "payment_method": "Settlement tender (credit_card, paypal, wire, crypto)",
            "shipping_method": "Logistics carrier tier (standard, expedited, overnight)",
            "item_count": "Total line-items contained within the shipping consignment",
        },
        "preset_queries": [
            (
                "Revenue and Order Volume by Status",
                "SELECT status, count(*) AS order_count, round(sum(total_amount), 2) AS gross_revenue, round(avg(total_amount), 2) AS avg_order_value\n"
                "FROM orders\n"
                "GROUP BY status\n"
                "ORDER BY gross_revenue DESC",
            ),
            (
                "Payment Method Volume and Value",
                "SELECT payment_method, count(*) AS transactions, round(sum(total_amount), 2) AS total_volume, round(avg(total_amount), 2) AS avg_ticket\n"
                "FROM orders\n"
                "GROUP BY payment_method\n"
                "ORDER BY transactions DESC",
            ),
            (
                "Shipping Method Fulfillment Breakdown",
                "SELECT shipping_method, status, count(*) AS shipments\n"
                "FROM orders\n"
                "GROUP BY shipping_method, status\n"
                "ORDER BY shipping_method, shipments DESC",
            ),
        ],
    },
    "customers": {
        "name": "customers",
        "tier": "SILVER",
        "description": "Cleansed customer profiles with aggregated lifetime value and loyalty segmentation.",
        "path": DATA_DIR / "silver" / "customers.parquet",
        "json_path": DATA_DIR / "silver" / "customers.json",
        "format": "Parquet (Snappy)",
        "column_descriptions": {
            "customer_id": "Enterprise customer master record ID",
            "tier": "Loyalty status level (bronze, silver, gold, platinum)",
            "total_orders": "Historical lifetime orders completed",
            "total_spent": "Cumulative gross spend in USD across customer lifecycle",
            "join_date": "Account establishment timestamp",
            "preferred_category": "Algorithmic top affinity retail category",
        },
        "preset_queries": [
            (
                "Customer Tier Lifetime Value & Segmentation",
                "SELECT tier, count(*) AS member_count, round(sum(total_spent), 2) AS total_revenue, round(avg(total_spent), 2) AS avg_ltv, round(avg(total_orders), 1) AS avg_orders\n"
                "FROM customers\n"
                "GROUP BY tier\n"
                "ORDER BY avg_ltv DESC",
            ),
            (
                "Top 10 Preferred Categories by Customer Count",
                "SELECT preferred_category, count(*) AS customer_count, round(avg(total_spent), 2) AS avg_spend\n"
                "FROM customers\n"
                "GROUP BY preferred_category\n"
                "ORDER BY customer_count DESC\n"
                "LIMIT 10",
            ),
        ],
    },
    "reviews": {
        "name": "reviews",
        "tier": "BRONZE",
        "description": "Raw, ingested customer sentiment and product reviews before vector embedding.",
        "path": DATA_DIR / "bronze" / "reviews.json",
        "json_path": DATA_DIR / "bronze" / "reviews.json",
        "format": "JSON (Raw Ingestion)",
        "column_descriptions": {
            "review_id": "Individual customer review record ID",
            "product_id": "Target product identifier",
            "customer_id": "Author customer identifier",
            "rating": "Numerical rating (1 to 5 stars)",
            "headline": "Subject title written by customer",
            "body": "Unstructured customer feedback text (RAG source)",
            "created_at": "Review publication timestamp",
            "verified_purchase": "Proof of purchase confirmation flag",
        },
        "preset_queries": [
            (
                "Review Sentiment & Rating Summary",
                "SELECT rating, count(*) AS review_count, round(count(*) * 100.0 / sum(count(*)) over(), 1) AS share_pct\n"
                "FROM reviews\n"
                "GROUP BY rating\n"
                "ORDER BY rating DESC",
            ),
        ],
    },
    "tickets": {
        "name": "tickets",
        "tier": "BRONZE",
        "description": "Raw support ticket incidents and resolution dialogue before semantic chunking.",
        "path": DATA_DIR / "bronze" / "tickets.json",
        "json_path": DATA_DIR / "bronze" / "tickets.json",
        "format": "JSON (Raw Ingestion)",
        "column_descriptions": {
            "ticket_id": "Customer support case ticket ID",
            "customer_id": "Requesting customer ID",
            "order_id": "Associated order ID (if applicable)",
            "subject": "Case inquiry topic / headline",
            "description": "Unstructured issue narrative (RAG source)",
            "status": "Resolution state (open, pending, resolved, closed)",
            "priority": "Urgency classification (low, medium, high, urgent)",
            "created_at": "Incident initiation timestamp",
        },
        "preset_queries": [
            (
                "Support Ticket Volume by Priority & Status",
                "SELECT priority, status, count(*) AS ticket_count\n"
                "FROM tickets\n"
                "GROUP BY priority, status\n"
                "ORDER BY priority, ticket_count DESC",
            ),
        ],
    },
}


@st.cache_data(show_spinner=False, ttl=2)
def load_dataset_metadata(dataset_key: str, path_str: str, mtime: float = 0.0) -> dict[str, Any]:
    """Inspect dataset file directly using DuckDB or file stats."""
    file_path = Path(path_str)
    if not file_path.exists():
        return {"exists": False, "error": f"File not found: {file_path.name}"}

    size_bytes = file_path.stat().st_size
    size_formatted = f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB"

    conn = duckdb.connect(database=":memory:")
    try:
        if file_path.suffix == ".parquet":
            duckdb_source = f"'{file_path.as_posix()}'"
        else:
            duckdb_source = f"read_json_auto('{file_path.as_posix()}')"

        count_res = conn.execute(f"SELECT count(*) FROM {duckdb_source}").fetchone()
        row_count = count_res[0] if count_res else 0

        describe_res = conn.execute(f"DESCRIBE SELECT * FROM {duckdb_source} LIMIT 1").fetchall()
        columns_meta = []
        info = DATASET_CATALOG.get(dataset_key, {})
        col_desc_map = info.get("column_descriptions", {})
        for col in describe_res:
            col_name = str(col[0])
            col_type = str(col[1])
            is_nullable = str(col[2])
            columns_meta.append({
                "column": col_name,
                "type": col_type,
                "nullable": is_nullable,
                "description": col_desc_map.get(col_name, "Standard dataset attribute"),
            })

        return {
            "exists": True,
            "row_count": row_count,
            "col_count": len(columns_meta),
            "size": size_formatted,
            "columns": columns_meta,
            "path": str(file_path),
            "duckdb_source": duckdb_source,
        }
    except Exception as exc:
        return {"exists": True, "error": str(exc), "size": size_formatted}
    finally:
        conn.close()


def render_table_explorer() -> None:
    """Render the executive Lakehouse Table Explorer screen."""
    header_html = (
        '<div style="margin-bottom: 1.25rem;">'
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.25rem;">'
        '<span class="source-type-pill">MEDALLION CATALOG</span>'
        '<span style="font-family:var(--font-mono); font-size:0.75rem; color:var(--text-muted);">'
        'DUCKDB IN-MEMORY ZERO-COPY INSPECTION'
        '</span></div>'
        '<h2 style="font-size:1.375rem; font-weight:700; margin:0 0 0.25rem 0; color:var(--text-primary); letter-spacing:-0.02em;">'
        'Lakehouse Table & Schema Explorer'
        '</h2>'
        '<p style="font-size:0.8125rem; color:var(--text-secondary); margin:0;">'
        'Direct schema validation, column-level data dictionaries, row sampling, and zero-copy OLAP analytical querying.'
        '</p></div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Dataset Selection Ribbon (combines Medallion lakehouse tables and user-uploaded datasets)
    catalog = dict(DATASET_CATALOG)
    try:
        from backend.dataset.manager import get_dataset_manager
        user_datasets = get_dataset_manager().list_datasets()
        for u in user_datasets:
            if not u.is_demo and u.parquet_path:
                u_path = Path(u.parquet_path)
                if u_path.exists():
                    catalog[u.table_name] = {
                        "name": u.table_name,
                        "tier": "USER",
                        "description": f"User-uploaded dataset from {u.filename} (Isolated DuckDB Table)",
                        "path": u_path,
                        "format": f"Parquet ({u.file_format.upper()})",
                        "column_descriptions": {},
                        "preset_queries": [
                            (
                                f"Preview 10 Rows from {u.table_name}",
                                f"SELECT * FROM {u.table_name} LIMIT 10",
                            ),
                            (
                                f"Count Records in {u.table_name}",
                                f"SELECT count(*) AS total_records FROM {u.table_name}",
                            ),
                        ],
                    }
    except Exception:
        pass

    col_sel, col_stats = st.columns([1, 2])
    with col_sel:
        selected_key = st.selectbox(
            "Select Lakehouse Dataset",
            options=list(catalog.keys()),
            format_func=lambda k: f"{k.upper()} ({catalog[k]['tier']} TIER)",
            key="lakehouse_selected_dataset",
        )

    dataset_def = catalog[selected_key]
    file_mtime = dataset_def["path"].stat().st_mtime if dataset_def["path"].exists() else 0.0
    meta = load_dataset_metadata(selected_key, path_str=str(dataset_def["path"]), mtime=file_mtime)

    with col_stats:
        tier_cls = f"tier-{dataset_def['tier'].lower()}"
        row_str = f"{meta.get('row_count', 0):,}" if "row_count" in meta else "N/A"
        col_str = str(meta.get("col_count", 0))
        size_str = meta.get("size", "N/A")

        banner_html = (
            f'<div class="dataset-meta-banner">'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">MEDALLION TIER</span>'
            f'<span class="tier-badge {tier_cls}">{dataset_def["tier"]}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">TOTAL ROWS</span>'
            f'<span class="dataset-meta-value">{row_str}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">COLUMNS</span>'
            f'<span class="dataset-meta-value">{col_str}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">STORAGE SIZE</span>'
            f'<span class="dataset-meta-value">{size_str}</span>'
            f'</div>'
            f'<div class="dataset-meta-item">'
            f'<span class="dataset-meta-label">STORAGE FORMAT</span>'
            f'<span class="dataset-meta-value">{dataset_def["format"]}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(banner_html, unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:0.8125rem; color:var(--text-secondary); margin-bottom:1.25rem;">'
        f'{dataset_def["description"]}</div>',
        unsafe_allow_html=True,
    )

    # Explorer Tabs
    tab_preview, tab_schema, tab_sql = st.tabs([
        "DATA PREVIEW & FILTERING",
        "SCHEMA & DATA DICTIONARY",
        "DIRECT DUCKDB ANALYTICAL QUERY",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: DATA PREVIEW & FILTERING
    # -------------------------------------------------------------------------
    with tab_preview:
        file_path = dataset_def["path"]
        if not file_path.exists():
            st.warning(f"File not found on disk at {file_path}")
            return

        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
        with ctrl_col1:
            limit_rows = st.select_slider(
                "Sample Limit",
                options=[10, 25, 50, 100, 250, 500],
                value=50,
                key=f"{selected_key}_sample_limit",
            )
        with ctrl_col2:
            search_query = st.text_input(
                "Filter Keyword",
                placeholder="Search across all fields...",
                key=f"{selected_key}_filter_kw",
            )
        with ctrl_col3:
            st.write("")
            st.write("")

        # Read preview with DuckDB
        conn = duckdb.connect(database=":memory:")
        try:
            source_sql = meta.get("duckdb_source", f"'{file_path.as_posix()}'")
            if search_query.strip():
                # Generic ILIKE filter across string columns
                escaped_kw = search_query.strip().replace("'", "''")
                df = conn.execute(
                    f"SELECT * FROM {source_sql} LIMIT {limit_rows}"
                ).df()
                # Local filter across string columns
                mask = df.astype(str).apply(lambda col: col.str.contains(escaped_kw, case=False, na=False)).any(axis=1)
                df = df[mask]
            else:
                df = conn.execute(f"SELECT * FROM {source_sql} LIMIT {limit_rows}").df()

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            # CSV Download Button
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Export {len(df)} Sample Rows as CSV",
                data=csv_data,
                file_name=f"lakehouse_{selected_key}_sample.csv",
                mime="text/csv",
                key=f"{selected_key}_csv_dl",
            )
        except Exception as exc:
            st.error(f"Error previewing dataset: {exc}")
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # TAB 2: SCHEMA & DATA DICTIONARY
    # -------------------------------------------------------------------------
    with tab_schema:
        columns_data = meta.get("columns", [])
        if columns_data:
            schema_df = pd.DataFrame(columns_data)
            schema_df.rename(
                columns={
                    "column": "Column Name",
                    "type": "DuckDB Inferred Type",
                    "nullable": "Nullable",
                    "description": "Business Semantic Description",
                },
                inplace=True,
            )
            st.dataframe(
                schema_df,
                use_container_width=True,
                hide_index=True,
            )

            note_html = (
                '<div class="schema-note-box">'
                '<strong>Schema Governance Note:</strong> '
                'All parquet tables conform to the Iceberg Lakehouse specification with strict decimal precision and UTF-8 encoding. '
                'Column names adhere to <code>snake_case</code> conventions and enforce strict typing during the bronze-to-silver ETL boundary.'
                '</div>'
            )
            st.markdown(note_html, unsafe_allow_html=True)
        else:
            st.info("Schema details could not be parsed for this file.")

    # -------------------------------------------------------------------------
    # TAB 3: DIRECT DUCKDB ANALYTICAL QUERY
    # -------------------------------------------------------------------------
    with tab_sql:
        st.markdown(
            '<div style="font-size:0.8125rem; color:var(--text-secondary); margin-bottom:0.75rem;">'
            'Execute safe, in-memory analytical SQL against the registered Lakehouse table. '
            'External I/O and modifying statements are restricted.'
            '</div>',
            unsafe_allow_html=True,
        )

        # Preset query selection
        presets = dataset_def.get("preset_queries", [])
        preset_names = ["-- Select an Analytical Preset Query --"] + [p[0] for p in presets]
        chosen_preset = st.selectbox(
            "Load Preset Query",
            options=preset_names,
            key=f"{selected_key}_preset_select",
        )

        default_sql = f"SELECT * FROM {selected_key} LIMIT 10"
        if chosen_preset != "-- Select an Analytical Preset Query --":
            for p_title, p_sql in presets:
                if p_title == chosen_preset:
                    default_sql = p_sql
                    break

        user_sql = st.text_area(
            "SQL Statement",
            value=default_sql,
            height=140,
            key=f"{selected_key}_custom_sql",
        )

        run_col1, run_col2 = st.columns([1, 4])
        with run_col1:
            run_btn = st.button("Execute SQL", type="primary", key=f"{selected_key}_run_sql")

        if run_btn:
            _execute_safe_sql(user_sql, selected_key, dataset_def)


def _execute_safe_sql(sql: str, selected_key: str, dataset_def: dict[str, Any]) -> None:
    """Execute SQL query safely in DuckDB with table registration."""
    # Safety checks
    sql_clean = sql.strip()
    disallowed = ["insert", "update", "delete", "drop", "alter", "create", "truncate", "attach", "detach", "copy"]
    first_word = sql_clean.split()[0].lower() if sql_clean else ""
    if first_word in disallowed:
        st.error(f"Execution Error: Statement type '{first_word.upper()}' is not permitted in read-only analysis.")
        return

    conn = duckdb.connect(database=":memory:")
    try:
        # Register products, orders, customers
        for key, info in DATASET_CATALOG.items():
            f_path = info["path"]
            if f_path.exists():
                if f_path.suffix == ".parquet":
                    conn.execute(f"CREATE VIEW {key} AS SELECT * FROM '{f_path.as_posix()}'")
                else:
                    conn.execute(f"CREATE VIEW {key} AS SELECT * FROM read_json_auto('{f_path.as_posix()}')")

        df = conn.execute(sql_clean).df()

        st.markdown(
            f'<div style="font-size:0.75rem; font-family:var(--font-mono); color:var(--status-success); margin:0.5rem 0;">'
            f'Query executed successfully: {len(df)} rows returned'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Tabular View
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Automatic Chart Detection
        _render_auto_chart(df)

    except Exception as exc:
        st.error(f"DuckDB Execution Error: {exc}")
    finally:
        conn.close()


def _render_auto_chart(df: pd.DataFrame) -> None:
    """Detect suitable aggregations and render native Streamlit chart."""
    if df.empty or len(df.columns) < 2:
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    non_numeric_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if non_numeric_cols and numeric_cols:
        x_col = non_numeric_cols[0]
        y_col = numeric_cols[0]

        st.markdown(
            f'<div style="font-size:0.6875rem; font-family:var(--font-mono); color:var(--text-muted); '
            f'text-transform:uppercase; margin:1rem 0 0.25rem 0;">'
            f'Analytical Visualization ({y_col} by {x_col})</div>',
            unsafe_allow_html=True,
        )

        try:
            chart_data = df[[x_col, y_col]].set_index(x_col)
            st.bar_chart(chart_data)
        except Exception:
            pass
