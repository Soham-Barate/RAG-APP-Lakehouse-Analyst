"""
Evidence & source citation display component for structured SQL tables and vector chunks.
Follows Linear-grade data layout, automatic chart synthesis, and ckw-design algorithm transparency principles.
"""

from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
import streamlit as st


def render_sources(
    sources: list[dict[str, Any]] | None = None,
    sql_query: str | None = None,
    sql_results: list[dict[str, Any]] | None = None,
    sql_error: str | None = None,
) -> None:
    """
    Render both structured lakehouse SQL evidence and unstructured vector citations.
    """
    has_sql = bool(sql_query or (sql_results and len(sql_results) > 0) or sql_error)
    # Only genuine vector chunks (filter out synthetic SQL documents)
    real_sources = [
        s for s in (sources or [])
        if not str(s.get("document_id", "")).startswith("lakehouse_sql_")
        and s.get("document_type") != "lakehouse_analytics"
    ]
    has_vector = bool(len(real_sources) > 0)

    if not has_sql and not has_vector:
        return

    # 1. Structured Lakehouse Evidence (DuckDB / SQL)
    if has_sql:
        _render_sql_evidence(sql_query, sql_results, sql_error)

    # 2. Unstructured Retrieval Citations (pgvector / BM25)
    if has_vector:
        _render_vector_citations(real_sources)


def _render_sql_evidence(
    sql_query: str | None,
    sql_results: list[dict[str, Any]] | None,
    sql_error: str | None = None,
) -> None:
    """Render DuckDB SQL query execution, visualization, and tabular analytical results."""
    row_count = len(sql_results) if sql_results else 0
    status_label = f"{row_count} rows returned" if not sql_error else "execution notice"

    with st.expander(f"STRUCTURED LAKEHOUSE EVIDENCE ({status_label})", expanded=bool(row_count > 0)):
        if sql_error:
            st.markdown(
                f'<div style="padding:0.625rem 0.875rem; background:rgba(239,68,68,0.1); '
                f'border:1px solid rgba(239,68,68,0.25); border-radius:var(--radius-sm); '
                f'font-size:0.75rem; color:#fca5a5; margin-bottom:0.75rem;">'
                f'<strong>Execution Notice:</strong> {html.escape(sql_error)}</div>',
                unsafe_allow_html=True,
            )

        if sql_query:
            st.markdown(
                '<div style="font-size:0.6875rem; font-family:var(--font-mono); '
                'color:var(--text-muted); text-transform:uppercase; margin-bottom:0.25rem;">'
                'Executed Analytical SQL (DuckDB Engine)</div>',
                unsafe_allow_html=True,
            )
            st.code(sql_query, language="sql")

        if sql_results and len(sql_results) > 0:
            df = pd.DataFrame(sql_results)

            # Automatic Chart Synthesis for Analytical Results
            _render_sql_chart(df)

            # Data Table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )

            # CSV Export
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Export {len(df)} Rows as CSV",
                data=csv_bytes,
                file_name="lakehouse_query_result.csv",
                mime="text/csv",
                key=f"dl_sql_{hash(sql_query or '')}_{len(df)}",
            )
        elif sql_query and not sql_error:
            st.markdown(
                '<div style="font-size:0.75rem; color:var(--text-muted); font-style:italic;">'
                'Query executed successfully with zero rows returned.</div>',
                unsafe_allow_html=True,
            )


def _render_sql_chart(df: pd.DataFrame) -> None:
    """Detect if dataframe contains suitable columns for charting."""
    if len(df) < 2 or len(df.columns) < 2:
        return

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]

    if categorical_cols and numeric_cols:
        cat_col = categorical_cols[0]
        num_col = numeric_cols[0]

        # Only chart if categories are bounded (<= 25 items) to avoid unreadable squished bars
        if df[cat_col].nunique() <= 25:
            st.markdown(
                f'<div style="font-size:0.6875rem; font-family:var(--font-mono); color:var(--text-muted); '
                f'text-transform:uppercase; margin:0.75rem 0 0.25rem 0;">'
                f'Result Visualization: {num_col} by {cat_col}</div>',
                unsafe_allow_html=True,
            )
            try:
                chart_df = df[[cat_col, num_col]].dropna().set_index(cat_col)
                st.bar_chart(chart_df, height=220)
            except Exception:
                pass


def _render_vector_citations(sources: list[dict[str, Any]]) -> None:
    """Render vector search chunks with relevance scores, metadata, and mathematical provenance."""
    count = len(sources)

    with st.expander(f"UNSTRUCTURED VECTOR EVIDENCE ({count} chunks cited)", expanded=False):
        for i, source in enumerate(sources):
            _render_single_source(source, i + 1)

        # Mathematical RRF Explainer Note (ckw-design §7: show equation & annotate terms)
        st.markdown(
            """
            <div class="formula-callout">
                <div style="font-weight:600; margin-bottom:0.375rem; color:var(--text-secondary); font-size:0.8125rem;">
                    Hybrid Retrieval & Reciprocal Rank Fusion (RRF) Mathematical Spec
                </div>
                <div style="font-size:0.75rem; color:var(--text-secondary); margin-bottom:0.375rem;">
                    Dense semantic vectors (pgvector cosine) and sparse lexical keywords (PostgreSQL tsvector BM25) are fused without scale bias:
                </div>
                <div style="margin:0.5rem 0; font-family:var(--font-mono); font-size:0.875rem; color:var(--accent-primary); letter-spacing:0.02em;">
                    RRF(d) = &sum;<sub>m &isin; M</sub> [ 1 / (k + rank<sub>m</sub>(d)) ]
                </div>
                <div style="font-size:0.6875rem; color:var(--text-muted); line-height:1.6; border-top:1px solid var(--border-subtle); padding-top:0.375rem;">
                    <div>&bull; <code>M</code> &isin; {Dense pgvector, Sparse BM25}: Independent retrieval model rankers</div>
                    <div>&bull; <code>rank<sub>m</sub>(d)</code>: 1-indexed rank position of document <code>d</code> within retrieval model <code>m</code></div>
                    <div>&bull; <code>k = 60</code>: Smoothing constant dampening outlier impact from single-channel dominance</div>
                    <div>&bull; <strong>Decision Rule:</strong> Top <code>N</code> fused candidate documents are passed to Cross-Encoder reranking.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_single_source(source: dict[str, Any], index: int) -> None:
    """Render an individual source chunk card."""
    doc_type = source.get("document_type", "document")
    relevance = float(source.get("relevance_score", 0.0))
    content = str(source.get("content", ""))
    metadata = source.get("metadata", {})
    doc_id = str(source.get("document_id", f"doc_{index}"))

    # Truncate and clean content for preview (remove markdown headers or fences that break HTML)
    clean_text = re.sub(r"^#+\s*", "", content, flags=re.MULTILINE)
    clean_text = re.sub(r"```[a-z]*", "", clean_text).strip()
    snippet = clean_text[:320] + "..." if len(clean_text) > 320 else clean_text
    safe_snippet = html.escape(snippet).replace("\n", "<br>")

    # Metadata chips
    meta_chips = []
    for key in ["category", "parent_id", "product_id", "order_id", "source_file", "rating", "dataset_id"]:
        val = metadata.get(key)
        if val is not None:
            meta_chips.append(f'<span class="source-meta-chip">{key}: {html.escape(str(val))}</span>')
    chips_html = "".join(meta_chips)
    meta_row_html = f'<div class="source-meta-row">{chips_html}</div>' if chips_html else ""

    rel_pct = max(0, min(100, int(relevance * 100)))

    # Use clean, un-indented HTML to prevent Python-Markdown from interpreting indentations as code
    card_html = (
        f'<div class="source-item-card">'
        f'<div class="source-item-header">'
        f'<div style="display:flex; align-items:center; gap:0.5rem;">'
        f'<span class="source-type-pill">{html.escape(str(doc_type))}</span>'
        f'<span style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted);">'
        f'ID: {html.escape(doc_id)}</span></div>'
        f'<div class="source-relevance-meter">'
        f'<span>{rel_pct}% relevance</span>'
        f'<div class="source-relevance-track">'
        f'<div class="source-relevance-fill" style="width:{rel_pct}%;"></div>'
        f'</div></div></div>'
        f'<div class="source-content-snippet">{safe_snippet}</div>'
        f'{meta_row_html}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)
