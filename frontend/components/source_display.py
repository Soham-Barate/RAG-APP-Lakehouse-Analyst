"""
Evidence & source citation display component for structured SQL tables and vector chunks.
Follows Linear-grade data layout and ckw-design algorithm transparency principles.
"""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st


def render_sources(
    sources: list[dict[str, Any]] | None = None,
    sql_query: str | None = None,
    sql_results: list[dict[str, Any]] | None = None,
) -> None:
    """
    Render both structured lakehouse SQL evidence and unstructured vector citations.
    """
    has_sql = bool(sql_query or (sql_results and len(sql_results) > 0))
    has_vector = bool(sources and len(sources) > 0)

    if not has_sql and not has_vector:
        return

    # 1. Structured Lakehouse Evidence (DuckDB / SQL)
    if has_sql:
        _render_sql_evidence(sql_query, sql_results)

    # 2. Unstructured Retrieval Citations (pgvector / BM25)
    if has_vector:
        _render_vector_citations(sources or [])


def _render_sql_evidence(
    sql_query: str | None,
    sql_results: list[dict[str, Any]] | None,
) -> None:
    """Render DuckDB SQL query execution and tabular analytical results."""
    row_count = len(sql_results) if sql_results else 0

    with st.expander(f"STRUCTURED LAKEHOUSE EVIDENCE ({row_count} rows returned)", expanded=False):
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
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
            )
        elif sql_query:
            st.markdown(
                '<div style="font-size:0.75rem; color:var(--text-muted); font-style:italic;">'
                'Query executed successfully with zero rows returned.</div>',
                unsafe_allow_html=True,
            )


def _render_vector_citations(sources: list[dict[str, Any]]) -> None:
    """Render vector search chunks with relevance scores, metadata, and mathematical provenance."""
    count = len(sources)

    with st.expander(f"UNSTRUCTURED VECTOR EVIDENCE ({count} chunks cited)", expanded=False):
        for i, source in enumerate(sources):
            _render_single_source(source, i + 1)

        # Mathematical RRF Explainer Note (ckw-design §7)
        st.markdown(
            """
            <div class="formula-callout">
                <div style="font-weight:600; margin-bottom:0.25rem; color:var(--text-secondary);">
                    Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
                </div>
                <div>
                    Document relevance combines pgvector cosine similarity and PostgreSQL tsvector BM25 keyword rankings via:
                </div>
                <div style="margin:0.375rem 0; font-family:var(--font-mono); color:var(--accent-primary);">
                    RRF(d) = &sum; [ 1 / (60 + rank_m(d)) ]
                </div>
                <div>
                    where <code>m &isin; {Dense, Sparse}</code> and constant <code>k = 60</code> dampens outlier bias before Cross-Encoder reranking.
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

    # Truncate content for preview
    snippet = content[:320] + "..." if len(content) > 320 else content
    safe_snippet = html.escape(snippet)

    # Metadata chips
    meta_chips = []
    for key in ["category", "parent_id", "product_id", "order_id", "source_file"]:
        val = metadata.get(key)
        if val:
            meta_chips.append(f'<span class="source-meta-chip">{key}: {html.escape(str(val))}</span>')
    chips_html = "".join(meta_chips)

    rel_pct = max(0, min(100, int(relevance * 100)))

    st.markdown(
        f"""
        <div class="source-item-card">
            <div class="source-item-header">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span class="source-type-pill">{doc_type}</span>
                    <span style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted);">
                        ID: {html.escape(doc_id)}
                    </span>
                </div>
                <div class="source-relevance-meter">
                    <span>{rel_pct}% relevance</span>
                    <div class="source-relevance-track">
                        <div class="source-relevance-fill" style="width:{rel_pct}%;"></div>
                    </div>
                </div>
            </div>
            <div class="source-content-snippet">
                {safe_snippet}
            </div>
            <div class="source-meta-row">
                {chips_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
