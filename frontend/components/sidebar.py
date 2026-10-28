"""
Sidebar component for Lakehouse Analyst.
Configures metadata filters, engine toggles, infrastructure health, and session lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

import streamlit as st


def render_sidebar() -> dict[str, Any]:
    """
    Render executive sidebar controls and return active filter & toggle dictionary.
    """
    with st.sidebar:
        # App Title & Brand
        st.markdown(
            """
            <div class="sidebar-brand-box">
                <div class="sidebar-title">RAG LAKEHOUSE ANALYST</div>
                <div class="sidebar-subtitle">PRODUCTION ENTERPRISE EDITION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # New Analysis Thread Button
        if st.button("New Analysis Thread", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.rerun()

        # Infrastructure Health Card
        system_status = st.session_state.get("system_status", "unknown")
        status_dot_cls = "online" if system_status == "healthy" else ("standby" if system_status == "degraded" else "offline")

        st.markdown(
            '<div class="sidebar-section-header">Infrastructure Cluster</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="infra-health-card">
                <div class="infra-node-item">
                    <span><span class="infra-node-dot {status_dot_cls}"></span>FastAPI Gateway</span>
                    <span style="color:var(--text-muted);">{system_status.upper()}</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot online"></span>DuckDB Analytical Engine</span>
                    <span style="color:var(--text-muted);">READY</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot online"></span>PostgreSQL pgvector</span>
                    <span style="color:var(--text-muted);">CONNECTED</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot online"></span>Redis Semantic Cache</span>
                    <span style="color:var(--text-muted);">ACTIVE</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Filters Section
        st.markdown(
            '<div class="sidebar-section-header">Retrieval Filters</div>',
            unsafe_allow_html=True,
        )

        with st.container():
            categories = st.multiselect(
                "Product Categories",
                [
                    "Electronics",
                    "Clothing",
                    "Home & Garden",
                    "Sports & Outdoors",
                    "Books",
                    "Beauty & Personal Care",
                    "Toys & Games",
                    "Automotive",
                    "Food & Grocery",
                    "Health & Wellness",
                ],
                key="filter_categories",
                help="Scope hybrid retrieval to specific commercial categories",
            )

            doc_types = st.multiselect(
                "Document Classes",
                ["product", "order", "review", "ticket", "knowledge_base"],
                key="filter_doc_types",
                help="Filter by source schema type",
            )

            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input(
                    "Date From",
                    value=datetime.now() - timedelta(days=365),
                    key="filter_date_from",
                )
            with col2:
                date_to = st.date_input(
                    "Date To",
                    value=datetime.now(),
                    key="filter_date_to",
                )

            max_results = st.slider(
                "Max Evidence Citations",
                min_value=1,
                max_value=15,
                value=5,
                key="filter_max_results",
            )

        # Inspection Toggles
        st.markdown(
            '<div class="sidebar-section-header">Evidence Inspection</div>',
            unsafe_allow_html=True,
        )

        with st.container():
            show_sql = st.toggle("Structured SQL Evidence", value=True, key="show_sql")
            show_sources = st.toggle("Vector Chunk Citations", value=True, key="show_sources")
            show_metrics = st.toggle("Pipeline Latency & Tokens", value=True, key="show_metrics")

        # Session Telemetry
        conv_id = st.session_state.get("conversation_id", "unassigned")
        short_id = conv_id[:8] if conv_id else "none"
        msg_count = len(st.session_state.get("messages", []))

        st.markdown(
            f"""
            <div style="margin-top:1.5rem; padding-top:0.75rem; border-top:1px solid var(--border-subtle);
                        font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); display:flex; justify-content:space-between;">
                <span>Session: {short_id}</span>
                <span>Queries: {msg_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return {
        "categories": categories or None,
        "document_types": doc_types or None,
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "max_results": max_results,
        "show_sql": show_sql,
        "show_sources": show_sources,
        "show_metrics": show_metrics,
    }
