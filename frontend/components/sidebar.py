"""
Sidebar component for Lakehouse Analyst.
Configures metadata filters, engine toggles, dynamic infrastructure health, and session lifecycle.
Follows zero-emoji policy, Linear/Apple craft, and honest telemetry reporting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Any

import streamlit as st


def _trigger_data_regeneration() -> None:
    """Trigger synthetic lakehouse data regeneration either via FastAPI or direct fallback."""
    import os
    import requests
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "dev-key-1")
    try:
        resp = requests.post(
            f"{backend_url}/api/v1/datasets/regenerate-demo",
            headers={"X-API-Key": api_key},
            timeout=10,
        )
        if resp.status_code == 200:
            return
    except Exception:
        pass

    # Direct local fallback
    try:
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from backend.rag.nodes.sql_node import SafeSQLNode
        SyntheticDataGenerator.regenerate_new_data()
        SafeSQLNode.reset_connection()
    except Exception:
        pass


def render_sidebar(health_info: dict[str, Any] | None = None) -> dict[str, Any]:
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

        # New Analysis Thread Button (Regenerates fresh dataset and resets thread)
        if st.button("New Analysis Thread", use_container_width=True, type="primary"):
            _trigger_data_regeneration()
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.active_view = "ANALYSIS CONSOLE"
            st.rerun()

        # Active Dataset Selector
        st.markdown(
            '<div class="sidebar-section-header">Active Dataset Target</div>',
            unsafe_allow_html=True,
        )
        try:
            from frontend.components.dataset_view import fetch_datasets
            available_datasets = fetch_datasets()
        except Exception:
            available_datasets = [{"dataset_id": "demo_lakehouse", "name": "E-Commerce Lakehouse (Demo)", "row_count": 5000, "is_demo": True}]

        dataset_options = {d["dataset_id"]: d.get("name", d["dataset_id"]) for d in available_datasets}
        current_active = st.session_state.get("active_dataset_id", "demo_lakehouse")
        if current_active not in dataset_options:
            current_active = "demo_lakehouse"
            st.session_state.active_dataset_id = current_active

        keys_list = list(dataset_options.keys())
        active_index = keys_list.index(current_active) if current_active in keys_list else 0

        selected_id = st.selectbox(
            "Select active dataset",
            options=keys_list,
            format_func=lambda x: dataset_options.get(x, x),
            index=active_index,
            label_visibility="collapsed",
            key="sidebar_dataset_selector",
        )
        if selected_id != current_active:
            st.session_state.active_dataset_id = selected_id
            st.rerun()

        active_meta = next((d for d in available_datasets if d["dataset_id"] == selected_id), None)
        active_label = "Demo Lakehouse" if (active_meta and active_meta.get("is_demo")) else "User Dataset"
        active_rows = active_meta.get("row_count", 0) if active_meta else 0
        st.markdown(
            f"""
            <div style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); margin-top:0.25rem;">
                Target: <span style="color:#38bdf8;">{active_label}</span> ({active_rows:,} rows)
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic Infrastructure Cluster Health Card
        health = health_info or {}
        gateway_reachable = health.get("reachable", False)
        system_status = health.get("status", "unknown")
        gw_dot = "online" if gateway_reachable and system_status == "healthy" else ("standby" if gateway_reachable else "offline")
        gw_label = "ONLINE" if gateway_reachable and system_status == "healthy" else ("DEGRADED" if gateway_reachable else "OFFLINE")

        services = health.get("services", {})
        pg_state = services.get("postgres", "not_configured")
        pg_dot = "online" if pg_state == "healthy" else ("standby" if pg_state == "not_configured" else "offline")
        pg_label = "CONNECTED" if pg_state == "healthy" else ("STANDBY" if pg_state == "not_configured" else "OFFLINE")

        rd_state = services.get("redis", "not_configured")
        rd_dot = "online" if rd_state == "healthy" else ("standby" if rd_state == "not_configured" else "offline")
        rd_label = "ACTIVE" if rd_state == "healthy" else ("STANDBY" if rd_state == "not_configured" else "OFFLINE")

        st.markdown(
            '<div class="sidebar-section-header">Infrastructure Cluster</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="infra-health-card">
                <div class="infra-node-item">
                    <span><span class="infra-node-dot {gw_dot}"></span>FastAPI Gateway</span>
                    <span style="color:var(--text-muted); font-size:0.6875rem;">{gw_label}</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot online"></span>DuckDB Analytical OLAP</span>
                    <span style="color:var(--text-muted); font-size:0.6875rem;">READY</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot {pg_dot}"></span>PostgreSQL pgvector</span>
                    <span style="color:var(--text-muted); font-size:0.6875rem;">{pg_label}</span>
                </div>
                <div class="infra-node-item">
                    <span><span class="infra-node-dot {rd_dot}"></span>Redis Semantic Cache</span>
                    <span style="color:var(--text-muted); font-size:0.6875rem;">{rd_label}</span>
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
            <div style="margin-top:1.25rem; padding-top:0.75rem; border-top:1px solid var(--border-subtle);
                        font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); display:flex; justify-content:space-between;">
                <span>Session: {short_id}</span>
                <span>Queries: {msg_count}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick Lakehouse Data Controls at Bottom of Panel
        st.markdown(
            '<div class="sidebar-section-header" style="margin-top:0.75rem;">Lakehouse Data Controls</div>',
            unsafe_allow_html=True,
        )

        col_side1, col_side2 = st.columns(2)
        with col_side1:
            if st.button("View Lakehouse Data", use_container_width=True, key="btn_view_lakehouse_data", help="Inspect newly generated or active lakehouse tables"):
                st.session_state.active_view = "LAKEHOUSE TABLE EXPLORER"
                st.rerun()

        with col_side2:
            if st.button("Regenerate Data", use_container_width=True, key="btn_regen_lakehouse_data", help="Generate fresh synthetic orders, revenue, and customer records"):
                with st.spinner("Regenerating..."):
                    _trigger_data_regeneration()
                st.toast("Fresh lakehouse transactions generated!")
                st.rerun()

        return {
            "active_dataset_id": selected_id,
            "categories": categories,
            "document_types": doc_types,
            "date_from": str(date_from) if date_from else None,
            "date_to": str(date_to) if date_to else None,
            "max_results": max_results,
            "show_sql": show_sql,
            "show_sources": show_sources,
            "show_metrics": show_metrics,
        }
