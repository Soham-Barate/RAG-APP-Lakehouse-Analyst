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


def _trigger_data_regeneration(scale: str = "quick") -> dict[str, Any] | None:
    """Trigger synthetic lakehouse data regeneration via FastAPI or direct fallback."""
    import os
    import requests
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "dev-key-1")

    # Use async endpoint for medium/large/enterprise to prevent HTTP gateway timeout
    if scale in ("medium", "large", "enterprise"):
        try:
            resp = requests.post(
                f"{backend_url}/api/v1/datasets/regenerate-async?scale={scale}",
                headers={"X-API-Key": api_key},
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass

    # Synchronous endpoint for quick demo
    try:
        resp = requests.post(
            f"{backend_url}/api/v1/datasets/regenerate-demo?scale={scale}",
            headers={"X-API-Key": api_key},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass

    # Direct local fallback
    try:
        from ingestion.synthetic.data_generator import SyntheticDataGenerator
        from backend.rag.nodes.sql_node import SafeSQLNode
        SyntheticDataGenerator.regenerate_new_data(scale=scale)
        SafeSQLNode.reset_connection()
        return {"status": "completed", "scale": scale}
    except Exception:
        return None


def _check_job_status(job_id: str) -> dict[str, Any] | None:
    """Query backend for status of an asynchronous background job."""
    import os
    import requests
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    api_key = os.getenv("API_KEY", "dev-key-1")
    try:
        resp = requests.get(
            f"{backend_url}/api/v1/datasets/jobs/{job_id}",
            headers={"X-API-Key": api_key},
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def render_sidebar(health_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Render executive sidebar controls and return active filter & toggle dictionary.
    """
    with st.sidebar:
        # App Title & Brand
        brand_html = (
            '<div class="sidebar-brand-box">'
            '<div class="sidebar-title">RAG LAKEHOUSE ANALYST</div>'
            '<div class="sidebar-subtitle">PRODUCTION ENTERPRISE EDITION</div>'
            '</div>'
        )
        st.markdown(brand_html, unsafe_allow_html=True)

        # New Analysis Thread Button (Regenerates fresh dataset and resets thread)
        if st.button("New Analysis Thread", use_container_width=True, type="primary"):
            with st.spinner("Regenerating lakehouse transactions..."):
                _trigger_data_regeneration()
            st.session_state.messages = []
            st.session_state.conversation_id = str(uuid.uuid4())
            st.session_state.active_dataset_id = "demo_lakehouse"
            st.session_state.pending_view_navigation = "ANALYSIS CONSOLE"
            st.toast("Fresh lakehouse data generated!")
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
        target_html = (
            f'<div style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); margin-top:0.25rem;">'
            f'Target: <span style="color:#38bdf8;">{active_label}</span> ({active_rows:,} rows)'
            f'</div>'
        )
        st.markdown(target_html, unsafe_allow_html=True)

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

        infra_html = (
            f'<div class="infra-health-card">'
            f'<div class="infra-node-item">'
            f'<span><span class="infra-node-dot {gw_dot}"></span>FastAPI Gateway</span>'
            f'<span style="color:var(--text-muted); font-size:0.6875rem;">{gw_label}</span>'
            f'</div>'
            f'<div class="infra-node-item">'
            f'<span><span class="infra-node-dot online"></span>DuckDB Analytical OLAP</span>'
            f'<span style="color:var(--text-muted); font-size:0.6875rem;">READY</span>'
            f'</div>'
            f'<div class="infra-node-item">'
            f'<span><span class="infra-node-dot {pg_dot}"></span>PostgreSQL pgvector</span>'
            f'<span style="color:var(--text-muted); font-size:0.6875rem;">{pg_label}</span>'
            f'</div>'
            f'<div class="infra-node-item">'
            f'<span><span class="infra-node-dot {rd_dot}"></span>Redis Semantic Cache</span>'
            f'<span style="color:var(--text-muted); font-size:0.6875rem;">{rd_label}</span>'
            f'</div>'
            f'</div>'
        )
        st.markdown(infra_html, unsafe_allow_html=True)

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

        session_html = (
            f'<div style="margin-top:1.25rem; padding-top:0.75rem; border-top:1px solid var(--border-subtle); '
            f'font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); display:flex; justify-content:space-between;">'
            f'<span>Session: {short_id}</span>'
            f'<span>Queries: {msg_count}</span>'
            f'</div>'
        )
        st.markdown(session_html, unsafe_allow_html=True)

        # Quick Lakehouse Data Controls at Bottom of Panel
        st.markdown(
            '<div class="sidebar-section-header" style="margin-top:1rem;">Lakehouse Data Controls</div>',
            unsafe_allow_html=True,
        )

        scale_options = {
            "quick": "Quick Demo (~50K rows, ~5MB)",
            "medium": "Medium Scale (~2M rows, ~200MB)",
            "large": "Large Scale (~10M rows, ~1GB)",
            "enterprise": "Enterprise Scale (~50M rows, ~5GB)",
        }
        chosen_scale = st.selectbox(
            "Generation Scale Target",
            options=list(scale_options.keys()),
            format_func=lambda x: scale_options[x],
            index=0,
            key="lakehouse_generation_scale",
            help="Choose scale volume for synthetic orders generation. Multi-gigabyte scales stream out-of-core without RAM exhaustion.",
        )

        # Active background job tracker
        active_job_id = st.session_state.get("active_job_id")
        if active_job_id:
            job_info = _check_job_status(active_job_id)
            if job_info:
                j_status = job_info.get("status", "running")
                j_pct = min(1.0, max(0.0, float(job_info.get("progress_pct", 0.0)) / 100.0))
                j_stage = job_info.get("stage", "processing").replace("_", " ").title()
                j_msg = job_info.get("message", "")
                j_rows = job_info.get("rows_processed", 0)

                if j_status == "running":
                    st.markdown(
                        f'<div style="font-size:0.6875rem; font-family:var(--font-mono); color:#38bdf8; margin:0.5rem 0 0.25rem 0;">'
                        f'BACKGROUND JOB: {active_job_id[:12]} ({j_stage})</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(j_pct)
                    st.markdown(
                        f'<div style="font-size:0.625rem; font-family:var(--font-mono); color:var(--text-muted);">'
                        f'{j_msg} | {j_rows:,} rows</div>',
                        unsafe_allow_html=True,
                    )
                    if st.button("Refresh Job Progress", key="btn_refresh_job", use_container_width=True):
                        st.rerun()
                elif j_status == "completed":
                    st.success(f"Generation Complete! {j_rows:,} rows ready.")
                    st.session_state.active_job_id = None
                    if st.button("Reload Dataset Views", key="btn_job_done_reload", use_container_width=True):
                        st.rerun()
                elif j_status == "failed":
                    st.error(f"Job Failed: {job_info.get('error', 'Unknown error')}")
                    st.session_state.active_job_id = None

        col_side1, col_side2 = st.columns(2)
        with col_side1:
            if st.button("View Generated Data", use_container_width=True, key="btn_view_lakehouse_data", help="Inspect newly generated or active lakehouse tables in Lakehouse Table Explorer"):
                st.session_state.pending_view_navigation = "LAKEHOUSE TABLE EXPLORER"
                st.rerun()

        with col_side2:
            if st.button("Regenerate Data", use_container_width=True, key="btn_regen_lakehouse_data", help="Generate fresh synthetic orders, revenue, and customer records"):
                if chosen_scale == "quick":
                    with st.spinner("Generating fresh transactions (~50K rows)..."):
                        _trigger_data_regeneration(scale=chosen_scale)
                    st.toast("Fresh lakehouse transactions generated and reloaded!")
                    st.rerun()
                else:
                    with st.spinner(f"Initiating background streaming generation ({chosen_scale})..."):
                        res = _trigger_data_regeneration(scale=chosen_scale)
                        if res and "job_id" in res:
                            st.session_state.active_job_id = res["job_id"]
                            st.toast(f"Async generation started: {res['job_id']}")
                        else:
                            st.toast(f"Dataset generated ({chosen_scale})")
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
