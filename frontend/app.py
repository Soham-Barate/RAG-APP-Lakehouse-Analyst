"""
RAG Lakehouse Analyst — Executive Streamlit Frontend

An enterprise dual-path analytics platform combining:
- DuckDB structured analytical queries (Safe Text-to-SQL execution over Parquet lakehouse)
- Hybrid vector & BM25 unstructured retrieval (pgvector + Reciprocal Rank Fusion)
- Interactive Lakehouse Table & Schema Explorer
- Real Cluster Telemetry & Observability Diagnostics
- Strict Apple/Linear-grade design restraint, zero-emoji policy, and deterministic typography.
"""

from __future__ import annotations

import html
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx
import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from frontend.components.chat_interface import (
    render_assistant_message,
    render_metrics_panel,
    render_starter_prompts,
    render_typing_indicator,
    render_user_message,
)
from frontend.components.diagnostics import render_diagnostics_hub
from frontend.components.feedback_widget import render_feedback
from frontend.components.sidebar import render_sidebar
from frontend.components.source_display import render_sources
from frontend.components.table_explorer import render_table_explorer

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Lakehouse Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Fonts and Custom Executive Stylesheet
font_tags = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200&display=swap" rel="stylesheet">
"""
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"{font_tags}\n<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state or not st.session_state.conversation_id:
    st.session_state.conversation_id = str(uuid.uuid4())
if "active_view" not in st.session_state:
    st.session_state.active_view = "ANALYSIS CONSOLE"
if "health_info" not in st.session_state:
    st.session_state.health_info = None

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-1")


# ---------------------------------------------------------------------------
# Backend Communications
# ---------------------------------------------------------------------------
def check_backend_health() -> dict[str, Any]:
    """Probe the backend health endpoint and return status dictionary."""
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/health",
            headers={"X-API-Key": API_KEY},
            timeout=3.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            data["reachable"] = True
            return data
        return {"reachable": True, "status": "degraded", "services": {}}
    except Exception:
        pass
    return {"reachable": False, "status": "offline", "services": {}}


def send_chat_request(
    query: str,
    filters: dict[str, Any],
    max_results: int = 5,
    dataset_id: str = "demo_lakehouse",
) -> dict[str, Any]:
    """Execute a chat query against the FastAPI Lakehouse Gateway."""
    payload = {
        "query": query,
        "conversation_id": st.session_state.conversation_id,
        "dataset_id": dataset_id,
        "filters": {k: v for k, v in filters.items() if v is not None},
        "max_results": max_results,
        "include_sources": True,
    }

    try:
        resp = httpx.post(
            f"{BACKEND_URL}/api/v1/chat",
            json=payload,
            headers={"X-API-Key": API_KEY},
            timeout=120.0,
        )

        if resp.status_code == 200:
            return resp.json()
        else:
            try:
                err_data = resp.json()
                detail = err_data.get("detail", str(err_data))
            except Exception:
                detail = resp.text or f"Status {resp.status_code}"
            return {
                "answer": f"[Service Notice] Gateway returned status {resp.status_code}: {detail}",
                "confidence_score": 0.0,
                "sources": [],
                "metrics": {"total_latency_ms": 0},
                "message_id": str(uuid.uuid4()),
            }
    except httpx.ConnectError:
        return {
            "answer": (
                "[Connection Warning] Cannot reach the Lakehouse Analyst API gateway at "
                f"`{BACKEND_URL}`.\n\n"
                "To start the local gateway server:\n"
                "```bash\n"
                "python -m backend.main\n"
                "```"
            ),
            "confidence_score": 0.0,
            "sources": [],
            "metrics": {"total_latency_ms": 0},
            "message_id": str(uuid.uuid4()),
        }
    except Exception as e:
        return {
            "answer": f"[Gateway Error] {str(e)}",
            "confidence_score": 0.0,
            "sources": [],
            "metrics": {"total_latency_ms": 0},
            "message_id": str(uuid.uuid4()),
        }


# ---------------------------------------------------------------------------
# Main Application Flow
# ---------------------------------------------------------------------------
def main() -> None:
    """Render executive page and handle chat interactions."""
    # Probe real health once or use cached
    health_info = check_backend_health()
    st.session_state.health_info = health_info

    sidebar_config = render_sidebar(health_info=health_info)

    # Executive Top Bar
    gw_status = health_info.get("status", "offline")
    gw_reachable = health_info.get("reachable", False)
    status_cls = "healthy" if (gw_reachable and gw_status == "healthy") else ("standby" if gw_reachable else "degraded")
    status_label = "ONLINE" if (gw_reachable and gw_status == "healthy") else ("DEGRADED" if gw_reachable else "OFFLINE")

    header_html = (
        f'<div class="app-header">'
        f'<div class="app-title-group">'
        f'<div class="app-badge-row">'
        f'<span class="app-version-pill">LAKEHOUSE v2.4</span>'
        f'<span class="app-version-pill">DUCKDB + PGVECTOR</span>'
        f'</div>'
        f'<h1 class="app-title">RAG Lakehouse Analyst</h1>'
        f'<p class="app-subtitle">'
        f'Dual-Path Enterprise Intelligence: Structured SQL Aggregations & Hybrid Semantic Retrieval'
        f'</p>'
        f'</div>'
        f'<div class="app-status-badge">'
        f'<span class="status-indicator-dot {status_cls}"></span>'
        f'<span>GATEWAY {status_label}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Top-Level Segmented Control Navigation
    active_view = st.segmented_control(
        "Navigation View",
        options=["ANALYSIS CONSOLE", "DATASET MANAGER", "LAKEHOUSE TABLE EXPLORER", "CLUSTER TELEMETRY"],
        default=st.session_state.get("active_view", "ANALYSIS CONSOLE"),
        key="active_view",
        label_visibility="collapsed",
    )

    # -------------------------------------------------------------------------
    # VIEW 1: ANALYSIS CONSOLE
    # -------------------------------------------------------------------------
    if active_view == "ANALYSIS CONSOLE":
        _render_analysis_console(sidebar_config)

    # -------------------------------------------------------------------------
    # VIEW 2: DATASET MANAGER & PROFILER
    # -------------------------------------------------------------------------
    elif active_view == "DATASET MANAGER":
        from frontend.components.dataset_view import render_dataset_manager
        render_dataset_manager()

    # -------------------------------------------------------------------------
    # VIEW 3: LAKEHOUSE TABLE EXPLORER
    # -------------------------------------------------------------------------
    elif active_view == "LAKEHOUSE TABLE EXPLORER":
        render_table_explorer()

    # -------------------------------------------------------------------------
    # VIEW 4: CLUSTER TELEMETRY
    # -------------------------------------------------------------------------
    elif active_view == "CLUSTER TELEMETRY":
        render_diagnostics_hub()


def _render_analysis_console(sidebar_config: dict[str, Any]) -> None:
    """Render the primary analytical conversation interface."""
    # -------------------------------------------------------------------------
    # Active Query Target Interactive Bar (Dropdown + New Dataset Creator)
    # -------------------------------------------------------------------------
    try:
        from frontend.components.dataset_view import fetch_datasets
        available_datasets = fetch_datasets()
    except Exception:
        available_datasets = [{"dataset_id": "demo_lakehouse", "name": "E-Commerce Lakehouse (Demo)", "row_count": 50000, "is_demo": True}]

    ds_map = {d["dataset_id"]: d for d in available_datasets}
    active_ds = st.session_state.get("active_dataset_id", "demo_lakehouse")
    if active_ds not in ds_map:
        active_ds = "demo_lakehouse"
        st.session_state.active_dataset_id = active_ds

    ds_keys = list(ds_map.keys())
    active_idx = ds_keys.index(active_ds) if active_ds in ds_keys else 0

    target_card = st.container()
    with target_card:
        bar_col1, bar_col2, bar_col3 = st.columns([1.5, 3.5, 1.3], vertical_alignment="center")

        with bar_col1:
            st.markdown(
                '<div style="font-family:var(--font-mono); font-size:0.6875rem; color:var(--text-muted); '
                'text-transform:uppercase; letter-spacing:0.04em; font-weight:600;">'
                'ACTIVE QUERY TARGET:'
                '</div>',
                unsafe_allow_html=True,
            )

        with bar_col2:
            def _format_ds_label(ds_id: str) -> str:
                d = ds_map.get(ds_id, {})
                name = d.get("name", ds_id)
                rows = d.get("row_count", 0)
                tag = "Demo Lakehouse" if d.get("is_demo") else "User Dataset"
                return f"{name} ({tag} - {rows:,} rows)"

            selected_ds = st.selectbox(
                "Change active dataset",
                options=ds_keys,
                index=active_idx,
                format_func=_format_ds_label,
                label_visibility="collapsed",
                key="console_active_dataset_selector",
            )
            if selected_ds != active_ds:
                st.session_state.active_dataset_id = selected_ds
                st.rerun()

        with bar_col3:
            # Inline popover to upload or make a new dataset directly from the console
            with st.popover("+ New Dataset", use_container_width=True):
                st.markdown(
                    '<div style="font-weight:600; font-size:0.875rem; margin-bottom:0.25rem;">Upload New Dataset</div>'
                    '<div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.75rem;">'
                    'Add CSV, Parquet, Excel, or JSON to analyze with isolated DuckDB SQL & RAG.</div>',
                    unsafe_allow_html=True,
                )
                quick_file = st.file_uploader(
                    "Upload file",
                    type=["csv", "xlsx", "xls", "parquet", "json"],
                    key="quick_console_file_upload",
                    label_visibility="collapsed",
                )
                if quick_file is not None:
                    if st.button("Ingest & Activate Dataset", type="primary", use_container_width=True, key="btn_quick_ingest"):
                        with st.spinner("Ingesting, profiling and registering DuckDB table..."):
                            from frontend.components.dataset_view import upload_file_to_backend
                            result = upload_file_to_backend(quick_file)
                            if result.get("success"):
                                new_id = result["metadata"]["dataset_id"]
                                st.session_state.active_dataset_id = new_id
                                st.toast(f"Dataset '{quick_file.name}' ingested and set as active target!")
                                st.rerun()
                            else:
                                st.error(result.get("error", "Upload failed"))

                st.divider()
                if st.button("Open Full Dataset Manager", use_container_width=True, key="btn_goto_dataset_mgr"):
                    st.session_state.active_view = "DATASET MANAGER"
                    st.rerun()

    # Isolation info tag
    active_item = ds_map.get(active_ds, {})
    table_tag = active_item.get("table_name", "products, orders, customers")
    info_html = (
        f'<div style="display:flex; align-items:center; justify-content:space-between; padding:0.25rem 0.75rem 0.5rem 0.75rem; '
        f'margin-bottom:0.75rem; border-bottom:1px solid var(--border-subtle); font-family:var(--font-mono); font-size:0.625rem; color:var(--text-muted);">'
        f'<span>ISOLATED TABLE(S): <strong style="color:var(--text-secondary);">{html.escape(table_tag)}</strong></span>'
        f'<span>SECURITY: READ-ONLY DUCKDB (NO DDL/DML, NO EXTERNAL ACCESS)</span>'
        f'</div>'
    )
    st.markdown(info_html, unsafe_allow_html=True)

    # Empty State or Message History
    if not st.session_state.messages:
        starter_query = render_starter_prompts()
        if starter_query:
            st.session_state.pending_query = starter_query
            st.rerun()
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_message(msg["content"])
            else:
                render_assistant_message(
                    msg["content"],
                    confidence=msg.get("confidence", 0.0),
                    route=msg.get("route"),
                    disclaimer=msg.get("disclaimer"),
                    message_id=msg.get("message_id", ""),
                )

                # Render Structured SQL + Vector Evidence
                sql_q = msg.get("sql_query") if sidebar_config.get("show_sql") else None
                sql_res = msg.get("sql_results") if sidebar_config.get("show_sql") else None
                sql_err = msg.get("sql_error") if sidebar_config.get("show_sql") else None
                vector_src = msg.get("sources") if sidebar_config.get("show_sources") else None

                if sql_q or sql_res or sql_err or vector_src:
                    render_sources(
                        sources=vector_src,
                        sql_query=sql_q,
                        sql_results=sql_res,
                        sql_error=sql_err,
                    )

                # Render Pipeline Metrics Ribbon
                if sidebar_config.get("show_metrics") and msg.get("metrics"):
                    render_metrics_panel(msg["metrics"])

                # Feedback Controls
                if msg.get("message_id"):
                    render_feedback(
                        msg["message_id"],
                        st.session_state.conversation_id,
                        BACKEND_URL,
                    )

    # Chat Input Field
    pending = st.session_state.pop("pending_query", None)
    user_input = st.chat_input(
        "Ask an analytical question (e.g. quarterly revenue by category, fulfillment delays)...",
        key="chat_input",
    )

    query = pending or user_input

    if query:
        # Append and immediately render user message
        st.session_state.messages.append({
            "role": "user",
            "content": query,
        })
        render_user_message(query)

        # Execute query with loading indicator
        with st.spinner(""):
            render_typing_indicator()

            filters = {
                "categories": sidebar_config.get("categories"),
                "document_types": sidebar_config.get("document_types"),
                "date_from": sidebar_config.get("date_from"),
                "date_to": sidebar_config.get("date_to"),
            }

            response = send_chat_request(
                query=query,
                filters=filters,
                max_results=sidebar_config.get("max_results", 5),
                dataset_id=sidebar_config.get("active_dataset_id", "demo_lakehouse"),
            )

        # Append assistant message with full dual-path metadata
        assistant_msg = {
            "role": "assistant",
            "content": response.get("answer", "No analysis returned from gateway."),
            "confidence": response.get("confidence_score", 0.0),
            "route": response.get("route"),
            "sql_query": response.get("sql_query"),
            "sql_results": response.get("sql_results"),
            "sql_error": response.get("sql_error"),
            "sources": response.get("sources", []),
            "metrics": response.get("metrics", {}),
            "disclaimer": response.get("disclaimer"),
            "message_id": response.get("message_id", str(uuid.uuid4())),
        }
        st.session_state.messages.append(assistant_msg)
        st.rerun()


if __name__ == "__main__":
    main()
