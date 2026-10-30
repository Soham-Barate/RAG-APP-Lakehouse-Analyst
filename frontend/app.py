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

# Load Custom Executive Stylesheet
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

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

    st.markdown(
        f"""
        <div class="app-header">
            <div class="app-title-group">
                <div class="app-badge-row">
                    <span class="app-version-pill">LAKEHOUSE v2.4</span>
                    <span class="app-version-pill">DUCKDB + PGVECTOR</span>
                </div>
                <h1 class="app-title">RAG Lakehouse Analyst</h1>
                <p class="app-subtitle">
                    Dual-Path Enterprise Intelligence: Structured SQL Aggregations & Hybrid Semantic Retrieval
                </p>
            </div>
            <div class="app-status-badge">
                <span class="status-indicator-dot {status_cls}"></span>
                <span>GATEWAY {status_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    # Active Dataset Status Banner
    active_ds = sidebar_config.get("active_dataset_id", "demo_lakehouse")
    active_label = "Demo Lakehouse" if active_ds == "demo_lakehouse" else active_ds
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; padding:0.375rem 0.75rem; 
                    background:rgba(255,255,255,0.02); border:1px solid var(--border-subtle); border-radius:var(--radius-sm); 
                    margin-bottom:1rem; font-family:var(--font-mono); font-size:0.6875rem;">
            <span>ACTIVE QUERY TARGET: <strong style="color:#38bdf8;">{html.escape(active_label)}</strong></span>
            <span style="color:var(--text-muted);">ISOLATION: STRICT READ-ONLY DUCKDB</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
                vector_src = msg.get("sources") if sidebar_config.get("show_sources") else None

                if sql_q or sql_res or vector_src:
                    render_sources(
                        sources=vector_src,
                        sql_query=sql_q,
                        sql_results=sql_res,
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
            "sources": response.get("sources", []),
            "metrics": response.get("metrics", {}),
            "disclaimer": response.get("disclaimer"),
            "message_id": response.get("message_id", str(uuid.uuid4())),
        }
        st.session_state.messages.append(assistant_msg)
        st.rerun()


if __name__ == "__main__":
    main()
