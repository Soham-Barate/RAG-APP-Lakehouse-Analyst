"""
RAG Lakehouse Analyst — Executive Streamlit Frontend

An enterprise dual-path analytics interface combining:
- DuckDB structured analytical queries (Text-to-SQL)
- Hybrid vector & BM25 unstructured retrieval (pgvector + RRF)
- Strict Apple/Linear-grade design restraint, zero-emoji policy, and deterministic typography.
"""

from __future__ import annotations

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
from frontend.components.feedback_widget import render_feedback
from frontend.components.sidebar import render_sidebar
from frontend.components.source_display import render_sources

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
if "system_status" not in st.session_state:
    st.session_state.system_status = "unknown"

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-1")


# ---------------------------------------------------------------------------
# Backend Communications
# ---------------------------------------------------------------------------
def check_backend_health() -> str:
    """Probe the backend health endpoint."""
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/health",
            headers={"X-API-Key": API_KEY},
            timeout=4.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status", "healthy")
    except Exception:
        pass
    return "degraded"


def send_chat_request(
    query: str,
    filters: dict[str, Any],
    max_results: int = 5,
) -> dict[str, Any]:
    """Execute a chat query against the FastAPI Lakehouse Gateway."""
    payload = {
        "query": query,
        "conversation_id": st.session_state.conversation_id,
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
    if st.session_state.system_status == "unknown":
        st.session_state.system_status = check_backend_health()

    sidebar_config = render_sidebar()

    # Executive Top Bar
    status_cls = "healthy" if st.session_state.system_status == "healthy" else "degraded"
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
                <span>GATEWAY {st.session_state.system_status.upper()}</span>
            </div>
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
