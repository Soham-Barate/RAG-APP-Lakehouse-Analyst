"""
E-commerce RAG Knowledge Assistant — Streamlit Frontend

A professional chat interface with source citations, confidence scores,
filtering, and real-time streaming.
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
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="E-commerce RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())
if "system_status" not in st.session_state:
    st.session_state.system_status = "unknown"

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "dev-key-1")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def check_backend_health() -> str:
    """Check backend health status."""
    try:
        resp = httpx.get(
            f"{BACKEND_URL}/api/v1/health",
            headers={"X-API-Key": API_KEY},
            timeout=5.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("status", "unknown")
    except Exception:
        pass
    return "unhealthy"


def send_chat_request(
    query: str,
    filters: dict[str, Any],
    max_results: int = 5,
) -> dict[str, Any]:
    """Send a chat request to the backend API."""
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
            return {
                "answer": f"Error: Backend returned status {resp.status_code}",
                "confidence_score": 0.0,
                "sources": [],
                "metrics": {"total_latency_ms": 0},
                "message_id": str(uuid.uuid4()),
            }
    except httpx.ConnectError:
        return {
            "answer": (
                "⚠️ Cannot connect to the backend server. "
                "Please ensure the FastAPI backend is running on "
                f"`{BACKEND_URL}`.\n\n"
                "**Quick start:**\n"
                "```bash\n"
                "cd 'd:/RAG app'\n"
                "pip install -r requirements.txt\n"
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
            "answer": f"Error communicating with backend: {str(e)}",
            "confidence_score": 0.0,
            "sources": [],
            "metrics": {"total_latency_ms": 0},
            "message_id": str(uuid.uuid4()),
        }


# ---------------------------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------------------------
def main() -> None:
    """Main application entry point."""

    # Check health on first load
    if st.session_state.system_status == "unknown":
        st.session_state.system_status = check_backend_health()

    # Render sidebar and get filter values
    sidebar_config = render_sidebar()

    # ----- Main content area -----
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <h1 style="background:linear-gradient(135deg, #6366f1, #818cf8, #22d3ee);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                       background-clip:text; font-size:2rem; font-weight:700;
                       margin-bottom:0.25rem;">
                🧠 E-commerce RAG Assistant
            </h1>
            <p style="color:#64748b; font-size:0.85rem; margin:0;">
                Intelligent answers from your product catalog, orders, reviews & support data
            </p>
        </div>
        <hr style="border-color:rgba(255,255,255,0.06); margin:0.5rem 0 1rem;">
        """,
        unsafe_allow_html=True,
    )

    # ----- Chat history -----
    if not st.session_state.messages:
        # Show starter prompts for new conversations
        st.markdown(
            """
            <div style="text-align:center; padding:2rem 0 1rem;">
                <div style="font-size:3rem; margin-bottom:0.5rem;">💡</div>
                <div style="color:#94a3b8; font-size:0.95rem;">
                    Start a conversation or try one of these suggestions:
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        starter_query = render_starter_prompts()
        if starter_query:
            st.session_state.pending_query = starter_query
            st.rerun()
    else:
        # Render existing messages
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                render_user_message(msg["content"])
            else:
                render_assistant_message(
                    msg["content"],
                    confidence=msg.get("confidence", 0.0),
                    disclaimer=msg.get("disclaimer"),
                    message_id=msg.get("message_id", ""),
                )

                # Show sources
                if sidebar_config.get("show_sources") and msg.get("sources"):
                    render_sources(msg["sources"])

                # Show metrics
                if sidebar_config.get("show_metrics") and msg.get("metrics"):
                    render_metrics_panel(msg["metrics"])

                # Feedback buttons
                if msg.get("message_id"):
                    render_feedback(
                        msg["message_id"],
                        st.session_state.conversation_id,
                        BACKEND_URL,
                    )

    # ----- Chat input -----
    # Check for pending query from starter prompts
    pending = st.session_state.pop("pending_query", None)

    user_input = st.chat_input(
        "Ask about products, orders, reviews, sales trends...",
        key="chat_input",
    )

    query = pending or user_input

    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query,
        })

        # Display the user message
        render_user_message(query)

        # Show typing indicator
        with st.spinner(""):
            render_typing_indicator()

            # Build filters
            filters = {
                "categories": sidebar_config.get("categories"),
                "document_types": sidebar_config.get("document_types"),
                "date_from": sidebar_config.get("date_from"),
                "date_to": sidebar_config.get("date_to"),
            }

            # Send request
            response = send_chat_request(
                query=query,
                filters=filters,
                max_results=sidebar_config.get("max_results", 5),
            )

        # Add assistant message to state
        assistant_msg = {
            "role": "assistant",
            "content": response.get("answer", "No response received."),
            "confidence": response.get("confidence_score", 0.0),
            "sources": response.get("sources", []),
            "metrics": response.get("metrics", {}),
            "disclaimer": response.get("disclaimer"),
            "message_id": response.get("message_id", str(uuid.uuid4())),
        }
        st.session_state.messages.append(assistant_msg)

        # Force rerun to properly render the new message
        st.rerun()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
else:
    main()
