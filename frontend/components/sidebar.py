"""
Sidebar component with filters, settings, and conversation history.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import streamlit as st


def render_sidebar() -> dict[str, Any]:
    """
    Render the sidebar with filters, settings, and status.

    Returns a dict of active filter/settings values.
    """
    with st.sidebar:
        # Header
        st.markdown(
            """
            <div style="text-align:center; padding:1rem 0;">
                <div style="font-size:2rem; margin-bottom:0.25rem;">🧠</div>
                <div class="sidebar-header">RAG Knowledge Assistant</div>
                <div style="color:#64748b; font-size:0.75rem;">
                    E-commerce Intelligence Platform
                </div>
            </div>
            <hr style="border-color:rgba(255,255,255,0.08); margin:1rem 0;">
            """,
            unsafe_allow_html=True,
        )

        # New conversation button
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        # ----- Filters Section -----
        st.markdown(
            '<div style="font-weight:600; color:#94a3b8; font-size:0.8rem; '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            '🔍 Search Filters</div>',
            unsafe_allow_html=True,
        )

        with st.container():
            categories = st.multiselect(
                "Categories",
                [
                    "Electronics", "Clothing", "Home & Garden",
                    "Sports & Outdoors", "Books", "Beauty & Personal Care",
                    "Toys & Games", "Automotive", "Food & Grocery",
                    "Health & Wellness",
                ],
                key="filter_categories",
            )

            doc_types = st.multiselect(
                "Document Types",
                ["product", "order", "review", "ticket", "customer"],
                key="filter_doc_types",
            )

            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input(
                    "From",
                    value=datetime.now() - timedelta(days=365),
                    key="filter_date_from",
                )
            with col2:
                date_to = st.date_input(
                    "To",
                    value=datetime.now(),
                    key="filter_date_to",
                )

            max_results = st.slider(
                "Max Sources",
                min_value=1,
                max_value=20,
                value=5,
                key="filter_max_results",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ----- Model Settings -----
        st.markdown(
            '<div style="font-weight:600; color:#94a3b8; font-size:0.8rem; '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            '⚙️ Model Settings</div>',
            unsafe_allow_html=True,
        )

        with st.container():
            show_sources = st.toggle("Show Sources", value=True, key="show_sources")
            show_metrics = st.toggle("Show Metrics", value=True, key="show_metrics")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----- System Status -----
        st.markdown(
            '<div style="font-weight:600; color:#94a3b8; font-size:0.8rem; '
            'text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.5rem;">'
            '💚 System Status</div>',
            unsafe_allow_html=True,
        )

        # Get status from session state
        system_status = st.session_state.get("system_status", "unknown")
        status_color = {
            "healthy": "#34d399",
            "degraded": "#fbbf24",
            "unhealthy": "#fb7185",
            "unknown": "#64748b",
        }.get(system_status, "#64748b")

        st.markdown(
            f"""
            <div style="background:rgba(255,255,255,0.03);
                        border:1px solid rgba(255,255,255,0.08);
                        border-radius:8px; padding:0.75rem;">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <div class="status-dot status-{system_status}"></div>
                    <span style="font-size:0.8rem; color:{status_color};
                                 text-transform:capitalize;">{system_status}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Conversation count
        msg_count = len(st.session_state.get("messages", []))
        st.caption(f"Messages in conversation: {msg_count}")

    return {
        "categories": categories if categories else None,
        "document_types": doc_types if doc_types else None,
        "date_from": date_from.isoformat() if date_from else None,
        "date_to": date_to.isoformat() if date_to else None,
        "max_results": max_results,
        "show_sources": show_sources,
        "show_metrics": show_metrics,
    }
