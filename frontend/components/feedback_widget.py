"""
Feedback widget component for per-message ratings.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_feedback(message_id: str, conversation_id: str, backend_url: str) -> None:
    """Render feedback buttons (thumbs up/down) for a message."""
    feedback_key = f"feedback_{message_id}"

    # Check if already rated
    if feedback_key in st.session_state:
        rating = st.session_state[feedback_key]
        icon = "👍" if rating == "thumbs_up" else "👎"
        st.markdown(
            f'<span style="font-size:0.75rem; color:#64748b;">'
            f'{icon} Feedback recorded</span>',
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("👍", key=f"thumbs_up_{message_id}", help="Good response"):
            _submit_feedback(
                message_id, conversation_id, "thumbs_up", backend_url
            )
            st.session_state[feedback_key] = "thumbs_up"
            st.rerun()

    with col2:
        if st.button("👎", key=f"thumbs_down_{message_id}", help="Poor response"):
            _submit_feedback(
                message_id, conversation_id, "thumbs_down", backend_url
            )
            st.session_state[feedback_key] = "thumbs_down"
            st.rerun()


def _submit_feedback(
    message_id: str,
    conversation_id: str,
    rating: str,
    backend_url: str,
) -> None:
    """Submit feedback to the backend API."""
    try:
        import httpx

        httpx.post(
            f"{backend_url}/api/v1/feedback",
            json={
                "message_id": message_id,
                "conversation_id": conversation_id,
                "rating": rating,
            },
            headers={"X-API-Key": "dev-key-1"},
            timeout=5.0,
        )
    except Exception:
        pass  # Silent failure — feedback is non-critical
