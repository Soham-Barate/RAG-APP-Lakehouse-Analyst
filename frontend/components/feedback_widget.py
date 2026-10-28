"""
Feedback widget component for per-message quality evaluation.
Clean, tactile buttons adhering to anti-emoji and accessibility standards.
"""

from __future__ import annotations

import httpx
import streamlit as st


def render_feedback(message_id: str, conversation_id: str, backend_url: str) -> None:
    """Render feedback actions for an assistant message."""
    feedback_key = f"feedback_{message_id}"

    if feedback_key in st.session_state:
        rating = st.session_state[feedback_key]
        label = "Helpful" if rating == "thumbs_up" else "Needs Improvement"
        st.markdown(
            f'<div class="feedback-record-msg">Recorded as: {label}</div>',
            unsafe_allow_html=True,
        )
        return

    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("Helpful", key=f"thumbs_up_{message_id}", help="Accurate and helpful response"):
            _submit_feedback(message_id, conversation_id, "thumbs_up", backend_url)
            st.session_state[feedback_key] = "thumbs_up"
            st.rerun()

    with col2:
        if st.button("Unhelpful", key=f"thumbs_down_{message_id}", help="Inaccurate or incomplete response"):
            _submit_feedback(message_id, conversation_id, "thumbs_down", backend_url)
            st.session_state[feedback_key] = "thumbs_down"
            st.rerun()


def _submit_feedback(
    message_id: str,
    conversation_id: str,
    rating: str,
    backend_url: str,
) -> None:
    """Submit feedback asynchronously to backend."""
    try:
        httpx.post(
            f"{backend_url}/api/v1/feedback",
            json={
                "message_id": message_id,
                "conversation_id": conversation_id,
                "rating": rating,
            },
            headers={"X-API-Key": "dev-key-1"},
            timeout=4.0,
        )
    except Exception:
        pass  # Non-blocking telemetry
