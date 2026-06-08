"""
Chat interface component for rendering messages.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render_user_message(message: str) -> None:
    """Render a user message bubble."""
    st.markdown(
        f"""
        <div class="user-message">
            <div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.5rem;">
                <span style="font-size:1.1rem;">👤</span>
                <span style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">You</span>
            </div>
            <div style="color:#cbd5e1; line-height:1.6;">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_assistant_message(
    message: str,
    confidence: float = 0.0,
    disclaimer: str | None = None,
    message_id: str = "",
) -> None:
    """Render an assistant message bubble with confidence badge."""
    # Confidence badge
    if confidence >= 0.7:
        badge_class = "confidence-high"
        badge_text = f"✓ High confidence ({confidence:.0%})"
    elif confidence >= 0.4:
        badge_class = "confidence-medium"
        badge_text = f"~ Medium confidence ({confidence:.0%})"
    else:
        badge_class = "confidence-low"
        badge_text = f"⚠ Low confidence ({confidence:.0%})"

    disclaimer_html = ""
    if disclaimer:
        disclaimer_html = f"""
        <div style="background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3);
                     border-radius:8px; padding:0.5rem 0.75rem; margin-top:0.75rem;
                     font-size:0.8rem; color:#fbbf24;">
            {disclaimer}
        </div>
        """

    st.markdown(
        f"""
        <div class="assistant-message">
            <div style="display:flex; align-items:center; justify-content:space-between;
                        margin-bottom:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.5rem;">
                    <span style="font-size:1.1rem;">🤖</span>
                    <span style="font-weight:600; color:#e2e8f0; font-size:0.85rem;">
                        RAG Assistant
                    </span>
                </div>
                <span class="confidence-badge {badge_class}">{badge_text}</span>
            </div>
            <div style="color:#cbd5e1; line-height:1.7;">{message}</div>
            {disclaimer_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_typing_indicator() -> None:
    """Show a typing indicator while the assistant is generating."""
    st.markdown(
        """
        <div class="typing-indicator">
            <span>🤖</span>
            <span>RAG Assistant is thinking</span>
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics_panel(metrics: dict[str, Any]) -> None:
    """Render pipeline performance metrics."""
    latency = metrics.get("total_latency_ms", 0)
    tokens = metrics.get("tokens_used", 0)
    iterations = metrics.get("retrieval_iterations", 1)
    docs_retrieved = metrics.get("documents_retrieved", 0)
    cache_hit = metrics.get("cache_hit", False)

    st.markdown(
        f"""
        <div class="metrics-panel">
            <div class="metric-card">
                <div class="metric-value">{latency:.0f}ms</div>
                <div class="metric-label">Latency</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{tokens:,}</div>
                <div class="metric-label">Tokens Used</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{docs_retrieved}</div>
                <div class="metric-label">Docs Retrieved</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{iterations}</div>
                <div class="metric-label">Iterations</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{"✓" if cache_hit else "✗"}</div>
                <div class="metric-label">Cache Hit</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_starter_prompts() -> str | None:
    """Render starter prompt cards for new conversations."""
    starters = [
        {
            "icon": "📊",
            "title": "Sales Analytics",
            "desc": "Analyze sales trends and revenue",
            "query": "What are the top-selling product categories this quarter and how do they compare to last quarter?",
        },
        {
            "icon": "📦",
            "title": "Order Insights",
            "desc": "Track orders and fulfillment",
            "query": "Show me the current order fulfillment rates and any shipping delays by region.",
        },
        {
            "icon": "⭐",
            "title": "Customer Reviews",
            "desc": "Sentiment and satisfaction analysis",
            "query": "What are customers saying about our electronics products? Summarize the main complaints and praises.",
        },
        {
            "icon": "🔄",
            "title": "Returns Analysis",
            "desc": "Understand return patterns",
            "query": "Which products have the highest return rates and what are the primary reasons for returns?",
        },
    ]

    cols = st.columns(2)
    for i, starter in enumerate(starters):
        with cols[i % 2]:
            if st.button(
                f"{starter['icon']} **{starter['title']}**\n\n{starter['desc']}",
                key=f"starter_{i}",
                use_container_width=True,
            ):
                return starter["query"]

    return None
