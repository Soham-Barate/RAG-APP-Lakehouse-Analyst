"""
Chat interface components for executive message rendering, starter queries, and metrics.
Adheres strictly to zero-emoji design system, deterministic typography, and Linear/Apple craft.
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

# Clean SVG Icon Primitives
SVG_USER = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>'
    '<circle cx="12" cy="7" r="4"></circle></svg>'
)

SVG_BOT = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="2" y="3" width="20" height="14" rx="2"></rect>'
    '<line x1="8" y1="21" x2="16" y2="21"></line>'
    '<line x1="12" y1="17" x2="12" y2="21"></line></svg>'
)

SVG_ANALYTICS = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<line x1="18" y1="20" x2="18" y2="10"></line>'
    '<line x1="12" y1="20" x2="12" y2="4"></line>'
    '<line x1="6" y1="20" x2="6" y2="14"></line></svg>'
)

SVG_ORDERS = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="1" y="3" width="15" height="13"></rect>'
    '<polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon>'
    '<circle cx="5.5" cy="18.5" r="2.5"></circle>'
    '<circle cx="18.5" cy="18.5" r="2.5"></circle></svg>'
)

SVG_PRODUCTS = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>'
    '<polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>'
    '<line x1="12" y1="22.08" x2="12" y2="12"></line></svg>'
)

SVG_RETURNS = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="1 4 1 10 7 10"></polyline>'
    '<path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>'
)


def render_user_message(message: str) -> None:
    """Render an executive user query bubble."""
    safe_msg = html.escape(message).replace("\n", "<br>")
    st.markdown(
        f'<div class="message-row-user"><div class="user-bubble">{safe_msg}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistant_message(
    message: str,
    confidence: float = 0.0,
    route: str | None = None,
    disclaimer: str | None = None,
    message_id: str = "",
) -> None:
    """
    Render an assistant analysis response with route badge and calibrated confidence score.
    """
    # Calibrated confidence categorization
    if confidence >= 0.70:
        conf_class = "confidence-high"
        conf_label = f"HIGH CONFIDENCE {confidence:.0%}"
    elif confidence >= 0.40:
        conf_class = "confidence-med"
        conf_label = f"MED CONFIDENCE {confidence:.0%}"
    else:
        conf_class = "confidence-low"
        conf_label = f"LOW CONFIDENCE {confidence:.0%}"

    # Route indicator badge
    route_html = ""
    if route:
        r_clean = route.lower()
        if "sql" in r_clean:
            route_html = '<span class="route-tag route-sql">SQL EXECUTION</span>'
        elif "vector" in r_clean:
            route_html = '<span class="route-tag route-vector">VECTOR RETRIEVAL</span>'
        else:
            route_html = '<span class="route-tag route-hybrid">HYBRID LAKEHOUSE</span>'

    # Low-confidence disclaimer
    disclaimer_html = ""
    if disclaimer:
        safe_disc = html.escape(disclaimer)
        disclaimer_html = (
            f'<div class="assistant-disclaimer">'
            f'<strong>System Notice:</strong> {safe_disc}'
            f'</div>'
        )

    header_html = (
        f'<div class="assistant-header-bar">'
        f'<div class="assistant-title-group">'
        f'<div class="assistant-icon-box">{SVG_BOT}</div>'
        f'<span class="assistant-name">Lakehouse Analyst</span>'
        f'{route_html}'
        f'</div>'
        f'<div class="header-badges-right">'
        f'<span class="confidence-pill {conf_class}">{conf_label}</span>'
        f'</div>'
        f'</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Render message markdown natively inside Streamlit for math and code formatting
    st.markdown(message)

    if disclaimer_html:
        st.markdown(disclaimer_html, unsafe_allow_html=True)


def render_typing_indicator() -> None:
    """Render a subtle, linear-style pulse loading indicator."""
    st.markdown(
        '<div class="skeleton-loader-bar">'
        '<div class="skeleton-pulse-ring"></div>'
        '<span class="skeleton-text">Executing RAG lakehouse pipeline...</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_metrics_panel(metrics: dict[str, Any]) -> None:
    """Render real pipeline performance metrics in an executive ribbon."""
    latency = metrics.get("total_latency_ms", 0)
    tokens = metrics.get("tokens_used", 0)
    docs_retrieved = metrics.get("documents_retrieved", 0)
    docs_reranked = metrics.get("documents_after_rerank", 0)
    cache_hit = metrics.get("cache_hit", False)

    cache_badge = "HIT" if cache_hit else "MISS"
    cache_color = "var(--status-success)" if cache_hit else "var(--text-muted)"

    metrics_html = (
        f'<div class="metrics-ribbon">'
        f'<span class="metric-pill-item">Latency: <strong>{latency:.0f}ms</strong></span>'
        f'<span class="metric-pill-item">Tokens: <strong>{tokens:,}</strong></span>'
        f'<span class="metric-pill-item">Retrieved: <strong>{docs_retrieved} docs</strong></span>'
        f'<span class="metric-pill-item">After Rerank: <strong>{docs_reranked}</strong></span>'
        f'<span class="metric-pill-item">Semantic Cache: <strong style="color:{cache_color};">{cache_badge}</strong></span>'
        f'</div>'
    )
    st.markdown(metrics_html, unsafe_allow_html=True)


def render_starter_prompts() -> str | None:
    """
    Render 4 focused analytical preset queries.
    Returns the selected query string if clicked, else None.
    """
    starters = [
        {
            "tag": "SQL REVENUE",
            "title": "Quarterly Revenue by Category",
            "desc": "Aggregate sales, revenue trends, and comparative performance across product lines.",
            "query": "Show me the top 5 product categories by revenue this quarter and compare them to last quarter.",
        },
        {
            "tag": "SQL LOGISTICS",
            "title": "Fulfillment Rates & Delays",
            "desc": "Analyze fulfillment times, regional logistics bottlenecks, and transit latency.",
            "query": "What are our current order fulfillment rates and what shipping delays exist by region?",
        },
        {
            "tag": "HYBRID AUDIT",
            "title": "Defect & Return Rate Correlation",
            "desc": "Cross-reference structured return volume against unstructured review complaints.",
            "query": "Which products have the highest return rates, and what are the primary customer complaints for them?",
        },
        {
            "tag": "VECTOR SENTIMENT",
            "title": "Customer Voice & Praise",
            "desc": "Synthesize unstructured review feedback, satisfaction patterns, and feature sentiment.",
            "query": "Summarize customer feedback and satisfaction themes for newly launched electronics products.",
        },
    ]

    st.markdown(
        '<div class="starter-headline">'
        '<span style="font-size:0.75rem; font-family:var(--font-mono); color:var(--text-muted); text-transform:uppercase; letter-spacing:0.06em;">'
        'Quick Analytical Directives'
        '</span></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    for i, item in enumerate(starters):
        with cols[i % 2]:
            button_label = f"[{item['tag']}] {item['title']}\n{item['desc']}"
            if st.button(button_label, key=f"starter_query_{i}", use_container_width=True):
                return item["query"]

    return None
