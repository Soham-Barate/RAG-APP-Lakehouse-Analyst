"""
Source citation display component.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


# Document type icons
DOC_TYPE_ICONS = {
    "product": "🏷️",
    "order": "📦",
    "review": "⭐",
    "ticket": "🎫",
    "customer": "👤",
    "knowledge_base": "📚",
}


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render source citation cards with relevance scores."""
    if not sources:
        return

    st.markdown(
        """<div style="margin-top:1rem; padding-top:0.75rem;
                       border-top:1px solid rgba(255,255,255,0.08);">
            <span style="font-size:0.8rem; font-weight:600; color:#94a3b8;
                         text-transform:uppercase; letter-spacing:0.05em;">
                📎 Sources ({count})
            </span>
        </div>""".format(count=len(sources)),
        unsafe_allow_html=True,
    )

    for i, source in enumerate(sources):
        render_source_card(source, i + 1)


def render_source_card(source: dict[str, Any], index: int) -> None:
    """Render a single source citation card."""
    doc_type = source.get("document_type", "product")
    icon = DOC_TYPE_ICONS.get(doc_type, "📄")
    relevance = source.get("relevance_score", 0.0)
    content = source.get("content", "No content available")
    metadata = source.get("metadata", {})
    doc_id = source.get("document_id", f"doc_{index}")

    # Relevance color
    if relevance >= 0.7:
        bar_class = "relevance-high"
        color = "#34d399"
    elif relevance >= 0.4:
        bar_class = "relevance-medium"
        color = "#fbbf24"
    else:
        bar_class = "relevance-low"
        color = "#fb7185"

    # Truncate content for display
    display_content = content[:300] + "..." if len(content) > 300 else content

    # Build metadata pills
    meta_html = ""
    meta_keys = ["category", "product_id", "order_id", "source_file"]
    for key in meta_keys:
        if key in metadata and metadata[key]:
            meta_html += (
                f'<span style="background:rgba(99,102,241,0.1); '
                f'border:1px solid rgba(99,102,241,0.2); '
                f'border-radius:999px; padding:0.125rem 0.5rem; '
                f'font-size:0.65rem; color:#818cf8;">'
                f'{key}: {metadata[key]}</span> '
            )

    with st.expander(
        f"{icon} Source {index} — {doc_type.replace('_', ' ').title()} "
        f"(Relevance: {relevance:.0%})",
        expanded=False,
    ):
        st.markdown(
            f"""
            <div class="source-card">
                <div style="display:flex; justify-content:space-between;
                            align-items:center; margin-bottom:0.5rem;">
                    <span style="font-size:0.75rem; color:#64748b;">
                        ID: {doc_id}
                    </span>
                    <span style="color:{color}; font-weight:600; font-size:0.8rem;">
                        {relevance:.0%} relevant
                    </span>
                </div>
                <div style="color:#cbd5e1; font-size:0.85rem; line-height:1.6;
                            margin-bottom:0.5rem;">
                    {display_content}
                </div>
                <div style="display:flex; flex-wrap:wrap; gap:0.375rem;
                            margin-top:0.5rem;">
                    {meta_html}
                </div>
                <div class="relevance-bar">
                    <div class="relevance-bar-fill {bar_class}"
                         style="width:{relevance*100:.0f}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
