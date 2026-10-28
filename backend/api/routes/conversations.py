"""
Conversation management API endpoints.

Provides CRUD operations for conversations and messages, plus export
functionality (JSON, Markdown).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from backend.api.middleware.rbac import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/conversations", tags=["Conversations"])


# ---------------------------------------------------------------------------
# In-memory store (production would use PostgreSQL ORM)
# ---------------------------------------------------------------------------
# This is a lightweight in-process store.  For persistence across restarts,
# swap with a database-backed implementation.
_conversations: dict[str, dict[str, Any]] = {}


def _get_or_create(conversation_id: str) -> dict[str, Any]:
    if conversation_id not in _conversations:
        _conversations[conversation_id] = {
            "id": conversation_id,
            "messages": [],
            "created_at": time.time(),
            "updated_at": time.time(),
        }
    return _conversations[conversation_id]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "",
    summary="List conversations",
    dependencies=[Depends(require_permission("conversations:list"))],
)
async def list_conversations() -> dict[str, Any]:
    """Return all conversation summaries."""
    summaries = []
    for conv in _conversations.values():
        summaries.append({
            "id": conv["id"],
            "message_count": len(conv["messages"]),
            "created_at": conv["created_at"],
            "updated_at": conv["updated_at"],
            "preview": (
                conv["messages"][-1]["content"][:100]
                if conv["messages"]
                else ""
            ),
        })

    # Sort by most recent
    summaries.sort(key=lambda c: c["updated_at"], reverse=True)

    return {"conversations": summaries, "total": len(summaries)}


@router.get(
    "/{conversation_id}",
    summary="Get conversation",
    dependencies=[Depends(require_permission("conversations:read"))],
)
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    """Return the full conversation with all messages."""
    if conversation_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NOT_FOUND",
                "message": f"Conversation '{conversation_id}' not found.",
            },
        )
    return _conversations[conversation_id]


@router.delete(
    "/{conversation_id}",
    summary="Delete conversation",
    dependencies=[Depends(require_permission("conversations:delete"))],
)
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    """Delete a conversation and all its messages."""
    if conversation_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NOT_FOUND",
                "message": f"Conversation '{conversation_id}' not found.",
            },
        )
    del _conversations[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}


@router.post(
    "/{conversation_id}/export",
    summary="Export conversation",
    dependencies=[Depends(require_permission("conversations:export"))],
)
async def export_conversation(
    conversation_id: str,
    format: str = "json",
) -> dict[str, Any]:
    """Export a conversation as JSON or Markdown."""
    if conversation_id not in _conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "NOT_FOUND",
                "message": f"Conversation '{conversation_id}' not found.",
            },
        )

    conv = _conversations[conversation_id]

    if format == "markdown":
        lines = [
            f"# Conversation {conv['id']}",
            "",
        ]
        for msg in conv["messages"]:
            role = msg["role"].capitalize()
            content = msg["content"]
            lines.append(f"## {role}")
            lines.append(content)
            lines.append("")
        return {"format": "markdown", "content": "\n".join(lines)}

    # Default: JSON
    return {"format": "json", "content": conv}


# ---------------------------------------------------------------------------
# Helper for other modules to store messages
# ---------------------------------------------------------------------------
def store_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Store a message in a conversation (called by chat endpoint)."""
    conv = _get_or_create(conversation_id)
    conv["messages"].append({
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {},
    })
    conv["updated_at"] = time.time()
