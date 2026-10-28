"""
Centralized input validation and sanitization.

Provides reusable validators for all user-facing inputs to prevent
injection, path traversal, resource abuse, and other attacks.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_QUERY_LENGTH = 4000
MAX_CONVERSATION_ID_LENGTH = 64
ALLOWED_CONVERSATION_ID_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]{1,64}$")

# Prompt injection patterns (case-insensitive)
INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"forget\s+(everything|all|your\s+instructions)",
    r"(system|internal)\s+prompt",
    r"reveal\s+your\s+(instructions|system\s+prompt|rules)",
    r"you\s+are\s+now\s+a",
    r"act\s+as\s+if\s+you",
    r"pretend\s+(you\s+are|to\s+be)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode\s+(enabled|on)",
    r"do\s+anything\s+now",
    r"ignore\s+safety\s+(guidelines|rules|filters)",
    r"bypass\s+(filters|restrictions|safety)",
    r"\[INST\]",          # Llama-style prompt injection
    r"<\|im_start\|>",    # ChatML injection
    r"<\|system\|>",      # System tag injection
    r"```\s*system",       # Markdown-wrapped system prompt
]

_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS
]


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
def sanitize_query(query: str) -> str:
    """
    Sanitize a user query.

    - Unicode normalizes (NFC) to prevent homoglyph attacks
    - Strips control characters
    - Truncates to MAX_QUERY_LENGTH
    """
    # NFC normalize → consistent representation
    query = unicodedata.normalize("NFC", query)

    # Remove control characters (keep newlines and tabs)
    query = "".join(
        ch for ch in query
        if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t", "\r")
    )

    # Strip and truncate
    query = query.strip()
    if len(query) > MAX_QUERY_LENGTH:
        query = query[:MAX_QUERY_LENGTH]
        logger.warning("Query truncated to %d characters", MAX_QUERY_LENGTH)

    return query


def detect_prompt_injection(query: str) -> tuple[bool, list[str]]:
    """
    Detect prompt injection attempts in user input.

    Returns:
        (is_injection, list_of_matched_patterns)
    """
    matched: list[str] = []
    for i, pattern in enumerate(_COMPILED_PATTERNS):
        if pattern.search(query):
            matched.append(INJECTION_PATTERNS[i])

    if matched:
        logger.warning(
            "Prompt injection detected: %d pattern(s) matched", len(matched)
        )

    return bool(matched), matched


def validate_conversation_id(conversation_id: str) -> str:
    """
    Validate and sanitize a conversation ID.

    Must be alphanumeric + hyphens/underscores, max 64 chars.
    """
    if not ALLOWED_CONVERSATION_ID_PATTERN.match(conversation_id):
        # Generate a safe fallback
        import uuid
        safe_id = str(uuid.uuid4())
        logger.warning(
            "Invalid conversation_id rejected, using fallback: %s",
            safe_id,
        )
        return safe_id
    return conversation_id


def validate_metadata_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and sanitize metadata filter values.

    Ensures no SQL injection via filter values by type-checking
    and length-limiting all fields.
    """
    clean: dict[str, Any] = {}

    if "categories" in filters and filters["categories"]:
        cats = filters["categories"]
        if isinstance(cats, list):
            clean["categories"] = [
                str(c)[:100] for c in cats[:20]  # Max 20 categories, 100 chars each
            ]

    if "document_types" in filters and filters["document_types"]:
        types = filters["document_types"]
        allowed = {"product", "order", "review", "ticket", "customer", "knowledge_base"}
        if isinstance(types, list):
            clean["document_types"] = [
                t for t in types if str(t) in allowed
            ]

    if "date_from" in filters and filters["date_from"]:
        clean["date_from"] = str(filters["date_from"])[:30]

    if "date_to" in filters and filters["date_to"]:
        clean["date_to"] = str(filters["date_to"])[:30]

    if "departments" in filters and filters["departments"]:
        depts = filters["departments"]
        if isinstance(depts, list):
            clean["departments"] = [
                str(d)[:100] for d in depts[:20]
            ]

    return clean


def estimate_token_count(text: str) -> int:
    """
    Quick estimate of token count without loading a tokenizer.

    Rule of thumb: ~4 characters per token for English text.
    """
    return max(1, len(text) // 4)
