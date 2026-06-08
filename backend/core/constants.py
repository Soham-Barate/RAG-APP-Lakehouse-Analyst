"""
Application-wide constants.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# API versioning
# ---------------------------------------------------------------------------
API_V1_PREFIX = "/api/v1"

# ---------------------------------------------------------------------------
# Document types in the e-commerce domain
# ---------------------------------------------------------------------------
DOCUMENT_TYPES = {
    "product": "Product Catalog",
    "order": "Order Record",
    "review": "Customer Review",
    "ticket": "Support Ticket",
    "customer": "Customer Profile",
    "knowledge_base": "Knowledge Base Article",
}

# ---------------------------------------------------------------------------
# Default LLM generation parameters
# ---------------------------------------------------------------------------
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048
STREAMING_CHUNK_SIZE = 50  # tokens per SSE chunk

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
PARENT_CHUNK_PREFIX = "parent::"
CHILD_CHUNK_PREFIX = "child::"
COLLECTION_NAME = "ecommerce_docs"

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------
PII_ENTITY_TYPES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "IBAN_CODE",
]

TOXIC_KEYWORDS = [
    "hack", "exploit", "steal", "illegal", "bypass security",
    "credit card fraud", "social engineering",
]

# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------
CACHE_PREFIX = "rag:cache:"
RATE_LIMIT_PREFIX = "rag:ratelimit:"

# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------
METRICS_NAMESPACE = "ecommerce_rag"
