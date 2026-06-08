"""
API key and JWT authentication middleware.

Provides a FastAPI dependency that validates the ``X-API-Key`` header
against a configured allowlist, with a JWT verification stub for future
expansion.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import Settings, get_settings

# ---------------------------------------------------------------------------
# API-Key scheme
# ---------------------------------------------------------------------------
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    Validate the API key from the request header.

    Returns the validated key so downstream handlers can use it for
    rate-limit bucketing.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "AUTHENTICATION_ERROR", "message": "Missing API key"},
        )
    if api_key not in settings.api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "AUTHENTICATION_ERROR", "message": "Invalid API key"},
        )
    return api_key


# ---------------------------------------------------------------------------
# JWT stub (for future expansion)
# ---------------------------------------------------------------------------
def create_jwt_token(
    data: dict[str, Any],
    settings: Settings | None = None,
) -> str:
    """Create a signed JWT token. Stub for future auth expansion."""
    try:
        from jose import jwt as jose_jwt
    except ImportError:
        raise RuntimeError("python-jose is required for JWT support")

    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.jwt_expiration_minutes
    )
    to_encode = {**data, "exp": expire}
    return jose_jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def decode_jwt_token(
    token: str,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Decode and verify a JWT token. Stub for future auth expansion."""
    try:
        from jose import JWTError, jwt as jose_jwt
    except ImportError:
        raise RuntimeError("python-jose is required for JWT support")

    settings = settings or get_settings()
    try:
        payload: dict[str, Any] = jose_jwt.decode(
            token,
            settings.jwt_secret_key.get_secret_value(),
            algorithms=[settings.jwt_algorithm],
        )
        return payload
    except JWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "AUTHENTICATION_ERROR", "message": str(exc)},
        ) from exc
