"""
Role-Based Access Control (RBAC) middleware.

Defines three roles:
  - ``admin``:   Full access — chat, ingestion, evaluation, analytics, management
  - ``analyst``: Chat, conversation history, export, analytics
  - ``viewer``:  Chat only, no export or analytics

Roles are resolved from the API key via settings, with a default
of ``viewer`` for unknown keys.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import Enum

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Application roles in order of decreasing privilege."""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


# Permissions per role
ROLE_PERMISSIONS: dict[Role, set[str]] = {
    Role.ADMIN: {
        "chat", "chat:stream",
        "conversations:list", "conversations:read", "conversations:delete",
        "conversations:export",
        "ingest", "ingest:upload",
        "evaluation:run", "evaluation:read",
        "analytics:read",
        "admin:keys", "admin:settings",
    },
    Role.ANALYST: {
        "chat", "chat:stream",
        "conversations:list", "conversations:read",
        "conversations:export",
        "analytics:read",
    },
    Role.VIEWER: {
        "chat", "chat:stream",
        "conversations:list", "conversations:read",
    },
}


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def resolve_role(api_key: str, settings: Settings) -> Role:
    """Resolve the role for an API key from configuration."""
    role_map: dict[str, str] = getattr(settings, "api_key_roles", {})
    role_name = role_map.get(api_key, "viewer")
    try:
        return Role(role_name)
    except ValueError:
        return Role.VIEWER


async def get_current_role(
    api_key: str | None = Security(_api_key_header),
    settings: Settings = Depends(get_settings),
) -> Role:
    """FastAPI dependency that resolves the current user's role."""
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
    return resolve_role(api_key, settings)


def require_permission(permission: str) -> Callable:
    """
    FastAPI dependency factory that enforces a specific permission.

    Usage:
        @router.get("/admin/settings", dependencies=[Depends(require_permission("admin:settings"))])
    """
    async def _check(
        role: Role = Depends(get_current_role),
    ) -> Role:
        allowed = ROLE_PERMISSIONS.get(role, set())
        if permission not in allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "FORBIDDEN",
                    "message": f"Role '{role.value}' lacks permission '{permission}'.",
                },
            )
        return role

    return _check
