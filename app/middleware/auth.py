"""
AuthN/AuthZ Middleware (Tenant Isolation)

Supports two modes:
- AUTH_MODE=client_token: expects client auth token + tenant mapping
- AUTH_MODE=platform_token: expects platform API key with tenant/role mapping
Default: AUTH_MODE=none (no enforcement)
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

from fastapi import Request
from starlette.responses import JSONResponse
from app.middleware import auth_hooks

logger = logging.getLogger(__name__)


def _auth_mode() -> str:
    return os.getenv("AUTH_MODE", "none").lower()


def _parse_platform_keys() -> dict:
    """
    PLATFORM_API_KEYS format:
    key1:tenantA:admin,key2:tenantB:analyst
    """
    mapping = {}
    raw = os.getenv("PLATFORM_API_KEYS", "")
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        parts = item.split(":")
        if len(parts) >= 2:
            key = parts[0].strip()
            tenant = parts[1].strip()
            role = parts[2].strip() if len(parts) >= 3 else "member"
            mapping[key] = (tenant, role)
    return mapping


def _extract_bearer(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


async def auth_middleware(request: Request, call_next):
    mode = _auth_mode()
    if mode == "none":
        return await call_next(request)

    tenant_id = request.headers.get("x-tenant-id")
    user_id = request.headers.get("x-user-id", "anonymous")
    roles = request.headers.get("x-roles", "member")

    if mode == "platform_token":
        api_key = request.headers.get("x-api-key") or _extract_bearer(request)
        if not api_key:
            return JSONResponse({"detail": "Missing API key."}, status_code=401)
        hook_result = auth_hooks.resolve_platform_key(api_key)
        if hook_result:
            mapped_tenant, mapped_role, mapped_user = hook_result
            user_id = mapped_user or user_id
        else:
            mapping = _parse_platform_keys()
            if api_key not in mapping:
                return JSONResponse({"detail": "Invalid API key."}, status_code=403)
            mapped_tenant, mapped_role = mapping[api_key]
        # Override tenant/roles with mapped values
        tenant_id = mapped_tenant
        roles = mapped_role
        request.state.user_id = user_id
        request.state.roles = roles
        request.state.tenant_id = tenant_id
        auth_hooks.post_auth(request, tenant_id, roles, user_id)
        return await call_next(request)

    if mode == "client_token":
        token = _extract_bearer(request)
        if not token:
            return JSONResponse({"detail": "Missing client auth token."}, status_code=401)
        hook_result = auth_hooks.resolve_client_token(token)
        if hook_result:
            tenant_id, roles, user_id = hook_result
        request.state.user_id = user_id
        request.state.roles = roles
        request.state.tenant_id = tenant_id
        auth_hooks.post_auth(request, tenant_id, roles, user_id)
        return await call_next(request)

    return await call_next(request)
