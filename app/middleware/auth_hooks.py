"""
Auth Hook Extensions — JWT Implementation

Supports HS256 and RS256 JWT tokens for client authentication.

Environment Variables for JWT Mode:
  AUTH_MODE=client_token        — Enable JWT validation (default: none = disabled)
  JWT_ALGORITHM=HS256           — HS256 (shared secret) or RS256 (public key)
  JWT_SECRET=your-secret-here   — Required for HS256
  JWT_PUBLIC_KEY=...            — Required for RS256 (PEM format, or base64-encoded PEM)
  JWT_AUDIENCE=your-api         — Optional: validates JWT 'aud' claim
  JWT_ISSUER=your-issuer        — Optional: validates JWT 'iss' claim

Expected JWT Payload Claims:
  {
    "sub": "user-123",         → maps to user_id
    "tenant_id": "acme-corp",  → maps to tenant_id (also accepts "tenant", "org")
    "role": "admin",           → maps to role (also accepts "roles") [default: "member"]
    "exp": 1700000000,         → expiry (validated automatically)
    "iat": 1699990000,         → issued-at (optional)
    "aud": "your-api",         → audience (validated if JWT_AUDIENCE is set)
    "iss": "your-issuer"       → issuer (validated if JWT_ISSUER is set)
  }

Platform Key Mode (AUTH_MODE=platform_token):
  PLATFORM_API_KEYS=key1:tenantA:admin,key2:tenantB:member

Client Dev Integration Guide:
  1. Set AUTH_MODE=client_token in your server .env
  2. Set JWT_ALGORITHM=HS256 and JWT_SECRET=<random 256-bit string>
  3. Your backend mints JWTs with the payload above and signs with the secret
  4. Client sends: Authorization: Bearer <jwt> on every API request
  5. The middleware extracts tenant_id, role, user_id transparently
  6. No changes needed in the API routes — auth is middleware-level

For RS256 (asymmetric, more secure for multi-service):
  1. Generate key pair: openssl genrsa -out private.pem 2048 && openssl rsa -in private.pem -pubout -out public.pem
  2. Set JWT_ALGORITHM=RS256, JWT_PUBLIC_KEY=$(cat public.pem)
  3. Sign JWTs on your auth server with the private key
  4. The API validates with the public key only — private key never leaves your auth server

Development Mode (AUTH_MODE=none):
  All requests are allowed. Pass x-tenant-id header for tenant scoping.
  No JWT required in development.
"""
from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _get_jwt_config() -> dict:
    return {
        "algorithm": os.getenv("JWT_ALGORITHM", "HS256"),
        "secret": os.getenv("JWT_SECRET", ""),
        "public_key": os.getenv("JWT_PUBLIC_KEY", ""),
        "audience": os.getenv("JWT_AUDIENCE") or None,
        "issuer": os.getenv("JWT_ISSUER") or None,
    }


def resolve_client_token(token: str) -> Optional[Tuple[str, str, str]]:
    """
    Validates a JWT bearer token and extracts (tenant_id, role, user_id).

    Returns:
        (tenant_id, role, user_id) if valid
        None if token is invalid (middleware will reject with 401)

    Raises:
        Nothing — all exceptions are caught and logged; invalid tokens return None.
    """
    if not token:
        return None

    cfg = _get_jwt_config()
    algorithm = cfg["algorithm"]

    # Choose signing key
    if algorithm == "HS256":
        key = cfg["secret"]
        if not key:
            logger.error(
                "[AUTH] JWT_ALGORITHM=HS256 but JWT_SECRET is not set. "
                "Set JWT_SECRET in .env or switch to AUTH_MODE=none for development."
            )
            return None
    elif algorithm == "RS256":
        key = cfg["public_key"]
        if not key:
            logger.error("[AUTH] JWT_ALGORITHM=RS256 but JWT_PUBLIC_KEY is not set.")
            return None
        # Support base64-encoded PEM
        if not key.strip().startswith("-----"):
            import base64
            try:
                key = base64.b64decode(key).decode("utf-8")
            except Exception:
                logger.error("[AUTH] JWT_PUBLIC_KEY is not valid PEM or base64-encoded PEM.")
                return None
    else:
        logger.error(f"[AUTH] Unsupported JWT_ALGORITHM: {algorithm}. Use HS256 or RS256.")
        return None

    try:
        from jose import jwt as jose_jwt, JWTError
        options = {"verify_exp": True, "verify_iat": True}
        kwargs = {"algorithms": [algorithm], "options": options}
        if cfg["audience"]:
            kwargs["audience"] = cfg["audience"]
        if cfg["issuer"]:
            kwargs["issuer"] = cfg["issuer"]

        payload = jose_jwt.decode(token, key, **kwargs)

        # Extract claims (support multiple field names for flexibility)
        user_id = str(
            payload.get("sub") or payload.get("user_id") or payload.get("userId") or "anonymous"
        )
        tenant_id = str(
            payload.get("tenant_id") or payload.get("tenant") or
            payload.get("org") or payload.get("organization") or "global"
        )
        role = str(
            payload.get("role") or
            (payload.get("roles", ["member"])[0] if isinstance(payload.get("roles"), list) else None) or
            "member"
        )

        logger.info(f"[AUTH] JWT validated: user={user_id} tenant={tenant_id} role={role}")
        return (tenant_id, role, user_id)

    except ImportError:
        logger.error(
            "[AUTH] python-jose is not installed. Install it: pip install python-jose[cryptography]"
        )
        return None
    except Exception as e:
        logger.warning(f"[AUTH] JWT validation failed: {type(e).__name__}: {e}")
        return None


def resolve_platform_key(api_key: str) -> Optional[Tuple[str, str, str]]:
    """
    Optionally map a platform API key to (tenant_id, role, user_id).
    Return None to defer to PLATFORM_API_KEYS env mapping.
    Implement custom logic here (e.g., DB lookup) if needed.
    """
    return None


def post_auth(request, tenant_id: str, role: str, user_id: str) -> None:
    """
    Optional callback after successful auth resolution.
    Use for audit logging, rate limit injection, or request enrichment.
    """
    logger.debug(f"[AUTH] post_auth: tenant={tenant_id} role={role} user={user_id} path={getattr(request, 'url', {})}")
    return None
