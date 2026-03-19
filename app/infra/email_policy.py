"""
Email policy enforcement: allow/deny list + rate limits + audit trail.
"""
from __future__ import annotations

import os
from typing import List, Dict, Any

from app.infra.database import count_recent_emails, record_email_audit


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _domain(email: str) -> str:
    if "@" not in email:
        return ""
    return email.split("@", 1)[1].lower().strip()


def check_email_policy(tenant_id: str, recipients: List[str]) -> Dict[str, Any]:
    allowlist = set(_split_csv(os.getenv("EMAIL_ALLOWLIST", "")))
    allow_domains = set(_split_csv(os.getenv("EMAIL_ALLOWLIST_DOMAINS", "")))
    denylist = set(_split_csv(os.getenv("EMAIL_DENYLIST", "")))
    deny_domains = set(_split_csv(os.getenv("EMAIL_DENYLIST_DOMAINS", "")))
    limit = int(os.getenv("EMAIL_RATE_LIMIT_PER_HOUR", "120"))

    normalized = [r.lower().strip() for r in recipients if r]
    for r in normalized:
        if r in denylist or _domain(r) in deny_domains:
            return {"allowed": False, "error": f"Recipient {r} is blocked by policy."}

    if allowlist or allow_domains:
        for r in normalized:
            if r not in allowlist and _domain(r) not in allow_domains:
                return {"allowed": False, "error": f"Recipient {r} not allowed by policy."}

    if limit > 0:
        sent = count_recent_emails(tenant_id, window_seconds=3600)
        if sent >= limit:
            return {"allowed": False, "error": "Email rate limit exceeded for this tenant."}

    return {"allowed": True}


def audit_email(tenant_id: str, recipients: List[str], subject: str, status: str, error: str = ""):
    record_email_audit(
        tenant_id=tenant_id,
        recipients=",".join(recipients),
        subject=subject or "",
        status=status,
        error=error or "",
    )
