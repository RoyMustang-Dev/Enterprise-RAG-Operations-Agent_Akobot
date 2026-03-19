"""
Lightweight WAF-like Request Guard

Provides basic safeguards (size and pattern checks). Enable via WAF_ENABLED=true.
"""
from __future__ import annotations

import os
import re
from fastapi import Request
from starlette.responses import JSONResponse


def _waf_enabled() -> bool:
    return os.getenv("WAF_ENABLED", "false").lower() == "true"


async def waf_middleware(request: Request, call_next):
    if not _waf_enabled():
        return await call_next(request)

    max_bytes = int(os.getenv("WAF_MAX_BODY_BYTES", "1048576"))
    body = await request.body()
    if len(body) > max_bytes:
        return JSONResponse({"detail": "Request too large."}, status_code=413)

    # Simple pattern checks for obvious abuse
    text = body.decode("utf-8", errors="ignore")
    patterns = [
        r"(?i)\bunion\s+select\b",
        r"(?i)\bdrop\s+table\b",
        r"(?i)\b<\s*script\b",
    ]
    for pat in patterns:
        if re.search(pat, text):
            return JSONResponse({"detail": "Request blocked by WAF."}, status_code=403)

    return await call_next(request)
