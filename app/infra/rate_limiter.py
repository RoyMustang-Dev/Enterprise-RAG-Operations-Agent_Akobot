"""
Simple rate limiter with Redis support.
Falls back to in-memory counters when Redis is unavailable.
"""
from __future__ import annotations

import time
import threading
from typing import Tuple

from app.infra.redis_client import get_redis_client

_lock = threading.Lock()
_local_buckets: dict[str, list[float]] = {}


def enforce_rate_limit(key: str, limit: int, window_seconds: int) -> Tuple[bool, int, int]:
    """
    Returns (allowed, remaining, reset_seconds).
    """
    if limit <= 0 or window_seconds <= 0:
        return True, limit, window_seconds

    client = get_redis_client()
    if client:
        bucket_key = f"rate:{key}:{window_seconds}"
        current = client.incr(bucket_key)
        if current == 1:
            client.expire(bucket_key, window_seconds)
        remaining = max(0, limit - int(current))
        ttl = client.ttl(bucket_key)
        allowed = int(current) <= limit
        return allowed, remaining, max(0, int(ttl))

    # In-memory fallback
    now = time.time()
    with _lock:
        timestamps = _local_buckets.get(key, [])
        cutoff = now - window_seconds
        timestamps = [t for t in timestamps if t > cutoff]
        allowed = len(timestamps) < limit
        if allowed:
            timestamps.append(now)
        _local_buckets[key] = timestamps
        remaining = max(0, limit - len(timestamps))
        reset_seconds = max(0, int(window_seconds - (now - min(timestamps))) if timestamps else window_seconds)
        return allowed, remaining, reset_seconds
