"""
Distributed Lock Helpers

Uses Redis to prevent duplicate ingestion/crawler/tool executions across workers.
"""
from __future__ import annotations

import os
import time
import logging
from contextlib import contextmanager
from typing import Optional

from app.infra.redis_client import get_redis_client

logger = logging.getLogger(__name__)


def _locks_enabled() -> bool:
    return os.getenv("REDIS_LOCKS_ENABLED", "false").lower() == "true"


@contextmanager
def distributed_lock(key: str, ttl_seconds: int = 600, wait_timeout: int = 0):
    """
    Acquire a Redis-based lock.
    If wait_timeout == 0, fail fast when lock is held.
    """
    if not _locks_enabled():
        yield True
        return

    client = get_redis_client()
    if not client:
        logger.warning("[LOCK] Redis unavailable; proceeding without lock.")
        yield True
        return

    lock_key = f"lock:{key}"
    start = time.time()
    acquired = False
    try:
        while True:
            acquired = client.set(lock_key, "1", nx=True, ex=ttl_seconds)
            if acquired:
                break
            if wait_timeout and (time.time() - start) < wait_timeout:
                time.sleep(0.1)
                continue
            break
        yield bool(acquired)
    finally:
        if acquired:
            try:
                client.delete(lock_key)
            except Exception as e:
                logger.warning(f"[LOCK] Failed to release lock {lock_key}: {e}")
