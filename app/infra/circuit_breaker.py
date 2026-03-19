"""
Simple Circuit Breaker for LLM Providers
"""
from __future__ import annotations

import time
from typing import Dict


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if (time.time() - self.opened_at) > self.recovery_seconds:
            self.opened_at = None
            self.failures = 0
            return True
        return False

    def record_success(self):
        self.failures = 0
        self.opened_at = None

    def record_failure(self):
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()


_breakers: Dict[str, CircuitBreaker] = {}


def get_breaker(provider: str, failure_threshold: int = 3, recovery_seconds: int = 30, agent: str | None = None) -> CircuitBreaker:
    key = f"{agent or 'global'}::{provider}"
    if key not in _breakers:
        _breakers[key] = CircuitBreaker(failure_threshold, recovery_seconds)
    return _breakers[key]
