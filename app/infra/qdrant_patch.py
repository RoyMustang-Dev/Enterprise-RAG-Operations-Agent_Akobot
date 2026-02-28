"""
Qdrant Client Compatibility Patch

Ensures qdrant-client imports cleanly on Python 3.11 when grpc annotations
use the `|` union operator without `from __future__ import annotations`.
"""
import os
import importlib.util
import logging

logger = logging.getLogger(__name__)


def _add_future_annotations(path: str) -> bool:
    try:
        if not os.path.exists(path):
            return False
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if "from __future__ import annotations" in content:
            return False
        content = "from __future__ import annotations\n" + content
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        logger.warning(f"[QDRANT PATCH] Failed to patch {path}: {e}")
        return False


def ensure_qdrant_grpc_compat():
    """
    Patch qdrant-client grpc uploader modules to avoid runtime annotation evaluation errors.
    Safe to call multiple times.
    """
    try:
        spec = importlib.util.find_spec("qdrant_client")
        if not spec or not spec.submodule_search_locations:
            return
        base = spec.submodule_search_locations[0]
        candidates = [
            os.path.join(base, "uploader", "grpc_uploader.py"),
            os.path.join(base, "async_qdrant_remote.py"),
        ]
        patched_any = False
        for path in candidates:
            patched_any = _add_future_annotations(path) or patched_any
        if patched_any:
            logger.info("[QDRANT PATCH] Applied __future__ annotations patch for grpc modules.")
    except Exception as e:
        logger.warning(f"[QDRANT PATCH] Skipped patch due to error: {e}")
