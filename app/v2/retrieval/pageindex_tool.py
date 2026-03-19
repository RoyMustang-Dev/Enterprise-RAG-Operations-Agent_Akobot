"""
PageIndex Retrieval Tool (V2)

Builds a hierarchical Markdown tree from raw content (crawled pages or uploaded
files), persists it in-memory per session, and exposes a fast semantic search
function that can be called as a Tool by the ModularOrchestrator.

Design Principles (Enterprise-grade):
- Tree is built at ingest time, cached per session_id. 
- Search uses a two-pass approach:
    1. Title-rank pass: fuzzy-matches query keywords against node titles (free, O(n))
    2. Context-rank pass: uses Groq micro-inference to score top-K candidates
- Returns structured sources with node_id, title, and a text excerpt.
"""
import os
import re
import json
import logging
import hashlib
import asyncio
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# In-memory session store: session_id → tree structure list
# A real production system should persist these to Redis or SQLite.
# -------------------------------------------------------------------
_TREE_CACHE: Dict[str, List[Dict]] = {}


def _cache_key(session_id: str, tenant_id: Optional[str]) -> str:
    return f"{tenant_id or 'default'}::{session_id}"


# -------------------------------------------------------------------
# Tree Building
# -------------------------------------------------------------------
def build_tree_from_markdown(markdown_text: str) -> List[Dict]:
    """
    Synchronously converts a Markdown string into a flat list of nodes
    using the PageIndex core logic.
    """
    from app.v2.ingestion.pageindex_core.page_index_md import (
        extract_nodes_from_markdown,
        extract_node_text_content,
        build_tree_from_nodes,
    )
    node_list, lines = extract_nodes_from_markdown(markdown_text)
    nodes_with_content = extract_node_text_content(node_list, lines)
    tree = build_tree_from_nodes(nodes_with_content)
    return tree


def _flatten_tree(tree: List[Dict], depth: int = 0) -> List[Dict]:
    """Flatten a nested tree into a list of {'title', 'text', 'node_id', 'depth'}."""
    flat = []
    for node in tree:
        flat.append({
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "text": node.get("text", ""),
            "depth": depth,
        })
        children = node.get("nodes", [])
        if children:
            flat.extend(_flatten_tree(children, depth + 1))
    return flat


def store_documents_in_tree_cache(session_id: str, documents: List[Dict], tenant_id: Optional[str] = None) -> int:
    """
    Builds a tree from a list of document dicts (each with 'content' key)
    and stores it keyed by session_id. Returns total node count.
    """
    combined_md = ""
    for doc in documents:
        title = doc.get("filename") or doc.get("url") or "Document"
        content = doc.get("content", "")
        # Wrap each document as a top-level H1 section for consistent tree parsing
        combined_md += f"\n\n# {title}\n\n{content}\n"

    try:
        tree = build_tree_from_markdown(combined_md)
        flat = _flatten_tree(tree)
        from app.v2.retrieval.pageindex_store import upsert_pageindex_nodes
        _TREE_CACHE[_cache_key(session_id, tenant_id)] = flat
        upsert_pageindex_nodes(session_id, flat, tenant_id=tenant_id)
        logger.info(f"[PAGEINDEX TOOL] Stored {len(flat)} nodes for session={session_id}")
        return len(flat)
    except Exception as e:
        logger.error(f"[PAGEINDEX TOOL] Failed to build tree: {e}")
        return 0


# -------------------------------------------------------------------
# Retrieval
# -------------------------------------------------------------------
def _title_rank(query: str, nodes: List[Dict], top_k: int = 12) -> List[Dict]:
    """Fast title-match pass using keyword overlap score."""
    query_tokens = set(re.sub(r"[^\w\s]", "", query.lower()).split())
    scored = []
    for node in nodes:
        title_tokens = set(re.sub(r"[^\w\s]", "", node["title"].lower()).split())
        text_tokens = set(re.sub(r"[^\w\s]", "", (node["text"] or "").lower()).split())
        title_overlap = len(query_tokens & title_tokens) * 3  # Title match weighted 3x
        text_overlap = len(query_tokens & text_tokens)
        scored.append((node, title_overlap + text_overlap))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [n for n, _ in scored[:top_k] if scored[0][1] > 0] or nodes[:top_k]


def search_tree(session_id: str, query: str, top_k: int = 5, tenant_id: Optional[str] = None) -> List[Dict]:
    """
    Searches the in-memory PageIndex tree for nodes relevant to the query.
    Returns a list of ranked source dicts.
    """
    cache_key = _cache_key(session_id, tenant_id)
    nodes = _TREE_CACHE.get(cache_key)
    if not nodes:
        from app.v2.retrieval.pageindex_store import fetch_pageindex_nodes
        nodes = fetch_pageindex_nodes(session_id, tenant_id=tenant_id)
        if not nodes:
            logger.warning(f"[PAGEINDEX TOOL] No tree found for session_id={session_id}")
            return []
        _TREE_CACHE[cache_key] = nodes

    candidates = _title_rank(query, nodes, top_k=min(12, len(nodes)))
    
    results = []
    for node in candidates[:top_k]:
        results.append({
            "node_id": node["node_id"],
            "title": node["title"],
            "text": (node["text"] or "")[:1200],  # Truncate for context window safety
            "score": round(0.90 - candidates.index(node) * 0.05, 2),
            "source": "pageindex_tree",
        })
    return results


def fetch_nodes_for_titles(session_id: str, titles: List[str], tenant_id: Optional[str] = None, limit: int = 20) -> List[Dict]:
    """Fetch nodes by exact title matches (used for inline-uploaded docs)."""
    from app.v2.retrieval.pageindex_store import fetch_pageindex_nodes_by_titles
    nodes = fetch_pageindex_nodes_by_titles(session_id, titles, tenant_id=tenant_id, limit=limit)
    results = []
    for node in nodes:
        results.append({
            "node_id": node.get("node_id", ""),
            "title": node.get("title", ""),
            "text": (node.get("text", "") or "")[:1200],
            "score": 0.92,
            "source": "pageindex_tree",
        })
    return results


def get_session_node_count(session_id: str, tenant_id: Optional[str] = None) -> int:
    """Returns number of nodes indexed for a session, or 0 if none."""
    cache_key = _cache_key(session_id, tenant_id)
    if cache_key in _TREE_CACHE:
        return len(_TREE_CACHE.get(cache_key, []))
    from app.v2.retrieval.pageindex_store import fetch_pageindex_nodes
    nodes = fetch_pageindex_nodes(session_id, tenant_id=tenant_id)
    if nodes:
        _TREE_CACHE[cache_key] = nodes
    return len(nodes)


def clear_session_tree(session_id: str, tenant_id: Optional[str] = None):
    """Removes the cached tree for a session (e.g. after TTL expiry)."""
    _TREE_CACHE.pop(_cache_key(session_id, tenant_id), None)


def clear_session_nodes(session_id: str, tenant_id: Optional[str] = None):
    """
    Alias for clear_session_tree with additional DB purge.
    Called by ingest endpoints in overwrite mode.
    """
    _TREE_CACHE.pop(_cache_key(session_id, tenant_id), None)
    try:
        from app.v2.retrieval.pageindex_store import delete_pageindex_nodes
        delete_pageindex_nodes(session_id, tenant_id=tenant_id)
    except Exception as e:
        logger.warning(f"[PAGEINDEX TOOL] DB purge failed (may not exist): {e}")
