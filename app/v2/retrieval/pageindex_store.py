"""
Persistent PageIndex Store (SQLite)
"""
from __future__ import annotations

import json
from typing import List, Dict, Any, Optional

from app.infra.database import _get_conn


def init_pageindex_db(tenant_id: Optional[str] = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS pageindex_nodes (
        session_id TEXT,
        node_id TEXT,
        title TEXT,
        text TEXT,
        depth INTEGER,
        PRIMARY KEY (session_id, node_id)
    )
    """)
    conn.commit()
    conn.close()


def upsert_pageindex_nodes(session_id: str, nodes: List[Dict[str, Any]], tenant_id: Optional[str] = None):
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("DELETE FROM pageindex_nodes WHERE session_id = ?", (session_id,))
    for node in nodes:
        c.execute(
            "INSERT INTO pageindex_nodes (session_id, node_id, title, text, depth) VALUES (?, ?, ?, ?, ?)",
            (session_id, node.get("node_id", ""), node.get("title", ""), node.get("text", ""), int(node.get("depth", 0))),
        )
    conn.commit()
    conn.close()


def insert_pageindex_nodes(session_id: str, nodes: List[Dict[str, Any]], tenant_id: Optional[str] = None):
    """
    Inserts nodes without deleting existing session nodes (append mode).
    """
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    for node in nodes:
        c.execute(
            "INSERT INTO pageindex_nodes (session_id, node_id, title, text, depth) VALUES (?, ?, ?, ?, ?)",
            (session_id, node.get("node_id", ""), node.get("title", ""), node.get("text", ""), int(node.get("depth", 0))),
        )
    conn.commit()
    conn.close()


def fetch_pageindex_nodes(session_id: str, tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("SELECT node_id, title, text, depth FROM pageindex_nodes WHERE session_id = ?", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"node_id": r[0], "title": r[1], "text": r[2], "depth": r[3]} for r in rows]


def fetch_pageindex_nodes_by_titles(session_id: str, titles: List[str], tenant_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches nodes matching any of the provided titles (exact match).
    Used to ensure inline-uploaded docs are always represented in sources.
    """
    if not titles:
        return []
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    conn.row_factory = None
    c = conn.cursor()
    placeholders = ",".join(["?"] * len(titles))
    c.execute(
        f"SELECT node_id, title, text, depth FROM pageindex_nodes WHERE session_id = ? AND title IN ({placeholders})",
        (session_id, *titles),
    )
    rows = c.fetchall()
    conn.close()
    nodes = [{"node_id": r[0], "title": r[1], "text": r[2], "depth": r[3]} for r in rows]
    return nodes[: max(1, int(limit))]


def fetch_pageindex_nodes_by_titles_like(
    session_id: str,
    titles: List[str],
    tenant_id: Optional[str] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fuzzy title match using LIKE (fallback for filename-based forced sources).
    """
    if not titles:
        return []
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    patterns = [f"%{t}%" for t in titles if t]
    clauses = " OR ".join(["title LIKE ?"] * len(patterns))
    c.execute(
        f"SELECT node_id, title, text, depth FROM pageindex_nodes WHERE session_id = ? AND ({clauses})",
        (session_id, *patterns),
    )
    rows = c.fetchall()
    conn.close()
    nodes = [{"node_id": r[0], "title": r[1], "text": r[2], "depth": r[3]} for r in rows]
    return nodes[: max(1, int(limit))]


def fetch_pageindex_session_stats(tenant_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns per-session node counts for PageIndex.
    """
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "SELECT session_id, COUNT(*) FROM pageindex_nodes GROUP BY session_id ORDER BY COUNT(*) DESC"
    )
    rows = c.fetchall()
    conn.close()
    return [{"session_id": r[0], "nodes_indexed": r[1]} for r in rows]


def delete_pageindex_nodes(session_id: str, tenant_id: Optional[str] = None) -> int:
    """
    Deletes all nodes for a session from the database.
    Used by overwrite mode in ingest endpoints.
    Returns the number of deleted rows.
    """
    init_pageindex_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("DELETE FROM pageindex_nodes WHERE session_id = ?", (session_id,))
    deleted = c.rowcount
    conn.commit()
    conn.close()
    return deleted
