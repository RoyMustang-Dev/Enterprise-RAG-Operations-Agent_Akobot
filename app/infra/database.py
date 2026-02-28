"""
SQLite Persistence Layer for Crawler & Analytics.
"""

import sqlite3
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Place DB in the repo-level data directory
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = str(DATA_DIR / "crawler_data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS crawled_pages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        url TEXT,
        title TEXT,
        content TEXT,
        depth INTEGER,
        status TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

def init_ingestion_db():
    """Ensures the ingestion job tracking table exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ingestion_jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT,
        payload_json TEXT,
        updated_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def init_chat_history_db():
    """Ensures the chat history table exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    conn.close()

def init_ephemeral_collections_db():
    """Ensures the ephemeral collections table exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ephemeral_collections (
        collection_name TEXT PRIMARY KEY,
        created_at REAL
    )
    """)
    conn.commit()
    conn.close()

def init_session_collections_db():
    """Ensures the session->collection mapping table exists."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS session_collections (
        session_id TEXT PRIMARY KEY,
        collection_name TEXT,
        created_at REAL,
        updated_at REAL
    )
    """)
    conn.commit()
    conn.close()

def enable_wal():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()

def insert_page_async(session_id, url, title, content, depth, status):
    """
    Called ONLY by DB writer coroutine.
    Never from crawler workers.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    ts = datetime.now().isoformat()

    c.execute("""
    INSERT INTO crawled_pages
    (session_id, url, title, content, depth, status, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, url, title, content, depth, status, ts))

    conn.commit()
    conn.close()

def upsert_ingestion_job(job_id: str, payload: dict, status: str = "pending"):
    """Insert or update ingestion job state."""
    init_ingestion_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.now().isoformat()
    payload_json = json.dumps(payload)
    c.execute("""
    INSERT INTO ingestion_jobs (job_id, status, payload_json, updated_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(job_id) DO UPDATE SET
        status=excluded.status,
        payload_json=excluded.payload_json,
        updated_at=excluded.updated_at
    """, (job_id, status, payload_json, ts))
    conn.commit()
    conn.close()

def get_ingestion_job(job_id: str):
    """Fetch a single ingestion job by id."""
    init_ingestion_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM ingestion_jobs WHERE job_id=?", (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
    payload["job_id"] = row["job_id"]
    payload["status"] = row["status"]
    payload["updated_at"] = row["updated_at"]
    return payload

def save_chat_turn(session_id: str, role: str, content: str):
    """Persist a single chat turn."""
    init_chat_history_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, ts),
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str, limit: int = 20):
    """Fetch recent chat turns for a session."""
    init_chat_history_db()
    try:
        limit = int(os.getenv("CHAT_HISTORY_LIMIT", limit))
    except Exception:
        limit = limit
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT role, content, timestamp FROM chat_history WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]

def record_ephemeral_collection(collection_name: str, created_at: float):
    init_ephemeral_collections_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO ephemeral_collections (collection_name, created_at) VALUES (?, ?)",
        (collection_name, created_at),
    )
    conn.commit()
    conn.close()

def upsert_session_collection(session_id: str, collection_name: str, created_at: float = None, updated_at: float = None):
    init_session_collections_db()
    now = time.time()
    created_at = created_at or now
    updated_at = updated_at or now
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO session_collections (session_id, collection_name, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            collection_name=excluded.collection_name,
            updated_at=excluded.updated_at
        """,
        (session_id, collection_name, created_at, updated_at),
    )
    conn.commit()
    conn.close()

def get_session_collection(session_id: str):
    init_session_collections_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM session_collections WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "session_id": row["session_id"],
        "collection_name": row["collection_name"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def delete_session_collection_by_collection(collection_name: str):
    init_session_collections_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM session_collections WHERE collection_name=?", (collection_name,))
    conn.commit()
    conn.close()

def list_expired_collections(ttl_hours: int):
    init_ephemeral_collections_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    cutoff = time.time() - (ttl_hours * 3600)
    c.execute(
        "SELECT collection_name FROM ephemeral_collections WHERE created_at < ?",
        (cutoff,),
    )
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

def delete_ephemeral_collection_record(collection_name: str):
    init_ephemeral_collections_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM ephemeral_collections WHERE collection_name=?", (collection_name,))
    conn.commit()
    conn.close()

def get_all_pages(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM crawled_pages WHERE session_id=?", (session_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_latest_session_data(limit=50):
    """Fetches pages from the most recent session."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    # Get latest session_id
    c.execute("SELECT session_id FROM crawled_pages ORDER BY id DESC LIMIT 1")
    result = c.fetchone()
    if not result:
        return []
    
    latest_sid = result["session_id"]
    c.execute("SELECT * FROM crawled_pages WHERE session_id=? ORDER BY id ASC LIMIT ?", (latest_sid, limit))
    rows = c.fetchall()
    conn.close()
    return rows
