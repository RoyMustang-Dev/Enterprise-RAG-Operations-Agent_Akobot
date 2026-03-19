"""
SQLite Persistence Layer for Crawler & Analytics.
"""

import sqlite3
import os
import time
import json
import time
from datetime import datetime
from pathlib import Path
import logging
from contextvars import ContextVar

# Place DBs in the repo-level data directory
REPO_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_DIR / "data"
TENANT_DIR = DATA_DIR / "tenants"
APP_DB_NAME = "app_data.db"
CRAWLER_DB_NAME = "crawler_data.db"
logger = logging.getLogger(__name__)

_CURRENT_TENANT: ContextVar[str] = ContextVar("CURRENT_TENANT_ID", default="global")

def set_current_tenant(tenant_id: str | None):
    if tenant_id:
        _CURRENT_TENANT.set(tenant_id)
    else:
        _CURRENT_TENANT.set("global")
    paths = get_db_paths(tenant_id)
    logger.info(
        "[DB] Tenant=%s app_db=%s crawler_db=%s",
        paths.get("tenant"),
        paths.get("app_db"),
        paths.get("crawler_db"),
    )

def get_current_tenant() -> str:
    return _CURRENT_TENANT.get() or "global"

def _sanitize_tenant(tenant_id: str | None) -> str:
    if not tenant_id:
        return "global"
    safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in tenant_id])
    return safe[:64] or "global"

def _resolve_db_path(db_kind: str, tenant_id: str | None) -> str:
    tenant = _sanitize_tenant(tenant_id)
    base_dir = TENANT_DIR / tenant
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        marker = base_dir / ".tenant_created"
        if not marker.exists():
            marker.write_text(str(time.time()))
    except Exception:
        pass
    name = APP_DB_NAME if db_kind == "app" else CRAWLER_DB_NAME
    return str(base_dir / name)

def get_db_paths(tenant_id: str | None) -> dict:
    tenant = _sanitize_tenant(tenant_id)
    base_dir = TENANT_DIR / tenant
    return {
        "tenant": tenant,
        "base_dir": str(base_dir),
        "app_db": str(base_dir / APP_DB_NAME),
        "crawler_db": str(base_dir / CRAWLER_DB_NAME),
    }

def cleanup_expired_tenant_dbs(ttl_days: int = 30):
    """Delete tenant DB folders older than ttl_days based on creation marker."""
    now = time.time()
    if not TENANT_DIR.exists():
        return 0
    removed = 0
    for tenant_dir in TENANT_DIR.iterdir():
        if not tenant_dir.is_dir():
            continue
        marker = tenant_dir / ".tenant_created"
        try:
            if marker.exists():
                created_ts = float(marker.read_text().strip())
            else:
                created_ts = tenant_dir.stat().st_mtime
            age_days = (now - created_ts) / 86400
            if age_days >= ttl_days:
                # Safety: only delete known db files and marker
                for name in [APP_DB_NAME, CRAWLER_DB_NAME, ".tenant_created"]:
                    path = tenant_dir / name
                    if path.exists():
                        path.unlink(missing_ok=True)
                # Remove dir if empty
                try:
                    tenant_dir.rmdir()
                except Exception:
                    pass
                removed += 1
        except Exception:
            continue
    return removed

def purge_tenant_db(tenant_id: str) -> bool:
    """Safely delete a single tenant DB folder."""
    tenant = _sanitize_tenant(tenant_id)
    tenant_dir = TENANT_DIR / tenant
    if not tenant_dir.exists():
        return False
    try:
        for name in [APP_DB_NAME, CRAWLER_DB_NAME, ".tenant_created"]:
            path = tenant_dir / name
            if path.exists():
                path.unlink(missing_ok=True)
        try:
            tenant_dir.rmdir()
        except Exception:
            pass
        return True
    except Exception:
        return False

def _get_conn(db_kind: str = "app", tenant_id: str | None = None):
    """Robust connection helper avoiding OperationalErrors during high threading."""
    last_err = None
    tenant = tenant_id or get_current_tenant()
    db_path = _resolve_db_path(db_kind, tenant)
    for attempt in range(3):
        try:
            os.makedirs(str(DATA_DIR), exist_ok=True)
            return sqlite3.connect(db_path, timeout=15, check_same_thread=False)
        except sqlite3.OperationalError as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))
        except Exception as e:
            last_err = e
            time.sleep(0.2 * (attempt + 1))

    # Fallback to a writable user directory to prevent hard crashes.
    try:
        fallback_dir = Path.home() / ".enterprise_rag" / "tenants" / _sanitize_tenant(tenant)
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_db = str(fallback_dir / (APP_DB_NAME if db_kind == "app" else CRAWLER_DB_NAME))
        logger.error(f"[DB] Falling back to user DB path: {fallback_db} (root cause: {last_err})")
        return sqlite3.connect(fallback_db, timeout=15, check_same_thread=False)
    except Exception:
        raise last_err

def init_db(tenant_id: str | None = None):
    conn = _get_conn("crawler", tenant_id)
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

def init_ingestion_db(tenant_id: str | None = None):
    """Ensures the ingestion job tracking table exists."""
    conn = _get_conn("app", tenant_id)
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

def init_chat_history_db(tenant_id: str | None = None):
    """Ensures the chat history table exists."""
    conn = _get_conn("app", tenant_id)
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

def init_analytics_memory_db(tenant_id: str | None = None):
    """Ensures the analytics memory table exists."""
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS analytics_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        role TEXT,
        content TEXT,
        kpi_json TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def init_analytics_jobs_db(tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS analytics_jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT,
        payload_json TEXT,
        updated_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def upsert_analytics_job(job_id: str, status: str, payload_json: str, tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    INSERT INTO analytics_jobs (job_id, status, payload_json, updated_at)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(job_id) DO UPDATE SET status=excluded.status, payload_json=excluded.payload_json, updated_at=excluded.updated_at
    """, (job_id, status, payload_json, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def fetch_analytics_job(job_id: str, tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("SELECT status, payload_json, updated_at FROM analytics_jobs WHERE job_id = ?", (job_id,))
    row = c.fetchone()
    conn.close()
    return row

def insert_analytics_memory(session_id: str, role: str, content: str, kpi_json: str = "", tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "INSERT INTO analytics_memory (session_id, role, content, kpi_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, kpi_json, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def fetch_analytics_memory(session_id: str, limit: int = 6, tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "SELECT role, content, kpi_json FROM analytics_memory WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = c.fetchall()
    conn.close()
    return list(reversed(rows))

def init_ephemeral_collections_db(tenant_id: str | None = None):
    """Ensures the ephemeral collections table exists."""
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS ephemeral_collections (
        collection_name TEXT PRIMARY KEY,
        created_at REAL
    )
    """)
    conn.commit()
    conn.close()

def init_session_collections_db(tenant_id: str | None = None):
    """Ensures the session->collection mapping table exists."""
    conn = _get_conn("app", tenant_id)
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

def init_seen_urls_db(tenant_id: str | None = None):
    """Ensures the seen URLs table exists for crawler de-dup across runs."""
    conn = _get_conn("crawler", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS seen_urls (
        url TEXT PRIMARY KEY,
        domain TEXT,
        first_seen TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_seen_urls(domain: str, limit: int = 50000, tenant_id: str | None = None):
    """Fetch a bounded set of previously seen URLs for a domain."""
    init_seen_urls_db(tenant_id)
    conn = _get_conn("crawler", tenant_id)
    c = conn.cursor()
    c.execute(
        "SELECT url FROM seen_urls WHERE domain=? ORDER BY first_seen DESC LIMIT ?",
        (domain, int(limit)),
    )
    rows = c.fetchall()
    conn.close()
    return {r[0] for r in rows}

def record_seen_url(url: str, domain: str, tenant_id: str | None = None):
    """Persist a seen URL (idempotent)."""
    if not url:
        return
    init_seen_urls_db(tenant_id)
    conn = _get_conn("crawler", tenant_id)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT OR IGNORE INTO seen_urls (url, domain, first_seen) VALUES (?, ?, ?)",
        (url, domain, ts),
    )
    conn.commit()
    conn.close()

def init_session_cache_db(tenant_id: str | None = None):
    """Ensures the session cache table exists."""
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS session_cache (
        session_id TEXT PRIMARY KEY,
        cache_json TEXT,
        last_query_hash TEXT,
        created_at REAL,
        updated_at REAL
    )
    """)
    conn.commit()
    conn.close()


def init_email_audit_db(tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS email_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT,
        recipients TEXT,
        subject TEXT,
        status TEXT,
        error TEXT,
        created_at REAL
    )
    """)
    conn.commit()
    conn.close()


def record_email_audit(tenant_id: str, recipients: str, subject: str, status: str, error: str = ""):
    init_email_audit_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "INSERT INTO email_audit (tenant_id, recipients, subject, status, error, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (tenant_id, recipients, subject, status, error or "", time.time()),
    )
    conn.commit()
    conn.close()


def count_recent_emails(tenant_id: str, window_seconds: int) -> int:
    init_email_audit_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    cutoff = time.time() - window_seconds
    c.execute(
        "SELECT COUNT(1) FROM email_audit WHERE tenant_id = ? AND created_at >= ?",
        (tenant_id, cutoff),
    )
    row = c.fetchone()
    conn.close()
    return int(row[0]) if row else 0


def init_tool_audit_db(tenant_id: str | None = None):
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS tool_audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tenant_id TEXT,
        tool_slug TEXT,
        status TEXT,
        error TEXT,
        created_at REAL
    )
    """)
    conn.commit()
    conn.close()


def record_tool_audit(tenant_id: str, tool_slug: str, status: str, error: str = ""):
    init_tool_audit_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "INSERT INTO tool_audit (tenant_id, tool_slug, status, error, created_at) VALUES (?, ?, ?, ?, ?)",
        (tenant_id, tool_slug, status, error or "", time.time()),
    )
    conn.commit()
    conn.close()

def enable_wal(db_kind: str = "crawler", tenant_id: str | None = None):
    conn = _get_conn(db_kind, tenant_id)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.close()

def insert_page_async(session_id, url, title, content, depth, status, tenant_id: str | None = None):
    """
    Called ONLY by DB writer coroutine.
    Never from crawler workers.
    """
    conn = _get_conn("crawler", tenant_id)
    c = conn.cursor()

    ts = datetime.now().isoformat()

    c.execute("""
    INSERT INTO crawled_pages
    (session_id, url, title, content, depth, status, timestamp)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (session_id, url, title, content, depth, status, ts))

    conn.commit()
    conn.close()

def upsert_ingestion_job(job_id: str, payload: dict, status: str = "pending", tenant_id: str | None = None):
    """Insert or update ingestion job state."""
    init_ingestion_db(tenant_id)
    conn = _get_conn("app", tenant_id)
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

def get_ingestion_job(job_id: str, tenant_id: str | None = None):
    """Fetch a single ingestion job by id."""
    init_ingestion_db(tenant_id)
    conn = _get_conn("app", tenant_id)
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

def save_chat_turn(session_id: str, role: str, content: str, tenant_id: str | None = None):
    """Persist a single chat turn."""
    init_chat_history_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    ts = datetime.utcnow().isoformat()
    c.execute(
        "INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, ts),
    )
    conn.commit()
    conn.close()

def get_chat_history(session_id: str, limit: int = 20, tenant_id: str | None = None):
    """Fetch recent chat turns for a session."""
    init_chat_history_db(tenant_id)
    try:
        limit = int(os.getenv("CHAT_HISTORY_LIMIT", limit))
    except Exception:
        limit = limit
    conn = _get_conn("app", tenant_id)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT role, content, timestamp FROM chat_history WHERE session_id=? ORDER BY id ASC LIMIT ?",
        (session_id, limit),
    )
    rows = c.fetchall()
    conn.close()
    return [{"role": r["role"], "content": r["content"], "timestamp": r["timestamp"]} for r in rows]

def record_ephemeral_collection(collection_name: str, created_at: float, tenant_id: str | None = None):
    init_ephemeral_collections_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO ephemeral_collections (collection_name, created_at) VALUES (?, ?)",
        (collection_name, created_at),
    )
    conn.commit()
    conn.close()

def upsert_session_collection(session_id: str, collection_name: str, created_at: float = None, updated_at: float = None, tenant_id: str | None = None):
    init_session_collections_db(tenant_id)
    now = time.time()
    created_at = created_at or now
    updated_at = updated_at or now
    conn = _get_conn("app", tenant_id)
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

def get_session_collection(session_id: str, tenant_id: str | None = None):
    init_session_collections_db(tenant_id)
    conn = _get_conn("app", tenant_id)
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

def get_session_cache(session_id: str, tenant_id: str | None = None):
    init_session_cache_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM session_cache WHERE session_id=?", (session_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    cache_payload = json.loads(row["cache_json"]) if row["cache_json"] else {}
    return {
        "session_id": row["session_id"],
        "cache": cache_payload,
        "last_query_hash": row["last_query_hash"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }

def upsert_session_cache(session_id: str, cache_payload: dict, last_query_hash: str = None, created_at: float = None, updated_at: float = None, tenant_id: str | None = None):
    init_session_cache_db(tenant_id)
    now = time.time()
    created_at = created_at or now
    updated_at = updated_at or now
    cache_json = json.dumps(cache_payload)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute(
        """
        INSERT INTO session_cache (session_id, cache_json, last_query_hash, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            cache_json=excluded.cache_json,
            last_query_hash=excluded.last_query_hash,
            updated_at=excluded.updated_at
        """,
        (session_id, cache_json, last_query_hash, created_at, updated_at),
    )
    conn.commit()
    conn.close()

def delete_session_cache(session_id: str, tenant_id: str | None = None):
    init_session_cache_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("DELETE FROM session_cache WHERE session_id=?", (session_id,))
    conn.commit()
    conn.close()

def delete_session_collection_by_collection(collection_name: str, tenant_id: str | None = None):
    init_session_collections_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("DELETE FROM session_collections WHERE collection_name=?", (collection_name,))
    conn.commit()
    conn.close()

def list_expired_collections(ttl_hours: int, tenant_id: str | None = None):
    init_ephemeral_collections_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    cutoff = time.time() - (ttl_hours * 3600)
    c.execute(
        "SELECT collection_name FROM ephemeral_collections WHERE created_at < ?",
        (cutoff,),
    )
    rows = c.fetchall()
    conn.close()
    return [r[0] for r in rows]

def delete_ephemeral_collection_record(collection_name: str, tenant_id: str | None = None):
    init_ephemeral_collections_db(tenant_id)
    conn = _get_conn("app", tenant_id)
    c = conn.cursor()
    c.execute("DELETE FROM ephemeral_collections WHERE collection_name=?", (collection_name,))
    conn.commit()
    conn.close()

def get_all_pages(session_id, tenant_id: str | None = None):
    conn = _get_conn("crawler", tenant_id)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM crawled_pages WHERE session_id=?", (session_id,))
    rows = c.fetchall()
    conn.close()
    return rows

def get_latest_session_data(limit=50, tenant_id: str | None = None):
    """Fetches pages from the most recent session."""
    conn = _get_conn("crawler", tenant_id)
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
