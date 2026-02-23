"""
SQLite Persistence Layer for Crawler & Analytics.
"""

import sqlite3
import os
from datetime import datetime

# Place DB in the data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "crawler_data.db")

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
