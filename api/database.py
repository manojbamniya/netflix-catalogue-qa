"""SQLite connection for the API. The database is built by ingest.py."""

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

DB_PATH = Path(os.getenv("NETFLIX_DB_PATH", "data/netflix.db"))


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open the database read-only, with rows that can be read by column name.

    check_same_thread=False because FastAPI may open the connection in one
    thread and run the endpoint in another; each request still uses its own
    connection, one step at a time.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_connection() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one connection per request, closed afterwards."""
    conn = open_connection(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()
