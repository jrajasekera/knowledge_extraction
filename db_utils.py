"""Database connection utilities for SQLite with concurrent access support."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


def get_sqlite_connection(
    db_path: str | Path,
    *,
    timeout: float = 30.0,
    read_only: bool = False,
    **kwargs: Any,
) -> sqlite3.Connection:
    """Get a SQLite connection configured for concurrent access.

    This function creates a connection with WAL mode and appropriate timeouts
    to prevent "database is locked" errors when multiple processes access the
    database concurrently.

    Args:
        db_path: Path to the SQLite database file.
        timeout: Connection timeout in seconds (default: 30.0).
        read_only: If True, open connection in read-only mode.
        **kwargs: Additional arguments to pass to sqlite3.connect().

    Returns:
        A configured SQLite connection.

    Example:
        >>> conn = get_sqlite_connection("./discord.db")
        >>> cursor = conn.execute("SELECT COUNT(*) FROM message")
        >>> conn.close()
    """
    uri = False
    if read_only:
        # For read-only connections, use URI mode
        db_path = f"file:{db_path}?mode=ro"
        uri = True

    conn = sqlite3.connect(str(db_path), timeout=timeout, uri=uri, **kwargs)

    # Enable foreign key constraints
    conn.execute("PRAGMA foreign_keys = ON")

    # Configure for concurrent access (WAL mode)
    # Note: WAL mode is persistent, so this is safe to call every time
    conn.execute("PRAGMA journal_mode = WAL")

    # Set busy timeout (how long to wait when database is locked)
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")

    # Configure for better performance in WAL mode
    conn.execute("PRAGMA synchronous = NORMAL")

    return conn


def init_wal_mode(db_path: str | Path) -> None:
    """Initialize WAL mode on a database.

    This should be called once when setting up a new database or migrating
    an existing one. WAL mode is persistent across connections.

    Args:
        db_path: Path to the SQLite database file.
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA wal_autocheckpoint = 1000")
    conn.close()
