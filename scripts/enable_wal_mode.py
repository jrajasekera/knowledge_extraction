#!/usr/bin/env python3
"""Enable WAL mode on SQLite database for better concurrent access.

WAL (Write-Ahead Logging) mode allows multiple readers and one writer to access
the database concurrently, preventing "database is locked" errors.

Usage:
    uv run python scripts/enable_wal_mode.py [--db-path ./discord.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def enable_wal_mode(db_path: Path) -> None:
    """Enable WAL mode and optimize settings for concurrent access.

    Args:
        db_path: Path to the SQLite database file.
    """
    if not db_path.exists():
        print(f"Error: Database file not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check current journal mode
    cursor.execute("PRAGMA journal_mode;")
    current_mode = cursor.fetchone()[0]
    print(f"Current journal mode: {current_mode}")

    if current_mode.lower() == "wal":
        print("WAL mode is already enabled.")
    else:
        # Enable WAL mode
        cursor.execute("PRAGMA journal_mode=WAL;")
        new_mode = cursor.fetchone()[0]
        print(f"Enabled WAL mode: {new_mode}")

    # Configure additional settings for better concurrency
    cursor.execute("PRAGMA synchronous=NORMAL;")  # Faster writes while still safe in WAL mode
    cursor.execute("PRAGMA busy_timeout=5000;")    # Wait up to 5 seconds when database is busy
    cursor.execute("PRAGMA wal_autocheckpoint=1000;")  # Checkpoint every 1000 pages

    # Verify settings
    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]
    cursor.execute("PRAGMA synchronous;")
    synchronous = cursor.fetchone()[0]
    cursor.execute("PRAGMA busy_timeout;")
    busy_timeout = cursor.fetchone()[0]

    print(f"\nFinal configuration:")
    print(f"  journal_mode: {journal_mode}")
    print(f"  synchronous: {synchronous}")
    print(f"  busy_timeout: {busy_timeout}ms")

    conn.close()
    print(f"\n✓ Successfully configured {db_path} for concurrent access")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enable WAL mode on SQLite database for concurrent access"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("./discord.db"),
        help="Path to SQLite database (default: ./discord.db)",
    )

    args = parser.parse_args()
    enable_wal_mode(args.db_path)


if __name__ == "__main__":
    main()
