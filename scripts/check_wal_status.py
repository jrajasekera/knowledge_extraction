#!/usr/bin/env python3
"""Check if WAL mode is properly enabled on the SQLite database.

Usage:
    uv run python scripts/check_wal_status.py [--db-path ./discord.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


def check_wal_status(db_path: Path) -> None:
    """Check WAL mode status and other important settings.

    Args:
        db_path: Path to the SQLite database file.
    """
    if not db_path.exists():
        print(f"❌ Error: Database file not found: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Check journal mode
    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0]

    # Check synchronous mode
    cursor.execute("PRAGMA synchronous;")
    synchronous = cursor.fetchone()[0]

    # Check busy timeout
    cursor.execute("PRAGMA busy_timeout;")
    busy_timeout = cursor.fetchone()[0]

    # Check if WAL files exist
    wal_file = db_path.parent / f"{db_path.name}-wal"
    shm_file = db_path.parent / f"{db_path.name}-shm"

    conn.close()

    print(f"\n{'=' * 60}")
    print("SQLite Database Configuration Check")
    print(f"{'=' * 60}")
    print(f"Database: {db_path}")
    print("\nConfiguration:")
    print(f"  journal_mode: {journal_mode}")
    print(f"  synchronous: {synchronous}")
    print(f"  busy_timeout: {busy_timeout}ms")
    print("\nWAL files:")
    print(f"  {wal_file.name}: {'✓ exists' if wal_file.exists() else '✗ not found'}")
    print(f"  {shm_file.name}: {'✓ exists' if shm_file.exists() else '✗ not found'}")

    # Overall status
    print(f"\n{'=' * 60}")
    if journal_mode.lower() == "wal":
        print("✓ WAL mode is ENABLED - concurrent access should work!")
        print("\n  You can now run:")
        print("  • Memory agent API (uvicorn)")
        print("  • Deduplication (deduplicate_facts.py)")
        print("  • IE runner (run_ie.py)")
        print("  ...all at the same time without locking issues.")
    else:
        print("❌ WAL mode is NOT enabled!")
        print("\nTo fix this, run:")
        print(f"  uv run python scripts/enable_wal_mode.py --db-path {db_path}")
        print("\nThis will enable WAL mode and allow concurrent access.")

    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check if WAL mode is enabled on SQLite database")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("./discord.db"),
        help="Path to SQLite database (default: ./discord.db)",
    )

    args = parser.parse_args()
    check_wal_status(args.db_path)


if __name__ == "__main__":
    main()
