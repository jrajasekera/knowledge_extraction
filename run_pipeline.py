#!/usr/bin/env python3
"""End-to-end Discord export ingestion pipeline."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from import_discord_json import ingest_exports
from loader import load_into_neo4j


def apply_schema(sqlite_path: Path, schema_path: Path) -> None:
    """Ensure the SQLite schema is applied before ingesting."""
    schema_sql = schema_path.read_text(encoding="utf-8")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(schema_sql)
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Discord knowledge extraction pipeline end-to-end.")
    parser.add_argument("--sqlite", type=Path, default=Path("./discord.db"), help="Path to the SQLite database file.")
    parser.add_argument("--schema", type=Path, default=Path("./schema.sql"), help="Path to the schema.sql file.")
    parser.add_argument("--json", type=Path, help="Optional path to a single Discord export JSON file.")
    parser.add_argument("--json-dir", type=Path, help="Optional directory containing Discord export JSON files.")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--neo4j-user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--neo4j-password", required=True, help="Neo4j password.")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-import files even if matching metadata already exists in import_batch.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json and not args.json_dir:
        raise SystemExit("Provide --json or --json-dir so there is data to ingest.")

    sqlite_path = args.sqlite.resolve()
    schema_path = args.schema.resolve()

    print(f"Applying schema from {schema_path} to {sqlite_path}...")
    apply_schema(sqlite_path, schema_path)

    print("Ingesting Discord exports into SQLite...")
    ingest_exports(
        sqlite_path,
        json=args.json.resolve() if args.json else None,
        json_dir=args.json_dir.resolve() if args.json_dir else None,
        skip_existing=not args.no_skip_existing,
    )

    print("Loading data into Neo4j...")
    load_into_neo4j(
        sqlite_path,
        neo4j_uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
