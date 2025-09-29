#!/usr/bin/env python3
"""End-to-end Discord export ingestion pipeline."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from facts_to_graph import materialize_facts
from ie import IEConfig, LlamaServerConfig, FactType, run_ie_job
from import_discord_json import ingest_exports
from loader import load_into_neo4j


def _ensure_official_name_column(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(member)")
    columns = {row[1] for row in cur}
    if "official_name" not in columns:
        conn.execute("ALTER TABLE member ADD COLUMN official_name TEXT")


def apply_schema(sqlite_path: Path, schema_path: Path) -> None:
    """Ensure the SQLite schema is applied before ingesting."""
    schema_sql = schema_path.read_text(encoding="utf-8")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(schema_sql)
        _ensure_official_name_column(conn)
        conn.commit()
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
    parser.add_argument("--no-ie", action="store_true", help="Skip the IE stage after loading data.")
    parser.add_argument("--ie-window-size", type=int, default=4, help="Message window size for IE prompts.")
    parser.add_argument("--ie-confidence", type=float, default=0.5, help="Minimum confidence to store IE facts.")
    parser.add_argument("--ie-max-windows", type=int, help="Optional cap on IE windows (for testing).")
    parser.add_argument("--llama-url", default="http://localhost:8080/v1/chat/completions", help="llama-server chat completions URL.")
    parser.add_argument("--llama-model", default="GLM-4.5-Air", help="Model name to request from llama-server.")
    parser.add_argument("--llama-temperature", type=float, default=0.2, help="Generation temperature for llama-server.")
    parser.add_argument("--llama-top-p", type=float, default=0.95, help="Top-p nucleus sampling for llama-server.")
    parser.add_argument("--llama-max-tokens", type=int, default=512, help="Max tokens for llama-server responses.")
    parser.add_argument("--llama-timeout", type=float, default=60.0, help="llama-server HTTP timeout (seconds).")
    parser.add_argument("--llama-api-key", help="Optional bearer token if llama-server requires auth.")
    parser.add_argument("--no-fact-graph", action="store_true", help="Skip materializing facts into Neo4j.")
    parser.add_argument("--fact-confidence", type=float, default=0.5, help="Minimum confidence when merging facts into Neo4j.")
    parser.add_argument(
        "--fact-types",
        nargs="*",
        help="Optional subset of fact types to materialize (e.g., WORKS_AT LIVES_IN).",
    )
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

    if args.no_ie:
        print("Skipping IE stage (per --no-ie).")
    else:
        ie_config = IEConfig(
            window_size=args.ie_window_size,
            confidence_threshold=args.ie_confidence,
            max_windows=args.ie_max_windows,
        )
        llama_config = LlamaServerConfig(
            base_url=args.llama_url,
            model=args.llama_model,
            temperature=args.llama_temperature,
            top_p=args.llama_top_p,
            max_tokens=args.llama_max_tokens,
            timeout=args.llama_timeout,
            api_key=args.llama_api_key,
        )
        print("Running IE stage with llama-server...")
        run_ie_job(sqlite_path, config=ie_config, client_config=llama_config)

    if args.no_fact_graph:
        print("Skipping fact materialization (per --no-fact-graph).")
    else:
        fact_types = (
            tuple(FactType(value) for value in args.fact_types)
            if args.fact_types
            else None
        )
        print("Materializing facts into Neo4j...")
        materialize_facts(
            sqlite_path,
            neo4j_uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            fact_types=fact_types,
            min_confidence=args.fact_confidence,
        )

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
