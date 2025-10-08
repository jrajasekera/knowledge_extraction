#!/usr/bin/env python3
"""End-to-end Discord export ingestion pipeline with resumable stages."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from facts_to_graph import MaterializeSummary, materialize_facts
from ie import IEConfig, LlamaServerConfig, FactType, IERunStats, reset_ie_progress, run_ie_job
from import_discord_json import ingest_exports
from loader import load_into_neo4j

PIPELINE_STAGES: tuple[str, ...] = ("ingest", "load", "ie", "facts")
StageRunner = Callable[[], Optional[Dict[str, Any]]]


def _ensure_official_name_column(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(member)")
    columns = {row[1] for row in cur}
    if "official_name" not in columns:
        conn.execute("ALTER TABLE member ADD COLUMN official_name TEXT")


def _ensure_fact_graph_synced_column(conn: sqlite3.Connection) -> None:
    cur = conn.execute("PRAGMA table_info(fact)")
    columns = {row[1] for row in cur}
    if "graph_synced_at" not in columns:
        conn.execute("ALTER TABLE fact ADD COLUMN graph_synced_at TEXT")


def apply_schema(sqlite_path: Path, schema_path: Path) -> None:
    """Ensure the SQLite schema is applied before ingesting."""
    schema_sql = schema_path.read_text(encoding="utf-8")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(sqlite_path))
    try:
        conn.executescript(schema_sql)
        _ensure_official_name_column(conn)
        _ensure_fact_graph_synced_column(conn)
        conn.commit()
    finally:
        conn.close()


def _serialize_details(details: Optional[Dict[str, Any]]) -> Optional[str]:
    if details is None:
        return None
    if isinstance(details, str):
        return details
    return json.dumps(details, sort_keys=True)


def _ensure_stage_rows(conn: sqlite3.Connection, run_id: int) -> None:
    for stage in PIPELINE_STAGES:
        conn.execute(
            "INSERT OR IGNORE INTO pipeline_stage_state (run_id, stage, status) VALUES (?, ?, 'pending')",
            (run_id, stage),
        )


def _get_stage_status(conn: sqlite3.Connection, run_id: int, stage: str) -> str:
    row = conn.execute(
        "SELECT status FROM pipeline_stage_state WHERE run_id = ? AND stage = ?",
        (run_id, stage),
    ).fetchone()
    if row is None:
        _ensure_stage_rows(conn, run_id)
        return "pending"
    status = str(row[0])
    if stage == "ie" and status == "completed":
        ie_row = conn.execute(
            "SELECT processed_windows, total_windows FROM ie_progress LIMIT 1"
        ).fetchone()
        if ie_row is not None and int(ie_row[0]) < int(ie_row[1]):
            return "pending"
    return status


def _set_stage_status(
    conn: sqlite3.Connection,
    run_id: int,
    stage: str,
    status: str,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    conn.execute(
        """
        UPDATE pipeline_stage_state
        SET status = ?, details = ?, updated_at = CURRENT_TIMESTAMP
        WHERE run_id = ? AND stage = ?
        """,
        (status, _serialize_details(details), run_id, stage),
    )
    conn.execute(
        "UPDATE pipeline_run SET current_stage = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (stage, run_id),
    )
    conn.commit()


def _get_active_pipeline_run(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id, status, current_stage
        FROM pipeline_run
        WHERE status IN ('running', 'paused')
        ORDER BY started_at DESC
        LIMIT 1
        """,
    ).fetchone()


def _first_pending_stage(conn: sqlite3.Connection, run_id: int) -> Optional[str]:
    rows = conn.execute(
        "SELECT stage, status FROM pipeline_stage_state WHERE run_id = ?",
        (run_id,),
    ).fetchall()
    status_map = {row[0]: row[1] for row in rows}
    for stage in PIPELINE_STAGES:
        if status_map.get(stage) != "completed":
            return stage
    return None


def _create_pipeline_run(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "INSERT INTO pipeline_run (status, current_stage) VALUES ('running', ?)",
        (PIPELINE_STAGES[0],),
    )
    run_id = int(cur.lastrowid)
    _ensure_stage_rows(conn, run_id)
    conn.commit()
    return run_id


def _update_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    *,
    stage: Optional[str] = None,
    completed: bool = False,
) -> None:
    assignments = ["status = :status", "updated_at = CURRENT_TIMESTAMP"]
    params: Dict[str, Any] = {"status": status, "id": run_id}
    if stage is not None:
        assignments.append("current_stage = :stage")
        params["stage"] = stage
    if completed:
        assignments.append("completed_at = CURRENT_TIMESTAMP")
    conn.execute(
        f"UPDATE pipeline_run SET {', '.join(assignments)} WHERE id = :id",
        params,
    )
    conn.commit()


def _prepare_pipeline_run(
    conn: sqlite3.Connection,
    *,
    resume: bool,
    restart: bool,
) -> int:
    active = _get_active_pipeline_run(conn)
    if resume:
        if active is None:
            raise SystemExit("No active pipeline run found to resume.")
        run_id = int(active["id"])
        _ensure_stage_rows(conn, run_id)
        _update_run_status(conn, run_id, "running")
        return run_id

    if active is not None:
        run_id = int(active["id"])
        stage = active["current_stage"] or "unknown"
        status = active["status"]
        if restart:
            _update_run_status(conn, run_id, "cancelled", completed=True)
            return _create_pipeline_run(conn)
        raise SystemExit(
            f"Pipeline run {run_id} is {status} at stage '{stage}'. Use --resume or --restart."
        )

    return _create_pipeline_run(conn)


def _has_ie_progress(sqlite_path: Path) -> bool:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        row = conn.execute("SELECT 1 FROM ie_progress LIMIT 1").fetchone()
        return row is not None
    finally:
        conn.close()


def _has_dirty_facts(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT 1 FROM fact WHERE graph_synced_at IS NULL LIMIT 1",
    ).fetchone()
    return row is not None


def _run_stage(
    conn: sqlite3.Connection,
    run_id: int,
    stage: str,
    runner: StageRunner,
) -> None:
    _set_stage_status(conn, run_id, stage, "in_progress")
    try:
        details = runner() or {}
    except KeyboardInterrupt:
        _set_stage_status(conn, run_id, stage, "pending")
        raise
    except Exception:
        _set_stage_status(conn, run_id, stage, "pending")
        raise
    else:
        desired_status = "completed"
        if isinstance(details, dict) and not details.get("completed", True):
            desired_status = "pending"
        _set_stage_status(conn, run_id, stage, desired_status, details)


def _run_pipeline(
    conn: sqlite3.Connection,
    run_id: int,
    args: argparse.Namespace,
    sqlite_path: Path,
) -> None:
    json_path = args.json.resolve() if args.json else None
    json_dir = args.json_dir.resolve() if args.json_dir else None
    fact_types = (
        tuple(FactType(value) for value in args.fact_types)
        if args.fact_types
        else None
    )

    if args.resume and not args.no_fact_graph and _has_dirty_facts(conn):
        print("[pipeline] Detected unsynced facts from previous run; materializing before resuming...")
        summary = materialize_facts(
            sqlite_path,
            neo4j_uri=args.neo4j_uri,
            user=args.neo4j_user,
            password=args.neo4j_password,
            fact_types=fact_types,
            min_confidence=args.fact_confidence,
        )
        if summary.processed > 0:
            details = summary.as_dict()
            details["completed"] = True
            _set_stage_status(conn, run_id, "facts", "pending", details)

    for stage in PIPELINE_STAGES:
        status = _get_stage_status(conn, run_id, stage)
        if status == "completed":
            continue

        print(f"[pipeline] Stage: {stage}")

        if stage == "ingest":
            def run_ingest() -> Dict[str, Any]:
                print("Applying Discord exports into SQLite...")
                total = ingest_exports(
                    sqlite_path,
                    json=json_path,
                    json_dir=json_dir,
                    skip_existing=not args.no_skip_existing,
                )
                return {"messages_loaded": total}

            _run_stage(conn, run_id, stage, run_ingest)
            continue

        if stage == "load":
            def run_load() -> Dict[str, Any]:
                print("Loading data into Neo4j...")
                load_into_neo4j(
                    sqlite_path,
                    neo4j_uri=args.neo4j_uri,
                    user=args.neo4j_user,
                    password=args.neo4j_password,
                )
                return {"status": "done"}

            _run_stage(conn, run_id, stage, run_load)
            continue

        if stage == "ie":
            if args.no_ie:
                print("Skipping IE stage (--no-ie).")
                _set_stage_status(conn, run_id, stage, "completed", {"skipped": True})
                continue

            def run_ie() -> Dict[str, Any]:
                resume_ie = _has_ie_progress(sqlite_path)
                if resume_ie and not args.resume:
                    print("[IE] Clearing residual progress for fresh run.")
                    reset_ie_progress(sqlite_path)
                    resume_ie = False

                ie_config = IEConfig(
                    window_size=args.ie_window_size,
                    confidence_threshold=args.ie_confidence,
                    max_windows=args.ie_max_windows,
                    max_concurrent_requests=args.ie_max_concurrent_requests,
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
                stats: IERunStats = run_ie_job(
                    sqlite_path,
                    config=ie_config,
                    client_config=llama_config,
                    resume=resume_ie,
                )
                return stats.as_dict()

            _run_stage(conn, run_id, stage, run_ie)
            continue

        if stage == "facts":
            if args.no_fact_graph:
                print("Skipping fact materialization (--no-fact-graph).")
                _set_stage_status(conn, run_id, stage, "completed", {"skipped": True})
                continue

            def run_facts() -> Dict[str, Any]:
                summary: MaterializeSummary = materialize_facts(
                    sqlite_path,
                    neo4j_uri=args.neo4j_uri,
                    user=args.neo4j_user,
                    password=args.neo4j_password,
                    fact_types=fact_types,
                    min_confidence=args.fact_confidence,
                )
                return summary.as_dict()

            _run_stage(conn, run_id, stage, run_facts)
            continue

        raise RuntimeError(f"Unknown pipeline stage: {stage}")


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
    parser.add_argument("--ie-window-size", type=int, default=8, help="Message window size for IE prompts.")
    parser.add_argument("--ie-confidence", type=float, default=0.5, help="Minimum confidence to store IE facts.")
    parser.add_argument(
        "--ie-max-windows",
        type=int,
        help="Process only this many IE windows during the current run; repeat with --resume to continue in chunks.",
    )
    parser.add_argument(
        "--ie-max-concurrent-requests",
        type=int,
        default=3,
        help="Maximum number of simultaneous IE LLM requests to issue.",
    )
    parser.add_argument("--llama-url", default="http://localhost:8080/v1/chat/completions", help="llama-server chat completions URL.")
    parser.add_argument("--llama-model", default="GLM-4.5-Air", help="Model name to request from llama-server.")
    parser.add_argument("--llama-temperature", type=float, default=0.3, help="Generation temperature for llama-server.")
    parser.add_argument("--llama-top-p", type=float, default=0.95, help="Top-p nucleus sampling for llama-server.")
    parser.add_argument("--llama-max-tokens", type=int, default=4096, help="Max tokens for llama-server responses.")
    parser.add_argument("--llama-timeout", type=float, default=600.0, help="llama-server HTTP timeout (seconds).")
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
    parser.add_argument("--resume", action="store_true", help="Resume the most recent paused pipeline run.")
    parser.add_argument("--restart", action="store_true", help="Start a new pipeline run even if one is active.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.json and not args.json_dir:
        raise SystemExit("Provide --json or --json-dir so there is data to ingest.")

    if args.resume and args.restart:
        raise SystemExit("--resume and --restart are mutually exclusive.")

    sqlite_path = args.sqlite.resolve()
    schema_path = args.schema.resolve()

    print(f"Applying schema from {schema_path} to {sqlite_path}...")
    apply_schema(sqlite_path, schema_path)

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")

    run_id = _prepare_pipeline_run(conn, resume=args.resume, restart=args.restart)
    print(f"[pipeline] Using run id {run_id}.")

    try:
        _run_pipeline(conn, run_id, args, sqlite_path)
    except KeyboardInterrupt:
        print("\nPipeline interrupted; progress saved. Resume with --resume to continue.")
        _update_run_status(conn, run_id, "paused")
        conn.close()
        sys.exit(130)
    except Exception:
        _update_run_status(conn, run_id, "failed")
        conn.close()
        raise
    else:
        pending_stage = _first_pending_stage(conn, run_id)
        if pending_stage is None:
            _update_run_status(conn, run_id, "completed", completed=True)
            conn.close()
            print("Pipeline complete.")
        else:
            remaining_msg = ""
            if pending_stage == "ie":
                details_row = conn.execute(
                    "SELECT details FROM pipeline_stage_state WHERE run_id = ? AND stage = ?",
                    (run_id, pending_stage),
                ).fetchone()
                if details_row and details_row[0]:
                    try:
                        detail_data = json.loads(details_row[0])
                        remaining = detail_data.get("remaining_windows")
                        if isinstance(remaining, int):
                            remaining_msg = f" ({remaining} windows remaining)"
                    except json.JSONDecodeError:
                        remaining_msg = ""

            _update_run_status(conn, run_id, "paused", stage=pending_stage)
            conn.close()
            print(
                f"Pipeline paused after stage '{pending_stage}'."
                f" Resume with --resume to continue{remaining_msg}."
            )


if __name__ == "__main__":
    main()
