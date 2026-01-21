from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime

from ie.types import FactType

from .models import Partition


@dataclass(slots=True)
class RunRecord:
    id: int
    status: str
    total_partitions: int
    processed_partitions: int


@dataclass(slots=True)
class RunSnapshot:
    id: int
    status: str
    total_partitions: int
    processed_partitions: int
    facts_processed: int
    facts_merged: int
    candidate_groups_processed: int
    started_at: datetime


class DeduplicationProgress:
    """Manage deduplication run state and partition tracking."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row

    def get_active_run(self) -> RunRecord | None:
        row = self._conn.execute(
            """
            SELECT id, status, total_partitions, processed_partitions
            FROM deduplication_run
            WHERE status IN ('running', 'paused', 'failed')
            ORDER BY started_at DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return RunRecord(
            id=int(row["id"]),
            status=str(row["status"]),
            total_partitions=int(row["total_partitions"] or 0),
            processed_partitions=int(row["processed_partitions"] or 0),
        )

    def start_new_run(self, total_partitions: int) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO deduplication_run (
                status,
                total_partitions,
                processed_partitions,
                facts_processed,
                facts_merged,
                candidate_groups_processed
            ) VALUES ('running', ?, 0, 0, 0, 0)
            """,
            (total_partitions,),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def get_run_snapshot(self, run_id: int) -> RunSnapshot:
        row = self._conn.execute(
            """
            SELECT id, status, total_partitions, processed_partitions,
                   facts_processed, facts_merged, candidate_groups_processed,
                   started_at
            FROM deduplication_run
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(f"No deduplication run found for id={run_id}")

        started_at_raw = row["started_at"]
        started_at = _coerce_utc_datetime(started_at_raw)

        return RunSnapshot(
            id=int(row["id"]),
            status=str(row["status"]),
            total_partitions=int(row["total_partitions"] or 0),
            processed_partitions=int(row["processed_partitions"] or 0),
            facts_processed=int(row["facts_processed"] or 0),
            facts_merged=int(row["facts_merged"] or 0),
            candidate_groups_processed=int(row["candidate_groups_processed"] or 0),
            started_at=started_at,
        )

    def resume_run(self, run_id: int) -> None:
        self._conn.execute(
            "UPDATE deduplication_run SET status = 'running' WHERE id = ?",
            (run_id,),
        )
        self._conn.commit()

    def update_totals(
        self,
        run_id: int,
        *,
        processed_partitions: int = 0,
        facts_processed: int = 0,
        facts_merged: int = 0,
        candidate_groups_processed: int = 0,
    ) -> None:
        self._conn.execute(
            """
            UPDATE deduplication_run
            SET processed_partitions = processed_partitions + ?,
                facts_processed = facts_processed + ?,
                facts_merged = facts_merged + ?,
                candidate_groups_processed = candidate_groups_processed + ?
            WHERE id = ?
            """,
            (
                processed_partitions,
                facts_processed,
                facts_merged,
                candidate_groups_processed,
                run_id,
            ),
        )
        self._conn.commit()

    def mark_run_completed(self, run_id: int, status: str) -> None:
        self._conn.execute(
            """
            UPDATE deduplication_run
            SET status = ?, completed_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (status, run_id),
        )
        self._conn.commit()

    def ensure_partition_entry(self, run_id: int, partition: Partition) -> None:
        self._conn.execute(
            """
            INSERT OR IGNORE INTO deduplication_partition_progress (
                run_id, fact_type, subject_id, status
            ) VALUES (?, ?, ?, 'pending')
            """,
            (run_id, partition.fact_type.value, partition.subject_id),
        )
        self._conn.commit()

    def mark_partition_status(
        self,
        run_id: int,
        partition: Partition,
        status: str,
    ) -> None:
        self._conn.execute(
            """
            UPDATE deduplication_partition_progress
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ? AND fact_type = ? AND subject_id = ?
            """,
            (status, run_id, partition.fact_type.value, partition.subject_id),
        )
        self._conn.commit()

    def partition_status(self, run_id: int, partition: Partition) -> str | None:
        row = self._conn.execute(
            """
            SELECT status
            FROM deduplication_partition_progress
            WHERE run_id = ? AND fact_type = ? AND subject_id = ?
            """,
            (run_id, partition.fact_type.value, partition.subject_id),
        ).fetchone()
        return str(row["status"]) if row else None

    def completed_partitions(self, run_id: int) -> set[tuple[FactType, str]]:
        rows = self._conn.execute(
            """
            SELECT fact_type, subject_id
            FROM deduplication_partition_progress
            WHERE run_id = ? AND status = 'completed'
            """,
            (run_id,),
        ).fetchall()
        return {(FactType(row["fact_type"]), str(row["subject_id"])) for row in rows}


def _coerce_utc_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(UTC)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
