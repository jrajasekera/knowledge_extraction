from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass

from data_structures.ingestion import normalize_iso_timestamp
from ie.types import FactType

from .models import FactRecord, Partition


@dataclass(slots=True)
class PartitionSelection:
    partition: Partition
    facts: list[FactRecord]


class FactPartitioner:
    """Iterate over fact partitions grouped by (fact_type, subject_id)."""

    def __init__(self, conn: sqlite3.Connection, *, min_confidence: float = 0.0) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._min_confidence = min_confidence

    def count_partitions(self) -> int:
        sql = """
            SELECT COUNT(*)
            FROM (
                SELECT 1
                FROM fact
                WHERE graph_synced_at IS NOT NULL
                  AND confidence >= ?
                GROUP BY type, subject_id
                HAVING COUNT(*) >= 2
            ) AS sub
        """
        row = self._conn.execute(sql, (self._min_confidence,)).fetchone()
        return int(row[0]) if row else 0

    def iter_partitions(
        self,
        *,
        limit: int | None = None,
    ) -> Iterator[PartitionSelection]:
        sql = """
            SELECT
                f.type,
                f.subject_id,
                subj.official_name AS subject_name,
                COUNT(*) AS fact_count
            FROM fact AS f
            LEFT JOIN member AS subj ON subj.id = f.subject_id
            WHERE f.graph_synced_at IS NOT NULL
              AND f.confidence >= ?
            GROUP BY f.type, f.subject_id
            HAVING COUNT(*) >= 2
            ORDER BY fact_count DESC, f.subject_id
        """
        params: list[object] = [self._min_confidence]
        if limit is not None:
            sql += "\n            LIMIT ?"
            params.append(limit)

        for row in self._conn.execute(sql, params):
            fact_type = FactType(row["type"])
            subject_id = str(row["subject_id"])
            subject_name = row["subject_name"] if row["subject_name"] else None
            facts = self._fetch_partition_facts(fact_type, subject_id)
            fact_ids = [fact.id for fact in facts]
            partition = Partition(
                fact_type=fact_type,
                subject_id=subject_id,
                subject_name=subject_name,
                fact_ids=fact_ids,
            )
            yield PartitionSelection(partition=partition, facts=facts)

    def _fetch_partition_facts(
        self,
        fact_type: FactType,
        subject_id: str,
    ) -> list[FactRecord]:
        sql = """
            SELECT
                f.id,
                f.ie_run_id,
                f.type,
                f.subject_id,
                subj.official_name AS subject_name,
                f.object_id,
                f.object_type,
                f.attributes,
                f.ts,
                f.confidence,
                COALESCE(json_group_array(fe.message_id), '[]') AS evidence
            FROM fact AS f
            LEFT JOIN member AS subj ON subj.id = f.subject_id
            LEFT JOIN fact_evidence AS fe ON fe.fact_id = f.id
            WHERE f.type = ?
              AND f.subject_id = ?
              AND f.graph_synced_at IS NOT NULL
              AND f.confidence >= ?
            GROUP BY f.id
            ORDER BY f.confidence DESC, f.id ASC
        """
        params = (fact_type.value, subject_id, self._min_confidence)
        facts: list[FactRecord] = []
        for row in self._conn.execute(sql, params):
            attributes_raw = row["attributes"]
            if isinstance(attributes_raw, bytes):
                attributes_raw = attributes_raw.decode("utf-8")
            attributes = json.loads(attributes_raw or "{}")
            evidence_raw = row["evidence"]
            evidence = json.loads(evidence_raw) if isinstance(evidence_raw, str) else []
            evidence = [
                str(message_id)
                for message_id in evidence
                if message_id is not None and str(message_id).strip()
            ]
            object_label = attributes.get("object_label")
            normalized_ts = normalize_iso_timestamp(row["ts"])
            timestamp = normalized_ts or ""
            fact = FactRecord(
                id=int(row["id"]),
                ie_run_id=int(row["ie_run_id"]),
                type=FactType(row["type"]),
                subject_id=str(row["subject_id"]),
                subject_name=row["subject_name"],
                object_label=object_label,
                object_id=row["object_id"],
                object_type=row["object_type"],
                attributes=attributes,
                confidence=float(row["confidence"]),
                evidence=evidence,
                timestamp=timestamp,
            )
            facts.append(fact)
        return facts
