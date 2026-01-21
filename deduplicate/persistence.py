from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Sequence

from data_structures.ingestion import normalize_iso_timestamp

from .models import CanonicalFact, FactRecord, PersistenceOutcome

logger = logging.getLogger(__name__)


class DeduplicationPersistence:
    def __init__(self, conn: sqlite3.Connection, *, dry_run: bool = False) -> None:
        self._conn = conn
        self._conn.row_factory = sqlite3.Row
        self._dry_run = dry_run

    def apply_group(
        self,
        canonical_facts: Sequence[CanonicalFact],
        source_facts: Sequence[FactRecord],
        similarity_matrix: dict[str, list[tuple[int, int, float]]],
    ) -> PersistenceOutcome:
        if not canonical_facts:
            logger.info("LLM returned no canonical facts; preserving originals")
            return PersistenceOutcome(canonical_fact_ids=[], deleted_fact_ids=[])

        if self._dry_run:
            logger.info(
                "Dry-run: would replace %d facts with %d canonical facts",
                len(source_facts),
                len(canonical_facts),
            )
            return PersistenceOutcome(
                canonical_fact_ids=[],
                deleted_fact_ids=[fact.id for fact in source_facts],
            )

        deleted_ids = [fact.id for fact in source_facts]
        inserted_ids: list[int] = []
        similarity_json = json.dumps(similarity_matrix, sort_keys=True)
        ie_run_id = self._select_ie_run_id(source_facts)

        with self._conn:
            for canonical in canonical_facts:
                fact_id = self._insert_canonical_fact(ie_run_id, canonical)
                inserted_ids.append(fact_id)
                self._insert_evidence(fact_id, canonical.evidence)
                self._insert_audit(
                    fact_id,
                    canonical.merged_from,
                    canonical.merge_reasoning,
                    similarity_json,
                )
            self._delete_originals(deleted_ids)

        logger.info(
            "Replaced %d original facts with %d canonical facts",
            len(source_facts),
            len(canonical_facts),
        )
        return PersistenceOutcome(
            canonical_fact_ids=inserted_ids,
            deleted_fact_ids=deleted_ids,
        )

    def _select_ie_run_id(self, source_facts: Sequence[FactRecord]) -> int:
        if not source_facts:
            raise ValueError("source_facts cannot be empty when persisting canonical facts")
        return max(source_facts, key=lambda fact: fact.confidence).ie_run_id

    def _insert_canonical_fact(self, ie_run_id: int, canonical: CanonicalFact) -> int:
        attributes_json = json.dumps(canonical.attributes or {}, sort_keys=True)
        normalized_timestamp = normalize_iso_timestamp(canonical.timestamp)
        cursor = self._conn.execute(
            """
            INSERT INTO fact (
                ie_run_id,
                type,
                subject_id,
                object_id,
                object_type,
                attributes,
                ts,
                confidence,
                graph_synced_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL)
            """,
            (
                ie_run_id,
                canonical.type.value,
                canonical.subject_id,
                canonical.object_id,
                canonical.object_type,
                attributes_json,
                normalized_timestamp or "",
                canonical.confidence,
            ),
        )
        return int(cursor.lastrowid)

    def _insert_evidence(self, fact_id: int, evidence: Sequence[str]) -> None:
        if not evidence:
            return
        message_ids = list(dict.fromkeys(evidence))
        placeholders = ",".join("?" for _ in message_ids)
        existing_ids = {
            row[0]
            for row in self._conn.execute(
                f"SELECT id FROM message WHERE id IN ({placeholders})",
                message_ids,
            )
        }
        rows = [(fact_id, message_id) for message_id in message_ids if message_id in existing_ids]
        dropped = len(message_ids) - len(rows)
        if dropped:
            logger.warning(
                "Dropped %d evidence IDs missing from message table.",
                dropped,
            )
        self._conn.executemany(
            "INSERT OR IGNORE INTO fact_evidence (fact_id, message_id) VALUES (?, ?)",
            rows,
        )

    def _insert_audit(
        self,
        canonical_fact_id: int,
        merged_from: Sequence[int],
        reasoning: str,
        similarity_json: str,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO fact_deduplication_audit (
                canonical_fact_id,
                original_fact_ids,
                merge_reasoning,
                similarity_scores
            ) VALUES (?, ?, ?, ?)
            """,
            (
                canonical_fact_id,
                json.dumps(list(merged_from), sort_keys=True),
                reasoning,
                similarity_json,
            ),
        )

    def _delete_originals(self, fact_ids: Sequence[int]) -> None:
        if not fact_ids:
            return
        placeholders = ",".join("?" for _ in fact_ids)
        self._conn.execute(
            f"DELETE FROM fact WHERE id IN ({placeholders})",
            list(fact_ids),
        )
