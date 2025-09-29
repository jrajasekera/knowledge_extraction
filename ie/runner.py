from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from typing import Iterable, Sequence

import httpx
from pydantic import ValidationError

from .client import LlamaServerClient, LlamaServerConfig
from .config import FACT_DEFINITION_INDEX, IEConfig
from .models import ExtractionResult
from .prompts import build_messages, window_hint
from .types import FactDefinition, FactType
from .windowing import MessageWindow, iter_message_windows


def _insert_ie_run(conn: sqlite3.Connection, *, model_name: str, model_params: dict, config: IEConfig) -> int:
    params_json = json.dumps(model_params, sort_keys=True)
    window_desc = f"window_size={config.window_size};fact_types={','.join(ft.value for ft in config.fact_types)}"
    cur = conn.execute(
        "INSERT INTO ie_run (model_name, model_params, window_hint) VALUES (?,?,?)",
        (model_name, params_json, window_desc),
    )
    return int(cur.lastrowid)


def _normalize_attributes(attributes: dict[str, object], *, object_label: str | None) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in attributes.items():
        if isinstance(value, str):
            value = value.strip()
        normalized[key] = value
    if object_label and "object_label" not in normalized:
        normalized["object_label"] = object_label
    return normalized


def _required_attributes_present(definition: FactDefinition, attributes: dict[str, object]) -> bool:
    for attr in definition.attributes:
        if not attr.required:
            continue
        value = attributes.get(attr.name)
        if value is None:
            return False
        if isinstance(value, str) and not value.strip():
            return False
    return True


def _evidence_list(focus_message_id: str, supplied: Iterable[str]) -> list[str]:
    evidence = [msg_id for msg_id in supplied if msg_id]
    if focus_message_id not in evidence:
        evidence.append(focus_message_id)
    return evidence


def _filter_existing_evidence(conn: sqlite3.Connection, message_ids: Sequence[str]) -> list[str]:
    unique = [msg_id for msg_id in dict.fromkeys(message_ids) if msg_id]
    if not unique:
        return []
    placeholders = ",".join("?" for _ in unique)
    rows = conn.execute(
        f"SELECT id FROM message WHERE id IN ({placeholders})",
        unique,
    )
    existing = {row[0] for row in rows}
    return [msg_id for msg_id in unique if msg_id in existing]


def _upsert_fact(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    fact_type: FactType,
    subject_id: str,
    object_id: str | None,
    object_type: str | None,
    attributes: dict[str, object],
    timestamp: str,
    confidence: float,
    evidence: list[str],
) -> int:
    attributes_json = json.dumps(attributes, sort_keys=True, ensure_ascii=False)

    row = conn.execute(
        """
        SELECT id FROM fact
        WHERE type = ?
          AND subject_id = ?
          AND ((? IS NULL AND object_id IS NULL) OR object_id = ?)
          AND attributes = ?
        """,
        (fact_type.value, subject_id, object_id, object_id, attributes_json),
    ).fetchone()

    if row:
        fact_id = int(row[0])
        conn.execute(
            """
            UPDATE fact
            SET ie_run_id = ?, object_id = ?, object_type = ?, attributes = ?, ts = ?, confidence = ?
            WHERE id = ?
            """,
            (run_id, object_id, object_type, attributes_json, timestamp, confidence, fact_id),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO fact (ie_run_id, type, subject_id, object_id, object_type, attributes, ts, confidence)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (run_id, fact_type.value, subject_id, object_id, object_type, attributes_json, timestamp, confidence),
        )
        fact_id = int(cur.lastrowid)

    for message_id in evidence:
        conn.execute(
            "INSERT OR IGNORE INTO fact_evidence (fact_id, message_id) VALUES (?, ?)",
            (fact_id, message_id),
        )
    return fact_id


def run_ie_job(
    sqlite_path: str | sqlite3.Connection,
    *,
    config: IEConfig | None = None,
    client_config: LlamaServerConfig | None = None,
) -> None:
    config = (config or IEConfig()).validate()
    client_config = client_config or LlamaServerConfig()

    if isinstance(sqlite_path, sqlite3.Connection):
        external_conn = sqlite_path
    else:
        external_conn = None

    conn = external_conn or sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    with (LlamaServerClient(client_config) as client, conn):
        run_id = _insert_ie_run(
            conn,
            model_name=client_config.model,
            model_params=asdict(client_config),
            config=config,
        )

        total_windows = 0
        total_facts = 0
        stored_facts = 0

        for window in iter_message_windows(
            conn,
            window_size=config.window_size,
            limit=config.max_windows,
        ):
            total_windows += 1
            messages = build_messages(window, config.fact_types)

            try:
                content = client.complete(messages)
            except httpx.HTTPError as exc:
                print(f"[IE] Request error for {window_hint(window)}: {exc}")
                continue

            if not content:
                continue

            try:
                result = ExtractionResult.model_validate_json(content)
            except ValidationError as exc:
                print(f"[IE] Failed to parse response for {window_hint(window)}: {exc}")
                continue

            for fact in result.facts:
                total_facts += 1
                if fact.type not in config.fact_types:
                    continue
                if fact.confidence < config.confidence_threshold:
                    continue

                definition: FactDefinition = FACT_DEFINITION_INDEX[fact.type]
                normalized_attributes = _normalize_attributes(fact.attributes, object_label=fact.object_label)
                if not _required_attributes_present(definition, normalized_attributes):
                    continue

                object_id = fact.object_id or fact.object_label
                timestamp = fact.timestamp or window.focus.timestamp.isoformat()
                evidence = _filter_existing_evidence(
                    conn,
                    _evidence_list(window.focus.id, fact.evidence),
                )
                if not evidence:
                    continue

                _upsert_fact(
                    conn,
                    run_id=run_id,
                    fact_type=fact.type,
                    subject_id=fact.subject_id,
                    object_id=object_id,
                    object_type=definition.object_type,
                    attributes=normalized_attributes,
                    timestamp=timestamp,
                    confidence=fact.confidence,
                    evidence=evidence,
                )
                stored_facts += 1

        print(
            f"[IE] Completed run {run_id}: windows={total_windows}, returned_facts={total_facts}, stored={stored_facts}"
        )

    if external_conn is None:
        conn.close()
