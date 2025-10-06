from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import httpx
from pydantic import ValidationError

from data_structures.ingestion import normalize_iso_timestamp

from .advanced_prompts import build_enhanced_prompt
from .client import LlamaServerClient, LlamaServerConfig
from .config import FACT_DEFINITION_INDEX, IEConfig
from .models import ExtractionResult
from .prompts import build_messages, window_hint
from .types import FactDefinition, FactType
from .windowing import MessageWindow, WindowBuilder

MAX_PROMPT_ATTEMPTS = 3


@dataclass(slots=True)
class IERunStats:
    run_id: int
    processed_windows: int
    skipped_windows: int
    returned_facts: int
    stored_facts: int
    total_windows: int
    total_processed: int
    target_windows: int
    completed: bool

    def as_dict(self) -> dict[str, int | bool]:
        remaining = max(self.total_windows - self.total_processed, 0)
        return {
            "run_id": self.run_id,
            "processed_windows": self.processed_windows,
            "skipped_windows": self.skipped_windows,
            "returned_facts": self.returned_facts,
            "stored_facts": self.stored_facts,
            "total_windows": self.total_windows,
            "total_processed": self.total_processed,
            "remaining_windows": remaining,
            "target_windows": self.target_windows,
            "completed": self.completed,
        }


@dataclass(slots=True)
class _WindowTaskResult:
    window: MessageWindow
    result: ExtractionResult | None


def _get_ie_progress(conn: sqlite3.Connection) -> dict[str, int] | None:
    row = conn.execute(
        "SELECT run_id, processed_windows, total_windows FROM ie_progress LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    return {
        "run_id": int(row[0]),
        "processed_windows": int(row[1]),
        "total_windows": int(row[2]),
    }


def _initialize_ie_progress(conn: sqlite3.Connection, run_id: int, total_windows: int) -> None:
    conn.execute(
        "INSERT INTO ie_progress (run_id, processed_windows, total_windows) VALUES (?, 0, ?)",
        (run_id, total_windows),
    )


def _update_ie_progress(conn: sqlite3.Connection, run_id: int, processed_windows: int) -> None:
    conn.execute(
        "UPDATE ie_progress SET processed_windows = ?, updated_at = CURRENT_TIMESTAMP WHERE run_id = ?",
        (processed_windows, run_id),
    )


def _clear_ie_progress(conn: sqlite3.Connection, run_id: int | None = None) -> None:
    if run_id is None:
        conn.execute("DELETE FROM ie_window_progress")
        conn.execute("DELETE FROM ie_progress")
    else:
        conn.execute("DELETE FROM ie_window_progress WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM ie_progress WHERE run_id = ?", (run_id,))


def _load_processed_window_ids(conn: sqlite3.Connection, run_id: int) -> set[str]:
    cursor = conn.execute(
        "SELECT focus_message_id FROM ie_window_progress WHERE run_id = ?",
        (run_id,),
    )
    return {row[0] for row in cursor}


def _record_window_processed(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    focus_message_id: str,
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO ie_window_progress (run_id, focus_message_id) VALUES (?, ?)",
        (run_id, focus_message_id),
    )


def reset_ie_progress(sqlite_path: str | sqlite3.Connection) -> None:
    if isinstance(sqlite_path, sqlite3.Connection):
        conn = sqlite_path
        external = True
    else:
        conn = sqlite3.connect(str(sqlite_path))
        external = False

    try:
        with conn:
            _clear_ie_progress(conn)
    finally:
        if not external:
            conn.close()


def _format_duration(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:
        return "unknown"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _insert_ie_run(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    model_params: dict,
    config: IEConfig,
) -> int:
    params_json = json.dumps(model_params, sort_keys=True)
    window_desc = (
        f"window_size={config.window_size};"
        f"fact_types={','.join(ft.value for ft in config.fact_types)};"
        f"prompt={config.prompt_version}"
    )
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
    normalized_timestamp = normalize_iso_timestamp(timestamp)
    stored_timestamp = normalized_timestamp or ""

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
            SET ie_run_id = ?, object_id = ?, object_type = ?, attributes = ?, ts = ?, confidence = ?, graph_synced_at = NULL
            WHERE id = ?
            """,
            (run_id, object_id, object_type, attributes_json, stored_timestamp, confidence, fact_id),
        )
    else:
        cur = conn.execute(
            """
            INSERT INTO fact (ie_run_id, type, subject_id, object_id, object_type, attributes, ts, confidence)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (run_id, fact_type.value, subject_id, object_id, object_type, attributes_json, stored_timestamp, confidence),
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
    resume: bool = False,
) -> IERunStats:
    config = (config or IEConfig()).validate()
    client_config = client_config or LlamaServerConfig()

    if isinstance(sqlite_path, sqlite3.Connection):
        external_conn = sqlite_path
    else:
        external_conn = None

    conn = external_conn or sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    builder = WindowBuilder(
        conn,
        window_size=config.window_size,
    )
    total_windows_available = builder.count_rows()

    progress = _get_ie_progress(conn)
    resume_mode = bool(progress)

    if resume and not resume_mode:
        raise RuntimeError("No IE run is currently in progress to resume.")
    if not resume and progress is not None:
        raise RuntimeError("Existing IE progress found; pass resume=True or reset_ie_progress first.")

    processed_ids: set[str]
    if progress is None:
        run_id = 0
        total_windows = 0
        total_processed = 0
        processed_ids = set()
    else:
        run_id = int(progress["run_id"])
        total_windows = int(progress["total_windows"])
        total_processed = int(progress["processed_windows"])
        processed_ids = _load_processed_window_ids(conn, run_id)
        # make sure counters stay in sync
        if len(processed_ids) > total_processed:
            total_processed = len(processed_ids)

    if total_windows == 0:
        total_windows = total_windows_available

    if total_windows == 0:
        print("[IE] No messages found for extraction.")
        if progress is not None:
            with conn:
                _clear_ie_progress(conn, run_id)
        if external_conn is None:
            conn.close()
        return IERunStats(
            run_id=run_id,
            processed_windows=0,
            skipped_windows=0,
            returned_facts=0,
            stored_facts=0,
            total_windows=0,
            total_processed=0,
            target_windows=0,
            completed=True,
        )

    if progress is None:
        run_id = _insert_ie_run(
            conn,
            model_name=client_config.model,
            model_params=asdict(client_config),
            config=config,
        )
        _initialize_ie_progress(conn, run_id, total_windows)
        conn.commit()
        processed_ids = set()
        total_processed = 0

    remaining_windows = max(total_windows - total_processed, 0)
    if remaining_windows == 0:
        print(f"[IE] Run {run_id} already complete; nothing to resume.")
        with conn:
            _clear_ie_progress(conn, run_id)
        if external_conn is None:
            conn.close()
        return IERunStats(
            run_id=run_id,
            processed_windows=0,
            skipped_windows=0,
            returned_facts=0,
            stored_facts=0,
            total_windows=total_windows,
            total_processed=total_processed,
            target_windows=0,
            completed=True,
        )

    session_target = remaining_windows
    if config.max_windows is not None:
        session_target = min(session_target, config.max_windows)

    with (LlamaServerClient(client_config) as client, conn):
        start_time = time.time()
        processed_this_run = 0
        skipped_windows = 0
        returned_facts = 0
        stored_facts = 0
        total_facts_returned = 0

        progress_template = "[IE] Progress: {done}/{total} ({percent:.1f}%) ETA {eta}"

        def update_progress(final: bool = False) -> None:
            elapsed = time.time() - start_time
            processed_for_rate = processed_this_run if processed_this_run else 0
            rate = processed_for_rate / elapsed if elapsed > 0 else 0.0
            remaining = max(total_windows - total_processed, 0)
            eta_seconds = remaining / rate if rate > 0 else float("inf")
            eta_text = _format_duration(eta_seconds)
            percent = (total_processed / total_windows) * 100 if total_windows else 0.0
            line = progress_template.format(
                done=total_processed,
                total=total_windows,
                percent=percent,
                eta=eta_text,
            )
            end = "\n" if final else "\r"
            print(line, end=end, flush=True)

        max_workers = max(1, config.max_concurrent_requests)
        in_flight: dict[Future[_WindowTaskResult], MessageWindow] = {}

        def process_window(window: MessageWindow) -> _WindowTaskResult:
            if config.prompt_version == "enhanced":
                messages = build_enhanced_prompt(window, config.fact_types)
            else:
                messages = build_messages(window, config.fact_types)

            result: ExtractionResult | None = None
            attempt = 0

            while attempt < MAX_PROMPT_ATTEMPTS:
                attempt += 1
                try:
                    content = client.complete(messages)
                except httpx.HTTPError as exc:
                    print()
                    print(f"[IE] Request error for {window_hint(window)}: {exc}")
                    return _WindowTaskResult(window=window, result=None)

                if not content:
                    break

                try:
                    result = ExtractionResult.model_validate_json(content)
                    break
                except ValidationError as exc:
                    print()
                    print(
                        f"[IE] Failed to parse response for {window_hint(window)} (attempt {attempt}/{MAX_PROMPT_ATTEMPTS}): {exc}"
                    )
                    print("[IE] Raw response:")
                    print(content)
                    result = None
                    if attempt < MAX_PROMPT_ATTEMPTS:
                        print("[IE] Retrying with identical prompt due to invalid JSON format.")
                    else:
                        print("[IE] Giving up on this window after repeated formatting failures.")

            return _WindowTaskResult(window=window, result=result)

        def handle_future(fut: Future[_WindowTaskResult]) -> None:
            nonlocal processed_this_run, total_processed, returned_facts, stored_facts, total_facts_returned
            window = in_flight.pop(fut)
            try:
                task_result = fut.result()
            except Exception as exc:
                print()
                print(f"[IE] Unexpected error processing {window_hint(window)}: {exc}")
                task_result = _WindowTaskResult(window=window, result=None)

            result = task_result.result
            if result:
                for fact in result.facts:
                    total_facts_returned += 1
                    if fact.type not in config.fact_types:
                        continue
                    if fact.confidence < config.confidence_threshold:
                        continue

                    definition: FactDefinition = FACT_DEFINITION_INDEX[fact.type]
                    normalized_attributes = _normalize_attributes(
                        fact.attributes,
                        object_label=fact.object_label,
                    )
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
                    returned_facts += 1

            processed_ids.add(window.focus.id)
            processed_this_run += 1
            total_processed += 1

            _record_window_processed(
                conn,
                run_id=run_id,
                focus_message_id=window.focus.id,
            )
            _update_ie_progress(conn, run_id, total_processed)

            update_progress()
            conn.commit()

        def drain(*, all_remaining: bool) -> None:
            if not in_flight:
                return
            iterator = as_completed(list(in_flight.keys()))
            if all_remaining:
                futures_to_handle = list(iterator)
            else:
                try:
                    futures_to_handle = [next(iterator)]
                except StopIteration:
                    return
            for fut in futures_to_handle:
                handle_future(fut)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for window in builder.iter_windows():
                if total_processed >= total_windows or processed_this_run >= session_target:
                    break

                if window.focus.id in processed_ids:
                    skipped_windows += 1
                    continue

                while in_flight and (
                    len(in_flight) >= max_workers
                    or processed_this_run + len(in_flight) >= session_target
                    or total_processed + len(in_flight) >= total_windows
                ):
                    drain(all_remaining=False)

                future = executor.submit(process_window, window)
                in_flight[future] = window

            while in_flight:
                drain(all_remaining=True)

        update_progress(final=True)

        completed = total_processed >= total_windows

        print(
            f"[IE] Run {run_id} {'completed' if completed else 'paused'}:"
            f" windows_processed={processed_this_run}, skipped={skipped_windows},"
            f" returned_facts={total_facts_returned}, stored={stored_facts},"
            f" target={session_target}"
        )

        if completed:
            _clear_ie_progress(conn, run_id)
            conn.commit()
        else:
            remaining_after = max(total_windows - total_processed, 0)
            print(f"[IE] Remaining windows: {remaining_after}")

        stats = IERunStats(
            run_id=run_id,
            processed_windows=processed_this_run,
            skipped_windows=skipped_windows,
            returned_facts=total_facts_returned,
            stored_facts=stored_facts,
            total_windows=total_windows,
            total_processed=total_processed,
            target_windows=session_target,
            completed=completed,
        )
        return stats

    if external_conn is None:
        conn.close()

    return IERunStats(
        run_id=run_id,
        processed_windows=0,
        skipped_windows=0,
        returned_facts=0,
        stored_facts=0,
        total_windows=total_windows,
        total_processed=total_processed,
        target_windows=0,
        completed=True,
    )
