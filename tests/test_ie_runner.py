from __future__ import annotations

import sqlite3
from pathlib import Path

import ie.runner as ie_runner
import pytest
from ie import IEConfig, run_ie_job, reset_ie_progress

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schema.sql"


def _build_sample_db(tmp_path: Path, message_count: int = 3) -> Path:
    db_path = tmp_path / "discord.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.execute(
            "INSERT INTO import_batch (id, source_path, exported_at_iso, reported_msg_count, loaded_msg_count)"
            " VALUES (1, 'test.json', '2025-01-01T00:00:00Z', ?, ?)",
            (message_count, message_count),
        )
        conn.execute(
            "INSERT INTO guild (id, name) VALUES ('g1', 'Guild 1')"
        )
        conn.execute(
            "INSERT INTO channel (id, guild_id, type, category, name) VALUES ('c1', 'g1', 'text', 'general', 'General')"
        )
        conn.execute(
            "INSERT INTO member (id, name, discriminator, nickname, is_bot) VALUES ('u1', 'User One', '0001', 'User', 0)"
        )
        for idx in range(message_count):
            ts = f"2025-01-01T00:00:{idx:02d}+00:00"
            conn.execute(
                """
                INSERT INTO message (
                  id, channel_id, guild_id, author_id, type, timestamp,
                  timestamp_edited, call_ended_at, is_pinned, content, import_batch_id
                ) VALUES (?, 'c1', 'g1', 'u1', 'Default', ?, NULL, NULL, 0, ?, 1)
                """,
                (f"m{idx}", ts, f"message {idx}"),
            )
        conn.commit()
    finally:
        conn.close()
    return db_path


def _make_config() -> IEConfig:
    return IEConfig(window_size=2, confidence_threshold=0.0, max_concurrent_requests=1)


def _count_cached_rows(db_path: Path) -> int:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM ie_window_state")
        row = cur.fetchone()
        return int(row[0] or 0)
    finally:
        conn.close()


def test_run_ie_job_skips_cached_windows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _build_sample_db(tmp_path)
    config = _make_config()

    first_calls = {"count": 0}

    class CountingClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "CountingClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            first_calls["count"] += 1
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", CountingClient)
    stats_first = run_ie_job(db_path, config=config, use_cache=True)

    assert stats_first.processed_windows == 3
    assert stats_first.cached_windows == 0
    assert first_calls["count"] == 3
    assert _count_cached_rows(db_path) == 3

    second_calls = {"count": 0}

    class SilentClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "SilentClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            second_calls["count"] += 1
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", SilentClient)
    stats_second = run_ie_job(db_path, config=config, use_cache=True)

    assert stats_second.processed_windows == 0
    assert stats_second.cached_windows == 3
    assert stats_second.skipped_windows == 0
    assert second_calls["count"] == 0


def test_run_ie_job_force_reprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _build_sample_db(tmp_path)
    config = _make_config()

    initial_calls = {"count": 0}

    class PrimingClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "PrimingClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            initial_calls["count"] += 1
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", PrimingClient)
    run_ie_job(db_path, config=config, use_cache=True)
    assert _count_cached_rows(db_path) == 3
    assert initial_calls["count"] == 3

    forced_calls = {"count": 0}

    class ForcedClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "ForcedClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            forced_calls["count"] += 1
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", ForcedClient)
    stats_forced = run_ie_job(db_path, config=config, use_cache=False)

    assert stats_forced.processed_windows == 3
    assert stats_forced.cached_windows == 0
    assert forced_calls["count"] == 3


def test_reset_ie_cache_clears_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = _build_sample_db(tmp_path)
    config = _make_config()

    class PrimingClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "PrimingClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", PrimingClient)
    run_ie_job(db_path, config=config, use_cache=True)
    assert _count_cached_rows(db_path) == 3

    reset_ie_progress(db_path, clear_cache=True)
    assert _count_cached_rows(db_path) == 0

    refresh_calls = {"count": 0}

    class RefreshClient:
        def __init__(self, config) -> None:
            self.config = config

        def __enter__(self) -> "RefreshClient":
            return self

        def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
            self.close()

        def close(self) -> None:
            return None

        def complete(self, messages):
            refresh_calls["count"] += 1
            return '{"facts":[]}'

    monkeypatch.setattr(ie_runner, "LlamaServerClient", RefreshClient)
    stats_refresh = run_ie_job(db_path, config=config, use_cache=True)

    assert stats_refresh.processed_windows == 3
    assert stats_refresh.cached_windows == 0
    assert refresh_calls["count"] == 3
    assert _count_cached_rows(db_path) == 3
