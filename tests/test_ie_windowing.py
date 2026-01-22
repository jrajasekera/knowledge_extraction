"""Tests for ie/windowing.py."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pytest

from ie.windowing import (
    MessageRecord,
    MessageWindow,
    WindowBuilder,
    _parse_aliases,
    _parse_timestamp,
    iter_message_windows,
)

# Tests for _parse_timestamp


def test_parse_timestamp_with_none() -> None:
    """_parse_timestamp should return None for None input."""
    result = _parse_timestamp(None)
    assert result is None


def test_parse_timestamp_with_z_suffix() -> None:
    """_parse_timestamp should handle Z suffix."""
    result = _parse_timestamp("2025-01-15T12:00:00Z")

    assert result is not None
    assert result.year == 2025
    assert result.month == 1
    assert result.day == 15
    assert result.hour == 12
    assert result.tzinfo is not None


def test_parse_timestamp_with_offset() -> None:
    """_parse_timestamp should handle explicit offset."""
    result = _parse_timestamp("2025-01-15T12:00:00+00:00")

    assert result is not None
    assert result.year == 2025


# Tests for _parse_aliases


def test_parse_aliases_with_none() -> None:
    """_parse_aliases should return empty tuple for None input."""
    result = _parse_aliases(None)
    assert result == ()


def test_parse_aliases_with_empty_string() -> None:
    """_parse_aliases should return empty tuple for empty string."""
    result = _parse_aliases("")
    assert result == ()


def test_parse_aliases_with_bytes_input() -> None:
    """_parse_aliases should decode bytes input."""
    data = json.dumps([{"alias": "nickname", "alias_type": "nick"}]).encode()
    result = _parse_aliases(data)

    assert len(result) == 1
    assert result[0].name == "nickname"
    assert result[0].alias_type == "nick"


def test_parse_aliases_with_invalid_json() -> None:
    """_parse_aliases should return empty tuple for invalid JSON."""
    result = _parse_aliases("not valid json")
    assert result == ()


def test_parse_aliases_with_missing_alias_key() -> None:
    """_parse_aliases should skip items without 'alias' key."""
    data = json.dumps([{"name": "test"}, {"alias": "valid", "alias_type": "nick"}])
    result = _parse_aliases(data)

    assert len(result) == 1
    assert result[0].name == "valid"


def test_parse_aliases_with_null_alias() -> None:
    """_parse_aliases should skip items with null alias."""
    data = json.dumps([{"alias": None}, {"alias": "valid"}])
    result = _parse_aliases(data)

    assert len(result) == 1
    assert result[0].name == "valid"


def test_parse_aliases_with_none_alias_type() -> None:
    """_parse_aliases should handle None alias_type."""
    data = json.dumps([{"alias": "test", "alias_type": None}])
    result = _parse_aliases(data)

    assert len(result) == 1
    assert result[0].name == "test"
    assert result[0].alias_type is None


def test_parse_aliases_non_dict_items() -> None:
    """_parse_aliases should handle non-dict items in list."""
    data = json.dumps(["string", 123, {"alias": "valid"}])
    result = _parse_aliases(data)

    assert len(result) == 1
    assert result[0].name == "valid"


# Tests for WindowBuilder


def test_window_builder_raises_on_zero_window_size() -> None:
    """WindowBuilder should raise ValueError for window_size=0."""
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="window_size must be at least 1"):
        WindowBuilder(conn, window_size=0)


def test_window_builder_raises_on_negative_window_size() -> None:
    """WindowBuilder should raise ValueError for negative window_size."""
    conn = sqlite3.connect(":memory:")

    with pytest.raises(ValueError, match="window_size must be at least 1"):
        WindowBuilder(conn, window_size=-5)


def test_window_builder_stores_filters() -> None:
    """WindowBuilder should store filter parameters."""
    conn = sqlite3.connect(":memory:")

    builder = WindowBuilder(
        conn,
        window_size=6,
        channel_ids=["c1", "c2"],
        guild_ids=["g1"],
        author_ids=["a1", "a2", "a3"],
    )

    assert builder.window_size == 6
    assert builder.channel_ids == ("c1", "c2")
    assert builder.guild_ids == ("g1",)
    assert builder.author_ids == ("a1", "a2", "a3")


# Tests for _build_filters


def test_build_filters_no_filters() -> None:
    """_build_filters should return empty clause when no filters."""
    conn = sqlite3.connect(":memory:")
    builder = WindowBuilder(conn)

    where_clause, params = builder._build_filters()

    assert where_clause == ""
    assert params == []


def test_build_filters_with_channel_ids() -> None:
    """_build_filters should build channel filter clause."""
    conn = sqlite3.connect(":memory:")
    builder = WindowBuilder(conn, channel_ids=["c1", "c2"])

    where_clause, params = builder._build_filters()

    assert "m.channel_id IN" in where_clause
    assert params == ["c1", "c2"]


def test_build_filters_with_guild_ids() -> None:
    """_build_filters should build guild filter clause."""
    conn = sqlite3.connect(":memory:")
    builder = WindowBuilder(conn, guild_ids=["g1"])

    where_clause, params = builder._build_filters()

    assert "m.guild_id IN" in where_clause
    assert params == ["g1"]


def test_build_filters_with_author_ids() -> None:
    """_build_filters should build author filter clause."""
    conn = sqlite3.connect(":memory:")
    builder = WindowBuilder(conn, author_ids=["a1", "a2"])

    where_clause, params = builder._build_filters()

    assert "m.author_id IN" in where_clause
    assert params == ["a1", "a2"]


def test_build_filters_combined() -> None:
    """_build_filters should combine multiple filters with AND."""
    conn = sqlite3.connect(":memory:")
    builder = WindowBuilder(
        conn,
        channel_ids=["c1"],
        guild_ids=["g1"],
        author_ids=["a1"],
    )

    where_clause, params = builder._build_filters()

    assert "AND" in where_clause
    assert "m.channel_id IN" in where_clause
    assert "m.guild_id IN" in where_clause
    assert "m.author_id IN" in where_clause
    assert params == ["c1", "g1", "a1"]


# Tests for MessageRecord and MessageWindow


def test_message_record_author_label_prefers_official_name() -> None:
    """MessageRecord.author_label should prefer official_name."""
    record = MessageRecord(
        id="m1",
        channel_id="c1",
        guild_id="g1",
        author_id="a1",
        author_display="Display Name",
        official_name="Official Name",
        aliases=(),
        content="Hello",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_type="Default",
        reply_to_id=None,
    )

    assert record.author_label() == "Official Name"


def test_message_record_author_label_falls_back_to_display() -> None:
    """MessageRecord.author_label should fallback to author_display."""
    record = MessageRecord(
        id="m1",
        channel_id="c1",
        guild_id="g1",
        author_id="a1",
        author_display="Display Name",
        official_name=None,
        aliases=(),
        content="Hello",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_type="Default",
        reply_to_id=None,
    )

    assert record.author_label() == "Display Name"


def test_message_window_focus_property() -> None:
    """MessageWindow.focus should return the message at focus_index."""
    record1 = MessageRecord(
        id="m1",
        channel_id="c1",
        guild_id="g1",
        author_id="a1",
        author_display="User1",
        official_name=None,
        aliases=(),
        content="First",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_type="Default",
        reply_to_id=None,
    )
    record2 = MessageRecord(
        id="m2",
        channel_id="c1",
        guild_id="g1",
        author_id="a2",
        author_display="User2",
        official_name=None,
        aliases=(),
        content="Second",
        timestamp=datetime(2025, 1, 2, tzinfo=UTC),
        message_type="Default",
        reply_to_id=None,
    )

    window = MessageWindow(messages=(record1, record2), focus_index=1)

    assert window.focus is record2


def test_message_window_as_text() -> None:
    """MessageWindow.as_text should format messages correctly."""
    record = MessageRecord(
        id="m1",
        channel_id="c1",
        guild_id="g1",
        author_id="a1",
        author_display="Alice",
        official_name="Alice Smith",
        aliases=(),
        content="Hello",
        timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=UTC),
        message_type="Default",
        reply_to_id=None,
    )

    window = MessageWindow(messages=(record,), focus_index=0)
    result = window.as_text()

    assert "message_id=m1" in result
    assert "author_id=a1" in result
    assert "Alice Smith: Hello" in result


# Tests with in-memory SQLite database


@pytest.fixture
def test_db() -> sqlite3.Connection:
    """Create a minimal in-memory SQLite database for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("""
        CREATE TABLE message (
            id TEXT PRIMARY KEY,
            channel_id TEXT,
            guild_id TEXT,
            author_id TEXT,
            content TEXT,
            timestamp TEXT,
            type TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE member (
            id TEXT PRIMARY KEY,
            nickname TEXT,
            name TEXT,
            official_name TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE member_alias (
            member_id TEXT,
            alias TEXT,
            alias_type TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE message_reference (
            message_id TEXT,
            ref_message_id TEXT
        )
    """)
    return conn


def test_iter_message_windows_empty_database(test_db: sqlite3.Connection) -> None:
    """iter_message_windows should yield nothing for empty database."""
    windows = list(iter_message_windows(test_db))
    assert windows == []


def test_iter_message_windows_with_limit(test_db: sqlite3.Connection) -> None:
    """iter_message_windows should respect limit parameter."""
    # Insert 5 messages
    for i in range(5):
        test_db.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"m{i}",
                "c1",
                "g1",
                "a1",
                f"Message {i}",
                f"2025-01-{i + 1:02d}T12:00:00Z",
                "Default",
            ),
        )
    test_db.execute("INSERT INTO member VALUES (?, ?, ?, ?)", ("a1", None, "Author", None))

    windows = list(iter_message_windows(test_db, limit=3))

    assert len(windows) == 3


def test_window_builder_count_rows(test_db: sqlite3.Connection) -> None:
    """WindowBuilder.count_rows should return correct count."""
    # Insert 3 messages
    for i in range(3):
        test_db.execute(
            "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                f"m{i}",
                "c1",
                "g1",
                "a1",
                f"Message {i}",
                f"2025-01-{i + 1:02d}T12:00:00Z",
                "Default",
            ),
        )

    builder = WindowBuilder(test_db)
    count = builder.count_rows()

    assert count == 3


def test_window_builder_count_rows_with_filter(test_db: sqlite3.Connection) -> None:
    """WindowBuilder.count_rows should respect filters."""
    # Insert messages in different channels
    test_db.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("m1", "c1", "g1", "a1", "Msg1", "2025-01-01T12:00:00Z", "Default"),
    )
    test_db.execute(
        "INSERT INTO message VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("m2", "c2", "g1", "a1", "Msg2", "2025-01-02T12:00:00Z", "Default"),
    )

    builder = WindowBuilder(test_db, channel_ids=["c1"])
    count = builder.count_rows()

    assert count == 1
