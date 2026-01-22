from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

from memory_agent.models import MessageModel
from memory_agent.serialization import json_dumps, to_serializable


def test_to_serializable_handles_nested_models() -> None:
    message = MessageModel(
        author_id="user-1",
        author_name="Bob",
        content="Hi",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_id="msg-1",
    )
    payload = {"messages": [message], "path": Path("./foo.txt")}

    serialized = to_serializable(payload)

    assert isinstance(serialized["messages"][0], dict)
    assert serialized["messages"][0]["author_id"] == "user-1"
    assert serialized["path"].endswith("foo.txt")


def test_json_dumps_is_stable() -> None:
    message = MessageModel(
        author_id="user-2",
        author_name="Alice",
        content="Hello",
        timestamp=datetime(2025, 2, 2, tzinfo=UTC),
        message_id="msg-2",
    )

    result = json_dumps({"messages": [message]})

    assert "user-2" in result
    assert result.startswith("{")


def test_to_serializable_handles_dataclass() -> None:
    """to_serializable should convert dataclasses to dicts."""

    @dataclass
    class SampleData:
        name: str
        value: int

    sample = SampleData(name="test", value=42)
    result = to_serializable(sample)

    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42


def test_to_serializable_handles_date() -> None:
    """to_serializable should convert date objects to ISO format strings."""
    test_date = date(2025, 6, 15)
    result = to_serializable(test_date)

    assert result == "2025-06-15"


def test_to_serializable_handles_decimal() -> None:
    """to_serializable should convert Decimal to float."""
    value = Decimal("3.14159")
    result = to_serializable(value)

    assert isinstance(result, float)
    assert abs(result - 3.14159) < 0.00001


def test_to_serializable_handles_set() -> None:
    """to_serializable should convert sets to lists."""
    test_set = {"apple", "banana", "cherry"}
    result = to_serializable(test_set)

    assert isinstance(result, list)
    assert set(result) == test_set


def test_to_serializable_handles_nested_set() -> None:
    """to_serializable should handle nested structures with sets."""
    payload = {
        "tags": {"python", "testing"},
        "count": 2,
    }
    result = to_serializable(payload)

    assert isinstance(result, dict)
    assert isinstance(result["tags"], list)
    assert set(result["tags"]) == {"python", "testing"}
    assert result["count"] == 2
