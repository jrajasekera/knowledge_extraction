from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from memory_agent.models import MessageModel
from memory_agent.serialization import json_dumps, to_serializable


def test_to_serializable_handles_nested_models() -> None:
    message = MessageModel(
        author_id="user-1",
        author_name="Bob",
        content="Hi",
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
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
        timestamp=datetime(2025, 2, 2, tzinfo=timezone.utc),
        message_id="msg-2",
    )

    result = json_dumps({"messages": [message]})

    assert "user-2" in result
    assert result.startswith("{")
