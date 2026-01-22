from __future__ import annotations

import json

from memory_agent.message_embedding_pipeline import GraphMessage, generate_message_embeddings


class DummyProvider:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):  # noqa: ANN001 - simplified stub for tests
        self.calls.append(list(texts))
        return [[0.5, 0.4] for _ in texts]


def _message(content: str | None) -> GraphMessage:
    return GraphMessage(
        message_id="msg-1" if content else "msg-empty",
        content=content,
        author_id="42",
        author_name="Alice",
        channel_id="chan-1",
        channel_name="general",
        channel_topic="Chit chat",
        channel_type="GUILD_TEXT",
        guild_id="guild-1",
        guild_name="Playground",
        timestamp="2024-01-01T00:00:00Z",
        edited_timestamp=None,
        is_pinned=False,
        message_type="DEFAULT",
        mentions=[{"id": "55", "name": "Bob"}],
        attachments=[
            {"id": "att-1", "url": "http://file", "file_name": "doc.pdf", "size_bytes": 10}
        ],
        reactions=[{"emoji_id": "1", "emoji_name": ":smile:", "count": 2}],
        thread_id=None,
    )


def test_generate_message_embeddings_serializes_payloads():
    provider = DummyProvider()
    rows, skipped = generate_message_embeddings(
        [
            _message("Hello **world**"),
            _message("   "),
        ],
        provider,  # type: ignore[arg-type]
        batch_size=2,
    )

    assert skipped == 1
    assert len(rows) == 1
    row = rows[0]
    assert row["mentions"] == ["55"]
    attachments = json.loads(row["attachments"])
    assert attachments[0]["file_name"] == "doc.pdf"
    assert row["clean_content"].startswith("Hello")
    assert "Alice" not in row["clean_content"]
    assert row["embedding"] == [0.5, 0.4]
