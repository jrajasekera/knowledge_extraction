from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

from memory_agent.embeddings import EmbeddingProvider
from memory_agent.tools.base import ToolContext
from memory_agent.tools.semantic_search_messages import (
    SemanticSearchMessagesInput,
    SemanticSearchMessagesTool,
)


class DummyEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(model_name="dummy-model")

    def embed(self, texts):  # noqa: ANN001 - simplified for tests
        return [[0.1, 0.2, 0.3] for _ in texts]


def _tool() -> SemanticSearchMessagesTool:
    context = ToolContext(driver=None, embeddings_model=DummyEmbeddingProvider())
    return SemanticSearchMessagesTool(context)


def _row(timestamp: str = "2024-01-01T00:00:00Z") -> dict:
    node = {
        "message_id": "123",
        "content": "Original **content**",
        "clean_content": "Original content",
        "author_id": "42",
        "author_name": "Alice",
        "channel_id": "chan-1",
        "channel_name": "general",
        "channel_topic": "chatter",
        "channel_type": "GUILD_TEXT",
        "guild_id": "guild-9",
        "guild_name": "Playground",
        "timestamp": timestamp,
        "edited_timestamp": None,
        "is_pinned": False,
        "message_type": "DEFAULT",
        "thread_id": None,
        "mentions": ["55"],
        "mention_names": ["Bob"],
        "attachments": json.dumps([{"id": "att-1", "file_name": "doc.pdf", "url": "http://file"}]),
        "reactions": json.dumps([{"emoji_name": ":smile:", "count": 2}]),
    }
    return {"node": node, "score": 0.87}


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_semantic_search_messages_returns_enriched_results(mock_query):
    mock_query.return_value = [_row()]
    tool = _tool()
    output = tool.run(SemanticSearchMessagesInput(queries=["project"], limit=5))

    assert len(output.results) == 1
    result = output.results[0]
    assert result.permalink == "https://discord.com/channels/guild-9/chan-1/123"
    assert result.mentions == ["55"]
    assert result.attachments[0]["file_name"] == "doc.pdf"
    assert result.similarity_score == 0.87


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_semantic_search_messages_applies_time_filters(mock_query):
    mock_query.return_value = [_row("2020-01-01T00:00:00Z")]
    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["project"],
            limit=5,
            start_timestamp=datetime(2023, 1, 1),
        )
    )
    assert output.results == []
