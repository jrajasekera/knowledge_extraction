from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

import pytest

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


def _row_with(
    message_id: str,
    score: float,
    *,
    author: str = "Alice",
    timestamp: str = "2024-01-01T00:00:00Z",
    clean_content: str | None = None,
    content: str | None = None,
) -> dict:
    base = _row(timestamp)["node"].copy()
    base.update({"message_id": message_id, "author_name": author})
    if clean_content is not None:
        base["clean_content"] = clean_content
    if content is not None:
        base["content"] = content
    return {"node": base, "score": score}


def test_messages_input_defaults_enable_adaptive():
    input_data = SemanticSearchMessagesInput(queries=["hello"])

    assert input_data.similarity_threshold is None
    assert input_data.adaptive_threshold is True


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_semantic_search_messages_returns_enriched_results(mock_query):
    mock_query.return_value = [_row()]
    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["project"],
            limit=5,
            fusion_method="score_max",
        )
    )

    assert len(output.results) == 1
    result = output.results[0]
    assert result.permalink == "https://discord.com/channels/guild-9/chan-1/123"
    assert result.mentions == ["55"]
    assert result.attachments[0]["file_name"] == "doc.pdf"
    assert result.similarity_score == 0.87
    assert result.query_scores == {1: 0.87}
    assert result.appeared_in_query_count == 1


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


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_rrf_favors_multi_query_presence(mock_query):
    mock_query.side_effect = [
        [
            _row_with("B", 0.94, author="Bea"),
            _row_with("A", 0.92, author="Al"),
        ],
        [
            _row_with("A", 0.85, author="Al"),
        ],
    ]

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["first", "second"],
            limit=5,
            similarity_threshold=0.0,
            fusion_method="rrf",
            results_per_query=10,
        )
    )

    assert [result.message_id for result in output.results] == ["A", "B"]
    expected_rrf = (1 / (60 + 2)) + (1 / (60 + 1))
    assert output.results[0].similarity_score == pytest.approx(expected_rrf, rel=1e-6)


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_score_sum_adds_scores_across_queries(mock_query):
    mock_query.side_effect = [
        [_row_with("A", 0.8)],
        [_row_with("A", 0.7)],
    ]

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["q1", "q2"],
            limit=1,
            similarity_threshold=0.0,
            fusion_method="score_sum",
            results_per_query=5,
        )
    )

    assert output.results[0].similarity_score == pytest.approx(1.5)
    assert output.results[0].appeared_in_query_count == 2


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_score_max_applies_multi_query_boost(mock_query):
    mock_query.side_effect = [
        [_row_with("A", 0.9)],
        [_row_with("A", 0.8)],
    ]

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["q1", "q2"],
            limit=1,
            similarity_threshold=0.0,
            fusion_method="score_max",
            multi_query_boost=0.1,
            results_per_query=5,
        )
    )

    assert output.results[0].similarity_score == pytest.approx(0.99)


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_results_per_query_passed_to_vector_search(mock_query):
    mock_query.return_value = [_row_with("A", 0.9)]
    tool = _tool()
    input_data = SemanticSearchMessagesInput(
        queries=["only"],
        limit=1,
        similarity_threshold=0.0,
        results_per_query=42,
    )

    tool.run(input_data)

    assert mock_query.call_args_list[0][0][3] == 42


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_deduplicates_messages_by_author_and_content(mock_query):
    mock_query.return_value = [
        _row_with("111", 0.95, author="Casey", timestamp="2024-02-01T00:00:00Z"),
        _row_with("222", 0.93, author="casey", timestamp="2024-02-02T00:00:00Z"),
        _row_with("333", 0.88, author="Priya", timestamp="2024-02-03T00:00:00Z"),
        _row_with("444", 0.85, author="Jon", timestamp="2024-02-04T00:00:00Z"),
    ]

    # Make duplicate content explicit while reusing helper defaults
    for entry in mock_query.return_value[:2]:
        entry["node"]["clean_content"] = "Deployment update"
        entry["node"]["content"] = "Deployment update"

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["deployment"],
            limit=3,
            similarity_threshold=0.0,
            results_per_query=10,
            fusion_method="score_sum",
        )
    )

    assert len(output.results) == 3
    message_ids = [result.message_id for result in output.results]
    assert message_ids == ["111", "333", "444"]


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_deduplication_does_not_reintroduce_duplicates_on_fallback(mock_query):
    mock_query.return_value = [
        _row_with("111", 0.95, author="Casey", timestamp="2024-02-01T00:00:00Z"),
        _row_with("222", 0.93, author="casey", timestamp="2024-02-02T00:00:00Z"),
        _row_with("333", 0.88, author="Priya", timestamp="2024-02-03T00:00:00Z"),
    ]

    for entry in mock_query.return_value[:2]:
        entry["node"]["clean_content"] = "Deployment update"
        entry["node"]["content"] = "Deployment update"

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["deployment"],
            limit=2,
            similarity_threshold=0.0,
            results_per_query=10,
            fusion_method="score_sum",
        )
    )

    assert len(output.results) == 2
    assert [result.message_id for result in output.results] == ["111", "333"]


@patch("memory_agent.tools.semantic_search_messages.run_vector_query")
def test_blacklisted_content_is_skipped(mock_query):
    mock_query.return_value = [
        _row_with("111", 0.95, author="Casey", clean_content=" SMH ", content=" smh "),
        _row_with("222", 0.90, author="Alex", clean_content="Deployment summary"),
        _row_with("333", 0.85, author="Priya", clean_content="Timeline adjustments"),
    ]

    tool = _tool()
    output = tool.run(
        SemanticSearchMessagesInput(
            queries=["deployment"],
            limit=2,
            similarity_threshold=0.0,
            results_per_query=10,
            fusion_method="score_sum",
        )
    )

    assert len(output.results) == 2
    assert all(result.message_id != "111" for result in output.results)
    contents = {result.clean_content for result in output.results}
    assert contents == {"Deployment summary", "Timeline adjustments"}
