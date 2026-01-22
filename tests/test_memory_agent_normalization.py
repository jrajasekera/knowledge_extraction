"""Tests for memory_agent/normalization.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

from memory_agent.models import RetrievedFact
from memory_agent.normalization import (
    _build_fact,
    _normalize_evidence,
    normalize_to_facts,
)

# Tests for _normalize_evidence


def test_normalize_evidence_with_none() -> None:
    """_normalize_evidence should return empty list for None."""
    result = _normalize_evidence(None)
    assert result == []


def test_normalize_evidence_with_dicts() -> None:
    """_normalize_evidence should pass through dict entries."""
    evidence = [{"source_id": "msg-1", "snippet": "text"}]
    result = _normalize_evidence(evidence)

    assert len(result) == 1
    assert result[0]["source_id"] == "msg-1"


def test_normalize_evidence_with_strings() -> None:
    """_normalize_evidence should wrap string entries as source_id."""
    evidence = ["msg-1", "msg-2"]
    result = _normalize_evidence(evidence)

    assert len(result) == 2
    assert result[0] == {"source_id": "msg-1"}
    assert result[1] == {"source_id": "msg-2"}


def test_normalize_evidence_mixed() -> None:
    """_normalize_evidence should handle mixed dict and string entries."""
    evidence = [{"source_id": "msg-1", "snippet": "text"}, "msg-2"]
    result = _normalize_evidence(evidence)

    assert len(result) == 2
    assert result[0]["snippet"] == "text"
    assert result[1] == {"source_id": "msg-2"}


# Tests for _build_fact


def test_build_fact_basic() -> None:
    """_build_fact should create a RetrievedFact with basic fields."""
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={"role": "Engineer"},
        confidence=0.8,
        evidence=["msg-1"],
    )

    assert isinstance(fact, RetrievedFact)
    assert fact.person_id == "user-1"
    assert fact.person_name == "Alice"
    assert fact.fact_type == "WORKS_AT"
    assert fact.fact_object == "Acme"
    assert fact.confidence == 0.8


def test_build_fact_defaults_person_name() -> None:
    """_build_fact should default person_name to person_id if None."""
    fact = _build_fact(
        person_id="user-123",
        person_name=None,
        fact_type="HAS_SKILL",
        fact_object="Python",
        attributes=None,
        confidence=None,
        evidence=None,
    )

    assert fact.person_name == "user-123"


def test_build_fact_defaults_attributes() -> None:
    """_build_fact should default None attributes to empty dict."""
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes=None,
        confidence=0.5,
        evidence=None,
    )

    assert fact.attributes == {}


def test_build_fact_defaults_confidence() -> None:
    """_build_fact should default None confidence to 0.0."""
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        confidence=None,
        evidence=None,
    )

    assert fact.confidence == 0.0


def test_build_fact_with_timestamp_string() -> None:
    """_build_fact should accept string timestamp."""
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        confidence=0.5,
        evidence=None,
        timestamp="2025-01-15T12:00:00Z",
    )

    assert fact.timestamp is not None
    assert fact.timestamp.year == 2025


def test_build_fact_with_timestamp_datetime() -> None:
    """_build_fact should accept datetime timestamp."""
    ts = datetime(2025, 6, 15, tzinfo=UTC)
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        confidence=0.5,
        evidence=None,
        timestamp=ts,
    )

    assert fact.timestamp == ts


def test_build_fact_with_similarity_score() -> None:
    """_build_fact should accept similarity_score."""
    fact = _build_fact(
        person_id="user-1",
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        confidence=0.5,
        evidence=None,
        similarity_score=0.92,
    )

    assert fact.similarity_score == 0.92


# Tests for normalize_to_facts


def test_normalize_to_facts_unknown_tool() -> None:
    """normalize_to_facts should return empty list for unknown tool."""
    result = normalize_to_facts("unknown_tool", {})
    assert result == []


def test_normalize_to_facts_semantic_search_facts() -> None:
    """normalize_to_facts should handle semantic_search_facts output."""
    # Create a mock output with results
    mock_result = MagicMock()
    mock_result.person_id = "user-1"
    mock_result.person_name = "Alice"
    mock_result.fact_type = "WORKS_AT"
    mock_result.fact_object = "Acme"
    mock_result.attributes = {"role": "Engineer"}
    mock_result.confidence = 0.85
    mock_result.evidence = ["msg-1"]
    mock_result.similarity_score = 0.92

    mock_output = MagicMock()
    mock_output.results = [mock_result]

    result = normalize_to_facts("semantic_search_facts", mock_output)

    assert len(result) == 1
    assert result[0].person_name == "Alice"
    assert result[0].fact_type == "WORKS_AT"


def test_normalize_to_facts_semantic_search_messages() -> None:
    """normalize_to_facts should handle semantic_search_messages output."""
    # Create a mock message result
    mock_message = MagicMock()
    mock_message.author_id = "user-1"
    mock_message.author_name = "Alice"
    mock_message.message_id = "msg-123"
    mock_message.content = "Hello world"
    mock_message.clean_content = "Hello world"
    mock_message.excerpt = None
    mock_message.channel_id = "channel-1"
    mock_message.channel_name = "general"
    mock_message.channel_topic = None
    mock_message.guild_id = "guild-1"
    mock_message.guild_name = "Test Server"
    mock_message.permalink = "https://discord.com/..."
    mock_message.mentions = []
    mock_message.mention_names = []
    mock_message.attachments = []
    mock_message.reactions = []
    mock_message.thread_id = None
    mock_message.message_type = "DEFAULT"
    mock_message.timestamp = "2025-01-15T12:00:00Z"
    mock_message.similarity_score = 0.88

    mock_output = MagicMock()
    mock_output.results = [mock_message]

    result = normalize_to_facts("semantic_search_messages", mock_output)

    assert len(result) == 1
    assert result[0].person_name == "Alice"
    assert result[0].fact_type == "MESSAGE_SEARCH_RESULT"
    assert result[0].fact_object == "Hello world"


def test_normalize_to_facts_message_without_author() -> None:
    """normalize_to_facts should handle messages without author info."""
    mock_message = MagicMock()
    mock_message.author_id = None
    mock_message.author_name = None
    mock_message.message_id = "msg-123"
    mock_message.content = "Anonymous message"
    mock_message.clean_content = None
    mock_message.excerpt = None
    mock_message.channel_id = None
    mock_message.channel_name = None
    mock_message.channel_topic = None
    mock_message.guild_id = None
    mock_message.guild_name = None
    mock_message.permalink = None
    mock_message.mentions = None
    mock_message.mention_names = None
    mock_message.attachments = None
    mock_message.reactions = None
    mock_message.thread_id = None
    mock_message.message_type = None
    mock_message.timestamp = None
    mock_message.similarity_score = 0.75

    mock_output = MagicMock()
    mock_output.results = [mock_message]

    result = normalize_to_facts("semantic_search_messages", mock_output)

    assert len(result) == 1
    assert result[0].person_id == "unknown"
    assert result[0].person_name == "unknown"
