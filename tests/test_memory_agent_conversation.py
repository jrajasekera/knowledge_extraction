"""Tests for memory_agent/conversation.py."""

from __future__ import annotations

from datetime import UTC, datetime

from memory_agent.conversation import MENTION_PATTERN, ConversationInsights, extract_insights
from memory_agent.models import MessageModel


def _make_message(content: str, author_id: str = "user-1") -> MessageModel:
    """Helper to create test MessageModel instances."""
    return MessageModel(
        author_id=author_id,
        author_name="TestUser",
        content=content,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_id="msg-1",
    )


def test_mention_pattern_matches_basic_mention() -> None:
    """MENTION_PATTERN should match Discord user mentions."""
    content = "Hello <@123456789>!"
    matches = MENTION_PATTERN.findall(content)

    assert matches == ["123456789"]


def test_mention_pattern_matches_nickname_mention() -> None:
    """MENTION_PATTERN should match Discord nickname mentions with !."""
    content = "Hey <@!987654321>, check this out"
    matches = MENTION_PATTERN.findall(content)

    assert matches == ["987654321"]


def test_mention_pattern_matches_multiple_mentions() -> None:
    """MENTION_PATTERN should match multiple mentions in same message."""
    content = "<@111> and <@!222> should see this"
    matches = MENTION_PATTERN.findall(content)

    assert matches == ["111", "222"]


def test_conversation_insights_defaults() -> None:
    """ConversationInsights should have empty defaults."""
    insights = ConversationInsights()

    assert insights.people == set()
    assert insights.organizations == set()
    assert insights.topics == set()
    assert insights.questions == []
    assert insights.hints == {}


def test_extract_insights_empty_messages() -> None:
    """extract_insights should return empty insights for no messages."""
    insights = extract_insights([])

    assert insights.people == set()
    assert insights.questions == []


def test_extract_insights_extracts_mentions() -> None:
    """extract_insights should extract user IDs from mentions."""
    messages = [_make_message("Hey <@123>, can you help?")]

    insights = extract_insights(messages)

    assert "123" in insights.people


def test_extract_insights_extracts_questions() -> None:
    """extract_insights should identify questions by ? character."""
    messages = [
        _make_message("How do I fix this?"),
        _make_message("This is a statement."),
        _make_message("What about this?"),
    ]

    insights = extract_insights(messages)

    assert len(insights.questions) == 2
    assert "How do I fix this?" in insights.questions
    assert "What about this?" in insights.questions


def test_extract_insights_extracts_organizations() -> None:
    """extract_insights should extract organizations from 'work at' patterns."""
    messages = [_make_message("I work at Google now")]

    insights = extract_insights(messages)

    assert "google now" in insights.organizations


def test_extract_insights_extracts_topics_with_skill() -> None:
    """extract_insights should identify skill-related topics."""
    messages = [_make_message("I have python skill and experience")]

    insights = extract_insights(messages)

    assert len(insights.topics) >= 1


def test_extract_insights_extracts_topics_with_experience() -> None:
    """extract_insights should identify experience-related topics."""
    messages = [_make_message("I have 5 years experience with React")]

    insights = extract_insights(messages)

    assert len(insights.topics) >= 1


def test_extract_insights_stores_hints() -> None:
    """extract_insights should store all message content as hints."""
    messages = [
        _make_message("First message"),
        _make_message("Second message"),
    ]

    insights = extract_insights(messages)

    assert insights.hints["message_0"] == "First message"
    assert insights.hints["message_1"] == "Second message"


def test_extract_insights_combined() -> None:
    """extract_insights should handle a conversation with multiple patterns."""
    messages = [
        _make_message("Hey <@111>, do you work at Acme?"),
        _make_message("Yes, I have skill in Python"),
        _make_message("That's cool"),
    ]

    insights = extract_insights(messages)

    assert "111" in insights.people
    assert len(insights.questions) == 1
    assert "acme?" in insights.organizations
    assert len(insights.hints) == 3
