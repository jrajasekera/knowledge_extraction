"""Tests for memory_agent/context_summarizer.py."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from memory_agent.context_summarizer import (
    build_context_summary_prompt,
    format_conversation_for_prompt,
    generate_context_summary,
)
from memory_agent.models import MessageModel


def _make_message(author_name: str, content: str, author_id: str = "user-1") -> MessageModel:
    """Helper to create test MessageModel instances."""
    return MessageModel(
        author_id=author_id,
        author_name=author_name,
        content=content,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_id="msg-1",
    )


# Tests for format_conversation_for_prompt


def test_format_conversation_empty_input() -> None:
    """format_conversation_for_prompt should return placeholder for empty list."""
    result = format_conversation_for_prompt([])

    assert result == "(No conversation messages)"


def test_format_conversation_basic_messages() -> None:
    """format_conversation_for_prompt should format messages correctly."""
    messages = [
        _make_message("Alice", "Hello"),
        _make_message("Bob", "Hi there"),
    ]

    result = format_conversation_for_prompt(messages)

    assert "Alice: Hello" in result
    assert "Bob: Hi there" in result


def test_format_conversation_max_messages_limit() -> None:
    """format_conversation_for_prompt should respect max_messages limit."""
    messages = [_make_message(f"User{i}", f"Message {i}") for i in range(15)]

    result = format_conversation_for_prompt(messages, max_messages=5)

    # Should only include the last 5 messages (indices 10-14)
    assert "User10" in result
    assert "User14" in result
    assert "User0" not in result
    assert "User9" not in result


def test_format_conversation_author_fallback_to_author_id() -> None:
    """format_conversation_for_prompt should fallback to author_id if name is empty."""
    message = MessageModel(
        author_id="user-123",
        author_name="",  # Empty string should fallback to author_id
        content="Test content",
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_id="msg-1",
    )

    result = format_conversation_for_prompt([message])

    assert "user-123: Test content" in result


def test_format_conversation_handles_empty_content() -> None:
    """format_conversation_for_prompt should handle empty or None content."""
    message = MessageModel(
        author_id="user-1",
        author_name="Alice",
        content="",  # Empty content
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        message_id="msg-1",
    )

    result = format_conversation_for_prompt([message])

    assert "Alice: " in result


# Tests for build_context_summary_prompt


def test_build_context_summary_prompt_with_facts_and_messages() -> None:
    """build_context_summary_prompt should include all sections."""
    result = build_context_summary_prompt(
        conversation="Alice: Hello",
        facts=["Fact 1", "Fact 2"],
        messages=["Message 1", "Message 2"],
    )

    assert "Alice: Hello" in result
    assert "Fact 1" in result
    assert "Fact 2" in result
    assert "Message 1" in result
    assert "Message 2" in result


def test_build_context_summary_prompt_empty_facts() -> None:
    """build_context_summary_prompt should handle empty facts list."""
    result = build_context_summary_prompt(
        conversation="Test conversation",
        facts=[],
        messages=["Some message"],
    )

    assert "(No facts retrieved)" in result
    assert "Some message" in result


def test_build_context_summary_prompt_empty_messages() -> None:
    """build_context_summary_prompt should handle empty messages list."""
    result = build_context_summary_prompt(
        conversation="Test conversation",
        facts=["Some fact"],
        messages=[],
    )

    assert "Some fact" in result
    assert "(No messages retrieved)" in result


# Tests for generate_context_summary (async)


@pytest.mark.asyncio
async def test_generate_context_summary_llm_unavailable() -> None:
    """generate_context_summary should return empty string when LLM is unavailable."""
    mock_llm = MagicMock()
    mock_llm.is_available = False

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == ""


@pytest.mark.asyncio
async def test_generate_context_summary_llm_is_none() -> None:
    """generate_context_summary should return empty string when LLM is None."""
    result = await generate_context_summary(
        llm=None,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == ""


@pytest.mark.asyncio
async def test_generate_context_summary_no_facts_or_messages() -> None:
    """generate_context_summary should return empty string if no facts or messages."""
    mock_llm = MagicMock()
    mock_llm.is_available = True

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=[],
        formatted_messages=[],
    )

    assert result == ""


@pytest.mark.asyncio
async def test_generate_context_summary_success() -> None:
    """generate_context_summary should return LLM response on success."""
    mock_llm = MagicMock()
    mock_llm.is_available = True
    mock_llm.agenerate_text = AsyncMock(return_value="Generated summary text")

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == "Generated summary text"
    mock_llm.agenerate_text.assert_called_once()


@pytest.mark.asyncio
async def test_generate_context_summary_retries_on_empty_response() -> None:
    """generate_context_summary should retry when LLM returns empty response."""
    mock_llm = MagicMock()
    mock_llm.is_available = True
    # Return empty twice, then a valid response
    mock_llm.agenerate_text = AsyncMock(side_effect=["", "  ", "Valid response"])

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == "Valid response"
    assert mock_llm.agenerate_text.call_count == 3


@pytest.mark.asyncio
async def test_generate_context_summary_returns_empty_after_max_retries() -> None:
    """generate_context_summary should return empty after exhausting retries."""
    mock_llm = MagicMock()
    mock_llm.is_available = True
    # Return empty on all attempts (max_retries=2 means 3 total attempts)
    mock_llm.agenerate_text = AsyncMock(return_value="   ")

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == ""
    assert mock_llm.agenerate_text.call_count == 3  # Initial + 2 retries


@pytest.mark.asyncio
async def test_generate_context_summary_handles_exception() -> None:
    """generate_context_summary should return empty string on exception."""
    mock_llm = MagicMock()
    mock_llm.is_available = True
    mock_llm.agenerate_text = AsyncMock(side_effect=RuntimeError("LLM failed"))

    result = await generate_context_summary(
        llm=mock_llm,
        conversation=[_make_message("Alice", "Hello")],
        formatted_facts=["Fact 1"],
        formatted_messages=["Message 1"],
    )

    assert result == ""
