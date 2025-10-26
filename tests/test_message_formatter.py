from __future__ import annotations

from memory_agent.message_formatter import format_message_for_embedding_text


def test_format_message_for_embedding_text_strips_markdown_and_mentions():
    text = format_message_for_embedding_text(
        author_name="Alice",
        content="**Hello** <@123> check [docs](https://example.com) now!",
        channel_name="general",
        guild_name="Playground",
        mentions=[{"id": "123", "name": "Bob"}],
    )
    assert "Alice" in text
    assert "@Bob" in text
    assert "docs" in text
    assert "**" not in text


def test_format_message_for_embedding_text_handles_empty_content():
    assert (
        format_message_for_embedding_text(
            author_name="Alice",
            content="   ",
            channel_name="random",
        )
        == ""
    )
