"""Helpers to normalize Discord messages for embedding generation."""

from __future__ import annotations

import re
from html import unescape
from typing import Mapping, Sequence


CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`]*)`")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
QUOTE_RE = re.compile(r"^>+\s?", re.MULTILINE)
MULTISPACE_RE = re.compile(r"\s+")


def _replace_mentions(text: str, mentions: Sequence[Mapping[str, str | None]] | None) -> str:
    if not mentions:
        return text
    updated = text
    for mention in mentions:
        mention_id = str(mention.get("id") or "")
        if not mention_id:
            continue
        name = (mention.get("name") or mention_id).strip()
        replacement = f"@{name}" if name else f"@{mention_id}"
        updated = updated.replace(f"<@{mention_id}>", replacement)
        updated = updated.replace(f"<@!{mention_id}>", replacement)
    return updated


def _replace_channel_tags(text: str, channel_id: str | None, channel_name: str | None) -> str:
    if not channel_id:
        return text
    replacement = f"#{channel_name}" if channel_name else f"#channel-{channel_id}"
    return text.replace(f"<#{channel_id}>", replacement)


def _strip_markdown(text: str) -> str:
    without_code = CODE_BLOCK_RE.sub(" ", text)
    without_inline = INLINE_CODE_RE.sub(r"\1", without_code)
    without_links = MARKDOWN_LINK_RE.sub(r"\1", without_inline)
    without_quotes = QUOTE_RE.sub("", without_links)
    cleaned = without_quotes.replace("**", "").replace("__", "").replace("*", "").replace("_", "").replace("~~", "")
    return unescape(cleaned)


def _collapse_whitespace(text: str) -> str:
    return MULTISPACE_RE.sub(" ", text).strip()


def _unique_mention_names(mentions: Sequence[Mapping[str, str | None]] | None) -> list[str]:
    if not mentions:
        return []
    seen: set[str] = set()
    names: list[str] = []
    for mention in mentions:
        name = (mention.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def format_message_for_embedding_text(
    *,
    author_name: str | None,
    content: str | None,
    channel_name: str | None = None,
    channel_topic: str | None = None,
    guild_name: str | None = None,
    mentions: Sequence[Mapping[str, str | None]] | None = None,
    channel_id: str | None = None,
) -> str:
    """Normalize a Discord message into a compact embedding-friendly string."""

    if not content:
        return ""
    text = _replace_mentions(content, mentions)
    text = _replace_channel_tags(text, channel_id, channel_name)
    text = _strip_markdown(text)
    text = _collapse_whitespace(text)
    if not text:
        return ""

    speaker = author_name or "Unknown author"
    context_bits: list[str] = []
    if channel_name:
        context_bits.append(f"#{channel_name}")
    elif channel_id:
        context_bits.append(f"channel {channel_id}")
    if guild_name:
        context_bits.append(guild_name)

    location = ""
    if context_bits:
        location = " in " + " / ".join(context_bits)

    parts = [f"{speaker}{location} said: {text}"]
    if channel_topic:
        topic = channel_topic.strip()
        if topic:
            parts.append(f"Topic: {topic}")

    mention_names = _unique_mention_names(mentions)
    if mention_names:
        parts.append("Mentions: " + ", ".join(mention_names))

    return " ".join(parts).strip()


__all__ = ["format_message_for_embedding_text"]
