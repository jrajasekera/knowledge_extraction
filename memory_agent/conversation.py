"""Conversation analysis utilities for the memory agent."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from re import Pattern

from .models import MessageModel

MENTION_PATTERN: Pattern[str] = re.compile(r"<@!?([0-9]+)>")


@dataclass(slots=True)
class ConversationInsights:
    """Summary of entities and intent extracted from conversation."""

    people: set[str] = field(default_factory=set)
    organizations: set[str] = field(default_factory=set)
    topics: set[str] = field(default_factory=set)
    questions: list[str] = field(default_factory=list)
    hints: dict[str, str] = field(default_factory=dict)


def extract_insights(messages: Iterable[MessageModel]) -> ConversationInsights:
    """Extract rough entities and questions from conversation messages."""
    insights = ConversationInsights()
    for idx, message in enumerate(messages):
        matches = MENTION_PATTERN.findall(message.content)
        insights.people.update(matches)
        if "?" in message.content:
            insights.questions.append(message.content.strip())
        lower_content = message.content.lower()
        if "work" in lower_content and "at" in lower_content:
            insights.organizations.add(lower_content.split("at")[-1].strip())
        if "skill" in lower_content or "experience" in lower_content:
            insights.topics.add(lower_content)
        insights.hints[f"message_{idx}"] = message.content
    return insights
