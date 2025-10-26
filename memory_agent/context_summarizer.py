"""Generate context summaries from retrieved facts and messages using LLM."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .llm import LLMClient
    from .models import MessageModel

logger = logging.getLogger(__name__)


def format_conversation_for_prompt(messages: list[MessageModel], max_messages: int = 10) -> str:
    """Format recent conversation messages for the LLM prompt.

    Args:
        messages: List of conversation messages
        max_messages: Maximum number of recent messages to include

    Returns:
        Formatted string with conversation history
    """
    if not messages:
        return "(No conversation messages)"

    # Take the most recent messages
    recent = messages[-max_messages:]

    formatted_lines = []
    for msg in recent:
        author = msg.author_name or msg.author_id or "Unknown"
        content = msg.content or ""
        formatted_lines.append(f"{author}: {content}")

    return "\n".join(formatted_lines)


def build_context_summary_prompt(
    conversation: str,
    facts: list[str],
    messages: list[str],
) -> str:
    """Build the LLM prompt for generating context summary.

    Args:
        conversation: Formatted conversation string
        facts: List of formatted facts
        messages: List of formatted retrieved messages

    Returns:
        Complete prompt string for the LLM
    """
    facts_section = "\n".join(facts) if facts else "(No facts retrieved)"
    messages_section = "\n".join(messages) if messages else "(No messages retrieved)"

    prompt = f"""Analyze the conversation below and provide relevant historical context using the retrieved facts and messages.

## Current Conversation
{conversation}

## Retrieved Facts
{facts_section}

## Retrieved Messages
{messages_section}

## Instructions
Generate a summary (1-3 paragraphs, max 200 words) that provides historical context for the current conversation. \
Provide as much relevant information as possible while being concise and natural. \
Focus primarily on the last message in the conversation, but consider the full context.

**Inclusion Criteria:**
- Information directly related to topics, people, or events in the current conversation
- Info that provide essential background
- Messages with unique insights not covered by facts

**Exclusion Criteria:**
- Off-topic or tangentially related information
- Ambiguous details
- Redundant information already clear from the conversation

**Conflict Resolution:**
- Prioritize facts over messages unless messages provide unique context
- Favor more recent information when sources conflict
- Prefer information that aligns with the current conversation

**Output Requirements:**
- Use neutral, professional tone
- Note uncertainties if information is incomplete
- Return empty response if no relevant context exists
- Avoid jargon unless used in the conversation
- Focus on: people mentioned, related past events, and clarifying context

Provide only the summary without preamble or meta-commentary.
"""

    return prompt


async def generate_context_summary(
    llm: LLMClient | None,
    conversation: list[MessageModel],
    formatted_facts: list[str],
    formatted_messages: list[str],
) -> str:
    """Generate a context summary using the LLM.

    Args:
        llm: LLM client instance (can be None)
        conversation: Original conversation messages from the request
        formatted_facts: List of formatted fact strings
        formatted_messages: List of formatted message strings

    Returns:
        Generated context summary string (empty string on failure or if LLM unavailable)
    """
    # Return empty if LLM is unavailable
    if not llm or not llm.is_available:
        logger.debug("LLM unavailable, skipping context summary generation")
        return ""

    # Return empty if no facts or messages to summarize
    if not formatted_facts and not formatted_messages:
        logger.debug("No facts or messages to summarize")
        return ""

    try:
        # Format the conversation
        conversation_text = format_conversation_for_prompt(conversation)

        # Build the prompt
        prompt = build_context_summary_prompt(
            conversation=conversation_text,
            facts=formatted_facts,
            messages=formatted_messages,
        )

        logger.debug("Generating context summary with prompt length: %d chars", len(prompt))

        # Call the LLM using the general text generation method
        system_message = (
            "You are a helpful assistant that analyzes conversations and provides relevant historical context. "
            "Generate concise, natural paragraph summaries focusing only on information relevant to the current discussion."
        )
        summary = await llm.agenerate_text(prompt, system_message=system_message)

        # Clean up the response
        summary = summary.strip()

        logger.info("Generated context summary: %s", summary)
        return summary

    except Exception as exc:
        logger.error("Failed to generate context summary: %s", exc, exc_info=True)
        return ""


__all__ = [
    "format_conversation_for_prompt",
    "build_context_summary_prompt",
    "generate_context_summary",
]
