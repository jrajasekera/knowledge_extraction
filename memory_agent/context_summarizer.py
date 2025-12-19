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

    prompt = f"""\
You are given a conversation and two retrieval blocks (“Facts” and “Messages”). Your task is to write a detailed historical context for the **current conversation**, prioritizing the **last message**, while using relevant material from the retrieval blocks.

### Objective
Produce **1–5 paragraphs (≤2000 words)** that add **relevant historical context** to the current conversation.

### Inputs
- **Current Conversation**
```
{conversation}
```
- **Retrieved Facts**
```
{facts_section}
```
- **Retrieved Messages**
```
{messages_section}
```

### Method (follow in order)
1. **Anchor on the last message:** Identify its topic(s), people, and immediate needs. Start your response with a one-sentence framing that ties directly to the last message.
2. **Select supporting context:** From **Facts** and **Messages**, extract items that:
   - directly relate to the current topic/people/events
   - fill background gaps
   - add unique insights not already obvious in the conversation
   - could be useful for understanding or responding to the last message
3. **Resolve conflicts:**
   - Prefer **Facts** over **Messages**, unless a message adds unique, situational nuance.
   - When sources conflict, favor **more recent** items.
   - Prefer items that align with the current conversation’s focus.
4. **De-duplicate & compress:** Merge overlapping items. Remove redundant or already-obvious details.
5. **Quote precisely:** Use **quotes** or specific references. Attribute them like **(Fact: My example fact)** or **(Msg: My example message)**. Include names/dates if available.
6. **Note uncertainty:** If key info is missing or inconsistent, state it neutrally (e.g., “It’s unclear whether …”).
7. **No invention:** Do **not** introduce external facts unless absolutely necessary to clarify a term; if you must, mark it as general knowledge and keep it minimal.

### Inclusion Criteria
- Items roughly tied to the conversation’s topics, people, or events
- Background to understand the current exchange
- Messages that add unique insights not covered by Facts

### Exclusion Criteria
- Off-topic info
- Repetition of what is already clear from the conversation

### Output Requirements
- **Tone:** neutral, casual
- **Focus:** people mentioned, related past events, clarifying context—**especially for the last message**
- **Attribution:** inline references like (Fact: <fact content>)/(Msg: <message content>); include names/dates if available
- **If nothing relevant exists:** return "No relevant context found."
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
            "Generate highly detailed reports focusing only on information relevant to the current discussion."
        )

        # Retry up to 2 times if LLM returns empty/blank response
        max_retries = 2
        for attempt in range(max_retries + 1):
            summary = await llm.agenerate_text(prompt, system_message=system_message)
            summary = summary.strip()

            # Check if we got a non-empty response
            if summary:
                logger.info("Generated context summary on attempt %d: %s", attempt + 1, summary)
                return summary

            # Log empty response and retry if attempts remain
            if attempt < max_retries:
                logger.warning("LLM returned empty response on attempt %d, retrying...", attempt + 1)
            else:
                logger.warning("LLM returned empty response after %d attempts", max_retries + 1)

        return ""

    except Exception as exc:
        logger.error("Failed to generate context summary: %s", exc, exc_info=True)
        return ""


__all__ = [
    "format_conversation_for_prompt",
    "build_context_summary_prompt",
    "generate_context_summary",
]
