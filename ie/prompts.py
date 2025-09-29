from __future__ import annotations

from typing import Iterable, Sequence

from .config import FACT_DEFINITION_INDEX
from .types import FactDefinition, FactType
from .windowing import MessageWindow

SYSTEM_PROMPT = (
    "You are an analyst extracting structured relationship facts from Discord conversations. "
    "Read the provided message window and return only validated facts in JSON format."
)


def format_fact_catalog(fact_types: Sequence[FactType]) -> str:
    lines: list[str] = []
    for fact_type in fact_types:
        definition: FactDefinition = FACT_DEFINITION_INDEX[fact_type]
        attr_lines = []
        for attr in definition.attributes:
            required = "(required)" if attr.required else "(optional)"
            attr_lines.append(f"      - {attr.name}: {attr.description} {required}")
        attr_text = "\n".join(attr_lines) if attr_lines else "      - (no attributes)"
        object_hint = (
            f"    Object ({definition.object_type or 'None'}): {definition.object_description or 'n/a'}\n"
            if definition.object_description
            else "    Object: none\n"
        )
        lines.append(
            f"- {definition.type}:\n"
            f"    Subject: {definition.subject_description}\n"
            f"{object_hint}"
            f"    Attributes:\n{attr_text}\n"
            f"    Rationale: {definition.rationale}"
        )
    return "\n".join(lines)


def build_messages(window: MessageWindow, fact_types: Sequence[FactType]) -> list[dict[str, str]]:
    participants = {}
    for record in window.messages:
        participants.setdefault(record.author_id, record.author_display)

    participant_lines = [
        f"- {name} (author_id={author_id})" for author_id, name in participants.items()
    ]
    participant_text = "\n".join(participant_lines)

    catalog_text = format_fact_catalog(fact_types)

    instructions = f"""
You will receive a short sequence of Discord messages (chronological). Each line includes author_id and message_id data. Use this context to extract zero or more structured facts from the supported catalogue. When you identify a fact:
1. Use the EXACT author_id values listed in Participants for subjects/objects when they refer to Discord members.
2. Populate attributes using plain text (ISO dates preferred when possible).
3. Provide a confidence score between 0.0 and 1.0 (float). Only include facts you are at least 0.3 confident about.
4. Always include the message_ids that directly support the fact (at least the focus message).

Return only valid JSON matching the schema:
{{
  "facts": [
    {{
      "type": "<FactType>",
      "subject_id": "<discord_author_id>",
      "object_label": "<human readable object or second person>",
      "object_id": "<optional stable identifier or null>",
      "attributes": {{ "key": "value" }},
      "confidence": 0.0,
      "evidence": ["<message_id>", ...],
      "timestamp": "<ISO8601 timestamp for primary evidence>",
      "notes": "<optional clarifications>"
    }}
  ]
}}

Ensure the JSON is parseable with double quotes and no trailing comments.

Supported fact types:
{catalog_text}

Participants:
{participant_text}

Focus message: {window.focus.id}

Conversation:
{window.as_text()}
"""

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instructions.strip()},
    ]


def window_hint(window: MessageWindow) -> str:
    return f"channel={window.focus.channel_id} message={window.focus.id}"
