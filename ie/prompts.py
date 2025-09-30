from __future__ import annotations

from typing import Sequence

from .config import FACT_DEFINITION_INDEX
from .types import FactDefinition, FactType
from .windowing import MessageRecord, MessageWindow

SYSTEM_PROMPT = (
    "You are an analyst extracting structured relationship facts from Discord conversations. "
    "Read the provided message window and return only validated facts in JSON format. "
    "Important: Participants may be referred to by their official names, nicknames, or other casual variants in the conversation text. "
    "Always resolve people to the provided author_id values when they are clearly identifiable."
)


def _alias_label(alias_type: str | None) -> str | None:
    if alias_type is None:
        return None
    cleaned = alias_type.replace("_", " ")
    if cleaned.lower() in {"nickname", "first name", "variation"}:
        return f"{cleaned} alias"
    return cleaned


def _collect_alias_entries(record: MessageRecord) -> list[tuple[str, str | None]]:
    seen: set[str] = set()
    entries: list[tuple[str, str | None]] = []

    def add(name: str | None, source: str | None) -> None:
        if not name:
            return
        cleaned = name.strip()
        if not cleaned:
            return
        key = cleaned.casefold()
        if key in seen:
            return
        seen.add(key)
        entries.append((cleaned, source))

    add(record.author_display, "Discord display")
    add(record.official_name, "official name")
    for alias in record.aliases:
        label = _alias_label(alias.alias_type)
        add(alias.name, label)

    return entries


def _generate_tag_handles(entries: list[tuple[str, str | None]], author_id: str) -> list[str]:
    handles: set[str] = {f"<@{author_id}>", f"<@!{author_id}>"}

    for name, _ in entries:
        variants: set[str] = set()
        stripped = name.strip()
        if not stripped:
            continue
        variants.add(stripped)
        variants.add(stripped.lower())
        collapsed = stripped.replace(" ", "")
        variants.add(collapsed)
        variants.add(collapsed.lower())
        first_word = stripped.split()[0]
        variants.add(first_word)
        variants.add(first_word.lower())
        for variant in variants:
            if variant:
                handles.add(f"@{variant}")

    return sorted(handles, key=str.casefold)


def build_participant_glossary(window: MessageWindow) -> str:
    participants: dict[str, MessageRecord] = {}
    for record in window.messages:
        participants.setdefault(record.author_id, record)

    lines: list[str] = []
    for author_id, record in participants.items():
        primary_label = record.official_name or record.author_display
        alias_entries = _collect_alias_entries(record)

        known_names: list[str] = []
        for name, source in alias_entries:
            if name == primary_label:
                continue
            if source:
                known_names.append(f"{name} ({source})")
            else:
                known_names.append(name)
        known_text = ", ".join(known_names) if known_names else "None"

        handles = _generate_tag_handles(alias_entries, author_id)
        handle_text = ", ".join(handles) if handles else "None"

        lines.append(
            f"- {primary_label} (author_id={author_id})\n"
            f"  Known names: {known_text}\n"
            f"  Mention handles: {handle_text}"
        )

    return "\n".join(lines)


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
    participant_text = build_participant_glossary(window)

    catalog_text = format_fact_catalog(fact_types)

    instructions = f"""
You will receive a short sequence of Discord messages (chronological). Each line includes author_id and message_id data. Use this context to extract zero or more structured facts from the supported catalogue. When you identify a fact:
1. Use the EXACT author_id values listed in Participants for subjects/objects when they refer to Discord members.
2. Resolve any names, nicknames, or aliases mentioned in the conversation to the matching participant listed in Participants when context makes the identity clear. If unsure, skip the fact.
3. Populate attributes using plain text (ISO dates preferred when possible).
4. Provide a confidence score between 0.0 and 1.0 (float). Only include facts you are at least 0.3 confident about.
5. Always include the message_ids that directly support the fact (at least the focus message).

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
