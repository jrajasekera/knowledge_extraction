"""Enhanced prompt construction with few-shot examples and guidance."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from .config import FACT_DEFINITION_INDEX
from .types import FactDefinition, FactType
from .windowing import MessageWindow

# Default location for prompt assets relative to this module
ASSET_PATH = Path(__file__).with_name("prompt_assets.json")

SYSTEM_PROMPT_V2 = """You are a specialized information extraction system analyzing Discord conversations.\n\nYour task is to identify factual, verifiable relationship statements and output them in structured JSON format.\n\nCore principles:\n1. ONLY extract facts explicitly stated or strongly implied by speakers\n2. Distinguish facts from speculation or joking statements\n3. Provide calibrated confidence scores between 0.0 and 1.0\n4. Prioritize precision over recall; skip uncertain facts\n5. Resolve people to the provided author_id values when identity is clear\n\nYou will see a focused conversation window. Extract only facts that can be confidently supported by the messages.\n"""


class PromptAssets(dict):
    """Thin wrapper to expose typed accessors."""

    @property
    def few_shot_examples(self) -> str:
        return self.get("few_shot_examples", "")

    @property
    def confidence_guidelines(self) -> str:
        return self.get("confidence_guidelines", "")

    @property
    def edge_case_guidance(self) -> str:
        return self.get("edge_case_guidance", "")


@lru_cache(maxsize=1)
def load_assets(path: Path = ASSET_PATH) -> PromptAssets:
    """Load reusable prompt fragments from JSON on first use."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):  # type: ignore[arg-type]
        raise TypeError(f"Prompt assets file {path} must contain a JSON object")
    return PromptAssets(data)  # type: ignore[arg-type]


def _format_fact_catalog(fact_types: Sequence[FactType]) -> str:
    lines: list[str] = []
    for fact_type in fact_types:
        definition: FactDefinition = FACT_DEFINITION_INDEX[fact_type]
        lines.append(f"**{definition.type.value}**")
        lines.append(f"  Subject: {definition.subject_description}")
        if definition.object_description:
            lines.append(f"  Object: {definition.object_description}")
        if definition.attributes:
            attr_lines = ", ".join(
                f"{attr.name}{' (required)' if attr.required else ''}"
                for attr in definition.attributes
            )
            lines.append(f"  Attributes: {attr_lines}")
        lines.append(f"  Purpose: {definition.rationale}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_participants(window: MessageWindow) -> str:
    participants: dict[str, str] = {}
    for record in window.messages:
        label = record.official_name or record.author_display
        participants.setdefault(record.author_id, label)
    lines = [f"- {label} (ID: {author_id})" for author_id, label in participants.items()]
    return "\n".join(lines)


def build_enhanced_prompt(
    window: MessageWindow,
    fact_types: Sequence[FactType],
    *,
    assets_path: Path | None = None,
) -> list[dict[str, str]]:
    """Construct the enhanced prompt payload for llama-server."""
    assets = load_assets(assets_path or ASSET_PATH)

    fact_catalog = _format_fact_catalog(fact_types)
    participants = _format_participants(window)

    user_sections = [
        "# Extraction Task",
        "You will analyze a Discord conversation window and extract structured facts.",
        "",
        "## Supported Fact Types",
        fact_catalog,
        "",
        assets.few_shot_examples,
        "",
        assets.confidence_guidelines,
        "",
        assets.edge_case_guidance,
        "",
        "## Output Format",
        "Return ONLY valid JSON matching this schema:",
        "```json",
        "{",
        "  \"facts\": [",
        "    {",
        "      \"type\": \"<FactType>\"",
        "      \"subject_id\": \"<exact_discord_id>\"",
        "      \"object_label\": \"<human_readable_name>\"",
        "      \"object_id\": \"<optional_stable_id>\"",
        "      \"attributes\": { \"key\": \"value\" }",
        "      \"confidence\": 0.0",
        "      \"evidence\": [\"<message_id>\"],",
        "      \"timestamp\": \"<ISO8601>\"",
        "      \"notes\": \"<optional_reasoning>\"",
        "    }",
        "  ]",
        "}",
        "```",
        "",
        "## Conversation to Analyze",
        "",
        "**Participants:**",
        participants,
        "",
        f"**Focus Message:** {window.focus.id}",
        "",
        "**Conversation:**",
        "```",
        window.as_text(),
        "```",
        "",
        "## Your Task",
        "1. Read the conversation carefully.",
        "2. Identify factual statements matching supported types.",
        "3. For each fact ensure confidence ≥ 0.5, use exact IDs, list all supporting message IDs, and add brief reasoning in notes.",
        "4. Output valid JSON only.",
        "5. Skip speculative, hypothetical, or joking statements.",
        "",
        "Remember: precision over recall. When in doubt, do not extract.",
    ]

    user_message = "\n".join(section for section in user_sections if section is not None)

    return [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_message},
    ]


__all__ = [
    "SYSTEM_PROMPT_V2",
    "build_enhanced_prompt",
    "load_assets",
]
