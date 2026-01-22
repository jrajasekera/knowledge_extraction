"""Helpers to map tool outputs into RetrievedFact objects."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from .models import RetrievedFact


def _normalize_evidence(evidence: Iterable[Any] | None) -> list[dict]:
    normalized: list[dict] = []
    if evidence is None:
        return normalized
    for entry in evidence:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            normalized.append({"source_id": str(entry)})
    return normalized


def _build_fact(
    person_id: str,
    person_name: str | None,
    fact_type: str,
    fact_object: str | None,
    attributes: dict | None,
    confidence: float | None,
    evidence: Iterable[Any] | None,
    timestamp: str | datetime | None = None,
    similarity_score: float | None = None,
) -> RetrievedFact:
    # Pydantic validators handle dict->FactEvidence and str->datetime conversions
    return RetrievedFact(
        person_id=person_id,
        person_name=person_name or person_id,
        fact_type=fact_type,
        fact_object=fact_object,
        attributes=attributes or {},
        confidence=confidence or 0.0,
        evidence=_normalize_evidence(evidence),  # type: ignore[arg-type]
        timestamp=timestamp,  # type: ignore[arg-type]
        similarity_score=similarity_score,
    )


def normalize_to_facts(tool_name: str, payload) -> list[RetrievedFact]:
    """Normalize tool outputs into RetrievedFact objects."""
    handler = TOOL_NORMALIZERS.get(tool_name)
    if handler is None:
        return []
    return handler(payload)


def _normalize_semantic_search(output) -> list[RetrievedFact]:
    facts = []
    for result in output.results:
        facts.append(
            _build_fact(
                person_id=result.person_id,
                person_name=result.person_name,
                fact_type=result.fact_type,
                fact_object=result.fact_object,
                attributes=result.attributes,
                confidence=result.confidence,
                evidence=result.evidence,
                similarity_score=result.similarity_score,
            )
        )
    return facts


def _normalize_semantic_search_messages(output) -> list[RetrievedFact]:
    facts = []
    for message in output.results:
        attributes = {
            "channel_id": message.channel_id,
            "channel_name": message.channel_name,
            "channel_topic": message.channel_topic,
            "guild_id": message.guild_id,
            "guild_name": message.guild_name,
            "permalink": message.permalink,
            "mentions": message.mentions,
            "mention_names": message.mention_names,
            "attachments": message.attachments,
            "reactions": message.reactions,
            "thread_id": message.thread_id,
            "message_type": message.message_type,
        }
        evidence = [
            {
                "source_id": message.message_id,
                "snippet": message.excerpt or message.clean_content or message.content,
                "url": message.permalink,
                "created_at": message.timestamp,
            }
        ]
        fact_object = message.excerpt or message.clean_content or message.content
        facts.append(
            _build_fact(
                person_id=message.author_id or "unknown",
                person_name=message.author_name or message.author_id or "unknown",
                fact_type="MESSAGE_SEARCH_RESULT",
                fact_object=fact_object,
                attributes={k: v for k, v in attributes.items() if v not in (None, [], {})},
                confidence=message.similarity_score,
                evidence=evidence,
                timestamp=message.timestamp,
                similarity_score=message.similarity_score,
            )
        )
    return facts


TOOL_NORMALIZERS = {
    "semantic_search_facts": _normalize_semantic_search,
    "semantic_search_messages": _normalize_semantic_search_messages,
}
