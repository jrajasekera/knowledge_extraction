"""Helpers to map tool outputs into RetrievedFact objects."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

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
    return RetrievedFact(
        person_id=person_id,
        person_name=person_name or person_id,
        fact_type=fact_type,
        fact_object=fact_object,
        attributes=attributes or {},
        confidence=confidence or 0.0,
        evidence=_normalize_evidence(evidence),
        timestamp=timestamp,
        similarity_score=similarity_score,
    )


def normalize_to_facts(tool_name: str, payload) -> list[RetrievedFact]:
    """Normalize tool outputs into RetrievedFact objects."""
    handler = TOOL_NORMALIZERS.get(tool_name)
    if handler is None:
        return []
    return handler(payload)


def _normalize_people_by_topic(output) -> list[RetrievedFact]:
    facts = []
    for person in output.people:
        attributes = {"relationship_type": person.relationship_type}
        if person.sentiment:
            attributes["sentiment"] = person.sentiment
        if person.details:
            attributes.update(person.details)
        facts.append(
            _build_fact(
                person_id=person.person_id,
                person_name=person.name,
                fact_type="TOPIC_RELATIONSHIP",
                fact_object=output.topic,
                attributes=attributes,
                confidence=person.confidence,
                evidence=person.evidence,
            )
        )
    return facts


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


def _normalize_person_profile(output) -> list[RetrievedFact]:
    facts = []
    for fact in output.facts:
        facts.append(
            _build_fact(
                person_id=fact.person_id,
                person_name=fact.person_name,
                fact_type=fact.fact_type,
                fact_object=fact.fact_object,
                attributes=fact.attributes,
                confidence=fact.confidence,
                evidence=fact.evidence,
                timestamp=fact.timestamp,
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
    "get_person_profile": _normalize_person_profile,
    "find_people_by_topic": _normalize_people_by_topic,
    "semantic_search_facts": _normalize_semantic_search,
    "semantic_search_messages": _normalize_semantic_search_messages,
}
