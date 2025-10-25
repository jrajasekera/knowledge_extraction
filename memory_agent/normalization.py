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


def _normalize_person_timeline(output) -> list[RetrievedFact]:
    facts = []
    for entry in output.timeline:
        attributes = dict(entry.attributes)
        if entry.start:
            attributes["start"] = entry.start
        if entry.end:
            attributes["end"] = entry.end
        facts.append(
            _build_fact(
                person_id=output.person_id,
                person_name=output.name,
                fact_type=entry.type,
                fact_object=entry.object,
                attributes=attributes,
                confidence=entry.confidence,
                evidence=entry.evidence,
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
            )
        )
    return facts


TOOL_NORMALIZERS = {
    "find_people_by_topic": _normalize_people_by_topic,
    "get_person_timeline": _normalize_person_timeline,
    "semantic_search_facts": _normalize_semantic_search,
}
