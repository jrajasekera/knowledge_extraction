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


def _normalize_person_profile(output) -> list[RetrievedFact]:
    facts: list[RetrievedFact] = []
    for fact in output.facts:
        facts.append(
            _build_fact(
                person_id=output.person_id,
                person_name=output.name,
                fact_type=fact.type,
                fact_object=fact.object,
                attributes=fact.attributes,
                confidence=fact.confidence,
                evidence=fact.evidence,
                timestamp=fact.timestamp,
            )
        )
    return facts


def _normalize_people_by_skill(output) -> list[RetrievedFact]:
    facts: list[RetrievedFact] = []
    for person in output.people:
        attributes = {}
        if person.proficiency:
            attributes["proficiency"] = person.proficiency
        if person.years_experience is not None:
            attributes["years_experience"] = person.years_experience
        facts.append(
            _build_fact(
                person_id=person.person_id,
                person_name=person.name,
                fact_type="HAS_SKILL",
                fact_object=output.skill,
                attributes=attributes,
                confidence=person.confidence,
                evidence=person.evidence,
            )
        )
    return facts


def _normalize_people_by_org(output) -> list[RetrievedFact]:
    facts: list[RetrievedFact] = []
    for person in output.people:
        attributes = {}
        if person.role:
            attributes["role"] = person.role
        if person.start_date:
            attributes["start_date"] = person.start_date
        if person.end_date:
            attributes["end_date"] = person.end_date
        if person.location:
            attributes["location"] = person.location
        facts.append(
            _build_fact(
                person_id=person.person_id,
                person_name=person.name,
                fact_type="WORKS_AT",
                fact_object=output.organization,
                attributes=attributes,
                confidence=person.confidence,
                evidence=person.evidence,
            )
        )
    return facts


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


def _normalize_people_by_location(output) -> list[RetrievedFact]:
    facts = []
    for person in output.people:
        facts.append(
            _build_fact(
                person_id=person.person_id,
                person_name=person.name,
                fact_type=person.relationship,
                fact_object=output.location,
                attributes=person.details,
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
            )
        )
    return facts


def _normalize_conversation_participants(output) -> list[RetrievedFact]:
    facts: list[RetrievedFact] = []

    def build_key(person_id: str, mention_type: str, reference: str | None) -> tuple[str, str, str | None]:
        return (person_id, mention_type, reference)

    seen: set[tuple[str, str, str | None]] = set()

    for mention in output.explicit_mentions:
        if not mention.person_id:
            continue
        key = build_key(mention.person_id, "explicit", None)
        if key in seen:
            continue
        seen.add(key)
        attributes = {
            "mention_type": "explicit",
            "mentioned_in_message": mention.mentioned_in_message,
            "mentioned_name": mention.name,
        }
        facts.append(
            _build_fact(
                person_id=mention.person_id,
                person_name=mention.name or mention.person_id,
                fact_type="CONVERSATION_MENTION",
                fact_object=None,
                attributes=attributes,
                confidence=0.5,
                evidence=[],
            )
        )

    for reference in output.implicit_references:
        for guess in reference.possible_matches:
            key = build_key(guess.person_id, "implicit", reference.reference)
            if key in seen:
                continue
            seen.add(key)
            attributes = {
                "mention_type": "implicit",
                "reference": reference.reference,
                "reason": guess.reason,
            }
            if guess.confidence is not None:
                attributes["llm_confidence"] = guess.confidence
            facts.append(
                _build_fact(
                    person_id=guess.person_id,
                    person_name=guess.name or guess.person_id,
                    fact_type="CONVERSATION_MENTION",
                    fact_object=None,
                    attributes=attributes,
                    confidence=guess.confidence,
                    evidence=[],
                )
            )

    return facts


TOOL_NORMALIZERS = {
    "get_person_profile": _normalize_person_profile,
    "find_people_by_skill": _normalize_people_by_skill,
    "find_people_by_organization": _normalize_people_by_org,
    "find_people_by_topic": _normalize_people_by_topic,
    "get_person_timeline": _normalize_person_timeline,
    "find_people_by_location": _normalize_people_by_location,
    "semantic_search_facts": _normalize_semantic_search,
    "get_conversation_participants": _normalize_conversation_participants,
}
