"""Utility helpers to normalize and format retrieved facts."""

from __future__ import annotations

from typing import Any, Iterable

from .models import RetrievedFact


def deduplicate_facts(facts: Iterable[RetrievedFact]) -> list[RetrievedFact]:
    """Remove duplicate facts by person, type, and object."""
    unique: dict[tuple[str, str, str | None], RetrievedFact] = {}
    for fact in facts:
        key = (fact.person_id, fact.fact_type, fact.fact_object)
        existing = unique.get(key)
        if existing is None:
            unique[key] = fact
            continue
        if (fact.confidence or 0) > (existing.confidence or 0):
            unique[key] = fact
    return list(unique.values())


def format_fact(fact: RetrievedFact) -> str:
    """Render a fact into a human readable sentence."""
    details = []
    if fact.fact_object:
        details.append(fact.fact_object)
    if fact.attributes:
        attribute_chunks = [f"{key}={value}" for key, value in fact.attributes.items() if value not in (None, "")]
        if attribute_chunks:
            details.append(", ".join(attribute_chunks))
    descriptor = " ".join(details).strip()
    evidence_ids = ", ".join(e.source_id for e in fact.evidence) if fact.evidence else "unknown"
    confidence_text = f"{fact.confidence:.2f}" if fact.confidence is not None else "unknown"
    if descriptor:
        return f"{fact.person_name} {fact.fact_type.lower()} {descriptor} (confidence: {confidence_text}, evidence: {evidence_ids})"
    return f"{fact.person_name} {fact.fact_type.lower()} (confidence: {confidence_text}, evidence: {evidence_ids})"


def format_facts(facts: Iterable[RetrievedFact]) -> list[str]:
    """Format multiple facts into strings."""
    return [format_fact(fact) for fact in facts]


def format_fact_for_embedding_text(
    *,
    person_name: str,
    fact_type: str,
    fact_object: str | None,
    attributes: dict[str, Any],
) -> str:
    """Render a fact into a concise embedding-friendly string."""
    cleaned_person = person_name or "Unknown person"
    relation = fact_type.replace("_", " ").lower()
    components = [cleaned_person, relation]
    if fact_object:
        components.append(str(fact_object))
    base = " ".join(component for component in components if component).strip()

    attribute_items = []
    for key, value in sorted(attributes.items()):
        if value in (None, "", []):
            continue
        attribute_items.append(f"{key}={value}")
    if attribute_items:
        return f"{base}. " + ", ".join(attribute_items)
    return base
