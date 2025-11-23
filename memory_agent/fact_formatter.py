"""Utility helpers to normalize and format retrieved facts."""

from __future__ import annotations

from typing import Any, Iterable

from .models import RetrievedFact


def deduplicate_facts(facts: Iterable[RetrievedFact]) -> list[RetrievedFact]:
    """Remove duplicate facts by person, type, and object."""
    unique: dict[tuple[str, str, str | None, str | None], RetrievedFact] = {}
    for fact in facts:
        relationship_type = None
        if isinstance(fact.attributes, dict):
            relationship_type = str(fact.attributes.get("relationship_type")) if fact.attributes.get("relationship_type") else None
        key = (fact.person_id, fact.fact_type, fact.fact_object, relationship_type)
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
        # Filter out assessment metadata before formatting
        attribute_chunks = [
            f"{key}={value}"
            for key, value in fact.attributes.items()
            if value not in (None, "") and key != "assessment"
        ]
        if attribute_chunks:
            details.append(", ".join(attribute_chunks))
    descriptor = " ".join(details).strip()

    # Format evidence with author and snippets if available
    evidence_parts = []
    for e in fact.evidence:
        if hasattr(e, 'snippet') and e.snippet:
            # Truncate long snippets
            snippet = e.snippet if len(e.snippet) <= 500 else e.snippet[:497] + "..."
            # Include author if available
            if hasattr(e, 'author') and e.author:
                evidence_parts.append(f'{e.author}: "{snippet}"')
            else:
                evidence_parts.append(f'"{snippet}"')
        elif hasattr(e, 'source_id'):
            evidence_parts.append(e.source_id)
    evidence_text = " | ".join(evidence_parts) if evidence_parts else "unknown"

    confidence_text = f"{fact.confidence:.2f}" if fact.confidence is not None else "unknown"
    if descriptor:
        return f"{fact.person_name} {fact.fact_type.lower()} {descriptor} (confidence: {confidence_text}, evidence: {evidence_text})"
    return f"{fact.person_name} {fact.fact_type.lower()} (confidence: {confidence_text}, evidence: {evidence_text})"


def format_facts(facts: Iterable[RetrievedFact]) -> list[str]:
    """Format multiple facts into strings."""
    return [format_fact(fact) for fact in facts]


def format_fact_for_embedding_text(
    *,
    person_name: str,
    fact_type: str,
    fact_object: str | None,
    attributes: dict[str, Any],
    evidence_text: Iterable[str] | None = None,
) -> str:
    """Render a fact into a concise embedding-friendly string with optional evidence context."""
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

    result = base
    if attribute_items:
        result = f"{base}. " + ", ".join(attribute_items)

    # Include evidence messages if available
    if evidence_text:
        evidence_snippets = []
        for msg in evidence_text:
            if msg:
                evidence_snippets.append(msg)
        if evidence_snippets:
            evidence_str = " | ".join(evidence_snippets)
            result = f"{result}. Evidence: {evidence_str}"

    return result
