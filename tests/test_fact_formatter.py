"""Tests for memory_agent/fact_formatter.py."""

from __future__ import annotations

from memory_agent.fact_formatter import (
    deduplicate_facts,
    format_fact,
    format_fact_for_embedding_text,
    format_facts,
)
from memory_agent.models import FactEvidence, RetrievedFact


def _make_fact(
    person_id: str = "p1",
    person_name: str = "Alice",
    fact_type: str = "WORKS_AT",
    fact_object: str | None = "Acme",
    confidence: float = 0.8,
    attributes: dict | None = None,
    evidence: list | None = None,
) -> RetrievedFact:
    """Helper to create test RetrievedFact instances."""
    return RetrievedFact(
        person_id=person_id,
        person_name=person_name,
        fact_type=fact_type,
        fact_object=fact_object,
        confidence=confidence,
        attributes=attributes or {},
        evidence=evidence or [],
    )


# Tests for deduplicate_facts


def test_deduplicate_facts_empty_input() -> None:
    """deduplicate_facts should handle empty input."""
    result = deduplicate_facts([])
    assert result == []


def test_deduplicate_facts_single_fact() -> None:
    """deduplicate_facts should return single fact unchanged."""
    fact = _make_fact()
    result = deduplicate_facts([fact])

    assert len(result) == 1
    assert result[0] is fact


def test_deduplicate_facts_removes_duplicates() -> None:
    """deduplicate_facts should remove facts with same person/type/object."""
    fact1 = _make_fact(confidence=0.5)
    fact2 = _make_fact(confidence=0.6)  # Same key, different confidence

    result = deduplicate_facts([fact1, fact2])

    assert len(result) == 1


def test_deduplicate_facts_keeps_higher_confidence() -> None:
    """deduplicate_facts should keep the fact with higher confidence."""
    fact1 = _make_fact(confidence=0.5)
    fact2 = _make_fact(confidence=0.9)

    result = deduplicate_facts([fact1, fact2])

    assert len(result) == 1
    assert result[0].confidence == 0.9


def test_deduplicate_facts_different_objects_kept() -> None:
    """deduplicate_facts should keep facts with different objects."""
    fact1 = _make_fact(fact_object="Acme")
    fact2 = _make_fact(fact_object="BigCorp")

    result = deduplicate_facts([fact1, fact2])

    assert len(result) == 2


def test_deduplicate_facts_relationship_type_in_key() -> None:
    """deduplicate_facts should consider relationship_type attribute in key."""
    fact1 = _make_fact(
        fact_type="RELATED_TO",
        fact_object="Bob",
        attributes={"relationship_type": "friend"},
    )
    fact2 = _make_fact(
        fact_type="RELATED_TO",
        fact_object="Bob",
        attributes={"relationship_type": "colleague"},
    )

    result = deduplicate_facts([fact1, fact2])

    # Different relationship types should be kept as separate
    assert len(result) == 2


# Tests for format_fact


def test_format_fact_basic() -> None:
    """format_fact should format a basic fact."""
    fact = _make_fact(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme Corp",
        confidence=0.85,
    )

    result = format_fact(fact)

    assert "Alice" in result
    assert "works_at" in result
    assert "Acme Corp" in result
    assert "0.85" in result


def test_format_fact_with_attributes() -> None:
    """format_fact should include attributes."""
    fact = _make_fact(
        attributes={"role": "Engineer", "since": "2020"},
    )

    result = format_fact(fact)

    assert "role=Engineer" in result
    assert "since=2020" in result


def test_format_fact_filters_assessment_attribute() -> None:
    """format_fact should filter out 'assessment' from attributes."""
    fact = _make_fact(
        attributes={"role": "Engineer", "assessment": {"score": 0.9}},
    )

    result = format_fact(fact)

    assert "role=Engineer" in result
    assert "assessment" not in result


def test_format_fact_filters_empty_attributes() -> None:
    """format_fact should filter out None and empty string attributes."""
    fact = _make_fact(
        attributes={"role": "Engineer", "location": None, "notes": ""},
    )

    result = format_fact(fact)

    assert "role=Engineer" in result
    assert "location" not in result
    assert "notes" not in result


def test_format_fact_with_evidence_snippets() -> None:
    """format_fact should include evidence snippets."""
    evidence = [
        FactEvidence(source_id="msg-1", snippet="I work at Acme", author="Alice"),
    ]
    fact = _make_fact(evidence=evidence)

    result = format_fact(fact)

    assert 'Alice: "I work at Acme"' in result


def test_format_fact_truncates_long_snippets() -> None:
    """format_fact should truncate snippets longer than 500 chars."""
    long_snippet = "x" * 600
    evidence = [FactEvidence(source_id="msg-1", snippet=long_snippet)]
    fact = _make_fact(evidence=evidence)

    result = format_fact(fact)

    assert "..." in result
    # Should not contain the full 600 character snippet
    assert "x" * 600 not in result


def test_format_fact_with_evidence_source_id_only() -> None:
    """format_fact should use source_id when snippet not available."""
    evidence = [FactEvidence(source_id="msg-123")]
    fact = _make_fact(evidence=evidence)

    result = format_fact(fact)

    assert "msg-123" in result


def test_format_fact_no_object() -> None:
    """format_fact should handle facts without object."""
    fact = _make_fact(fact_object=None)

    result = format_fact(fact)

    assert "Alice works_at" in result


def test_format_fact_zero_confidence() -> None:
    """format_fact should handle zero confidence."""
    fact = RetrievedFact(
        person_id="p1",
        person_name="Alice",
        fact_type="WORKS_AT",
        confidence=0.0,
        evidence=[],
    )

    result = format_fact(fact)

    assert "confidence: 0.00" in result


# Tests for format_facts


def test_format_facts_multiple() -> None:
    """format_facts should format multiple facts."""
    facts = [
        _make_fact(person_name="Alice", fact_object="Acme"),
        _make_fact(person_name="Bob", fact_object="BigCorp"),
    ]

    result = format_facts(facts)

    assert len(result) == 2
    assert any("Alice" in r for r in result)
    assert any("Bob" in r for r in result)


def test_format_facts_empty() -> None:
    """format_facts should handle empty input."""
    result = format_facts([])
    assert result == []


# Tests for format_fact_for_embedding_text


def test_format_fact_for_embedding_basic() -> None:
    """format_fact_for_embedding_text should create embedding-friendly text."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme Corp",
        attributes={},
    )

    assert "Alice" in result
    assert "works at" in result
    assert "Acme Corp" in result


def test_format_fact_for_embedding_with_attributes() -> None:
    """format_fact_for_embedding_text should include sorted attributes."""
    result = format_fact_for_embedding_text(
        person_name="Bob",
        fact_type="HAS_SKILL",
        fact_object="Python",
        attributes={"years": 5, "level": "expert"},
    )

    # Attributes should be sorted
    assert "level=expert" in result
    assert "years=5" in result


def test_format_fact_for_embedding_filters_empty_attributes() -> None:
    """format_fact_for_embedding_text should filter empty attribute values."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={"role": "Engineer", "empty": None, "blank": "", "empty_list": []},
    )

    assert "role=Engineer" in result
    assert "empty" not in result or "empty=" not in result
    assert "blank" not in result
    assert "empty_list" not in result


def test_format_fact_for_embedding_no_object() -> None:
    """format_fact_for_embedding_text should handle None object."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="HAS_SKILL",
        fact_object=None,
        attributes={},
    )

    assert "Alice" in result
    assert "has skill" in result


def test_format_fact_for_embedding_unknown_person() -> None:
    """format_fact_for_embedding_text should fallback for None person_name."""
    result = format_fact_for_embedding_text(
        person_name=None,  # type: ignore[arg-type]
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
    )

    assert "Unknown person" in result


def test_format_fact_for_embedding_with_evidence() -> None:
    """format_fact_for_embedding_text should include evidence text."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        evidence_text=["I started at Acme last year", "Great company"],
    )

    assert "Evidence:" in result
    assert "I started at Acme last year" in result
    assert "Great company" in result


def test_format_fact_for_embedding_filters_empty_evidence() -> None:
    """format_fact_for_embedding_text should filter empty evidence strings."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        evidence_text=["Valid message", "", None, "Another message"],  # type: ignore[list-item]
    )

    assert "Evidence:" in result
    assert "Valid message" in result
    assert "Another message" in result


def test_format_fact_for_embedding_no_evidence() -> None:
    """format_fact_for_embedding_text should not include Evidence section if empty."""
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="Acme",
        attributes={},
        evidence_text=None,
    )

    assert "Evidence:" not in result
