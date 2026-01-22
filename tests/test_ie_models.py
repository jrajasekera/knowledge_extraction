"""Tests for ie/models.py."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ie.models import ExtractionFact, ExtractionResult
from ie.types import FactType


def test_extraction_fact_clamps_negative_confidence() -> None:
    """ExtractionFact should clamp negative confidence to 0.0."""
    fact = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        confidence=-0.5,
    )

    assert fact.confidence == 0.0


def test_extraction_fact_clamps_confidence_above_one() -> None:
    """ExtractionFact should clamp confidence > 1.0 to 1.0."""
    # Note: 0.8+ requires notes, 0.9+ requires evidence
    fact = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        confidence=1.5,
        notes="Clamping test",
        evidence=["msg-1"],
    )

    assert fact.confidence == 1.0


def test_extraction_fact_high_confidence_requires_notes() -> None:
    """ExtractionFact with confidence >= 0.8 must have notes."""
    with pytest.raises(ValidationError) as exc_info:
        ExtractionFact(
            type=FactType.WORKS_AT,
            subject_id="user-1",
            confidence=0.85,
        )

    assert "confidence ≥ 0.8 must include reasoning notes" in str(exc_info.value)


def test_extraction_fact_high_confidence_accepts_notes() -> None:
    """ExtractionFact with confidence >= 0.8 should accept valid notes."""
    fact = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        confidence=0.85,
        notes="User explicitly mentioned employment",
    )

    assert fact.confidence == 0.85
    assert fact.notes is not None


def test_extraction_fact_very_high_confidence_requires_evidence() -> None:
    """ExtractionFact with confidence >= 0.9 must have evidence."""
    with pytest.raises(ValidationError) as exc_info:
        ExtractionFact(
            type=FactType.WORKS_AT,
            subject_id="user-1",
            confidence=0.95,
            notes="Valid notes here",
            evidence=[],  # Empty evidence
        )

    assert "confidence ≥ 0.9 must include at least one evidence" in str(exc_info.value)


def test_extraction_fact_very_high_confidence_accepts_evidence() -> None:
    """ExtractionFact with confidence >= 0.9 should accept valid evidence."""
    fact = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        confidence=0.95,
        notes="User explicitly mentioned employment",
        evidence=["msg-123"],
    )

    assert fact.confidence == 0.95
    assert len(fact.evidence) == 1


def test_extraction_fact_empty_notes_string_fails() -> None:
    """ExtractionFact should reject whitespace-only notes at high confidence."""
    with pytest.raises(ValidationError) as exc_info:
        ExtractionFact(
            type=FactType.WORKS_AT,
            subject_id="user-1",
            confidence=0.85,
            notes="   ",  # Whitespace-only
        )

    assert "confidence ≥ 0.8 must include reasoning notes" in str(exc_info.value)


def test_extraction_result_deduplicates_facts() -> None:
    """ExtractionResult should remove duplicate facts."""
    fact1 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        confidence=0.5,
    )
    fact2 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        confidence=0.6,  # Different confidence, same key
    )
    fact3 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-2",  # Different user
        object_label="Acme",
        confidence=0.5,
    )

    result = ExtractionResult(facts=[fact1, fact2, fact3])

    # Should have 2 unique facts (user-1 and user-2)
    assert len(result.facts) == 2


def test_extraction_result_keeps_first_of_duplicates() -> None:
    """ExtractionResult should keep the first occurrence of duplicates."""
    fact1 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        confidence=0.5,
    )
    fact2 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        confidence=0.9,
        notes="Higher confidence",
        evidence=["msg-1"],
    )

    result = ExtractionResult(facts=[fact1, fact2])

    # Should keep the first one (fact1)
    assert len(result.facts) == 1
    assert result.facts[0].confidence == 0.5


def test_extraction_result_empty_facts() -> None:
    """ExtractionResult should accept empty facts list."""
    result = ExtractionResult(facts=[])

    assert result.facts == []


def test_extraction_fact_default_values() -> None:
    """ExtractionFact should have sensible defaults."""
    fact = ExtractionFact(
        type=FactType.HAS_SKILL,
        subject_id="user-1",
    )

    assert fact.object_label is None
    assert fact.object_id is None
    assert fact.attributes == {}
    assert fact.confidence == 0.0
    assert fact.evidence == []
    assert fact.timestamp is None
    assert fact.notes is None


def test_extraction_fact_with_attributes() -> None:
    """ExtractionFact should store attributes correctly."""
    fact = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme Corp",
        attributes={"role": "Engineer", "since": "2020"},
        confidence=0.7,
    )

    assert fact.attributes["role"] == "Engineer"
    assert fact.attributes["since"] == "2020"


def test_extraction_result_different_attributes_not_duplicates() -> None:
    """ExtractionResult should not deduplicate facts with different attributes."""
    fact1 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        attributes={"role": "Engineer"},
        confidence=0.5,
    )
    fact2 = ExtractionFact(
        type=FactType.WORKS_AT,
        subject_id="user-1",
        object_label="Acme",
        attributes={"role": "Manager"},  # Different role
        confidence=0.5,
    )

    result = ExtractionResult(facts=[fact1, fact2])

    # Should have 2 facts since attributes differ
    assert len(result.facts) == 2
