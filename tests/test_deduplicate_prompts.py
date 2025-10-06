from __future__ import annotations

from deduplicate.llm.prompts import build_messages
from deduplicate.models import FactRecord, Partition
from ie.types import FactType


def _fact_record(
    *,
    fact_id: int,
    fact_type: FactType,
    subject_id: str = "user_123",
    subject_name: str | None = "Alex",
    object_label: str | None = "Example",
    object_type: str | None = None,
    attributes: dict[str, object] | None = None,
) -> FactRecord:
    return FactRecord(
        id=fact_id,
        ie_run_id=1,
        type=fact_type,
        subject_id=subject_id,
        subject_name=subject_name,
        object_label=object_label,
        object_id=None,
        object_type=object_type,
        attributes=attributes or {},
        confidence=0.7,
        evidence=["msg1"],
        timestamp="2024-01-01T00:00:00Z",
    )


def test_build_messages_includes_type_specific_guidance() -> None:
    fact = _fact_record(
        fact_id=1,
        fact_type=FactType.WORKS_AT,
        object_label="Google",
        object_type="Organization",
        attributes={"organization": "Google", "start_date": "2023-01-01"},
    )
    partition = Partition(
        fact_type=FactType.WORKS_AT,
        subject_id="user_123",
        subject_name="Alex",
        fact_ids=[1],
    )

    messages = build_messages(partition, [fact])

    assert messages[0]["content"].startswith(
        "You are a knowledge graph deduplication specialist working on a Discord conversation analysis system."
    )
    user_prompt = messages[1]["content"]
    assert "WORKS_AT Rules" in user_prompt
    assert "Confidence Calculation Rules" in user_prompt
    assert "Few-Shot Examples:" in user_prompt
    assert "Example: clear_duplicate_works_at" in user_prompt


def test_build_messages_uses_generic_guidance_when_missing_fact_specific_rules() -> None:
    fact = _fact_record(
        fact_id=2,
        fact_type=FactType.CLOSE_TO,
        object_label="Jordan",
        object_type="Person",
        attributes={"relationship_type": "friend"},
    )
    partition = Partition(
        fact_type=FactType.CLOSE_TO,
        subject_id="user_123",
        subject_name=None,
        fact_ids=[2],
    )

    messages = build_messages(partition, [fact])

    user_prompt = messages[1]["content"]
    assert "Generic Deduplication Guidance" in user_prompt
    assert "Quality Validation Checklist" in user_prompt
    assert "Output Schema" in user_prompt
    assert "Example:" in user_prompt


def test_build_messages_includes_attribute_priorities() -> None:
    fact = _fact_record(
        fact_id=3,
        fact_type=FactType.LIVES_IN,
        object_label="San Francisco",
        object_type="Place",
        attributes={"location": "San Francisco"},
    )
    partition = Partition(
        fact_type=FactType.LIVES_IN,
        subject_id="user_123",
        subject_name="Alex",
        fact_ids=[3],
    )

    messages = build_messages(partition, [fact])

    user_prompt = messages[1]["content"]
    assert "Attribute Priorities" in user_prompt
    assert "Critical: location" in user_prompt
