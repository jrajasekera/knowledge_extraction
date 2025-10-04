from deduplicate.llm.parser import CanonicalFactsParser
from deduplicate.models import FactRecord, Partition
from ie.types import FactType


def _make_fact(fact_id: int, *, label: str, evidence: list[str], timestamp: str) -> FactRecord:
    return FactRecord(
        id=fact_id,
        ie_run_id=1,
        type=FactType.WORKS_AT,
        subject_id="subject-1",
        subject_name="Subject",
        object_label=label,
        object_id=None,
        object_type="Organization",
        attributes={"organization": label},
        confidence=0.75,
        evidence=evidence,
        timestamp=timestamp,
    )


def test_parser_fills_missing_fields_from_sources() -> None:
    parser = CanonicalFactsParser()
    partition = Partition(
        fact_type=FactType.WORKS_AT,
        subject_id="subject-1",
        subject_name="Subject",
        fact_ids=[1, 2],
    )
    facts_by_id = {
        1: _make_fact(1, label="Org", evidence=["msg-1"], timestamp="2024-01-02T00:00:00Z"),
        2: _make_fact(2, label="Org Inc", evidence=["msg-2"], timestamp="2024-01-01T00:00:00Z"),
    }
    response = """
    {
      "canonical_facts": [
        {
          "type": "WORKS_AT",
          "subject_id": "subject-1",
          "attributes": {"organization": "Org"},
          "confidence": 0.9,
          "evidence": ["msg-1", "msg-2"],
          "timestamp": "",
          "merged_from": [1, 2],
          "merge_reasoning": "Normalized organization naming and combined evidence."
        }
      ]
    }
    """

    canonical = parser.parse(response, partition=partition, facts_by_id=facts_by_id)
    assert len(canonical) == 1
    fact = canonical[0]
    assert fact.object_label == "Org"
    assert fact.timestamp == "2024-01-01T00:00:00Z"
    assert set(fact.evidence) == {"msg-1", "msg-2"}
    assert fact.merged_from == [1, 2]
