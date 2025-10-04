from pathlib import Path

from deduplicate.core import DeduplicationConfig, DeduplicationOrchestrator
from deduplicate.models import CandidateGroup, FactRecord, Partition, SimilarityPair
from deduplicate.similarity.grouping import CandidateGrouper
from ie.types import FactType


def _make_fact(fact_id: int) -> FactRecord:
    return FactRecord(
        id=fact_id,
        ie_run_id=1,
        type=FactType.WORKS_AT,
        subject_id="subject-1",
        subject_name="Subject",
        object_label="Org",
        object_id=None,
        object_type=None,
        attributes={"organization": "Org"},
        confidence=0.8,
        evidence=["msg-1"],
        timestamp="2024-01-01T00:00:00Z",
    )


def test_candidate_grouper_builds_connected_group() -> None:
    partition = Partition(
        fact_type=FactType.WORKS_AT,
        subject_id="subject-1",
        subject_name="Subject",
        fact_ids=[1, 2, 3],
    )
    facts = [_make_fact(1), _make_fact(2), _make_fact(3)]
    minhash_pairs = [SimilarityPair(1, 2, 0.9)]
    embedding_pairs = [SimilarityPair(2, 3, 0.88)]

    grouper = CandidateGrouper()
    groups = grouper.build_groups(
        partition,
        facts,
        minhash_pairs=minhash_pairs,
        embedding_pairs=embedding_pairs,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.fact_ids == {1, 2, 3}
    assert "minhash" in group.similarity
    assert "embedding" in group.similarity


def test_enforce_group_limits_splits_large_group() -> None:
    config = DeduplicationConfig(
        sqlite_path=Path("./discord.db"),
        neo4j_password="dummy",
        dry_run=True,
        max_group_size=2,
    )
    orchestrator = DeduplicationOrchestrator(config)

    partition = Partition(
        fact_type=FactType.WORKS_AT,
        subject_id="subject-1",
        subject_name="Subject",
        fact_ids=[1, 2, 3, 4],
    )

    facts = {
        1: _make_fact(1),
        2: _make_fact(2),
        3: _make_fact(3),
        4: _make_fact(4),
    }
    facts[1].attributes["organization"] = "OrgA"
    facts[2].attributes["organization"] = "OrgA"
    facts[3].attributes["organization"] = "OrgB"
    facts[4].attributes["organization"] = "OrgB"

    group = CandidateGroup(partition=partition, fact_ids={1, 2, 3, 4})
    limited = orchestrator._enforce_group_limits([group], facts)

    sizes = sorted(len(g.fact_ids) for g in limited)
    assert sizes == [2, 2]
