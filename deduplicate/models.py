from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ie.types import FactType


@dataclass(slots=True)
class FactRecord:
    id: int
    ie_run_id: int
    type: FactType
    subject_id: str
    subject_name: str | None
    object_label: str | None
    object_id: str | None
    object_type: str | None
    attributes: dict[str, Any]
    confidence: float
    evidence: list[str]
    timestamp: str
    notes: str | None = None


@dataclass(slots=True)
class Partition:
    fact_type: FactType
    subject_id: str
    subject_name: str | None
    fact_ids: list[int]

    def fact_count(self) -> int:
        return len(self.fact_ids)


@dataclass(slots=True)
class SimilarityPair:
    source_id: int
    target_id: int
    score: float

    def as_tuple(self) -> Tuple[int, int, float]:
        return (self.source_id, self.target_id, self.score)


SimilarityMatrix = Dict[str, List[SimilarityPair]]


@dataclass(slots=True)
class CandidateGroup:
    partition: Partition
    fact_ids: set[int]
    similarity: SimilarityMatrix = field(default_factory=dict)

    def sorted_fact_ids(self) -> List[int]:
        return sorted(self.fact_ids)

    def add_similarity(self, method: str, pairs: Iterable[SimilarityPair]) -> None:
        existing = self.similarity.setdefault(method, [])
        existing.extend(pairs)

    def has_multiple_facts(self) -> bool:
        return len(self.fact_ids) > 1


@dataclass(slots=True)
class CanonicalFact:
    type: FactType
    subject_id: str
    object_label: str | None
    object_id: str | None
    object_type: str | None
    attributes: dict[str, Any]
    confidence: float
    evidence: list[str]
    timestamp: str
    merged_from: list[int]
    merge_reasoning: str


@dataclass(slots=True)
class DeduplicationStats:
    run_id: int
    total_partitions: int = 0
    processed_partitions: int = 0
    facts_processed: int = 0
    facts_merged: int = 0
    candidate_groups_processed: int = 0
    average_group_size: float = 0.0
    elapsed_time: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "total_partitions": self.total_partitions,
            "processed_partitions": self.processed_partitions,
            "facts_processed": self.facts_processed,
            "facts_merged": self.facts_merged,
            "candidate_groups_processed": self.candidate_groups_processed,
            "average_group_size": self.average_group_size,
            "elapsed_time": self.elapsed_time,
        }


@dataclass(slots=True)
class PersistenceOutcome:
    canonical_fact_ids: list[int]
    deleted_fact_ids: list[int]


def flatten_similarity(similarity: SimilarityMatrix) -> Dict[str, List[Tuple[int, int, float]]]:
    return {
        method: [pair.as_tuple() for pair in pairs]
        for method, pairs in similarity.items()
    }
