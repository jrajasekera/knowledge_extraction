"""Fact deduplication pipeline for the knowledge extraction project."""

from .core import DeduplicationConfig, DeduplicationOrchestrator
from .models import CandidateGroup, CanonicalFact, DeduplicationStats, FactRecord, Partition

__all__ = [
    "DeduplicationConfig",
    "DeduplicationOrchestrator",
    "CandidateGroup",
    "CanonicalFact",
    "DeduplicationStats",
    "FactRecord",
    "Partition",
]
