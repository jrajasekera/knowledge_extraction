from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from neo4j import GraphDatabase

from facts_to_graph import materialize_facts
from ie.client import LlamaServerConfig
from ie.config import FACT_DEFINITION_INDEX
from ie.types import FactType

from .llm.client import DeduplicationLLMClient
from .llm.parser import CanonicalFactsParser
from .models import CanonicalFact, CandidateGroup, DeduplicationStats, FactRecord, Partition
from .partitioning import FactPartitioner
from .persistence import DeduplicationPersistence
from .progress import DeduplicationProgress
from .similarity.embeddings import EmbeddingConfig, EmbeddingSimilarityDetector
from .similarity.grouping import CandidateGrouper
from .similarity.minhash_lsh import MinHashConfig, MinHashLSHDetector

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DeduplicationConfig:
    sqlite_path: Path
    neo4j_password: str
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    minhash_threshold: float = 0.7
    minhash_num_perm: int = 128
    minhash_ngram_size: int = 3
    embedding_model: str = "google/embeddinggemma-300m"
    embedding_threshold: float = 0.85
    embedding_batch_size: int = 32
    embedding_device: str | None = None
    embedding_max_neighbors: int = 25
    llm_config: LlamaServerConfig = field(default_factory=LlamaServerConfig)
    llm_max_retries: int = 3
    min_confidence: float = 0.5
    resume: bool = False
    dry_run: bool = False
    max_partitions: int | None = None
    graph_delete_batch_size: int = 100
    max_group_size: int = 25

    def validate(self) -> "DeduplicationConfig":
        if self.minhash_threshold <= 0.0 or self.minhash_threshold > 1.0:
            raise ValueError("minhash_threshold must be within (0, 1]")
        if self.embedding_threshold <= 0.0 or self.embedding_threshold > 1.0:
            raise ValueError("embedding_threshold must be within (0, 1]")
        if self.minhash_num_perm <= 0:
            raise ValueError("minhash_num_perm must be > 0")
        if self.embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be > 0")
        if self.embedding_max_neighbors <= 0:
            raise ValueError("embedding_max_neighbors must be > 0")
        return self


class DeduplicationOrchestrator:
    def __init__(self, config: DeduplicationConfig) -> None:
        self.config = config.validate()
        self._attribute_index: Mapping[FactType, tuple[str, ...]] = {
            fact_type: definition.attribute_names()
            for fact_type, definition in FACT_DEFINITION_INDEX.items()
        }

    def run(self) -> DeduplicationStats:
        start_time = time.time()
        conn = sqlite3.connect(str(self.config.sqlite_path))
        conn.row_factory = sqlite3.Row
        partitioner = FactPartitioner(conn, min_confidence=self.config.min_confidence)
        progress = DeduplicationProgress(conn)
        persistence = DeduplicationPersistence(conn, dry_run=self.config.dry_run)

        minhash_detector = MinHashLSHDetector(
            self._attribute_index,
            MinHashConfig(
                threshold=self.config.minhash_threshold,
                num_perm=self.config.minhash_num_perm,
                ngram_size=self.config.minhash_ngram_size,
            ),
        )
        embedding_detector = EmbeddingSimilarityDetector(
            self._attribute_index,
            EmbeddingConfig(
                model_name=self.config.embedding_model,
                threshold=self.config.embedding_threshold,
                batch_size=self.config.embedding_batch_size,
                device=self.config.embedding_device,
                max_neighbors=self.config.embedding_max_neighbors,
            ),
        )
        grouper = CandidateGrouper()
        llm_parser = CanonicalFactsParser()

        run_id, total_partitions = self._prepare_run(progress, partitioner)
        stats = DeduplicationStats(run_id=run_id, total_partitions=total_partitions)
        completed_keys = progress.completed_partitions(run_id)
        deleted_fact_ids: set[int] = set()
        canonical_fact_ids: set[int] = set()
        group_size_sum = 0
        failure_detected = False

        partition_iterator = partitioner.iter_partitions(limit=self.config.max_partitions)

        with DeduplicationLLMClient(
            self.config.llm_config,
            parser=llm_parser,
            max_retries=self.config.llm_max_retries,
        ) as llm_client:
            for selection in partition_iterator:
                partition = selection.partition
                key = (partition.fact_type, partition.subject_id)
                if key in completed_keys:
                    logger.info(
                        "Skipping partition %s/%s (already completed)",
                        partition.fact_type.value,
                        partition.subject_id,
                    )
                    continue

                logger.info(
                    "Processing partition %s/%s (%d facts)",
                    partition.fact_type.value,
                    partition.subject_id,
                    len(selection.facts),
                )

                progress.ensure_partition_entry(run_id, partition)
                progress.mark_partition_status(run_id, partition, "in_progress")

                stats.facts_processed += len(selection.facts)

                facts_lookup = {fact.id: fact for fact in selection.facts}
                minhash_pairs = minhash_detector.find_candidate_pairs(selection.facts)
                embedding_pairs = embedding_detector.find_candidate_pairs(selection.facts)
                candidate_groups = grouper.build_groups(
                    partition,
                    selection.facts,
                    minhash_pairs=minhash_pairs,
                    embedding_pairs=embedding_pairs,
                )
                candidate_groups = self._enforce_group_limits(
                    candidate_groups,
                    facts_lookup,
                )

                if not candidate_groups:
                    logger.info(
                        "No candidate groups identified for partition %s/%s; skipping LLM invocation.",
                        partition.fact_type.value,
                        partition.subject_id,
                    )
                    progress.mark_partition_status(run_id, partition, "completed")
                    progress.update_totals(
                        run_id,
                        processed_partitions=1,
                        facts_processed=len(selection.facts),
                        facts_merged=0,
                        candidate_groups_processed=0,
                    )
                    stats.processed_partitions += 1
                    continue

                partition_group_count = len(candidate_groups)
                partition_group_size_sum = sum(len(group.fact_ids) for group in candidate_groups)

                logger.info(
                    "Evaluating %d candidate groups (avg size %.1f) for partition %s/%s",
                    partition_group_count,
                    partition_group_size_sum / partition_group_count
                    if partition_group_count
                    else 0.0,
                    partition.fact_type.value,
                    partition.subject_id,
                )

                partition_deleted: list[int] = []
                partition_canonical: list[int] = []
                partition_merge_delta = 0
                partition_failed = False

                for group in candidate_groups:
                    self._log_group_preview(group, facts_lookup)
                    try:
                        canonical_facts = llm_client.generate_canonical_facts(group, facts_lookup)
                    except Exception as exc:  # noqa: BLE001 - surface all issues
                        logger.exception(
                            "LLM processing failed for partition %s/%s: %s",
                            partition.fact_type.value,
                            partition.subject_id,
                            exc,
                        )
                        partition_failed = True
                        break

                    source_facts = [facts_lookup[fact_id] for fact_id in group.sorted_fact_ids()]
                    similarity_payload = {
                        method: [pair.as_tuple() for pair in pairs]
                        for method, pairs in group.similarity.items()
                    }
                    self._log_canonical_preview(partition, canonical_facts)
                    outcome = persistence.apply_group(
                        canonical_facts,
                        source_facts,
                        similarity_payload,
                    )
                    partition_deleted.extend(outcome.deleted_fact_ids)
                    partition_canonical.extend(outcome.canonical_fact_ids)
                    if outcome.deleted_fact_ids:
                        merge_delta = max(
                            0,
                            len(outcome.deleted_fact_ids) - len(outcome.canonical_fact_ids),
                        )
                        partition_merge_delta += merge_delta

                if partition_failed:
                    failure_detected = True
                    progress.mark_partition_status(run_id, partition, "failed")
                    continue

                if partition_deleted:
                    deleted_fact_ids.update(partition_deleted)
                if partition_canonical:
                    canonical_fact_ids.update(partition_canonical)

                stats.candidate_groups_processed += partition_group_count
                if partition_group_count:
                    group_size_sum += partition_group_size_sum
                stats.facts_merged += partition_merge_delta
                stats.processed_partitions += 1
                progress.mark_partition_status(run_id, partition, "completed")
                progress.update_totals(
                    run_id,
                    processed_partitions=1,
                    facts_processed=len(selection.facts),
                    facts_merged=partition_merge_delta,
                    candidate_groups_processed=len(candidate_groups),
                )

        conn.close()

        stats.elapsed_time = time.time() - start_time
        if stats.candidate_groups_processed:
            stats.average_group_size = group_size_sum / stats.candidate_groups_processed

        if not self.config.dry_run and (deleted_fact_ids or canonical_fact_ids):
            try:
                self._refresh_graph(deleted_fact_ids)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Graph refresh failed: %s", exc)
                failure_detected = True

            materialize_facts(
                self.config.sqlite_path,
                neo4j_uri=self.config.neo4j_uri,
                user=self.config.neo4j_user,
                password=self.config.neo4j_password,
                min_confidence=self.config.min_confidence,
            )
        elif self.config.dry_run:
            logger.info("Dry-run complete; skipping graph synchronization.")

        conn = sqlite3.connect(str(self.config.sqlite_path))
        try:
            run_progress = DeduplicationProgress(conn)
            run_progress.mark_run_completed(
                run_id,
                "failed" if failure_detected else "completed",
            )
        finally:
            conn.close()

        return stats

    def _enforce_group_limits(
        self,
        groups: Sequence[CandidateGroup],
        facts_lookup: Mapping[int, FactRecord],
    ) -> list[CandidateGroup]:
        max_size = self.config.max_group_size
        if max_size <= 0:
            return list(groups)

        result: list[CandidateGroup] = []
        queue: list[CandidateGroup] = list(groups)
        while queue:
            current = queue.pop(0)
            size = len(current.fact_ids)
            if size <= max_size:
                result.append(current)
                continue

            split_groups = self._split_group_by_attribute(current, facts_lookup)
            if split_groups:
                logger.info(
                    "Splitting oversized group (%d facts) into %d sub-groups for partition %s/%s",
                    size,
                    len(split_groups),
                    current.partition.fact_type.value,
                    current.partition.subject_id,
                )
                queue.extend(split_groups)
                continue

            # If splitting did not reduce size, chunk the group to avoid over-large prompts
            chunked = list(self._chunk_group(current, max_size))
            if chunked:
                logger.info(
                    "Chunking oversized group (%d facts) into %d chunks (max size %d) for partition %s/%s",
                    size,
                    len(chunked),
                    max_size,
                    current.partition.fact_type.value,
                    current.partition.subject_id,
                )
                queue.extend(chunked)

        return result

    def _split_group_by_attribute(
        self,
        group: CandidateGroup,
        facts_lookup: Mapping[int, FactRecord],
    ) -> list[CandidateGroup]:
        attr_names = self._attribute_index.get(group.partition.fact_type)
        primary_attr = attr_names[0] if attr_names else None

        buckets: dict[str, list[int]] = defaultdict(list)
        for fact_id in sorted(group.fact_ids):
            fact = facts_lookup[fact_id]
            key = self._normalize_fact_key(fact, primary_attr)
            buckets[key].append(fact_id)

        new_groups: list[CandidateGroup] = []
        for fact_ids in buckets.values():
            if len(fact_ids) < 2:
                continue
            if len(fact_ids) == len(group.fact_ids):
                # No effective split
                continue
            new_groups.append(self._build_candidate_group(group, fact_ids))

        return new_groups

    def _chunk_group(
        self,
        group: CandidateGroup,
        max_size: int,
    ) -> Iterable[CandidateGroup]:
        fact_ids = sorted(group.fact_ids)
        for idx in range(0, len(fact_ids), max_size):
            chunk = fact_ids[idx : idx + max_size]
            if len(chunk) < 2:
                continue
            yield self._build_candidate_group(group, chunk)

    def _build_candidate_group(
        self,
        base_group: CandidateGroup,
        fact_ids: Iterable[int],
    ) -> CandidateGroup:
        fact_set = set(fact_ids)
        new_group = CandidateGroup(partition=base_group.partition, fact_ids=fact_set)
        for method, pairs in base_group.similarity.items():
            filtered = [
                pair
                for pair in pairs
                if pair.source_id in fact_set and pair.target_id in fact_set
            ]
            if filtered:
                new_group.add_similarity(method, filtered)
        return new_group

    @staticmethod
    def _normalize_fact_key(fact: FactRecord, primary_attr: str | None) -> str:
        candidates = []
        if primary_attr:
            candidates.append(fact.attributes.get(primary_attr))
        candidates.append(fact.object_label)
        candidates.append(fact.object_id)

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value.strip().lower()
            if value is not None:
                return str(value).strip().lower()
        return f"fact-{fact.id}"

    def _prepare_run(
        self,
        progress: DeduplicationProgress,
        partitioner: FactPartitioner,
    ) -> tuple[int, int]:
        if self.config.resume:
            active = progress.get_active_run()
            if active is None:
                raise RuntimeError("No active deduplication run to resume")
            progress.resume_run(active.id)
            total = active.total_partitions or partitioner.count_partitions()
            if active.total_partitions == 0 and total:
                self._update_total_partitions(progress, active.id, total)
            return active.id, total

        total_partitions = partitioner.count_partitions()
        run_id = progress.start_new_run(total_partitions)
        return run_id, total_partitions

    def _update_total_partitions(
        self,
        progress: DeduplicationProgress,
        run_id: int,
        total: int,
    ) -> None:
        conn = progress._conn  # pylint: disable=protected-access
        conn.execute(
            "UPDATE deduplication_run SET total_partitions = ? WHERE id = ?",
            (total, run_id),
        )
        conn.commit()

    def _refresh_graph(self, fact_ids: Iterable[int]) -> None:
        fact_list = list(dict.fromkeys(fact_ids))
        if not fact_list:
            return
        logger.info("Removing %d fact relationships from Neo4j", len(fact_list))
        driver = GraphDatabase.driver(
            self.config.neo4j_uri,
            auth=(self.config.neo4j_user, self.config.neo4j_password),
        )
        try:
            with driver.session() as session:
                for chunk in _chunk(fact_list, self.config.graph_delete_batch_size):
                    session.run(
                        "MATCH ()-[r]-() WHERE r.factId IN $ids DELETE r",
                        {"ids": chunk},
                    )
        finally:
            driver.close()

    def _log_group_preview(
        self,
        group: CandidateGroup,
        facts_lookup: Mapping[int, FactRecord],
        *,
        preview_limit: int = 10,
    ) -> None:
        fact_ids = group.sorted_fact_ids()
        facts = [facts_lookup[fact_id] for fact_id in fact_ids]
        if not facts:
            return

        similarity_counts = {
            method: len(pairs)
            for method, pairs in sorted(group.similarity.items())
        }
        logger.info(
            "Candidate group: %d facts for partition %s/%s (similarity pairs: %s)",
            len(facts),
            group.partition.fact_type.value,
            group.partition.subject_id,
            similarity_counts or "{}",
        )
        for fact in facts[:preview_limit]:
            logger.info(
                "  - Fact %d | conf=%.2f | object=%s | attrs={%s} | evidence=%d",
                fact.id,
                fact.confidence,
                self._format_object(fact),
                self._summarize_attributes(fact.attributes),
                len(fact.evidence),
            )
        if len(facts) > preview_limit:
            logger.info("  - ... %d additional facts omitted", len(facts) - preview_limit)

    def _log_canonical_preview(
        self,
        partition: Partition,
        canonical_facts: Sequence[CanonicalFact],
        *,
        preview_limit: int = 3,
    ) -> None:
        if not canonical_facts:
            logger.info(
                "LLM produced no canonical facts for partition %s/%s",
                partition.fact_type.value,
                partition.subject_id,
            )
            return

        logger.info(
            "LLM canonical suggestions (%d) for partition %s/%s:",
            len(canonical_facts),
            partition.fact_type.value,
            partition.subject_id,
        )
        for canonical in canonical_facts[:preview_limit]:
            logger.info(
                "  - Canonical object=%s | conf=%.2f | attrs={%s} | merged_from=%s | reasoning=%s",
                self._format_canonical_object(canonical),
                canonical.confidence,
                self._summarize_attributes(canonical.attributes),
                canonical.merged_from,
                self._truncate(canonical.merge_reasoning),
            )
        if len(canonical_facts) > preview_limit:
            logger.info(
                "  - ... %d additional canonical facts omitted",
                len(canonical_facts) - preview_limit,
            )

    @staticmethod
    def _format_object(fact: FactRecord) -> str:
        label = fact.object_label or fact.object_id or "<unknown>"
        if fact.object_type:
            return f"{label} ({fact.object_type})"
        return label

    @staticmethod
    def _format_canonical_object(canonical: CanonicalFact) -> str:
        label = canonical.object_label or canonical.object_id or "<unknown>"
        if canonical.object_type:
            return f"{label} ({canonical.object_type})"
        return label

    @staticmethod
    def _summarize_attributes(attributes: Mapping[str, object] | None, *, limit: int = 3) -> str:
        if not attributes:
            return ""
        items: list[str] = []
        for idx, (key, value) in enumerate(sorted(attributes.items())):
            if idx >= limit:
                items.append("...")
                break
            items.append(f"{key}={value!r}")
        return ", ".join(items)

    @staticmethod
    def _truncate(text: str, limit: int = 160) -> str:
        if not text:
            return ""
        collapsed = " ".join(text.split())
        if len(collapsed) <= limit:
            return collapsed
        return f"{collapsed[: limit - 3]}..."


def _chunk(values: Sequence[int], size: int) -> Iterable[list[int]]:
    if size <= 0:
        size = 100
    it = iter(values)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch
