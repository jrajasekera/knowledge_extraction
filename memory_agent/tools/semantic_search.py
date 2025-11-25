"""Implementation for semantic_search_facts."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from ..models import RetrievedFact
from .base import ToolBase, ToolContext, ToolError
from .utils import run_keyword_query, run_vector_query

# Configure logger with handler to ensure logs are visible
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

DEFAULT_VECTOR_INDEX = "fact_embeddings"

# RRF constants
RRF_K = 60
DEFAULT_FUSION_METHOD: Literal["rrf", "score_sum", "score_max"] = "rrf"
DEFAULT_MULTI_QUERY_BOOST = 0.0

# Hybrid search: offset for keyword query indices to treat them as separate "voters" in RRF
KEYWORD_QUERY_OFFSET = 1000

# Adaptive threshold configuration
ADAPTIVE_THRESHOLD_MAX = 0.75
ADAPTIVE_THRESHOLD_MIN = 0.35
ADAPTIVE_THRESHOLD_STEP = 0.05
ADAPTIVE_TARGET_RATIO = 0.8
ADAPTIVE_SEARCH_RESULT_MULTIPLIER = 2.0


@dataclass
class FactOccurrence:
    """Track per-fact observations across multiple semantic queries."""

    properties: dict[str, Any]
    best_score: float
    evidence: list[str | dict[str, Any]] = field(default_factory=list)
    query_scores: dict[int, float] = field(default_factory=dict)
    query_ranks: dict[int, int] = field(default_factory=dict)

    def add_observation(
        self,
        query_idx: int,
        score: float,
        rank: int,
        properties: dict[str, Any],
        evidence: list[str | dict[str, Any]],
    ) -> None:
        """Record an observation for this fact from a specific query.

        Args:
            query_idx: Index of the query that returned this fact (1-based)
            score: Similarity score for this observation
            rank: Rank position in the result list (1-based)
            properties: Node properties from Neo4j
            evidence: Evidence list (message content or IDs)
        """
        # Update query_scores: keep the highest score if we see this fact multiple times in same query
        existing_score = self.query_scores.get(query_idx)
        if existing_score is None or score > existing_score:
            self.query_scores[query_idx] = score

        # Update query_ranks: keep the best (lowest) rank for this query
        if query_idx not in self.query_ranks or rank < self.query_ranks[query_idx]:
            self.query_ranks[query_idx] = rank

        # Update best_score, properties, and evidence if this is the best observation overall
        if score > self.best_score:
            self.best_score = score
            self.properties = properties
            self.evidence = evidence


class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    queries: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    adaptive_threshold: bool = Field(default=True)
    adaptive_threshold_max: float = Field(default=ADAPTIVE_THRESHOLD_MAX, ge=0.0, le=1.0)
    adaptive_threshold_min: float = Field(default=ADAPTIVE_THRESHOLD_MIN, ge=0.0, le=1.0)
    fusion_method: Literal["rrf", "score_sum", "score_max"] = Field(default=DEFAULT_FUSION_METHOD)
    multi_query_boost: float = Field(default=DEFAULT_MULTI_QUERY_BOOST, ge=0.0, le=1.0)


class SemanticSearchResult(BaseModel):
    """Output entry for semantic_search_facts."""

    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    confidence: float | None = None
    evidence: list[str | dict[str, Any]] = Field(default_factory=list)
    query_scores: dict[int, float] | None = None
    appeared_in_query_count: int | None = None


class SemanticSearchOutput(BaseModel):
    """Outputs for semantic_search_facts."""

    queries: list[str]
    results: list[SemanticSearchResult] = Field(default_factory=list)


class SemanticSearchFactsTool(ToolBase[SemanticSearchInput, SemanticSearchOutput]):
    """Find facts semantically similar to a query."""

    input_model = SemanticSearchInput
    output_model = SemanticSearchOutput

    def __init__(self, context: ToolContext, index_name: str = DEFAULT_VECTOR_INDEX) -> None:
        super().__init__(context)
        self.index_name = index_name

    @property
    def embeddings(self) -> EmbeddingProvider:
        model = self.context.embeddings_model
        if model is None:
            raise ToolError("Embedding model not configured")
        if not isinstance(model, EmbeddingProvider):
            msg = f"Unexpected embedding model type: {type(model)}"
            raise ToolError(msg)
        return model

    @staticmethod
    def _calculate_combined_score(
        occurrence: FactOccurrence,
        fusion_method: str,
        multi_query_boost: float,
    ) -> float:
        """Calculate combined score from multiple query observations.

        Args:
            occurrence: FactOccurrence with scores and ranks from multiple queries
            fusion_method: Method to use for combining scores (rrf, score_sum, score_max)
            multi_query_boost: Boost factor for facts appearing in multiple queries (score_max only)

        Returns:
            Combined score for ranking
        """
        if not occurrence.query_scores:
            return occurrence.best_score

        if fusion_method == "rrf":
            # Reciprocal Rank Fusion: sum of 1/(K+rank) for each query
            return sum(1.0 / (RRF_K + rank) for rank in occurrence.query_ranks.values())

        if fusion_method == "score_sum":
            # Simple sum of all scores
            return sum(occurrence.query_scores.values())

        if fusion_method == "score_max":
            # Max score with boost for appearing in multiple queries
            max_score = max(occurrence.query_scores.values())
            query_count = len(occurrence.query_scores)
            return max_score * (1.0 + multi_query_boost * (query_count - 1))

        # Fallback to best_score if method is unrecognized
        logger.warning("Unrecognized fusion method: %s, using best_score", fusion_method)
        return occurrence.best_score

    def _build_result(
        self,
        properties: dict[str, Any],
        score: float,
        evidence: list[str | dict[str, Any]],
        *,
        query_scores: dict[int, float] | None = None,
    ) -> SemanticSearchResult:
        """Build a SemanticSearchResult from node properties and calculated score.

        Args:
            properties: Node properties from Neo4j
            score: Combined similarity score
            evidence: Evidence list (may include content snippets)
            query_scores: Optional mapping of query index to score

        Returns:
            SemanticSearchResult with all fields populated
        """
        # Parse attributes (may be JSON string or dict)
        attributes_raw = properties.get("attributes")
        attributes = {}
        if isinstance(attributes_raw, str):
            try:
                import json
                attributes = json.loads(attributes_raw)
            except json.JSONDecodeError:
                attributes = {}
        elif isinstance(attributes_raw, dict):
            attributes = attributes_raw

        appeared_in_query_count = len(query_scores) if query_scores else None

        return SemanticSearchResult(
            person_id=properties.get("person_id", ""),
            person_name=properties.get("person_name", properties.get("person_id", "")),
            fact_type=properties.get("fact_type", ""),
            fact_object=properties.get("fact_object"),
            attributes=attributes,
            similarity_score=score,
            confidence=properties.get("confidence"),
            evidence=evidence,
            query_scores=query_scores,
            appeared_in_query_count=appeared_in_query_count,
        )

    @staticmethod
    def _extract_relationship_type(properties: dict[str, Any]) -> str | None:
        """Extract relationship_type from attributes for deduplication."""

        attributes_raw = properties.get("attributes")

        if isinstance(attributes_raw, str):
            try:
                import json

                attributes = json.loads(attributes_raw)
                return (
                    str(attributes.get("relationship_type"))
                    if attributes.get("relationship_type")
                    else None
                )
            except json.JSONDecodeError:
                return None

        if isinstance(attributes_raw, dict):
            return (
                str(attributes_raw.get("relationship_type"))
                if attributes_raw.get("relationship_type")
                else None
            )

        return None

    def _execute_search_pass(
        self,
        input_data: SemanticSearchInput,
        similarity_threshold: float,
        *,
        search_limit: int | None = None,
    ) -> dict[tuple[str, str, str | None, str | None], FactOccurrence]:
        """Execute a single search pass at the given threshold."""

        occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}
        per_query_limit = int(search_limit or input_data.limit)

        for query_idx, query in enumerate(input_data.queries, 1):
            logger.info(
                "Processing query %d/%d at threshold %.2f: %r",
                query_idx,
                len(input_data.queries),
                similarity_threshold,
                query,
            )

            vector_rows: list[dict[str, Any]] = []
            embedding = self.embeddings.embed_single(query)
            if embedding:
                try:
                    vector_rows = run_vector_query(
                        self.context,
                        self.index_name,
                        embedding,
                        per_query_limit,
                        None,
                    )
                    logger.info(
                        "Query %d vector search returned %d results",
                        query_idx,
                        len(vector_rows),
                    )
                except ToolError as exc:
                    logger.warning("Query %d vector search failed: %s", query_idx, exc)
            else:
                logger.warning("Failed to generate embedding for query %d: %r", query_idx, query)

            keyword_rows: list[dict[str, Any]] = []
            try:
                keyword_rows = run_keyword_query(
                    self.context,
                    query,
                    per_query_limit,
                )
                logger.info(
                    "Query %d keyword search returned %d results",
                    query_idx,
                    len(keyword_rows),
                )
            except ToolError as exc:
                logger.warning("Query %d keyword search failed: %s", query_idx, exc)

            search_batches = [
                (vector_rows, query_idx, "vector"),
                (keyword_rows, query_idx + KEYWORD_QUERY_OFFSET, "keyword"),
            ]

            for rows, effective_query_idx, source_type in search_batches:
                for rank, row in enumerate(rows, start=1):
                    score = row.get("score", 0.0)
                    if score < similarity_threshold:
                        logger.debug(
                            "Query %d (%s): Filtered result with score %.3f (below threshold %.2f)",
                            query_idx,
                            source_type,
                            score,
                            similarity_threshold,
                        )
                        continue

                    node = row.get("node")
                    if not node:
                        logger.debug("Query %d (%s): Skipping row with missing node", query_idx, source_type)
                        continue

                    properties = dict(node)
                    person_id = properties.get("person_id", "")
                    fact_type = properties.get("fact_type", "")
                    fact_object = properties.get("fact_object")
                    relationship_type = self._extract_relationship_type(properties)

                    dedup_key = (person_id, fact_type, fact_object, relationship_type)

                    evidence_with_content = row.get("evidence_with_content", [])
                    evidence = evidence_with_content if evidence_with_content else properties.get("evidence") or []

                    occurrence = occurrences.get(dedup_key)
                    if occurrence is None:
                        occurrence = FactOccurrence(
                            properties=properties,
                            best_score=score,
                            evidence=evidence,
                        )
                        occurrences[dedup_key] = occurrence

                    occurrence.add_observation(
                        effective_query_idx, score, rank, properties, evidence
                    )

        return occurrences

    def _search_with_adaptive_threshold(
        self, input_data: SemanticSearchInput
    ) -> tuple[dict[tuple[str, str, str | None, str | None], FactOccurrence], float]:
        """Execute search with adaptive threshold adjustment."""

        target_count = int(input_data.limit * ADAPTIVE_TARGET_RATIO)
        current_threshold = input_data.adaptive_threshold_max
        best_occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}
        best_threshold = current_threshold

        max_iterations = int(
            (input_data.adaptive_threshold_max - input_data.adaptive_threshold_min)
            / ADAPTIVE_THRESHOLD_STEP
        ) + 1

        search_limit = int(
            max(input_data.limit * ADAPTIVE_SEARCH_RESULT_MULTIPLIER, input_data.limit)
        )

        iterations = 0
        while current_threshold >= input_data.adaptive_threshold_min and iterations < max_iterations:
            iterations += 1

            logger.info(
                "Adaptive search iteration %d: threshold=%.2f, target=%d",
                iterations,
                current_threshold,
                target_count,
            )

            occurrences = self._execute_search_pass(
                input_data=input_data,
                similarity_threshold=current_threshold,
                search_limit=search_limit,
            )

            unique_count = len(occurrences)
            logger.info(
                "Adaptive search iteration %d: found %d unique facts",
                iterations,
                unique_count,
            )

            if unique_count > len(best_occurrences):
                best_occurrences = occurrences
                best_threshold = current_threshold

            if unique_count >= target_count:
                logger.info(
                    "Adaptive threshold converged: threshold=%.2f, results=%d, target=%d",
                    current_threshold,
                    unique_count,
                    target_count,
                )
                return occurrences, current_threshold

            current_threshold -= ADAPTIVE_THRESHOLD_STEP

        logger.info(
            "Adaptive threshold exhausted: final_threshold=%.2f, results=%d, target=%d",
            best_threshold,
            len(best_occurrences),
            target_count,
        )

        return best_occurrences, best_threshold

    def _build_results_from_occurrences(
        self,
        occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence],
        fusion_method: str,
        multi_query_boost: float,
    ) -> list[SemanticSearchResult]:
        results: list[SemanticSearchResult] = []

        for occurrence in occurrences.values():
            combined_score = self._calculate_combined_score(
                occurrence,
                fusion_method,
                multi_query_boost,
            )

            result = self._build_result(
                occurrence.properties,
                combined_score,
                occurrence.evidence,
                query_scores=dict(occurrence.query_scores),
            )
            results.append(result)

        return results

    def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
        logger.info(
            "semantic_search_facts called: queries=%r, limit=%d, adaptive=%s, index=%s",
            input_data.queries,
            input_data.limit,
            input_data.adaptive_threshold,
            self.index_name,
        )

        adaptive_mode = input_data.adaptive_threshold and input_data.similarity_threshold is None

        if adaptive_mode:
            occurrences, final_threshold = self._search_with_adaptive_threshold(input_data)
        else:
            threshold = input_data.similarity_threshold if input_data.similarity_threshold is not None else 0.6
            occurrences = self._execute_search_pass(
                input_data,
                threshold,
                search_limit=input_data.limit,
            )
            final_threshold = threshold

        logger.info(
            "Semantic search finished: mode=%s, final_threshold=%.2f, unique_facts=%d",
            "adaptive" if adaptive_mode else "fixed",
            final_threshold,
            len(occurrences),
        )

        results_with_scores = self._build_results_from_occurrences(
            occurrences,
            input_data.fusion_method,
            input_data.multi_query_boost,
        )

        ordered = sorted(results_with_scores, key=lambda r: r.similarity_score, reverse=True)
        final_results = ordered[: input_data.limit]

        return SemanticSearchOutput(queries=input_data.queries, results=final_results)
