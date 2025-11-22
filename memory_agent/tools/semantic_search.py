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
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
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

    def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
        """Execute semantic search with RRF-based multi-query fusion.

        Process:
        1. Generate embeddings for each query
        2. Execute vector searches
        3. Track observations in FactOccurrence objects
        4. Apply similarity threshold filtering
        5. Calculate combined scores using selected fusion method
        6. Sort and limit results

        Args:
            input_data: Search parameters including queries and fusion method

        Returns:
            SemanticSearchOutput with fused results
        """
        logger.info(
            "semantic_search_facts called: queries=%r, limit=%d, similarity_threshold=%.2f, "
            "fusion_method=%s, index=%s",
            input_data.queries,
            input_data.limit,
            input_data.similarity_threshold,
            input_data.fusion_method,
            self.index_name,
        )

        # Track fact occurrences across queries
        occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}

        # Metrics for logging
        total_raw_results = 0
        total_filtered_by_threshold = 0
        total_missing_node = 0
        queries_processed = 0

        # Process each query
        for query_idx, query in enumerate(input_data.queries, 1):
            logger.info("Processing query %d/%d: %r", query_idx, len(input_data.queries), query)

            # --- HYBRID SEARCH: Execute both vector and keyword searches ---

            # 1. Vector Search
            vector_rows = []
            embedding = self.embeddings.embed_single(query)
            if embedding:
                logger.debug("Generated embedding vector of length %d for query %d", len(embedding), query_idx)
                try:
                    vector_rows = run_vector_query(
                        self.context,
                        self.index_name,
                        embedding,
                        input_data.limit,
                        None,
                    )
                    logger.info("Query %d vector search returned %d results", query_idx, len(vector_rows))
                except ToolError as exc:
                    logger.warning("Query %d vector search failed: %s", query_idx, exc)
            else:
                logger.warning("Failed to generate embedding for query %d: %r", query_idx, query)

            # 2. Keyword Search
            keyword_rows = []
            try:
                keyword_rows = run_keyword_query(
                    self.context,
                    query,
                    input_data.limit,
                )
                logger.info("Query %d keyword search returned %d results", query_idx, len(keyword_rows))
            except ToolError as exc:
                logger.warning("Query %d keyword search failed: %s", query_idx, exc)

            # Skip this query if both searches failed
            if not vector_rows and not keyword_rows:
                logger.warning("Query %d: Both vector and keyword search failed or returned no results", query_idx)
                continue

            queries_processed += 1
            total_raw_results += len(vector_rows) + len(keyword_rows)

            # --- Process results from both sources ---
            # Vector results use query_idx, keyword results use query_idx + KEYWORD_QUERY_OFFSET
            # This treats them as separate "voters" in the RRF system

            search_batches = [
                (vector_rows, query_idx, "vector"),
                (keyword_rows, query_idx + KEYWORD_QUERY_OFFSET, "keyword"),
            ]

            filtered_by_threshold = 0
            missing_node = 0
            added_new = 0
            updated_existing = 0

            for rows, effective_query_idx, source_type in search_batches:
                for rank, row in enumerate(rows, start=1):
                    node = row.get("node")
                    score = row.get("score", 0.0)
                    evidence_with_content = row.get("evidence_with_content", [])

                    # Apply similarity threshold
                    if input_data.similarity_threshold and score < input_data.similarity_threshold:
                        filtered_by_threshold += 1
                        logger.debug(
                            "Query %d (%s): Filtered result with score %.3f (below threshold %.2f)",
                            query_idx,
                            source_type,
                            score,
                            input_data.similarity_threshold,
                        )
                        continue

                    if not node:
                        missing_node += 1
                        logger.debug("Query %d (%s): Skipping row with missing node", query_idx, source_type)
                        continue

                    # Parse node properties
                    properties = dict(node)

                    # Create deduplication key
                    person_id = properties.get("person_id", "")
                    fact_type = properties.get("fact_type", "")
                    fact_object = properties.get("fact_object")

                    # Extract relationship_type from attributes for proper deduplication
                    attributes_raw = properties.get("attributes")
                    relationship_type = None
                    if isinstance(attributes_raw, str):
                        try:
                            import json
                            attributes = json.loads(attributes_raw)
                            relationship_type = str(attributes.get("relationship_type")) if attributes.get("relationship_type") else None
                        except json.JSONDecodeError:
                            pass
                    elif isinstance(attributes_raw, dict):
                        relationship_type = str(attributes_raw.get("relationship_type")) if attributes_raw.get("relationship_type") else None

                    dedup_key = (person_id, fact_type, fact_object, relationship_type)

                    # Use evidence_with_content if available, fallback to evidence IDs
                    evidence = evidence_with_content if evidence_with_content else properties.get("evidence") or []

                    # Track or update occurrence
                    occurrence = occurrences.get(dedup_key)
                    if occurrence is None:
                        occurrence = FactOccurrence(properties=properties, best_score=score, evidence=evidence)
                        occurrence.add_observation(effective_query_idx, score, rank, properties, evidence)
                        occurrences[dedup_key] = occurrence
                        added_new += 1
                        logger.debug(
                            "Query %d (%s): Added new fact: person=%s, fact_type=%s, object=%s, score=%.3f, rank=%d",
                            query_idx,
                            source_type,
                            properties.get("person_name", person_id),
                            fact_type,
                            fact_object,
                            score,
                            rank,
                        )
                    else:
                        occurrence.add_observation(effective_query_idx, score, rank, properties, evidence)
                        updated_existing += 1
                        logger.debug(
                            "Query %d (%s): Updated existing fact: person=%s, fact_type=%s, object=%s, score=%.3f, rank=%d",
                            query_idx,
                            source_type,
                        properties.get("person_name", person_id),
                        fact_type,
                        fact_object,
                        score,
                        rank,
                        )

            total_filtered_by_threshold += filtered_by_threshold
            total_missing_node += missing_node

            logger.info(
                "Query %d summary: vector_results=%d, keyword_results=%d, filtered=%d, missing_node=%d, added_new=%d, updated_existing=%d",
                query_idx,
                len(vector_rows),
                len(keyword_rows),
                filtered_by_threshold,
                missing_node,
                added_new,
                updated_existing,
            )

        logger.info("Total unique facts before fusion: %d", len(occurrences))

        # Calculate combined scores and build results
        results_with_scores: list[SemanticSearchResult] = []
        for dedup_key, occurrence in occurrences.items():
            combined_score = self._calculate_combined_score(
                occurrence,
                input_data.fusion_method,
                input_data.multi_query_boost,
            )

            # Use evidence from the best observation (includes message content, not just IDs)
            evidence = occurrence.evidence

            result = self._build_result(
                occurrence.properties,
                combined_score,
                evidence,
                query_scores=dict(occurrence.query_scores),
            )
            results_with_scores.append(result)

        # Log fusion statistics
        facts_in_multiple_queries = sum(1 for occ in occurrences.values() if len(occ.query_scores) > 1)
        avg_queries_per_fact = (
            sum(len(occ.query_scores) for occ in occurrences.values()) / len(occurrences)
            if occurrences
            else 0.0
        )

        logger.info(
            "Fact fusion summary: total_unique_facts=%d, facts_in_multiple_queries=%d, "
            "avg_queries_per_fact=%.2f, fusion_method=%s",
            len(occurrences),
            facts_in_multiple_queries,
            avg_queries_per_fact,
            input_data.fusion_method,
        )

        # Sort by combined score descending
        ordered = sorted(results_with_scores, key=lambda r: r.similarity_score, reverse=True)

        # Apply final limit
        final_results = ordered[: input_data.limit]

        # Log top results for debugging
        for result in final_results[:5]:
            occurrence_key = (result.person_id, result.fact_type, result.fact_object,
                             result.attributes.get("relationship_type") if result.attributes else None)
            occurrence = occurrences.get(occurrence_key)
            if occurrence:
                logger.debug(
                    "Top result after fusion: person=%s, fact_type=%s, combined_score=%.3f, "
                    "appeared_in=%d, query_scores=%s",
                    result.person_name,
                    result.fact_type,
                    result.similarity_score,
                    len(occurrence.query_scores),
                    occurrence.query_scores,
                )

        logger.info(
            "semantic_search_facts completed: queries=%d, queries_processed=%d, total_raw_results=%d, "
            "total_filtered=%d, total_missing_node=%d, unique_facts=%d, final_results=%d",
            len(input_data.queries),
            queries_processed,
            total_raw_results,
            total_filtered_by_threshold,
            total_missing_node,
            len(ordered),
            len(final_results),
        )

        return SemanticSearchOutput(queries=input_data.queries, results=final_results)
