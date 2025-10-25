"""Implementation for semantic_search_facts."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from ..models import RetrievedFact
from .base import ToolBase, ToolContext, ToolError
from .utils import run_vector_query

# Configure logger with handler to ensure logs are visible
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)

DEFAULT_VECTOR_INDEX = "fact_embeddings"


class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    queries: list[str] = Field(min_length=1, max_length=5)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class SemanticSearchResult(BaseModel):
    """Output entry for semantic_search_facts."""

    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


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

    def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
        logger.info(
            "semantic_search_facts called: queries=%r, limit=%d, similarity_threshold=%.2f, index=%s",
            input_data.queries,
            input_data.limit,
            input_data.similarity_threshold,
            self.index_name,
        )

        # Dictionary to store unique results keyed by (person_id, fact_type, fact_object)
        results_dict: dict[tuple[str, str, str | None], SemanticSearchResult] = {}
        total_raw_results = 0
        total_filtered_by_threshold = 0
        total_missing_node = 0
        queries_processed = 0

        # Process each query
        for query_idx, query in enumerate(input_data.queries, 1):
            logger.info("Processing query %d/%d: %r", query_idx, len(input_data.queries), query)

            # Generate embedding for this query
            embedding = self.embeddings.embed_single(query)
            if not embedding:
                logger.warning("Failed to generate embedding for query %d: %r", query_idx, query)
                continue

            logger.debug("Generated embedding vector of length %d for query %d", len(embedding), query_idx)

            # Execute vector query
            try:
                rows = run_vector_query(
                    self.context,
                    self.index_name,
                    embedding,
                    input_data.limit,
                    None,
                )
                logger.info("Query %d returned %d raw results from index %s", query_idx, len(rows), self.index_name)
                total_raw_results += len(rows)
                queries_processed += 1
            except ToolError as exc:
                logger.warning("Query %d failed: %s", query_idx, exc)
                continue

            # Process results from this query
            filtered_by_threshold = 0
            missing_node = 0
            added_new = 0
            updated_existing = 0

            for row in rows:
                node = row.get("node")
                score = row.get("score", 0.0)

                # Apply similarity threshold
                if input_data.similarity_threshold and score < input_data.similarity_threshold:
                    filtered_by_threshold += 1
                    logger.debug(
                        "Query %d: Filtered result with score %.3f (below threshold %.2f)",
                        query_idx,
                        score,
                        input_data.similarity_threshold,
                    )
                    continue

                if not node:
                    missing_node += 1
                    logger.debug("Query %d: Skipping row with missing node", query_idx)
                    continue

                # Parse node properties
                properties = dict(node)
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

                # Create deduplication key
                person_id = properties.get("person_id", "")
                fact_type = properties.get("fact_type", "")
                fact_object = properties.get("fact_object")
                dedup_key = (person_id, fact_type, fact_object)

                # Check if we should add/update this result
                if dedup_key not in results_dict or score > results_dict[dedup_key].similarity_score:
                    result = SemanticSearchResult(
                        person_id=person_id,
                        person_name=properties.get("person_name", person_id),
                        fact_type=fact_type,
                        fact_object=fact_object,
                        attributes=attributes,
                        similarity_score=score,
                        confidence=properties.get("confidence"),
                        evidence=properties.get("evidence") or [],
                    )

                    if dedup_key in results_dict:
                        updated_existing += 1
                        logger.debug(
                            "Query %d: Updated existing result with higher score %.3f (was %.3f): person=%s, fact_type=%s, object=%s",
                            query_idx,
                            score,
                            results_dict[dedup_key].similarity_score,
                            properties.get("person_name", person_id),
                            fact_type,
                            fact_object,
                        )
                    else:
                        added_new += 1
                        logger.debug(
                            "Query %d: Added new result: person=%s, fact_type=%s, object=%s, score=%.3f",
                            query_idx,
                            properties.get("person_name", person_id),
                            fact_type,
                            fact_object,
                            score,
                        )

                    results_dict[dedup_key] = result
                else:
                    logger.debug(
                        "Query %d: Skipped duplicate with lower score %.3f (existing %.3f): person=%s, fact_type=%s",
                        query_idx,
                        score,
                        results_dict[dedup_key].similarity_score,
                        properties.get("person_name", person_id),
                        fact_type,
                    )

            total_filtered_by_threshold += filtered_by_threshold
            total_missing_node += missing_node

            logger.info(
                "Query %d summary: raw=%d, filtered=%d, missing_node=%d, added_new=%d, updated_existing=%d",
                query_idx,
                len(rows),
                filtered_by_threshold,
                missing_node,
                added_new,
                updated_existing,
            )

        # Convert dictionary to sorted list
        unique_results = list(results_dict.values())
        logger.info("Total unique facts before limit: %d", len(unique_results))

        # Sort by similarity score descending
        unique_results.sort(key=lambda r: r.similarity_score, reverse=True)

        # Apply final limit
        final_results = unique_results[: input_data.limit]

        logger.info(
            "semantic_search_facts completed: queries=%d, queries_processed=%d, total_raw_results=%d, "
            "total_filtered=%d, total_missing_node=%d, unique_facts=%d, final_results=%d",
            len(input_data.queries),
            queries_processed,
            total_raw_results,
            total_filtered_by_threshold,
            total_missing_node,
            len(unique_results),
            len(final_results),
        )

        return SemanticSearchOutput(queries=input_data.queries, results=final_results)
