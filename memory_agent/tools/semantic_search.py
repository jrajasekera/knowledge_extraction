"""Implementation for semantic_search_facts."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from ..models import RetrievedFact
from .base import ToolBase, ToolContext, ToolError
from .utils import run_vector_query

DEFAULT_VECTOR_INDEX = "fact_embeddings"


class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    query: str
    fact_types: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


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

    query: str
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
        embedding = self.embeddings.embed_single(input_data.query)
        if not embedding:
            return SemanticSearchOutput(query=input_data.query, results=[])
        filters: dict[str, Any] | None = None
        if input_data.fact_types:
            filters = {"fact_type": input_data.fact_types}

        rows = run_vector_query(
            self.context,
            self.index_name,
            embedding,
            input_data.limit,
            filters,
        )
        results = []
        for row in rows:
            node = row.get("node")
            score = row.get("score", 0.0)
            if input_data.similarity_threshold and score < input_data.similarity_threshold:
                continue
            if not node:
                continue
            properties = dict(node)
            result = SemanticSearchResult(
                person_id=properties.get("person_id", ""),
                person_name=properties.get("person_name", properties.get("person_id", "")),
                fact_type=properties.get("fact_type", ""),
                fact_object=properties.get("fact_object"),
                attributes=properties.get("attributes", {}),
                similarity_score=score,
                confidence=properties.get("confidence"),
                evidence=properties.get("evidence") or [],
            )
            results.append(result)
        return SemanticSearchOutput(query=input_data.query, results=results)
