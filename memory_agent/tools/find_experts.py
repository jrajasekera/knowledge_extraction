"""Implementation for find_experts."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from .base import ToolBase, ToolError
from .semantic_search import DEFAULT_VECTOR_INDEX
from .utils import run_vector_query


class FindExpertsInput(BaseModel):
    """Inputs for find_experts."""

    query: str
    limit: int = Field(default=5, ge=1, le=20)


class ExpertFactModel(BaseModel):
    """Relevant fact supporting the expert ranking."""

    type: str
    description: str
    confidence: float | None = None


class ExpertResultModel(BaseModel):
    """Expert result entry."""

    person_id: str
    name: str | None = None
    relevance_score: float
    relevant_facts: list[ExpertFactModel] = Field(default_factory=list)


class FindExpertsOutput(BaseModel):
    """Outputs for find_experts."""

    query: str
    experts: list[ExpertResultModel] = Field(default_factory=list)


class FindExpertsTool(ToolBase[FindExpertsInput, FindExpertsOutput]):
    """Find people best suited to answer a question."""

    input_model = FindExpertsInput
    output_model = FindExpertsOutput

    def __init__(self, context, index_name: str = DEFAULT_VECTOR_INDEX) -> None:
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

    def run(self, input_data: FindExpertsInput) -> FindExpertsOutput:
        embedding = self.embeddings.embed_single(input_data.query)
        if not embedding:
            return FindExpertsOutput(query=input_data.query, experts=[])
        rows = run_vector_query(self.context, self.index_name, embedding, input_data.limit * 4)
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"name": None, "score": 0.0, "facts": []}
        )
        for row in rows:
            node = row.get("node")
            score = row.get("score", 0.0)
            if not node:
                continue
            properties = dict(node)
            person_id = properties.get("person_id")
            if not person_id:
                continue
            grouped_entry = grouped[person_id]
            if grouped_entry["name"] is None:
                grouped_entry["name"] = properties.get("person_name")
            grouped_entry["score"] = max(grouped_entry["score"], float(score))
            description_parts = [
                properties.get("fact_type", ""),
                properties.get("fact_object") or "",
            ]
            raw_attributes = properties.get("attributes")
            attributes: dict[str, Any] = {}
            if isinstance(raw_attributes, dict):
                attributes = raw_attributes
            elif isinstance(raw_attributes, str):
                try:
                    parsed = json.loads(raw_attributes)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    attributes = parsed
                elif raw_attributes.strip():
                    description_parts.append(raw_attributes.strip())
            elif raw_attributes is not None:
                # Neo4j lists arrive as list[Any]; convert to keyless description text.
                description_parts.append(str(raw_attributes))
            if attributes:
                description_parts.append(", ".join(f"{k}={v}" for k, v in attributes.items()))
            description = " ".join(part for part in description_parts if part).strip()
            grouped_entry["facts"].append(
                ExpertFactModel(
                    type=properties.get("fact_type", ""),
                    description=description or properties.get("text", ""),
                    confidence=properties.get("confidence"),
                )
            )
        sorted_experts = sorted(grouped.items(), key=lambda item: item[1]["score"], reverse=True)[: input_data.limit]
        experts = [
            ExpertResultModel(
                person_id=person_id,
                name=data["name"],
                relevance_score=float(data["score"]),
                relevant_facts=data["facts"],
            )
            for person_id, data in sorted_experts
        ]
        return FindExpertsOutput(query=input_data.query, experts=experts)
