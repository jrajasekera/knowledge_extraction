"""Implementation for find_people_by_topic."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from .base import ToolBase, ToolError
from .person_profile import run_read_query
from .semantic_search import DEFAULT_VECTOR_INDEX
from .utils import run_vector_query


class FindPeopleByTopicInput(BaseModel):
    """Inputs for find_people_by_topic."""

    topic: str
    relationship_types: list[str] = Field(
        default_factory=lambda: ["TALKS_ABOUT", "CARES_ABOUT", "CURIOUS_ABOUT"],
        min_length=1,
    )
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=50)


class TopicPersonModel(BaseModel):
    """Output entry for find_people_by_topic."""

    person_id: str
    name: str | None = None
    relationship_type: str
    sentiment: str | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class FindPeopleByTopicOutput(BaseModel):
    """Outputs for find_people_by_topic."""

    topic: str
    people: list[TopicPersonModel] = Field(default_factory=list)


class FindPeopleByTopicTool(ToolBase[FindPeopleByTopicInput, FindPeopleByTopicOutput]):
    """Find people who discuss or care about a topic."""

    input_model = FindPeopleByTopicInput
    output_model = FindPeopleByTopicOutput

    def __init__(self, context, index_name: str = DEFAULT_VECTOR_INDEX) -> None:
        super().__init__(context)
        self.index_name = index_name

    @property
    def embeddings(self) -> EmbeddingProvider | None:
        model = self.context.embeddings_model
        if model is None:
            return None
        if not isinstance(model, EmbeddingProvider):
            raise TypeError(f"Unexpected embedding model type: {type(model)}")
        return model

    def run(self, input_data: FindPeopleByTopicInput) -> FindPeopleByTopicOutput:
        graph_rows = self._query_graph(input_data)
        if not graph_rows:
            graph_rows = self._semantic_fallback(input_data) or []

        grouped = defaultdict(list)
        for row in graph_rows:
            person_id = row.get("person_id")
            if not person_id:
                continue
            grouped[person_id].append(row)

        people: list[TopicPersonModel] = []
        for person_id, entries in grouped.items():
            best_entry = max(entries, key=lambda item: item.get("confidence", 0.0))
            details = {
                "topic": best_entry.get("topic_name") or input_data.topic,
                "target_labels": best_entry.get("target_labels"),
            }
            for key in ("importance_level", "manifests_as", "related_actions", "curiosity_level", "spark"):
                if best_entry.get(key):
                    details[key] = best_entry.get(key)

            if not best_entry.get("sentiment") and best_entry.get("importance_level"):
                sentiment = best_entry.get("importance_level")
            else:
                sentiment = best_entry.get("sentiment")

            people.append(
                TopicPersonModel(
                    person_id=person_id,
                    name=best_entry.get("name"),
                    relationship_type=best_entry.get("relationship_type", ""),
                    sentiment=sentiment,
                    confidence=best_entry.get("confidence"),
                    evidence=best_entry.get("evidence") or [],
                    details={k: v for k, v in details.items() if v not in (None, [], "")},
                )
            )

        people.sort(key=lambda item: item.confidence or 0.0, reverse=True)
        people = people[: input_data.limit]
        return FindPeopleByTopicOutput(topic=input_data.topic, people=people)

    def _query_graph(self, input_data: FindPeopleByTopicInput) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Person)-[r]->(target)
        WHERE type(r) IN $relationship_types
          AND (
                (target.name IS NOT NULL AND toLower(target.name) CONTAINS toLower($topic))
             OR (toLower(coalesce(r.topic, '')) CONTAINS toLower($topic))
             OR (toLower(coalesce(r.manifestsAs, '')) CONTAINS toLower($topic))
             OR (toLower(coalesce(r.relatedActions, '')) CONTAINS toLower($topic))
            )
          AND coalesce(r.confidence, 0.0) >= $min_confidence
        RETURN p.id AS person_id,
               p.name AS name,
               type(r) AS relationship_type,
               r.sentiment AS sentiment,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence,
               labels(target) AS target_labels,
               coalesce(target.name, r.topic, r.manifestsAs, r.relatedActions, $topic) AS topic_name,
               r.importanceLevel AS importance_level,
               r.manifestsAs AS manifests_as,
               r.relatedActions AS related_actions,
               r.curiosityLevel AS curiosity_level,
               r.spark AS spark
        ORDER BY confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "relationship_types": input_data.relationship_types,
            "topic": input_data.topic,
            "min_confidence": input_data.min_confidence,
            "limit": input_data.limit * 3,
        }
        return run_read_query(self.context, query, params)

    def _semantic_fallback(self, input_data: FindPeopleByTopicInput) -> list[dict[str, Any]]:
        provider = self.embeddings
        if provider is None:
            return []
        embedding = provider.embed_single(input_data.topic)
        if not embedding:
            return []
        filters = {"fact_type": input_data.relationship_types}
        try:
            rows = run_vector_query(
                self.context,
                self.index_name,
                embedding,
                input_data.limit * 3,
                filters,
            )
        except ToolError:
            return []

        transformed: list[dict[str, Any]] = []
        for row in rows:
            node = row.get("node")
            if not node:
                continue
            properties = dict(node)
            person_id = properties.get("person_id")
            if not person_id:
                continue
            attributes = properties.get("attributes")
            if isinstance(attributes, str):
                try:
                    attributes = json.loads(attributes)
                except json.JSONDecodeError:
                    attributes = {}
            if attributes is None:
                attributes = {}
            transformed.append(
                {
                    "person_id": person_id,
                    "name": properties.get("person_name"),
                    "relationship_type": properties.get("fact_type"),
                    "confidence": properties.get("confidence", row.get("score")),
                    "evidence": properties.get("evidence") or [],
                    "sentiment": attributes.get("sentiment"),
                    "importance_level": attributes.get("importanceLevel"),
                    "manifests_as": attributes.get("manifestsAs"),
                    "related_actions": attributes.get("relatedActions"),
                    "curiosity_level": attributes.get("curiosityLevel"),
                    "spark": attributes.get("spark"),
                    "topic_name": properties.get("fact_object") or attributes.get("topic") or input_data.topic,
                    "target_labels": attributes.get("target_labels"),
                }
            )
        return transformed
