"""Implementation for find_people_by_topic."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


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


class FindPeopleByTopicOutput(BaseModel):
    """Outputs for find_people_by_topic."""

    topic: str
    people: list[TopicPersonModel] = Field(default_factory=list)


class FindPeopleByTopicTool(ToolBase[FindPeopleByTopicInput, FindPeopleByTopicOutput]):
    """Find people who discuss or care about a topic."""

    input_model = FindPeopleByTopicInput
    output_model = FindPeopleByTopicOutput

    def run(self, input_data: FindPeopleByTopicInput) -> FindPeopleByTopicOutput:
        query = """
        MATCH (p:Person)-[r]->(t:Topic)
        WHERE type(r) IN $relationship_types
          AND toLower(t.name) CONTAINS toLower($topic)
          AND coalesce(r.confidence, 0.0) >= $min_confidence
        RETURN p.id AS person_id,
               p.name AS name,
               type(r) AS relationship_type,
               r.sentiment AS sentiment,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence
        ORDER BY confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "relationship_types": input_data.relationship_types,
            "topic": input_data.topic,
            "min_confidence": input_data.min_confidence,
            "limit": input_data.limit,
        }
        rows = run_read_query(self.context, query, params)
        people = [
            TopicPersonModel(
                person_id=row.get("person_id"),
                name=row.get("name"),
                relationship_type=row.get("relationship_type"),
                sentiment=row.get("sentiment"),
                confidence=row.get("confidence"),
                evidence=row.get("evidence") or [],
            )
            for row in rows
        ]
        return FindPeopleByTopicOutput(topic=input_data.topic, people=people)
