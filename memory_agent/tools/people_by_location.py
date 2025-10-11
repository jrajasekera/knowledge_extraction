"""Implementation for find_people_by_location."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


class FindPeopleByLocationInput(BaseModel):
    """Inputs for find_people_by_location."""

    location: str
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=50)


class LocationPersonModel(BaseModel):
    """Output entry for find_people_by_location."""

    person_id: str
    name: str | None = None
    relationship: str
    details: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class FindPeopleByLocationOutput(BaseModel):
    """Outputs for find_people_by_location."""

    location: str
    people: list[LocationPersonModel] = Field(default_factory=list)


class FindPeopleByLocationTool(ToolBase[FindPeopleByLocationInput, FindPeopleByLocationOutput]):
    """Find people connected to a particular location."""

    input_model = FindPeopleByLocationInput
    output_model = FindPeopleByLocationOutput

    def run(self, input_data: FindPeopleByLocationInput) -> FindPeopleByLocationOutput:
        query = """
        MATCH (p:Person)-[r]->(loc:Location)
        WHERE toLower(loc.name) CONTAINS toLower($location)
          AND type(r) IN ['LIVES_IN', 'WORKS_IN', 'STUDIED_IN', 'VISITED']
          AND coalesce(r.confidence, 0.0) >= $min_confidence
        RETURN p.id AS person_id,
               p.name AS name,
               type(r) AS relationship,
               coalesce(r.details, {}) AS details,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence
        ORDER BY confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "location": input_data.location,
            "min_confidence": input_data.min_confidence,
            "limit": input_data.limit,
        }
        rows = run_read_query(self.context, query, params)
        people = [
            LocationPersonModel(
                person_id=row.get("person_id"),
                name=row.get("name"),
                relationship=row.get("relationship"),
                details=row.get("details") or {},
                confidence=row.get("confidence"),
                evidence=row.get("evidence") or [],
            )
            for row in rows
        ]
        return FindPeopleByLocationOutput(location=input_data.location, people=people)
