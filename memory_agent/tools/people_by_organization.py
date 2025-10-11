"""Implementation for the find_people_by_organization tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


class FindPeopleByOrganizationInput(BaseModel):
    """Inputs for find_people_by_organization."""

    organization: str
    current_only: bool = False
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=50)


class OrganizationPersonModel(BaseModel):
    """Output entry for find_people_by_organization."""

    person_id: str
    name: str | None = None
    role: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class FindPeopleByOrganizationOutput(BaseModel):
    """Outputs for find_people_by_organization."""

    organization: str
    people: list[OrganizationPersonModel] = Field(default_factory=list)


class FindPeopleByOrganizationTool(ToolBase[FindPeopleByOrganizationInput, FindPeopleByOrganizationOutput]):
    """Find people who work or worked at an organization."""

    input_model = FindPeopleByOrganizationInput
    output_model = FindPeopleByOrganizationOutput

    def run(self, input_data: FindPeopleByOrganizationInput) -> FindPeopleByOrganizationOutput:
        relationship_types = ["WORKS_AT"]
        if not input_data.current_only:
            relationship_types.append("PREVIOUSLY")

        query = """
        MATCH (p:Person)-[r]->(o:Organization)
        WHERE toLower(o.name) CONTAINS toLower($organization)
          AND type(r) IN $relationship_types
          AND coalesce(r.confidence, 0.0) >= $min_confidence
        RETURN p.id AS person_id,
               p.name AS name,
               r.role AS role,
               r.start_date AS start_date,
               r.end_date AS end_date,
               coalesce(r.location, o.headquarters) AS location,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence
        ORDER BY confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "organization": input_data.organization,
            "relationship_types": relationship_types,
            "min_confidence": input_data.min_confidence,
            "limit": input_data.limit,
        }
        rows = run_read_query(self.context, query, params)
        people = [
            OrganizationPersonModel(
                person_id=row.get("person_id"),
                name=row.get("name"),
                role=row.get("role"),
                start_date=row.get("start_date"),
                end_date=row.get("end_date"),
                location=row.get("location"),
                confidence=row.get("confidence"),
                evidence=row.get("evidence") or [],
            )
            for row in rows
        ]
        return FindPeopleByOrganizationOutput(organization=input_data.organization, people=people)
