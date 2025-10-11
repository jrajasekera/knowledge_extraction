"""Implementation for the find_people_by_skill tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase, ToolContext
from .utils import run_read_query


class FindPeopleBySkillInput(BaseModel):
    """Inputs for the find_people_by_skill tool."""

    skill: str
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    limit: int = Field(default=10, ge=1, le=50)


class SkillPersonModel(BaseModel):
    """Output entry for find_people_by_skill."""

    person_id: str
    name: str | None = None
    proficiency: str | None = None
    years_experience: int | None = None
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class FindPeopleBySkillOutput(BaseModel):
    """Outputs for find_people_by_skill."""

    skill: str
    people: list[SkillPersonModel] = Field(default_factory=list)


class FindPeopleBySkillTool(ToolBase[FindPeopleBySkillInput, FindPeopleBySkillOutput]):
    """Find people who have a specific skill."""

    input_model = FindPeopleBySkillInput
    output_model = FindPeopleBySkillOutput

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)

    def run(self, input_data: FindPeopleBySkillInput) -> FindPeopleBySkillOutput:
        query = """
        MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill)
        WHERE toLower(s.name) = toLower($skill)
          AND coalesce(r.confidence, 0.0) >= $min_confidence
        RETURN p.id AS person_id,
               p.name AS name,
               r.proficiency AS proficiency,
               r.years_experience AS years_experience,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence
        ORDER BY confidence DESC
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "skill": input_data.skill,
            "min_confidence": input_data.min_confidence,
            "limit": input_data.limit,
        }
        rows = run_read_query(self.context, query, params)
        people = [
            SkillPersonModel(
                person_id=row.get("person_id"),
                name=row.get("name"),
                proficiency=row.get("proficiency"),
                years_experience=row.get("years_experience"),
                confidence=row.get("confidence"),
                evidence=row.get("evidence") or [],
            )
            for row in rows
        ]
        return FindPeopleBySkillOutput(skill=input_data.skill, people=people)
