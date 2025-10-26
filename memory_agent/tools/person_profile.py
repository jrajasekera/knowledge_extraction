"""Tool implementation for get_person_profile."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


PROFILE_QUERY = """
MATCH (person:Person {id: $person_id})-[rel]->(target)
RETURN person.id AS person_id,
       coalesce(person.realName, person.name, person.id) AS person_name,
       type(rel) AS relationship_type,
       rel AS relationship,
       coalesce(target.name, target.title, target.id) AS target_name,
       target.id AS target_id,
       labels(target) AS target_labels,
       target AS target
ORDER BY rel.lastUpdated DESC, rel.confidence DESC
LIMIT 200
"""


class PersonProfileInput(BaseModel):
    """Inputs for get_person_profile."""

    person_id: str


class PersonProfileFact(BaseModel):
    """Structured representation of a person's relationship."""

    person_id: str
    person_name: str | None = None
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    evidence: list[str | dict[str, Any]] = Field(default_factory=list)
    timestamp: Any | None = None


class PersonProfileOutput(BaseModel):
    """Output payload for get_person_profile."""

    facts: list[PersonProfileFact] = Field(default_factory=list)


class PersonProfileTool(ToolBase[PersonProfileInput, PersonProfileOutput]):
    """Fetch the most recent relationships for a specific person."""

    input_model = PersonProfileInput
    output_model = PersonProfileOutput

    def run(self, input_data: PersonProfileInput) -> PersonProfileOutput:
        params = {"person_id": input_data.person_id}
        rows = run_read_query(self.context, PROFILE_QUERY, params)
        facts: list[PersonProfileFact] = []
        for row in rows:
            relationship = dict(row.get("relationship") or {})
            target = row.get("target") or {}
            attributes = {
                "target_id": row.get("target_id"),
                "target_labels": row.get("target_labels") or [],
                "target_properties": target,
            }
            extra_attrs = {
                key: value
                for key, value in relationship.items()
                if key not in {"confidence", "evidence", "timestamp"}
                and value not in (None, "", [], {})
            }
            attributes.update(extra_attrs)
            facts.append(
                PersonProfileFact(
                    person_id=row.get("person_id") or input_data.person_id,
                    person_name=row.get("person_name"),
                    fact_type=row.get("relationship_type") or "RELATED_TO",
                    fact_object=row.get("target_name") or row.get("target_id"),
                    attributes={k: v for k, v in attributes.items() if v not in (None, "", [], {})},
                    confidence=relationship.get("confidence"),
                    evidence=relationship.get("evidence") or [],
                    timestamp=relationship.get("timestamp"),
                )
            )
        return PersonProfileOutput(facts=facts)


__all__ = ["PersonProfileTool", "run_read_query"]
