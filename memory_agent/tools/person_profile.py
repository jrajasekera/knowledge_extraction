"""Implementation for the get_person_profile tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase, ToolContext
from .utils import run_read_query


class PersonProfileInput(BaseModel):
    """Inputs for get_person_profile."""

    person_id: str
    fact_types: list[str] | None = Field(default=None, min_length=1)


class PersonFactModel(BaseModel):
    """Representation of a fact tied to a person."""

    type: str
    object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)
    timestamp: str | None = None


class PersonProfileOutput(BaseModel):
    """Outputs for get_person_profile."""

    person_id: str
    name: str | None = None
    facts: list[PersonFactModel] = Field(default_factory=list)


class GetPersonProfileTool(ToolBase[PersonProfileInput, PersonProfileOutput]):
    """Retrieve all facts about a specific person."""

    input_model = PersonProfileInput
    output_model = PersonProfileOutput

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)

    def run(self, input_data: PersonProfileInput) -> PersonProfileOutput:
        filters = ""
        params: dict[str, Any] = {"person_id": input_data.person_id}
        if input_data.fact_types:
            filters = "AND type(r) IN $fact_types"
            params["fact_types"] = input_data.fact_types

        query = f"""
        MATCH (p:Person {{id: $person_id}})-[r]->(target)
        WHERE r.confidence IS NOT NULL {filters}
        RETURN p.name AS name,
               type(r) AS relationship_type,
               target.name AS target_name,
               coalesce(target.id, target.name) AS target_id,
               r AS relationship,
               target AS target
        """
        rows = run_read_query(self.context, query, params)
        facts: list[PersonFactModel] = []
        person_name: str | None = None
        for row in rows:
            person_name = row.get("name") or person_name
            relationship_obj = row.get("relationship", {})
            relationship = dict(relationship_obj) if hasattr(relationship_obj, "items") else relationship_obj or {}
            target = row.get("target", {})
            attributes = {k: v for k, v in dict(relationship).items() if k not in {"confidence", "evidence", "timestamp"}}
            fact = PersonFactModel(
                type=row.get("relationship_type", ""),
                object=row.get("target_name") or row.get("target_id"),
                attributes=attributes,
                confidence=relationship.get("confidence"),
                evidence=relationship.get("evidence", []),
                timestamp=(relationship.get("timestamp") or relationship.get("updated_at")),
            )
            facts.append(fact)

        return PersonProfileOutput(person_id=input_data.person_id, name=person_name, facts=facts)
