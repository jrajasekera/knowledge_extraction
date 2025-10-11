"""Implementation for get_relationships_between."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


class GetRelationshipsBetweenInput(BaseModel):
    """Inputs for get_relationships_between."""

    person_a_id: str
    person_b_id: str


class RelationshipModel(BaseModel):
    """Direct relationship descriptor."""

    type: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class SharedContextModel(BaseModel):
    """Shared context descriptor."""

    type: str
    context: str
    details: dict[str, Any] = Field(default_factory=dict)


class GetRelationshipsBetweenOutput(BaseModel):
    """Outputs for get_relationships_between."""

    relationships: list[RelationshipModel] = Field(default_factory=list)
    shared_contexts: list[SharedContextModel] = Field(default_factory=list)


class GetRelationshipsBetweenTool(
    ToolBase[GetRelationshipsBetweenInput, GetRelationshipsBetweenOutput],
):
    """Find connections between two people."""

    input_model = GetRelationshipsBetweenInput
    output_model = GetRelationshipsBetweenOutput

    def run(self, input_data: GetRelationshipsBetweenInput) -> GetRelationshipsBetweenOutput:
        direct_query = """
        MATCH (a:Person {id: $person_a_id})-[r]-(b:Person {id: $person_b_id})
        RETURN type(r) AS type,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence,
               r AS rel
        """
        shared_query = """
        MATCH (a:Person {id: $person_a_id})-[r1]->(shared)<-[r2]-(b:Person {id: $person_b_id})
        WHERE NOT shared:Person
        RETURN labels(shared) AS labels,
               shared.name AS name,
               shared.id AS id,
               type(r1) AS rel_a,
               type(r2) AS rel_b
        LIMIT 25
        """
        params = {"person_a_id": input_data.person_a_id, "person_b_id": input_data.person_b_id}

        direct_rows = run_read_query(self.context, direct_query, params)
        relationships = []
        for row in direct_rows:
            rel = row.get("rel", {})
            attributes = {k: v for k, v in dict(rel).items() if k not in {"confidence", "evidence"}}
            relationships.append(
                RelationshipModel(
                    type=row.get("type"),
                    attributes=attributes,
                    confidence=row.get("confidence"),
                    evidence=row.get("evidence") or [],
                )
            )

        shared_rows = run_read_query(self.context, shared_query, params)
        shared_contexts = []
        for row in shared_rows:
            labels = row.get("labels") or []
            shared_type = next((label.lower() for label in labels if label != "Entity"), "context")
            details = {"relation_from_a": row.get("rel_a"), "relation_from_b": row.get("rel_b")}
            context_name = row.get("name") or row.get("id") or shared_type
            shared_contexts.append(
                SharedContextModel(
                    type=shared_type,
                    context=context_name,
                    details=details,
                )
            )

        return GetRelationshipsBetweenOutput(relationships=relationships, shared_contexts=shared_contexts)
