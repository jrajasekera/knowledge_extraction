"""Implementation for get_person_timeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from .base import ToolBase
from .utils import run_read_query


DEFAULT_TEMPORAL_FACTS = [
    "WORKS_AT",
    "WORKING_ON",
    "PREVIOUSLY",
    "STUDIED_AT",
    "ATTENDED_EVENT",
    "EXPERIENCED",
]


class GetPersonTimelineInput(BaseModel):
    """Inputs for get_person_timeline."""

    person_id: str
    fact_types: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None


class TimelineEntryModel(BaseModel):
    """Output entry for get_person_timeline."""

    type: str
    object: str | None = None
    start: str | None = None
    end: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    confidence: float | None = None
    evidence: list[str] = Field(default_factory=list)


class GetPersonTimelineOutput(BaseModel):
    """Outputs for get_person_timeline."""

    person_id: str
    name: str | None = None
    timeline: list[TimelineEntryModel] = Field(default_factory=list)


class GetPersonTimelineTool(ToolBase[GetPersonTimelineInput, GetPersonTimelineOutput]):
    """Retrieve temporal facts about a person."""

    input_model = GetPersonTimelineInput
    output_model = GetPersonTimelineOutput

    def run(self, input_data: GetPersonTimelineInput) -> GetPersonTimelineOutput:
        fact_types = input_data.fact_types or DEFAULT_TEMPORAL_FACTS
        query = """
        MATCH (p:Person {id: $person_id})-[r]->(target)
        WHERE type(r) IN $fact_types
        WITH p, r, target
        WHERE ($start_date IS NULL OR coalesce(r.start_date, r.timestamp, r.occurred_at) >= $start_date)
          AND ($end_date IS NULL OR coalesce(r.end_date, r.timestamp, r.occurred_at) <= $end_date)
        RETURN p.name AS name,
               type(r) AS type,
               coalesce(target.name, target.id) AS object,
               coalesce(r.start_date, r.timestamp, r.occurred_at) AS start,
               coalesce(r.end_date, r.occurred_at) AS end,
               coalesce(r.confidence, 0.0) AS confidence,
               coalesce(r.evidence, []) AS evidence,
               r AS rel
        ORDER BY start ASC
        """
        params: dict[str, Any] = {
            "person_id": input_data.person_id,
            "fact_types": fact_types,
            "start_date": input_data.start_date,
            "end_date": input_data.end_date,
        }
        rows = run_read_query(self.context, query, params)
        timeline = []
        person_name: str | None = None
        for row in rows:
            person_name = row.get("name") or person_name
            rel = row.get("rel", {})
            attributes = {k: v for k, v in dict(rel).items() if k not in {"confidence", "evidence", "start_date", "end_date"}}
            timeline.append(
                TimelineEntryModel(
                    type=row.get("type"),
                    object=row.get("object"),
                    start=row.get("start"),
                    end=row.get("end"),
                    attributes=attributes,
                    confidence=row.get("confidence"),
                    evidence=row.get("evidence") or [],
                )
            )
        return GetPersonTimelineOutput(person_id=input_data.person_id, name=person_name, timeline=timeline)
