#!/usr/bin/env python3
"""Materialize high-confidence facts from SQLite into Neo4j."""

from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from neo4j import GraphDatabase

from ie.types import FactType


@dataclass(slots=True)
class FactRecord:
    id: int
    type: FactType
    subject_id: str
    subject_official_name: str | None
    object_id: str | None
    object_official_name: str | None
    object_type: str | None
    attributes: dict[str, Any]
    timestamp: str
    confidence: float
    evidence: list[str]


@dataclass(slots=True)
class MaterializeSummary:
    candidates: int
    processed: int

    def as_dict(self) -> dict[str, int]:
        return {
            "candidates": self.candidates,
            "processed": self.processed,
        }


def _ensure_person_node(tx, member_id: str, official_name: str | None) -> None:
    if not member_id:
        return
    params = {"id": member_id}
    cleaned_name = _sanitize_value(official_name)
    if cleaned_name:
        params["official_name"] = cleaned_name
        tx.run(
            "MERGE (p:Person {id:$id}) SET p.realName=$official_name",
            params,
        )
    else:
        tx.run("MERGE (p:Person {id:$id})", params)


def _fetch_facts(
    conn: sqlite3.Connection,
    *,
    fact_types: Sequence[FactType],
    min_confidence: float,
) -> Iterable[FactRecord]:
    type_values = [fact_type.value for fact_type in fact_types]
    placeholders = ",".join("?" for _ in type_values)
    sql = f"""
        SELECT
          f.id,
          f.type,
          f.subject_id,
          subj.official_name,
          f.object_id,
          obj.official_name,
          f.object_type,
          f.attributes,
          f.ts,
          f.confidence,
          COALESCE(json_group_array(fe.message_id), '[]') AS evidence
        FROM fact AS f
        LEFT JOIN member AS subj ON subj.id = f.subject_id
        LEFT JOIN member AS obj ON obj.id = f.object_id
        LEFT JOIN fact_evidence AS fe ON fe.fact_id = f.id
        WHERE f.confidence >= ?
          AND f.type IN ({placeholders})
          AND f.graph_synced_at IS NULL
        GROUP BY f.id
    """
    params: list[Any] = [min_confidence]
    params.extend(type_values)

    for row in conn.execute(sql, params):
        raw_attributes = row[7]
        if isinstance(raw_attributes, bytes):
            raw_attributes = raw_attributes.decode()
        attributes = json.loads(raw_attributes)

        evidence_raw = row[10]
        evidence = json.loads(evidence_raw) if isinstance(evidence_raw, str) else []
        yield FactRecord(
            id=int(row[0]),
            type=FactType(row[1]),
            subject_id=row[2],
            subject_official_name=row[3],
            object_id=row[4],
            object_official_name=row[5],
            object_type=row[6],
            attributes=attributes,
            timestamp=row[8] or "",
            confidence=float(row[9]),
            evidence=evidence,
        )


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
    if value == "":
        return None
    return value


def _clean_attr(fact: FactRecord, key: str, *, fallback: Any | None = None) -> Any:
    value = fact.attributes.get(key)
    if value is None:
        value = fallback
    return _sanitize_value(value)


def _dedupe_evidence(values: Iterable[str]) -> list[str]:
    return [value for value in dict.fromkeys(v for v in values if v)]


def _ensure_external_person_node(tx, name: str) -> None:
    if not name:
        return
    tx.run("MERGE (p:ExternalPerson {name:$name})", {"name": name})


def _set_relationship_properties(tx, query: str, params: dict[str, Any]) -> None:
    tx.run(query, params)


def _mark_fact_synced(conn: sqlite3.Connection, fact_id: int) -> None:
    conn.execute(
        "UPDATE fact SET graph_synced_at = CURRENT_TIMESTAMP WHERE id = ?",
        (fact_id,),
    )


def _handle_works_at(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    organization = _sanitize_value(fact.attributes.get("organization"))
    if not organization:
        return

    params = {
        "subject_id": fact.subject_id,
        "organization": organization,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "role": _sanitize_value(fact.attributes.get("role")),
        "location": _sanitize_value(fact.attributes.get("location")),
        "startDate": _sanitize_value(fact.attributes.get("start_date")),
        "endDate": _sanitize_value(fact.attributes.get("end_date")),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (o:Org {name:$organization})
    MERGE (p)-[r:WORKS_AT {factId:$fact_id}]->(o)
    SET r.role = $role,
        r.location = $location,
        r.startDate = $startDate,
        r.endDate = $endDate,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_lives_in(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    location = _sanitize_value(fact.attributes.get("location"))
    if not location:
        return

    params = {
        "subject_id": fact.subject_id,
        "location": location,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "since": _sanitize_value(fact.attributes.get("since")),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (place:Place {label:$location})
    MERGE (p)-[r:LIVES_IN {factId:$fact_id}]->(place)
    SET r.since = $since,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_talks_about(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    topic = _sanitize_value(fact.attributes.get("topic"))
    if not topic:
        return

    params = {
        "subject_id": fact.subject_id,
        "topic": topic,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "sentiment": _sanitize_value(fact.attributes.get("sentiment")),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (t:Topic {name:$topic})
    MERGE (p)-[r:TALKS_ABOUT {factId:$fact_id}]->(t)
    SET r.sentiment = $sentiment,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_close_to(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    other_id = _sanitize_value(fact.object_id)
    if not other_id:
        other_label = _clean_attr(fact, "object_label")
        if not other_label:
            return

        params = {
            "subject_id": fact.subject_id,
            "other_label": other_label,
            "fact_id": fact.id,
            "confidence": fact.confidence,
            "timestamp": fact.timestamp,
            "basis": _clean_attr(fact, "closeness_basis"),
            "evidence": _dedupe_evidence(fact.evidence),
        }

        _ensure_external_person_node(tx, other_label)

        query = """
        MATCH (a:Person {id:$subject_id})
        MATCH (b:ExternalPerson {name:$other_label})
        MERGE (a)-[r:CLOSE_TO {factId:$fact_id}]->(b)
        SET r.basis = $basis,
            r.confidence = $confidence,
            r.lastUpdated = datetime($timestamp),
            r.evidence = $evidence
        MERGE (b)-[rb:CLOSE_TO {factId:$fact_id}]->(a)
        SET rb.basis = $basis,
            rb.confidence = $confidence,
            rb.lastUpdated = datetime($timestamp),
            rb.evidence = $evidence
        """
        _set_relationship_properties(tx, query, params)
        return

    _ensure_person_node(tx, other_id, fact.object_official_name)

    params = {
        "subject_id": fact.subject_id,
        "other_id": other_id,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "basis": _sanitize_value(fact.attributes.get("closeness_basis")),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (a:Person {id:$subject_id})
    MATCH (b:Person {id:$other_id})
    MERGE (a)-[r:CLOSE_TO {factId:$fact_id}]->(b)
    SET r.basis = $basis,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    MERGE (b)-[rb:CLOSE_TO {factId:$fact_id}]->(a)
    SET rb.basis = $basis,
        rb.confidence = $confidence,
        rb.lastUpdated = datetime($timestamp),
        rb.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_studied_at(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    institution = _clean_attr(fact, "institution", fallback=_clean_attr(fact, "object_label"))
    if not institution:
        return

    params = {
        "subject_id": fact.subject_id,
        "institution": institution,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "degree_type": _clean_attr(fact, "degree_type"),
        "field_of_study": _clean_attr(fact, "field_of_study"),
        "graduation_year": _clean_attr(fact, "graduation_year"),
        "status": _clean_attr(fact, "status"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (i:Institution {name:$institution})
    MERGE (p)-[r:STUDIED_AT {factId:$fact_id}]->(i)
    SET r.degreeType = $degree_type,
        r.fieldOfStudy = $field_of_study,
        r.graduationYear = $graduation_year,
        r.status = $status,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_has_skill(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    skill = _clean_attr(fact, "skill", fallback=_clean_attr(fact, "object_label"))
    if not skill:
        return

    params = {
        "subject_id": fact.subject_id,
        "skill": skill,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "proficiency_level": _clean_attr(fact, "proficiency_level"),
        "years_experience": _clean_attr(fact, "years_experience"),
        "learning_status": _clean_attr(fact, "learning_status"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (s:Skill {name:$skill})
    MERGE (p)-[r:HAS_SKILL {factId:$fact_id}]->(s)
    SET r.proficiency = $proficiency_level,
        r.yearsExperience = $years_experience,
        r.learningStatus = $learning_status,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_working_on(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    project = _clean_attr(fact, "project", fallback=_clean_attr(fact, "object_label"))
    if not project:
        return

    params = {
        "subject_id": fact.subject_id,
        "project": project,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "role": _clean_attr(fact, "role"),
        "start_date": _clean_attr(fact, "start_date"),
        "project_type": _clean_attr(fact, "project_type"),
        "collaboration_mode": _clean_attr(fact, "collaboration_mode"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (proj:Project {name:$project})
    MERGE (p)-[r:WORKING_ON {factId:$fact_id}]->(proj)
    SET r.role = $role,
        r.startDate = $start_date,
        r.projectType = $project_type,
        r.collaborationMode = $collaboration_mode,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_related_to(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    relationship_type = _clean_attr(fact, "relationship_type")
    relationship_basis = _clean_attr(fact, "relationship_basis")
    params_common = {
        "subject_id": fact.subject_id,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "relationship_type": relationship_type,
        "relationship_basis": relationship_basis,
        "evidence": _dedupe_evidence(fact.evidence),
    }

    other_id = _sanitize_value(fact.object_id)
    if other_id:
        _ensure_person_node(tx, other_id, fact.object_official_name)
        params = {
            **params_common,
            "other_id": other_id,
        }
        query = """
        MATCH (a:Person {id:$subject_id})
        MATCH (b:Person {id:$other_id})
        MERGE (a)-[r:RELATED_TO {factId:$fact_id}]->(b)
        SET r.relationshipType = $relationship_type,
            r.relationshipBasis = $relationship_basis,
            r.confidence = $confidence,
            r.lastUpdated = datetime($timestamp),
            r.evidence = $evidence
        MERGE (b)-[rb:RELATED_TO {factId:$fact_id}]->(a)
        SET rb.relationshipType = $relationship_type,
            rb.relationshipBasis = $relationship_basis,
            rb.confidence = $confidence,
            rb.lastUpdated = datetime($timestamp),
            rb.evidence = $evidence
        """
        _set_relationship_properties(tx, query, params)
        return

    other_label = _clean_attr(fact, "object_label")
    if not other_label:
        return

    _ensure_external_person_node(tx, other_label)

    params = {
        **params_common,
        "other_label": other_label,
    }

    query = """
    MATCH (a:Person {id:$subject_id})
    MATCH (b:ExternalPerson {name:$other_label})
    MERGE (a)-[r:RELATED_TO {factId:$fact_id}]->(b)
    SET r.relationshipType = $relationship_type,
        r.relationshipBasis = $relationship_basis,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    MERGE (b)-[rb:RELATED_TO {factId:$fact_id}]->(a)
    SET rb.relationshipType = $relationship_type,
        rb.relationshipBasis = $relationship_basis,
        rb.confidence = $confidence,
        rb.lastUpdated = datetime($timestamp),
        rb.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_attended_event(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    event_name = _clean_attr(fact, "event_name", fallback=_clean_attr(fact, "object_label"))
    if not event_name:
        return

    params = {
        "subject_id": fact.subject_id,
        "event_name": event_name,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "event_type": _clean_attr(fact, "event_type"),
        "date": _clean_attr(fact, "date"),
        "role": _clean_attr(fact, "role"),
        "format": _clean_attr(fact, "format"),
        "location": _clean_attr(fact, "location"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (e:Event {name:$event_name})
    ON CREATE SET e.eventType = $event_type,
                  e.format = $format,
                  e.location = $location
    MERGE (p)-[r:ATTENDED_EVENT {factId:$fact_id}]->(e)
    SET r.eventType = $event_type,
        r.date = $date,
        r.role = $role,
        r.format = $format,
        r.location = $location,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_recommends(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    target = _clean_attr(fact, "target", fallback=_clean_attr(fact, "object_label"))
    if not target:
        return

    params = {
        "subject_id": fact.subject_id,
        "target": target,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "strength": _clean_attr(fact, "recommendation_strength"),
        "context": _clean_attr(fact, "context"),
        "reason": _clean_attr(fact, "reason"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (item:Resource {name:$target})
    MERGE (p)-[r:RECOMMENDS {factId:$fact_id}]->(item)
    SET r.strength = $strength,
        r.context = $context,
        r.reason = $reason,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_avoids(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    target = _clean_attr(fact, "target", fallback=_clean_attr(fact, "object_label"))
    if not target:
        return

    params = {
        "subject_id": fact.subject_id,
        "target": target,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "reason": _clean_attr(fact, "reason"),
        "severity": _clean_attr(fact, "severity"),
        "timeframe": _clean_attr(fact, "timeframe"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (item:Resource {name:$target})
    MERGE (p)-[r:AVOIDS {factId:$fact_id}]->(item)
    SET r.reason = $reason,
        r.severity = $severity,
        r.timeframe = $timeframe,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_plans_to(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    plan = _clean_attr(fact, "plan", fallback=_clean_attr(fact, "object_label"))
    if not plan:
        return

    params = {
        "subject_id": fact.subject_id,
        "plan": plan,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "goal_type": _clean_attr(fact, "goal_type"),
        "timeframe": _clean_attr(fact, "timeframe"),
        "confidence_level": _clean_attr(fact, "confidence_level"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (g:Goal {description:$plan})
    MERGE (p)-[r:PLANS_TO {factId:$fact_id}]->(g)
    SET r.goalType = $goal_type,
        r.timeframe = $timeframe,
        r.planConfidence = $confidence_level,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_previously(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    fact_type = _clean_attr(fact, "fact_type")
    entity_label = _clean_attr(fact, "object_label")
    if not fact_type or not entity_label:
        return

    params = {
        "subject_id": fact.subject_id,
        "fact_type": fact_type,
        "entity_label": entity_label,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "start_date": _clean_attr(fact, "start_date"),
        "end_date": _clean_attr(fact, "end_date"),
        "transition_reason": _clean_attr(fact, "transition_reason"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (h:HistoricalEntity {name:$entity_label, factType:$fact_type})
    MERGE (p)-[r:PREVIOUSLY {factId:$fact_id}]->(h)
    SET r.factType = $fact_type,
        r.startDate = $start_date,
        r.endDate = $end_date,
        r.transitionReason = $transition_reason,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_prefers(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    target = _clean_attr(fact, "preference_target", fallback=_clean_attr(fact, "object_label"))
    if not target:
        return

    params = {
        "subject_id": fact.subject_id,
        "target": target,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "category": _clean_attr(fact, "preference_category"),
        "strength": _clean_attr(fact, "preference_strength"),
        "alternatives": _clean_attr(fact, "alternatives_considered"),
        "reason": _clean_attr(fact, "reason"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (pref:Preference {name:$target})
    MERGE (p)-[r:PREFERS {factId:$fact_id}]->(pref)
    SET r.category = $category,
        r.strength = $strength,
        r.alternatives = $alternatives,
        r.reason = $reason,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_believes(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    topic = _clean_attr(fact, "object_label", fallback=_clean_attr(fact, "topic"))
    if not topic:
        return

    params = {
        "subject_id": fact.subject_id,
        "topic": topic,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "stance": _clean_attr(fact, "stance"),
        "conviction": _clean_attr(fact, "conviction_strength"),
        "reasoning": _clean_attr(fact, "reasoning"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (t:Topic {name:$topic})
    MERGE (p)-[r:BELIEVES {factId:$fact_id}]->(t)
    SET r.stance = $stance,
        r.conviction = $conviction,
        r.reasoning = $reasoning,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_dislikes(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    target = _clean_attr(fact, "target", fallback=_clean_attr(fact, "object_label"))
    if not target:
        return

    params = {
        "subject_id": fact.subject_id,
        "target": target,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "intensity": _clean_attr(fact, "dislike_intensity"),
        "reason": _clean_attr(fact, "reason"),
        "still_engages": _clean_attr(fact, "still_engages"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (item:Preference {name:$target})
    MERGE (p)-[r:DISLIKES {factId:$fact_id}]->(item)
    SET r.intensity = $intensity,
        r.reason = $reason,
        r.stillEngages = $still_engages,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_enjoys(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    activity = _clean_attr(fact, "activity", fallback=_clean_attr(fact, "object_label"))
    if not activity:
        return

    params = {
        "subject_id": fact.subject_id,
        "activity": activity,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "enjoyment_level": _clean_attr(fact, "enjoyment_level"),
        "frequency": _clean_attr(fact, "frequency"),
        "social_mode": _clean_attr(fact, "social_mode"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (act:Activity {name:$activity})
    MERGE (p)-[r:ENJOYS {factId:$fact_id}]->(act)
    SET r.enjoymentLevel = $enjoyment_level,
        r.frequency = $frequency,
        r.socialMode = $social_mode,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_experienced(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    event_label = _clean_attr(fact, "object_label", fallback=_clean_attr(fact, "event_type"))
    if not event_label:
        return

    params = {
        "subject_id": fact.subject_id,
        "event_label": event_label,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "event_type": _clean_attr(fact, "event_type"),
        "event_date": _clean_attr(fact, "event_date"),
        "impact_level": _clean_attr(fact, "impact_level"),
        "current_stage": _clean_attr(fact, "current_stage"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (le:LifeEvent {name:$event_label})
    MERGE (p)-[r:EXPERIENCED {factId:$fact_id}]->(le)
    SET r.eventType = $event_type,
        r.eventDate = $event_date,
        r.impactLevel = $impact_level,
        r.currentStage = $current_stage,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_cares_about(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    cause = _clean_attr(fact, "object_label")
    if not cause:
        return

    params = {
        "subject_id": fact.subject_id,
        "cause": cause,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "importance_level": _clean_attr(fact, "importance_level"),
        "manifestation": _clean_attr(fact, "how_it_manifests"),
        "actions": _clean_attr(fact, "related_actions"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (c:Cause {name:$cause})
    MERGE (p)-[r:CARES_ABOUT {factId:$fact_id}]->(c)
    SET r.importanceLevel = $importance_level,
        r.manifestsAs = $manifestation,
        r.relatedActions = $actions,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_remembers(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    memory_label = _clean_attr(fact, "object_label", fallback=_clean_attr(fact, "memory_type"))
    if not memory_label:
        return

    params = {
        "subject_id": fact.subject_id,
        "memory_label": memory_label,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "memory_type": _clean_attr(fact, "memory_type"),
        "emotional_valence": _clean_attr(fact, "emotional_valence"),
        "approximate_date": _clean_attr(fact, "approximate_date"),
        "significance": _clean_attr(fact, "significance"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (m:Memory {name:$memory_label})
    MERGE (p)-[r:REMEMBERS {factId:$fact_id}]->(m)
    SET r.memoryType = $memory_type,
        r.emotionalValence = $emotional_valence,
        r.approximateDate = $approximate_date,
        r.significance = $significance,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_curious_about(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    topic = _clean_attr(fact, "topic", fallback=_clean_attr(fact, "object_label"))
    if not topic:
        return

    params = {
        "subject_id": fact.subject_id,
        "topic": topic,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "curiosity_level": _clean_attr(fact, "curiosity_level"),
        "spark": _clean_attr(fact, "what_sparked_it"),
        "exploration_status": _clean_attr(fact, "exploration_status"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (t:Topic {name:$topic})
    MERGE (p)-[r:CURIOUS_ABOUT {factId:$fact_id}]->(t)
    SET r.curiosityLevel = $curiosity_level,
        r.spark = $spark,
        r.explorationStatus = $exploration_status,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


def _handle_witnessed(tx, fact: FactRecord) -> None:
    _ensure_person_node(tx, fact.subject_id, fact.subject_official_name)

    occurrence = _clean_attr(fact, "object_label", fallback=_clean_attr(fact, "context"))
    if not occurrence:
        return

    params = {
        "subject_id": fact.subject_id,
        "occurrence": occurrence,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "context": _clean_attr(fact, "context"),
        "date": _clean_attr(fact, "date"),
        "role": _clean_attr(fact, "their_role"),
        "impact": _clean_attr(fact, "impact"),
        "evidence": _dedupe_evidence(fact.evidence),
    }

    query = """
    MATCH (p:Person {id:$subject_id})
    MERGE (occ:Occurrence {name:$occurrence})
    MERGE (p)-[r:WITNESSED {factId:$fact_id}]->(occ)
    SET r.context = $context,
        r.date = $date,
        r.role = $role,
        r.impact = $impact,
        r.confidence = $confidence,
        r.lastUpdated = datetime($timestamp),
        r.evidence = $evidence
    """
    _set_relationship_properties(tx, query, params)


HANDLERS = {
    FactType.WORKS_AT: _handle_works_at,
    FactType.LIVES_IN: _handle_lives_in,
    FactType.TALKS_ABOUT: _handle_talks_about,
    FactType.CLOSE_TO: _handle_close_to,
    FactType.STUDIED_AT: _handle_studied_at,
    FactType.HAS_SKILL: _handle_has_skill,
    FactType.WORKING_ON: _handle_working_on,
    FactType.RELATED_TO: _handle_related_to,
    FactType.ATTENDED_EVENT: _handle_attended_event,
    FactType.RECOMMENDS: _handle_recommends,
    FactType.AVOIDS: _handle_avoids,
    FactType.PLANS_TO: _handle_plans_to,
    FactType.PREVIOUSLY: _handle_previously,
    FactType.PREFERS: _handle_prefers,
    FactType.BELIEVES: _handle_believes,
    FactType.DISLIKES: _handle_dislikes,
    FactType.ENJOYS: _handle_enjoys,
    FactType.EXPERIENCED: _handle_experienced,
    FactType.CARES_ABOUT: _handle_cares_about,
    FactType.REMEMBERS: _handle_remembers,
    FactType.CURIOUS_ABOUT: _handle_curious_about,
    FactType.WITNESSED: _handle_witnessed,
}


def materialize_facts(
    sqlite_path: str | Path,
    *,
    neo4j_uri: str,
    user: str,
    password: str,
    fact_types: Sequence[FactType] | None = None,
    min_confidence: float = 0.5,
) -> MaterializeSummary:
    selected_types = tuple(fact_types or HANDLERS.keys())

    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))
    processed = 0
    missing_handlers: set[FactType] = set()

    try:
        with driver.session() as session:
            for fact in _fetch_facts(
                conn,
                fact_types=selected_types,
                min_confidence=min_confidence,
            ):
                handler = HANDLERS.get(fact.type)
                if handler:
                    session.execute_write(handler, fact)
                else:
                    missing_handlers.add(fact.type)

                _mark_fact_synced(conn, fact.id)
                processed += 1
                if processed % 100 == 0:
                    conn.commit()

        conn.commit()
    finally:
        driver.close()
        conn.close()

    if processed == 0:
        print("[facts_to_graph] No facts ready for materialization.")
    else:
        print(
            f"[facts_to_graph] Materialized {processed} facts (min_confidence={min_confidence})."
        )

    if missing_handlers:
        missing_names = ", ".join(sorted(ft.value for ft in missing_handlers))
        print(f"[facts_to_graph] Skipped unsupported fact types: {missing_names}")

    return MaterializeSummary(candidates=processed, processed=processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize SQLite facts into Neo4j.")
    parser.add_argument("--sqlite", type=Path, default=Path("./discord.db"), help="Path to SQLite DB.")
    parser.add_argument("--neo4j", default="bolt://localhost:7687", help="Neo4j bolt URI.")
    parser.add_argument("--user", default="neo4j", help="Neo4j username.")
    parser.add_argument("--password", required=True, help="Neo4j password.")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum confidence filter.")
    parser.add_argument(
        "--fact-types",
        nargs="*",
        help="Optional subset of fact types to materialize (e.g., WORKS_AT LIVES_IN)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fact_types = (
        tuple(FactType(value) for value in args.fact_types)
        if args.fact_types
        else None
    )
    materialize_facts(
        args.sqlite,
        neo4j_uri=args.neo4j,
        user=args.user,
        password=args.password,
        fact_types=fact_types,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
