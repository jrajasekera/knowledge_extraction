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


def _set_relationship_properties(tx, query: str, params: dict[str, Any]) -> None:
    tx.run(query, params)


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
        "evidence": list(dict.fromkeys(fact.evidence)),
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
        "evidence": list(dict.fromkeys(fact.evidence)),
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
        "evidence": list(dict.fromkeys(fact.evidence)),
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
        return

    _ensure_person_node(tx, other_id, fact.object_official_name)

    params = {
        "subject_id": fact.subject_id,
        "other_id": other_id,
        "fact_id": fact.id,
        "confidence": fact.confidence,
        "timestamp": fact.timestamp,
        "basis": _sanitize_value(fact.attributes.get("closeness_basis")),
        "evidence": list(dict.fromkeys(fact.evidence)),
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


HANDLERS = {
    FactType.WORKS_AT: _handle_works_at,
    FactType.LIVES_IN: _handle_lives_in,
    FactType.TALKS_ABOUT: _handle_talks_about,
    FactType.CLOSE_TO: _handle_close_to,
}


def materialize_facts(
    sqlite_path: str | Path,
    *,
    neo4j_uri: str,
    user: str,
    password: str,
    fact_types: Sequence[FactType] | None = None,
    min_confidence: float = 0.5,
) -> None:
    fact_types = tuple(fact_types or HANDLERS.keys())

    conn = sqlite3.connect(str(sqlite_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        facts = list(
            _fetch_facts(conn, fact_types=fact_types, min_confidence=min_confidence)
        )
    finally:
        conn.close()

    if not facts:
        print("[facts_to_graph] No facts found meeting the criteria.")
        return

    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    try:
        with driver.session() as session:
            for fact in facts:
                handler = HANDLERS.get(fact.type)
                if not handler:
                    continue
                session.execute_write(handler, fact)
    finally:
        driver.close()

    print(
        f"[facts_to_graph] Materialized {len(facts)} facts (min_confidence={min_confidence})."
    )


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
