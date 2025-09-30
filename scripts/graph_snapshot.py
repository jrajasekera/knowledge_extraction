#!/usr/bin/env python3
"""Inspect Neo4j knowledge graph contents to validate materialized facts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from neo4j import GraphDatabase


DEFAULT_NODE_LABELS: tuple[str, ...] = (
    "Person",
    "Org",
    "Place",
    "Topic",
    "Skill",
    "Activity",
    "LifeEvent",
    "Memory",
)

SAMPLE_QUERIES: Mapping[str, str] = {
    "STUDIED_AT": (
        "MATCH (p:Person)-[r:STUDIED_AT]->(i) "
        "WITH COALESCE(p.realName, p.id) AS person, i.name AS institution, properties(r) AS rel "
        "RETURN person, institution, rel['degreeType'] AS degree, rel['fieldOfStudy'] AS field, "
        "rel['status'] AS status, rel['graduationYear'] AS graduation_year, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
    "HAS_SKILL": (
        "MATCH (p:Person)-[r:HAS_SKILL]->(s) "
        "WITH COALESCE(p.realName, p.id) AS person, s.name AS skill, properties(r) AS rel "
        "RETURN person, skill, rel['proficiency'] AS proficiency, rel['yearsExperience'] AS years_experience, "
        "rel['learningStatus'] AS learning_status, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
    "WORKING_ON": (
        "MATCH (p:Person)-[r:WORKING_ON]->(proj) "
        "WITH COALESCE(p.realName, p.id) AS person, proj.name AS project, properties(r) AS rel "
        "RETURN person, project, rel['role'] AS role, rel['projectType'] AS project_type, "
        "rel['collaborationMode'] AS collaboration_mode, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
    "RELATED_TO": (
        "MATCH (a:Person)-[r:RELATED_TO]->(b) "
        "WITH a, r, CASE WHEN b:Person THEN COALESCE(b.realName, b.id) ELSE b.name END AS counterpart "
        "RETURN DISTINCT COALESCE(a.realName, a.id) AS person, counterpart, r.relationshipType AS relationship_type, "
        "r.relationshipBasis AS relationship_basis, r.confidence AS confidence "
        "ORDER BY r.confidence DESC LIMIT $limit"
    ),
    "ATTENDED_EVENT": (
        "MATCH (p:Person)-[r:ATTENDED_EVENT]->(e) "
        "WITH COALESCE(p.realName, p.id) AS person, e.name AS event, properties(r) AS rel, properties(e) AS event_props "
        "RETURN person, event, rel['date'] AS date, rel['role'] AS role, rel['format'] AS format, "
        "COALESCE(rel['location'], event_props['location']) AS location, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
    "PREFERS": (
        "MATCH (p:Person)-[r:PREFERS]->(pref) "
        "WITH COALESCE(p.realName, p.id) AS person, pref.name AS preference, properties(r) AS rel "
        "RETURN person, preference, rel['category'] AS category, rel['strength'] AS strength, "
        "rel['reason'] AS reason, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
    "BELIEVES": (
        "MATCH (p:Person)-[r:BELIEVES]->(t) "
        "WITH COALESCE(p.realName, p.id) AS person, t.name AS topic, properties(r) AS rel "
        "RETURN person, topic, rel['stance'] AS stance, rel['conviction'] AS conviction, rel['reasoning'] AS reasoning, "
        "rel['confidence'] AS confidence ORDER BY confidence DESC LIMIT $limit"
    ),
    "ENJOYS": (
        "MATCH (p:Person)-[r:ENJOYS]->(act) "
        "WITH COALESCE(p.realName, p.id) AS person, act.name AS activity, properties(r) AS rel "
        "RETURN person, activity, rel['enjoymentLevel'] AS enjoyment_level, rel['frequency'] AS frequency, "
        "rel['socialMode'] AS social_mode, rel['confidence'] AS confidence "
        "ORDER BY confidence DESC LIMIT $limit"
    ),
}


@dataclass(slots=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str


def _format_table(items: Iterable[Mapping[str, object]]) -> str:
    rows = list(items)
    if not rows:
        return "  (no rows)"
    headers = list(rows[0].keys())
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for idx, header in enumerate(headers):
            widths[idx] = max(widths[idx], len(str(row.get(header, ""))))

    def render_row(values: Sequence[object]) -> str:
        return "  " + "  |  ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values))

    header_line = render_row(headers)
    separator = "  " + "--+-".join("-" * width for width in widths)
    data_lines = [render_row([row.get(header, "") for header in headers]) for row in rows]
    return "\n".join([header_line, separator, *data_lines])


def fetch_existing_labels(session) -> set[str]:
    # Neo4j 4.x uses CALL db.labels(), 5.x supports SHOW NODE LABELS.
    try:
        result = session.run("SHOW NODE LABELS YIELD name")
        return {record["name"] for record in result if record.get("name")}
    except Exception:
        result = session.run("CALL db.labels()")
        return {record.get("label") for record in result if record.get("label")}


def fetch_existing_relationship_types(session) -> set[str]:
    try:
        result = session.run("SHOW RELATIONSHIP TYPES YIELD name")
        return {record["name"] for record in result if record.get("name")}
    except Exception:
        result = session.run("CALL db.relationshipTypes()")
        return {record.get("relationshipType") for record in result if record.get("relationshipType")}


def node_counts(session, labels: Sequence[str]) -> Mapping[str, int]:
    existing = fetch_existing_labels(session)
    counts: dict[str, int] = {}
    for label in labels:
        if label not in existing:
            continue
        result = session.run(f"MATCH (n:{label}) RETURN count(n) AS count")
        counts[label] = result.single().get("count", 0)
    return counts


def relationship_counts(session, *, limit: int = 25) -> list[Mapping[str, object]]:
    result = session.run(
        "MATCH ()-[r]->() "
        "RETURN type(r) AS type, count(r) AS count "
        "ORDER BY count DESC LIMIT $limit",
        limit=limit,
    )
    return [record.data() for record in result]


def sample_relationships(session, *, per_type_limit: int, available_types: set[str]) -> Mapping[str, list[Mapping[str, object]]]:
    samples: dict[str, list[Mapping[str, object]]] = {}
    for rel_type, query in SAMPLE_QUERIES.items():
        if rel_type not in available_types:
            continue
        result = session.run(query, limit=per_type_limit)
        rows = [record.data() for record in result]
        if rows:
            samples[rel_type] = rows
    return samples


def run_snapshot(config: Neo4jConfig, *, per_type_limit: int) -> None:
    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))
    try:
        with driver.session() as session:
            print("=== Node Counts ===")
            counts = node_counts(session, DEFAULT_NODE_LABELS)
            for label, count in counts.items():
                if count:
                    print(f"  {label}: {count}")

            print("\n=== Relationship Counts ===")
            rel_counts = relationship_counts(session)
            print(_format_table(rel_counts))

            print("\n=== Sample Facts ===")
            relationship_types = fetch_existing_relationship_types(session)
            samples = sample_relationships(
                session,
                per_type_limit=per_type_limit,
                available_types=relationship_types,
            )
            if not samples:
                print("  (no facts returned)")
            for rel_type, rows in samples.items():
                print(f"\n-- {rel_type} --")
                print(_format_table(rows))
    finally:
        driver.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display a snapshot of materialized facts from Neo4j."
    )
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", required=True, help="Neo4j password")
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Maximum rows to fetch per fact type sample",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Neo4jConfig(uri=args.uri, user=args.user, password=args.password)
    run_snapshot(config, per_type_limit=max(1, args.sample_limit))


if __name__ == "__main__":
    main()
