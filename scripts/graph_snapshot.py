#!/usr/bin/env python3
"""Inspect Neo4j knowledge graph contents to validate materialized facts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence
import statistics

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


@dataclass(slots=True)
class WeeklyMessageSeries:
    member: str
    total_messages: int
    weekly_counts: list[tuple[str, int]]


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


def facts_temporal_breakdown(session) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    month_result = session.run(
        """
        MATCH ()-[r]->()
        WHERE r.factId IS NOT NULL AND r.lastUpdated IS NOT NULL
        WITH date(r.lastUpdated) AS d
        WITH date({year: d.year, month: d.month, day: 1}) AS month, count(*) AS fact_count
        RETURN month, fact_count
        ORDER BY month
        """
    )
    month_rows = [record.data() for record in month_result]

    quarter_result = session.run(
        """
        MATCH ()-[r]->()
        WHERE r.factId IS NOT NULL AND r.lastUpdated IS NOT NULL
        WITH date(r.lastUpdated) AS d
        WITH d.year AS year, toInteger(floor((d.month - 1) / 3)) + 1 AS quarter, count(*) AS fact_count
        RETURN year, quarter, fact_count
        ORDER BY year, quarter
        """
    )
    quarter_rows = [record.data() for record in quarter_result]
    return month_rows, quarter_rows


def _detect_spikes(
    rows: Sequence[Mapping[str, object]], value_key: str
) -> tuple[list[Mapping[str, object]], float, float, float]:
    values = [float(row.get(value_key, 0) or 0) for row in rows]
    if not values:
        return [], 0.0, 0.0, 0.0
    if len(values) == 1:
        value = values[0]
        return list(rows), value, 0.0, value
    avg = statistics.mean(values)
    std = statistics.pstdev(values)
    threshold = avg + max(std, 0.5 * avg)
    spikes = [row for row in rows if float(row.get(value_key, 0) or 0) >= threshold]
    if not spikes:
        spikes = sorted(rows, key=lambda row: float(row.get(value_key, 0) or 0))[-3:]
    return spikes, avg, std, threshold


def weekly_message_series(
    session,
    *,
    member_limit: int,
) -> list[WeeklyMessageSeries]:
    """Return weekly message counts for the most active members."""
    query = (
        "MATCH (p:Person)-[:SENT]->(m:Message) "
        "WHERE m.timestamp IS NOT NULL "
        "WITH COALESCE(p.realName, p.id) AS member, date(datetime(m.timestamp)) AS message_date "
        "WITH member, message_date.weekYear AS week_year, message_date.week AS week, count(*) AS message_count "
        "ORDER BY member, week_year, week "
        "WITH member, collect({week_year: week_year, week: week, message_count: message_count}) AS weekly_counts, "
        "sum(message_count) AS total_messages "
        "ORDER BY total_messages DESC "
        "LIMIT $member_limit "
        "RETURN member, total_messages, weekly_counts"
    )
    result = session.run(query, member_limit=member_limit)
    series: list[WeeklyMessageSeries] = []
    for record in result:
        weekly_counts: list[tuple[str, int]] = []
        for entry in record["weekly_counts"]:
            week_year = entry.get("week_year")
            week = entry.get("week")
            if week_year is None or week is None:
                continue
            week_label = f"{int(week_year)}-W{int(week):02d}"
            weekly_counts.append((week_label, int(entry["message_count"])))
        series.append(
            WeeklyMessageSeries(
                member=record["member"],
                total_messages=int(record["total_messages"]),
                weekly_counts=weekly_counts,
            )
        )
    return series


def run_snapshot(
    config: Neo4jConfig,
    *,
    per_type_limit: int,
    message_member_limit: int,
) -> None:
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

            print("\n=== Facts Over Time ===")
            month_rows, quarter_rows = facts_temporal_breakdown(session)

            if month_rows:
                month_table = [
                    {
                        "month": str(row["month"]),
                        "fact_count": row["fact_count"],
                    }
                    for row in month_rows
                ]
                print("\n-- By Month --")
                print(_format_table(month_table))
                spikes, avg, std, threshold = _detect_spikes(month_rows, "fact_count")
                if spikes:
                    print(
                        f"  Spikes (avg={avg:.1f}, std={std:.1f}, threshold≈{threshold:.1f}):"
                    )
                    for row in spikes:
                        print(f"    - {row['month']}: {row['fact_count']} facts")
            else:
                print("  (no timestamped facts)")

            if quarter_rows:
                quarter_table = [
                    {
                        "quarter": f"Q{row['quarter']} {row['year']}",
                        "fact_count": row["fact_count"],
                    }
                    for row in quarter_rows
                ]
                print("\n-- By Quarter --")
                print(_format_table(quarter_table))
                spikes, avg, std, threshold = _detect_spikes(quarter_rows, "fact_count")
                if spikes:
                    print(
                        f"  Spikes (avg={avg:.1f}, std={std:.1f}, threshold≈{threshold:.1f}):"
                    )
                    for row in spikes:
                        print(
                            f"    - Q{row['quarter']} {row['year']}: {row['fact_count']} facts"
                        )

            print("\n=== Member Message Volume ===")
            message_series = weekly_message_series(
                session, member_limit=message_member_limit
            )
            if not message_series:
                print("  (no messages)")
            else:
                summary_rows = []
                member_week_counts: dict[str, dict[str, int]] = {}
                for series in message_series:
                    week_count = len(series.weekly_counts)
                    avg_per_week = (
                        series.total_messages / week_count
                        if week_count
                        else series.total_messages
                    )
                    summary_rows.append(
                        {
                            "member": series.member,
                            "total_messages": series.total_messages,
                            "weeks_active": week_count,
                            "avg_per_week": f"{avg_per_week:.1f}",
                        }
                    )
                    counts: dict[str, int] = {}
                    for week, count in series.weekly_counts:
                        counts[week] = count
                    member_week_counts[series.member] = counts

                print("\n-- Summary --")
                print(_format_table(summary_rows))

                members = [row["member"] for row in summary_rows]
                all_weeks = sorted(
                    {
                        week
                        for counts in member_week_counts.values()
                        for week in counts.keys()
                    },
                    key=lambda label: (
                        int(label.split("-W")[0]),
                        int(label.split("-W")[1]),
                    ),
                )

                weekly_table = []
                for week in all_weeks:
                    row = {"week": week}
                    for member in members:
                        row[member] = member_week_counts.get(member, {}).get(week, 0)
                    weekly_table.append(row)

                print("\n-- Weekly Counts --")
                print(_format_table(weekly_table))
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
    parser.add_argument(
        "--message-member-limit",
        type=int,
        default=5,
        help="Number of members to include when displaying weekly message counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Neo4jConfig(uri=args.uri, user=args.user, password=args.password)
    run_snapshot(
        config,
        per_type_limit=max(1, args.sample_limit),
        message_member_limit=max(1, args.message_member_limit),
    )


if __name__ == "__main__":
    main()
