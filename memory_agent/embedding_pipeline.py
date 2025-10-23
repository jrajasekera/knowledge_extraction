"""Utilities to generate and persist fact embeddings in Neo4j."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Sequence

from neo4j import Driver, Session

from .embeddings import EmbeddingProvider
from .fact_formatter import format_fact_for_embedding_text


logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "fact_embeddings"
VECTOR_DIMENSIONS = 768


@dataclass(slots=True)
class GraphFact:
    """Representation of a fact stored in Neo4j relationships."""

    fact_id: int
    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None
    attributes: dict[str, Any]
    confidence: float | None
    evidence: Sequence[str]
    target_labels: Sequence[str]


def _session_kwargs(database: str | None) -> dict[str, str]:
    return {"database": database} if database else {}


def _sanitize_property_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _serialize_attributes(attributes: dict[str, Any] | None) -> str | None:
    if not attributes:
        return None
    sanitized = {key: _sanitize_property_value(val) for key, val in attributes.items()}
    return json.dumps(sanitized, sort_keys=True)


def _sanitize_array(values: Sequence[Any] | None) -> list[str]:
    if not values:
        return []
    return [str(value) for value in values]


def _sanitize_evidence(evidence: Sequence[Any] | None) -> list[str]:
    if not evidence:
        return []
    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for value in evidence:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def ensure_vector_index(session: Session, index_name: str = VECTOR_INDEX_NAME) -> None:
    """Create the vector index used for semantic search if it is absent."""
    logger.info("Ensuring Neo4j vector index %s exists", index_name)
    session.run(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (f:FactEmbedding)
        ON f.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )


def fetch_graph_facts(session: Session) -> list[GraphFact]:
    """Retrieve all relationship-backed facts from Neo4j."""
    logger.info("Fetching fact relationships from Neo4j")
    result = session.run(
        """
        MATCH (person:Person)-[r]->(target)
        WHERE r.factId IS NOT NULL
        RETURN
            r.factId AS fact_id,
            person.id AS person_id,
            coalesce(person.realName, person.name, person.id) AS person_name,
            type(r) AS fact_type,
            coalesce(target.name, target.label, target.title, target.id) AS fact_object,
            properties(r) AS relationship_properties,
            labels(target) AS target_labels
        """
    )
    facts: list[GraphFact] = []
    for row in result:
        rel_props = dict(row["relationship_properties"] or {})
        confidence = rel_props.get("confidence")
        evidence = rel_props.get("evidence") or []
        attributes = {
            key: value
            for key, value in rel_props.items()
            if key not in {"factId", "confidence", "evidence", "lastUpdated"}
        }
        facts.append(
            GraphFact(
                fact_id=int(row["fact_id"]),
                person_id=row["person_id"],
                person_name=row["person_name"],
                fact_type=row["fact_type"],
                fact_object=row["fact_object"],
                attributes=attributes,
                confidence=confidence,
                evidence=list(evidence),
                target_labels=list(row["target_labels"] or []),
            )
        )
    logger.info("Loaded %d facts from graph relationships", len(facts))
    return facts


def chunk_iterable(values: Sequence[GraphFact], size: int) -> Iterator[Sequence[GraphFact]]:
    """Yield successive chunks from the sequence."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def generate_embeddings(
    facts: Sequence[GraphFact],
    provider: EmbeddingProvider,
    *,
    batch_size: int = 64,
) -> list[dict[str, Any]]:
    """Create embedding payloads ready to be persisted in Neo4j."""
    rows: list[dict[str, Any]] = []
    for chunk in chunk_iterable(facts, batch_size):
        texts = [
            format_fact_for_embedding_text(
                person_name=fact.person_name,
                fact_type=fact.fact_type,
                fact_object=fact.fact_object,
                attributes=fact.attributes,
            )
            for fact in chunk
        ]
        embeddings = provider.embed(texts)
        for fact, embedding, text in zip(chunk, embeddings, texts, strict=False):
            rows.append(
                {
                    "fact_id": fact.fact_id,
                    "person_id": fact.person_id,
                    "person_name": fact.person_name,
                    "fact_type": fact.fact_type,
                    "fact_object": fact.fact_object,
                    "attributes_json": _serialize_attributes(fact.attributes),
                    "confidence": fact.confidence,
                    "evidence": _sanitize_evidence(fact.evidence),
                    "target_labels": _sanitize_array(fact.target_labels),
                    "text": text,
                    "embedding": embedding,
                }
            )
    logger.info("Generated embeddings for %d facts", len(rows))
    return rows


def upsert_embeddings(
    session: Session,
    rows: Sequence[dict[str, Any]],
    *,
    embedding_model: str,
) -> None:
    """Persist embedding payloads in Neo4j."""
    if not rows:
        logger.info("No fact embeddings to persist")
        return
    logger.info("Persisting %d fact embeddings", len(rows))
    session.run(
        """
        UNWIND $rows AS row
        MERGE (f:FactEmbedding {fact_id: row.fact_id})
        SET
            f.person_id = row.person_id,
            f.person_name = row.person_name,
            f.fact_type = row.fact_type,
            f.fact_object = row.fact_object,
            f.attributes = row.attributes_json,
            f.confidence = row.confidence,
            f.evidence = row.evidence,
            f.target_labels = row.target_labels,
            f.text = row.text,
            f.embedding = row.embedding,
            f.embedding_model = $embedding_model,
            f.updated_at = datetime(),
            f.created_at = coalesce(f.created_at, datetime())
        """,
        {"rows": [dict(row) for row in rows], "embedding_model": embedding_model},
    )


def cleanup_orphan_embeddings(session: Session) -> None:
    """Remove FactEmbedding nodes whose corresponding relationships no longer exist."""
    logger.info("Cleaning up orphan fact embeddings")
    session.run(
        """
        MATCH (f:FactEmbedding)
        WHERE NOT EXISTS {
            MATCH ()-[r]->()
            WHERE r.factId = f.fact_id
        }
        DETACH DELETE f
        """
    )


def run_embedding_pipeline(
    driver: Driver,
    provider: EmbeddingProvider,
    *,
    database: str | None = None,
    cleanup: bool = True,
) -> dict[str, Any]:
    """High-level pipeline orchestration."""
    summary: dict[str, Any] = {}
    with driver.session(**_session_kwargs(database)) as session:
        ensure_vector_index(session)
        facts = fetch_graph_facts(session)
    if not facts:
        logger.info("No facts found; skipping embedding generation")
        summary["facts_processed"] = 0
        return summary
    rows = generate_embeddings(facts, provider)
    with driver.session(**_session_kwargs(database)) as session:
        upsert_embeddings(session, rows, embedding_model=provider.model_name)
        if cleanup:
            cleanup_orphan_embeddings(session)
    summary["facts_processed"] = len(facts)
    summary["embeddings_written"] = len(rows)
    return summary
