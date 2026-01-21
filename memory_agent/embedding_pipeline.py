"""Utilities to generate and persist fact embeddings in Neo4j."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from neo4j import Driver, Session
from tqdm import tqdm

from constants import EMBEDDING_VECTOR_DIMENSIONS
from .embedding_utils import chunk_iterable, sanitize_array, sanitize_evidence, serialize_attributes
from .embeddings import EmbeddingProvider
from .fact_formatter import format_fact_for_embedding_text


logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "fact_embeddings"
VECTOR_DIMENSIONS = EMBEDDING_VECTOR_DIMENSIONS


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
    evidence_text: Sequence[str]
    target_labels: Sequence[str]


def _session_kwargs(database: str | None) -> dict[str, str]:
    return {"database": database} if database else {}


def ensure_indices(session: Session, index_name: str = VECTOR_INDEX_NAME) -> None:
    """Create both vector and fulltext indices for hybrid search if absent."""
    # Vector index for semantic search
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

    # Fulltext index for keyword/exact matching
    logger.info("Ensuring Neo4j fulltext index fact_fulltext exists")
    session.run(
        """
        CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
        FOR (f:FactEmbedding)
        ON EACH [f.text, f.person_name, f.fact_object]
        """
    )


def fetch_graph_facts(session: Session) -> list[GraphFact]:
    """Retrieve all relationship-backed facts from Neo4j."""
    logger.info("Fetching fact relationships from Neo4j")
    result = session.run(
        """
        MATCH (person:Person)-[r]->(target)
        WHERE r.factId IS NOT NULL
        WITH person, r, target, r.evidence AS evidence_ids
        OPTIONAL MATCH (author:Person)-[:SENT]->(msg:Message)
        WHERE msg.id IN evidence_ids
        WITH person, r, target, evidence_ids,
             collect(coalesce(author.realName, author.name, author.id) + ': ' + msg.content) AS evidence_texts
        RETURN
            r.factId AS fact_id,
            person.id AS person_id,
            coalesce(person.realName, person.name, person.id) AS person_name,
            type(r) AS fact_type,
            coalesce(target.name, target.label, target.title, target.id) AS fact_object,
            properties(r) AS relationship_properties,
            labels(target) AS target_labels,
            evidence_ids,
            evidence_texts
        """
    )
    facts: list[GraphFact] = []
    for row in result:
        rel_props = dict(row["relationship_properties"] or {})
        confidence = rel_props.get("confidence")
        evidence = list(row["evidence_ids"] or [])
        evidence_text = list(row["evidence_texts"] or [])
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
                evidence=evidence,
                evidence_text=evidence_text,
                target_labels=list(row["target_labels"] or []),
            )
        )
    logger.info("Loaded %d facts from graph relationships", len(facts))
    return facts


def _embed_fact_batch(
    batch: Sequence[GraphFact],
    model_name: str,
    device: str,
    cache_dir: str | None,
) -> list[tuple[GraphFact, str, list[float]]]:
    """Worker function to embed a single batch of facts (runs in subprocess)."""
    # Each worker creates its own provider instance
    provider = EmbeddingProvider(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    texts = [
        format_fact_for_embedding_text(
            person_name=fact.person_name,
            fact_type=fact.fact_type,
            fact_object=fact.fact_object,
            attributes=fact.attributes,
            evidence_text=fact.evidence_text,
        )
        for fact in batch
    ]

    embeddings = provider.embed(texts)
    return list(zip(batch, texts, embeddings, strict=False))


def generate_embeddings(
    facts: Sequence[GraphFact],
    provider: EmbeddingProvider,
    *,
    batch_size: int = 64,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """Create embedding payloads ready to be persisted in Neo4j.

    Args:
        facts: Sequence of facts to embed
        provider: Embedding provider (config will be extracted for workers)
        batch_size: Number of facts per batch
        workers: Number of parallel workers (1 = sequential, >1 = parallel)
    """
    rows: list[dict[str, Any]] = []
    batches = list(chunk_iterable(facts, batch_size))

    if workers <= 1:
        # Sequential processing (original behavior)
        logger.info("Processing %d batches sequentially", len(batches))
        with tqdm(
            total=len(facts),
            desc="Embedding facts",
            unit="fact",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            for batch in batches:
                texts = [
                    format_fact_for_embedding_text(
                        person_name=fact.person_name,
                        fact_type=fact.fact_type,
                        fact_object=fact.fact_object,
                        attributes=fact.attributes,
                        evidence_text=fact.evidence_text,
                    )
                    for fact in batch
                ]
                embeddings = provider.embed(texts)
                results_count = min(len(batch), len(embeddings))
                for fact, text, embedding in zip(batch, texts, embeddings, strict=False):
                    rows.append(
                        {
                            "fact_id": fact.fact_id,
                            "person_id": fact.person_id,
                            "person_name": fact.person_name,
                            "fact_type": fact.fact_type,
                            "fact_object": fact.fact_object,
                            "attributes_json": serialize_attributes(fact.attributes),
                            "confidence": fact.confidence,
                            "evidence": sanitize_evidence(fact.evidence),
                            "target_labels": sanitize_array(fact.target_labels),
                            "text": text,
                            "embedding": embedding,
                        }
                    )
                pbar.update(results_count)
    else:
        # Parallel processing with multiple workers
        logger.info("Processing %d batches with %d workers", len(batches), workers)
        with tqdm(
            total=len(facts),
            desc="Embedding facts",
            unit="fact",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ) as pbar:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # Submit all batches
                futures = {
                    executor.submit(
                        _embed_fact_batch,
                        batch,
                        provider.model_name,
                        provider.device,
                        provider.cache_dir,
                    ): batch_idx
                    for batch_idx, batch in enumerate(batches)
                }

                # Collect results as they complete
                for future in as_completed(futures):
                    batch_idx = futures[future]
                    try:
                        results = future.result()
                        for fact, text, embedding in results:
                            rows.append(
                                {
                                    "fact_id": fact.fact_id,
                                    "person_id": fact.person_id,
                                    "person_name": fact.person_name,
                                    "fact_type": fact.fact_type,
                                    "fact_object": fact.fact_object,
                                    "attributes_json": serialize_attributes(fact.attributes),
                                    "confidence": fact.confidence,
                                    "evidence": sanitize_evidence(fact.evidence),
                                    "target_labels": sanitize_array(fact.target_labels),
                                    "text": text,
                                    "embedding": embedding,
                                }
                            )
                        pbar.update(len(results))
                    except Exception as exc:
                        logger.error("Batch %d failed with error: %s", batch_idx, exc)
                        raise

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
    batch_size: int = 64,
    workers: int = 1,
) -> dict[str, Any]:
    """High-level pipeline orchestration.

    Args:
        driver: Neo4j driver
        provider: Embedding provider
        database: Optional Neo4j database name
        cleanup: Whether to remove orphan embeddings
        batch_size: Number of facts per batch
        workers: Number of parallel workers (1 = sequential)
    """
    summary: dict[str, Any] = {}
    with driver.session(**_session_kwargs(database)) as session:
        ensure_indices(session)
        facts = fetch_graph_facts(session)
    if not facts:
        logger.info("No facts found; skipping embedding generation")
        summary["facts_processed"] = 0
        return summary
    rows = generate_embeddings(facts, provider, batch_size=batch_size, workers=workers)
    with driver.session(**_session_kwargs(database)) as session:
        upsert_embeddings(session, rows, embedding_model=provider.model_name)
        if cleanup:
            cleanup_orphan_embeddings(session)
    summary["facts_processed"] = len(facts)
    summary["embeddings_written"] = len(rows)
    return summary
