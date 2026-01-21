"""Shared helper utilities for tool implementations."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Session

from .base import ToolContext, ToolError

logger = logging.getLogger(__name__)

LUCENE_MAX_CLAUSES = int(os.getenv("LUCENE_MAX_CLAUSES", "4096"))
LUCENE_TERM_CLAUSE_COST = 2  # Each term is expanded to `term OR term~`.
LUCENE_MIN_TERM_LEN = 2
LUCENE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


@contextmanager
def managed_session(context: ToolContext) -> Iterator[Session]:
    """Context manager yielding a Neo4j session with consistent error handling."""
    session = context.session()
    try:
        yield session
    except Exception as exc:  # noqa: BLE001
        logger.exception("Neo4j session execution failed: %s", exc)
        raise ToolError(str(exc)) from exc
    finally:
        session.close()


def run_read_query(
    context: ToolContext, query: str, parameters: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Execute a read-only Cypher query and return list of dictionaries."""
    with managed_session(context) as session:
        result = session.run(query, parameters or {})
        return [record.data() for record in result]


def run_vector_query(
    context: ToolContext,
    index_name: str,
    embedding: list[float],
    limit: int,
    filters: dict[str, Any] | None = None,
    *,
    include_evidence: bool = True,
) -> list[dict[str, Any]]:
    """Execute a vector index query, applying optional filters client-side."""
    parameters: dict[str, Any] = {
        "index_name": index_name,
        "limit": limit,
        "embedding": embedding,
    }
    if include_evidence:
        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        WITH node, score, node.evidence AS evidence_ids
        OPTIONAL MATCH (author:Person)-[:SENT]->(msg:Message)
        WHERE msg.id IN evidence_ids
        WITH node, score,
             COLLECT({source_id: msg.id, snippet: msg.content, created_at: msg.timestamp, author: coalesce(author.realName, author.name)}) AS evidence_with_content
        RETURN node, score, evidence_with_content
        """
    else:
        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        RETURN node, score
        """
    rows = run_read_query(context, query, parameters)
    if not filters:
        return rows

    filtered_rows: list[dict[str, Any]] = []
    for row in rows:
        node = row.get("node")
        if node is None:
            continue
        properties = dict(node)
        match = True
        for key, expected in filters.items():
            value = properties.get(key)
            if isinstance(expected, (list, tuple, set)):
                expected_set = set(expected)
                if isinstance(value, (list, tuple, set)):
                    if not expected_set.intersection(value):
                        match = False
                        break
                else:
                    if value not in expected_set:
                        match = False
                        break
            else:
                if value != expected:
                    match = False
                    break
        if match:
            filtered_rows.append(row)
    return filtered_rows


def run_keyword_query(
    context: ToolContext,
    query_text: str,
    limit: int,
    *,
    index_name: str = "fact_fulltext",
) -> list[dict[str, Any]]:
    """Execute a fulltext keyword search using Lucene syntax with fuzzy matching.

    Args:
        context: Tool execution context with Neo4j session
        query_text: The search query string
        limit: Maximum number of results to return
        index_name: Name of the fulltext index (default: fact_fulltext)

    Returns:
        List of dictionaries with 'node', 'score', and 'evidence_with_content' keys,
        matching the format returned by run_vector_query.
    """
    # Sanitize query for Lucene: keep alphanumeric, spaces, hyphens, underscores
    safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query_text).strip()

    if not safe_text:
        logger.warning("Keyword query sanitized to empty string, returning no results")
        return []

    # Construct Lucene query: exact terms OR fuzzy terms (~ for edit distance)
    # For multi-word queries, apply fuzzy matching to each term
    raw_terms = safe_text.split()
    seen_terms: set[str] = set()
    terms: list[str] = []
    for term in raw_terms:
        lowered = term.lower()
        if lowered in seen_terms or len(lowered) < LUCENE_MIN_TERM_LEN:
            continue
        seen_terms.add(lowered)
        terms.append(lowered)

    max_terms = max(1, LUCENE_MAX_CLAUSES // LUCENE_TERM_CLAUSE_COST)
    if len(terms) > max_terms:
        filtered_terms = [term for term in terms if term not in LUCENE_STOPWORDS]
        if filtered_terms:
            terms = filtered_terms

    if len(terms) > max_terms:

        def term_rank(term: str) -> tuple[int, int, int]:
            has_alpha = any(char.isalpha() for char in term)
            is_numeric = term.isdigit()
            return (int(has_alpha), int(not is_numeric), len(term))

        ranked = sorted(
            enumerate(terms),
            key=lambda item: (term_rank(item[1]), -item[0]),
            reverse=True,
        )
        keep = {idx for idx, _ in ranked[:max_terms]}
        original_count = len(terms)
        terms = [term for idx, term in enumerate(terms) if idx in keep]
        logger.info(
            "Keyword query trimmed from %d to %d terms to respect Lucene max clauses (%d)",
            original_count,
            len(terms),
            LUCENE_MAX_CLAUSES,
        )

    if not terms:
        logger.warning("Keyword query reduced to zero terms, returning no results")
        return []

    lucene_parts = [f"{term} OR {term}~" for term in terms]
    lucene_query = " ".join(lucene_parts)

    query = """
    CALL db.index.fulltext.queryNodes($index_name, $lucene_query)
    YIELD node, score
    WITH node, score
    LIMIT $limit
    WITH node, score, node.evidence AS evidence_ids
    OPTIONAL MATCH (author:Person)-[:SENT]->(msg:Message)
    WHERE msg.id IN evidence_ids
    WITH node, score,
         COLLECT({source_id: msg.id, snippet: msg.content, created_at: msg.timestamp, author: coalesce(author.realName, author.name)}) AS evidence_with_content
    RETURN node, score, evidence_with_content
    """

    parameters = {
        "index_name": index_name,
        "lucene_query": lucene_query,
        "limit": limit,
    }

    logger.debug("Keyword query: %r (Lucene: %r)", query_text, lucene_query)
    return run_read_query(context, query, parameters)
