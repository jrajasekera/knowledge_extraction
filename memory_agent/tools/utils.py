"""Shared helper utilities for tool implementations."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from neo4j import Session

from .base import ToolContext, ToolError


logger = logging.getLogger(__name__)


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


def run_read_query(context: ToolContext, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
) -> list[dict[str, Any]]:
    """Execute a vector index query, applying optional filters client-side."""
    parameters: dict[str, Any] = {
        "index_name": index_name,
        "limit": limit,
        "embedding": embedding,
    }
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
