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
    """Execute a vector index query."""
    parameters: dict[str, Any] = {
        "index_name": index_name,
        "limit": limit,
        "embedding": embedding,
    }
    if filters:
        parameters["filters"] = filters
        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding, $filters)
        YIELD node, score
        RETURN node, score
        """
    else:
        query = """
        CALL db.index.vector.queryNodes($index_name, $limit, $embedding)
        YIELD node, score
        RETURN node, score
        """
    return run_read_query(context, query, parameters)
