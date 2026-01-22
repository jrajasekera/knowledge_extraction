"""Tests for memory_agent/tools/utils.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from memory_agent.tools.base import ToolContext, ToolError
from memory_agent.tools.utils import (
    LUCENE_MAX_CLAUSES,
    LUCENE_MIN_TERM_LEN,
    LUCENE_STOPWORDS,
    LUCENE_TERM_CLAUSE_COST,
    managed_session,
    run_keyword_query,
    run_read_query,
    run_vector_query,
)


@pytest.fixture
def mock_context() -> ToolContext:
    """Create a mock ToolContext with a Neo4j driver."""
    mock_driver = MagicMock()
    return ToolContext(driver=mock_driver)


@pytest.fixture
def mock_session():
    """Create a mock Neo4j session."""
    session = MagicMock()
    return session


# Tests for managed_session


def test_managed_session_yields_session(mock_context: ToolContext) -> None:
    """managed_session should yield a Neo4j session."""
    mock_session = MagicMock()
    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    with managed_session(mock_context) as session:
        assert session is mock_session

    mock_session.close.assert_called_once()


def test_managed_session_closes_on_success(mock_context: ToolContext) -> None:
    """managed_session should close session after successful execution."""
    mock_session = MagicMock()
    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    with managed_session(mock_context):
        pass

    mock_session.close.assert_called_once()


def test_managed_session_closes_on_exception(mock_context: ToolContext) -> None:
    """managed_session should close session even on exception."""
    mock_session = MagicMock()
    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    with pytest.raises(ToolError), managed_session(mock_context):
        raise RuntimeError("Test error")

    mock_session.close.assert_called_once()


def test_managed_session_wraps_exception_in_tool_error(mock_context: ToolContext) -> None:
    """managed_session should wrap exceptions in ToolError."""
    mock_session = MagicMock()
    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    with pytest.raises(ToolError) as exc_info, managed_session(mock_context):
        raise ValueError("Original error")

    assert "Original error" in str(exc_info.value)


# Tests for run_read_query


def test_run_read_query_returns_records(mock_context: ToolContext) -> None:
    """run_read_query should return list of record dictionaries."""
    mock_session = MagicMock()
    mock_record1 = MagicMock()
    mock_record1.data.return_value = {"name": "Alice", "age": 30}
    mock_record2 = MagicMock()
    mock_record2.data.return_value = {"name": "Bob", "age": 25}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record1, mock_record2])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    result = run_read_query(mock_context, "MATCH (n) RETURN n")

    assert len(result) == 2
    assert result[0]["name"] == "Alice"
    assert result[1]["name"] == "Bob"


def test_run_read_query_with_parameters(mock_context: ToolContext) -> None:
    """run_read_query should pass parameters to session.run."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_read_query(mock_context, "MATCH (n:Person {id: $id})", {"id": "user-1"})

    mock_session.run.assert_called_once_with("MATCH (n:Person {id: $id})", {"id": "user-1"})


def test_run_read_query_empty_parameters(mock_context: ToolContext) -> None:
    """run_read_query should pass empty dict when no parameters."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_read_query(mock_context, "MATCH (n) RETURN n")

    mock_session.run.assert_called_once_with("MATCH (n) RETURN n", {})


# Tests for run_vector_query


def test_run_vector_query_basic(mock_context: ToolContext) -> None:
    """run_vector_query should execute vector index query."""
    mock_session = MagicMock()
    mock_node = MagicMock()
    mock_node.__iter__ = lambda self: iter({"id": "fact-1"}.items())
    mock_record = MagicMock()
    mock_record.data.return_value = {"node": mock_node, "score": 0.95}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    embedding = [0.1, 0.2, 0.3]
    result = run_vector_query(mock_context, "fact_embeddings", embedding, limit=10)

    assert len(result) == 1
    assert result[0]["score"] == 0.95


def test_run_vector_query_without_evidence(mock_context: ToolContext) -> None:
    """run_vector_query with include_evidence=False should use simpler query."""
    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.data.return_value = {"node": MagicMock(), "score": 0.9}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_vector_query(
        mock_context,
        "fact_embeddings",
        [0.1, 0.2],
        limit=5,
        include_evidence=False,
    )

    # Verify the query doesn't include evidence matching
    call_args = mock_session.run.call_args
    query = call_args[0][0]
    assert "evidence_with_content" not in query


def test_run_vector_query_with_filters(mock_context: ToolContext) -> None:
    """run_vector_query should apply client-side filters."""
    mock_session = MagicMock()

    # Create dict-like nodes that work with dict()
    class DictLikeNode(dict):
        pass

    node1 = DictLikeNode({"fact_type": "WORKS_AT"})
    node2 = DictLikeNode({"fact_type": "HAS_SKILL"})

    mock_record1 = MagicMock()
    mock_record1.data.return_value = {"node": node1, "score": 0.95}
    mock_record2 = MagicMock()
    mock_record2.data.return_value = {"node": node2, "score": 0.90}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record1, mock_record2])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    result = run_vector_query(
        mock_context,
        "fact_embeddings",
        [0.1, 0.2],
        limit=10,
        filters={"fact_type": "WORKS_AT"},
    )

    assert len(result) == 1
    assert result[0]["node"]["fact_type"] == "WORKS_AT"


def test_run_vector_query_filter_list_match(mock_context: ToolContext) -> None:
    """run_vector_query should filter with list of expected values."""
    mock_session = MagicMock()

    # Create dict-like node
    class DictLikeNode(dict):
        pass

    node = DictLikeNode({"fact_type": "HAS_SKILL"})

    mock_record = MagicMock()
    mock_record.data.return_value = {"node": node, "score": 0.9}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    result = run_vector_query(
        mock_context,
        "fact_embeddings",
        [0.1],
        limit=10,
        filters={"fact_type": ["WORKS_AT", "HAS_SKILL"]},
    )

    assert len(result) == 1


def test_run_vector_query_skips_none_nodes(mock_context: ToolContext) -> None:
    """run_vector_query should skip records with None nodes when filtering."""
    mock_session = MagicMock()

    mock_record = MagicMock()
    mock_record.data.return_value = {"node": None, "score": 0.9}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    result = run_vector_query(
        mock_context,
        "fact_embeddings",
        [0.1],
        limit=10,
        filters={"fact_type": "WORKS_AT"},
    )

    assert len(result) == 0


# Tests for run_keyword_query


def test_run_keyword_query_basic(mock_context: ToolContext) -> None:
    """run_keyword_query should execute fulltext search."""
    mock_session = MagicMock()
    mock_record = MagicMock()
    mock_record.data.return_value = {"node": MagicMock(), "score": 0.8}

    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([mock_record])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    result = run_keyword_query(mock_context, "python programming", limit=10)

    assert len(result) == 1


def test_run_keyword_query_sanitizes_input(mock_context: ToolContext) -> None:
    """run_keyword_query should sanitize special characters."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_keyword_query(mock_context, "test!@#$%^&*()query", limit=10)

    call_args = mock_session.run.call_args
    params = call_args[0][1]
    lucene_query = params["lucene_query"]

    # Should not contain special characters
    assert "!" not in lucene_query
    assert "@" not in lucene_query
    assert "#" not in lucene_query


def test_run_keyword_query_empty_after_sanitize(mock_context: ToolContext) -> None:
    """run_keyword_query should return empty list if query sanitizes to empty."""
    result = run_keyword_query(mock_context, "!@#$%", limit=10)

    assert result == []


def test_run_keyword_query_filters_short_terms(mock_context: ToolContext) -> None:
    """run_keyword_query should filter terms shorter than LUCENE_MIN_TERM_LEN."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_keyword_query(mock_context, "a b python", limit=10)

    call_args = mock_session.run.call_args
    params = call_args[0][1]
    lucene_query = params["lucene_query"]

    # Should only contain "python" (a and b are too short)
    assert "python" in lucene_query


def test_run_keyword_query_removes_duplicates(mock_context: ToolContext) -> None:
    """run_keyword_query should remove duplicate terms."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_keyword_query(mock_context, "python Python PYTHON", limit=10)

    call_args = mock_session.run.call_args
    params = call_args[0][1]
    lucene_query = params["lucene_query"]

    # Should only have one occurrence of "python"
    assert lucene_query.count("python") == 2  # "python OR python~"


def test_run_keyword_query_applies_fuzzy_matching(mock_context: ToolContext) -> None:
    """run_keyword_query should add fuzzy matching to terms."""
    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter([])
    mock_session.run.return_value = mock_result

    mock_context.driver.session.return_value = mock_session  # type: ignore[union-attr]

    run_keyword_query(mock_context, "python", limit=10)

    call_args = mock_session.run.call_args
    params = call_args[0][1]
    lucene_query = params["lucene_query"]

    assert "python OR python~" in lucene_query


def test_run_keyword_query_zero_terms_after_filter(mock_context: ToolContext) -> None:
    """run_keyword_query should return empty if all terms are filtered out."""
    # All single-character terms that get filtered
    result = run_keyword_query(mock_context, "a b c d", limit=10)

    assert result == []


def test_lucene_constants() -> None:
    """Verify LUCENE constants have expected values."""
    assert LUCENE_TERM_CLAUSE_COST == 2
    assert LUCENE_MIN_TERM_LEN == 2
    assert "the" in LUCENE_STOPWORDS
    assert "and" in LUCENE_STOPWORDS
    assert isinstance(LUCENE_MAX_CLAUSES, int)
