"""Tests for memory_agent.request_logger module."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest

from memory_agent.models import (
    MessageModel,
    RetrievalMetadata,
    RetrievalRequest,
    RetrievalResponse,
)
from memory_agent.request_logger import RequestLogger


@pytest.fixture
def temp_db():
    """Create a temporary database with the request log schema."""
    with NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
        db_path = Path(temp_file.name)

    # Initialize schema
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_agent_request_log (
          id              TEXT PRIMARY KEY,
          requested_at    TEXT NOT NULL,
          completed_at    TEXT,
          duration_ms     INTEGER,
          query           TEXT NOT NULL,
          request_payload TEXT NOT NULL,
          status_code     INTEGER NOT NULL,
          response_payload TEXT,
          error_detail    TEXT,
          facts_returned  INTEGER,
          confidence      REAL,
          client_ip       TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


def test_log_request_start(temp_db):
    """Test logging request start."""
    logger = RequestLogger(temp_db)

    request = RetrievalRequest(
        messages=[
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="Test query",
                timestamp=datetime.now(timezone.utc),
            )
        ],
        channel_id="test-channel",
    )

    logger.log_request_start("test-request-id", request, "127.0.0.1")

    # Verify the record was created
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.execute("SELECT id, query, status_code, client_ip FROM memory_agent_request_log WHERE id = ?", ("test-request-id",))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "test-request-id"
    assert row[1] == "Test query"
    assert row[2] == 0  # Initial status code
    assert row[3] == "127.0.0.1"


def test_log_request_complete(temp_db):
    """Test logging successful request completion."""
    logger = RequestLogger(temp_db)

    # First, log the request start
    request = RetrievalRequest(
        messages=[
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="Test query",
                timestamp=datetime.now(timezone.utc),
            )
        ],
        channel_id="test-channel",
    )
    logger.log_request_start("test-request-id", request, "127.0.0.1")

    # Now log the completion
    response = RetrievalResponse(
        facts=["fact1", "fact2", "fact3"],
        messages=["msg1", "msg2"],
        context_summary="Test summary",
        confidence="high",
        metadata=RetrievalMetadata(
            queries_executed=2,
            facts_retrieved=3,
            processing_time_ms=1450,
            iterations_used=3,
            tool_calls=[],
        ),
    )
    logger.log_request_complete("test-request-id", response, 1500)

    # Verify the record was updated
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.execute(
        """
        SELECT status_code, duration_ms, facts_returned, confidence, completed_at
        FROM memory_agent_request_log WHERE id = ?
        """,
        ("test-request-id",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 200  # Success status
    assert row[1] == 1500  # Duration
    assert row[2] == 3  # Facts count
    assert row[3] == 0.75  # High confidence mapped to 0.75
    assert row[4] is not None  # Completed timestamp


def test_log_request_error(temp_db):
    """Test logging request error."""
    logger = RequestLogger(temp_db)

    # First, log the request start
    request = RetrievalRequest(
        messages=[
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="Test query",
                timestamp=datetime.now(timezone.utc),
            )
        ],
        channel_id="test-channel",
    )
    logger.log_request_start("test-request-id", request, "127.0.0.1")

    # Now log the error
    logger.log_request_error("test-request-id", 500, "Internal server error", 800)

    # Verify the record was updated
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.execute(
        """
        SELECT status_code, duration_ms, error_detail, completed_at
        FROM memory_agent_request_log WHERE id = ?
        """,
        ("test-request-id",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 500  # Error status
    assert row[1] == 800  # Duration
    assert row[2] == "Internal server error"
    assert row[3] is not None  # Completed timestamp


def test_extract_confidence_value():
    """Test confidence value extraction from different types."""
    logger = RequestLogger(":memory:")

    # Test with string confidence levels
    assert logger._extract_confidence_value("very_high") == 0.9
    assert logger._extract_confidence_value("high") == 0.75
    assert logger._extract_confidence_value("medium") == 0.5
    assert logger._extract_confidence_value("low") == 0.25
    assert logger._extract_confidence_value("very_low") == 0.1

    # Test with numeric values
    assert logger._extract_confidence_value(0.85) == 0.85
    assert logger._extract_confidence_value(1) == 1.0

    # Test with case-insensitive strings
    assert logger._extract_confidence_value("HIGH") == 0.75
    assert logger._extract_confidence_value("MEDIUM") == 0.5

    # Test with None
    assert logger._extract_confidence_value(None) is None


def test_query_extraction_from_multiple_messages(temp_db):
    """Test that query is extracted from the last message."""
    logger = RequestLogger(temp_db)

    now = datetime.now(timezone.utc)
    request = RetrievalRequest(
        messages=[
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="First message",
                timestamp=now,
            ),
            MessageModel(
                author_id="assistant-1",
                author_name="Assistant",
                content="Response",
                timestamp=now,
            ),
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="Last message",
                timestamp=now,
            ),
        ],
        channel_id="test-channel",
    )

    logger.log_request_start("test-request-id", request, None)

    # Verify the query is from the last message
    conn = sqlite3.connect(str(temp_db))
    cursor = conn.execute("SELECT query FROM memory_agent_request_log WHERE id = ?", ("test-request-id",))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "Last message"


def test_logging_failure_is_non_fatal(temp_db):
    """Test that logging failures don't raise exceptions."""
    logger = RequestLogger("/nonexistent/path/db.sqlite")

    # These should not raise exceptions even though the path is invalid
    request = RetrievalRequest(
        messages=[
            MessageModel(
                author_id="user-123",
                author_name="Test User",
                content="Test",
                timestamp=datetime.now(timezone.utc),
            )
        ],
        channel_id="test-channel",
    )

    # Should complete without raising exceptions
    logger.log_request_start("test-id", request, "127.0.0.1")
    logger.log_request_error("test-id", 500, "Error", 100)
