"""Database logger for memory agent API requests and responses."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add parent directory to path for db_utils import
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_utils import get_sqlite_connection

from .models import RetrievalRequest, RetrievalResponse


logger = logging.getLogger(__name__)


class RequestLogger:
    """Logs memory agent API requests and responses to SQLite database."""

    def __init__(self, db_path: Path | str) -> None:
        """Initialize the request logger.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)

    def _get_connection(self):
        """Get a database connection with optimized settings for concurrent access."""
        return get_sqlite_connection(self.db_path, timeout=30.0)

    def log_request_start(
        self,
        request_id: str,
        request_body: RetrievalRequest,
        client_ip: str | None = None,
    ) -> None:
        """Log the start of a request.

        Args:
            request_id: Unique identifier for this request.
            request_body: The retrieval request payload.
            client_ip: Optional client IP address.
        """
        try:
            requested_at = datetime.now(timezone.utc).isoformat()
            request_payload = json.dumps(request_body.model_dump(mode="json"), ensure_ascii=False)

            # Extract query from the last message
            query = request_body.messages[-1].content if request_body.messages else ""

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO memory_agent_request_log (
                        id, requested_at, query, request_payload, status_code, client_ip
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (request_id, requested_at, query, request_payload, 0, client_ip),
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log request start for %s: %s", request_id, exc)

    def log_request_complete(
        self,
        request_id: str,
        response: RetrievalResponse,
        duration_ms: int,
    ) -> None:
        """Log successful completion of a request.

        Args:
            request_id: Unique identifier for this request.
            response: The retrieval response payload.
            duration_ms: Request duration in milliseconds.
        """
        try:
            completed_at = datetime.now(timezone.utc).isoformat()
            response_payload = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
            facts_returned = len(response.facts)
            confidence = self._extract_confidence_value(response.confidence)

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE memory_agent_request_log
                    SET completed_at = ?,
                        duration_ms = ?,
                        status_code = ?,
                        response_payload = ?,
                        facts_returned = ?,
                        confidence = ?
                    WHERE id = ?
                    """,
                    (completed_at, duration_ms, 200, response_payload, facts_returned, confidence, request_id),
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log request completion for %s: %s", request_id, exc)

    def log_request_error(
        self,
        request_id: str,
        status_code: int,
        error_detail: str,
        duration_ms: int,
    ) -> None:
        """Log failed completion of a request.

        Args:
            request_id: Unique identifier for this request.
            status_code: HTTP status code.
            error_detail: Error message or detail.
            duration_ms: Request duration in milliseconds.
        """
        try:
            completed_at = datetime.now(timezone.utc).isoformat()

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE memory_agent_request_log
                    SET completed_at = ?,
                        duration_ms = ?,
                        status_code = ?,
                        error_detail = ?
                    WHERE id = ?
                    """,
                    (completed_at, duration_ms, status_code, error_detail, request_id),
                )
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to log request error for %s: %s", request_id, exc)

    @staticmethod
    def _extract_confidence_value(confidence: Any) -> float | None:
        """Extract numeric confidence value from ConfidenceLevel enum or string.

        Args:
            confidence: Confidence level (enum, string, or number).

        Returns:
            Numeric confidence value between 0 and 1, or None if not extractable.
        """
        if confidence is None:
            return None

        # If it's already a number, return it
        if isinstance(confidence, (int, float)):
            return float(confidence)

        # If it has a value attribute (enum), extract it
        if hasattr(confidence, "value"):
            confidence = confidence.value

        # Map string confidence levels to numeric values
        confidence_str = str(confidence).lower()
        confidence_map = {
            "very_high": 0.9,
            "high": 0.75,
            "medium": 0.5,
            "low": 0.25,
            "very_low": 0.1,
        }

        return confidence_map.get(confidence_str)
