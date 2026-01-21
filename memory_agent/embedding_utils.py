"""Shared helpers for embedding pipelines."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from typing import Any


def chunk_iterable[T](values: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield successive chunks from the sequence."""

    if size <= 0:
        msg = "size must be a positive integer"
        raise ValueError(msg)
    for start in range(0, len(values), size):
        yield values[start : start + size]


def sanitize_property_value(value: Any) -> Any:
    """Convert arbitrary values into Neo4j-friendly scalars."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def serialize_attributes(attributes: dict[str, Any] | None) -> str | None:
    """Serialize attribute dictionaries into sorted JSON strings."""

    if not attributes:
        return None
    sanitized = {key: sanitize_property_value(val) for key, val in attributes.items()}
    return json.dumps(sanitized, sort_keys=True)


def sanitize_array(values: Sequence[Any] | None) -> list[str]:
    """Convert a sequence into a list of strings, skipping None values."""

    if not values:
        return []
    return [str(value) for value in values if value is not None]


def sanitize_evidence(evidence: Sequence[Any] | None) -> list[str]:
    """Deduplicate evidence identifiers while preserving order."""

    if not evidence:
        return []
    seen: set[str] = set()
    deduped: list[str] = []
    for value in evidence:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


__all__ = [
    "chunk_iterable",
    "sanitize_property_value",
    "sanitize_array",
    "sanitize_evidence",
    "serialize_attributes",
]
