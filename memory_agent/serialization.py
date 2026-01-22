from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def to_serializable(value: Any) -> Any:
    """Convert nested structures into JSON-serializable data."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, Mapping):
        return {str(key): to_serializable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_serializable(item) for item in value]
    if isinstance(value, set):
        return [to_serializable(item) for item in value]
    return value


def json_dumps(value: Any, **kwargs: Any) -> str:
    import json

    return json.dumps(to_serializable(value), **kwargs)
