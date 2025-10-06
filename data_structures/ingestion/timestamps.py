from __future__ import annotations

import re
from datetime import datetime, timezone

_OFFSET_PATTERN = re.compile(r"([+-]\d{2}):?(\d{2})(?::(\d{2}))?$")


def normalize_iso_timestamp(raw: str | None) -> str | None:
    """Return a canonical ISO-8601 timestamp or ``None`` if parsing fails."""
    if raw is None:
        return None

    candidate = raw.strip()
    if not candidate:
        return None

    if "T" not in candidate and " " in candidate:
        candidate = candidate.replace(" ", "T", 1)

    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"

    match = _OFFSET_PATTERN.search(candidate)
    if match:
        hours, minutes, _seconds = match.groups()
        candidate = candidate[: match.start()] + f"{hours}:{minutes}" + candidate[match.end():]

    try:
        dt_obj = datetime.fromisoformat(candidate)
    except ValueError:
        return None

    if dt_obj.tzinfo is None:
        dt_obj = dt_obj.replace(tzinfo=timezone.utc)

    timespec = "microseconds" if dt_obj.microsecond else "seconds"
    iso_value = dt_obj.isoformat(timespec=timespec)
    if iso_value.endswith("+00:00"):
        iso_value = iso_value[:-6] + "Z"
    return iso_value
