from data_structures.ingestion import normalize_iso_timestamp


def test_normalize_converts_offset_seconds() -> None:
    value = "2025-05-13T18:47:48.983000-04:04:00"
    assert normalize_iso_timestamp(value) == "2025-05-13T18:47:48.983000-04:04"


def test_normalize_restores_trailing_z_and_separator() -> None:
    value = "2024-01-01 00:00:00Z"
    assert normalize_iso_timestamp(value) == "2024-01-01T00:00:00Z"


def test_normalize_handles_shorthand_offset() -> None:
    value = "2023-12-31T23:59:59+0930"
    assert normalize_iso_timestamp(value) == "2023-12-31T23:59:59+09:30"


def test_normalize_rejects_invalid() -> None:
    assert normalize_iso_timestamp("not-a-timestamp") is None
