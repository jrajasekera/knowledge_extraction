from __future__ import annotations

import json
import logging
from collections.abc import Mapping

from data_structures.ingestion import normalize_iso_timestamp

from ..models import CanonicalFact, FactRecord, Partition

logger = logging.getLogger(__name__)


class CanonicalFactsParser:
    """Parse and validate canonical fact JSON returned by the LLM."""

    def parse(
        self,
        payload: str,
        *,
        partition: Partition,
        facts_by_id: Mapping[int, FactRecord],
    ) -> list[CanonicalFact]:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response was not valid JSON") from exc

        canonical_items = data.get("canonical_facts")
        if canonical_items is None:
            raise ValueError("LLM response missing 'canonical_facts'")
        if not isinstance(canonical_items, list):
            raise ValueError("'canonical_facts' must be a list")

        results: list[CanonicalFact] = []
        for item in canonical_items:
            if not isinstance(item, dict):
                raise ValueError("canonical fact entries must be objects")
            results.append(self._parse_item(item, partition=partition, facts_by_id=facts_by_id))
        return results

    def _parse_item(
        self,
        item: dict[str, object],
        *,
        partition: Partition,
        facts_by_id: Mapping[int, FactRecord],
    ) -> CanonicalFact:
        fact_type = item.get("type") or partition.fact_type.value
        if fact_type != partition.fact_type.value:
            raise ValueError("canonical fact type must match partition type")

        subject_id = item.get("subject_id") or partition.subject_id
        if subject_id != partition.subject_id:
            raise ValueError("canonical fact subject_id must match partition subject")

        merged_from = item.get("merged_from")
        if not isinstance(merged_from, list) or not merged_from:
            raise ValueError("canonical facts require non-empty merged_from list")

        merged_ids: list[int] = []
        for value in merged_from:
            if not isinstance(value, int):
                raise ValueError("merged_from entries must be integers")
            if value not in facts_by_id:
                raise ValueError(f"merged_from fact {value} not part of candidate group")
            merged_ids.append(value)

        source_facts = [facts_by_id[fact_id] for fact_id in merged_ids]
        evidence_union = self._collect_evidence(item.get("evidence"), source_facts)
        attributes = self._collect_attributes(item.get("attributes"))
        confidence = self._normalize_confidence(item.get("confidence"))
        timestamp = self._normalize_timestamp(item.get("timestamp"), source_facts)
        merge_reasoning = self._extract_reasoning(item.get("merge_reasoning"))
        object_label = self._resolve_value(
            item.get("object_label"), [fact.object_label for fact in source_facts]
        )
        object_id = self._resolve_value(
            item.get("object_id"), [fact.object_id for fact in source_facts]
        )
        object_type = self._resolve_value(
            item.get("object_type"), [fact.object_type for fact in source_facts]
        )

        return CanonicalFact(
            type=partition.fact_type,
            subject_id=partition.subject_id,
            object_label=object_label,
            object_id=object_id,
            object_type=object_type,
            attributes=attributes,
            confidence=confidence,
            evidence=evidence_union,
            timestamp=timestamp,
            merged_from=merged_ids,
            merge_reasoning=merge_reasoning,
        )

    def _collect_evidence(
        self,
        value: object,
        source_facts: list[FactRecord],
    ) -> list[str]:
        allowed = {message_id for fact in source_facts for message_id in fact.evidence}

        if value is None:
            evidence = []
        elif isinstance(value, list):
            if not all(isinstance(item, (str, int)) for item in value):
                raise ValueError("evidence entries must be strings or ints")
            evidence = [str(item) for item in value]
        else:
            raise ValueError("evidence must be an array")

        if evidence:
            filtered = [message_id for message_id in evidence if message_id in allowed]
            dropped = [message_id for message_id in evidence if message_id not in allowed]
            if dropped:
                logger.warning(
                    "Dropped %d invalid evidence IDs from LLM response.",
                    len(dropped),
                )
            if filtered:
                return list(dict.fromkeys(filtered))

        if not evidence:
            evidence = sorted(allowed)
        return list(dict.fromkeys(evidence))

    def _collect_attributes(self, value: object) -> dict[str, object]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("attributes must be an object")
        return value

    def _normalize_confidence(self, value: object) -> float:
        if value is None:
            return 0.0
        try:
            confidence = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return confidence

    def _normalize_timestamp(
        self,
        value: object,
        source_facts: list[FactRecord],
    ) -> str:
        if isinstance(value, str) and value.strip():
            sanitized = normalize_iso_timestamp(value)
            if sanitized is not None:
                return sanitized
        timestamps = [fact.timestamp for fact in source_facts if fact.timestamp]
        if not timestamps:
            return ""
        for ts in sorted(timestamps):
            sanitized = normalize_iso_timestamp(ts)
            if sanitized is not None:
                return sanitized
        return ""

    def _extract_reasoning(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("merge_reasoning must be a non-empty string")
        return value.strip()

    def _resolve_value(
        self,
        provided: object,
        fallbacks: list[str | None],
    ) -> str | None:
        if isinstance(provided, str) and provided.strip():
            return provided.strip()
        for option in fallbacks:
            if option:
                return option
        return None
