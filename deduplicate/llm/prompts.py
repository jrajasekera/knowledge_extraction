from __future__ import annotations

import json
from typing import Iterable

from ..models import FactRecord, Partition

SYSTEM_PROMPT = (
    "You are a knowledge graph deduplication specialist. "
    "Given potential duplicate facts, output a consolidated list of canonical facts."
)

MERGE_RULES = """
Rules for merging:
1. Merge facts that represent the same real-world assertion allowing minor textual differences and complementary attributes.
2. Do not merge when entities, timeframes, or core attributes conflict.
3. Confidence rules: identical facts use max confidence; complementary merges use weighted average plus 0.05; cap at 0.95; add 0.1 if merged fact has 5+ evidence messages.
4. Evidence should be the union of supporting message IDs with duplicates removed.
5. Prefer specific attributes, normalize variants, and keep conflicting attributes separate.

Output format:
Return strict JSON with a top-level key canonical_facts. Each fact requires fields type, subject_id, object_label, object_id, object_type, attributes, confidence, evidence, timestamp, merged_from, and merge_reasoning.
""".strip()


def build_messages(partition: Partition, candidate_facts: Iterable[FactRecord]) -> list[dict[str, str]]:
    facts_payload = [
        {
            "fact_id": fact.id,
            "object_label": fact.object_label,
            "object_id": fact.object_id,
            "object_type": fact.object_type,
            "attributes": fact.attributes,
            "confidence": fact.confidence,
            "evidence": fact.evidence,
            "timestamp": fact.timestamp,
        }
        for fact in candidate_facts
    ]
    request = {
        "task": "deduplicate_facts",
        "fact_type": partition.fact_type.value,
        "subject_id": partition.subject_id,
        "subject_name": partition.subject_name,
        "candidate_facts": facts_payload,
    }
    facts_json = json.dumps(request, indent=2, sort_keys=True)
    user_prompt = f"""
FACT TYPE: {partition.fact_type.value}
SUBJECT ID: {partition.subject_id}

Review the candidate facts below and return canonical_facts JSON following the specification.

{MERGE_RULES}

{facts_json}
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]
