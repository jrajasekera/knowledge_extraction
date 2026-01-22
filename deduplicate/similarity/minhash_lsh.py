from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from ie.types import FactType

from ..models import FactRecord, SimilarityPair

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MinHashConfig:
    threshold: float = 0.7
    num_perm: int = 128
    ngram_size: int = 3


class MinHashLSHDetector:
    def __init__(
        self,
        attribute_index: Mapping[FactType, Sequence[str]],
        config: MinHashConfig | None = None,
    ) -> None:
        self._attribute_index = attribute_index
        self._config = config or MinHashConfig()

    def find_candidate_pairs(self, facts: Sequence[FactRecord]) -> list[SimilarityPair]:
        if len(facts) < 2:
            return []

        lsh = MinHashLSH(
            threshold=self._config.threshold,
            num_perm=self._config.num_perm,
        )
        signatures: dict[int, MinHash] = {}

        for fact in facts:
            text = self._extract_text(fact)
            if not text:
                continue
            signature = self._generate_signature(text)
            key = str(fact.id)
            signatures[fact.id] = signature
            lsh.insert(key, signature)

        if not signatures:
            return []

        pairs: dict[tuple[int, int], float] = {}
        for fact_id, signature in signatures.items():
            for candidate_key in lsh.query(signature):
                candidate_id = int(candidate_key)
                if candidate_id == fact_id:
                    continue
                # Use min/max instead of sorted to get tuple[int, int] type
                ordered = (min(fact_id, candidate_id), max(fact_id, candidate_id))
                if ordered in pairs:
                    continue
                other_signature = signatures.get(candidate_id)
                if not other_signature:
                    continue
                score = signature.jaccard(other_signature)
                pairs[ordered] = score

        return [SimilarityPair(a, b, score) for (a, b), score in pairs.items()]

    def _extract_text(self, fact: FactRecord) -> str:
        attributes = self._attribute_index.get(fact.type)
        parts: list[str] = []
        if fact.object_label:
            label = fact.object_label.strip()
            if label:
                parts.append(label)
        if attributes:
            for name in attributes:
                value = fact.attributes.get(name)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    parts.append(text)
        else:
            for _name, value in sorted(fact.attributes.items()):
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    parts.append(text)
        return "|".join(parts)

    def _generate_signature(self, text: str) -> MinHash:
        mh = MinHash(num_perm=self._config.num_perm)
        for token in self._tokenize(text):
            mh.update(token.encode("utf-8"))
        return mh

    def _tokenize(self, text: str) -> list[str]:
        n = self._config.ngram_size
        cleaned = text.lower()
        if len(cleaned) <= n:
            return [cleaned]
        return [cleaned[i : i + n] for i in range(len(cleaned) - n + 1)]
