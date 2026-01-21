from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from constants import DEFAULT_EMBEDDING_MODEL
from ie.types import FactType

from ..models import FactRecord, SimilarityPair

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = DEFAULT_EMBEDDING_MODEL
    threshold: float = 0.85
    batch_size: int = 32
    device: str | None = None
    max_neighbors: int = 25


class EmbeddingSimilarityDetector:
    def __init__(
        self,
        attribute_index: Mapping[FactType, Sequence[str]],
        config: EmbeddingConfig | None = None,
    ) -> None:
        self._attribute_index = attribute_index
        self._config = config or EmbeddingConfig()
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading sentence-transformer model '%s'", self._config.model_name)
            self._model = SentenceTransformer(
                self._config.model_name,
                trust_remote_code=True,
                device=self._config.device,
            )
        return self._model

    def find_candidate_pairs(self, facts: Sequence[FactRecord]) -> list[SimilarityPair]:
        if len(facts) < 2:
            return []

        model = self._ensure_model()
        texts = [self._format_fact_text(fact) for fact in facts]
        embeddings = model.encode(
            texts,
            batch_size=self._config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        if embeddings.size == 0:
            return []

        normalized = self._normalize(embeddings)
        similarity_matrix = normalized @ normalized.T

        ids = [fact.id for fact in facts]
        pairs: dict[tuple[int, int], float] = {}
        max_neighbors = max(1, self._config.max_neighbors)

        for i in range(len(ids)):
            scores = similarity_matrix[i]
            sorted_indices = np.argsort(-scores)
            neighbors_added = 0
            for idx in sorted_indices:
                if idx == i:
                    continue
                score = float(scores[idx])
                if score < self._config.threshold:
                    break
                a, b = ids[i], ids[idx]
                ordered = (a, b) if a < b else (b, a)
                existing = pairs.get(ordered)
                if existing is None or score > existing:
                    pairs[ordered] = score
                neighbors_added += 1
                if neighbors_added >= max_neighbors:
                    break

        return [SimilarityPair(a, b, score) for (a, b), score in pairs.items()]

    def _format_fact_text(self, fact: FactRecord) -> str:
        parts = [f"Fact type: {fact.type.value}", f"Subject: {fact.subject_id}"]
        if fact.subject_name:
            parts.append(f"Subject name: {fact.subject_name}")
        if fact.object_label:
            parts.append(f"Object label: {fact.object_label}")
        if fact.object_id:
            parts.append(f"Object id: {fact.object_id}")
        attributes = self._attribute_index.get(fact.type)
        if attributes:
            for name in attributes:
                value = fact.attributes.get(name)
                if value is None:
                    continue
                parts.append(f"{name}: {value}")
        else:
            for name, value in sorted(fact.attributes.items()):
                if value is None:
                    continue
                parts.append(f"{name}: {value}")
        if fact.timestamp:
            parts.append(f"Timestamp: {fact.timestamp}")
        return "\n".join(parts)

    def _normalize(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return embeddings / norms
