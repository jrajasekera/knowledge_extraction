"""Embedding utilities for semantic search."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class EmbeddingProvider:
    """Thin wrapper around a sentence-transformers model."""

    model_name: str
    device: str = "cpu"
    cache_dir: Path | None = None
    normalize: bool = True

    _model: "SentenceTransformer | None" = None  # type: ignore[name-defined]

    @property
    def model(self) -> "SentenceTransformer":  # type: ignore[name-defined]
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            kwargs = {"device": self.device}
            if self.cache_dir is not None:
                kwargs["cache_folder"] = str(self.cache_dir)
            self._model = SentenceTransformer(self.model_name, **kwargs)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed one or more texts."""
        if not texts:
            return []
        embeddings = self.model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        )
        return [
            emb.tolist() if hasattr(emb, "tolist") else [float(x) for x in emb]
            for emb in embeddings
        ]

    def embed_single(self, text: str) -> list[float]:
        """Embed a single string."""
        results = self.embed([text])
        return results[0] if results else []
