from __future__ import annotations

import logging
import time
from contextlib import suppress

import httpx

from ie.client import LlamaServerClient, LlamaServerConfig

from ..models import CandidateGroup, CanonicalFact, FactRecord
from .parser import CanonicalFactsParser
from .prompts import build_messages

logger = logging.getLogger(__name__)


class DeduplicationLLMClient:
    def __init__(
        self,
        config: LlamaServerConfig,
        *,
        parser: CanonicalFactsParser | None = None,
        max_retries: int = 3,
        backoff_base: float = 2.0,
    ) -> None:
        self._config = config
        self._client = LlamaServerClient(config)
        self._parser = parser or CanonicalFactsParser()
        self._max_retries = max_retries
        self._backoff_base = backoff_base

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DeduplicationLLMClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def generate_canonical_facts(
        self,
        group: CandidateGroup,
        facts_lookup: dict[int, FactRecord],
    ) -> list[CanonicalFact]:
        candidate_ids = sorted(group.fact_ids)
        candidate_facts = [facts_lookup[fact_id] for fact_id in candidate_ids]
        messages = build_messages(group.partition, candidate_facts)

        attempt = 0
        while True:
            try:
                response = self._client.complete(messages)
            except httpx.HTTPError as exc:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff_base**attempt
                logger.warning("LLM HTTP error: %s. Retrying in %.1fs", exc, delay)
                time.sleep(delay)
                continue

            if response is None:
                raise ValueError("LLM returned empty response content")

            try:
                return self._parser.parse(
                    response,
                    partition=group.partition,
                    facts_by_id=facts_lookup,
                )
            except ValueError as exc:
                attempt += 1
                if attempt >= self._max_retries:
                    raise
                delay = self._backoff_base**attempt
                logger.warning("LLM parse failure: %s. Retrying in %.1fs", exc, delay)
                time.sleep(delay)

    def __del__(self) -> None:
        with suppress(Exception):  # pragma: no cover - best effort
            self.close()
