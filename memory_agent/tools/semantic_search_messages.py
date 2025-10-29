"""Semantic search tool for raw Discord messages."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from .base import ToolBase, ToolContext, ToolError
from .utils import run_vector_query


logger = logging.getLogger(__name__)

DEFAULT_VECTOR_INDEX = "message_embeddings"
MAX_EXCERPT_LENGTH = 240
DEFAULT_RESULTS_PER_QUERY = 50
DEFAULT_FUSION_METHOD: Literal["rrf", "score_sum", "score_max"] = "rrf"
DEFAULT_MULTI_QUERY_BOOST = 0.0
RRF_K = 60
BLACKLIST_ENV_VAR = "SEMANTIC_MESSAGE_BLACKLIST_PATH"
_DEFAULT_BLACKLIST_PATH = Path(__file__).resolve().parent.parent / "assets" / "semantic_message_blacklist.json"


def _load_blacklisted_content() -> set[str]:
    candidates: list[Path] = []

    env_path = os.getenv(BLACKLIST_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates.append(_DEFAULT_BLACKLIST_PATH)

    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            if candidate == candidates[-1]:
                logger.warning("Default blacklist file missing at %s", candidate)
            else:
                logger.warning("Blacklist override path does not exist: %s", candidate)
            continue
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse blacklist JSON at %s: %s", candidate, exc)
            continue
        except OSError as exc:
            logger.warning("Unable to read blacklist file at %s: %s", candidate, exc)
            continue

        if not isinstance(payload, list):
            logger.warning("Blacklist JSON at %s must be a list, got %s", candidate, type(payload))
            continue

        normalized_entries = {
            str(entry).strip().lower()
            for entry in payload
            if isinstance(entry, (str, int, float)) and str(entry).strip()
        }
        if not normalized_entries:
            logger.warning("Blacklist JSON at %s contained no usable entries", candidate)
            continue

        logger.info("Loaded %d blacklist terms from %s", len(normalized_entries), candidate)
        return normalized_entries

    logger.warning("Falling back to empty blacklist; no valid blacklist source found")
    return set()


BLACKLISTED_CONTENT = _load_blacklisted_content()


@dataclass
class MessageOccurrence:
    """Track per-message observations across multiple semantic queries."""

    properties: dict[str, Any]
    best_score: float
    query_scores: dict[int, float] = field(default_factory=dict)
    query_ranks: dict[int, int] = field(default_factory=dict)

    def add_observation(self, query_idx: int, score: float, rank: int, properties: dict[str, Any]) -> None:
        """Record an observation for this message from a specific query."""

        existing_score = self.query_scores.get(query_idx)
        if existing_score is None or score > existing_score:
            self.query_scores[query_idx] = score
        if query_idx not in self.query_ranks or rank < self.query_ranks[query_idx]:
            self.query_ranks[query_idx] = rank
        if score > self.best_score:
            self.best_score = score
            self.properties = properties


class SemanticSearchMessagesInput(BaseModel):
    """Inputs for semantic_search_messages."""

    queries: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    results_per_query: int = Field(default=DEFAULT_RESULTS_PER_QUERY, ge=1, le=100)
    fusion_method: Literal["rrf", "score_sum", "score_max"] = Field(default=DEFAULT_FUSION_METHOD)
    multi_query_boost: float = Field(default=DEFAULT_MULTI_QUERY_BOOST, ge=0.0, le=1.0)
    channel_ids: list[str] | None = None
    author_ids: list[str] | None = None
    guild_ids: list[str] | None = None
    start_timestamp: datetime | None = None
    end_timestamp: datetime | None = None


class SemanticSearchMessageResult(BaseModel):
    """Output entry for semantic_search_messages."""

    message_id: str
    content: str = ""
    clean_content: str = ""
    author_id: str | None = None
    author_name: str | None = None
    channel_id: str | None = None
    channel_name: str | None = None
    channel_topic: str | None = None
    channel_type: str | None = None
    guild_id: str | None = None
    guild_name: str | None = None
    timestamp: str | None = None
    edited_timestamp: str | None = None
    is_pinned: bool | None = None
    message_type: str | None = None
    thread_id: str | None = None
    permalink: str | None = None
    mentions: list[str] = Field(default_factory=list)
    mention_names: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    reactions: list[dict[str, Any]] = Field(default_factory=list)
    similarity_score: float
    excerpt: str | None = None
    query_scores: dict[int, float] | None = None
    appeared_in_query_count: int | None = None


class SemanticSearchMessagesOutput(BaseModel):
    """Outputs for semantic_search_messages."""

    queries: list[str]
    results: list[SemanticSearchMessageResult] = Field(default_factory=list)


def _parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00") if "Z" in value else value
        try:
            parsed = datetime.fromisoformat(cleaned)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def _build_permalink(guild_id: str | None, channel_id: str | None, message_id: str | None) -> str | None:
    if not (guild_id and channel_id and message_id):
        return None
    return f"https://discord.com/channels/{guild_id}/{channel_id}/{message_id}"


def _load_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(entry) for entry in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [dict(entry) for entry in parsed]
        except json.JSONDecodeError:
            logger.info("Failed to decode JSON payload: %s", value[:80])
    return []


def _safe_excerpt(text: str | None) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if len(stripped) <= MAX_EXCERPT_LENGTH:
        return stripped
    return stripped[: MAX_EXCERPT_LENGTH - 3].rstrip() + "..."


class SemanticSearchMessagesTool(ToolBase[SemanticSearchMessagesInput, SemanticSearchMessagesOutput]):
    """Return the most similar Discord messages for a free-text query."""

    input_model = SemanticSearchMessagesInput
    output_model = SemanticSearchMessagesOutput

    def __init__(self, context: ToolContext, index_name: str = DEFAULT_VECTOR_INDEX) -> None:
        super().__init__(context)
        self.index_name = index_name

    @property
    def embeddings(self) -> EmbeddingProvider:
        model = self.context.embeddings_model
        if model is None:
            raise ToolError("Embedding model not configured")
        if not isinstance(model, EmbeddingProvider):
            msg = f"Unexpected embedding model type: {type(model)}"
            raise ToolError(msg)
        return model

    def run(self, input_data: SemanticSearchMessagesInput) -> SemanticSearchMessagesOutput:
        logger.info(
            "semantic_search_messages called: queries=%r, limit=%d, similarity_threshold=%.2f, index=%s",
            input_data.queries,
            input_data.limit,
            input_data.similarity_threshold,
            self.index_name,
        )

        filters: dict[str, Any] = {}
        # Disable filtering by IDs for now to improve recall
        # if input_data.channel_ids:
        #     filters["channel_id"] = input_data.channel_ids
        # if input_data.author_ids:
        #     filters["author_id"] = input_data.author_ids
        # if input_data.guild_ids:
        #     filters["guild_id"] = input_data.guild_ids

        if filters:
            logger.info("Applying filters: %s", filters)

        start_dt = self._normalize_range(input_data.start_timestamp)
        end_dt = self._normalize_range(input_data.end_timestamp)
        if start_dt or end_dt:
            logger.info("Time range filter: start=%s, end=%s", start_dt, end_dt)

        occurrences: dict[str, MessageOccurrence] = {}
        total_raw_results = 0
        total_filtered_by_threshold = 0
        total_filtered_by_time = 0
        total_missing_node = 0
        total_filtered_by_content = 0
        queries_processed = 0

        for query_idx, query in enumerate(input_data.queries, 1):
            logger.info("Processing query %d/%d: %r", query_idx, len(input_data.queries), query)

            embedding = self.embeddings.embed_single(query)
            if not embedding:
                logger.warning("Failed to generate embedding for query %d: %r", query_idx, query)
                continue

            logger.debug("Generated embedding vector of length %d for query %d", len(embedding), query_idx)

            try:
                rows = run_vector_query(
                    self.context,
                    self.index_name,
                    embedding,
                    input_data.results_per_query,
                    filters,
                    include_evidence=False,
                )
                logger.info("Query %d returned %d raw results from index %s", query_idx, len(rows), self.index_name)
                total_raw_results += len(rows)
                queries_processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Query %d failed: %s", query_idx, exc)
                continue

            filtered_by_threshold = 0
            filtered_by_time = 0
            missing_node = 0
            added_new = 0
            updated_existing = 0
            filtered_by_content = 0

            for rank, row in enumerate(rows, start=1):
                node = row.get("node")
                score = float(row.get("score", 0.0))

                # Apply similarity threshold
                if score < input_data.similarity_threshold:
                    filtered_by_threshold += 1
                    logger.debug(
                        "Query %d: Filtered result with score %.3f (below threshold %.2f)",
                        query_idx,
                        score,
                        input_data.similarity_threshold,
                    )
                    continue

                if not node:
                    missing_node += 1
                    logger.debug("Query %d: Skipping row with missing node", query_idx)
                    continue

                properties = dict(node)
                timestamp_str = properties.get("timestamp")
                timestamp_dt = _parse_timestamp(timestamp_str)

                normalized_content = (properties.get("clean_content") or properties.get("content") or "").strip().lower()
                if normalized_content in BLACKLISTED_CONTENT:
                    filtered_by_content += 1
                    logger.debug(
                        "Query %d: Skipped blacklisted content '%s' for message_id=%s",
                        query_idx,
                        normalized_content,
                        properties.get("message_id"),
                    )
                    continue

                # Apply time range filters
                if start_dt and (timestamp_dt is None or timestamp_dt < start_dt):
                    filtered_by_time += 1
                    logger.debug(
                        "Query %d: Filtered message before start time: %s < %s",
                        query_idx,
                        timestamp_dt,
                        start_dt,
                    )
                    continue
                if end_dt and (timestamp_dt is None or timestamp_dt > end_dt):
                    filtered_by_time += 1
                    logger.debug(
                        "Query %d: Filtered message after end time: %s > %s",
                        query_idx,
                        timestamp_dt,
                        end_dt,
                    )
                    continue

                key = str(properties.get("message_id"))
                occurrence = occurrences.get(key)

                if occurrence is None:
                    occurrence = MessageOccurrence(properties=properties, best_score=score)
                    occurrence.add_observation(query_idx, score, rank, properties)
                    occurrences[key] = occurrence
                    added_new += 1
                    logger.debug(
                        "Query %d: Added new message: message_id=%s, author=%s, score=%.3f, rank=%d",
                        query_idx,
                        key,
                        properties.get("author_name"),
                        score,
                        rank,
                    )
                else:
                    occurrence.add_observation(query_idx, score, rank, properties)
                    updated_existing += 1
                    logger.debug(
                        "Query %d: Updated existing message: message_id=%s, author=%s, score=%.3f, rank=%d",
                        query_idx,
                        key,
                        properties.get("author_name"),
                        score,
                        rank,
                    )

            total_filtered_by_threshold += filtered_by_threshold
            total_filtered_by_time += filtered_by_time
            total_missing_node += missing_node
            total_filtered_by_content += filtered_by_content
            logger.info(
                "Query %d summary: raw=%d, filtered_threshold=%d, filtered_time=%d, missing_node=%d, filtered_content=%d, added_new=%d, updated_existing=%d",
                query_idx,
                len(rows),
                filtered_by_threshold,
                filtered_by_time,
                missing_node,
                filtered_by_content,
                added_new,
                updated_existing,
            )

            logger.info(
                "Query %d summary: raw=%d, filtered_threshold=%d, filtered_time=%d, missing_node=%d, added_new=%d, updated_existing=%d",
                query_idx,
                len(rows),
                filtered_by_threshold,
                filtered_by_time,
                missing_node,
                added_new,
                updated_existing,
            )

        logger.info("Total unique messages before fusion: %d", len(occurrences))

        results_with_scores: list[SemanticSearchMessageResult] = []
        for message_id, occurrence in occurrences.items():
            combined_score = self._calculate_combined_score(
                occurrence,
                input_data.fusion_method,
                input_data.multi_query_boost,
            )
            result = self._build_result(
                occurrence.properties,
                combined_score,
                query_scores=dict(occurrence.query_scores),
            )
            results_with_scores.append(result)

        messages_in_multiple_queries = sum(1 for occ in occurrences.values() if len(occ.query_scores) > 1)
        avg_queries_per_message = (
            sum(len(occ.query_scores) for occ in occurrences.values()) / len(occurrences)
            if occurrences
            else 0.0
        )

        logger.info(
            "Message fusion summary: total_unique_messages=%d, messages_in_multiple_queries=%d, avg_queries_per_message=%.2f",
            len(occurrences),
            messages_in_multiple_queries,
            avg_queries_per_message,
        )

        def _sort_key(result: SemanticSearchMessageResult) -> tuple[float, float, str]:
            timestamp_dt = _parse_timestamp(result.timestamp)
            timestamp_value = timestamp_dt.timestamp() if timestamp_dt else float("-inf")
            return (result.similarity_score, timestamp_value, result.message_id)

        ordered = sorted(results_with_scores, key=_sort_key, reverse=True)

        limited: list[SemanticSearchMessageResult] = []
        seen_dedupe_keys: set[tuple[str, str]] = set()
        duplicates_filtered = 0

        for result in ordered:
            if self._try_append_result(result, limited, seen_dedupe_keys, enforce_unique=True):
                if len(limited) >= input_data.limit:
                    break
            else:
                duplicates_filtered += 1

        if len(limited) < input_data.limit:
            logger.debug(
                "Not enough unique messages to satisfy limit; attempting fallback with remaining candidates",
            )
            for result in ordered:
                if result in limited:
                    continue
                if self._try_append_result(result, limited, seen_dedupe_keys, enforce_unique=True):
                    if len(limited) >= input_data.limit:
                        break

        duplicates_reintroduced = 0
        if len(limited) < input_data.limit:
            logger.debug(
                "Still short after unique fallback; allowing duplicates to fill remaining slots",
            )
            for result in ordered:
                if result in limited:
                    continue
                if self._try_append_result(result, limited, seen_dedupe_keys, enforce_unique=False):
                    duplicates_reintroduced += 1
                if len(limited) >= input_data.limit:
                    break

        logger.info(
            "Deduplicated messages by author/content: filtered_duplicates=%d, reintroduced_duplicates=%d, final_results=%d",
            duplicates_filtered,
            duplicates_reintroduced,
            len(limited),
        )

        for result in limited[:5]:
            occurrence = occurrences[result.message_id]
            logger.debug(
                "Top result after fusion: message_id=%s, combined_score=%.3f, appeared_in=%d, query_scores=%s",
                result.message_id,
                result.similarity_score,
                len(occurrence.query_scores),
                occurrence.query_scores,
            )

        logger.info(
            "semantic_search_messages completed: queries=%d, queries_processed=%d, total_raw_results=%d, "
            "total_filtered_threshold=%d, total_filtered_time=%d, total_filtered_content=%d, total_missing_node=%d, unique_messages=%d, final_results=%d",
            len(input_data.queries),
            queries_processed,
            total_raw_results,
            total_filtered_by_threshold,
            total_filtered_by_time,
            total_filtered_by_content,
            total_missing_node,
            len(ordered),
            len(limited),
        )

        return SemanticSearchMessagesOutput(queries=input_data.queries, results=limited)

    @staticmethod
    def _normalize_range(value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _calculate_combined_score(occurrence: MessageOccurrence, fusion_method: str, multi_query_boost: float) -> float:
        if not occurrence.query_scores:
            return occurrence.best_score

        if fusion_method == "rrf":
            return sum(1.0 / (RRF_K + rank) for rank in occurrence.query_ranks.values())

        if fusion_method == "score_sum":
            return sum(occurrence.query_scores.values())

        if fusion_method == "score_max":
            max_score = max(occurrence.query_scores.values())
            query_count = len(occurrence.query_scores)
            return max_score * (1.0 + multi_query_boost * (query_count - 1))

        msg = f"Unsupported fusion method: {fusion_method}"
        raise ToolError(msg)

    def _build_result(
        self,
        properties: dict[str, Any],
        score: float,
        *,
        query_scores: dict[int, float] | None = None,
    ) -> SemanticSearchMessageResult:
        attachments = _load_json_list(properties.get("attachments"))
        reactions = _load_json_list(properties.get("reactions"))
        mentions = [str(value) for value in properties.get("mentions", []) if value is not None]
        mention_names = [str(value) for value in properties.get("mention_names", []) if value is not None]
        clean_content = properties.get("clean_content") or ""
        content = properties.get("content") or ""
        excerpt = _safe_excerpt(clean_content or content)
        permalink = _build_permalink(properties.get("guild_id"), properties.get("channel_id"), properties.get("message_id"))
        appeared_in_query_count = len(query_scores) if query_scores else None
        return SemanticSearchMessageResult(
            message_id=str(properties.get("message_id")),
            content=content,
            clean_content=clean_content,
            author_id=properties.get("author_id"),
            author_name=properties.get("author_name"),
            channel_id=properties.get("channel_id"),
            channel_name=properties.get("channel_name"),
            channel_topic=properties.get("channel_topic"),
            channel_type=properties.get("channel_type"),
            guild_id=properties.get("guild_id"),
            guild_name=properties.get("guild_name"),
            timestamp=properties.get("timestamp"),
            edited_timestamp=properties.get("edited_timestamp"),
            is_pinned=properties.get("is_pinned"),
            message_type=properties.get("message_type"),
            thread_id=properties.get("thread_id"),
            permalink=permalink,
            mentions=mentions,
            mention_names=mention_names,
            attachments=attachments,
            reactions=reactions,
            similarity_score=score,
            excerpt=excerpt,
            query_scores=query_scores,
            appeared_in_query_count=appeared_in_query_count,
        )

    @staticmethod
    def _dedupe_key(result: SemanticSearchMessageResult) -> tuple[str, str] | None:
        author = (result.author_name or "").strip().lower()
        content = (result.clean_content or result.content or "").strip().lower()
        if not author and not content:
            return None
        return (author, content)

    def _try_append_result(
        self,
        result: SemanticSearchMessageResult,
        bucket: list[SemanticSearchMessageResult],
        seen_dedupe_keys: set[tuple[str, str]],
        *,
        enforce_unique: bool,
    ) -> bool:
        dedupe_key = self._dedupe_key(result)
        if enforce_unique and dedupe_key and dedupe_key in seen_dedupe_keys:
            return False
        if dedupe_key:
            seen_dedupe_keys.add(dedupe_key)
        bucket.append(result)
        return True


__all__ = [
    "SemanticSearchMessagesInput",
    "SemanticSearchMessagesOutput",
    "SemanticSearchMessagesTool",
]
