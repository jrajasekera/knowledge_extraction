"""Semantic search tool for raw Discord messages."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from ..embeddings import EmbeddingProvider
from .base import ToolBase, ToolContext, ToolError
from .utils import run_vector_query


logger = logging.getLogger(__name__)

DEFAULT_VECTOR_INDEX = "message_embeddings"
MAX_EXCERPT_LENGTH = 240


class SemanticSearchMessagesInput(BaseModel):
    """Inputs for semantic_search_messages."""

    queries: list[str] = Field(min_length=1, max_length=5)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
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
            logger.debug("Failed to decode JSON payload: %s", value[:80])
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
        if input_data.channel_ids:
            filters["channel_id"] = input_data.channel_ids
        if input_data.author_ids:
            filters["author_id"] = input_data.author_ids
        if input_data.guild_ids:
            filters["guild_id"] = input_data.guild_ids

        if filters:
            logger.info("Applying filters: %s", filters)

        start_dt = self._normalize_range(input_data.start_timestamp)
        end_dt = self._normalize_range(input_data.end_timestamp)
        if start_dt or end_dt:
            logger.info("Time range filter: start=%s, end=%s", start_dt, end_dt)

        dedup: dict[str, SemanticSearchMessageResult] = {}
        total_raw_results = 0
        total_filtered_by_threshold = 0
        total_filtered_by_time = 0
        total_missing_node = 0
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
                    input_data.limit,
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

            for row in rows:
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

                result = self._build_result(properties, score)
                key = result.message_id
                existing = dedup.get(key)

                if existing is None or result.similarity_score > existing.similarity_score:
                    if existing is not None:
                        updated_existing += 1
                        logger.debug(
                            "Query %d: Updated existing message with higher score %.3f (was %.3f): message_id=%s, author=%s",
                            query_idx,
                            score,
                            existing.similarity_score,
                            result.message_id,
                            result.author_name,
                        )
                    else:
                        added_new += 1
                        logger.debug(
                            "Query %d: Added new message: message_id=%s, author=%s, score=%.3f",
                            query_idx,
                            result.message_id,
                            result.author_name,
                            score,
                        )
                    dedup[key] = result
                else:
                    logger.debug(
                        "Query %d: Skipped duplicate with lower score %.3f (existing %.3f): message_id=%s",
                        query_idx,
                        score,
                        existing.similarity_score,
                        result.message_id,
                    )

            total_filtered_by_threshold += filtered_by_threshold
            total_filtered_by_time += filtered_by_time
            total_missing_node += missing_node

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

        logger.info("Total unique messages before limit: %d", len(dedup))

        ordered = sorted(dedup.values(), key=lambda item: item.similarity_score, reverse=True)
        limited = ordered[: input_data.limit]

        logger.info(
            "semantic_search_messages completed: queries=%d, queries_processed=%d, total_raw_results=%d, "
            "total_filtered_threshold=%d, total_filtered_time=%d, total_missing_node=%d, unique_messages=%d, final_results=%d",
            len(input_data.queries),
            queries_processed,
            total_raw_results,
            total_filtered_by_threshold,
            total_filtered_by_time,
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

    def _build_result(self, properties: dict[str, Any], score: float) -> SemanticSearchMessageResult:
        attachments = _load_json_list(properties.get("attachments"))
        reactions = _load_json_list(properties.get("reactions"))
        mentions = [str(value) for value in properties.get("mentions", []) if value is not None]
        mention_names = [str(value) for value in properties.get("mention_names", []) if value is not None]
        clean_content = properties.get("clean_content") or ""
        content = properties.get("content") or ""
        excerpt = _safe_excerpt(clean_content or content)
        permalink = _build_permalink(properties.get("guild_id"), properties.get("channel_id"), properties.get("message_id"))
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
        )


__all__ = [
    "SemanticSearchMessagesInput",
    "SemanticSearchMessagesOutput",
    "SemanticSearchMessagesTool",
]
