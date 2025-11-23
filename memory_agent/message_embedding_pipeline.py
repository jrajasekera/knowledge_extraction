"""Pipeline to generate embeddings for raw Discord messages."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from neo4j import Driver, Session

from .embedding_utils import chunk_iterable
from .embeddings import EmbeddingProvider
from .message_formatter import format_message_for_embedding_text


logger = logging.getLogger(__name__)

VECTOR_INDEX_NAME = "message_embeddings"
VECTOR_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 128


@dataclass(slots=True)
class GraphMessage:
    """Representation of a Discord message stored in Neo4j."""

    message_id: str
    content: str | None
    author_id: str | None
    author_name: str | None
    channel_id: str | None
    channel_name: str | None
    channel_topic: str | None
    channel_type: str | None
    guild_id: str | None
    guild_name: str | None
    timestamp: str | None
    edited_timestamp: str | None
    is_pinned: bool | None
    message_type: str | None
    mentions: list[dict[str, str | None]]
    attachments: list[dict[str, Any]]
    reactions: list[dict[str, Any]]
    thread_id: str | None


def _session_kwargs(database: str | None) -> dict[str, str]:
    return {"database": database} if database else {}


def _stringify_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_native"):
        native = value.to_native()
        if isinstance(native, datetime):
            return native.isoformat()
    return str(value)


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _sanitize_mentions(raw_mentions: Sequence[dict[str, Any]] | None) -> list[dict[str, str | None]]:
    sanitized: list[dict[str, str | None]] = []
    for entry in raw_mentions or []:
        mention_id = entry.get("id")
        if mention_id is None:
            continue
        sanitized.append(
            {
                "id": str(mention_id),
                "name": _stringify(entry.get("name")) or str(mention_id),
            }
        )
    return sanitized


def _sanitize_attachments(raw_attachments: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    attachments: list[dict[str, Any]] = []
    for entry in raw_attachments or []:
        if not entry:
            continue
        attachments.append(
            {
                "id": _stringify(entry.get("id")),
                "url": entry.get("url"),
                "file_name": entry.get("file_name"),
                "size_bytes": entry.get("size_bytes"),
            }
        )
    return attachments


def _sanitize_reactions(raw_reactions: Sequence[dict[str, Any]] | None) -> list[dict[str, Any]]:
    reactions: list[dict[str, Any]] = []
    for entry in raw_reactions or []:
        if not entry:
            continue
        reactions.append(
            {
                "emoji_id": _stringify(entry.get("emoji_id")),
                "emoji_name": entry.get("emoji_name"),
                "count": entry.get("count"),
            }
        )
    return reactions


def _json_dumps(data: Any) -> str:
    return json.dumps(data, sort_keys=True, default=str)


def ensure_indices(session: Session) -> None:
    """Ensure both vector and fulltext indices for message hybrid search exist."""

    # Vector index for semantic search
    logger.info("Ensuring Neo4j vector index %s exists", VECTOR_INDEX_NAME)
    session.run(
        f"""
        CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
        FOR (m:MessageEmbedding)
        ON m.embedding
        OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {VECTOR_DIMENSIONS},
                `vector.similarity_function`: 'cosine'
            }}
        }}
        """
    )

    # Fulltext index for keyword/exact matching
    logger.info("Ensuring Neo4j fulltext index message_fulltext exists")
    session.run(
        """
        CREATE FULLTEXT INDEX message_fulltext IF NOT EXISTS
        FOR (m:MessageEmbedding)
        ON EACH [m.content, m.clean_content, m.author_name, m.channel_name, m.guild_name]
        """
    )


def fetch_graph_messages(session: Session) -> list[GraphMessage]:
    """Load message metadata required for embedding generation."""

    logger.info("Fetching Message nodes from Neo4j")
    result = session.run(
        """
        MATCH (msg:Message)
        OPTIONAL MATCH (author:Person)-[:SENT]->(msg)
        OPTIONAL MATCH (msg)-[:IN_CHANNEL]->(channel:Channel)
        OPTIONAL MATCH (channel)-[:IN_GUILD]->(guild:Guild)
        OPTIONAL MATCH (msg)-[:MENTIONS]->(mention:Person)
        OPTIONAL MATCH (msg)-[:HAS_ATTACHMENT]->(attachment:Attachment)
        OPTIONAL MATCH (msg)-[reaction:REACTED_WITH]->(emoji:Emoji)
        WITH msg, author, channel, guild,
             collect(DISTINCT {id: mention.id, name: coalesce(mention.realName, mention.name, mention.id)}) AS mentions,
             collect(DISTINCT {id: attachment.id, url: attachment.url, file_name: attachment.fileName, size_bytes: attachment.sizeBytes}) AS attachments,
             collect(DISTINCT {emoji_id: emoji.id, emoji_name: emoji.name, count: reaction.count}) AS reactions
        RETURN
            msg.id AS message_id,
            msg.content AS content,
            msg.timestamp AS timestamp,
            msg.edited AS edited_timestamp,
            msg.isPinned AS is_pinned,
            msg.type AS message_type,
            msg.threadId AS thread_id,
            author.id AS author_id,
            coalesce(author.realName, author.name, author.id) AS author_name,
            channel.id AS channel_id,
            channel.name AS channel_name,
            channel.topic AS channel_topic,
            channel.type AS channel_type,
            guild.id AS guild_id,
            guild.name AS guild_name,
            mentions,
            attachments,
            reactions
        """
    )
    messages: list[GraphMessage] = []
    for record in result:
        data = record.data()
        message_id = data.get("message_id")
        if message_id is None:
            continue
        mentions = _sanitize_mentions(data.get("mentions"))
        attachments = _sanitize_attachments(data.get("attachments"))
        reactions = _sanitize_reactions(data.get("reactions"))
        messages.append(
            GraphMessage(
                message_id=str(message_id),
                content=data.get("content"),
                author_id=_stringify(data.get("author_id")),
                author_name=data.get("author_name"),
                channel_id=_stringify(data.get("channel_id")),
                channel_name=data.get("channel_name"),
                channel_topic=data.get("channel_topic"),
                channel_type=data.get("channel_type"),
                guild_id=_stringify(data.get("guild_id")),
                guild_name=data.get("guild_name"),
                timestamp=_stringify_timestamp(data.get("timestamp")),
                edited_timestamp=_stringify_timestamp(data.get("edited_timestamp")),
                is_pinned=bool(data.get("is_pinned")) if data.get("is_pinned") is not None else None,
                message_type=data.get("message_type"),
                mentions=mentions,
                attachments=attachments,
                reactions=reactions,
                thread_id=_stringify(data.get("thread_id")),
            )
        )
    logger.info("Loaded %d messages from Neo4j", len(messages))
    return messages


def _embed_message_batch(
    batch: Sequence[GraphMessage],
    model_name: str,
    device: str,
    cache_dir: str | None,
) -> list[tuple[GraphMessage, str, list[float]]]:
    """Worker function to embed a single batch of messages (runs in subprocess)."""
    # Each worker creates its own provider instance
    provider = EmbeddingProvider(
        model_name=model_name,
        device=device,
        cache_dir=cache_dir,
    )

    results: list[tuple[GraphMessage, str, list[float]]] = []
    texts = []
    valid_messages = []

    for message in batch:
        clean_text = format_message_for_embedding_text(
            author_name=message.author_name,
            content=message.content,
            channel_name=message.channel_name,
            channel_topic=message.channel_topic,
            guild_name=message.guild_name,
            mentions=message.mentions,
            channel_id=message.channel_id,
        )
        if clean_text:
            texts.append(clean_text)
            valid_messages.append((message, clean_text))

    if not texts:
        return []

    embeddings = provider.embed(texts)
    for (message, clean_text), embedding in zip(valid_messages, embeddings, strict=False):
        results.append((message, clean_text, embedding))

    return results


def _build_row(message: GraphMessage, clean_text: str, embedding: list[float]) -> dict[str, Any]:
    attachments_json = _json_dumps(message.attachments)
    reactions_json = _json_dumps(message.reactions)
    mention_ids = [entry["id"] for entry in message.mentions if entry.get("id")]
    mention_names = [entry.get("name") for entry in message.mentions if entry.get("name")]
    return {
        "message_id": message.message_id,
        "content": message.content or "",
        "clean_content": clean_text,
        "author_id": message.author_id,
        "author_name": message.author_name,
        "channel_id": message.channel_id,
        "channel_name": message.channel_name,
        "channel_topic": message.channel_topic,
        "channel_type": message.channel_type,
        "guild_id": message.guild_id,
        "guild_name": message.guild_name,
        "timestamp": message.timestamp,
        "edited_timestamp": message.edited_timestamp,
        "is_pinned": message.is_pinned,
        "message_type": message.message_type,
        "mentions": mention_ids,
        "mention_names": mention_names,
        "attachments": attachments_json,
        "reactions": reactions_json,
        "thread_id": message.thread_id,
        "embedding": embedding,
    }


def generate_message_embeddings(
    messages: Sequence[GraphMessage],
    provider: EmbeddingProvider,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], int]:
    """Generate embedding payloads and report how many messages were skipped.

    Args:
        messages: Sequence of messages to embed
        provider: Embedding provider (config will be extracted for workers)
        batch_size: Number of messages per batch
        workers: Number of parallel workers (1 = sequential, >1 = parallel)
    """
    rows: list[dict[str, Any]] = []
    batches = list(chunk_iterable(messages, batch_size))
    total_messages = len(messages)

    if workers <= 1:
        # Sequential processing (original behavior)
        logger.info("Processing %d batches sequentially", len(batches))
        skipped = 0
        for batch in batches:
            results = _embed_message_batch(
                batch,
                provider.model_name,
                provider.device,
                provider.cache_dir,
            )
            skipped += len(batch) - len(results)
            for message, clean_text, embedding in results:
                rows.append(_build_row(message, clean_text, embedding))
        return rows, skipped
    else:
        # Parallel processing with multiple workers
        logger.info("Processing %d batches with %d workers", len(batches), workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all batches
            futures = {
                executor.submit(
                    _embed_message_batch,
                    batch,
                    provider.model_name,
                    provider.device,
                    provider.cache_dir,
                ): batch_idx
                for batch_idx, batch in enumerate(batches)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    results = future.result()
                    for message, clean_text, embedding in results:
                        rows.append(_build_row(message, clean_text, embedding))
                    completed += 1
                    if completed % 10 == 0:
                        logger.info("Completed %d/%d batches", completed, len(batches))
                except Exception as exc:
                    logger.error("Batch %d failed with error: %s", batch_idx, exc)
                    raise

        skipped = total_messages - len(rows)
        return rows, skipped


def upsert_message_embeddings(
    session: Session,
    rows: Sequence[dict[str, Any]],
    *,
    embedding_model: str,
) -> None:
    if not rows:
        logger.info("No message embeddings to persist")
        return
    logger.info("Persisting %d message embeddings", len(rows))
    session.run(
        """
        UNWIND $rows AS row
        MERGE (m:MessageEmbedding {message_id: row.message_id})
        SET
            m.content = row.content,
            m.clean_content = row.clean_content,
            m.author_id = row.author_id,
            m.author_name = row.author_name,
            m.channel_id = row.channel_id,
            m.channel_name = row.channel_name,
            m.channel_topic = row.channel_topic,
            m.channel_type = row.channel_type,
            m.guild_id = row.guild_id,
            m.guild_name = row.guild_name,
            m.timestamp = row.timestamp,
            m.edited_timestamp = row.edited_timestamp,
            m.is_pinned = row.is_pinned,
            m.message_type = row.message_type,
            m.mentions = row.mentions,
            m.mention_names = row.mention_names,
            m.attachments = row.attachments,
            m.reactions = row.reactions,
            m.thread_id = row.thread_id,
            m.embedding = row.embedding,
            m.embedding_model = $embedding_model,
            m.updated_at = datetime(),
            m.created_at = coalesce(m.created_at, datetime())
        """,
        {"rows": [dict(row) for row in rows], "embedding_model": embedding_model},
    )


def cleanup_orphan_message_embeddings(session: Session) -> int:
    """Delete MessageEmbedding nodes whose source messages vanished."""

    logger.info("Removing orphan MessageEmbedding nodes")
    result = session.run(
        """
        MATCH (m:MessageEmbedding)
        WHERE NOT EXISTS {
            MATCH (:Message {id: m.message_id})
        }
        DETACH DELETE m
        """
    )
    summary = result.consume()
    deleted = summary.counters.nodes_deleted if summary else 0
    logger.info("Deleted %d orphan embeddings", deleted)
    return deleted


def run_message_embedding_pipeline(
    driver: Driver,
    provider: EmbeddingProvider,
    *,
    database: str | None = None,
    cleanup: bool = False,
    dry_run: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    workers: int = 1,
) -> dict[str, Any]:
    """High level orchestration for message embedding generation.

    Args:
        driver: Neo4j driver
        provider: Embedding provider
        database: Optional Neo4j database name
        cleanup: Whether to remove orphan embeddings
        dry_run: Skip database writes
        batch_size: Number of messages per batch
        workers: Number of parallel workers (1 = sequential)
    """
    summary: dict[str, Any] = {
        "messages_scanned": 0,
        "messages_embedded": 0,
        "skipped_empty": 0,
        "embeddings_written": 0,
        "cleaned_orphans": 0,
    }
    with driver.session(**_session_kwargs(database)) as session:
        ensure_indices(session)
        messages = fetch_graph_messages(session)
    summary["messages_scanned"] = len(messages)
    if not messages:
        logger.info("No messages available; skipping embedding generation")
        return summary

    rows, skipped = generate_message_embeddings(messages, provider, batch_size=batch_size, workers=workers)
    summary["skipped_empty"] = skipped
    summary["messages_embedded"] = len(rows)

    if dry_run or not rows:
        logger.info("Dry run enabled; skipping persistence")
        return summary

    with driver.session(**_session_kwargs(database)) as session:
        upsert_message_embeddings(session, rows, embedding_model=provider.model_name)
        summary["embeddings_written"] = len(rows)
        if cleanup:
            summary["cleaned_orphans"] = cleanup_orphan_message_embeddings(session)
    return summary


__all__ = [
    "GraphMessage",
    "fetch_graph_messages",
    "generate_message_embeddings",
    "run_message_embedding_pipeline",
]
