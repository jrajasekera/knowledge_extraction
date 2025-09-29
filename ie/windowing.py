from __future__ import annotations

import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Dict, Iterable, Iterator, Sequence


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    cleaned = value.replace("Z", "+00:00")
    return datetime.fromisoformat(cleaned)


@dataclass(slots=True)
class MessageRecord:
    id: str
    channel_id: str
    guild_id: str
    author_id: str
    author_display: str
    content: str
    timestamp: datetime
    message_type: str
    reply_to_id: str | None


@dataclass(slots=True)
class MessageWindow:
    messages: tuple[MessageRecord, ...]
    focus_index: int

    @property
    def focus(self) -> MessageRecord:
        return self.messages[self.focus_index]

    def as_text(self) -> str:
        return "\n\n".join(
            f"[{record.timestamp.isoformat()}] {record.author_display}: {record.content.strip()}"
            for record in self.messages
        )


class WindowBuilder:
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        window_size: int = 4,
        channel_ids: Sequence[str] | None = None,
        guild_ids: Sequence[str] | None = None,
        author_ids: Sequence[str] | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.conn = conn
        self.window_size = window_size
        self.channel_ids = tuple(channel_ids or ())
        self.guild_ids = tuple(guild_ids or ())
        self.author_ids = tuple(author_ids or ())

    def _build_filters(self) -> tuple[str, list[str]]:
        filters: list[str] = []
        params: list[str] = []

        if self.channel_ids:
            placeholders = ",".join("?" for _ in self.channel_ids)
            filters.append(f"m.channel_id IN ({placeholders})")
            params.extend(self.channel_ids)
        if self.guild_ids:
            placeholders = ",".join("?" for _ in self.guild_ids)
            filters.append(f"m.guild_id IN ({placeholders})")
            params.extend(self.guild_ids)
        if self.author_ids:
            placeholders = ",".join("?" for _ in self.author_ids)
            filters.append(f"m.author_id IN ({placeholders})")
            params.extend(self.author_ids)

        where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
        return where_clause, params

    def iter_rows(self) -> Iterator[MessageRecord]:
        where_clause, params = self._build_filters()

        sql = f"""
            SELECT
              m.id,
              m.channel_id,
              m.guild_id,
              m.author_id,
              COALESCE(member.nickname, member.name) AS display_name,
              m.content,
              m.timestamp,
              m.type,
              ref.ref_message_id
            FROM message AS m
            LEFT JOIN member ON member.id = m.author_id
            LEFT JOIN message_reference AS ref ON ref.message_id = m.id
            {where_clause}
            ORDER BY m.channel_id, m.timestamp
        """

        cur = self.conn.execute(sql, params)
        for row in cur:
            timestamp = _parse_timestamp(row[6])
            if timestamp is None:
                continue
            yield MessageRecord(
                id=row[0],
                channel_id=row[1],
                guild_id=row[2],
                author_id=row[3],
                author_display=row[4] or row[3],
                content=row[5] or "",
                timestamp=timestamp,
                message_type=row[7] or "Default",
                reply_to_id=row[8],
            )

    def iter_windows(self) -> Iterator[MessageWindow]:
        buffers: Dict[str, Deque[MessageRecord]] = defaultdict(lambda: deque(maxlen=self.window_size))

        for record in self.iter_rows():
            buffer = buffers[record.channel_id]
            buffer.append(record)
            messages = tuple(buffer)
            focus_index = len(messages) - 1
            yield MessageWindow(messages=messages, focus_index=focus_index)

    def count_rows(self) -> int:
        where_clause, params = self._build_filters()
        sql = f"SELECT COUNT(*) FROM message AS m {where_clause}"
        cur = self.conn.execute(sql, params)
        row = cur.fetchone()
        return int(row[0] or 0)


def iter_message_windows(
    conn: sqlite3.Connection,
    *,
    window_size: int = 4,
    channel_ids: Sequence[str] | None = None,
    guild_ids: Sequence[str] | None = None,
    author_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> Iterable[MessageWindow]:
    builder = WindowBuilder(
        conn,
        window_size=window_size,
        channel_ids=channel_ids,
        guild_ids=guild_ids,
        author_ids=author_ids,
    )
    count = 0
    for window in builder.iter_windows():
        yield window
        count += 1
        if limit is not None and count >= limit:
            break
