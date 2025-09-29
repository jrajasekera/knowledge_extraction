#!/usr/bin/env python3
"""
import_discord_json.py
Ingest Discord JSON exports into the provided SQLite schema (schema.sql).

Usage examples:
  python import_discord_json.py --db ./discord.db --json ./sample.json
  python import_discord_json.py --db ./discord.db --json-dir ./exports
"""

import argparse
import sqlite3
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import orjson
import datetime

def iso(dt: Optional[str]) -> Optional[str]:
    return dt if dt else None

def as_bool(v: Any) -> int:
    return 1 if bool(v) else 0

def ensure(conn: sqlite3.Connection, sql: str, params: Tuple):
    conn.execute(sql, params)

def insert_import_batch(conn: sqlite3.Connection, source_path: str, exported_at: str, message_count: int) -> int:
    cur = conn.execute(
        "INSERT INTO import_batch (source_path, exported_at_iso, reported_msg_count, loaded_msg_count) VALUES (?,?,?,0)",
        (source_path, exported_at, int(message_count)),
    )
    return int(cur.lastrowid)

def finalize_import_batch(conn: sqlite3.Connection, batch_id: int, loaded_count: int):
    conn.execute("UPDATE import_batch SET loaded_msg_count = ? WHERE id = ?", (int(loaded_count), int(batch_id)))

def upsert_guild(conn: sqlite3.Connection, g: Dict[str, Any]):
    ensure(conn, "INSERT OR IGNORE INTO guild (id, name, icon_url) VALUES (?,?,?)",
           (str(g.get("id")), g.get("name",""), g.get("iconUrl")))

def upsert_channel(conn: sqlite3.Connection, c: Dict[str, Any], guild_id: str):
    ensure(conn, "INSERT OR IGNORE INTO channel (id, guild_id, type, category_id, category, name, topic) VALUES (?,?,?,?,?,?,?)",
           (str(c.get("id")), str(guild_id), c.get("type",""), c.get("categoryId"), c.get("category"), c.get("name",""), c.get("topic")))

def upsert_role(conn: sqlite3.Connection, role: Dict[str, Any], guild_id: str):
    ensure(conn, "INSERT OR IGNORE INTO role (id, guild_id, name, color_hex, position) VALUES (?,?,?,?,?)",
           (str(role.get("id")), str(guild_id), role.get("name",""), role.get("color"), role.get("position") or 0))

def upsert_member(conn: sqlite3.Connection, m: Dict[str, Any]):
    ensure(conn, "INSERT OR IGNORE INTO member (id, name, discriminator, nickname, color_hex, is_bot, avatar_url) VALUES (?,?,?,?,?,?,?)",
           (str(m.get("id")), m.get("name",""), m.get("discriminator","0000"), m.get("nickname"),
            (m.get("color") or {}).get("hex") if isinstance(m.get("color"), dict) else m.get("color"),
            as_bool(m.get("isBot", False)), m.get("avatarUrl")))

def link_member_roles(conn: sqlite3.Connection, member_id: str, roles: List[Dict[str, Any]]):
    for r in roles or []:
        ensure(conn, "INSERT OR IGNORE INTO member_role (member_id, role_id) VALUES (?,?)",
               (str(member_id), str(r.get("id"))))

def insert_message(conn: sqlite3.Connection, msg: Dict[str, Any], guild_id: str, channel_id: str, batch_id: int):
    ensure(conn, textwrap.dedent("""
        INSERT OR IGNORE INTO message
          (id, channel_id, guild_id, author_id, type, timestamp, timestamp_edited, call_ended_at, is_pinned, content, import_batch_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """),
    (str(msg.get("id")), str(channel_id), str(guild_id), str(msg["author"]["id"]),
     msg.get("type","Default"), iso(msg.get("timestamp")), iso(msg.get("timestampEdited")), iso(msg.get("callEndedTimestamp")),
     as_bool(msg.get("isPinned", False)), msg.get("content",""), int(batch_id))
    )

def insert_reference(conn: sqlite3.Connection, msg_id: str, ref: Dict[str, Any]):
    ensure(conn, "INSERT OR IGNORE INTO message_reference (message_id, ref_message_id, ref_channel_id, ref_guild_id) VALUES (?,?,?,?)",
           (str(msg_id), str(ref.get("messageId")), str(ref.get("channelId")), str(ref.get("guildId")) if ref.get("guildId") else None))

def insert_attachment(conn: sqlite3.Connection, msg_id: str, att: Dict[str, Any]):
    ensure(conn, "INSERT OR IGNORE INTO attachment (id, message_id, url, file_name, file_size_bytes) VALUES (?,?,?,?,?)",
           (str(att.get("id") or f"{msg_id}:{att.get('url')}"), str(msg_id), att.get("url",""),
            att.get("fileName") or att.get("filename") or "", int(att.get("fileSizeBytes") or att.get("sizeBytes") or att.get("size") or 0)))

def insert_embed(conn: sqlite3.Connection, msg_id: str, e: Dict[str, Any], idx: int):
    embed_id = int(f"{hash(str(msg_id)) & 0x7fffffff}{idx}") if e.get("id") is None else int(e.get("id"))
    ensure(conn, textwrap.dedent("""
        INSERT OR IGNORE INTO embed
          (id, message_id, title, url, timestamp, description, color_hex,
           author_name, author_url, thumbnail_url, thumbnail_w, thumbnail_h,
           video_url, video_w, video_h)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """),
    (embed_id, str(msg_id), e.get("title"), e.get("url"), iso(e.get("timestamp")), e.get("description"),
     (e.get("color") or {}).get("hex") if isinstance(e.get("color"), dict) else e.get("color"),
     (e.get("author") or {}).get("name"), (e.get("author") or {}).get("url"),
     (e.get("thumbnail") or {}).get("url"), (e.get("thumbnail") or {}).get("width"), (e.get("thumbnail") or {}).get("height"),
     (e.get("video") or {}).get("url"), (e.get("video") or {}).get("width"), (e.get("video") or {}).get("height"))
    )
    # embed images array
    for img in (e.get("images") or []):
        ensure(conn, "INSERT OR IGNORE INTO embed_image (embed_id, url, width, height) VALUES (?,?,?,?)",
               (embed_id, img.get("url",""), img.get("width"), img.get("height")))

def insert_inline_emoji(conn: sqlite3.Connection, msg_id: str, ie: Dict[str, Any]):
    ensure(conn, "INSERT INTO inline_emoji (message_id, emoji_id, name, code, is_animated, image_url) VALUES (?,?,?,?,?,?)",
           (str(msg_id), ie.get("id"), ie.get("name",""), ie.get("code") or "", as_bool(ie.get("isAnimated", False)), ie.get("imageUrl")))

def upsert_emoji(conn: sqlite3.Connection, em: Dict[str, Any]):
    # composite PK (id, name). id may be null, so use None in tuple.
    ensure(conn, "INSERT OR IGNORE INTO emoji (id, name, code, is_animated, image_url) VALUES (?,?,?,?,?)",
           (em.get("id"), em.get("name",""), em.get("code") or "", as_bool(em.get("isAnimated", False)), em.get("imageUrl")))

def insert_reaction(conn: sqlite3.Connection, msg_id: str, r: Dict[str, Any]):
    # Ensure emoji exists
    upsert_emoji(conn, r.get("emoji", {}))
    emoji_id = r.get("emoji", {}).get("id")
    emoji_name = r.get("emoji", {}).get("name", "")
    ensure(conn, "INSERT OR IGNORE INTO reaction (message_id, emoji_id, emoji_name, count) VALUES (?,?,?,?)",
           (str(msg_id), emoji_id, emoji_name, int(r.get("count", 0))))

def insert_reaction_users(conn: sqlite3.Connection, msg_id: str, r: Dict[str, Any]):
    emoji_name = r.get("emoji", {}).get("name", "")
    for u in r.get("users", []):
        upsert_member(conn, u)
        ensure(conn, "INSERT OR IGNORE INTO reaction_user (message_id, emoji_name, user_id) VALUES (?,?,?)",
               (str(msg_id), emoji_name, str(u.get("id"))))

def insert_mentions(conn: sqlite3.Connection, msg_id: str, mentions: List[Dict[str, Any]]):
    for m in mentions or []:
        upsert_member(conn, m)
        ensure(conn, "INSERT OR IGNORE INTO message_mention (message_id, member_id) VALUES (?,?)",
               (str(msg_id), str(m.get("id"))))

def process_file(conn: sqlite3.Connection, path: Path) -> int:
    raw = path.read_bytes()
    data = orjson.loads(raw)

    guild = data.get("guild", {})
    channel = data.get("channel", {})
    exported_at = data.get("exportedAt") or datetime.datetime.utcnow().isoformat()
    message_count = int(data.get("messageCount") or len(data.get("messages", [])))

    batch_id = insert_import_batch(conn, str(path), exported_at, message_count)

    # upsert guild/channel first
    upsert_guild(conn, guild)
    upsert_channel(conn, channel, str(guild.get("id")))

    loaded = 0
    for msg in data.get("messages", []):
        # authors & roles
        author = msg.get("author", {})
        upsert_member(conn, author)
        for role in (author.get("roles") or []):
            upsert_role(conn, role, str(guild.get("id")))
        link_member_roles(conn, str(author.get("id")), author.get("roles") or [])

        # the message itself
        insert_message(conn, msg, str(guild.get("id")), str(channel.get("id")), batch_id)

        # reference (reply)
        if msg.get("reference"):
            insert_reference(conn, str(msg.get("id")), msg["reference"])

        # attachments
        for att in msg.get("attachments", []):
            insert_attachment(conn, str(msg.get("id")), att)

        # embeds
        for idx, e in enumerate(msg.get("embeds", [])):
            insert_embed(conn, str(msg.get("id")), e, idx)

        # inline emojis
        for ie in msg.get("inlineEmojis", []):
            insert_inline_emoji(conn, str(msg.get("id")), ie)

        # reactions (and users)
        for r in msg.get("reactions", []):
            insert_reaction(conn, str(msg.get("id")), r)
            insert_reaction_users(conn, str(msg.get("id")), r)

        # mentions
        insert_mentions(conn, str(msg.get("id")), msg.get("mentions", []))

        loaded += 1

    finalize_import_batch(conn, batch_id, loaded)
    return loaded

def iter_json_files(json: Optional[Path], json_dir: Optional[Path]) -> Iterable[Path]:
    if json and json.exists():
        yield json
    if json_dir and json_dir.exists():
        for p in sorted(json_dir.rglob("*.json")):
            yield p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="Path to SQLite DB (schema.sql must be applied)")
    ap.add_argument("--json", type=Path, help="Path to a single Discord export .json")
    ap.add_argument("--json-dir", type=Path, help="Directory to crawl for *.json exports")
    args = ap.parse_args()

    if not args.json and not args.json_dir:
        ap.error("Provide --json or --json-dir")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA foreign_keys = ON;")
    total = 0
    try:
        with conn:
            for p in iter_json_files(args.json, args.json_dir):
                print(f"Ingesting {p} ...")
                total += process_file(conn, p)
        print(f"Done. Loaded {total} messages.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()
