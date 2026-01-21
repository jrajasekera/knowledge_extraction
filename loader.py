# loader.py
# Stream data from SQLite into Neo4j with batched MERGEs.
# Usage:
#   uv pip install neo4j orjson
#   python loader.py --sqlite ./discord.db --neo4j bolt://localhost:7687 --user neo4j --password test --batch 1000

import argparse
import sqlite3
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase

from db_utils import get_sqlite_connection


def batched(cursor: sqlite3.Cursor, fetchsize: int):
    while True:
        rows = cursor.fetchmany(fetchsize)
        if not rows:
            break
        yield rows


def run_cypher(tx, query: str, params: dict[str, Any] | None = None):
    tx.run(query, params or {})


CONSTRAINTS = [
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person)   REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT guild_id  IF NOT EXISTS FOR (n:Guild)    REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT channel_id IF NOT EXISTS FOR (n:Channel) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT message_id IF NOT EXISTS FOR (n:Message) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT role_id    IF NOT EXISTS FOR (n:Role)    REQUIRE n.id IS UNIQUE",
]

MERGE_GUILD = """
MERGE (g:Guild {id:$id})
SET g.name=$name, g.iconUrl=$icon_url
"""

MERGE_CHANNEL = """
MERGE (c:Channel {id:$id})
SET c.name=$name, c.type=$type, c.category=$category, c.topic=$topic
WITH c
MATCH (g:Guild {id:$guild_id})
MERGE (c)-[:IN_GUILD]->(g)
"""

MERGE_MEMBER = """
MERGE (p:Person {id:$id})
SET p.name=$name, p.nickname=$nickname, p.discriminator=$discriminator,
    p.avatarUrl=$avatar_url, p.colorHex=$color_hex, p.isBot=$is_bot,
    p.realName=$official_name
"""

MERGE_ROLE = """
MERGE (r:Role {id:$id})
SET r.name=$name, r.colorHex=$color_hex, r.position=$position
WITH r
MATCH (g:Guild {id:$guild_id})
MERGE (r)-[:IN_GUILD]->(g)
"""

MERGE_MEMBER_ROLE = """
MATCH (p:Person {id:$member_id}), (r:Role {id:$role_id})
MERGE (p)-[:HAS_ROLE]->(r)
"""

MERGE_MESSAGE = """
MERGE (m:Message {id:$id})
SET m.content=$content, m.timestamp=$timestamp, m.edited=$edited, m.isPinned=$is_pinned, m.type=$type
WITH m
MATCH (p:Person {id:$author_id}), (c:Channel {id:$channel_id})
MERGE (p)-[:SENT {ts:$timestamp}]->(m)
MERGE (m)-[:IN_CHANNEL]->(c)
"""

MERGE_REPLY = """
MATCH (m:Message {id:$msg_id}), (parent:Message {id:$ref_msg_id})
MERGE (m)-[:REPLIES_TO]->(parent)
"""

MERGE_MENTION = """
MATCH (m:Message {id:$msg_id}), (p:Person {id:$person_id})
MERGE (m)-[:MENTIONS]->(p)
"""

MERGE_EMOJI = """
MERGE (e:Emoji {id: coalesce($emoji_id,$emoji_name), name:$emoji_name})
SET e.code=coalesce($code,''), e.isAnimated=coalesce($is_animated,false), e.imageUrl=$image_url
"""

MERGE_REACTION = """
MATCH (m:Message {id:$msg_id})
MERGE (m)-[r:REACTED_WITH {name:$emoji_name}]->(:Emoji {id:coalesce($emoji_id,$emoji_name), name:$emoji_name})
SET r.count = coalesce(r.count,0) + $count
"""

MERGE_ATTACHMENT = """
MATCH (m:Message {id:$msg_id})
MERGE (a:Attachment {id:$att_id})
SET a.url=$url, a.fileName=$file_name, a.sizeBytes=$size
MERGE (m)-[:HAS_ATTACHMENT]->(a)
"""

MERGE_EMBED = """
MATCH (m:Message {id:$msg_id})
MERGE (e:Embed {id:$embed_id})
SET e.title=$title, e.url=$url, e.description=$description, e.timestamp=$timestamp, e.color=$color_hex,
    e.authorName=$author_name, e.authorUrl=$author_url, e.thumbnailUrl=$thumbnail_url,
    e.thumbnailW=$thumbnail_w, e.thumbnailH=$thumbnail_h, e.videoUrl=$video_url, e.videoW=$video_w, e.videoH=$video_h
MERGE (m)-[:HAS_EMBED]->(e)
"""


def load_guilds(cur, driver):
    cur.execute("SELECT id, name, icon_url FROM guild")
    with driver.session() as sess:
        for rows in batched(cur, 1000):

            def txfun(tx, rows=rows):
                for gid, name, icon in rows:
                    run_cypher(tx, MERGE_GUILD, {"id": gid, "name": name, "icon_url": icon})

            sess.execute_write(txfun)


def load_channels(cur, driver):
    cur.execute("SELECT id, guild_id, type, category, name, topic FROM channel")
    with driver.session() as sess:
        for rows in batched(cur, 1000):

            def txfun(tx, rows=rows):
                for cid, gid, typ, cat, name, topic in rows:
                    run_cypher(
                        tx,
                        MERGE_CHANNEL,
                        {
                            "id": cid,
                            "guild_id": gid,
                            "type": typ,
                            "category": cat,
                            "name": name,
                            "topic": topic,
                        },
                    )

            sess.execute_write(txfun)


def load_members(cur, driver):
    cur.execute(
        "SELECT id, name, discriminator, nickname, official_name, color_hex, is_bot, avatar_url FROM member"
    )
    with driver.session() as sess:
        for rows in batched(cur, 1000):

            def txfun(tx, rows=rows):
                for pid, name, disc, nick, official_name, color, is_bot, avatar in rows:
                    run_cypher(
                        tx,
                        MERGE_MEMBER,
                        {
                            "id": pid,
                            "name": name,
                            "discriminator": disc,
                            "nickname": nick,
                            "official_name": official_name,
                            "color_hex": color,
                            "is_bot": bool(is_bot),
                            "avatar_url": avatar,
                        },
                    )

            sess.execute_write(txfun)


def load_roles(cur, driver):
    cur.execute("SELECT id, guild_id, name, color_hex, position FROM role")
    with driver.session() as sess:
        for rows in batched(cur, 1000):

            def txfun(tx, rows=rows):
                for rid, gid, name, color, pos in rows:
                    run_cypher(
                        tx,
                        MERGE_ROLE,
                        {
                            "id": rid,
                            "guild_id": gid,
                            "name": name,
                            "color_hex": color,
                            "position": pos,
                        },
                    )

            sess.execute_write(txfun)


def load_member_roles(cur, driver):
    cur.execute("SELECT member_id, role_id FROM member_role")
    with driver.session() as sess:
        for rows in batched(cur, 2000):

            def txfun(tx, rows=rows):
                for mid, rid in rows:
                    run_cypher(tx, MERGE_MEMBER_ROLE, {"member_id": mid, "role_id": rid})

            sess.execute_write(txfun)


def load_messages(cur, driver):
    cur.execute(
        "SELECT id, channel_id, guild_id, author_id, type, timestamp, timestamp_edited, is_pinned, content FROM message"
    )
    with driver.session() as sess:
        for rows in batched(cur, 1000):

            def txfun(tx, rows=rows):
                for mid, cid, _gid, aid, typ, ts, edited, pinned, content in rows:
                    run_cypher(
                        tx,
                        MERGE_MESSAGE,
                        {
                            "id": mid,
                            "channel_id": cid,
                            "author_id": aid,
                            "type": typ,
                            "timestamp": ts,
                            "edited": edited,
                            "is_pinned": bool(pinned),
                            "content": content,
                        },
                    )

            sess.execute_write(txfun)


def load_replies(cur, driver):
    cur.execute("SELECT message_id, ref_message_id FROM message_reference")
    with driver.session() as sess:
        for rows in batched(cur, 2000):

            def txfun(tx, rows=rows):
                for mid, ref_mid in rows:
                    run_cypher(tx, MERGE_REPLY, {"msg_id": mid, "ref_msg_id": ref_mid})

            sess.execute_write(txfun)


def load_mentions(cur, driver):
    cur.execute("SELECT message_id, member_id FROM message_mention")
    with driver.session() as sess:
        for rows in batched(cur, 5000):

            def txfun(tx, rows=rows):
                for mid, pid in rows:
                    run_cypher(tx, MERGE_MENTION, {"msg_id": mid, "person_id": pid})

            sess.execute_write(txfun)


def load_reactions(cur, driver):
    # ensure emojis first (distinct by id+name pairs)
    cur.execute("SELECT DISTINCT emoji_id, emoji_name FROM reaction")
    with driver.session() as sess:
        for rows in batched(cur, 2000):

            def txfun(tx, rows=rows):
                for eid, ename in rows:
                    run_cypher(
                        tx,
                        MERGE_EMOJI,
                        {
                            "emoji_id": eid,
                            "emoji_name": ename,
                            "code": "",
                            "is_animated": False,
                            "image_url": None,
                        },
                    )

            sess.execute_write(txfun)

    cur.execute("SELECT message_id, emoji_id, emoji_name, count FROM reaction")
    with driver.session() as sess:
        for rows in batched(cur, 5000):

            def txfun(tx, rows=rows):
                for mid, eid, ename, count in rows:
                    run_cypher(
                        tx,
                        MERGE_REACTION,
                        {"msg_id": mid, "emoji_id": eid, "emoji_name": ename, "count": count},
                    )

            sess.execute_write(txfun)


def load_attachments(cur, driver):
    cur.execute("SELECT id, message_id, url, file_name, file_size_bytes FROM attachment")
    with driver.session() as sess:
        for rows in batched(cur, 2000):

            def txfun(tx, rows=rows):
                for att_id, mid, url, fname, size in rows:
                    run_cypher(
                        tx,
                        MERGE_ATTACHMENT,
                        {
                            "att_id": att_id,
                            "msg_id": mid,
                            "url": url,
                            "file_name": fname,
                            "size": size,
                        },
                    )

            sess.execute_write(txfun)


def load_embeds(cur, driver):
    cur.execute(
        "SELECT id, message_id, title, url, timestamp, description, color_hex, author_name, author_url, thumbnail_url, thumbnail_w, thumbnail_h, video_url, video_w, video_h FROM embed"
    )
    with driver.session() as sess:
        for rows in batched(cur, 2000):

            def txfun(tx, rows=rows):
                for (
                    eid,
                    mid,
                    title,
                    url,
                    ts,
                    desc,
                    color,
                    aname,
                    aurl,
                    th_url,
                    th_w,
                    th_h,
                    vurl,
                    vw,
                    vh,
                ) in rows:
                    run_cypher(
                        tx,
                        MERGE_EMBED,
                        {
                            "embed_id": eid,
                            "msg_id": mid,
                            "title": title,
                            "url": url,
                            "timestamp": ts,
                            "description": desc,
                            "color_hex": color,
                            "author_name": aname,
                            "author_url": aurl,
                            "thumbnail_url": th_url,
                            "thumbnail_w": th_w,
                            "thumbnail_h": th_h,
                            "video_url": vurl,
                            "video_w": vw,
                            "video_h": vh,
                        },
                    )

            sess.execute_write(txfun)


def materialize_interactions(driver):
    # Replies weight 3; Mentions weight 1; then symmetrize.
    delete_stmt = "MATCH ()-[r:INTERACTED_WITH]-() DELETE r"
    build_stmt = """
    // Accumulate weights in two subqueries
    CALL {
      // Replies contribute weight 3
      MATCH (a:Person)-[:SENT]->(m2:Message)-[:REPLIES_TO]->(m1:Message)<-[:SENT]-(b:Person)
      WITH a,b,count(*)*3 AS w
      MERGE (a)-[r:INTERACTED_WITH]->(b)
      SET r.weight = coalesce(r.weight,0) + w
    }
    CALL {
      // Mentions contribute weight 1
      MATCH (a:Person)-[:SENT]->(m:Message)-[:MENTIONS]->(b:Person)
      WITH a,b,count(*) AS w
      MERGE (a)-[r:INTERACTED_WITH]->(b)
      SET r.weight = coalesce(r.weight,0) + w
    }
    // Symmetrize in a separate scope so variables are defined
    WITH 1 AS _
    MATCH (a:Person)-[r:INTERACTED_WITH]->(b:Person)
    OPTIONAL MATCH (b)-[rb:INTERACTED_WITH]->(a)
    SET r.weight = (coalesce(r.weight,0) + coalesce(rb.weight,0)) / 2.0
    """
    with driver.session() as sess:
        sess.run(delete_stmt)
        sess.run(build_stmt)


def load_into_neo4j(
    sqlite_path: str | Path,
    *,
    neo4j_uri: str = "bolt://localhost:7687",
    user: str = "neo4j",
    password: str,
) -> None:
    conn = get_sqlite_connection(sqlite_path, timeout=60.0, read_only=True)
    cur = conn.cursor()
    driver = GraphDatabase.driver(neo4j_uri, auth=(user, password))

    try:
        # Ensure constraints once
        with driver.session() as sess:
            for c in CONSTRAINTS:
                sess.run(c)

        print("Loading guilds...")
        load_guilds(cur, driver)
        print("Loading channels...")
        load_channels(cur, driver)
        print("Loading members...")
        load_members(cur, driver)
        print("Loading roles...")
        load_roles(cur, driver)
        print("Linking member roles...")
        load_member_roles(cur, driver)
        print("Loading messages...")
        load_messages(cur, driver)
        print("Loading replies...")
        load_replies(cur, driver)
        print("Loading mentions...")
        load_mentions(cur, driver)
        print("Loading reactions...")
        load_reactions(cur, driver)
        print("Loading attachments...")
        load_attachments(cur, driver)
        print("Loading embeds...")
        load_embeds(cur, driver)

        print("Materializing INTERACTED_WITH edges...")
        materialize_interactions(driver)

        print("Done. Consider running ingest.cql to build the GDS projection.")
    finally:
        driver.close()
        conn.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sqlite", required=True, help="Path to SQLite db with schema.sql applied")
    ap.add_argument("--neo4j", default="bolt://localhost:7687")
    ap.add_argument("--user", default="neo4j")
    ap.add_argument("--password", required=True)
    args = ap.parse_args()

    load_into_neo4j(
        args.sqlite,
        neo4j_uri=args.neo4j,
        user=args.user,
        password=args.password,
    )


if __name__ == "__main__":
    main()
