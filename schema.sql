-- schema.sql
-- SQLite schema for Discord ingest, provenance, and IE facts

PRAGMA foreign_keys = ON;

-- 0) Meta
CREATE TABLE IF NOT EXISTS import_batch (
  id                  INTEGER PRIMARY KEY,
  source_path         TEXT NOT NULL,
  exported_at_iso     TEXT NOT NULL,
  reported_msg_count  INTEGER NOT NULL,
  loaded_msg_count    INTEGER NOT NULL,
  created_at          TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 1) Discord structure
CREATE TABLE IF NOT EXISTS guild (
  id              TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  icon_url        TEXT
);

CREATE TABLE IF NOT EXISTS channel (
  id              TEXT PRIMARY KEY,
  guild_id        TEXT NOT NULL REFERENCES guild(id) ON DELETE CASCADE,
  type            TEXT NOT NULL,
  category_id     TEXT,
  category        TEXT,
  name            TEXT NOT NULL,
  topic           TEXT
);

CREATE TABLE IF NOT EXISTS role (
  id              TEXT PRIMARY KEY,
  guild_id        TEXT NOT NULL REFERENCES guild(id) ON DELETE CASCADE,
  name            TEXT NOT NULL,
  color_hex       TEXT,
  position        INTEGER
);

CREATE TABLE IF NOT EXISTS member (
  id              TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  discriminator   TEXT NOT NULL,
  nickname        TEXT,
  official_name   TEXT,
  color_hex       TEXT,
  is_bot          INTEGER NOT NULL CHECK (is_bot IN (0,1)),
  avatar_url      TEXT
);

CREATE TABLE IF NOT EXISTS member_alias (
  member_id   TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  alias       TEXT NOT NULL,
  alias_type  TEXT CHECK (alias_type IS NULL OR alias_type IN ('nickname', 'first_name', 'variation')),
  PRIMARY KEY (member_id, alias)
);

CREATE TABLE IF NOT EXISTS member_role (
  member_id       TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  role_id         TEXT NOT NULL REFERENCES role(id) ON DELETE CASCADE,
  PRIMARY KEY (member_id, role_id)
);

-- 2) Messages & children
CREATE TABLE IF NOT EXISTS message (
  id                  TEXT PRIMARY KEY,
  channel_id          TEXT NOT NULL REFERENCES channel(id) ON DELETE CASCADE,
  guild_id            TEXT NOT NULL REFERENCES guild(id) ON DELETE CASCADE,
  author_id           TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  type                TEXT NOT NULL,
  timestamp           TEXT NOT NULL,
  timestamp_edited    TEXT,
  call_ended_at       TEXT,
  is_pinned           INTEGER NOT NULL CHECK (is_pinned IN (0,1)),
  content             TEXT NOT NULL,
  import_batch_id     INTEGER NOT NULL REFERENCES import_batch(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_message_channel_ts ON message(channel_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_message_author_ts  ON message(author_id, timestamp);

CREATE TABLE IF NOT EXISTS message_reference (
  message_id      TEXT PRIMARY KEY REFERENCES message(id) ON DELETE CASCADE,
  ref_message_id  TEXT NOT NULL,
  ref_channel_id  TEXT NOT NULL,
  ref_guild_id    TEXT
);

CREATE TABLE IF NOT EXISTS attachment (
  id              TEXT PRIMARY KEY,
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  url             TEXT NOT NULL,
  file_name       TEXT NOT NULL,
  file_size_bytes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embed (
  id              INTEGER PRIMARY KEY,
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  title           TEXT,
  url             TEXT,
  timestamp       TEXT,
  description     TEXT,
  color_hex       TEXT,
  author_name     TEXT,
  author_url      TEXT,
  thumbnail_url   TEXT,
  thumbnail_w     INTEGER,
  thumbnail_h     INTEGER,
  video_url       TEXT,
  video_w         INTEGER,
  video_h         INTEGER
);

CREATE TABLE IF NOT EXISTS embed_image (
  id              INTEGER PRIMARY KEY,
  embed_id        INTEGER NOT NULL REFERENCES embed(id) ON DELETE CASCADE,
  url             TEXT NOT NULL,
  width           INTEGER,
  height          INTEGER
);

CREATE TABLE IF NOT EXISTS inline_emoji (
  id              INTEGER PRIMARY KEY,
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  emoji_id        TEXT,
  name            TEXT NOT NULL,
  code            TEXT NOT NULL,
  is_animated     INTEGER NOT NULL CHECK (is_animated IN (0,1)),
  image_url       TEXT
);

-- 3) Mentions & reactions
CREATE TABLE IF NOT EXISTS message_mention (
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  member_id       TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  PRIMARY KEY (message_id, member_id)
);

CREATE TABLE IF NOT EXISTS emoji (
  id              TEXT,
  name            TEXT NOT NULL,
  code            TEXT NOT NULL,
  is_animated     INTEGER NOT NULL CHECK (is_animated IN (0,1)),
  image_url       TEXT,
  PRIMARY KEY (id, name)
);

CREATE TABLE IF NOT EXISTS reaction (
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  emoji_id        TEXT,
  emoji_name      TEXT NOT NULL,
  count           INTEGER NOT NULL,
  PRIMARY KEY (message_id, emoji_name)
);

CREATE TABLE IF NOT EXISTS reaction_user (
  message_id      TEXT NOT NULL,
  emoji_name      TEXT NOT NULL,
  user_id         TEXT NOT NULL REFERENCES member(id) ON DELETE CASCADE,
  PRIMARY KEY (message_id, emoji_name, user_id),
  FOREIGN KEY (message_id, emoji_name) REFERENCES reaction(message_id, emoji_name) ON DELETE CASCADE
);

-- 4) Information Extraction (facts) & runs
CREATE TABLE IF NOT EXISTS ie_run (
  id              INTEGER PRIMARY KEY,
  started_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  model_name      TEXT NOT NULL,
  model_params    TEXT,
  window_hint     TEXT
);

CREATE TABLE IF NOT EXISTS fact (
  id              INTEGER PRIMARY KEY,
  ie_run_id       INTEGER NOT NULL REFERENCES ie_run(id) ON DELETE CASCADE,
  type            TEXT NOT NULL,
  subject_id      TEXT NOT NULL,
  object_id       TEXT,
  object_type     TEXT,
  attributes      TEXT NOT NULL,
  ts              TEXT,
  confidence      REAL NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
  graph_synced_at TEXT
);

CREATE TABLE IF NOT EXISTS fact_evidence (
  fact_id         INTEGER NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
  message_id      TEXT NOT NULL REFERENCES message(id) ON DELETE CASCADE,
  PRIMARY KEY (fact_id, message_id)
);

-- 5) Fact deduplication audit and progress tracking
CREATE TABLE IF NOT EXISTS fact_deduplication_audit (
  id                   INTEGER PRIMARY KEY,
  canonical_fact_id    INTEGER NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
  original_fact_ids    TEXT NOT NULL,
  merge_reasoning      TEXT,
  similarity_scores    TEXT,
  processed_at         TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dedup_audit_canonical
  ON fact_deduplication_audit(canonical_fact_id);

CREATE TABLE IF NOT EXISTS deduplication_run (
  id                          INTEGER PRIMARY KEY,
  started_at                  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at                TEXT,
  status                      TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'paused')),
  total_partitions            INTEGER,
  processed_partitions        INTEGER DEFAULT 0,
  facts_processed             INTEGER DEFAULT 0,
  facts_merged                INTEGER DEFAULT 0,
  candidate_groups_processed  INTEGER DEFAULT 0,
  details                     TEXT
);

CREATE TABLE IF NOT EXISTS deduplication_partition_progress (
  run_id       INTEGER NOT NULL REFERENCES deduplication_run(id) ON DELETE CASCADE,
  fact_type    TEXT NOT NULL,
  subject_id   TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
  updated_at   TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, fact_type, subject_id)
);

-- 6) Pipeline orchestration & progress tracking
CREATE TABLE IF NOT EXISTS pipeline_run (
  id             INTEGER PRIMARY KEY,
  started_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at   TEXT,
  status         TEXT NOT NULL CHECK (status IN ('running','paused','completed','failed','cancelled')),
  current_stage  TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_stage_state (
  run_id      INTEGER NOT NULL REFERENCES pipeline_run(id) ON DELETE CASCADE,
  stage       TEXT NOT NULL,
  status      TEXT NOT NULL CHECK (status IN ('pending','in_progress','completed')),
  details     TEXT,
  updated_at  TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, stage)
);

CREATE TABLE IF NOT EXISTS ie_progress (
  run_id            INTEGER PRIMARY KEY REFERENCES ie_run(id) ON DELETE CASCADE,
  processed_windows INTEGER NOT NULL DEFAULT 0,
  total_windows     INTEGER NOT NULL,
  started_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at        TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ie_window_progress (
  run_id           INTEGER NOT NULL REFERENCES ie_run(id) ON DELETE CASCADE,
  focus_message_id TEXT NOT NULL,
  processed_at     TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (run_id, focus_message_id)
);

CREATE INDEX IF NOT EXISTS idx_ie_window_progress_run_id
  ON ie_window_progress(run_id);

-- 5) Derived view
CREATE VIEW IF NOT EXISTS message_interactions AS
SELECT
  m1.author_id AS a,
  m2.author_id AS b,
  m2.id        AS reply_id,
  m1.id        AS parent_id,
  m2.timestamp AS reply_ts
FROM message m2
JOIN message_reference r ON r.message_id = m2.id
JOIN message m1 ON m1.id = r.ref_message_id;
