# Knowledge Extraction from Discord — README

> Build detailed person + relationship profiles from Discord exports using a hybrid stack: **SQLite (staging/provenance)** → **Neo4j (graph)** → **LLM/NLP extraction** → **graph analytics**.

---

## Table of Contents

* [What is this?](#what-is-this)
* [Architecture](#architecture)
* [Data Model](#data-model)

  * [SQLite (staging + provenance)](#sqlite-staging--provenance)
  * [Neo4j (graph)](#neo4j-graph)
* [Repository layout](#repository-layout)
* [Getting started](#getting-started)

  * [1) Create the SQLite DB](#1-create-the-sqlite-db)
  * [2) Import Discord JSON → SQLite](#2-import-discord-json--sqlite)
  * [3) Load SQLite → Neo4j](#3-load-sqlite--neo4j)
  * [4) Create a GDS projection](#4-create-a-gds-projection)
* [Query cookbook (Neo4j/Cypher)](#query-cookbook-neo4jcypher)
* [Current focus: Ingesting the data](#current-focus-ingesting-the-data)
* [Roadmap](#roadmap)
* [Troubleshooting](#troubleshooting)
* [Performance tips](#performance-tips)
* [License](#license)

---

## What is this?

This project turns raw **Discord message exports** into a **knowledge graph** that captures:

* Who the people are (profiles, aliases, roles),
* Who interacts with whom (reply/mention/reaction patterns → closeness),
* What topics/events/organizations/places are discussed,
* When things happen (time-aware edges & provenance).

We keep the original data **lossless** in SQLite (with message-level provenance) and project meaningful entities/relationships into **Neo4j** for analytical queries, community detection, and profile generation.

---

## Architecture

```
Discord JSON exports
        │
        ▼
   (importer)
  import_discord_json.py
        │
        ▼
SQLite (staging, provenance)
  - Raw messages, members, roles, reactions, embeds
  - IE runs, facts, evidence (later stage)
        │
        ▼
  loader.py  ──► Neo4j (graph)
                 - Person, Message, Channel, Guild
                 - Mentions, Replies, Reactions, Attachments, Embeds
                 - Derived: INTERACTED_WITH (weights)
                 - Later: TALKS_ABOUT, WORKS_AT, LIVES_IN, CLOSE_TO…
        │
        ▼
 Neo4j GDS analytics + profile generation
 (communities, influence, similarity; LLM narrative)
```

---

## Memory Agent Service

This repository now includes a standalone **memory agent** microservice designed to surface relevant facts from the Neo4j knowledge graph for downstream chat agents. The service exposes a FastAPI application (`memory_agent.app:create_app`) that orchestrates the LangGraph-powered retrieval workflow and a catalog of Neo4j-backed tools.

Start the service locally once your graph is populated:

```bash
uv run uvicorn memory_agent.app:create_app --host 0.0.0.0 --port 8000
```

Key endpoints:

* `POST /api/memory/retrieve` – accept recent Discord messages and return formatted fact strings plus confidence + metadata.
* `POST /api/memory/retrieve/debug` – optional verbose trace (enable via `ENABLE_DEBUG_ENDPOINT=true`).
* `GET /health` – liveness + dependency checks.

Environment variables cover Neo4j credentials, LLM provider/model, embedding settings, iteration limits, and CORS/rate-limiting controls (see `memory_agent/config.py`).

### Semantic Search Embeddings

For the semantic search tool to return useful results, populate the fact embedding index in Neo4j:

```bash
# Ensure Neo4j is running and contains materialised fact relationships (factId on edges)
uv run python scripts/embed_facts.py --cleanup
```

The script will:

- generate text summaries for each fact-bearing relationship,
- embed them with the configured sentence-transformers model (`EMBEDDING_MODEL`),
- upsert `:FactEmbedding` nodes with the vector payload, and
- create / refresh the `fact_embeddings` vector index (cosine similarity, 768 dims).

Re-run the script whenever facts change or after backfills. The `--cleanup` flag removes embeddings for facts that have been deleted from the graph.

---

## Data Model

### SQLite (staging & provenance)

Purpose:

* Lossless storage of Discord exports.
* Fast re-ingest & reproducibility.
* Fact/provenance tables to audit and regenerate graph edges.

Key tables:

* `guild`, `channel`, `member`, `role`, `member_role`
* `message` (+ `message_reference` for replies)
* `attachment`, `embed`, `embed_image`, `inline_emoji`
* `emoji`, `reaction`, `reaction_user`
* `message_mention`
* `import_batch` (source file & counts)
* **For LLM/IE (future)**: `ie_run`, `fact`, `fact_evidence`

> Schema is defined in `schema.sql`. It enforces foreign keys and provides useful indices.

### Neo4j (graph)

Node labels:

* `Person {id, name, nickname, discriminator, avatarUrl, colorHex, isBot}`
* `Guild {id, name, iconUrl}`
* `Channel {id, name, type, category, topic}`
* `Message {id, content, timestamp, edited, isPinned, type}`
* `Role {id, name, colorHex, position}`
* (Later) `Topic`, `Org`, `Place`, `Event`, etc.

Relationships:

* `(:Person)-[:SENT {ts}]->(:Message)`
* `(:Message)-[:IN_CHANNEL]->(:Channel)-[:IN_GUILD]->(:Guild)`
* `(:Message)-[:REPLIES_TO]->(:Message)`
* `(:Message)-[:MENTIONS]->(:Person)`
* `(:Message)-[:REACTED_WITH {name,count}]->(:Emoji)`
* `(:Person)-[:HAS_ROLE]->(:Role)`
* **Derived**: `(:Person)-[:INTERACTED_WITH {weight}]-(:Person)` (from replies & mentions; adjustable)
* (Later) `TALKS_ABOUT`, `CLOSE_TO`, `WORKS_AT`, `LIVES_IN`, `ATTENDS`, etc.

---

## Repository layout

```
.
├─ schema.sql                # SQLite schema (staging + provenance)
├─ import_discord_json.py    # Importer: Discord JSON → SQLite
├─ loader.py                 # Loader: SQLite → Neo4j (nodes/edges + INTERACTED_WITH)
├─ ingest.cql                # Constraints + GDS graph projection
├─ sample.json               # Example Discord export (for quick testing)
└─ README.md                 # This file
```

---

## Getting started

### 0) One-shot pipeline

```bash
python run_pipeline.py --sqlite ./discord.db --schema ./schema.sql --json-dir ./data --neo4j-password 'test'
```

This command applies the schema (idempotently), ingests every `*.json` export under `./data` while skipping files that were already recorded in `import_batch`, and pushes the data into Neo4j at `bolt://localhost:7687`.

By default the pipeline will also:

- Trigger the llama-server IE pass (`GLM-4.5-Air` at `http://localhost:8080/v1/chat/completions`) using window size 4 and store high-confidence facts in SQLite.
- Materialize those facts into Neo4j via `facts_to_graph.py`, creating `Org`, `Place`, `Topic` nodes and relationship edges such as `WORKS_AT`, `LIVES_IN`, `TALKS_ABOUT`, `CLOSE_TO`.

Use `--no-ie` or `--no-fact-graph` to skip either stage, and tweak IE behaviour with flags such as `--ie-window-size`, `--ie-confidence`, `--llama-model`, etc.

### 1) Create the SQLite DB

```bash
sqlite3 discord.db < schema.sql
```

### 2) Import Discord JSON → SQLite

Single file:

```bash
python import_discord_json.py --db ./discord.db --json ./sample.json
```

Directory of exports (recurses for `*.json`):

```bash
python import_discord_json.py --db ./discord.db --json-dir ./exports
```

What the importer handles:

* Members (incl. `color.hex`, `nickname`, `avatarUrl`, `isBot`)
* Roles & member-role links
* Messages (edits, pinned, call-ended timestamp)
* Replies (message references)
* Attachments (file sizes, name)
* Embeds (author, thumbnail, video, images)
* Inline emojis
* Mentions (ensures mentioned members exist)
* Reactions (`emoji`, `reaction`, `reaction_user`)
* `import_batch` counters

### 3) Load SQLite → Neo4j

Start Neo4j (example with Docker):

```bash
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.22
```

Install Python deps and run the loader:

```bash
uv pip install neo4j
python loader.py --sqlite ./discord.db --neo4j bolt://localhost:7687 --user neo4j --password 'test'
```

What the loader does:

* Creates constraints (`Person`, `Guild`, `Channel`, `Message`, `Role`)
* Merges guilds, channels, people, roles
* Loads messages, replies, mentions, reactions, attachments, embeds
* **Materializes `INTERACTED_WITH`** edges from:

  * Replies (weight **3**)
  * Mentions (weight **1**)
  * Symmetrizes A↔B by averaging

> You can tune weights inside `materialize_interactions()` in `loader.py`.

### 4) Create a GDS projection

```bash
# After loader has created INTERACTED_WITH relationships
cat ingest.cql | cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'
```

This creates a GDS in-memory graph `peopleInteractions` with undirected `INTERACTED_WITH` edges (property: `weight`).

---

## Query cookbook (Neo4j/Cypher)

Top interaction partners for a given person:

```cypher
MATCH (:Person {id:$personId})-[:INTERACTED_WITH]->(p:Person)
RETURN p.id AS person, p.name AS name, r.weight AS weight
ORDER BY weight DESC
LIMIT 10;
```

People in the same guild & channel activity:

```cypher
MATCH (p:Person)-[:SENT]->(:Message)-[:IN_CHANNEL]->(c:Channel)-[:IN_GUILD]->(g:Guild {id:$guildId})
RETURN p.name AS person, c.name AS channel, count(*) AS msgs
ORDER BY msgs DESC
LIMIT 20;
```

Reply chains (who replies to whom the most):

```cypher
MATCH (a:Person)-[:SENT]->(m2:Message)-[:REPLIES_TO]->(m1:Message)<-[:SENT]-(b:Person)
RETURN b.name AS from, a.name AS to, count(*) AS replies
ORDER BY replies DESC
LIMIT 20;
```

Mentions received:

```cypher
MATCH (a:Person)-[:SENT]->(m:Message)-[:MENTIONS]->(b:Person)
RETURN b.name AS mentioned, count(*) AS times
ORDER BY times DESC
LIMIT 20;
```

---

## Current focus: Ingesting the data

We’re currently focused on **robust ingestion**:

1. **Import Discord JSON → SQLite**
   `import_discord_json.py` ensures *lossless* capture of messages, replies, mentions, reactions, embeds, attachments, and roles, with batch-level provenance in `import_batch`.

2. **Load SQLite → Neo4j**
   `loader.py` merges core entities/edges and computes **`INTERACTED_WITH`** edges based on replies and mentions so we can immediately run community & influence analyses.

**Once ingestion is solid**, we’ll:

* Add an **IE (information extraction)** pass to populate `fact` and `fact_evidence` (WORKS_AT, LIVES_IN, TALKS_ABOUT, CLOSE_TO, etc.).
* Materialize those facts into Neo4j edges (with provenance & confidence).
* Generate readable **profiles** with citations back to `message_id`s.

---

## Information Extraction (IE)

IE uses a local `llama.cpp`/`llama-server` deployment (OpenAI-compatible API) to convert message windows into structured facts stored in SQLite (`fact` + `fact_evidence`) before being materialized back into Neo4j.

### Initial fact catalogue

| Fact type   | Subject                         | Object / target               | Key attributes (required **bold**)                   | Purpose |
|-------------|----------------------------------|-------------------------------|------------------------------------------------------|---------|
| `WORKS_AT`  | Discord member                   | Organization / company        | **organization**, role, location, start_date, end_date | Org graph + profile context |
| `LIVES_IN`  | Discord member                   | Place / region                | **location**, since                                   | Geo clustering |
| `TALKS_ABOUT` | Discord member                 | Topic / project               | **topic**, sentiment                                  | Interest detection |
| `CLOSE_TO`  | Discord member (Person A)        | Discord member (Person B)     | closeness_basis                                       | Strengthen relationship edges |

Facts below a configurable confidence threshold are discarded before graph materialization. Evidence (message IDs) is recorded in `fact_evidence` for traceability.

> Tip: populate `member.official_name` in SQLite (`UPDATE member SET official_name='John Smith' WHERE id='...';`) so the IE prompts can disambiguate real names from Discord handles.

### Windowing scaffolding

`ie/windowing.py` streams channel-ordered message windows (default size 4) so the IE runner can provide the language model with short conversational context while preserving provenance. You can narrow extraction to specific guilds/channels/authors or cap the number of windows for testing.

### Usage

- Run the pipeline end-to-end (ingest → IE → fact materialization):

  ```bash
  python run_pipeline.py --sqlite ./discord.db --schema ./schema.sql --json-dir ./exports --neo4j-password 'test'
  ```

- Rerun just the IE stage (adjust thresholds/window size as needed):

  ```bash
  python run_ie.py --sqlite ./discord.db --window-size 6 --confidence-threshold 0.6
  ```

- Rematerialize facts into Neo4j without re-ingesting data:

  ```bash
  python facts_to_graph.py --sqlite ./discord.db --password 'test' --min-confidence 0.6
  ```

### Verifying results

- Inspect extracted facts in SQLite:

  ```bash
  sqlite3 ./discord.db "SELECT type, subject_id, object_id, json_extract(attributes, '$.organization'), confidence FROM fact ORDER BY confidence DESC LIMIT 10;"
  ```

- Confirm Neo4j relationships:

  ```cypher
  MATCH (p:Person)-[r:WORKS_AT]->(o:Org)
  RETURN p.name, o.name, r.role, r.confidence, r.evidence
  ORDER BY r.confidence DESC LIMIT 10;
  ```

---

## Roadmap

* [x] SQLite schema (`schema.sql`)
* [x] Importer: Discord JSON → SQLite (`import_discord_json.py`)
* [x] Loader: SQLite → Neo4j (`loader.py`)
* [x] GDS projection (`ingest.cql`)
* [ ] IE pass: windowed extraction to `fact`/`fact_evidence` (Pydantic schema + local LLM)
* [ ] “facts_to_graph.py”: materialize `WORKS_AT`, `LIVES_IN`, `TALKS_ABOUT`, `CLOSE_TO`
* [ ] Profile generator: aggregate facts & graph signals → LLM narrative w/ evidence pointers
* [ ] Streamlit dashboard (people, communities, topics over time)
* [ ] Privacy controls (per-person redaction / “forget me”, sensitive-topic filters)

---

## Troubleshooting

**Neo4j driver error: “Expected exactly one statement per query but got: N”**

* The driver requires a single statement per `session.run()`. `loader.py`’s `materialize_interactions()` already splits the delete and build steps into two runs.

**Download links not working in your environment**

* If you’re missing files, copy from this README or use the setup script pattern in your shell to create/overwrite the files.

**Foreign key errors in SQLite**

* Ensure `PRAGMA foreign_keys = ON;` (the importer sets this).
* Insert order is handled by the importer (guild/channel/members → messages → children).

**Empty graph**

* Confirm the importer actually loaded messages (`SELECT count(*) FROM message;`).
* Ensure `loader.py` ran without errors and check for `Message` and `SENT` relationships in Neo4j Browser.

---

## Performance tips

* Import in **batches** (default fetch sizes are reasonable in `loader.py`).
* Add more indices if you filter on time (`message.timestamp`) heavily.
* For very large exports, consider:

  * Running Neo4j in Docker with increased memory,
  * Temporarily disabling logging,
  * Using periodic `apoc.periodic.iterate` (if APOC is enabled) for massive merges.

---

## License

Choose a license and put it here (e.g., MIT). If this is private/internal, state that clearly.

---

### Quickstart (copy/paste)

```bash
# 1) Create DB
sqlite3 discord.db < schema.sql

# 2) Import JSON (single file or directory)
python import_discord_json.py --db ./discord.db --json ./data/sample.json
# or
python import_discord_json.py --db ./discord.db --json-dir ./exports

# 3) Start Neo4j
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.22

# 4) Load SQLite -> Neo4j
uv pip install neo4j
python loader.py --sqlite ./discord.db --neo4j bolt://localhost:7687 --user neo4j --password 'test'

# 5) GDS projection
cat ingest.cql | cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'
```

That’s the full pipeline. When you’re ready, I can add the **facts_to_graph** step and a first **profile generator** pass.
