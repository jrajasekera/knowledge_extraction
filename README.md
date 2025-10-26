# Knowledge Extraction from Discord

> Turn raw Discord exports into a living people + relationship knowledge graph backed by SQLite, Neo4j, and local LLM-powered information extraction.

---

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Data Model](#data-model)
- [Repository Layout](#repository-layout)
- [Environment & Setup](#environment--setup)
- [Running the Pipeline](#running-the-pipeline)
- [Stage Entry Points](#stage-entry-points)
- [Information Extraction & Fact Catalogue](#information-extraction--fact-catalogue)
- [Prompt Scaffolding](#prompt-scaffolding)
- [Memory Agent Service](#memory-agent-service)
- [Embedding Jobs](#embedding-jobs)
- [Query Cookbook](#query-cookbook)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Roadmap](#roadmap)
- [Data & Safety](#data--safety)
- [Quickstart](#quickstart)
- [License](#license)

---

## Overview
This repo ingests exported Discord JSON dumps, stages them losslessly in SQLite, builds an attributed Neo4j graph, runs local LLM information extraction (IE) to capture higher-order signals (work history, education, preferences, relationships, events), and surfaces everything through a retrieval-ready memory service. The `run_pipeline.py` entry point orchestrates ingest → load → IE → fact materialization so you can push new exports end-to-end or resume partially completed runs without reprocessing prior work.

---

## Architecture
```
Discord JSON exports
        │
        ▼
import_discord_json.py        (lossless staging + provenance)
        │
        ▼
SQLite (messages, roles, IE runs, facts, evidence)
        │
        ▼
loader.py → Neo4j core graph (Person/Message/Channel/Guild/Role + INTERACTED_WITH)
        │
        ▼
run_ie.py / pipeline "ie" stage (windowed llama-server IE, fact tables)
        │
        ▼
facts_to_graph.py (Org/Place/Topic/Event/Skill/etc. nodes + rich relationships)
        │
        ▼
Neo4j GDS analytics, LangGraph memory agent, embeddings, downstream apps
```
`run_pipeline.py` tracks stage status in `pipeline_run`/`pipeline_stage_state` so you can `--resume` or `--restart` long-running IE or fact-materialization jobs. Each stage records structured details (counts, remaining windows) for easy monitoring.

---

## Data Model
### SQLite (staging + IE provenance)
Purpose:
- Lossless storage of Discord exports (`guild`, `channel`, `member`, `role`, `message`, `reaction`, attachments, embeds, mentions, emoji).
- Provenance & IE control tables (`import_batch`, `pipeline_run`, `pipeline_stage_state`, `ie_progress`).
- Structured IE outputs (`fact`, `fact_evidence`, confidence + attributes, `graph_synced_at`).

The schema lives in `schema.sql`; helper utilities in `data_structures/ingestion/` provide typed accessors when loading rows.

### Neo4j (graph + embeddings)
Node labels:
- `Person`, `ExternalPerson`, `Guild`, `Channel`, `Message`, `Role`.
- IE-derived types: `Org`, `Place`, `Topic`, `Project`, `Skill`, `Event`, `Preference`, `Recommendation`.
- Vector helpers: `FactEmbedding`, `MessageEmbedding` (populated via the scripts under `scripts/`).

Relationships:
- Core: `SENT`, `IN_CHANNEL`, `IN_GUILD`, `HAS_ROLE`, `MENTIONS`, `REPLIES_TO`, `REACTED_WITH`, `HAS_ATTACHMENT`, `HAS_EMBED`.
- Derived interactions: `INTERACTED_WITH {weight}` (mentions + replies).
- Fact edges: `WORKS_AT`, `STUDIED_AT`, `HAS_SKILL`, `WORKING_ON`, `RELATED_TO`, `ATTENDED_EVENT`, `LIVES_IN`, `TALKS_ABOUT`, `CLOSE_TO`, plus preference/plan edges such as `PREFERS`, `DISLIKES`, `ENJOYS`, `RECOMMENDS`, `AVOIDS`, `PLANS_TO`, `CARES_ABOUT`, `CURIOUS_ABOUT`, `BELIEVES`, `REMEMBERS`, `EXPERIENCED`, `WITNESSED`.
- Vector similarity support via `fact_embeddings` and `message_embeddings` indexes (cosine, 768 dims by default).

`ingest.cql` defines the GDS projection and constraints for quick graph analytics.

---

## Repository Layout
```
.
├─ run_pipeline.py            # Orchestrates ingest → load → IE → fact graph with resume support
├─ import_discord_json.py     # Discord JSON → SQLite staging
├─ loader.py                  # SQLite → Neo4j core graph + INTERACTED_WITH edges
├─ run_ie.py                  # Standalone IE runner (window controls, confidence gating)
├─ facts_to_graph.py          # Materialize IE facts into Neo4j
├─ deduplicate/               # Utils for cleaning duplicated facts prior to graph writes
├─ ie/
│  ├─ advanced_prompts.py     # Prompt scaffolding helpers
│  ├─ prompt_assets.json      # Few-shot + framing assets editable without touching code
│  ├─ client.py / runner.py   # llama-server adapter + execution loop
│  └─ windowing.py            # Channel-ordered streaming windows
├─ data_structures/ingestion  # Typed domain objects shared by importer/loader/IE
├─ memory_agent/              # FastAPI service + LangGraph retrieval workflow
├─ scripts/
│  ├─ embed_facts.py          # Populate/refresh :FactEmbedding nodes & vector index
│  ├─ embed_messages.py       # Populate/refresh :MessageEmbedding nodes & index
│  └─ graph_snapshot.py       # Export Neo4j snapshots for regression checks
├─ docs/                      # Design notes (llama-server setup, memory agent plans, profiles)
├─ tests/                     # pytest coverage for core modules
├─ schema.sql, ingest.cql     # SQLite schema + Neo4j constraints/projection
├─ data/                      # Place large Discord exports here (gitignored)
├─ discord.db                 # Local staging DB (not checked in)
└─ README.md
```

---

## Environment & Setup
- **Python 3.13**: `pyenv install 3.13.0 && pyenv local 3.13.0`.
- **Dependency management**: `uv sync` reads `pyproject.toml` / `uv.lock` and installs into the local virtual env.
- **Secrets**: pass credentials via env vars (`NEO4J_PASSWORD`, `NEO4J_URI`, `LLAMA_API_KEY`, `LLAMA_BASE_URL`, etc.). Never hard-code keys or commit raw exports.
- **Execution**: run everything through `uv run ...` so dependencies resolve consistently.

---

## Running the Pipeline
One-shot (all stages, resumable):
```bash
uv run python run_pipeline.py \
  --sqlite ./discord.db \
  --schema ./schema.sql \
  --json-dir ./data \
  --neo4j-password "$NEO4J_PASSWORD"
```
Key flags:
- `--json` or `--json-dir`: input exports (files already recorded in `import_batch` are skipped unless `--no-skip-existing`).
- `--resume` / `--restart`: continue or replace the most recent `pipeline_run` without re-ingesting.
- `--ie-window-size`, `--ie-max-windows`, `--ie-max-concurrent-requests`: throttle llama-server load for long chats.
- `--ie-confidence` & `--fact-confidence`: tune minimum thresholds per stage.
- `--fact-types`: focus fact materialization on a subset (e.g., `WORKS_AT LIVES_IN`).
- `--no-ie`, `--no-fact-graph`: skip downstream stages if you only need raw ingest or the core interaction graph.

Stage progress, details, and timestamps persist in SQLite so partially completed IE runs can be paused (Ctrl+C) and resumed without losing processed windows.

---

## Stage Entry Points
### 1. Import Discord JSON → SQLite
```bash
uv run python import_discord_json.py --db ./discord.db --json-dir ./data
```
- Recurses through `*.json`, deduplicates via `import_batch`, enforces foreign keys, and captures roles, reactions, embeds, attachments, mentions, and inline emoji.
- Use `--json` for single files and `--no-skip-existing` to force re-import.

### 2. Load SQLite → Neo4j
```bash
uv run python loader.py \
  --sqlite ./discord.db \
  --neo4j bolt://localhost:7687 \
  --user neo4j \
  --password "$NEO4J_PASSWORD"
```
- Merges core nodes/edges, creates constraints, and materializes symmetric `INTERACTED_WITH` edges (reply weight 3, mention weight 1 by default; adjust inside `materialize_interactions`).

### 3. IE (windowed llama-server extraction)
```bash
uv run python run_ie.py \
  --sqlite ./discord.db \
  --window-size 6 \
  --confidence-threshold 0.6 \
  --llama-url "$LLAMA_BASE_URL" \
  --llama-model "$LLAMA_MODEL"
```
- Streams chronological windows per channel/guild, calls llama-server via `ie/client.py`, validates output against `ie/models.py`, and writes into `fact` / `fact_evidence`. Resume by re-running with `--resume`.

### 4. Materialize facts → Neo4j
```bash
uv run python facts_to_graph.py \
  --sqlite ./discord.db \
  --password "$NEO4J_PASSWORD" \
  --min-confidence 0.55
```
- Handles education, skills, projects, relationships, events, preferences, recommendations, and avoidance facts. Marks processed facts with `graph_synced_at` to keep Neo4j in sync.

---

## Information Extraction & Fact Catalogue
IE runs inside `ie/` combine:
- **Windowing** (`ie/windowing.py`): deterministic channel-ordered sliding windows with per-guild/per-author filters.
- **Prompting** (`ie/advanced_prompts.py`, `ie/prompt_assets.json`): reusable scaffolding + JSON assets for few-shots and rubric tweaks without code changes.
- **Runner** (`ie/runner.py`): concurrency limits, retries, confidence gating, and SQLite persistence.

Current fact coverage (see `ie/types.py` for the canonical list):

| Category | Fact types | Graph targets |
| --- | --- | --- |
| Work & Education | `WORKS_AT`, `STUDIED_AT`, `PREVIOUSLY` | `(:Person)-[:WORKS_AT]->(:Org)` and `(:Person)-[:STUDIED_AT]->(:Org {type:'School'})` with role/location/dates. |
| Skills & Projects | `HAS_SKILL`, `WORKING_ON` | `Person` → `Skill` / `Project` nodes including proficiency, scope, and timeframe attributes. |
| Relationships & Memory | `CLOSE_TO`, `RELATED_TO`, `REMEMBERS` | Weighted `Person`↔`Person` edges with relationship basis + evidence arrays. |
| Topics & Beliefs | `TALKS_ABOUT`, `CARES_ABOUT`, `CURIOUS_ABOUT`, `BELIEVES` | `Person` → `Topic` edges capturing sentiment, rationale, and confidence. |
| Location & Events | `LIVES_IN`, `ATTENDED_EVENT`, `WITNESSED`, `EXPERIENCED` | `Person` → `Place` / `Event` nodes with normalized timestamps and context. |
| Preferences & Plans | `PREFERS`, `DISLIKES`, `ENJOYS`, `PLANS_TO` | `Person` → `Preference` nodes describing likes/dislikes and future intent. |
| Recommendations & Warnings | `RECOMMENDS`, `AVOIDS` | `Person` → `Recommendation` nodes detailing endorsements or cautions with reasons. |

Each fact stores structured attributes (organization, role, since/until, sentiment basis, etc.), a confidence score, and evidence message IDs. `facts_to_graph.py` sanitizes values, ensures people/org nodes exist, and deduplicates evidence before writing graph relationships.

---

## Prompt Scaffolding
- Edit **few-shots & rubrics** in `ie/prompt_assets.json`; they are loaded at runtime without code changes.
- Extend or override prompt templates in `ie/advanced_prompts.py` to add new fact families.
- Domain objects in `data_structures/ingestion/models.py` keep serialization consistent when swapping models/providers.

---

## Memory Agent Service
`memory_agent/` contains a FastAPI application that turns the Neo4j graph into LangGraph-powered retrieval tools consumable by downstream chat agents.

Run locally after facts & embeddings are populated:
```bash
uv run uvicorn memory_agent.app:create_app --host 0.0.0.0 --port 8000
```
Endpoints:
- `POST /api/memory/retrieve`: main entry point returning stitched fact summaries, supporting metadata, and confidence values.
- `POST /api/memory/retrieve/debug`: verbose trace when `ENABLE_DEBUG_ENDPOINT=true`.
- `GET /health`: dependency liveness (Neo4j, embedding indexes, llama-server reachability).

Environment toggles (see `memory_agent/config.py`): Neo4j credentials/URIs, llama/embedding providers, per-tool limits, timeouts, rate limiting, and optional tracing.

The service automatically calls the **semantic fact** and **message** search tools when embeddings are available. See `docs/memory_agent_*` for architectural notes and prompt evolution plans.

---

## Embedding Jobs
Populate or refresh vector indexes whenever facts or messages change:
```bash
# Fact embeddings (used by semantic_search_facts tool)
uv run python scripts/embed_facts.py --cleanup

# Message embeddings (search verbatim phrasing alongside structured facts)
uv run python scripts/embed_messages.py --cleanup
```
Both scripts:
- Sanitize text payloads with channel/person context.
- Use the configured sentence-transformers model (`EMBEDDING_MODEL` / `MESSAGE_EMBEDDING_MODEL`).
- Batch requests (`MESSAGE_EMBEDDING_BATCH_SIZE` defaults to 128).
- Upsert `:FactEmbedding` / `:MessageEmbedding` nodes and rebuild the associated vector index (cosine similarity, 768 dims by default).
- Accept `--dry-run` to preview work without touching Neo4j.

Run `scripts/graph_snapshot.py` to capture a Cypher export for regression testing before/after IE changes.

---

## Query Cookbook
- **Top interaction partners**
  ```cypher
  MATCH (:Person {id:$personId})-[r:INTERACTED_WITH]->(p:Person)
  RETURN p.name AS name, r.weight AS weight
  ORDER BY weight DESC
  LIMIT 10;
  ```
- **Guild + channel activity**
  ```cypher
  MATCH (p:Person)-[:SENT]->(:Message)-[:IN_CHANNEL]->(c:Channel)-[:IN_GUILD]->(g:Guild {id:$guild})
  RETURN p.name, c.name, count(*) AS msgs
  ORDER BY msgs DESC LIMIT 20;
  ```
- **Work history rollup**
  ```cypher
  MATCH (p:Person)-[r:WORKS_AT]->(o:Org)
  RETURN p.name, o.name, r.role, r.location, r.confidence
  ORDER BY r.confidence DESC LIMIT 15;
  ```
- **Skill graph**
  ```cypher
  MATCH (p:Person)-[:HAS_SKILL]->(s:Skill)
  RETURN s.name AS skill, collect(p.name)[0..5] AS sample_people, count(*) AS holders
  ORDER BY holders DESC;
  ```
- **Preferences & recommendations**
  ```cypher
  MATCH (p:Person)-[r:PREFERS|RECOMMENDS]->(t)
  RETURN labels(t)[0] AS type, t.name AS target, collect(p.name)[0..3] AS advocates
  ORDER BY size(advocates) DESC;
  ```

---

## Troubleshooting
- **Foreign key errors**: ensure `PRAGMA foreign_keys = ON;` (the importer sets this) and run `sqlite3 ./discord.db "SELECT count(*) FROM message;"` to confirm ingest.
- **Neo4j driver complaining about multiple statements**: loader already separates Cypher statements; check for custom Cypher you injected elsewhere.
- **Pipeline stalled mid-IE**: rerun `uv run python run_pipeline.py --resume --neo4j-password "$NEO4J_PASSWORD" ...` to pick up remaining windows; see `pipeline_stage_state.details` (JSON) for queue length.
- **Embeddings out of sync**: rerun `embed_facts.py --cleanup` or `embed_messages.py --cleanup` to rebuild indexes after deleting facts/messages.

---

## Performance Tips
- Ingest large exports in batches; the importer deduplicates via `import_batch` so you can safely re-run.
- For massive Neo4j loads, increase Docker memory + page cache and consider running loader with `NEO4J_MAX_CONNECTION_POOL_SIZE` tuned higher.
- Temporarily disable llama-server streaming if CPU bound; `--ie-max-concurrent-requests` throttles concurrency.
- Use `scripts/graph_snapshot.py` before/after IE prompt tweaks to confirm relationship deltas.

---

## Roadmap
- [x] SQLite schema + importer with provenance tracking
- [x] Neo4j loader + interaction graph + GDS projection
- [x] Windowed IE runner (Pydantic validation, resume support, llama-server client)
- [x] Fact materialization covering education, skills, relationships, events, preferences, and memory cues
- [x] LangGraph-powered memory agent + semantic search embeddings
- [ ] Profile generator: combine facts + graph metrics + evidence into narrative dossiers
- [ ] Streamlit/Notebooks for people/topic exploration and fact QA
- [ ] Privacy controls (per-person redaction, “forget me” requests, sensitive topic filters)
- [ ] Automated eval harness for IE prompts + embedding relevance (gold fixtures in `tests/`)

---

## Data & Safety
- Store raw Discord exports inside `data/` (gitignored) and keep only sanitized fixtures in the repo.
- SQLite artifacts (`discord.db`) live at the repo root but must not be committed.
- Use environment variables for all credentials (Neo4j, llama-server, embedding providers). Document overrides in PRs instead of committing .env files.
- See `docs/` for setup notes (llama-server, profile generation) and keep personal data out of screenshots.

---

## Quickstart
```bash
# 0) Ensure Python 3.13 + deps
pyenv install 3.13.0
pyenv local 3.13.0
uv sync

# 1) Apply schema (idempotent)
uv run python - <<'PY'
from pathlib import Path
from run_pipeline import apply_schema
apply_schema(Path('discord.db'), Path('schema.sql'))
PY

# 2) Import Discord exports
uv run python import_discord_json.py --db ./discord.db --json-dir ./data

# 3) Start Neo4j locally
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.22

# 4) Load SQLite → Neo4j
uv run python loader.py --sqlite ./discord.db --neo4j bolt://localhost:7687 --user neo4j --password 'test'

# 5) Run IE + fact materialization (optionally through run_pipeline)
uv run python run_pipeline.py --sqlite ./discord.db --schema ./schema.sql --json-dir ./data --neo4j-password 'test'

# 6) Populate embeddings + start memory agent
uv run python scripts/embed_facts.py --cleanup
uv run python scripts/embed_messages.py --cleanup
uv run uvicorn memory_agent.app:create_app --host 0.0.0.0 --port 8000
```

---

## License
TBD (private/internal by default). Add an explicit license before distribution.
