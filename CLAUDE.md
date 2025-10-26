# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Knowledge extraction system that transforms Discord message exports into a multi-layer knowledge graph. The architecture uses **SQLite** for staging and provenance, **Neo4j** for the graph database, and **LLM-powered information extraction** to discover relationships, skills, organizations, and other structured facts from unstructured conversations.

## Development Commands

### Environment Setup
```bash
# Requires Python 3.13+
pyenv install 3.13.0
pyenv local 3.13.0

# Install dependencies (uses uv)
uv sync
```

### Database Setup
```bash
# Initialize SQLite schema (idempotent)
sqlite3 ./discord.db < schema.sql

# Start Neo4j (Docker example)
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.22
```

### Running the Pipeline

**Full pipeline** (Discord JSON → SQLite → Neo4j → IE → fact materialization):
```bash
uv run python run_pipeline.py \
  --sqlite ./discord.db \
  --schema ./schema.sql \
  --json-dir ./exports \
  --neo4j-password 'test'
```

**Individual stages**:
```bash
# 1. Import Discord JSON to SQLite only
uv run python import_discord_json.py --db ./discord.db --json-dir ./exports

# 2. Load SQLite to Neo4j only
uv run python loader.py \
  --sqlite ./discord.db \
  --neo4j bolt://localhost:7687 \
  --user neo4j \
  --password 'test'

# 3. Run information extraction only
uv run python run_ie.py \
  --sqlite ./discord.db \
  --window-size 6 \
  --confidence-threshold 0.6

# 4. Materialize facts to Neo4j graph
uv run python facts_to_graph.py \
  --sqlite ./discord.db \
  --password 'test' \
  --min-confidence 0.6
```

### Neo4j Operations
```bash
# Apply constraints and create GDS projection
cat ingest.cql | cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'
```

### Memory Agent Service
```bash
# Start the FastAPI microservice
uv run uvicorn memory_agent.app:create_app --host 0.0.0.0 --port 8000

# Generate and index fact embeddings for semantic search
uv run python scripts/embed_facts.py --cleanup
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_memory_agent_agent.py

# Run with verbose output
uv run pytest -v
```

### Fact Deduplication
```bash
# Deduplicate extracted facts using MinHash LSH + embeddings + LLM verification
uv run python deduplicate_facts.py \
  --sqlite ./discord.db \
  --neo4j-password 'test' \
  --minhash-threshold 0.80 \
  --embedding-threshold 0.95 \
  --min-confidence 0.5

# Resume an interrupted deduplication run
uv run python deduplicate_facts.py --sqlite ./discord.db --neo4j-password 'test' --resume

# Dry run (no changes persisted)
uv run python deduplicate_facts.py --sqlite ./discord.db --neo4j-password 'test' --dry-run
```

### Graph Inspection
```bash
# Validate materialized facts and get graph statistics
uv run python scripts/graph_snapshot.py --password 'test'

# Show samples of specific relationship types
uv run python scripts/graph_snapshot.py --password 'test' --sample-limit 20
```

### Data Inspection
```bash
# Check SQLite data
sqlite3 ./discord.db "SELECT count(*) FROM message;"
sqlite3 ./discord.db "SELECT type, subject_id, confidence FROM fact ORDER BY confidence DESC LIMIT 10;"

# Neo4j Browser: http://localhost:7474
# Example Cypher queries in README.md Query Cookbook section
```

## Architecture

### Three-Layer Data Flow

1. **SQLite (Staging & Provenance)**
   - Lossless Discord data storage
   - Schema: `schema.sql` defines ~20 tables including `guild`, `channel`, `member`, `message`, `attachment`, `embed`, `reaction`
   - IE provenance: `ie_run`, `fact`, `fact_evidence` tables track extraction runs and fact sources
   - Key indices: `idx_message_channel_ts`, `idx_message_author_ts`

2. **Neo4j (Graph Database)**
   - Core entities: `Person`, `Guild`, `Channel`, `Message`, `Role`
   - Derived entities: `Topic`, `Org`, `Place`, `Project`, `Skill`, `Event`
   - Key relationships:
     - `(:Person)-[:SENT]->(:Message)-[:IN_CHANNEL]->(:Channel)`
     - `(:Message)-[:REPLIES_TO]->(:Message)`
     - `(:Message)-[:MENTIONS]->(:Person)`
     - `(:Person)-[:INTERACTED_WITH {weight}]->(:Person)` - materialized from replies (weight 3) and mentions (weight 1)
     - Fact-based: `[:WORKS_AT]`, `[:LIVES_IN]`, `[:TALKS_ABOUT]`, `[:HAS_SKILL]`, `[:STUDIED_AT]`, etc.

3. **LLM Information Extraction**
   - Uses local llama-server (OpenAI-compatible API) with model GLM-4.5-Air
   - Windowed extraction: processes conversations in sliding windows (default size 4)
   - 20+ fact types defined in `ie/config.py`: employment, education, skills, relationships, preferences, beliefs
   - Confidence thresholding before graph materialization

### Key Modules

- **`run_pipeline.py`**: Orchestrates full ingest → IE → fact materialization flow with resumable stages
- **`import_discord_json.py`**: Parses Discord JSON exports and loads into SQLite with batch tracking
- **`loader.py`**: Transforms SQLite data into Neo4j nodes/relationships, computes `INTERACTED_WITH` weights
- **`facts_to_graph.py`**: Materializes high-confidence facts from SQLite into Neo4j graph edges
- **`ie/`**: Information extraction subsystem
  - `ie/runner.py`: IE job orchestration
  - `ie/windowing.py`: Message window generation for context
  - `ie/client.py`: llama-server API client
  - `ie/config.py`: 20+ fact type definitions with schemas
  - `ie/advanced_prompts.py`: Enhanced prompt scaffolding
  - `ie/prompt_assets.json`: Few-shot examples for prompt engineering (update without touching code)
- **`memory_agent/`**: FastAPI microservice for fact retrieval
  - `memory_agent/app.py`: FastAPI application with `/api/memory/retrieve` endpoint
  - `memory_agent/agent.py`: LangGraph-based agentic retrieval workflow
  - `memory_agent/tools/`: Neo4j-backed retrieval tools (person profile, timeline, semantic search, etc.)
  - `memory_agent/config.py`: Environment-based configuration
- **`deduplicate/`**: Fact deduplication subsystem
  - `deduplicate/core.py`: Orchestration and merge logic
  - `deduplicate/similarity/`: MinHash LSH and embedding-based similarity detection
  - `deduplicate/llm/`: LLM-powered merge decision making
  - `deduplicate/partitioning.py`: Partition facts by type for parallel processing
- **`scripts/`**: Utility scripts
  - `scripts/embed_facts.py`: Generates embeddings for facts and maintains `fact_embeddings` vector index
  - `scripts/embed_messages.py`: Generates embeddings for raw Discord messages (backfills `message_embeddings` index)
  - `scripts/graph_snapshot.py`: Validates materialized facts and reports graph statistics
- **`deduplicate_facts.py`**: CLI entry point for fact deduplication (hybrid MinHash + embeddings + LLM)

## Environment Variables

### Neo4j
- `NEO4J_URI`: Connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Username (default: `neo4j`)
- `NEO4J_PASSWORD`: Password (default: `test`)
- `NEO4J_DATABASE`: Optional database name

### LLM (llama-server)
- `LLAMA_MODEL`: Model name (default: `GLM-4.5-Air`)
- `LLAMA_BASE_URL`: API endpoint (default: `http://localhost:8080/v1/chat/completions`)
- `LLAMA_API_KEY`: Optional API key
- `LLAMA_TEMPERATURE`: Sampling temperature (default: `0.3`)
- `LLAMA_TOP_P`: Nucleus sampling (default: `0.95`)
- `LLAMA_MAX_TOKENS`: Max completion tokens (default: `4096`)
- `LLAMA_TIMEOUT`: Request timeout in seconds (default: `1200`)

### Embeddings (for semantic search)
- `EMBEDDING_MODEL`: Model name (default: `google/embeddinggemma-300m`)
- `EMBEDDING_DEVICE`: Device (default: `cpu`)
- `EMBEDDING_CACHE_DIR`: Optional cache directory

### Message Embeddings Job
- `MESSAGE_EMBEDDING_MODEL`: Override model for message embeddings (default: `EMBEDDING_MODEL`)
- `MESSAGE_EMBEDDING_DEVICE`: Override device for message embeddings (default: `EMBEDDING_DEVICE`)
- `MESSAGE_EMBEDDING_CACHE_DIR`: Optional cache dir for message jobs (default: `EMBEDDING_CACHE_DIR`)
- `MESSAGE_EMBEDDING_BATCH_SIZE`: Batch size for `scripts/embed_messages.py` (default: `128`)

### Memory Agent Service
- `API_HOST`: FastAPI host (default: `0.0.0.0`)
- `API_PORT`: FastAPI port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `ENABLE_CORS`: Enable CORS (default: `false`)
- `CORS_ALLOW_ORIGINS`: Comma-separated allowed origins
- `ENABLE_DEBUG_ENDPOINT`: Enable `/api/memory/retrieve/debug` (default: `false`)
- `MAX_ITERATIONS`: Agent max iterations (default: `10`)
- `MAX_FACTS`: Max facts to return (default: `30`)

## Important Implementation Details

### IE Fact Schema
Facts are defined in `ie/config.py` as `FactDefinition` objects with:
- `type`: Enum value from `FactType` (e.g., `WORKS_AT`, `LIVES_IN`)
- `subject_description`: What the subject represents
- `object_type`: Category of the object (Organization, Place, Topic, etc.)
- `attributes`: Tuple of `FactAttribute` (name, description, required flag)

Example fact types:
- Employment: `WORKS_AT`, `WORKING_ON`
- Education: `STUDIED_AT`, `HAS_SKILL`
- Location: `LIVES_IN`
- Relationships: `CLOSE_TO`, `RELATED_TO`
- Engagement: `TALKS_ABOUT`, `CURIOUS_ABOUT`, `ENJOYS`, `DISLIKES`
- History: `ATTENDED_EVENT`, `EXPERIENCED`, `PREVIOUSLY`
- Preferences: `PREFERS`, `RECOMMENDS`, `AVOIDS`, `BELIEVES`

### Interaction Weight Calculation
In `loader.py:materialize_interactions()`:
- Replies: weight = 3 (stronger signal of engagement)
- Mentions: weight = 1 (weaker signal)
- Bidirectional: weights are averaged for A↔B symmetry
- Can be tuned by modifying the weight constants

### Official Names
Populate `member.official_name` in SQLite to help IE disambiguate real names from Discord handles:
```sql
UPDATE member SET official_name='John Smith' WHERE id='...';
```

### Fact Evidence Provenance
Each fact in the `fact` table has corresponding `fact_evidence` entries linking back to specific `message_id`s, maintaining full traceability.

### Pipeline Resumability
`run_pipeline.py` tracks completion in `pipeline_stage` table, allowing re-runs to skip completed stages. Use `--reset-stage <stage>` to force re-execution.

### Fact Deduplication Pipeline
`deduplicate_facts.py` uses a three-stage approach:
1. **MinHash LSH**: Fast candidate detection using character n-grams (default threshold: 0.80 Jaccard similarity)
2. **Embedding similarity**: Refine candidates using semantic embeddings (default threshold: 0.95 cosine similarity)
3. **LLM verification**: Final merge decision with confidence scoring and attribute reconciliation
- Deduplication runs are persisted in `dedup_run` table with resumability support
- Use `--dry-run` to preview merges without modifying the database or graph

## File Naming & Organization

- Scripts use `snake_case.py` (e.g., `import_discord_json.py`)
- Modules under `ie/` and `memory_agent/` use `snake_case.py`
- Data files: `discord.db` (SQLite), `schema.sql` (schema definition), `ingest.cql` (Neo4j setup)
- Exports directory: `data/` or `./exports` (not committed to git)
- Test fixtures: `tests/fixtures/`

## Common Pitfalls

1. **Foreign key violations**: SQLite requires `PRAGMA foreign_keys = ON;` - the importer sets this automatically
2. **Neo4j driver errors**: Each `session.run()` accepts only one statement; multi-statement queries must be split
3. **Missing embeddings**: Run `scripts/embed_facts.py --cleanup` after loading facts to enable semantic search
4. **IE confidence tuning**: Default threshold is 0.5; increase to 0.7+ for higher precision, lower to 0.3 for recall
5. **llama-server connection**: Ensure llama-server is running at `LLAMA_BASE_URL` before running IE
6. **GDS projection timing**: Run `ingest.cql` AFTER `loader.py` has materialized `INTERACTED_WITH` relationships
7. **Duplicate facts**: After running IE multiple times, use `deduplicate_facts.py` to merge semantically identical facts
8. **Transformers dependency**: The project uses a specific git revision of transformers for Embedding Gemma support (see `pyproject.toml`)

## Coding Style & Conventions

- Follow PEP 8 with four-space indentation and type hints on all public functions
- Use `Path` from `pathlib` for filesystem paths (not bare strings)
- Modules and files use `snake_case`; classes use `PascalCase`; constants are `UPPER_SNAKE_CASE`
- Prefer Pydantic models (see `ie/models.py`, `memory_agent/models.py`) for structured data
- Document command-line scripts with module docstrings
- Add short comments before complex SQL or Cypher operations
- Keep commit messages imperative and ≤72 chars (e.g., "Handle duplicate facts")

## Testing Strategy

- Use `pytest` with test files mirroring module structure (e.g., `tests/test_loader.py`)
- Mock external dependencies (llama-server, Neo4j) via adapters in `ie/client.py`
- Test edge cases: duplicate facts, null confidence values, malformed Discord exports
- Integration tests use `conftest.py` fixtures for shared test data
- Run `uv run pytest -v` for detailed test output
