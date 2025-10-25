# Embedded Message Semantic Search

## 1. Goal & Scope
- Add semantic search over raw Discord messages to complement fact-level search.
- Reuse the existing `memory_agent` embedding infrastructure and Neo4j vector indexes.
- Keep the pipeline orthogonal: message embeddings run independently but in a similar fashion to `scripts/embed_facts.py` so teams can refresh either index without re-ingesting the graph.

## 2. Target Experience
- **Offline job**: `uv run python scripts/embed_messages.py --cleanup` (new) to backfill and refresh embeddings for all `Message` nodes.
- **Realtime usage**: expose a `semantic_search_messages` tool (or API endpoint) that takes free-text queries and returns high-similarity messages with metadata (author, channel, timestamp, link).
- **Ops ergonomics**: the job should be resumable, support dry-runs, and emit structured summaries like the fact pipeline does.

## 3. Data Model Updates (Neo4j)
1. Create a new label `MessageEmbedding` with unique key `message_id` mirroring how `FactEmbedding` works.
2. Vector index definition (Cypher):
   ```cypher
   CREATE VECTOR INDEX message_embeddings IF NOT EXISTS
   FOR (m:MessageEmbedding)
   ON m.embedding
   OPTIONS {
     indexConfig: {
       `vector.dimensions`: 768,
       `vector.similarity_function`: 'cosine'
     }
   }
   ```
3. Stored properties per node:
   - `message_id`, `content`, `clean_content` (after preprocessing), `author_id`, `author_name`, `channel_id`, `channel_name`, `timestamp`, `guild_id`, `guild_name`.
   - `mentions` (list of person IDs), `attachments` (serialized info), `reactions` (JSON summary), `thread_id` (optional).
   - `embedding` (vector), `embedding_model`, `created_at`, `updated_at`.
4. Relationship back-references are optional; linking `(m:MessageEmbedding)-[:EMBEDS]->(msg:Message)` keeps navigation simple.

## 4. Embedding Generation Pipeline
1. **Module layout**
   - Create `memory_agent/message_embedding_pipeline.py` mirroring `memory_agent/embedding_pipeline.py`.
   - Reuse `EmbeddingProvider` and shared sanitization helpers where possible (consider extracting them into `memory_agent/embedding_utils.py`).
2. **Fetcher**
   - Query Neo4j for all `Message` nodes (or a subset via `WHERE m.timestamp > $since` CLI flag later).
   - Fetch related info in the same Cypher to minimize round-trips (channels, people, mentions, reactions).
   - Only include messages that have `content` after trimming, skipping pure attachment stubs.
3. **Text formatting**
   - Add `format_message_for_embedding_text` helper inside `memory_agent/message_formatter.py` (or extend `memory_agent/fact_formatter.py`).
   - Normalization steps: strip markdown, collapse whitespace, optionally append channel/topic context and mention names.
4. **Batching & encoding**
   - Same `chunk_iterable` utility; default batch size `128` because messages are shorter than fact descriptions.
   - Use `EmbeddingProvider.embed` with the configured model; keep normalization enabled for cosine index.
5. **Persistence**
   - Create `upsert_message_embeddings` function to write `MessageEmbedding` nodes and update timestamps.
   - Implement `cleanup_orphan_message_embeddings` (delete nodes when `Message` no longer exists).
6. **Summary output**
   - Return counts: `messages_scanned`, `messages_embedded`, `skipped_empty`, `embeddings_written`, `cleaned_orphans`.

## 5. CLI Entry Point (`scripts/embed_messages.py`)
1. Copy the structure of `scripts/embed_facts.py`.
2. CLI flags:
   - `--settings-from-env` (hidden, for parity).
   - `--channel` / `--guild` filters (optional) to embed subsets while testing.
   - `--cleanup` boolean.
   - `--dry-run` to log summary without writing (skip upsert/cleanup).
   - `--since` ISO timestamp to embed only recent messages.
3. Steps:
   - Load `Settings` (extend config with message-embedding defaults under `settings.embeddings_messages` if needed).
   - Instantiate `EmbeddingProvider` (optionally allow a different model/env variable `MESSAGE_EMBEDDING_MODEL`).
   - Run `run_message_embedding_pipeline` with filters/flags.

## 6. Semantic Search Tooling
1. **Tool class**: `memory_agent/tools/semantic_search_messages.py` modeled after `semantic_search_facts`.
2. Differences:
   - Input may include `channel_ids`, `author_ids`, `time_range` filters.
   - Output includes message permalink (constructed via guild/channel/message IDs) plus excerpt.
3. Share runtime pieces:
   - Use `run_vector_query` against `message_embeddings` index.
   - Add helper to trim/clean the stored `content` so results are human-readable.
4. Register the tool inside `memory_agent/app.py` or wherever tool registry lives so the agent can call it.

## 7. Configuration & Ops
- Extend `.env.example` / README to cover:
  - `MESSAGE_EMBEDDING_MODEL` (default to fact model for now).
  - Optional `MESSAGE_EMBEDDING_DEVICE`, `MESSAGE_EMBEDDING_BATCH_SIZE`.
- Document the new maintenance workflow in `CLAUDE.md` + `README.md` (similar to fact embedding instructions).
- Consider scheduling via cron or Airflow if message volume is high; pipeline should support partial updates via `--since`.

## 8. Testing Strategy
1. **Unit tests**
   - `tests/test_message_formatter.py` verifying markdown stripping, mention handling, attribute serialization.
   - `tests/test_message_embedding_pipeline.py` using fake `EmbeddingProvider` to assert Cypher payloads.
2. **Integration tests**
   - Use `neo4j` test harness or mock driver to ensure `ensure_vector_index` and upserts generate expected Cypher.
   - End-to-end smoke test that loads a temporary in-memory Neo4j via Testcontainers (optional).
3. **Tool tests**
   - Mirror `tests/test_memory_agent_embedding.py` but for the message search tool; mock `run_vector_query` results and ensure dedup/filters work.

## 9. Rollout Plan
1. Land the pipeline + CLI + tests.
2. Backfill production Neo4j with `scripts/embed_messages.py --cleanup` after verifying on staging.
3. Enable the semantic search tool in the agent UI and capture metrics (query latency, hit counts).
4. Iterate on prompt-side usage (e.g., retrieving top-K messages for persona analysis) once embeddings prove stable.

## 10. Open Questions / Follow-Ups
- Should we store embeddings alongside the `Message` node instead of a sibling `MessageEmbedding` node to reduce hops? (Current plan keeps parity with facts.)
- Do we need different models for messages vs. facts (e.g., instruct-tuned vs. general)? Config already allows override.
- How do we treat non-textual messages (images, stickers)? Future extension could embed captions or OCR output.
- Rate limiting: if the message corpus is huge, do we need incremental checkpoints to avoid re-embedding unchanged messages?

