# Pyright Fixes Plan

## Context
- Pyright version: 1.1.408 (installed via `uv add --dev pyright`).
- Config: `pyrightconfig.json` (basic mode + warnings for unknown/missing types, excludes data/exports/etc).
- Latest run: `uv run pyright` and `uv run pyright --stats`.
- Diagnostics: 259 errors, 824 warnings (1083 total).

## High-Impact Areas (by diagnostic volume)
- Core ingestion/graph code: `loader.py`, `facts_to_graph.py`, `scripts/graph_snapshot.py`.
- Memory agent pipelines: `memory_agent/embedding_pipeline.py`, `memory_agent/message_embedding_pipeline.py`, `memory_agent/embeddings.py`, `memory_agent/normalization.py`.
- Tests: `tests/test_semantic_search_rrf.py`, `tests/test_fact_query_extraction.py`.
- Deduplication: `deduplicate/*` (parser, persistence, progress, minhash, logging_support).

## Plan (no fixes applied yet)

### 1) Normalize data model types (Discord ingest + ingestion models)
- Add concrete type parameters to collection fields in `data_structures/ingestion/models.py` (e.g., `list[Role]`, `list[Attachment]`, `list[EmbedField]`, etc.) and ensure Pydantic defaults use `Field(default_factory=list)`.
- Introduce `TypedDict` (or Pydantic models) for Discord JSON shapes used in `import_discord_json.py` to replace `dict[str, Any]`/`Any` and remove `Unknown` propagation.
- Guard `int(...)` conversions for Optional values (`None` checks or `or 0` where appropriate) to resolve `ConvertibleToInt` errors in `import_discord_json.py`.

### 2) Add explicit Neo4j/SQLite typing to graph pipeline code
- In `loader.py`, `facts_to_graph.py`, and `scripts/graph_snapshot.py`:
  - Type annotate Neo4j objects (`neo4j.Driver`, `neo4j.Session`, `neo4j.Transaction`, `neo4j.Record`).
  - Type annotate SQLite connections/cursors (`sqlite3.Connection`, `sqlite3.Cursor`).
  - Use `Mapping[str, Any]` for query params and return types for `tx.run(...)` results.
- In `run_pipeline.py` and `ie/runner.py`, reconcile `Path` vs `str` arguments (accept `Path | str`, or cast before calling helper functions) to address argument type mismatches.

### 3) Fix configuration/env value types in embedding pipelines
- In `memory_agent/embeddings.py`, `memory_agent/embedding_pipeline.py`, `memory_agent/message_embedding_pipeline.py`:
  - Parse environment values into correct types (bool/int/float/Path) before passing into Neo4j and embedding constructors.
  - Replace stringly-typed values with typed config objects (dataclass/Pydantic) to stop `str`-to-`bool/int` errors.
- In `scripts/embed_facts.py` and `scripts/embed_messages.py`:
  - Replace integer literals used for Neo4j `driver(...)` options with proper enum/boolean values, or pass only the supported keyword args.
  - Annotate config parsing helpers to return correct types.

### 4) Resolve unknown types from external libs and missing stubs
- Provide local stubs or protocols for third-party libs without type hints:
  - `datasketch` (used in `deduplicate/similarity/minhash_lsh.py`) → add `typings/datasketch.pyi` and set `stubPath` in `pyrightconfig.json`.
  - If needed, add minimal stubs for `sentence_transformers` encode return types or create a small wrapper module with explicit return types used across embedding code.
- Where stubs are unavailable, use `typing.cast` with well-defined aliases (e.g., `EmbeddingArray = np.ndarray[Any, Any] | Tensor`).

### 5) Tighten LLM/deduplication typing and Optional handling
- In `deduplicate/llm/parser.py` and `deduplicate/llm/prompts.py`:
  - Introduce `TypedDict` for LLM payloads and responses, and type helper functions accordingly.
  - Ensure numeric conversions (`float`, `int`) guard against `object`/`None` values.
- In `deduplicate/persistence.py` and `deduplicate/progress.py`:
  - Guard Optional integers before `int(...)` conversions.
- In `deduplicate/logging_support.py`:
  - Align filter types to `logging.Filter` (or a Protocol) so `_copy_filters` accepts the actual list type used.

### 6) Memory agent normalization and tools
- In `memory_agent/normalization.py`:
  - Convert untyped lists/dicts into `FactEvidence` and other domain types explicitly (construct typed models rather than passing raw dicts).
- In `memory_agent/agent.py`, `memory_agent/llm.py`, and tool modules:
  - Add missing parameter/return annotations and narrow `Any` to `Mapping[str, Any]` or specific model types.

### 7) Test typing alignment
- In `tests/test_semantic_search_rrf.py`:
  - Update evidence parameter types to accept `Sequence[...]` (covariant) or cast test inputs to the expected union type.
  - Use `typing.cast`/`# pyright: ignore` for deliberately invalid literals and `None` driver cases that are part of negative tests.
- In `tests/test_fact_query_extraction.py`:
  - Update calls to match current function signatures (remove/rename `channel_id`, `guild_id`) or adjust function signatures if the test reflects intended API.
  - Avoid direct assignment to private attrs by using a helper or test hook that is explicitly typed.

### 8) Verification pass
- Re-run `uv run pyright` after each module group is addressed.
- Target: 0 errors, warnings only where explicitly justified (and documented in `pyrightconfig.json` or with inline `# pyright: ignore`).

