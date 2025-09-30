# Repository Guidelines

## Project Structure & Module Organization
- `run_pipeline.py` orchestrates the full ingest → IE → graph flow; use it once exports are staged in `data/` and the schema in `schema.sql`.
- `import_discord_json.py`, `loader.py`, and `facts_to_graph.py` are standalone entry points for Discord → SQLite, SQLite → Neo4j, and fact materialization (now covering education, skills, relationships, events, preferences, and more).
- IE-specific logic lives in `ie/` (client, prompts, runner, windowing) with reusable domain objects under `data_structures/ingestion/`.
- Enhanced prompt scaffolding is split between `ie/advanced_prompts.py` and `ie/prompt_assets.json`; update the JSON to tweak few-shots without touching code.
- Persisted artifacts (`discord.db`, JSON exports) sit at the repository root; keep large inputs in `data/` and check in only fixtures.
- Reference `docs/` for service notes (e.g., llama-server setup) and `ingest.cql` for Neo4j constraints and projections.

## Build, Test, and Development Commands
- Ensure Python 3.13 is active (`pyenv install 3.13.0`, `pyenv local 3.13.0`) and install deps with `uv sync` (reads `pyproject.toml` / `uv.lock`).
- Run the end-to-end pipeline: `uv run python run_pipeline.py --sqlite ./discord.db --schema ./schema.sql --json-dir ./exports --neo4j-password 'test'`.
- Re-run targeted stages: `uv run python run_ie.py --sqlite ./discord.db --window-size 6` or `uv run python facts_to_graph.py --sqlite ./discord.db --password 'test'`.
- Quick schema reset: `sqlite3 ./discord.db < schema.sql` (safe to rerun; migrations handled in code).

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation, type hints on all public functions, and `Path` over bare strings for filesystem paths.
- Modules and files use `snake_case`; classes use `PascalCase`; constants are `UPPER_SNAKE_CASE`.
- Prefer Pydantic models (see `ie/models.py`) for structured data and keep serialization in dedicated helpers.
- Document command-line scripts with module docstrings and short comments before complex SQL or graph operations.

## Testing Guidelines
- Use `pytest` for new tests; create `tests/` mirrors of modules (e.g., `tests/test_loader.py`) and isolate fixtures under `tests/fixtures/`.
- Target edge cases around SQLite writes, Neo4j merges, and IE confidence thresholds; mock external services (llama-server, Neo4j) via adapters in `ie/client.py`.
- Run `uv run pytest` locally; for manual spot checks, execute `sqlite3 ./discord.db "SELECT count(*) FROM message;"` and inspect Neo4j via cypher-shell.

## Commit & Pull Request Guidelines
- Keep commit subjects imperative and ≤72 chars (e.g., "Handle duplicate facts"), mirroring recent history (`git log --oneline`).
- Each PR should describe scope, data sources touched, and include screenshots or Cypher snippets when graph changes are user-visible.
- Link related issues, note schema impacts, and call out any required re-ingest or backfills before requesting review.

## Environment & Data Safety
- Never commit raw Discord exports or API keys; store them outside the repo or add to `.gitignore` entries under `data/`.
- Use environment variables (e.g., `NEO4J_PASSWORD`, `LLAMA_API_KEY`) rather than hard-coding credentials, and document overrides in the PR description.
