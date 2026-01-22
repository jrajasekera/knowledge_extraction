# Suggested Commands

## Environment Setup
```bash
# Install Python 3.13+
pyenv install 3.13.0
pyenv local 3.13.0

# Install dependencies
uv sync
```

## Database Setup
```bash
# Initialize SQLite schema (idempotent)
sqlite3 ./discord.db < schema.sql

# Enable WAL mode for concurrent access
uv run python scripts/enable_wal_mode.py --db-path ./discord.db

# Start Neo4j (Docker)
docker run -p7474:7474 -p7687:7687 -e NEO4J_AUTH=neo4j/test neo4j:5.22
```

## Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_memory_agent_agent.py

# Run with verbose output
uv run pytest -v

# Run with coverage
uv run coverage run -m pytest
uv run coverage report
```

## Linting & Formatting
```bash
# Lint check
uv run ruff check .

# Auto-fix lint issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Type checking
uv run pyright
```

## Running the Pipeline

### Full Pipeline
```bash
uv run python run_pipeline.py \
  --sqlite ./discord.db \
  --schema ./schema.sql \
  --json-dir ./exports \
  --neo4j-password 'test'
```

### Individual Stages
```bash
# 1. Import Discord JSON to SQLite
uv run python import_discord_json.py --db ./discord.db --json-dir ./exports

# 2. Load SQLite to Neo4j
uv run python loader.py \
  --sqlite ./discord.db \
  --neo4j bolt://localhost:7687 \
  --user neo4j \
  --password 'test'

# 3. Run information extraction
uv run python run_ie.py \
  --sqlite ./discord.db \
  --window-size 6 \
  --confidence-threshold 0.6

# 4. Materialize facts to Neo4j
uv run python facts_to_graph.py \
  --sqlite ./discord.db \
  --password 'test' \
  --min-confidence 0.6
```

## Memory Agent Service
```bash
# Start the FastAPI microservice
uv run uvicorn memory_agent.app:create_app --host 0.0.0.0 --port 8000

# Generate fact embeddings
uv run python scripts/embed_facts.py --cleanup
```

## Fact Deduplication
```bash
# Deduplicate facts
uv run python deduplicate_facts.py \
  --sqlite ./discord.db \
  --neo4j-password 'test' \
  --minhash-threshold 0.80 \
  --embedding-threshold 0.95

# Dry run (no changes)
uv run python deduplicate_facts.py --sqlite ./discord.db --neo4j-password 'test' --dry-run
```

## Graph Inspection
```bash
# Graph statistics and validation
uv run python scripts/graph_snapshot.py --password 'test'

# Check SQLite data
sqlite3 ./discord.db "SELECT count(*) FROM message;"
sqlite3 ./discord.db "SELECT type, subject_id, confidence FROM fact ORDER BY confidence DESC LIMIT 10;"
```

## Git
```bash
git status
git add <file>
git commit -m "message"
git push
```
