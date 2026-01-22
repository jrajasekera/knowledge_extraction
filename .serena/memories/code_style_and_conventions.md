# Code Style and Conventions

## General Style
- Follow **PEP 8** with four-space indentation
- Line length: **100 characters** (configured in ruff)
- Quote style: **double quotes**
- Indent style: **spaces**
- Line endings: **LF**

## Naming Conventions
- **Modules and files**: `snake_case.py` (e.g., `import_discord_json.py`)
- **Classes**: `PascalCase` (e.g., `ExtractionFact`, `RetrievalRequest`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_EMBEDDING_MODEL`)
- **Functions/methods/variables**: `snake_case`

## Type Hints
- **Required** on all public functions
- Pyright configured with `basic` type checking mode (Python 3.13)
- Use `Path` from `pathlib` for filesystem paths (not bare strings)
- Warnings enabled for unknown types

## Data Models
- Use **Pydantic v2** models for structured data
- See `ie/models.py` and `memory_agent/models.py` for patterns
- Pydantic models include validators using `@field_validator` decorators

## Imports
- Ruff handles import sorting (isort style)
- Selected lint rules: B, E, F, I, SIM, UP
- E501 (line too long) is ignored

## Documentation
- Document command-line scripts with module docstrings
- Add short comments before complex SQL or Cypher operations
- Keep commit messages imperative and ≤72 chars

## Project Structure
```
knowledge_extraction/
├── ie/                    # Information extraction subsystem
├── memory_agent/          # FastAPI microservice
│   └── tools/            # Neo4j-backed retrieval tools
├── deduplicate/          # Fact deduplication
│   ├── similarity/       # MinHash LSH and embeddings
│   └── llm/              # LLM-powered merge decisions
├── scripts/              # Utility scripts
├── tests/                # Test files mirror module structure
├── data_structures/      # Data structure definitions
├── schema.sql            # SQLite schema
├── ingest.cql            # Neo4j setup
└── *.py                  # Entry point scripts
```

## Test Structure
- Use `pytest` with test files mirroring module structure
- Test file naming: `test_<module>.py`
- Fixtures in `conftest.py`
- Mock external dependencies (llama-server, Neo4j)
