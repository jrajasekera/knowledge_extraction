# Task Completion Checklist

When completing a task, run through these steps:

## 1. Code Quality
```bash
# Format code
uv run ruff format .

# Lint and auto-fix
uv run ruff check --fix .

# Type check
uv run pyright
```

## 2. Testing
```bash
# Run all tests
uv run pytest

# Run with coverage if needed
uv run coverage run -m pytest
uv run coverage report
```

## 3. Pre-commit Checks
The project uses pre-commit hooks that run:
- `ruff check` (linting)
- `ruff format` (formatting)

If pre-commit is installed, these run automatically on commit.
Otherwise, run manually before committing.

## 4. Common Pitfalls to Check
- [ ] Foreign key violations: Ensure `PRAGMA foreign_keys = ON;` where needed
- [ ] Neo4j driver: Each `session.run()` accepts only one statement
- [ ] Missing embeddings: Run `scripts/embed_facts.py --cleanup` after loading facts
- [ ] Database locks: If concurrent access needed, ensure WAL mode is enabled
- [ ] LLM connection: Verify llama-server is running before IE operations

## 5. If Modifying Database Schema
- Update `schema.sql`
- Update relevant model classes
- Consider migration implications

## 6. If Adding New Dependencies
- Add to `pyproject.toml`
- Run `uv sync`

## 7. Commit Guidelines
- Imperative mood, ≤72 characters
- Examples: "Add fact deduplication", "Fix null confidence handling"
