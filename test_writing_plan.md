# Test Writing Plan to Reach 65% Coverage

## Goal
Increase line+branch coverage from ~41% to at least 65% without changing production behavior.

## Ground Rules
- Add tests only; no production logic changes unless required for testability.
- Use `pytest` with existing fixtures; mock external services (Neo4j, LLMs) via adapters in `ie/client.py`.
- Target high-leverage modules with large statement counts and low coverage first.
- Prefer small, deterministic unit tests over broad integration tests.

## Coverage Baseline (Key Low-Coverage Targets)
Prioritize these modules based on size and current coverage gaps:
1. `memory_agent/llm.py` (large, ~41% coverage)
2. `memory_agent/agent.py` (very large, ~10% coverage)
3. `facts_to_graph.py` (large, ~17% coverage)
4. `deduplicate/core.py` (large, ~32% coverage)
5. `memory_agent/message_embedding_pipeline.py` (~37% coverage)
6. `ie/runner.py` (~59% coverage)
7. CLI entry points: `run_pipeline.py`, `run_ie.py`, `loader.py`, `import_discord_json.py` (0% coverage)

## Phased Plan

### Phase 1: Quick Wins (Target +10-12%)
Focus on modules with small units and straightforward branches.
- `ie/windowing.py`: add tests for window sizing, overlap, edge cases (empty input, 1 item, exact boundary).
- `data_structures/ingestion/timestamps.py`: add tests for parsing and failure modes.
- `memory_agent/serialization.py`: add tests for serialize/deserialize round trips and invalid inputs.
- `memory_agent/tools/base.py`: validate tool registration, input validation, and error paths.

### Phase 2: Medium Complexity (Target +8-10%)
Increase coverage in core but testable utilities.
- `memory_agent/message_embedding_pipeline.py`: mock embedding provider; test batching, error handling, and output shapes.
- `memory_agent/embedding_pipeline.py`: cover tokenization/normalization guards and retry branches.
- `deduplicate/partitioning.py`: deterministic tests for grouping logic with small inputs.
- `deduplicate/similarity/grouping.py`: add tests for minhash/grouping thresholds and edge conditions.

### Phase 3: High-Impact Modules (Target +10-12%)
Move into large modules with many branches; use mocks to isolate external dependencies.
- `memory_agent/llm.py`: mock `ie/client.py` adapters; test prompt construction, retry logic, and error paths.
- `memory_agent/agent.py`: test agent state transitions with a minimal fake LLM client and in-memory store.
- `deduplicate/core.py`: test parser + merging workflows with small fixtures and stubbed LLM outputs.

### Phase 4: CLI/Entry Points (Target +4-6%)
Cover argument parsing and basic orchestration with temporary files.
- `run_ie.py`, `run_pipeline.py`, `loader.py`, `import_discord_json.py`: test CLI parsing and that top-level functions are invoked with expected params (mock heavy operations).

## Test Artifacts & Fixtures
- Add fixtures in `tests/fixtures/` for small JSON and SQLite seeds.
- Use `pytest` fixtures to isolate temp directories and in-memory DBs.
- Add helper factories for LLM responses and Neo4j mocks.

## Execution Plan
1. Add tests in Phase 1; run `uv run pytest` and `uv run coverage report -m`.
2. Add Phase 2 tests; validate coverage increase and keep runtime <10s.
3. Add Phase 3 tests; use mocks to avoid network/DB.
4. Add Phase 4 tests; ensure CLI coverage without slow end-to-end runs.

## Verification
- Run `uv run coverage run -m pytest` followed by `uv run coverage report -m`.
- Verify total coverage >= 65%.
- Optional: open `htmlcov/index.html` to identify remaining gaps.

## Success Criteria
- Overall coverage >= 65%.
- All tests deterministic and run locally within a few seconds.
- No production code changes required.
