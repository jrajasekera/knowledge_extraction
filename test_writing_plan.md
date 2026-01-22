# Test Writing Plan to Reach 80% Coverage

## Goal
Increase line+branch coverage from ~41% to at least 80% without changing production behavior.

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

### Phase 1: Quick Wins (Target +12-15%)
Focus on small units and straightforward branches to lift the baseline quickly.
- `ie/windowing.py`: test window sizing/overlap; empty input; exact boundary; odd/even window sizes.
- `data_structures/ingestion/timestamps.py`: parse valid strings; invalid formats; timezone handling.
- `memory_agent/serialization.py`: round-trip serialize/deserialize; invalid payloads; missing fields.
- `memory_agent/tools/base.py`: tool registration; duplicate tool names; input validation; error path.

Test setup specifics:
- Use `pytest.mark.parametrize` for boundary cases.
- No mocks required.

### Phase 2: Medium Complexity (Target +10-12%)
Increase coverage in core but testable utilities.
- `memory_agent/message_embedding_pipeline.py`: mock embedding provider; test batching, error handling, and output shapes.
- `memory_agent/embedding_pipeline.py`: cover tokenization/normalization guards and retry branches.
- `deduplicate/partitioning.py`: deterministic tests for grouping logic with small inputs.
- `deduplicate/similarity/grouping.py`: add tests for minhash/grouping thresholds and edge conditions.

Test setup specifics:
- Mock embedding provider with a simple callable returning deterministic vectors.
- Use `pytest` fixtures to generate small message batches (3-10 items).
- For retry branches, inject a provider that fails once then succeeds.

### Phase 3: High-Impact Modules (Target +15-18%)
Move into large modules with many branches; use mocks to isolate external dependencies.
- `memory_agent/llm.py`: mock `ie/client.py` adapters; test prompt construction, retry logic, and error paths.
- `memory_agent/agent.py`: test agent state transitions with a minimal fake LLM client and in-memory store.
- `deduplicate/core.py`: test parser + merging workflows with small fixtures and stubbed LLM outputs.

Test setup specifics:
- Mock `ie.client` to return fixed responses and to raise transient errors.
- Use an in-memory SQLite DB or temp files for any storage (avoid real disk).
- Stub any time-dependent code with `freezegun`-like patterns using `monkeypatch` on `datetime` if needed.

### Phase 4: CLI/Entry Points (Target +6-8%)
Cover argument parsing and basic orchestration with temporary files.
- `run_ie.py`, `run_pipeline.py`, `loader.py`, `import_discord_json.py`: test CLI parsing and that top-level functions are invoked with expected params (mock heavy operations).

Test setup specifics:
- Use `pytest` `monkeypatch` to replace heavy functions (Neo4j, LLM) with no-ops.
- Use `tmp_path` for any file args and inject minimal fixture content.

### Phase 5: Graph + Runner Coverage (Target +8-10%)
Attack the large, low-coverage orchestration and graph materialization paths.
- `facts_to_graph.py`: unit-test query builders and per-entity merge functions using a fake Neo4j session.
- `ie/runner.py`: test windowing orchestration, error handling, and retry logic with a fake LLM client.

Test setup specifics:
- Create a `FakeSession` that records `run()` calls and returns minimal rows; assert cypher and params.
- Monkeypatch `neo4j.GraphDatabase.driver` to return a fake driver with `session()` context manager.
- Use a tiny in-memory SQLite DB seeded with a handful of rows for facts/messages.

## Test Artifacts & Fixtures
- Add fixtures in `tests/fixtures/` for small JSON and SQLite seeds.
- Use `pytest` fixtures to isolate temp directories and in-memory DBs.
- Add helper factories for LLM responses and Neo4j mocks.
- Provide a minimal fake LLM response schema that matches `ie/models.py` expectations.
- Include a tiny Discord export JSON for `import_discord_json.py` CLI tests (3-5 messages).

## Execution Plan
1. Add tests in Phase 1; run `uv run pytest` and `uv run coverage report -m`.
2. Add Phase 2 tests; validate coverage increase and keep runtime <10s.
3. Add Phase 3 tests; use mocks to avoid network/DB.
4. Add Phase 4 tests; ensure CLI coverage without slow end-to-end runs.
5. Add Phase 5 tests to cover graph materialization and runner orchestration.
6. Re-run coverage; if <80%, add focused tests for the next-largest gaps (likely `facts_to_graph.py` and `memory_agent/agent.py` branches).

## Verification
- Run `uv run coverage run -m pytest` followed by `uv run coverage report -m`.
- Verify total coverage >= 80%.
- Optional: open `htmlcov/index.html` to identify remaining gaps.

## Success Criteria
- Overall coverage >= 80%.
- All tests deterministic and run locally within a few seconds.
- No production code changes required.
