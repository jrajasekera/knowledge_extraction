# Memory Agent: Iterative Refinement Improvements

**Date:** 2026-01-06
**Context:** The agent's iterative loop is designed to try different search queries when initial attempts don't find relevant information. This document identifies issues and proposes improvements to make iteration more effective.

---

## Current Iteration Flow

```
plan_queries
    │
    ├─ LLM: aplan_tool_usage (selects tool)
    │
    ├─ if previous call returned 0 results:
    │      LLM: refine_tool_parameters
    │
    ├─ LLM: extract_fact_search_queries (generates queries)
    │
    └─ return pending_tool with parameters
         │
         ▼
execute_tool
    │
    └─ runs semantic_search_facts with queries
         │
         ▼
evaluate_progress
    │
    └─ LLM: should_continue_searching
         │
         ▼
    [loop back to plan_queries or finish]
```

---

## Issues with Current Refinement

### Issue 1: Query Generation is Stateless

**Problem:** `extract_fact_search_queries()` only receives the conversation - it doesn't know what queries were already tried or what results came back.

**Location:** `memory_agent/llm.py:427-519`

```python
async def extract_fact_search_queries(
    self,
    messages: list[MessageModel],  # Only conversation!
    *,
    max_queries: int = 15,
) -> list[str]:
```

**Impact:**
- May regenerate similar queries each iteration
- Can't learn from what worked or didn't work
- No "try something different" signal

### Issue 2: Refinement Only Triggers on Zero Results

**Problem:** The `should_refine` flag only activates when a previous tool call returned 0 results.

**Location:** `memory_agent/agent.py:224-227`

```python
should_refine = any(
    call.get("name") == tool_name and call.get("result_count", 0) == 0
    for call in state.get("tool_calls", [])
)
```

**Impact:**
- Low-quality results don't trigger refinement
- If search returns 5 irrelevant facts, no refinement happens
- Binary (0 vs >0) is too coarse

### Issue 3: LLM Sees Limited Context

**Problem:** The LLM only sees the last 5 facts and last 5 tool calls when planning.

**Location:** `memory_agent/llm.py:829-845`

```python
def _summarize_retrieved_facts(self, facts: Sequence[RetrievedFact]) -> str:
    # ...
    for fact in facts[-5:]:  # Only last 5!
        summaries.append(f"- {format_fact(fact)}")

def _format_tool_history(self, tool_calls: Sequence[dict[str, Any]]) -> str:
    # ...
    for index, call in enumerate(tool_calls[-5:], start=...):  # Only last 5!
```

**Impact:**
- After many iterations, early context is lost
- Can't make decisions based on full retrieval history

### Issue 4: No Query Deduplication Across Iterations

**Problem:** There's no mechanism to prevent regenerating the same or highly similar queries.

**Impact:**
- Wastes iterations on duplicate searches
- Similarity threshold may change, but queries don't diversify

### Issue 5: Refinement Doesn't See Retrieved Content

**Problem:** `refine_tool_parameters()` receives `previous_results` which is the tool call metadata, not the actual facts retrieved.

**Location:** `memory_agent/agent.py:236-241`

```python
refinement = await llm.refine_tool_parameters(
    tool_name,
    parameters,
    conversation_context,
    [call for call in state.get("tool_calls", []) if call.get("name") == tool_name],
    # ^ This is call metadata, not retrieved facts!
)
```

**Impact:**
- Can't refine based on "we found X, but need Y"
- No content-aware refinement

---

## Proposed Improvements

### Improvement 1: Context-Aware Query Generation

Pass previous queries and results to query extraction:

```python
async def extract_fact_search_queries(
    self,
    messages: list[MessageModel],
    *,
    max_queries: int = 15,
    previous_queries: list[str] | None = None,  # NEW
    retrieved_facts: list[RetrievedFact] | None = None,  # NEW
) -> list[str]:
    """Derive diverse search queries, avoiding previous attempts."""

    # Build context about what's been tried
    tried_context = ""
    if previous_queries:
        tried_context = f"""
## Previously Tried Queries (DO NOT repeat these or similar)
{chr(10).join(f'- "{q}"' for q in previous_queries[-10:])}
"""

    found_context = ""
    if retrieved_facts:
        found_context = f"""
## Already Retrieved Facts (search for DIFFERENT angles)
{chr(10).join(f'- {format_fact(f)}' for f in retrieved_facts[-10:])}
"""

    prompt = f"""
...existing prompt...

{tried_context}

{found_context}

Generate queries that explore NEW angles not covered above.
"""
```

**Caller update in `plan_queries`:**

```python
# Collect previous queries from tool history
previous_queries = []
for call in state.get("tool_calls", []):
    if call.get("name") == "semantic_search_facts":
        queries = call.get("input", {}).get("queries", [])
        previous_queries.extend(queries)

extracted_queries = await llm.extract_fact_search_queries(
    conversation,
    max_queries=15,
    previous_queries=previous_queries,
    retrieved_facts=state.get("retrieved_facts", []),
)
```

### Improvement 2: Quality-Based Refinement Trigger

Replace binary (0 results) trigger with quality assessment:

```python
def should_refine_search(state: AgentState) -> bool:
    """Determine if search parameters need refinement."""
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        return False

    recent_fact_calls = [
        c for c in tool_calls[-3:]
        if c.get("name") == "semantic_search_facts"
    ]

    if not recent_fact_calls:
        return False

    last_call = recent_fact_calls[-1]

    # Trigger refinement if:
    # 1. Zero results
    if last_call.get("result_count", 0) == 0:
        return True

    # 2. Diminishing returns (fewer results than previous)
    if len(recent_fact_calls) >= 2:
        prev_count = recent_fact_calls[-2].get("result_count", 0)
        curr_count = last_call.get("result_count", 0)
        if curr_count < prev_count * 0.5:  # 50% drop
            return True

    # 3. Low average confidence in retrieved facts
    facts = state.get("retrieved_facts", [])
    if facts:
        recent_facts = facts[-last_call.get("result_count", 0):]
        avg_conf = sum(f.confidence or 0 for f in recent_facts) / len(recent_facts)
        if avg_conf < 0.5:
            return True

    return False
```

### Improvement 3: Explicit Query Diversification Strategy

Add a diversification mode that tries different search strategies:

```python
class SearchStrategy(Enum):
    DIRECT = "direct"           # Exact terms from conversation
    SYNONYMS = "synonyms"       # Related terms and synonyms
    BROADER = "broader"         # More general concepts
    NARROWER = "narrower"       # More specific details
    RELATED_PEOPLE = "people"   # Focus on person names
    RELATED_ORGS = "orgs"       # Focus on organizations
    TEMPORAL = "temporal"       # Time-based queries

def get_next_strategy(tool_calls: list[dict]) -> SearchStrategy:
    """Rotate through strategies to ensure diversity."""
    used_strategies = set()
    for call in tool_calls:
        if call.get("name") == "semantic_search_facts":
            strategy = call.get("input", {}).get("strategy")
            if strategy:
                used_strategies.add(strategy)

    # Return first unused strategy
    for strategy in SearchStrategy:
        if strategy.value not in used_strategies:
            return strategy

    # All tried - start over with least recently used
    return SearchStrategy.DIRECT
```

Then include strategy in the query extraction prompt:

```python
strategy = get_next_strategy(state.get("tool_calls", []))
prompt = f"""
...
## Search Strategy for This Iteration: {strategy.value}
{STRATEGY_INSTRUCTIONS[strategy]}

Generate queries following this strategy.
"""
```

### Improvement 4: Accumulated Query History in State

Track all queries tried across iterations:

```python
# Add to AgentState
class AgentState(TypedDict, total=False):
    # ... existing fields ...

    # NEW: Track query history for deduplication
    tried_queries: list[str]
    query_results_map: dict[str, int]  # query -> result count
```

Update in `execute_tool`:

```python
# After tool execution
queries_used = tool_input.get("queries", [])
tried = list(state.get("tried_queries", []))
tried.extend(queries_used)

query_map = dict(state.get("query_results_map", {}))
for q in queries_used:
    query_map[q] = len(facts)

return {
    # ... existing returns ...
    "tried_queries": tried,
    "query_results_map": query_map,
}
```

### Improvement 5: Content-Aware Refinement

Pass actual retrieved facts to refinement:

```python
async def refine_tool_parameters(
    self,
    tool_name: str,
    initial_parameters: dict[str, Any],
    conversation_context: str,
    previous_calls: Sequence[dict],
    retrieved_facts: list[RetrievedFact] | None = None,  # NEW
) -> dict[str, Any]:

    facts_context = ""
    if retrieved_facts:
        facts_context = f"""
## What We Found So Far
{chr(10).join(f'- {format_fact(f)}' for f in retrieved_facts[-10:])}

Consider: What's missing? What related topics should we explore?
"""

    prompt = f"""
...existing prompt...

{facts_context}

Suggest refined parameters that will find DIFFERENT, COMPLEMENTARY information.
"""
```

### Improvement 6: Intelligent Early Stopping

Add smarter stopping conditions based on search quality:

```python
async def should_continue_searching(self, ...) -> dict[str, Any]:
    # ... existing logic ...

    # NEW: Detect search exhaustion
    if self._detect_search_exhaustion(tool_calls, retrieved_facts):
        return {
            "should_continue": False,
            "reasoning": "Search space appears exhausted - recent queries found diminishing new information",
        }

def _detect_search_exhaustion(
    self,
    tool_calls: list[dict],
    facts: list[RetrievedFact]
) -> bool:
    """Detect when further searching is unlikely to help."""
    recent_calls = [c for c in tool_calls[-3:] if c.get("name") == "semantic_search_facts"]

    if len(recent_calls) < 3:
        return False

    # Check if last 3 calls all returned same/similar results
    result_counts = [c.get("result_count", 0) for c in recent_calls]
    if all(count == 0 for count in result_counts):
        return True  # 3 consecutive zero-result searches

    # Check for repeated fact retrieval (getting same facts)
    # Would need dedup tracking to implement properly

    return False
```

---

## Implementation Priority

| Improvement | Impact | Effort | Priority |
|-------------|--------|--------|----------|
| 1. Context-aware query generation | High | Medium | **P0** |
| 2. Quality-based refinement trigger | Medium | Low | **P1** |
| 4. Query history tracking | High | Low | **P1** |
| 5. Content-aware refinement | Medium | Medium | **P2** |
| 3. Diversification strategies | Medium | High | **P2** |
| 6. Intelligent early stopping | Low | Medium | **P3** |

---

## Quick Win: Minimum Viable Improvement

The highest-impact, lowest-effort change is to pass previous queries and retrieved facts to `extract_fact_search_queries`:

```python
# In plan_queries, before calling extract_fact_search_queries:

# Collect all previously used queries
previous_queries = []
for call in state.get("tool_calls", []):
    if call.get("name") == "semantic_search_facts":
        qs = call.get("input", {}).get("queries", [])
        previous_queries.extend(qs)

# Pass context to query extraction
extracted_queries = await llm.extract_fact_search_queries(
    conversation,
    max_queries=15,
    previous_queries=previous_queries,  # ADD THIS
    retrieved_facts=state.get("retrieved_facts", []),  # ADD THIS
)
```

And update `extract_fact_search_queries` to include these in the prompt with instructions to diversify.

This single change would make iteration significantly more effective by ensuring each iteration explores genuinely new search angles.
