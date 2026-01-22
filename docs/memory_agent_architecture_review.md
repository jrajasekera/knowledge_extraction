# Memory Agent Architecture Review

**Date:** 2026-01-06
**Module:** `memory_agent/`
**Reviewer:** Claude Code

---

## Executive Summary

The memory_agent implements a LangGraph-based agentic workflow for retrieving contextual facts and messages from a knowledge graph. While functional, the architecture has several design issues that impact performance, maintainability, and scalability. This document analyzes the current design and proposes alternatives.

---

## Current Architecture

### Workflow State Machine

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ┌──────────────────┐                                                   │
│  │ analyze_         │                                                   │
│  │ conversation     │ ──────────────────────────────────────────┐       │
│  └──────────────────┘                                           │       │
│           │                                                     │       │
│           ▼                                                     │       │
│  ┌──────────────────┐    ┌─────────────┐                        │       │
│  │  plan_queries    │───▶│should_      │──"finish"──────────────┼───┐   │
│  └──────────────────┘    │continue     │                        │   │   │
│           ▲              └─────────────┘                        │   │   │
│           │                     │                               │   │   │
│           │                "continue"                           │   │   │
│           │                     ▼                               │   │   │
│           │              ┌──────────────────┐                   │   │   │
│           │              │  execute_tool    │                   │   │   │
│           │              └──────────────────┘                   │   │   │
│           │                     │                               │   │   │
│           │                     ▼                               │   │   │
│           │              ┌──────────────────┐                   │   │   │
│           │              │evaluate_progress │                   │   │   │
│           │              └──────────────────┘                   │   │   │
│           │                     │                               │   │   │
│           │              ┌─────────────┐                        │   │   │
│           └──"continue"──│evaluate_    │──"finish"──────────────┼───┤   │
│                          │next_step    │                        │   │   │
│                          └─────────────┘                        │   │   │
│                                                                 │   │   │
│                                                                 ▼   ▼   │
│                                                          ┌────────────┐ │
│                                                          │ synthesize │ │
│                                                          └────────────┘ │
│                                                                 │       │
│                                                                 ▼       │
│                                                              [END]      │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Inventory

| Component | Location | Purpose |
|-----------|----------|---------|
| `MemoryAgent` | `agent.py:26-104` | Main orchestrator |
| `create_memory_agent_graph` | `agent.py:107-542` | LangGraph workflow builder |
| `LLMClient` | `llm.py` | LLM interaction layer |
| `SemanticSearchFactsTool` | `tools/semantic_search.py` | Hybrid fact retrieval |
| `SemanticSearchMessagesTool` | `tools/semantic_search_messages.py` | Hybrid message retrieval |
| `EmbeddingProvider` | `embeddings.py` | Sentence transformers wrapper |
| `AgentState` | `state.py` | TypedDict state container |

### LLM Call Points

The agent makes **6+ LLM calls** per request:

1. `extract_goal_from_conversation` - Initial goal extraction
2. `aplan_tool_usage` - Tool selection (per iteration)
3. `extract_fact_search_queries` - Query generation for facts
4. `should_continue_searching` - Stop decision (per iteration)
5. `extract_message_search_queries` - Query generation for messages
6. `generate_context_summary` - Final synthesis

---

## Architectural Issues

### Issue 1: Over-Engineered for Two Tools

**Problem:** The agentic loop with LLM-based tool selection is designed for scenarios with many tools. With only 2 tools (`semantic_search_facts`, `semantic_search_messages`), this is overkill.

**Evidence:**
```python
# build_toolkit only registers 2 tools
tools: Dict[str, ToolBase] = {
    "semantic_search_facts": SemanticSearchFactsTool(context),
    "semantic_search_messages": SemanticSearchMessagesTool(context),
}
```

**Impact:**
- Unnecessary LLM calls for tool selection
- Added latency (~500ms-2s per LLM call)
- Complex code path for simple decisions

### Issue 2: Asymmetric Message/Fact Retrieval

**Problem:** Facts are retrieved iteratively in the main loop, but messages are only retrieved once in `synthesize` after fact retrieval completes.

**Evidence:**
```python
# In synthesize (agent.py:438-523)
# Messages retrieved AFTER facts, with no opportunity for iteration
if max_messages > 0:
    tool = tools.get("semantic_search_messages")
    # ... single retrieval, no refinement loop
```

**Impact:**
- Message context can't inform fact retrieval
- No iterative refinement for messages
- Inconsistent treatment of two data sources

### Issue 3: Excessive LLM Call Overhead

**Problem:** Each iteration requires multiple LLM calls, creating latency and cost overhead.

**Typical Request Flow:**
```
Request
  └─ LLM: extract_goal (1 call)
     └─ Iteration 1
        ├─ LLM: plan_tool_usage (1 call)
        ├─ LLM: extract_fact_search_queries (1 call)
        ├─ Tool: semantic_search_facts
        └─ LLM: should_continue_searching (1 call)
     └─ Iteration 2 (if continues)
        └─ ... (3 more LLM calls)
     └─ Synthesize
        ├─ LLM: extract_message_search_queries (1 call)
        └─ LLM: generate_context_summary (1 call)

Minimum LLM calls: 6
Typical LLM calls: 8-12
```

**Impact:**
- Response times of 10-30+ seconds
- High API costs
- Increased failure surface (any LLM call can fail)

### Issue 4: Goal is Static

**Problem:** The goal is extracted once at the start and never updated based on retrieved context.

**Evidence:**
```python
async def analyze_conversation(state: AgentState) -> AgentState:
    # Goal extracted once, never refined
    goal = await llm.extract_goal_from_conversation(state["conversation"])
    return {"current_goal": goal, ...}
```

**Impact:**
- Can't pivot based on what's found
- Initial misinterpretation persists through entire workflow

### Issue 5: State Accumulation Without Pruning

**Problem:** `retrieved_facts` grows unbounded during iterations with no deduplication until final output.

**Evidence:**
```python
# In execute_tool
retrieved = list(state.get("retrieved_facts", []))
retrieved.extend(meaningful_facts)  # Always append, never dedupe
```

**Impact:**
- Memory grows with iterations
- Duplicate facts in intermediate state
- Final deduplication is wasteful

### Issue 6: Redundant Decision Functions

**Problem:** Two functions (`should_continue`, `evaluate_next_step`) check overlapping conditions.

**Evidence:**
```python
# should_continue (agent.py:545-556)
def should_continue(state: AgentState) -> str:
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        return "finish"
    if len(state.get("retrieved_facts", [])) >= state.get("max_facts", 1):
        return "finish"
    # ...

# evaluate_next_step (agent.py:559-572)
def evaluate_next_step(state: AgentState) -> str:
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        return "finish"  # Same check!
    if len(state.get("retrieved_facts", [])) >= state.get("max_facts", 1):
        return "finish"  # Same check!
    # ...
```

**Impact:**
- Confusing logic split across functions
- Changes must be synchronized

### Issue 7: Heuristics are Underutilized

**Problem:** `determine_tool_from_goal` has heuristic logic that's mostly bypassed when LLM is available.

**Evidence:**
```python
# Heuristics exist but LLM takes precedence
if llm and llm.is_available:
    llm_result = await llm.aplan_tool_usage(...)  # LLM decides
    # Heuristics only used for fallback or parameter filling
```

**Impact:**
- Wasted code complexity
- Could use simpler deterministic logic

---

## Proposed Alternatives

### Option A: Parallel Retrieval Pipeline (Recommended)

**Eliminate the agentic loop entirely.** For a system with 2 retrieval tools, a simpler pipeline is more appropriate:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  Request                                                        │
│     │                                                           │
│     ▼                                                           │
│  ┌──────────────────────┐                                       │
│  │ Extract Search       │  (1 LLM call)                         │
│  │ Queries              │                                       │
│  └──────────────────────┘                                       │
│     │                                                           │
│     ├────────────────────────────┐                              │
│     ▼                            ▼                              │
│  ┌──────────────┐         ┌──────────────┐                      │
│  │ Search Facts │         │ Search Msgs  │  (parallel)          │
│  └──────────────┘         └──────────────┘                      │
│     │                            │                              │
│     └────────────┬───────────────┘                              │
│                  ▼                                              │
│  ┌──────────────────────┐                                       │
│  │ Merge, Dedupe, Rank  │                                       │
│  └──────────────────────┘                                       │
│                  │                                              │
│                  ▼                                              │
│  ┌──────────────────────┐                                       │
│  │ Generate Summary     │  (1 LLM call)                         │
│  └──────────────────────┘                                       │
│                  │                                              │
│                  ▼                                              │
│              Response                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Benefits:**
- 2 LLM calls instead of 6-12
- Parallel retrieval reduces latency
- Simpler code, easier to maintain
- Deterministic behavior

**Implementation Sketch:**
```python
class SimpleMemoryAgent:
    async def run(self, request: RetrievalRequest) -> dict:
        # 1. Extract queries (single LLM call)
        queries = await self.llm.extract_search_queries(request.messages)

        # 2. Parallel retrieval
        facts_task = asyncio.create_task(self.search_facts(queries))
        msgs_task = asyncio.create_task(self.search_messages(queries))
        facts, messages = await asyncio.gather(facts_task, msgs_task)

        # 3. Merge and rank
        combined = self.merge_and_rank(facts, messages)

        # 4. Generate summary (single LLM call)
        summary = await self.llm.generate_summary(request.messages, combined)

        return {"facts": combined.facts, "messages": combined.messages, "summary": summary}
```

### Option B: Two-Phase Retrieval

**Phase 1:** Broad retrieval with low threshold (high recall)
**Phase 2:** LLM-based filtering and re-ranking (high precision)

```
Request
   │
   ▼
┌─────────────────────┐
│ Phase 1: Broad      │  - Low similarity threshold (0.3)
│ Retrieval           │  - Retrieve 100+ candidates
└─────────────────────┘
   │
   ▼
┌─────────────────────┐
│ Phase 2: LLM        │  - Score relevance to query
│ Re-ranking          │  - Filter to top-k
└─────────────────────┘
   │
   ▼
Response
```

**Benefits:**
- Better precision through LLM re-ranking
- Single retrieval pass (no iteration)
- Explicit recall/precision tradeoff

### Option C: Streaming RAG Architecture

For lowest latency, implement streaming:

```
Request ──▶ [Embed] ──▶ [Retrieve] ──▶ [Stream LLM Response]
                              │
                              └─▶ (facts injected as context)
```

**Benefits:**
- Time-to-first-token is fast
- User sees response building
- Modern UX pattern

---

## Recommendations

### Short-Term (Low Effort)

1. **Remove redundant decision functions** - Consolidate `should_continue` and `evaluate_next_step`
2. **Add deduplication in execute_tool** - Prevent duplicate facts from accumulating
3. **Make message retrieval part of the main loop** - Or at least make it optional to skip

### Medium-Term (Medium Effort)

4. **Implement Option A (Parallel Pipeline)** - Simplify to a non-agentic flow
5. **Reduce LLM calls** - Combine query extraction for facts and messages into one call
6. **Add caching for embeddings** - Avoid re-embedding similar queries

### Long-Term (High Effort)

7. **Implement streaming responses** - Modern UX with faster perceived latency
8. **Add LLM re-ranking phase** - For better precision on large result sets
9. **Consider tool expansion** - If more tools are added, the agentic loop becomes justified

---

## Metrics to Track

Before/after any changes, measure:

| Metric | Current (Est.) | Target |
|--------|----------------|--------|
| Avg response time | 15-30s | <5s |
| LLM calls per request | 6-12 | 2-3 |
| P95 latency | 45s+ | <10s |
| Cost per request | High | 50% reduction |

---

## Conclusion

The current architecture is over-engineered for its use case. The agentic loop with LLM-based tool selection adds complexity and latency without proportional benefit when there are only 2 tools. A simpler parallel retrieval pipeline would be faster, cheaper, and easier to maintain.

The key insight is that **tool selection doesn't need an LLM when you only have 2 tools** - you can run both in parallel and merge results. The LLM's value is in query extraction and final synthesis, not in deciding which tool to call.
