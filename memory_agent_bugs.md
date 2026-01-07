# Memory Agent Bug Report

**Date:** 2026-01-06
**Module:** `memory_agent/`
**Reviewer:** Claude Code

---

## Bug 1: Duration Timing Overwrite in `execute_tool`

**Severity:** HIGH
**File:** `memory_agent/agent.py`
**Line:** ~328

### Description

In the `execute_tool` function, the `duration_ms` for successful tool executions is recalculated **after** the retry loop exits, overwriting the correct timing that was measured inside the loop.

### Code

```python
for attempt in range(1, max_retries + 2):
    attempts = attempt
    start = time.perf_counter()
    try:
        result = tool(tool_input)
        duration_ms = int((time.perf_counter() - start) * 1000)  # Correct value
        break
    except (ToolError, ValidationError, Exception) as exc:
        ...

# Success path - BUG HERE:
facts = normalize_to_facts(tool_name, result)
...
retrieved.extend(meaningful_facts)
tool_calls = list(state.get("tool_calls", []))
duration_ms = int((time.perf_counter() - start) * 1000)  # Overwrites with wrong value!
tool_calls.append({
    ...
    "duration_ms": duration_ms,
    ...
})
```

### Impact

- Duration metrics for successful tool calls are inflated
- Includes time spent on `normalize_to_facts()`, list operations, and other post-processing
- Makes performance analysis and debugging unreliable

### Fix

Remove the redundant `duration_ms` recalculation on line ~328. The value computed inside the `try` block (immediately after `tool(tool_input)`) is correct and should be preserved.

```python
# Remove this line:
duration_ms = int((time.perf_counter() - start) * 1000)
```

---

## Bug 2: `retrieved_messages` Never Populated

**Severity:** MEDIUM
**File:** `memory_agent/agent.py`
**Lines:** 57, 103, 438-523

### Description

The `AgentState` TypedDict defines a `retrieved_messages` field of type `list[SemanticSearchMessageResult]`, and the debug output attempts to serialize it. However, the `synthesize` function never populates this field - it only stores formatted string representations in `formatted_messages`.

### Code

**State definition (`memory_agent/state.py:29`):**
```python
retrieved_messages: list[SemanticSearchMessageResult]
```

**Initialization (`memory_agent/agent.py:57`):**
```python
initial_state: AgentState = {
    ...
    "retrieved_messages": [],  # Initialized but never filled
    ...
}
```

**Debug output (`memory_agent/agent.py:103`):**
```python
if debug_mode:
    result["debug"] = {
        ...
        "retrieved_messages": [msg.model_dump() for msg in final_state.get("retrieved_messages", [])],
    }
```

**Synthesize function (`memory_agent/agent.py:486-489`):**
```python
result = tool(search_input.model_dump())
formatted_messages = format_messages(result.results)  # Only strings stored
# retrieved_messages is never updated!
```

### Impact

- Debug mode provides incomplete information
- Raw message search results (with scores, metadata) are lost
- Developers cannot inspect the actual `SemanticSearchMessageResult` objects

### Fix

Update the `synthesize` function to also store raw results:

```python
result = tool(search_input.model_dump())
formatted_messages = format_messages(result.results)

# Add: Store raw results for debugging
retrieved_messages_raw = list(state.get("retrieved_messages", []))
retrieved_messages_raw.extend(result.results)

return {
    ...
    "formatted_messages": formatted_messages[:max_messages],
    "retrieved_messages": retrieved_messages_raw,  # Add this
    ...
}
```

---

## Issue 3: Redundant Stop Condition Checks

**Severity:** LOW (Code Quality)
**File:** `memory_agent/agent.py`
**Lines:** 545-556, 559-572

### Description

Two functions check overlapping stop conditions at different points in the workflow:

| Condition | `should_continue` | `evaluate_next_step` |
|-----------|-------------------|----------------------|
| `iteration >= max_iterations` | Yes | Yes |
| `len(retrieved_facts) >= max_facts` | Yes | Yes |
| `should_stop_evaluation.should_continue is False` | Yes | Yes |
| `pending_tool is None` | Yes | No |
| `detect_tool_loop()` | No | Yes |
| `goal_accomplished` | No | Yes |

### Workflow Position

- `should_continue`: Called after `plan_queries`, before `execute_tool`
- `evaluate_next_step`: Called after `execute_tool`, before next `plan_queries`

### Impact

- Redundant checks add minor overhead
- Logic is split across two functions, making it harder to understand the full stopping criteria
- Changes to stopping logic must be synchronized in both places

### Recommendation

Consider consolidating into a single decision function or clearly documenting why each check exists in its specific location.

---

## Issue 4: Silent Logging Failures

**Severity:** LOW
**File:** `memory_agent/request_logger.py`
**Lines:** 51-69, 88-106, 122-140

### Description

All `RequestLogger` methods catch exceptions and only emit warnings:

```python
def log_request_start(...) -> None:
    try:
        ...
    except Exception as exc:
        logger.warning("Failed to log request start for %s: %s", request_id, exc)

def log_request_complete(...) -> None:
    try:
        ...
    except Exception as exc:
        logger.warning("Failed to log request completion for %s: %s", request_id, exc)

def log_request_error(...) -> None:
    try:
        ...
    except Exception as exc:
        logger.warning("Failed to log request error for %s: %s", request_id, exc)
```

### Impact

- Audit trail may be incomplete without operators noticing
- Database issues (disk full, permissions, corruption) are silently ignored
- Compliance requirements may not be met if logging is critical

### Recommendation

Consider:
1. Adding metrics for logging failures
2. Optionally propagating errors in strict mode
3. Implementing a fallback logging mechanism (e.g., file-based)

---

## Summary

| Bug | Severity | Status | Effort to Fix |
|-----|----------|--------|---------------|
| Duration timing overwrite | HIGH | Open | Low (1 line) |
| `retrieved_messages` not populated | MEDIUM | Open | Low (~5 lines) |
| Redundant stop conditions | LOW | Open | Medium (refactor) |
| Silent logging failures | LOW | Open | Medium (design decision) |
