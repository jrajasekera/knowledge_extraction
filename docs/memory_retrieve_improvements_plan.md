# POST /api/memory/retrieve Improvement Plan

This document captures four targeted improvements to increase retrieval quality for `POST /api/memory/retrieve`.

## 1. Fix Early-Stop Behavior

### Problem
The agent can stop too early after a single tool call that returns any non-zero number of facts. This improves latency but hurts recall on multi-facet questions.

### Goal
Continue searching until the agent has enough *new and relevant* coverage, not just any result.

### Proposed Changes
1. Replace the current binary "last call returned results" stop signal with a novelty-aware stop policy.
2. Track per-iteration "new facts added" and "coverage gain" metrics in state.
3. Require at least one of these before stopping:
1. Iteration limit reached.
2. No meaningful novelty for `N` consecutive iterations.
3. LLM stop decision is high confidence and supported by metrics.
4. Add a minimum exploration floor (for example 2 iterations unless the first pass is very strong).

### Implementation Notes
1. Extend `AgentState` with fields such as:
1. `new_facts_last_iteration`
2. `novelty_streak_without_gain`
3. `coverage_score`
2. Compute novelty after each tool execution by comparing current retrieved facts to a canonical key set.
3. Update `evaluate_progress` and `evaluate_next_step` to use these metrics.
4. Keep behavior configurable with env-backed thresholds.

### Suggested Config Knobs
1. `EARLY_STOP_MIN_ITERATIONS` (default: 2)
2. `NOVELTY_MIN_NEW_FACTS` (default: 1)
3. `NOVELTY_PATIENCE` (default: 2)
4. `STOP_CONFIDENCE_REQUIRED` (default: `"high"`)

### Validation
1. Unit tests where first pass returns partial results and second pass adds important new facts.
2. Regression tests proving the agent still stops promptly when repeated iterations return no novelty.
3. Compare before/after on labeled prompts:
1. Recall@K
2. Average iterations used
3. Latency impact

## 2. Improve Fallback Query Generation

### Problem
When LLM query extraction fails or is empty, fallback is often a single broad query string. This limits retrieval coverage and makes results sensitive to phrasing.

### Goal
Generate multiple, diverse fallback queries that preserve high recall even without LLM support.

### Proposed Changes
1. Build a deterministic fallback query expansion module.
2. Produce a small query set (for example 6-12 queries) from:
1. Noun phrases and entities in the latest message and recent context.
2. Synonym/related-term expansion for skills, roles, organizations, and domains.
3. Short keyword forms plus medium-length phrase forms.
3. Deduplicate and rank fallback queries before execution.
4. Log which fallback strategy produced each query for later tuning.

### Implementation Notes
1. Add a helper module such as `memory_agent/query_fallback.py`.
2. Inputs:
1. Last message
2. Recent message window
3. Current goal
4. Previously tried queries
3. Outputs:
1. Ordered list of unique queries
2. Optional per-query metadata (`source=entity|synonym|keyword`)
4. Integrate in both:
1. `determine_tool_from_goal`
2. The `"queries"` fallback in `plan_queries`

### Guardrails
1. Cap fallback query count and token length per query.
2. Remove stopwords and trivial reformulations.
3. Avoid repeating previously failed queries unless context changed.

### Validation
1. Unit tests for query expansion on:
1. Skill lookup prompts
2. Role/organization prompts
3. Pronoun-heavy follow-ups
2. Offline replay to measure uplift in unique relevant facts found when LLM is unavailable.

## 3. Make Ranking and Merging Novelty-Aware Across Iterations

### Problem
Current ranking is strong within one tool call, but cross-iteration merging can over-prioritize repeated variants of the same fact while under-prioritizing truly new evidence.

### Goal
Rank final results using both relevance and novelty so each iteration contributes complementary value.

### Proposed Changes
1. Introduce a cross-iteration aggregation layer before final synthesis.
2. Track a canonical fact identity and evidence-level identity separately.
3. Score facts with a composite function:
1. Relevance score (hybrid similarity/fusion)
2. Novelty bonus (new fact key or new evidence)
3. Recency signal (if timestamps exist)
4. Query diversity signal (matched by multiple distinct queries)
4. Re-rank globally at synthesis time, then trim to `max_facts`.

### Implementation Notes
1. Maintain two keys:
1. `fact_key`: stable semantic identity of a fact.
2. `evidence_key`: source-level uniqueness (`source_id` or timestamp+snippet hash).
2. During merge:
1. Preserve top evidence snippets per fact.
2. Count how many *new* evidence items were added this iteration.
3. Feed novelty counters into stopping logic and confidence scoring.
4. Optionally expose debug metadata:
1. `novelty_added`
2. `supporting_evidence_count`
3. `query_coverage_count`

### Validation
1. Test that duplicate facts from repeated queries do not crowd out novel facts.
2. Test that a fact with multiple independent evidence sources ranks above a single weak source.
3. Compare NDCG/Recall across multi-iteration replay sets.

## 4. Calibrate Confidence with Retrieval Signals

### Problem
Final confidence can be disconnected from actual retrieval quality because it relies heavily on tool-call success and simple count thresholds.

### Goal
Produce confidence labels that better reflect relevance, evidence strength, and coverage completeness.

### Proposed Changes
1. Replace the current rule set with a calibrated scoring model (still deterministic).
2. Build a scalar confidence score in `[0,1]` from weighted signals:
1. Mean and top-k similarity score
2. Evidence richness (sources/snippets per fact)
3. Novelty and coverage gained across iterations
4. Query diversity and retrieval consistency
5. Tool error rate and retry behavior
3. Map scalar score to labels:
1. `low`
2. `medium`
3. `high`
4. Include raw score and contributing components in metadata for debugging.

### Implementation Notes
1. Add a `compute_confidence_score(...) -> float` helper.
2. Keep `compute_confidence(...) -> ConfidenceLevel` as a thin mapper for API compatibility.
3. Store score breakdown in logs or debug endpoint response.
4. Keep thresholds configurable so they can be tuned on replay data.

### Suggested Initial Weights
1. Relevance quality: 0.35
2. Evidence strength: 0.20
3. Novelty/coverage: 0.25
4. Execution reliability: 0.20

### Validation
1. Build a small labeled set of requests with expected confidence levels.
2. Compute confusion matrix for label mapping before and after calibration.
3. Verify that low-quality broad matches are less likely to be labeled `high`.

## Recommended Rollout Order

1. Early-stop fix and novelty tracking foundation.
2. Fallback query generation improvements.
3. Cross-iteration ranking/merge enhancements.
4. Confidence calibration using the new novelty/relevance signals.

## Observability Checklist

1. Add per-request counters:
1. Queries generated
2. Queries executed
3. New facts per iteration
4. Duplicates merged
5. Final confidence score
2. Add structured logs for stop decisions with explicit reason codes.
3. Add a replay script to compare old/new behavior on a fixed prompt set.
