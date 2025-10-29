# Implementation Plan: Add Reciprocal Rank Fusion (RRF) to semantic_search.py

## Overview
This document provides a detailed plan to add Reciprocal Rank Fusion (RRF) support to `memory_agent/tools/semantic_search.py`, following the pattern already implemented in `memory_agent/tools/semantic_search_messages.py`.

## Background

### Current State of semantic_search.py
- Processes multiple queries (1-10) and embeds each one
- Uses dictionary deduplication by `(person_id, fact_type, fact_object, relationship_type)`
- Keeps only the highest scoring result for each unique fact
- Applies similarity threshold filtering
- Sorts by similarity score descending and limits results

### Current State of semantic_search_messages.py (Reference Implementation)
- Uses `MessageOccurrence` dataclass to track multiple observations of the same message
- Tracks both scores and ranks for each query that returned a result
- Supports three fusion methods: `rrf`, `score_sum`, `score_max`
- Uses RRF formula: `sum(1.0 / (RRF_K + rank))` where RRF_K = 60
- Supports `multi_query_boost` parameter for score_max method

### Why RRF Matters
When a fact appears in results for multiple queries, it's a strong signal that the fact is highly relevant. RRF (Reciprocal Rank Fusion) is a proven method for combining rankings from multiple queries that:
- Rewards facts that appear in multiple query results
- Takes into account the rank position in each result list
- Is more robust than simple score averaging
- Helps surface facts that are broadly relevant vs. narrowly matched

## Implementation Steps

### Step 1: Add Constants and Imports

**File:** `memory_agent/tools/semantic_search.py`

**Location:** After existing imports, before class definitions

**Add:**
```python
from dataclasses import dataclass, field
from typing import Literal

# RRF constants
RRF_K = 60
DEFAULT_FUSION_METHOD: Literal["rrf", "score_sum", "score_max"] = "rrf"
DEFAULT_MULTI_QUERY_BOOST = 0.0
```

**Rationale:** These constants match the semantic_search_messages.py implementation and provide sensible defaults.

---

### Step 2: Create FactOccurrence Dataclass

**File:** `memory_agent/tools/semantic_search.py`

**Location:** After constants, before `SemanticSearchInput` class

**Add:**
```python
@dataclass
class FactOccurrence:
    """Track per-fact observations across multiple semantic queries."""

    properties: dict[str, Any]
    best_score: float
    query_scores: dict[int, float] = field(default_factory=dict)
    query_ranks: dict[int, int] = field(default_factory=dict)

    def add_observation(self, query_idx: int, score: float, rank: int, properties: dict[str, Any]) -> None:
        """Record an observation for this fact from a specific query.
        
        Args:
            query_idx: Index of the query that returned this fact (1-based)
            score: Similarity score for this observation
            rank: Rank position in the result list (1-based)
            properties: Node properties from Neo4j
        """
        # Update query_scores: keep the highest score if we see this fact multiple times in same query
        existing_score = self.query_scores.get(query_idx)
        if existing_score is None or score > existing_score:
            self.query_scores[query_idx] = score
        
        # Update query_ranks: keep the best (lowest) rank for this query
        if query_idx not in self.query_ranks or rank < self.query_ranks[query_idx]:
            self.query_ranks[query_idx] = rank
        
        # Update best_score and properties if this is the best observation overall
        if score > self.best_score:
            self.best_score = score
            self.properties = properties
```

**Rationale:** This dataclass mirrors `MessageOccurrence` and provides the infrastructure to track multiple observations of the same fact across different queries.

---

### Step 3: Update SemanticSearchInput Model

**File:** `memory_agent/tools/semantic_search.py`

**Location:** Modify existing `SemanticSearchInput` class

**Current:**
```python
class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    queries: list[str] = Field(min_length=1, max_length=10)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
```

**Change to:**
```python
class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    queries: list[str] = Field(min_length=1, max_length=10)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    fusion_method: Literal["rrf", "score_sum", "score_max"] = Field(default=DEFAULT_FUSION_METHOD)
    multi_query_boost: float = Field(default=DEFAULT_MULTI_QUERY_BOOST, ge=0.0, le=1.0)
```

**Rationale:** Adds the same fusion parameters as semantic_search_messages.py for consistency.

---

### Step 4: Update SemanticSearchResult Model

**File:** `memory_agent/tools/semantic_search.py`

**Location:** Modify existing `SemanticSearchResult` class

**Current:**
```python
class SemanticSearchResult(BaseModel):
    """Output entry for semantic_search_facts."""

    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    confidence: float | None = None
    evidence: list[str | dict[str, Any]] = Field(default_factory=list)
```

**Change to:**
```python
class SemanticSearchResult(BaseModel):
    """Output entry for semantic_search_facts."""

    person_id: str
    person_name: str
    fact_type: str
    fact_object: str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    similarity_score: float
    confidence: float | None = None
    evidence: list[str | dict[str, Any]] = Field(default_factory=list)
    query_scores: dict[int, float] | None = None
    appeared_in_query_count: int | None = None
```

**Rationale:** Adds metadata about which queries returned this fact and how many, useful for debugging and transparency.

---

### Step 5: Add Score Calculation Method to SemanticSearchFactsTool

**File:** `memory_agent/tools/semantic_search.py`

**Location:** Add as a static method in `SemanticSearchFactsTool` class, before the `run` method

**Add:**
```python
@staticmethod
def _calculate_combined_score(
    occurrence: FactOccurrence,
    fusion_method: str,
    multi_query_boost: float,
) -> float:
    """Calculate combined score from multiple query observations.
    
    Args:
        occurrence: FactOccurrence with scores and ranks from multiple queries
        fusion_method: Method to use for combining scores (rrf, score_sum, score_max)
        multi_query_boost: Boost factor for facts appearing in multiple queries (score_max only)
    
    Returns:
        Combined score for ranking
    """
    if not occurrence.query_scores:
        return occurrence.best_score

    if fusion_method == "rrf":
        # Reciprocal Rank Fusion: sum of 1/(K+rank) for each query
        return sum(1.0 / (RRF_K + rank) for rank in occurrence.query_ranks.values())

    if fusion_method == "score_sum":
        # Simple sum of all scores
        return sum(occurrence.query_scores.values())

    if fusion_method == "score_max":
        # Max score with boost for appearing in multiple queries
        max_score = max(occurrence.query_scores.values())
        query_count = len(occurrence.query_scores)
        return max_score * (1.0 + multi_query_boost * (query_count - 1))

    # Fallback to best_score if method is unrecognized
    logger.warning("Unrecognized fusion method: %s, using best_score", fusion_method)
    return occurrence.best_score
```

**Rationale:** Centralizes score calculation logic, matching semantic_search_messages.py implementation.

---

### Step 6: Add Result Building Helper Method

**File:** `memory_agent/tools/semantic_search.py`

**Location:** Add as a method in `SemanticSearchFactsTool` class, after `_calculate_combined_score`

**Add:**
```python
def _build_result(
    self,
    properties: dict[str, Any],
    score: float,
    evidence: list[str | dict[str, Any]],
    *,
    query_scores: dict[int, float] | None = None,
) -> SemanticSearchResult:
    """Build a SemanticSearchResult from node properties and calculated score.
    
    Args:
        properties: Node properties from Neo4j
        score: Combined similarity score
        evidence: Evidence list (may include content snippets)
        query_scores: Optional mapping of query index to score
    
    Returns:
        SemanticSearchResult with all fields populated
    """
    # Parse attributes (may be JSON string or dict)
    attributes_raw = properties.get("attributes")
    attributes = {}
    if isinstance(attributes_raw, str):
        try:
            import json
            attributes = json.loads(attributes_raw)
        except json.JSONDecodeError:
            attributes = {}
    elif isinstance(attributes_raw, dict):
        attributes = attributes_raw

    appeared_in_query_count = len(query_scores) if query_scores else None

    return SemanticSearchResult(
        person_id=properties.get("person_id", ""),
        person_name=properties.get("person_name", properties.get("person_id", "")),
        fact_type=properties.get("fact_type", ""),
        fact_object=properties.get("fact_object"),
        attributes=attributes,
        similarity_score=score,
        confidence=properties.get("confidence"),
        evidence=evidence,
        query_scores=query_scores,
        appeared_in_query_count=appeared_in_query_count,
    )
```

**Rationale:** Encapsulates result building logic for cleaner code and easier maintenance.

---

### Step 7: Refactor the run() Method

**File:** `memory_agent/tools/semantic_search.py`

**Location:** Replace the existing `run` method in `SemanticSearchFactsTool` class

**Current Structure:**
```python
def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
    logger.info(...)
    results_dict: dict[...] = {}
    total_raw_results = 0
    # ... process each query
    # ... add to results_dict
    unique_results = list(results_dict.values())
    unique_results.sort(...)
    final_results = unique_results[:input_data.limit]
    logger.info(...)
    return SemanticSearchOutput(...)
```

**New Structure:**
```python
def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
    """Execute semantic search with RRF-based multi-query fusion.
    
    Process:
    1. Generate embeddings for each query
    2. Execute vector searches
    3. Track observations in FactOccurrence objects
    4. Apply similarity threshold filtering
    5. Calculate combined scores using selected fusion method
    6. Sort and limit results
    
    Args:
        input_data: Search parameters including queries and fusion method
    
    Returns:
        SemanticSearchOutput with fused results
    """
    logger.info(
        "semantic_search_facts called: queries=%r, limit=%d, similarity_threshold=%.2f, "
        "fusion_method=%s, index=%s",
        input_data.queries,
        input_data.limit,
        input_data.similarity_threshold,
        input_data.fusion_method,
        self.index_name,
    )

    # Track fact occurrences across queries
    occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}
    
    # Metrics for logging
    total_raw_results = 0
    total_filtered_by_threshold = 0
    total_missing_node = 0
    queries_processed = 0

    # Process each query
    for query_idx, query in enumerate(input_data.queries, 1):
        logger.info("Processing query %d/%d: %r", query_idx, len(input_data.queries), query)

        # Generate embedding for this query
        embedding = self.embeddings.embed_single(query)
        if not embedding:
            logger.warning("Failed to generate embedding for query %d: %r", query_idx, query)
            continue

        logger.debug("Generated embedding vector of length %d for query %d", len(embedding), query_idx)

        # Execute vector query
        try:
            rows = run_vector_query(
                self.context,
                self.index_name,
                embedding,
                input_data.limit,
                None,
            )
            logger.info("Query %d returned %d raw results from index %s", query_idx, len(rows), self.index_name)
            total_raw_results += len(rows)
            queries_processed += 1
        except ToolError as exc:
            logger.warning("Query %d failed: %s", query_idx, exc)
            continue

        # Process results from this query
        filtered_by_threshold = 0
        missing_node = 0
        added_new = 0
        updated_existing = 0

        for rank, row in enumerate(rows, start=1):
            node = row.get("node")
            score = row.get("score", 0.0)
            evidence_with_content = row.get("evidence_with_content", [])

            # Apply similarity threshold
            if input_data.similarity_threshold and score < input_data.similarity_threshold:
                filtered_by_threshold += 1
                logger.debug(
                    "Query %d: Filtered result with score %.3f (below threshold %.2f)",
                    query_idx,
                    score,
                    input_data.similarity_threshold,
                )
                continue

            if not node:
                missing_node += 1
                logger.debug("Query %d: Skipping row with missing node", query_idx)
                continue

            # Parse node properties
            properties = dict(node)
            
            # Create deduplication key
            person_id = properties.get("person_id", "")
            fact_type = properties.get("fact_type", "")
            fact_object = properties.get("fact_object")
            
            # Extract relationship_type from attributes for proper deduplication
            attributes_raw = properties.get("attributes")
            relationship_type = None
            if isinstance(attributes_raw, str):
                try:
                    import json
                    attributes = json.loads(attributes_raw)
                    relationship_type = str(attributes.get("relationship_type")) if attributes.get("relationship_type") else None
                except json.JSONDecodeError:
                    pass
            elif isinstance(attributes_raw, dict):
                relationship_type = str(attributes_raw.get("relationship_type")) if attributes_raw.get("relationship_type") else None
            
            dedup_key = (person_id, fact_type, fact_object, relationship_type)

            # Use evidence_with_content if available, fallback to evidence IDs
            evidence = evidence_with_content if evidence_with_content else properties.get("evidence") or []

            # Track or update occurrence
            occurrence = occurrences.get(dedup_key)
            if occurrence is None:
                occurrence = FactOccurrence(properties=properties, best_score=score)
                occurrence.add_observation(query_idx, score, rank, properties)
                occurrences[dedup_key] = occurrence
                added_new += 1
                logger.debug(
                    "Query %d: Added new fact: person=%s, fact_type=%s, object=%s, score=%.3f, rank=%d",
                    query_idx,
                    properties.get("person_name", person_id),
                    fact_type,
                    fact_object,
                    score,
                    rank,
                )
            else:
                occurrence.add_observation(query_idx, score, rank, properties)
                updated_existing += 1
                logger.debug(
                    "Query %d: Updated existing fact: person=%s, fact_type=%s, object=%s, score=%.3f, rank=%d",
                    query_idx,
                    properties.get("person_name", person_id),
                    fact_type,
                    fact_object,
                    score,
                    rank,
                )

        total_filtered_by_threshold += filtered_by_threshold
        total_missing_node += missing_node

        logger.info(
            "Query %d summary: raw=%d, filtered=%d, missing_node=%d, added_new=%d, updated_existing=%d",
            query_idx,
            len(rows),
            filtered_by_threshold,
            missing_node,
            added_new,
            updated_existing,
        )

    logger.info("Total unique facts before fusion: %d", len(occurrences))

    # Calculate combined scores and build results
    results_with_scores: list[SemanticSearchResult] = []
    for dedup_key, occurrence in occurrences.items():
        combined_score = self._calculate_combined_score(
            occurrence,
            input_data.fusion_method,
            input_data.multi_query_boost,
        )
        
        # Use evidence_with_content from the best observation
        evidence = occurrence.properties.get("evidence") or []
        
        result = self._build_result(
            occurrence.properties,
            combined_score,
            evidence,
            query_scores=dict(occurrence.query_scores),
        )
        results_with_scores.append(result)

    # Log fusion statistics
    facts_in_multiple_queries = sum(1 for occ in occurrences.values() if len(occ.query_scores) > 1)
    avg_queries_per_fact = (
        sum(len(occ.query_scores) for occ in occurrences.values()) / len(occurrences)
        if occurrences
        else 0.0
    )

    logger.info(
        "Fact fusion summary: total_unique_facts=%d, facts_in_multiple_queries=%d, "
        "avg_queries_per_fact=%.2f, fusion_method=%s",
        len(occurrences),
        facts_in_multiple_queries,
        avg_queries_per_fact,
        input_data.fusion_method,
    )

    # Sort by combined score descending
    ordered = sorted(results_with_scores, key=lambda r: r.similarity_score, reverse=True)

    # Apply final limit
    final_results = ordered[: input_data.limit]

    # Log top results for debugging
    for result in final_results[:5]:
        occurrence_key = (result.person_id, result.fact_type, result.fact_object, 
                         result.attributes.get("relationship_type") if result.attributes else None)
        occurrence = occurrences.get(occurrence_key)
        if occurrence:
            logger.debug(
                "Top result after fusion: person=%s, fact_type=%s, combined_score=%.3f, "
                "appeared_in=%d, query_scores=%s",
                result.person_name,
                result.fact_type,
                result.similarity_score,
                len(occurrence.query_scores),
                occurrence.query_scores,
            )

    logger.info(
        "semantic_search_facts completed: queries=%d, queries_processed=%d, total_raw_results=%d, "
        "total_filtered=%d, total_missing_node=%d, unique_facts=%d, final_results=%d",
        len(input_data.queries),
        queries_processed,
        total_raw_results,
        total_filtered_by_threshold,
        total_missing_node,
        len(ordered),
        len(final_results),
    )

    return SemanticSearchOutput(queries=input_data.queries, results=final_results)
```

**Rationale:** This is a complete rewrite following the semantic_search_messages.py pattern, with proper RRF support, detailed logging, and cleaner structure.

---

### Step 8: Update LLM Integration in llm.py

**File:** `memory_agent/llm.py`

**Location:** In the `_fallback_tool_selection` method

**Current code (around line 600):**
```python
if "semantic_search_facts" in available_tools:
    # Use goal text as a single query in fallback mode
    queries = [goal_text] if goal_text else [conversation_text]
    # Over-fetch for LLM quality filtering (3x multiplier, capped at tool maximum)
    retrieval_limit = min(state.get("max_facts", 10) * 3, 50)
    return {
        "tool_name": "semantic_search_facts",
        "parameters": {"queries": queries, "limit": retrieval_limit},
        "reasoning": "Fallback heuristic defaulted to semantic search.",
        "confidence": "low",
        "should_stop": False,
        "stop_reason": None,
    }
```

**Change to:**
```python
if "semantic_search_facts" in available_tools:
    # Use goal text as a single query in fallback mode
    queries = [goal_text] if goal_text else [conversation_text]
    # Over-fetch for LLM quality filtering (3x multiplier, capped at tool maximum)
    retrieval_limit = min(state.get("max_facts", 10) * 3, 50)
    return {
        "tool_name": "semantic_search_facts",
        "parameters": {
            "queries": queries,
            "limit": retrieval_limit,
            "fusion_method": "rrf",  # Use RRF by default
        },
        "reasoning": "Fallback heuristic defaulted to semantic search with RRF.",
        "confidence": "low",
        "should_stop": False,
        "stop_reason": None,
    }
```

**Rationale:** Ensures LLM integration uses the new RRF capability by default.

---

### Step 9: Update Tool Prompt Info

**File:** `memory_agent/llm.py`

**Location:** In the `TOOL_PROMPT_INFO` dictionary

**Current:**
```python
"semantic_search_facts": {
    "description": "Perform semantic search using multiple keywords or key phrases to find relevant facts across all types.",
    "use_when": "The goal requires broad discovery across fact types, or when you need to search for concepts using multiple related terms.",
    "inputs": "queries (required, list of 1-5 keywords/phrases), limit (optional, will be auto-adjusted for quality filtering), similarity_threshold (optional)",
    "example": "Use when asked 'Who has startup experience?' - extract keywords like ['startup', 'founder', 'entrepreneur', 'early-stage company'] and search with all of them to maximize recall.",
},
```

**Change to:**
```python
"semantic_search_facts": {
    "description": "Perform semantic search using multiple keywords or key phrases to find relevant facts across all types. Uses Reciprocal Rank Fusion (RRF) to intelligently combine results from multiple queries.",
    "use_when": "The goal requires broad discovery across fact types, or when you need to search for concepts using multiple related terms. RRF will boost facts that appear in multiple query results.",
    "inputs": "queries (required, list of 1-5 keywords/phrases), limit (optional), similarity_threshold (optional), fusion_method (optional: rrf/score_sum/score_max)",
    "example": "Use when asked 'Who has startup experience?' - extract keywords like ['startup', 'founder', 'entrepreneur', 'early-stage company']. RRF will surface people who match multiple terms higher in rankings.",
},
```

**Rationale:** Updates documentation to reflect RRF capability and guide the LLM to use multiple queries effectively.

---

## Testing Plan

### Unit Tests to Add

Create `tests/test_semantic_search_rrf.py` with the following test cases:

1. **test_fact_occurrence_tracking**
   - Verify FactOccurrence correctly tracks multiple observations
   - Test score and rank updates
   - Test best_score tracking

2. **test_rrf_calculation**
   - Verify RRF formula: 1/(K+rank) summation
   - Test with different rank combinations
   - Compare with expected values

3. **test_score_sum_calculation**
   - Verify simple sum of scores
   - Test with multiple query observations

4. **test_score_max_calculation**
   - Verify max score selection
   - Test multi_query_boost application
   - Test with different boost values (0.0, 0.5, 1.0)

5. **test_single_query_backward_compatibility**
   - Verify that single-query searches still work
   - Ensure results match old behavior when queries=[single_query]

6. **test_multi_query_fusion**
   - Test with 2-5 queries
   - Verify facts appearing in multiple queries rank higher
   - Test with different fusion methods

7. **test_deduplication_with_rrf**
   - Verify same fact from multiple queries is deduplicated
   - Verify dedup_key includes relationship_type
   - Test that best observations are preserved

### Integration Tests to Add

Create `tests/integration/test_semantic_search_integration.py`:

1. **test_real_neo4j_rrf_search**
   - Set up test Neo4j database with known facts
   - Execute multi-query search
   - Verify RRF scores are calculated correctly
   - Verify facts in multiple results rank higher

2. **test_agent_workflow_with_rrf**
   - Run full agent workflow
   - Verify semantic_search_facts is called with multiple queries
   - Verify RRF fusion works end-to-end

### Manual Testing Checklist

- [ ] Test with single query (backward compatibility)
- [ ] Test with 2 queries where facts overlap
- [ ] Test with 5 queries with varying overlap
- [ ] Test each fusion method (rrf, score_sum, score_max)
- [ ] Test with different similarity thresholds
- [ ] Test with multi_query_boost values
- [ ] Compare rankings with old implementation
- [ ] Verify logging output is informative
- [ ] Test edge cases (empty queries, no results, all filtered)

---

## Migration Strategy

### Phase 1: Implementation (Week 1)
1. Implement FactOccurrence class
2. Add new input parameters with defaults
3. Refactor run() method with RRF logic
4. Update helper methods

### Phase 2: Testing (Week 1-2)
1. Write and run unit tests
2. Perform integration testing
3. Manual testing with real data
4. Performance benchmarking

### Phase 3: Deployment (Week 2)
1. Deploy to staging environment
2. Monitor logs for issues
3. A/B test if possible (compare rankings)
4. Deploy to production with monitoring

### Phase 4: Optimization (Week 3)
1. Analyze performance metrics
2. Tune RRF_K value if needed
3. Adjust default fusion_method based on results
4. Update documentation

---

## Backward Compatibility

### Guaranteed Compatibility
- Single-query searches will work identically to before
- Default parameters maintain current behavior
- All existing code calling semantic_search_facts will continue to work

### Default Behavior Changes
- When multiple queries are provided, RRF is now used by default
- Results may be ranked differently with multiple queries (this is the desired improvement)

### Migration for Existing Callers
No changes required. However, callers can benefit from:
```python
# Old style (still works)
semantic_search_facts(queries=["machine learning"], limit=10)

# New style (takes advantage of RRF)
semantic_search_facts(
    queries=["machine learning", "ML", "AI", "neural networks"],
    limit=10,
    fusion_method="rrf"
)
```

---

## Performance Considerations

### Expected Impact
- **Latency**: Minimal increase (< 5ms) due to additional bookkeeping
- **Memory**: Slight increase to store FactOccurrence objects
- **Accuracy**: Significant improvement in ranking quality for multi-query searches

### Optimization Opportunities
- Cache embeddings if same query appears multiple times
- Batch process rankings for better performance
- Consider parallelizing query execution (future enhancement)

---

## Monitoring and Metrics

### Key Metrics to Track
1. **facts_in_multiple_queries**: How many facts appear in results of >1 query
2. **avg_queries_per_fact**: Average number of queries returning each fact
3. **fusion_method_usage**: Distribution of fusion methods used
4. **ranking_changes**: Compare rankings with/without RRF

### Log Messages to Watch
- "Fact fusion summary" - shows RRF statistics
- "Top result after fusion" - shows winning facts and their scores
- Query processing summaries for each query

---

## Documentation Updates

### Files to Update
1. **README.md** - Add section on RRF and fusion methods
2. **API documentation** - Document new input parameters
3. **User guide** - Explain when to use multiple queries
4. **Developer guide** - Explain RRF implementation details

### Example Documentation
```markdown
## Semantic Search with RRF

The semantic_search_facts tool now supports Reciprocal Rank Fusion (RRF) for 
combining results from multiple queries. This is particularly useful when searching
for concepts that can be expressed in different ways.

### Fusion Methods

- **rrf** (default): Uses Reciprocal Rank Fusion to combine rankings
- **score_sum**: Sums similarity scores across queries
- **score_max**: Uses maximum score with optional multi-query boost

### Example Usage

```python
# Search for startup experience using multiple terms
result = semantic_search_facts(
    queries=["startup founder", "entrepreneur", "early-stage company"],
    limit=10,
    fusion_method="rrf"
)
```

Facts that match multiple queries will rank higher, improving precision.
```

---

## Success Criteria

### Implementation Complete When:
- [ ] All code changes implemented per plan
- [ ] Unit tests pass with >90% coverage
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Code reviewed and approved
- [ ] Documentation updated

### Deployment Successful When:
- [ ] No errors in production logs
- [ ] Ranking quality improved (manual evaluation)
- [ ] Performance within acceptable bounds
- [ ] Monitoring dashboards show expected metrics

---

## Rollback Plan

### If Issues Arise:
1. **Minor issues**: Fix forward with hotfix
2. **Major issues**: Revert to previous version

### Rollback Steps:
1. Revert commit with changes
2. Deploy previous version
3. Monitor for stability
4. Analyze what went wrong
5. Fix in development branch
6. Redeploy when ready

### Code Preserved:
Keep old run() method in comments or separate branch for reference during rollback.

---

## Questions for Discussion

Before implementation, clarify:

1. **Default fusion_method**: Should it be "rrf" or keep current behavior initially?
2. **RRF_K value**: Should we use 60 like messages, or tune differently?
3. **Multi-query strategy**: Should agent be updated to generate multiple queries by default?
4. **Performance budget**: What's acceptable latency increase?
5. **A/B testing**: Should we run parallel implementations for comparison?

---

## References

- Original semantic_search_messages.py implementation (lines 1-700+)
- RRF paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- Neo4j vector search documentation
- LangGraph agent documentation

---

## Appendix: Code Comparison

### Before (Current Implementation)
```python
# Simple dictionary with best score only
results_dict: dict[tuple[str, str, str | None, str | None], SemanticSearchResult] = {}

# Process each query
for query in queries:
    rows = run_vector_query(...)
    for row in rows:
        dedup_key = (person_id, fact_type, fact_object, relationship_type)
        if dedup_key not in results_dict or score > results_dict[dedup_key].similarity_score:
            results_dict[dedup_key] = result

# Sort and return
results = list(results_dict.values())
results.sort(key=lambda r: r.similarity_score, reverse=True)
```

### After (RRF Implementation)
```python
# Track occurrences with ranks
occurrences: dict[tuple, FactOccurrence] = {}

# Process each query
for query_idx, query in enumerate(queries, 1):
    rows = run_vector_query(...)
    for rank, row in enumerate(rows, 1):
        dedup_key = (person_id, fact_type, fact_object, relationship_type)
        if dedup_key not in occurrences:
            occurrences[dedup_key] = FactOccurrence(properties, score)
        occurrences[dedup_key].add_observation(query_idx, score, rank, properties)

# Calculate fused scores
results = []
for occurrence in occurrences.values():
    combined_score = _calculate_combined_score(occurrence, fusion_method, boost)
    result = _build_result(occurrence.properties, combined_score, ...)
    results.append(result)

# Sort by combined score
results.sort(key=lambda r: r.similarity_score, reverse=True)
```

Key differences:
- Tracking ranks in addition to scores
- Multiple observations per fact
- Fusion calculation before sorting
- Richer logging and metadata
