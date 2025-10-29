# Semantic Message Search Scoring Enhancement Specification

## Overview

This specification describes changes to the `SemanticSearchMessagesTool` to implement a multi-query fusion scoring system. The new system rewards messages that appear in the results of multiple queries, improving the relevance of search results when using multiple related queries.

## Current Behavior

The existing implementation:
1. Processes each query independently
2. For each query, retrieves results and filters by similarity threshold
3. Deduplicates messages across queries, keeping only the highest similarity score
4. Sorts by similarity score and returns top `limit` results

**Limitation**: A message that appears in multiple query results gets no additional credit for its multi-query relevance, potentially missing messages that are broadly relevant to the overall search intent.

## Proposed Behavior

The enhanced implementation will:
1. Process each query independently and retrieve top `n` results per query
2. Track all occurrences of each message across queries
3. Calculate a combined score that considers both:
   - Individual similarity scores from each query
   - A boost for appearing in multiple queries (query frequency)
4. Sort by the combined score and return top `limit` results

**Benefit**: Messages relevant to multiple aspects of the search intent will rank higher, improving overall result quality for complex, multi-faceted searches.

## Scoring Algorithm

### Parameters

Add new configuration parameters to `SemanticSearchMessagesInput`:

```python
results_per_query: int = Field(default=20, ge=1, le=100)
    # Number of results to fetch per query before fusion
    
fusion_method: str = Field(default="rrf", pattern="^(rrf|score_sum|score_max)$")
    # Scoring fusion method:
    # - "rrf": Reciprocal Rank Fusion
    # - "score_sum": Sum of similarity scores
    # - "score_max": Maximum similarity score with multi-query bonus

multi_query_boost: float = Field(default=0.1, ge=0.0, le=1.0)
    # Bonus multiplier for appearing in multiple queries (only for score_max method)
```

### Fusion Methods

#### 1. Reciprocal Rank Fusion (RRF) - Recommended Default

**Formula**: 
```
RRF_score = Σ(1 / (k + rank_i))
```

Where:
- `k` is a constant (typically 60) to prevent division by zero and reduce impact of high ranks
- `rank_i` is the rank (1-indexed) of the message in query `i`'s results
- Sum is over all queries where the message appears

**Properties**:
- Rank-based, reducing impact of similarity score variations
- Naturally rewards messages appearing in multiple queries
- Well-studied and proven effective in information retrieval
- Scale-independent (no need to normalize scores across different embedding models)

**Example**:
```
Message A appears in:
- Query 1: rank 3, score 0.85
- Query 2: rank 1, score 0.92

RRF score = 1/(60+3) + 1/(60+1) = 0.0159 + 0.0164 = 0.0323

Message B appears in:
- Query 1: rank 1, score 0.88

RRF score = 1/(60+1) = 0.0164

Message A ranks higher despite Message B having a good score in one query.
```

#### 2. Score Sum

**Formula**:
```
combined_score = Σ(similarity_score_i)
```

Where sum is over all queries where the message appears.

**Properties**:
- Simple and intuitive
- Directly sums similarity scores
- Messages in multiple queries naturally get higher scores
- Can be sensitive to score scale variations

**Example**:
```
Message A: appears in queries 1 (0.85) and 2 (0.78)
Combined score = 0.85 + 0.78 = 1.63

Message B: appears in query 1 (0.95)
Combined score = 0.95

Message A ranks higher due to appearing in multiple queries.
```

#### 3. Score Max with Multi-Query Bonus

**Formula**:
```
combined_score = max_score × (1 + multi_query_boost × (query_count - 1))
```

Where:
- `max_score` is the highest similarity score across all queries
- `query_count` is the number of queries the message appears in
- `multi_query_boost` is a configurable parameter (default 0.1)

**Properties**:
- Preserves the best individual score
- Applies incremental boost for each additional query
- Configurable boost strength
- More conservative than score_sum

**Example** (with boost=0.1):
```
Message A: appears in queries 1 (0.85) and 2 (0.78)
Combined score = 0.85 × (1 + 0.1 × 1) = 0.935

Message B: appears in query 1 (0.95)
Combined score = 0.95 × (1 + 0.1 × 0) = 0.95

Message B still ranks higher, but the gap is reduced.
```

## Implementation Details

### Data Structures

```python
# Track all occurrences of each message
class MessageOccurrence:
    message_id: str
    properties: dict[str, Any]
    query_scores: dict[int, float]  # {query_index: similarity_score}
    query_ranks: dict[int, int]     # {query_index: rank (1-indexed)}
    
# Collection for all messages
message_occurrences: dict[str, MessageOccurrence] = {}
```

### Modified Algorithm Flow

1. **Query Processing Loop**:
   ```python
   for query_idx, query in enumerate(input_data.queries):
       # Generate embedding
       embedding = self.embeddings.embed_single(query)
       
       # Fetch results (use results_per_query, not limit)
       rows = run_vector_query(
           self.context,
           self.index_name,
           embedding,
           input_data.results_per_query,  # Changed from limit
           filters,
           include_evidence=False,
       )
       
       # Filter by similarity threshold
       filtered_rows = [
           row for row in rows 
           if float(row.get("score", 0.0)) >= input_data.similarity_threshold
       ]
       
       # Track rank and score for each message
       for rank, row in enumerate(filtered_rows, start=1):
           node = row.get("node")
           score = float(row.get("score", 0.0))
           
           # Apply time filters here
           # ...
           
           message_id = str(properties.get("message_id"))
           
           if message_id not in message_occurrences:
               message_occurrences[message_id] = MessageOccurrence(
                   message_id=message_id,
                   properties=properties,
                   query_scores={},
                   query_ranks={}
               )
           
           occurrence = message_occurrences[message_id]
           occurrence.query_scores[query_idx] = score
           occurrence.query_ranks[query_idx] = rank
   ```

2. **Score Fusion**:
   ```python
   def calculate_combined_score(
       occurrence: MessageOccurrence,
       fusion_method: str,
       multi_query_boost: float,
       rrf_k: int = 60
   ) -> float:
       if fusion_method == "rrf":
           return sum(
               1.0 / (rrf_k + rank)
               for rank in occurrence.query_ranks.values()
           )
       
       elif fusion_method == "score_sum":
           return sum(occurrence.query_scores.values())
       
       elif fusion_method == "score_max":
           max_score = max(occurrence.query_scores.values())
           query_count = len(occurrence.query_scores)
           return max_score * (1 + multi_query_boost * (query_count - 1))
       
       else:
           raise ValueError(f"Unknown fusion method: {fusion_method}")
   
   # Calculate combined scores for all messages
   results_with_scores = []
   for occurrence in message_occurrences.values():
       combined_score = calculate_combined_score(
           occurrence,
           input_data.fusion_method,
           input_data.multi_query_boost
       )
       
       result = self._build_result(occurrence.properties, combined_score)
       results_with_scores.append(result)
   ```

3. **Final Ranking**:
   ```python
   # Sort by combined score (descending)
   sorted_results = sorted(
       results_with_scores,
       key=lambda r: r.similarity_score,  # Now contains combined score
       reverse=True
   )
   
   # Return top limit results
   final_results = sorted_results[:input_data.limit]
   ```

### Logging Enhancements

Add detailed logging for the fusion process:

```python
logger.info(
    "Message fusion summary: "
    "total_unique_messages=%d, "
    "messages_in_multiple_queries=%d, "
    "avg_queries_per_message=%.2f",
    len(message_occurrences),
    sum(1 for occ in message_occurrences.values() if len(occ.query_scores) > 1),
    sum(len(occ.query_scores) for occ in message_occurrences.values()) / len(message_occurrences)
)

# Optional: Log top messages with multi-query presence
for result in final_results[:5]:
    occurrence = message_occurrences[result.message_id]
    logger.debug(
        "Top result: message_id=%s, combined_score=%.3f, "
        "appeared_in=%d queries, scores=%s",
        result.message_id,
        result.similarity_score,
        len(occurrence.query_scores),
        occurrence.query_scores
    )
```

## Updated Result Model

Consider adding metadata about the fusion process to the result:

```python
class SemanticSearchMessageResult(BaseModel):
    # ... existing fields ...
    
    similarity_score: float  # Now contains the combined/fused score
    
    # New optional fields for debugging/transparency
    query_scores: dict[int, float] | None = None  # Individual scores per query
    appeared_in_query_count: int | None = None    # Number of queries this appeared in
```

Make these fields optional and populate them only when a debug/verbose flag is set.

## Configuration Examples

### Example 1: High Recall with RRF (Recommended)
```python
input_data = SemanticSearchMessagesInput(
    queries=["machine learning", "neural networks", "deep learning"],
    limit=10,
    results_per_query=30,
    fusion_method="rrf",
    similarity_threshold=0.5
)
```
Use case: Want messages relevant to any aspect of ML/DL, with preference for broadly relevant messages.

### Example 2: Conservative with Score Max
```python
input_data = SemanticSearchMessagesInput(
    queries=["bug report", "error in production"],
    limit=10,
    results_per_query=20,
    fusion_method="score_max",
    multi_query_boost=0.15,
    similarity_threshold=0.7
)
```
Use case: Want high-quality matches but with slight boost for messages matching both queries.

### Example 3: Aggressive Multi-Query Fusion
```python
input_data = SemanticSearchMessagesInput(
    queries=["API design", "REST endpoints", "documentation"],
    limit=10,
    results_per_query=25,
    fusion_method="score_sum",
    similarity_threshold=0.6
)
```
Use case: Want to heavily favor messages that discuss multiple related topics.

## Edge Cases and Considerations

### 1. Single Query
When only one query is provided, all fusion methods should produce identical results (ordered by similarity score). No special handling needed.

### 2. Empty Results from Some Queries
If a query returns no results after filtering, it simply contributes nothing to any message's combined score. This is handled naturally by the algorithm.

### 3. Very Different Query Specificity
If one query is very specific (few results) and another is broad (many results), RRF handles this well due to its rank-based nature. Score-based methods may need normalization, but this is acceptable for a first implementation.

### 4. Score Scale Differences
RRF is immune to this. For score-based methods, assume all queries use the same embedding model with consistent score scales.

### 5. Tie Breaking
When combined scores are identical, maintain stable sorting by:
- Secondary sort by timestamp (most recent first), or
- Lexicographic sort by message_id

```python
sorted_results = sorted(
    results_with_scores,
    key=lambda r: (r.similarity_score, r.timestamp or ""),
    reverse=True
)
```

## Performance Considerations

### Memory Impact
- Old: Stores up to `limit` messages total
- New: Stores up to `results_per_query × num_queries` messages before fusion
- Recommendation: Default `results_per_query=20` keeps memory reasonable

### Computational Impact
- Additional overhead is O(n) for score fusion where n = unique messages
- Negligible compared to embedding generation and vector search
- No performance concerns for typical use cases

### Vector Query Optimization
The code already fetches results per query. The main change is:
- Use `results_per_query` instead of `limit` in the vector query call
- This may fetch slightly more results but improves fusion quality

## Testing Recommendations

### Unit Tests

1. **Test RRF Calculation**
   ```python
   def test_rrf_single_query():
       # Message in one query at rank 1
       # Should get score of 1/(60+1)
       
   def test_rrf_multiple_queries():
       # Message in two queries at ranks 1 and 3
       # Should get sum of 1/(60+1) + 1/(60+3)
   ```

2. **Test Score Sum**
   ```python
   def test_score_sum_additive():
       # Message with scores 0.8 and 0.7 should get 1.5
   ```

3. **Test Score Max with Boost**
   ```python
   def test_score_max_boost():
       # Message with scores [0.9, 0.8] and boost=0.1
       # Should get 0.9 × (1 + 0.1 × 1) = 0.99
   ```

4. **Test Edge Cases**
   - Single query (all methods equivalent)
   - Empty query results
   - All messages unique across queries
   - All messages appear in all queries

### Integration Tests

1. **Real Query Comparison**
   - Run same queries with old and new system
   - Verify that multi-query relevant messages rank higher

2. **Regression Test**
   - Single query should produce identical results to old system (when fusion_method="score_max" and boost=0)

### Performance Tests

1. **Memory Usage**
   - Measure memory with various `results_per_query` values
   - Ensure no memory leaks with large result sets

2. **Latency**
   - Measure end-to-end latency for 1, 5, and 10 queries
   - Fusion overhead should be <5% of total query time

## Migration Strategy

### Backward Compatibility

1. **Default Behavior**: Set `fusion_method="score_max"` and `multi_query_boost=0.0` to approximate current behavior
2. **Feature Flag**: Consider adding a feature flag to enable/disable fusion during rollout
3. **A/B Testing**: Test RRF vs current system with real queries to measure improvement

### Rollout Plan

1. **Phase 1**: Deploy with conservative defaults (score_max, boost=0.0)
2. **Phase 2**: Enable RRF as default after validation
3. **Phase 3**: Deprecate single-query mode optimizations if no longer needed

## Documentation Updates

Update the following documentation:

1. **API Documentation**: 
   - Document new parameters: `results_per_query`, `fusion_method`, `multi_query_boost`
   - Explain when to use each fusion method
   - Provide usage examples

2. **Developer Guide**:
   - Explain the multi-query fusion concept
   - Provide guidance on choosing `results_per_query` values
   - Describe score interpretation

3. **Migration Guide**:
   - For existing users, explain behavior changes
   - Provide recommended settings for common use cases

## Future Enhancements

Consider these potential future improvements:

1. **Query Weights**: Allow assigning different weights to different queries
2. **Score Normalization**: Normalize scores across queries before fusion for more consistent behavior
3. **Diversity Boosting**: Penalize very similar messages to increase result diversity
4. **Learned Fusion**: Use ML to learn optimal fusion weights based on user feedback
5. **Explain API**: Return detailed explanation of why each message ranked where it did

## Summary

This specification introduces a multi-query fusion scoring system that:
- ✅ Rewards messages appearing in multiple query results
- ✅ Provides three fusion methods (RRF, score_sum, score_max)
- ✅ Maintains backward compatibility through configuration
- ✅ Adds minimal computational overhead
- ✅ Improves result quality for complex, multi-faceted searches

**Recommended Defaults**:
- `fusion_method`: "rrf" (most robust)
- `results_per_query`: 20 (balances recall and memory)
- `similarity_threshold`: 0.6 (unchanged)
- `multi_query_boost`: 0.1 (only relevant for score_max)
