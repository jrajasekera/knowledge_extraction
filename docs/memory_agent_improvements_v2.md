# Memory Agent Recall Improvements: Implementation Plan (v2)

This document provides implementation plans for two features to improve the memory agent's recall ability and context quality:

1. **Adaptive Similarity Thresholds** - Dynamically adjust search thresholds to optimize result counts
2. **Cross-Encoder Re-ranking via llama-server** - Re-rank candidates using Qwen3-Reranker-8B on a local llama-server instance

---

## Feature 1: Adaptive Similarity Thresholds

### What

Adaptive Similarity Thresholds is a search strategy that dynamically adjusts the similarity score cutoff based on result quality and quantity, rather than using a fixed threshold value.

Currently, the semantic search tools use a hardcoded similarity threshold (default 0.6). This creates two failure modes:

- **Threshold too high**: Relevant results are filtered out, leading to incomplete context
- **Threshold too low**: Irrelevant results pollute the context, reducing quality

Adaptive thresholds solve this by starting with a high threshold for precision, then progressively lowering it until a target result count is achieved, ensuring we always return enough context while prioritizing the most relevant results.

### Why

#### Current Problems

1. **Fixed threshold fails across query types**: A threshold optimal for specific queries (e.g., "Who knows Kubernetes?") performs poorly for broad queries (e.g., "Tell me about the engineering team") and vice versa.

2. **Embedding quality varies**: Some fact types embed better than others. Work history facts may cluster tightly while preference facts are more diffuse. A single threshold cannot account for this variance.

3. **Query specificity mismatch**: Highly specific queries naturally produce fewer high-similarity matches. The current system may return 2 results when 20 were requested, leaving valuable context on the table.

4. **No graceful degradation**: When the exact information isn't available, the system should return "close enough" results rather than nothing.

#### Expected Benefits

- **Improved recall**: Queries that currently return few/no results will return relevant context
- **Consistent result counts**: Users requesting N facts will receive closer to N facts
- **Better precision at top**: High-confidence results still appear first due to score-based ranking
- **Automatic adaptation**: No manual threshold tuning required per deployment or dataset

### How

#### Implementation Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Adaptive Threshold Search                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. Execute search with HIGH threshold (0.75)                   │
│                         │                                        │
│                         ▼                                        │
│  2. Count unique results after deduplication                    │
│                         │                                        │
│                         ▼                                        │
│  3. Results >= target?  ──YES──► Return results                 │
│           │                                                      │
│           NO                                                     │
│           │                                                      │
│           ▼                                                      │
│  4. Lower threshold by step (0.05)                              │
│                         │                                        │
│                         ▼                                        │
│  5. Threshold > minimum (0.35)?  ──NO──► Return best results    │
│           │                                                      │
│           YES                                                    │
│           │                                                      │
│           └──────► Go to step 1                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

#### File Changes

##### 1. New Configuration Constants

**File**: `memory_agent/tools/semantic_search.py`

Add configuration constants at the module level:

```python
# Adaptive threshold configuration
ADAPTIVE_THRESHOLD_MAX = 0.75      # Starting threshold (high precision)
ADAPTIVE_THRESHOLD_MIN = 0.35      # Floor threshold (maximum recall)
ADAPTIVE_THRESHOLD_STEP = 0.05     # Decrement per iteration
ADAPTIVE_TARGET_RATIO = 0.8        # Target 80% of requested limit
```

##### 2. Update Input Model

**File**: `memory_agent/tools/semantic_search.py`

Modify `SemanticSearchInput` to support adaptive mode:

```python
class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""

    queries: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=10, ge=1, le=50)
    
    # Change: Make similarity_threshold optional to enable adaptive mode
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    
    # New: Explicit adaptive mode toggle
    adaptive_threshold: bool = Field(default=True)
    
    # New: Adaptive mode configuration overrides
    adaptive_threshold_max: float = Field(default=ADAPTIVE_THRESHOLD_MAX, ge=0.0, le=1.0)
    adaptive_threshold_min: float = Field(default=ADAPTIVE_THRESHOLD_MIN, ge=0.0, le=1.0)
    
    fusion_method: Literal["rrf", "score_sum", "score_max"] = Field(default=DEFAULT_FUSION_METHOD)
    multi_query_boost: float = Field(default=DEFAULT_MULTI_QUERY_BOOST, ge=0.0, le=1.0)
```

##### 3. Implement Adaptive Search Method

**File**: `memory_agent/tools/semantic_search.py`

Add a new method to `SemanticSearchFactsTool`:

```python
def _search_with_adaptive_threshold(
    self,
    input_data: SemanticSearchInput,
) -> tuple[dict[tuple[str, str, str | None, str | None], FactOccurrence], float]:
    """
    Execute search with adaptive threshold adjustment.
    
    Returns:
        Tuple of (occurrences dict, final threshold used)
    """
    target_count = int(input_data.limit * ADAPTIVE_TARGET_RATIO)
    current_threshold = input_data.adaptive_threshold_max
    best_occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}
    best_threshold = current_threshold
    
    iterations = 0
    max_iterations = int(
        (input_data.adaptive_threshold_max - input_data.adaptive_threshold_min) 
        / ADAPTIVE_THRESHOLD_STEP
    ) + 1
    
    while current_threshold >= input_data.adaptive_threshold_min and iterations < max_iterations:
        iterations += 1
        
        logger.info(
            "Adaptive search iteration %d: threshold=%.2f, target=%d",
            iterations,
            current_threshold,
            target_count,
        )
        
        # Execute search at current threshold
        occurrences = self._execute_search_pass(
            input_data=input_data,
            similarity_threshold=current_threshold,
        )
        
        unique_count = len(occurrences)
        
        logger.info(
            "Adaptive search iteration %d: found %d unique facts",
            iterations,
            unique_count,
        )
        
        # Track best results seen (most results while above minimum threshold)
        if unique_count > len(best_occurrences):
            best_occurrences = occurrences
            best_threshold = current_threshold
        
        # Check if we've met target
        if unique_count >= target_count:
            logger.info(
                "Adaptive threshold converged: threshold=%.2f, results=%d, target=%d",
                current_threshold,
                unique_count,
                target_count,
            )
            return occurrences, current_threshold
        
        # Lower threshold for next iteration
        current_threshold -= ADAPTIVE_THRESHOLD_STEP
    
    logger.info(
        "Adaptive threshold exhausted: final_threshold=%.2f, results=%d, target=%d",
        best_threshold,
        len(best_occurrences),
        target_count,
    )
    
    return best_occurrences, best_threshold


def _execute_search_pass(
    self,
    input_data: SemanticSearchInput,
    similarity_threshold: float,
) -> dict[tuple[str, str, str | None, str | None], FactOccurrence]:
    """
    Execute a single search pass at the given threshold.
    
    This method contains the core search logic extracted from the current run() method.
    Returns the occurrences dictionary for deduplication counting.
    """
    occurrences: dict[tuple[str, str, str | None, str | None], FactOccurrence] = {}
    
    for query_idx, query in enumerate(input_data.queries, 1):
        # Vector search
        embedding = self.embeddings.embed_single(query)
        if not embedding:
            continue
            
        vector_rows = run_vector_query(
            self.context,
            self.index_name,
            embedding,
            input_data.limit,
            None,
        )
        
        # Keyword search
        keyword_rows = run_keyword_query(
            self.context,
            query,
            input_data.limit,
        )
        
        # Process both result sets
        for rows, effective_query_idx, source_type in [
            (vector_rows, query_idx, "vector"),
            (keyword_rows, query_idx + KEYWORD_QUERY_OFFSET, "keyword"),
        ]:
            for rank, row in enumerate(rows, start=1):
                score = row.get("score", 0.0)
                
                # Apply threshold filter
                if score < similarity_threshold:
                    continue
                
                node = row.get("node")
                if not node:
                    continue
                
                properties = dict(node)
                
                # Build deduplication key
                person_id = properties.get("person_id", "")
                fact_type = properties.get("fact_type", "")
                fact_object = properties.get("fact_object")
                relationship_type = self._extract_relationship_type(properties)
                
                dedup_key = (person_id, fact_type, fact_object, relationship_type)
                
                # Track occurrence
                evidence = row.get("evidence_with_content", []) or properties.get("evidence", [])
                
                if dedup_key not in occurrences:
                    occurrences[dedup_key] = FactOccurrence(
                        properties=properties,
                        best_score=score,
                        evidence=evidence,
                    )
                
                occurrences[dedup_key].add_observation(
                    effective_query_idx, score, rank, properties, evidence
                )
    
    return occurrences


def _extract_relationship_type(self, properties: dict[str, Any]) -> str | None:
    """Extract relationship_type from attributes for deduplication."""
    attributes_raw = properties.get("attributes")
    
    if isinstance(attributes_raw, str):
        try:
            attributes = json.loads(attributes_raw)
            return str(attributes.get("relationship_type")) if attributes.get("relationship_type") else None
        except json.JSONDecodeError:
            return None
    elif isinstance(attributes_raw, dict):
        return str(attributes_raw.get("relationship_type")) if attributes_raw.get("relationship_type") else None
    
    return None
```

##### 4. Update Main run() Method

**File**: `memory_agent/tools/semantic_search.py`

Modify the `run()` method to use adaptive thresholds:

```python
def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
    """Execute semantic search with optional adaptive thresholds."""
    
    logger.info(
        "semantic_search_facts called: queries=%r, limit=%d, adaptive=%s, index=%s",
        input_data.queries,
        input_data.limit,
        input_data.adaptive_threshold,
        self.index_name,
    )
    
    # Determine search mode
    if input_data.adaptive_threshold and input_data.similarity_threshold is None:
        # Adaptive mode: dynamically adjust threshold
        occurrences, final_threshold = self._search_with_adaptive_threshold(input_data)
        
        logger.info(
            "Adaptive search completed: final_threshold=%.2f, unique_facts=%d",
            final_threshold,
            len(occurrences),
        )
    else:
        # Fixed threshold mode: use provided or default threshold
        threshold = input_data.similarity_threshold or 0.6
        occurrences = self._execute_search_pass(input_data, threshold)
        
        logger.info(
            "Fixed threshold search completed: threshold=%.2f, unique_facts=%d",
            threshold,
            len(occurrences),
        )
    
    # Calculate combined scores and build results (existing logic)
    results_with_scores = self._build_results_from_occurrences(
        occurrences, 
        input_data.fusion_method,
        input_data.multi_query_boost,
    )
    
    # Sort and limit
    ordered = sorted(results_with_scores, key=lambda r: r.similarity_score, reverse=True)
    final_results = ordered[:input_data.limit]
    
    return SemanticSearchOutput(queries=input_data.queries, results=final_results)
```

##### 5. Apply Same Pattern to Message Search

**File**: `memory_agent/tools/semantic_search_messages.py`

Apply the same adaptive threshold pattern to `SemanticSearchMessagesTool`:

- Add `adaptive_threshold` field to `SemanticSearchMessagesInput`
- Implement `_search_with_adaptive_threshold()` method
- Implement `_execute_search_pass()` method  
- Update `run()` to use adaptive mode

The implementation is analogous to the facts search, with message-specific deduplication logic.

##### 6. Update Agent to Use Adaptive Mode

**File**: `memory_agent/agent.py`

Update the `plan_queries` function to enable adaptive thresholds:

```python
# In plan_queries, when building semantic_search_facts parameters:
if tool_name == "semantic_search_facts" and isinstance(parameters, dict):
    if "limit" not in parameters:
        parameters["limit"] = state.get("max_facts", config.max_facts)
    
    # Enable adaptive thresholds by default
    if "adaptive_threshold" not in parameters:
        parameters["adaptive_threshold"] = True
    
    # Remove fixed threshold to enable adaptive mode
    if parameters.get("adaptive_threshold") and "similarity_threshold" in parameters:
        del parameters["similarity_threshold"]
```

#### Testing Plan

##### Unit Tests

**File**: `tests/test_adaptive_threshold.py`

```python
import pytest
from memory_agent.tools.semantic_search import (
    SemanticSearchFactsTool,
    SemanticSearchInput,
    ADAPTIVE_THRESHOLD_MAX,
    ADAPTIVE_THRESHOLD_MIN,
)

class TestAdaptiveThreshold:
    """Tests for adaptive similarity threshold feature."""
    
    def test_adaptive_mode_enabled_by_default(self):
        """Verify adaptive mode is on when no threshold specified."""
        input_data = SemanticSearchInput(queries=["test"], limit=10)
        assert input_data.adaptive_threshold is True
        assert input_data.similarity_threshold is None
    
    def test_fixed_mode_when_threshold_specified(self):
        """Verify fixed mode when explicit threshold provided."""
        input_data = SemanticSearchInput(
            queries=["test"], 
            limit=10,
            similarity_threshold=0.7,
            adaptive_threshold=False,
        )
        assert input_data.adaptive_threshold is False
        assert input_data.similarity_threshold == 0.7
    
    def test_adaptive_lowers_threshold_on_few_results(self, mock_context):
        """Verify threshold decreases when results are insufficient."""
        tool = SemanticSearchFactsTool(mock_context)
        
        # Mock embedding and search to return few results at high threshold
        # ... test implementation
    
    def test_adaptive_stops_at_minimum_threshold(self, mock_context):
        """Verify threshold doesn't go below minimum."""
        tool = SemanticSearchFactsTool(mock_context)
        
        # Mock to always return 0 results
        # Verify final threshold >= ADAPTIVE_THRESHOLD_MIN
    
    def test_adaptive_stops_when_target_reached(self, mock_context):
        """Verify search stops early when target count achieved."""
        tool = SemanticSearchFactsTool(mock_context)
        
        # Mock to return sufficient results at threshold 0.65
        # Verify threshold stopped at 0.65, not lower
```

##### Integration Tests

```python
@pytest.mark.integration
async def test_adaptive_threshold_improves_recall():
    """Compare recall between fixed and adaptive modes."""
    
    # Query that returns few results at 0.6 threshold
    sparse_query = "obscure technical topic"
    
    fixed_result = tool.run(SemanticSearchInput(
        queries=[sparse_query],
        limit=20,
        similarity_threshold=0.6,
        adaptive_threshold=False,
    ))
    
    adaptive_result = tool.run(SemanticSearchInput(
        queries=[sparse_query],
        limit=20,
        adaptive_threshold=True,
    ))
    
    # Adaptive should return more results
    assert len(adaptive_result.results) >= len(fixed_result.results)
```

#### Metrics and Observability

Add Prometheus metrics to track adaptive behavior:

```python
from prometheus_client import Counter, Histogram, Gauge

adaptive_iterations = Histogram(
    "memory_agent_adaptive_threshold_iterations",
    "Number of iterations in adaptive threshold search",
    buckets=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)

adaptive_final_threshold = Histogram(
    "memory_agent_adaptive_final_threshold",
    "Final threshold value after adaptive search",
    buckets=(0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75),
)

adaptive_result_count = Histogram(
    "memory_agent_adaptive_result_count",
    "Number of results returned by adaptive search",
    buckets=(0, 5, 10, 15, 20, 25, 30, 40, 50),
)
```


---

## Feature 2: Cross-Encoder Re-ranking via llama-server

### What

Cross-encoder re-ranking is a two-stage retrieval approach that first retrieves candidates using fast bi-encoder embeddings, then re-scores them using a more accurate cross-encoder model. This implementation uses **Qwen3-Reranker-8B** running on a local llama-server instance.

**Current architecture (bi-encoder only)**:
```
Query ──► Encode ──► Vector Search ──► Top K Results
```

**Proposed architecture (bi-encoder + cross-encoder)**:
```
Query ──► Encode ──► Vector Search ──► 3x Candidates ──► Qwen3-Reranker ──► Top K Results
                                                              │
                                                        llama-server
                                                     (http://localhost:8080)
```

### Why

#### The Bi-Encoder Limitation

Bi-encoders (like the current sentence-transformers model) encode queries and documents independently into fixed-size vectors. This means:

1. **No query-document interaction**: The model cannot attend to both query and document simultaneously
2. **Information compression**: Complex semantic relationships are compressed into a fixed-size vector
3. **Asymmetric understanding**: A document about "Python programming" may not score highly for "snake handling skills" even though the terms overlap

#### Cross-Encoder Advantages

Cross-encoders see the query and document together as a single input. Qwen3-Reranker-8B uses an LLM-based approach that judges document relevance:

```
System: "Judge whether the Document meets the requirements based on the Query 
         and the Instruct provided. Note that the answer can only be 'yes' or 'no'."

User: "<Instruct>: {instruction}
       <Query>: {query}
       <Document>: {document}"

Output: yes/no logits → softmax → relevance score
```

This enables:
- **Full attention**: Every query token can attend to every document token
- **Fine-grained matching**: Understands "Python (programming)" vs "Python (snake)"
- **Negation handling**: Correctly scores "NOT interested in frontend"
- **Instruction-aware**: Custom instructions can tune ranking behavior

#### Why Qwen3-Reranker-8B

| Model | MTEB-R (English) | MMTEB-R (Multilingual) | Parameters |
|-------|------------------|------------------------|------------|
| **Qwen3-Reranker-8B** | **69.02** | **72.94** | 8B |
| Qwen3-Reranker-4B | 69.76 | 72.74 | 4B |
| bge-reranker-v2-m3 | 57.03 | 58.36 | 568M |
| ms-marco-MiniLM-L-6-v2 | ~50 | N/A | 22M |

Qwen3-Reranker-8B achieves state-of-the-art performance:
- **#1 on MTEB multilingual leaderboard** (as of June 2025)
- **100+ language support** including code
- **32K context length** for long documents
- **Apache 2.0 license** for commercial use

#### Why llama-server

Using llama-server provides several advantages:
- **Shared infrastructure**: Reuse existing llama.cpp deployment for both chat and reranking
- **Quantization**: Run 8B model efficiently with Q4/Q5 quantization
- **GPU acceleration**: Native CUDA/Metal support
- **No Python dependencies**: Model runs in separate process, avoids memory conflicts
- **Flexibility**: Easy to swap models or scale independently

#### Expected Improvements

Research shows cross-encoder re-ranking typically improves:
- **Precision@10**: 10-30% improvement
- **MRR (Mean Reciprocal Rank)**: 15-25% improvement
- **User satisfaction**: Higher-quality top results

### How

#### llama-server Setup

The memory agent expects a llama-server instance running Qwen3-Reranker-8B.

**Starting llama-server for reranking**:

```bash
# Download GGUF model (Q4_K_M quantization recommended)
# From: https://huggingface.co/Mungert/Qwen3-Reranker-8B-GGUF

# Start server (default: http://localhost:8080)
llama-server \
    -m Qwen3-Reranker-8B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 8192 \
    -ngl 99 \
    --jinja
```

**⚠️ Important**: Do NOT use the `--rerank` flag. The native `/v1/rerank` endpoint does not work correctly with Qwen3-Reranker models (produces garbage scores like 1e-28). Instead, we use the `/v1/chat/completions` endpoint with proper prompt formatting and logprobs extraction.

#### API Interaction

**Request format**:

```json
POST http://localhost:8080/v1/chat/completions
{
    "model": "qwen3-reranker",
    "messages": [
        {
            "role": "system",
            "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
        },
        {
            "role": "user", 
            "content": "<Instruct>: Given a query, retrieve relevant facts that answer the query\n\n<Query>: Who knows Python?\n\n<Document>: Alice has 5 years of Python experience"
        }
    ],
    "max_tokens": 1,
    "temperature": 0.0,
    "logprobs": true,
    "top_logprobs": 10
}
```

**Response parsing**:

The response contains logprobs for the generated token. We look for "yes" and "no" in the top logprobs and compute:

```python
import math

def compute_relevance_score(logprobs_content: list) -> float:
    """Extract relevance score from llama-server logprobs response."""
    if not logprobs_content:
        return 0.5  # Default uncertain score
    
    # Get the first token's top logprobs
    first_token = logprobs_content[0]
    top_logprobs = first_token.get("top_logprobs", [])
    
    yes_logprob = None
    no_logprob = None
    
    for entry in top_logprobs:
        token = entry.get("token", "").lower().strip()
        logprob = entry.get("logprob", -100)
        
        if token == "yes":
            yes_logprob = logprob
        elif token == "no":
            no_logprob = logprob
    
    # If we found both, compute softmax
    if yes_logprob is not None and no_logprob is not None:
        # softmax([yes, no])[0] = exp(yes) / (exp(yes) + exp(no))
        max_logprob = max(yes_logprob, no_logprob)
        yes_exp = math.exp(yes_logprob - max_logprob)
        no_exp = math.exp(no_logprob - max_logprob)
        return yes_exp / (yes_exp + no_exp)
    
    # Fallback: check if generated token is yes/no
    generated_token = first_token.get("token", "").lower().strip()
    if generated_token == "yes":
        return 1.0
    elif generated_token == "no":
        return 0.0
    
    return 0.5  # Uncertain
```

#### Implementation Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Cross-Encoder Re-ranking Pipeline                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  1. Request limit=N from user                                            │
│                         │                                                 │
│                         ▼                                                 │
│  2. Execute bi-encoder search with limit=N×3 (oversample)                │
│                         │                                                 │
│                         ▼                                                 │
│  3. Collect unique candidates (deduped)                                  │
│                         │                                                 │
│                         ▼                                                 │
│  4. Extract text from each candidate                                     │
│     - Facts: "{subject} {predicate} {object}"                            │
│     - Messages: "{author}: {content}"                                    │
│                         │                                                 │
│                         ▼                                                 │
│  5. For each candidate, send request to llama-server                     │
│     - POST /v1/chat/completions                                          │
│     - Use Qwen3-Reranker prompt template                                 │
│     - Request logprobs for yes/no scoring                                │
│                         │                                                 │
│                         ▼                                                 │
│  6. Parse responses, extract yes/no logprobs, compute scores             │
│                         │                                                 │
│                         ▼                                                 │
│  7. Sort by cross-encoder score (descending)                             │
│                         │                                                 │
│                         ▼                                                 │
│  8. Return top N results                                                 │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

#### File Changes

##### 1. New Reranker Module

**File**: `memory_agent/reranker.py` (new file)

```python
"""Cross-encoder re-ranking via llama-server with Qwen3-Reranker."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence, TypeVar

import httpx

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Qwen3-Reranker prompt template
SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query '
    'and the Instruct provided. Note that the answer can only be "yes" or "no".'
)

DEFAULT_INSTRUCTION = "Given a query, retrieve relevant facts that answer the query"


@dataclass
class LlamaServerRerankerConfig:
    """Configuration for llama-server based reranking."""
    
    base_url: str = "http://localhost:8080"
    model: str = "qwen3-reranker"  # Model name (informational, server uses loaded model)
    instruction: str = DEFAULT_INSTRUCTION
    timeout: float = 30.0
    max_concurrent: int = 10  # Max concurrent requests
    batch_size: int = 1  # Requests per batch (1 = sequential, safe for most setups)


@dataclass
class LlamaServerReranker:
    """
    Re-ranker that uses Qwen3-Reranker via llama-server.
    
    Sends requests to /v1/chat/completions with the Qwen3-Reranker prompt format
    and extracts relevance scores from yes/no token logprobs.
    """
    
    config: LlamaServerRerankerConfig = field(default_factory=LlamaServerRerankerConfig)
    _client: httpx.Client | None = field(default=None, repr=False)
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None
    
    def _build_messages(self, query: str, document: str) -> list[dict[str, str]]:
        """Build chat messages for Qwen3-Reranker."""
        user_content = (
            f"<Instruct>: {self.config.instruction}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {document}"
        )
        
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    
    def _parse_logprobs(self, response_data: dict[str, Any]) -> float:
        """
        Extract relevance score from llama-server response logprobs.
        
        Returns a score between 0.0 (not relevant) and 1.0 (relevant).
        """
        try:
            choices = response_data.get("choices", [])
            if not choices:
                logger.warning("No choices in response")
                return 0.5
            
            choice = choices[0]
            logprobs_data = choice.get("logprobs")
            
            if logprobs_data is None:
                # Fallback: parse generated text
                message = choice.get("message", {})
                content = message.get("content", "").lower().strip()
                if content.startswith("yes"):
                    return 1.0
                elif content.startswith("no"):
                    return 0.0
                return 0.5
            
            # Get content logprobs (OpenAI-compatible format)
            content_logprobs = logprobs_data.get("content", [])
            if not content_logprobs:
                return 0.5
            
            # Look at first token
            first_token_data = content_logprobs[0]
            top_logprobs = first_token_data.get("top_logprobs", [])
            
            yes_logprob = None
            no_logprob = None
            
            for entry in top_logprobs:
                token = entry.get("token", "").lower().strip()
                logprob = entry.get("logprob")
                
                if logprob is None:
                    continue
                
                if token == "yes" and yes_logprob is None:
                    yes_logprob = logprob
                elif token == "no" and no_logprob is None:
                    no_logprob = logprob
            
            # Compute softmax if we have both
            if yes_logprob is not None and no_logprob is not None:
                # Numerical stability: subtract max
                max_lp = max(yes_logprob, no_logprob)
                yes_exp = math.exp(yes_logprob - max_lp)
                no_exp = math.exp(no_logprob - max_lp)
                return yes_exp / (yes_exp + no_exp)
            
            # Fallback: use generated token
            generated_token = first_token_data.get("token", "").lower().strip()
            if generated_token == "yes":
                return 1.0
            elif generated_token == "no":
                return 0.0
            
            # If yes was found but not no (or vice versa), use that info
            if yes_logprob is not None:
                return 0.8  # Likely yes
            if no_logprob is not None:
                return 0.2  # Likely no
            
            return 0.5
            
        except Exception as e:
            logger.warning("Error parsing logprobs: %s", e)
            return 0.5
    
    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: The search query
            document: The document text to score
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        messages = self._build_messages(query, document)
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 10,
        }
        
        try:
            response = self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return self._parse_logprobs(response.json())
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error scoring pair: %s", e)
            return 0.5
        except httpx.RequestError as e:
            logger.error("Request error scoring pair: %s", e)
            return 0.5
    
    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """
        Score multiple query-document pairs.
        
        Args:
            query: The search query
            documents: List of document texts to score
            
        Returns:
            List of relevance scores (same order as documents)
        """
        if not documents:
            return []
        
        scores = []
        for i, doc in enumerate(documents):
            score = self.score_pair(query, doc)
            scores.append(score)
            
            if (i + 1) % 10 == 0:
                logger.debug("Scored %d/%d documents", i + 1, len(documents))
        
        return scores
    
    def rerank(
        self,
        query: str,
        items: Sequence[T],
        text_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[tuple[T, float]]:
        """
        Re-rank items by relevance to query.
        
        Args:
            query: The search query
            items: Sequence of items to re-rank
            text_extractor: Function to extract text from each item
            top_k: Return only top-k items (None = return all)
            
        Returns:
            List of (item, score) tuples sorted by score descending
        """
        if not items:
            return []
        
        # Extract texts
        texts = [text_extractor(item) for item in items]
        
        # Score all pairs
        scores = self.score_pairs(query, texts)
        
        # Combine and sort
        scored_items = list(zip(items, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k
        if top_k is not None:
            scored_items = scored_items[:top_k]
        
        return scored_items
    
    def is_available(self) -> bool:
        """Check if llama-server is reachable."""
        try:
            response = self.client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


# Async version for high-throughput scenarios
@dataclass
class AsyncLlamaServerReranker:
    """
    Async version of LlamaServerReranker for concurrent scoring.
    """
    
    config: LlamaServerRerankerConfig = field(default_factory=LlamaServerRerankerConfig)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy-initialize async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    def _build_messages(self, query: str, document: str) -> list[dict[str, str]]:
        """Build chat messages for Qwen3-Reranker."""
        user_content = (
            f"<Instruct>: {self.config.instruction}\n\n"
            f"<Query>: {query}\n\n"
            f"<Document>: {document}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    
    def _parse_logprobs(self, response_data: dict[str, Any]) -> float:
        """Parse logprobs - same logic as sync version."""
        # (Same implementation as LlamaServerReranker._parse_logprobs)
        try:
            choices = response_data.get("choices", [])
            if not choices:
                return 0.5
            
            choice = choices[0]
            logprobs_data = choice.get("logprobs")
            
            if logprobs_data is None:
                message = choice.get("message", {})
                content = message.get("content", "").lower().strip()
                if content.startswith("yes"):
                    return 1.0
                elif content.startswith("no"):
                    return 0.0
                return 0.5
            
            content_logprobs = logprobs_data.get("content", [])
            if not content_logprobs:
                return 0.5
            
            first_token_data = content_logprobs[0]
            top_logprobs = first_token_data.get("top_logprobs", [])
            
            yes_logprob = None
            no_logprob = None
            
            for entry in top_logprobs:
                token = entry.get("token", "").lower().strip()
                logprob = entry.get("logprob")
                
                if logprob is None:
                    continue
                
                if token == "yes" and yes_logprob is None:
                    yes_logprob = logprob
                elif token == "no" and no_logprob is None:
                    no_logprob = logprob
            
            if yes_logprob is not None and no_logprob is not None:
                max_lp = max(yes_logprob, no_logprob)
                yes_exp = math.exp(yes_logprob - max_lp)
                no_exp = math.exp(no_logprob - max_lp)
                return yes_exp / (yes_exp + no_exp)
            
            generated_token = first_token_data.get("token", "").lower().strip()
            if generated_token == "yes":
                return 1.0
            elif generated_token == "no":
                return 0.0
            
            if yes_logprob is not None:
                return 0.8
            if no_logprob is not None:
                return 0.2
            
            return 0.5
            
        except Exception as e:
            logger.warning("Error parsing logprobs: %s", e)
            return 0.5
    
    async def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair asynchronously."""
        messages = self._build_messages(query, document)
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 10,
        }
        
        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return self._parse_logprobs(response.json())
        except httpx.HTTPStatusError as e:
            logger.error("HTTP error scoring pair: %s", e)
            return 0.5
        except httpx.RequestError as e:
            logger.error("Request error scoring pair: %s", e)
            return 0.5
    
    async def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        """Score multiple pairs with controlled concurrency."""
        import asyncio
        
        if not documents:
            return []
        
        semaphore = asyncio.Semaphore(self.config.max_concurrent)
        
        async def score_with_semaphore(doc: str) -> float:
            async with semaphore:
                return await self.score_pair(query, doc)
        
        tasks = [score_with_semaphore(doc) for doc in documents]
        return await asyncio.gather(*tasks)
    
    async def rerank(
        self,
        query: str,
        items: Sequence[T],
        text_extractor: Callable[[T], str],
        top_k: int | None = None,
    ) -> list[tuple[T, float]]:
        """Re-rank items asynchronously."""
        if not items:
            return []
        
        texts = [text_extractor(item) for item in items]
        scores = await self.score_pairs(query, texts)
        
        scored_items = list(zip(items, scores))
        scored_items.sort(key=lambda x: x[1], reverse=True)
        
        if top_k is not None:
            scored_items = scored_items[:top_k]
        
        return scored_items
```

##### 2. Add Configuration

**File**: `memory_agent/config.py`

Add reranker configuration:

```python
@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Configuration for llama-server based reranking."""
    
    enabled: bool = True
    base_url: str = "http://localhost:8080"
    model: str = "qwen3-reranker"
    instruction: str = "Given a query, retrieve relevant facts that answer the query"
    timeout: float = 30.0
    max_concurrent: int = 10
    oversample_factor: float = 3.0  # Retrieve 3x candidates for re-ranking


@dataclass(frozen=True, slots=True)
class Settings:
    """Aggregate application settings."""
    
    neo4j: Neo4jConfig
    llm: LLMConfig
    sqlite: SQLiteConfig
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    message_embeddings: MessageEmbeddingConfig = field(default_factory=MessageEmbeddingConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)  # NEW
    agent: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    
    @classmethod
    def from_env(cls) -> "Settings":
        # ... existing code ...
        
        # Add reranker config loading
        reranker = RerankerConfig(
            enabled=cls._bool_env("RERANKER_ENABLED", True),
            base_url=os.getenv("RERANKER_BASE_URL", "http://localhost:8080"),
            model=os.getenv("RERANKER_MODEL", "qwen3-reranker"),
            instruction=os.getenv(
                "RERANKER_INSTRUCTION",
                "Given a query, retrieve relevant facts that answer the query"
            ),
            timeout=float(os.getenv("RERANKER_TIMEOUT", "30.0")),
            max_concurrent=int(os.getenv("RERANKER_MAX_CONCURRENT", "10")),
            oversample_factor=float(os.getenv("RERANKER_OVERSAMPLE", "3.0")),
        )
        
        return cls(
            # ... existing fields ...
            reranker=reranker,
        )
```

**Environment variables**:

| Variable | Default | Description |
|----------|---------|-------------|
| `RERANKER_ENABLED` | `true` | Enable/disable reranking |
| `RERANKER_BASE_URL` | `http://localhost:8080` | llama-server URL |
| `RERANKER_MODEL` | `qwen3-reranker` | Model name (informational) |
| `RERANKER_INSTRUCTION` | (see above) | Custom instruction for reranking |
| `RERANKER_TIMEOUT` | `30.0` | Request timeout in seconds |
| `RERANKER_MAX_CONCURRENT` | `10` | Max concurrent requests (async) |
| `RERANKER_OVERSAMPLE` | `3.0` | Oversample factor for candidates |

##### 3. Update Tool Context

**File**: `memory_agent/tools/base.py`

Add reranker to tool context:

```python
from ..reranker import LlamaServerReranker, LlamaServerRerankerConfig

@dataclass(slots=True)
class ToolContext:
    """Shared resources needed by tool implementations."""
    
    driver: Driver
    embeddings_model: Any | None = None
    reranker: LlamaServerReranker | None = None  # NEW
    
    def session(self) -> Session:
        """Create a Neo4j session."""
        return self.driver.session()
```

##### 4. Update Tool Initialization

**File**: `memory_agent/app.py`

Initialize reranker in app startup:

```python
from .reranker import LlamaServerReranker, LlamaServerRerankerConfig

@app.on_event("startup")
async def on_startup() -> None:
    cfg = app.state.settings
    
    # ... existing initialization ...
    
    # Initialize reranker if enabled
    reranker = None
    if cfg.reranker.enabled:
        reranker_config = LlamaServerRerankerConfig(
            base_url=cfg.reranker.base_url,
            model=cfg.reranker.model,
            instruction=cfg.reranker.instruction,
            timeout=cfg.reranker.timeout,
            max_concurrent=cfg.reranker.max_concurrent,
        )
        reranker = LlamaServerReranker(config=reranker_config)
        
        # Check if server is available
        if reranker.is_available():
            logger.info("Reranker connected: %s", cfg.reranker.base_url)
        else:
            logger.warning(
                "Reranker server not available at %s - reranking will be disabled",
                cfg.reranker.base_url,
            )
            reranker = None
    
    tool_context = ToolContext(
        driver=driver,
        embeddings_model=embedding_provider,
        reranker=reranker,  # NEW
    )
    
    # ... rest of initialization ...
```

##### 5. Update Input Models

**File**: `memory_agent/tools/semantic_search.py`

Add re-ranking options to input model:

```python
class SemanticSearchInput(BaseModel):
    """Inputs for semantic_search_facts."""
    
    queries: list[str] = Field(min_length=1, max_length=20)
    limit: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    adaptive_threshold: bool = Field(default=True)
    adaptive_threshold_max: float = Field(default=ADAPTIVE_THRESHOLD_MAX, ge=0.0, le=1.0)
    adaptive_threshold_min: float = Field(default=ADAPTIVE_THRESHOLD_MIN, ge=0.0, le=1.0)
    fusion_method: Literal["rrf", "score_sum", "score_max"] = Field(default=DEFAULT_FUSION_METHOD)
    multi_query_boost: float = Field(default=DEFAULT_MULTI_QUERY_BOOST, ge=0.0, le=1.0)
    
    # NEW: Re-ranking options
    enable_reranking: bool = Field(default=True)
    rerank_query: str | None = Field(
        default=None,
        description="Override query for re-ranking (uses first query if not set)",
    )
```

##### 6. Implement Re-ranking in Search Tool

**File**: `memory_agent/tools/semantic_search.py`

Add re-ranking method and integrate into run():

```python
from ..reranker import LlamaServerReranker

class SemanticSearchFactsTool(ToolBase[SemanticSearchInput, SemanticSearchOutput]):
    
    # ... existing code ...
    
    @property
    def reranker(self) -> LlamaServerReranker | None:
        """Get reranker if available."""
        return self.context.reranker
    
    def _rerank_results(
        self,
        results: list[SemanticSearchResult],
        query: str,
        top_k: int,
    ) -> list[SemanticSearchResult]:
        """
        Re-rank results using llama-server Qwen3-Reranker.
        
        Args:
            results: Candidate results from bi-encoder search
            query: Query to re-rank against
            top_k: Number of results to return
            
        Returns:
            Re-ranked and filtered results
        """
        reranker = self.reranker
        if reranker is None:
            logger.debug("Reranker not available, skipping re-ranking")
            return results[:top_k]
        
        if len(results) <= top_k:
            logger.debug("Fewer candidates than top_k, skipping re-ranking")
            return results
        
        logger.info(
            "Re-ranking %d candidates to select top %d",
            len(results),
            top_k,
        )
        
        import time
        start = time.perf_counter()
        
        def extract_text(result: SemanticSearchResult) -> str:
            """Extract searchable text from result."""
            parts = []
            if result.subject:
                parts.append(result.subject)
            if result.predicate:
                parts.append(result.predicate)
            if result.object:
                parts.append(result.object)
            return " ".join(parts)
        
        # Re-rank
        reranked = reranker.rerank(
            query=query,
            items=results,
            text_extractor=extract_text,
            top_k=top_k,
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Re-ranking completed in %.1fms, selected %d results",
            elapsed_ms,
            len(reranked),
        )
        
        # Extract just the results (drop scores)
        return [item for item, score in reranked]
    
    def run(self, input_data: SemanticSearchInput) -> SemanticSearchOutput:
        """Execute semantic search with optional re-ranking."""
        
        # Calculate oversample limit if re-ranking is enabled
        oversample_factor = getattr(
            self.context, 
            'oversample_factor', 
            3.0
        ) if input_data.enable_reranking else 1.0
        
        search_limit = int(input_data.limit * oversample_factor)
        
        # ... existing search logic with search_limit ...
        
        # After deduplication and before returning:
        if input_data.enable_reranking and len(results) > input_data.limit:
            rerank_query = input_data.rerank_query or input_data.queries[0]
            results = self._rerank_results(
                results=results,
                query=rerank_query,
                top_k=input_data.limit,
            )
        else:
            results = results[:input_data.limit]
        
        return SemanticSearchOutput(results=results)
```

##### 7. Similar Changes for Messages Tool

**File**: `memory_agent/tools/semantic_search_messages.py`

Apply similar changes for message search:

```python
def _rerank_messages(
    self,
    results: list[MessageSearchResult],
    query: str,
    top_k: int,
) -> list[MessageSearchResult]:
    """Re-rank message results using llama-server."""
    reranker = self.reranker
    if reranker is None:
        return results[:top_k]
    
    if len(results) <= top_k:
        return results
    
    def extract_text(result: MessageSearchResult) -> str:
        """Extract searchable text from message."""
        author = result.author_name or result.author_id or "Unknown"
        content = result.content or ""
        return f"{author}: {content}"
    
    reranked = reranker.rerank(
        query=query,
        items=results,
        text_extractor=extract_text,
        top_k=top_k,
    )
    
    return [item for item, score in reranked]
```

#### Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    # ... existing dependencies ...
    "httpx>=0.27.0",  # For llama-server communication
]
```

#### Testing Plan

##### Unit Tests

**File**: `tests/test_reranker.py`

```python
import pytest
from unittest.mock import MagicMock, patch
from memory_agent.reranker import (
    LlamaServerReranker,
    LlamaServerRerankerConfig,
)


class TestLlamaServerReranker:
    """Tests for LlamaServerReranker."""
    
    @pytest.fixture
    def config(self):
        return LlamaServerRerankerConfig(
            base_url="http://localhost:8080",
            timeout=5.0,
        )
    
    @pytest.fixture
    def reranker(self, config):
        return LlamaServerReranker(config=config)
    
    def test_build_messages(self, reranker):
        """Verify message format matches Qwen3-Reranker spec."""
        messages = reranker._build_messages(
            query="Who knows Python?",
            document="Alice has 5 years of Python experience",
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert 'yes' in messages[0]["content"]
        assert 'no' in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "<Query>: Who knows Python?" in messages[1]["content"]
        assert "<Document>: Alice has 5 years" in messages[1]["content"]
    
    def test_parse_logprobs_yes_higher(self, reranker):
        """Test parsing when yes logprob is higher."""
        response_data = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "yes",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "yes", "logprob": -0.1},
                            {"token": "no", "logprob": -2.5},
                        ]
                    }]
                }
            }]
        }
        
        score = reranker._parse_logprobs(response_data)
        assert score > 0.8  # Should be high since yes >> no
    
    def test_parse_logprobs_no_higher(self, reranker):
        """Test parsing when no logprob is higher."""
        response_data = {
            "choices": [{
                "logprobs": {
                    "content": [{
                        "token": "no",
                        "logprob": -0.1,
                        "top_logprobs": [
                            {"token": "no", "logprob": -0.1},
                            {"token": "yes", "logprob": -3.0},
                        ]
                    }]
                }
            }]
        }
        
        score = reranker._parse_logprobs(response_data)
        assert score < 0.2  # Should be low since no >> yes
    
    def test_parse_logprobs_fallback_text(self, reranker):
        """Test fallback to text parsing when no logprobs."""
        response_data = {
            "choices": [{
                "message": {"content": "yes"},
                "logprobs": None,
            }]
        }
        
        score = reranker._parse_logprobs(response_data)
        assert score == 1.0
    
    def test_rerank_ordering(self, reranker):
        """Test that rerank orders by score descending."""
        with patch.object(reranker, 'score_pairs') as mock_score:
            mock_score.return_value = [0.3, 0.9, 0.1]
            
            items = ["low", "high", "lowest"]
            reranked = reranker.rerank(
                query="test",
                items=items,
                text_extractor=lambda x: x,
            )
            
            assert reranked[0] == ("high", 0.9)
            assert reranked[1] == ("low", 0.3)
            assert reranked[2] == ("lowest", 0.1)
    
    def test_rerank_top_k(self, reranker):
        """Test that top_k limits results."""
        with patch.object(reranker, 'score_pairs') as mock_score:
            mock_score.return_value = [0.3, 0.9, 0.1, 0.5]
            
            items = ["a", "b", "c", "d"]
            reranked = reranker.rerank(
                query="test",
                items=items,
                text_extractor=lambda x: x,
                top_k=2,
            )
            
            assert len(reranked) == 2
            assert reranked[0][1] == 0.9  # Highest score
            assert reranked[1][1] == 0.5  # Second highest


@pytest.mark.integration
class TestLlamaServerRerankerIntegration:
    """Integration tests requiring a running llama-server."""
    
    @pytest.fixture
    def reranker(self):
        config = LlamaServerRerankerConfig(
            base_url="http://localhost:8080",
        )
        reranker = LlamaServerReranker(config=config)
        
        if not reranker.is_available():
            pytest.skip("llama-server not available")
        
        return reranker
    
    def test_score_relevant_pair(self, reranker):
        """Test scoring a clearly relevant pair."""
        score = reranker.score_pair(
            query="Who knows Python programming?",
            document="Alice is a senior Python developer with 10 years experience",
        )
        
        assert score > 0.7
    
    def test_score_irrelevant_pair(self, reranker):
        """Test scoring a clearly irrelevant pair."""
        score = reranker.score_pair(
            query="Who knows Python programming?",
            document="The weather in Paris is sunny today",
        )
        
        assert score < 0.3
    
    def test_rerank_improves_ordering(self, reranker):
        """Test that reranking improves result ordering."""
        query = "machine learning experience"
        
        # Documents in suboptimal order
        documents = [
            "Likes coffee and tea",  # Irrelevant
            "Built neural networks at Google",  # Very relevant
            "Has a dog named Max",  # Irrelevant
            "Trained ML models on large datasets",  # Very relevant
        ]
        
        reranked = reranker.rerank(
            query=query,
            items=documents,
            text_extractor=lambda x: x,
        )
        
        # Relevant docs should be at top
        top_2 = [doc for doc, _ in reranked[:2]]
        assert "neural networks" in top_2[0] or "ML models" in top_2[0]
        assert "neural networks" in top_2[1] or "ML models" in top_2[1]
```

##### Integration Tests

**File**: `tests/test_reranking_integration.py`

```python
@pytest.mark.integration
async def test_semantic_search_with_reranking():
    """Test full semantic search with reranking enabled."""
    
    # Setup: Ensure test data exists
    query = "Who has experience with Kubernetes?"
    
    # Without reranking
    no_rerank = tool.run(SemanticSearchInput(
        queries=[query],
        limit=10,
        enable_reranking=False,
    ))
    
    # With reranking
    with_rerank = tool.run(SemanticSearchInput(
        queries=[query],
        limit=10,
        enable_reranking=True,
    ))
    
    # Both should return results
    assert len(no_rerank.results) > 0
    assert len(with_rerank.results) > 0
    
    # Manual inspection: reranked results should have more K8s-related
    # facts in top positions


@pytest.mark.integration
async def test_reranking_latency():
    """Verify reranking latency is acceptable."""
    import time
    
    query = "Python developer"
    input_data = SemanticSearchInput(
        queries=[query],
        limit=10,  # Will fetch 30 candidates with 3x oversample
        enable_reranking=True,
    )
    
    start = time.perf_counter()
    result = tool.run(input_data)
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    # Reranking 30 candidates should complete in under 10 seconds
    # (depends heavily on llama-server throughput)
    assert elapsed_ms < 10000
```

#### Metrics and Observability

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

rerank_calls = Counter(
    "memory_agent_rerank_calls_total",
    "Total re-ranking operations",
    ["tool", "status"],
)

rerank_candidates = Histogram(
    "memory_agent_rerank_candidates",
    "Number of candidates re-ranked",
    buckets=(5, 10, 20, 30, 50, 75, 100),
)

rerank_latency = Histogram(
    "memory_agent_rerank_latency_seconds",
    "Re-ranking latency in seconds",
    buckets=(0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0),
)

rerank_server_errors = Counter(
    "memory_agent_rerank_server_errors_total",
    "Errors communicating with rerank server",
    ["error_type"],
)
```

#### Performance Considerations

1. **Sequential vs Concurrent**: By default, requests are sequential for simplicity. Enable async reranker with `max_concurrent > 1` for parallelism if llama-server can handle it.

2. **Latency**: Each candidate requires a round-trip to llama-server. For 30 candidates:
   - Sequential: ~100-200ms per request = 3-6 seconds total
   - Concurrent (10 parallel): ~300-600ms total

3. **Max candidates**: Cap at 50-100 candidates to prevent excessive latency.

4. **Batch inference**: llama-server doesn't support true batch inference for chat completions. Each query-document pair requires a separate request.

5. **Caching**: Consider caching scores for repeated query-document pairs if the same documents are frequently searched.

#### Rollout Plan

1. **Phase 1**: Deploy llama-server with Qwen3-Reranker-8B alongside existing infrastructure
2. **Phase 2**: Deploy memory_agent with `RERANKER_ENABLED=false`
3. **Phase 3**: Enable for debug endpoint only, measure latency and quality
4. **Phase 4**: Enable for 20% of production traffic (A/B test)
5. **Phase 5**: Full rollout after validating precision improvements

---

## Combined Implementation Order

Recommended implementation sequence:

1. **Week 1**: Implement Adaptive Similarity Thresholds
   - Refactor search into `_execute_search_pass`
   - Implement `_search_with_adaptive_threshold`
   - Add unit tests
   - Deploy with feature flag

2. **Week 2**: Deploy llama-server Infrastructure
   - Set up llama-server with Qwen3-Reranker-8B-Q4_K_M
   - Configure GPU offloading and context length
   - Test /v1/chat/completions endpoint with logprobs
   - Benchmark throughput

3. **Week 3**: Implement Cross-Encoder Re-ranking
   - Add `LlamaServerReranker` module
   - Update config and tool context
   - Implement `_rerank_results` in both search tools
   - Add unit tests and mocks

4. **Week 4**: Integration and Tuning
   - Enable both features in staging
   - Run evaluation benchmarks
   - Tune oversample factors and timeouts
   - Document performance characteristics

5. **Week 5**: Production Rollout
   - Gradual rollout with monitoring
   - Collect user feedback
   - Adjust defaults based on metrics

---

## Appendix: llama-server Command Reference

```bash
# Basic setup for reranking
llama-server \
    -m /path/to/Qwen3-Reranker-8B-Q4_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 8192 \                    # Context length
    -ngl 99 \                    # GPU layers (99 = all)
    --jinja \                    # Enable Jinja templates
    -np 4 \                      # Number of parallel slots
    --threads 8                  # CPU threads

# Test the endpoint
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": "<Instruct>: Given a query, retrieve relevant facts\n\n<Query>: Python experience\n\n<Document>: Alice is a Python developer"}
        ],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": true,
        "top_logprobs": 10
    }'
```
