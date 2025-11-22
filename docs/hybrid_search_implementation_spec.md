Here is the technical specification for implementing Hybrid Search.

# Technical Spec: Hybrid Search Implementation (Vector + Keyword)

## 1. Overview & Motivation

### Problem
The current implementation relies exclusively on **Vector Search** (Cosine Similarity). While excellent for understanding conceptual intent (e.g., "coding" $\leftrightarrow$ "programming"), vector search struggles with:
1.  **Exact Identifiers:** Specific project codes (e.g., "PROJ-99"), error codes, or user IDs.
2.  **Rare Entities:** Names or terms that the embedding model (Gemma-300m) has rarely seen.
3.  **Acronyms:** Specific industry abbreviations that may look semantically arguably similar to other text but require exact matching.

### Solution
Implement **Hybrid Search** using Reciprocal Rank Fusion (RRF).
For every user query, we will execute two database searches in parallel:
1.  **Vector Search:** Finds conceptually related facts.
2.  **Fulltext (Keyword) Search:** Finds textual matches using BM25/Lucene scoring (fuzzy matching).

These results will be merged at the application layer using the existing RRF logic in `semantic_search.py`.

---

## 2. Architecture Changes

### Database Schema (Neo4j)
*   **New Index:** We must create a Neo4j `FULLTEXT` index on `FactEmbedding` nodes targeting the `text`, `person_name`, and `fact_object` properties.

### Tool Logic (`semantic_search_facts`)
*   The tool currently iterates through a list of generated queries (`input_data.queries`).
*   For each query, it currently runs `run_vector_query`.
*   **Change:** It must now run `run_vector_query` **AND** a new `run_keyword_query`.
*   **Fusion:** The results from both sources will be fed into the existing `FactOccurrence` tracking system, which handles the RRF scoring math.

---

## 3. Detailed Implementation Plan

### Task 1: Create Fulltext Index
**File:** `memory_agent/embedding_pipeline.py`

We need to ensure the keyword index exists when the application starts.

**Implementation Details:**
Modify the `ensure_vector_index` function (can be renamed to `ensure_indices`):

```python
def ensure_indices(session: Session) -> None:
    """Ensure both Vector and Fulltext indices exist."""
    
    # 1. Existing Vector Index logic (KEEP AS IS)
    # ...

    # 2. NEW: Fulltext Index
    logger.info("Ensuring Neo4j fulltext index fact_fulltext exists")
    session.run(
        """
        CREATE FULLTEXT INDEX fact_fulltext IF NOT EXISTS
        FOR (f:FactEmbedding)
        ON EACH [f.text, f.person_name, f.fact_object]
        """
    )
```
*Note: Update `run_embedding_pipeline` to call this renamed function.*

### Task 2: Create Keyword Search Utility
**File:** `memory_agent/tools/utils.py`

Add a function to execute the Lucene-based keyword query. This mirrors `run_vector_query` but uses string matching.

**Implementation Details:**

```python
def run_keyword_query(
    context: ToolContext,
    query_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Execute a Fulltext keyword search using Lucene syntax."""
    
    # Clean query for Lucene: remove special chars that break syntax, 
    # and append fuzzy match tilde (~).
    # Simple sanitation: keep alphanumeric and spaces.
    import re
    safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query_text).strip()
    
    if not safe_text:
        return []

    # Construct Lucene query: standard terms OR fuzzy terms
    # e.g., "Python" -> "Python OR Python~"
    lucene_query = f"{safe_text} OR {safe_text}~"

    statement = """
    CALL db.index.fulltext.queryNodes("fact_fulltext", $lucene_query, {limit: $limit})
    YIELD node, score
    
    // Format evidence exactly like run_vector_query does
    WITH node, score, node.evidence AS evidence_ids
    OPTIONAL MATCH (author:Person)-[:SENT]->(msg:Message)
    WHERE msg.id IN evidence_ids
    WITH node, score,
            COLLECT({
                source_id: msg.id, 
                snippet: msg.content, 
                created_at: msg.timestamp, 
                author: coalesce(author.realName, author.name)
            }) AS evidence_with_content
            
    RETURN node, score, evidence_with_content
    """
    
    return run_read_query(context, statement, {"lucene_query": lucene_query, "limit": limit})
```

### Task 3: Integrate into Semantic Search Tool
**File:** `memory_agent/tools/semantic_search.py`

Update the `SemanticSearchFactsTool.run` method to execute both searches and merge results.

**Implementation Steps:**

1.  **Identify the Loop:** Locate `for query_idx, query in enumerate(input_data.queries, 1):`.
2.  **Refactor Loop Body:**

```python
# Inside SemanticSearchFactsTool.run

for query_idx, query in enumerate(input_data.queries, 1):
    logger.info("Processing query %d/%d: %r", query_idx, len(input_data.queries), query)

    # --- 1. VECTOR SEARCH (Existing Logic) ---
    embedding = self.embeddings.embed_single(query)
    vector_rows = []
    if embedding:
         try:
             vector_rows = run_vector_query(self.context, self.index_name, embedding, input_data.limit, None)
         except Exception as e:
             logger.warning(f"Vector search failed: {e}")

    # --- 2. NEW: KEYWORD SEARCH ---
    keyword_rows = []
    try:
        from .utils import run_keyword_query # Ensure import
        # Use a slightly stricter limit for keywords to avoid noise, or match input_data.limit
        keyword_rows = run_keyword_query(self.context, query, input_data.limit)
        logger.info("Query %d keyword results: %d", query_idx, len(keyword_rows))
    except Exception as e:
        logger.warning(f"Keyword search failed: {e}")

    # --- 3. MERGE RESULTS ---
    # We treat keyword results exactly like vector results for RRF.
    # Since they come from different sources, the "score" ranges differ 
    # (Cosine 0-1 vs Lucene unbounded), but RRF relies on RANK, so this is fine.
    
    all_rows = []
    
    # Tag rows to identify source for logging/debugging if needed
    for row in vector_rows:
        row['source'] = 'vector'
        all_rows.append(row)
        
    for row in keyword_rows:
        row['source'] = 'keyword'
        all_rows.append(row)

    # --- 4. PROCESS (Existing Logic) ---
    # The existing iteration over 'rows' handles deduplication via 'dedup_key'
    # and ranking via 'occurrences'. 
    
    # Important: Reset existing loop variable
    rows = all_rows 

    # ... Continue with existing processing loop ...
    for rank, row in enumerate(rows, start=1):
        # NOTE: Because we merged two lists, 'rank' here is slightly artificial.
        # Ideally, we process vector_rows and keyword_rows separately to give them 
        # their own clean 1..N ranks for RRF calculation.
        pass
```

**Refined Merge Logic (Crucial for RRF accuracy):**

Instead of merging `all_rows` blindly, we should process them sequentially so `rank` is accurate for the specific method.

```python
    # Helper to process a batch of rows
    def process_batch(rows, source_type):
        for rank, row in enumerate(rows, start=1):
            node = row.get("node")
            if not node: continue
            
            properties = dict(node)
            score = row.get("score", 0.0)
            
            # ... (Existing logic to extract properties, dedup_key) ...
            
            # Update occurrence
            occurrence = occurrences.get(dedup_key)
            if occurrence is None:
                occurrence = FactOccurrence(properties=properties, best_score=score, evidence=evidence)
                occurrences[dedup_key] = occurrence
            
            # IMPORTANT: add_observation needs to handle the fact that we have multiple 
            # "virtual queries" happening per query string. 
            # We can formulate a unique ID like: (query_idx * 10) + (1 if vector else 2)
            # OR simply treat keyword matches as a boost to the existing query_idx.
            
            # PREFERRED APPROACH: Treat the Keyword Search as a distinct signal 
            # for the SAME query_idx.
            occurrence.add_observation(query_idx, score, rank, properties, evidence)

    # Execute processing
    process_batch(vector_rows, "vector")
    process_batch(keyword_rows, "keyword")
```

**Note on `FactOccurrence` Update:**
The `FactOccurrence.add_observation` method (in `semantic_search.py`) currently stores `query_scores[query_idx] = score`.
If we call it twice for the same `query_idx` (once vector, once keyword), the last one overwrites.
**Optimization:** We should modify `add_observation` to take the *max* score or better yet, modify input to treat keyword search as a separate "virtual" query index (e.g., negative index or offset) to allow RRF to sum up the contributions of *both* vector and keyword finding the same fact.

**Proposed Logic Adjustment in `semantic_search.py`:**
When calling `process_batch` for KeyWord search, pass `query_idx + 1000` (an offset).
This treats the keyword search as a distinct "voter" in the RRF system.
Example:
1. Query "Apple" (Vector) -> Rank 1
2. Query "Apple" (Keyword) -> Rank 1
3. RRF calculates $1/(60+1) + 1/(60+1)$, resulting in a very high score for facts found by both.

---

## 4. Testing Plan

### 1. Exact ID Match
*   **Input:** "Who worked on ticket JIRA-9988?"
*   **Expected:** The existing system likely fails. The Hybrid system should find the fact containing "JIRA-9988" via Fulltext index and rank it highly.

### 2. Typo Tolerance
*   **Input:** "Exprnce with Pyton" (Typo in Python)
*   **Expected:**
    *   Vector search might handle this via semantic robustness.
    *   Keyword search (using `~` fuzzy matching) should explicitly match "Python".
    *   The fusion should rank "Python" facts at the top.

### 3. No Regressions
*   **Input:** "Tell me about coding skills"
*   **Expected:** Vector search results should dominate. Keyword search might return few precise matches. The system should degrade gracefully to Vector performance.