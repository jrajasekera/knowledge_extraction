# Fact Deduplication System Specification

## Overview

This document specifies a system for detecting and merging duplicate facts in the knowledge extraction pipeline. The deduplication process uses a combination of MinHash/LSH for textual similarity, embedding-based semantic similarity, and LLM-based consolidation to identify and merge duplicate facts while preserving evidence provenance.

---

## Goals

- **Reduce duplicate facts** by 70%+ while preserving unique information
- **Scale efficiently** to handle 100,000+ facts without O(n²) comparisons
- **Maintain provenance** through complete audit trails
- **Be resumable** to handle interruptions and process incrementally
- **Preserve data quality** by using LLM judgment for final merge decisions

---

## Non-Goals

- Real-time deduplication (this is a batch process)
- Deduplication across different subjects (facts about different people remain separate)
- Temporal modeling of fact evolution (we don't track how facts change over time)
- Cross-fact-type merging (WORKS_AT never merges with LIVES_IN)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Deduplication Pipeline                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  Stage 1: Partition by Type + Subject  │
         │  (fact_type, subject_id) groups        │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  Stage 2: Similarity Detection         │
         │                                         │
         │  ┌──────────────────────────────────┐  │
         │  │  Method A: MinHash + LSH         │  │
         │  │  (textual near-duplicates)       │  │
         │  └──────────────────────────────────┘  │
         │               ∪ (union)                 │
         │  ┌──────────────────────────────────┐  │
         │  │  Method B: Embedding Similarity  │  │
         │  │  (semantic equivalence)          │  │
         │  └──────────────────────────────────┘  │
         │                                         │
         │  Output: Candidate groups               │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  Stage 3: LLM Canonical Generation     │
         │  - Analyze each candidate group        │
         │  - Output canonical facts              │
         │  - Provide merge reasoning             │
         └────────────────────────────────────────┘
                              │
                              ▼
         ┌────────────────────────────────────────┐
         │  Stage 4: Database Update              │
         │  - Insert canonical facts              │
         │  - Insert merged evidence              │
         │  - Delete original facts               │
         │  - Record audit trail                  │
         └────────────────────────────────────────┘
```

---

## Strategy Details

### Stage 1: Partition by Fact Type and Subject

**Purpose**: Reduce comparison space by grouping facts that could potentially be duplicates.

**Algorithm**:
1. Query all facts from the database with `graph_synced_at IS NOT NULL` (only deduplicate facts that have been materialized to Neo4j)
2. Group facts by `(fact_type, subject_id)` tuples
3. Skip partitions with fewer than 2 facts (no duplicates possible)
4. Process each partition independently

**Key insight**: Facts can only be duplicates if they describe the same person (subject_id) and the same type of relationship (fact_type).

**Complexity**: O(n) where n is the number of facts

**Example partition**:
```
Partition: (fact_type='WORKS_AT', subject_id='12345')
Facts: [147, 148, 150, 152, 189]
```

---

### Stage 2: Similarity Detection

Within each partition, identify candidate groups of potentially duplicate facts using two complementary methods.

#### Method A: MinHash + LSH (Textual Near-Duplicates)

**Purpose**: Detect facts that differ only by minor textual variations (typos, formatting, abbreviations).

**Algorithm**:
1. For each fact, extract key textual attributes based on fact type:
   - `WORKS_AT`: organization, role, location
   - `LIVES_IN`: location
   - `TALKS_ABOUT`: topic
   - `STUDIED_AT`: institution, degree_type, field_of_study
   - `HAS_SKILL`: skill, proficiency_level
   - etc.

2. Concatenate these attributes into a single text representation:
   ```
   "Google|Software Engineer|San Francisco"
   ```

3. Tokenize using character n-grams (n=3):
   ```
   ["Goo", "oog", "ogl", "gle", ...]
   ```

4. Generate MinHash signature (128 hash functions recommended):
   ```python
   from datasketch import MinHash
   m = MinHash(num_perm=128)
   for token in tokens:
       m.update(token.encode('utf-8'))
   ```

5. Insert into LSH index with Jaccard threshold of 0.7:
   ```python
   from datasketch import MinHashLSH
   lsh = MinHashLSH(threshold=0.7, num_perm=128)
   lsh.insert(fact_id, minhash_signature)
   ```

6. Query LSH for each fact to find similar facts:
   ```python
   candidates = lsh.query(minhash_signature)
   ```

**Output**: Set of fact ID pairs with high textual similarity

**Parameters**:
- `num_perm=128`: Number of hash functions (trade-off: accuracy vs speed)
- `threshold=0.7`: Jaccard similarity threshold (0.7 = 70% overlap)
- `ngram_size=3`: Character n-gram size

**Libraries**: `datasketch` (Python implementation of MinHash/LSH)

#### Method B: Embedding-Based Semantic Similarity

**Purpose**: Detect facts that are semantically equivalent but lexically different.

**Algorithm**:
1. For each fact, create a structured text representation:
   ```
   Fact type: WORKS_AT
   Subject: person_12345
   Organization: Google
   Role: Software Engineer
   Location: San Francisco
   ```

2. Generate embedding using EmbeddingGemma or similar model:
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('google/gemma-2-9b-it-embedding')
   embedding = model.encode(fact_text)
   ```

3. Store embeddings in a vector database or in-memory structure

4. For each fact, find k-nearest neighbors using cosine similarity:
   ```python
   from sklearn.metrics.pairwise import cosine_similarity
   similarities = cosine_similarity([fact_embedding], all_embeddings)
   candidates = similarities[0] > 0.85  # threshold
   ```

5. Group facts that exceed the similarity threshold

**Output**: Set of fact ID groups with high semantic similarity

**Parameters**:
- `embedding_model`: "google/gemma-2-9b-it-embedding" or alternative
- `similarity_threshold=0.85`: Cosine similarity threshold (85% similarity)
- `batch_size=32`: Batch size for embedding generation

**Optimization**: Use FAISS or Annoy for approximate nearest neighbor search if dealing with 10k+ facts per partition

**Libraries**: 
- `sentence-transformers` for embedding generation
- `faiss-cpu` or `faiss-gpu` for efficient similarity search (optional)
- `annoy` as lightweight alternative to FAISS

#### Combining Methods A and B

**Algorithm**:
1. Collect all candidate pairs from MinHash+LSH (Method A)
2. Collect all candidate pairs from embedding similarity (Method B)
3. Take the union of both sets
4. Build connected components to form candidate groups:
   ```
   If MinHash says: (147, 148), (148, 150)
   And Embeddings say: (150, 152)
   Then group: [147, 148, 150, 152]
   ```

5. Use Union-Find (Disjoint Set Union) algorithm for efficient grouping:
   ```python
   class UnionFind:
       def __init__(self, n):
           self.parent = list(range(n))
       
       def find(self, x):
           if self.parent[x] != x:
               self.parent[x] = self.find(self.parent[x])
           return self.parent[x]
       
       def union(self, x, y):
           px, py = self.find(x), self.find(y)
           if px != py:
               self.parent[px] = py
   ```

**Output**: List of candidate groups, where each group contains fact IDs that should be reviewed together

**Example output**:
```python
[
    [147, 148, 150],      # Group 1: 3 similar WORKS_AT facts
    [152, 189],           # Group 2: 2 similar WORKS_AT facts
    [201],                # Group 3: singleton (no duplicates found)
    [205, 206, 207, 208]  # Group 4: 4 similar facts
]
```

---

### Stage 3: LLM Canonical Fact Generation

**Purpose**: Use LLM judgment to determine which facts in a candidate group should be merged and produce canonical output facts.

#### Input Format

For each candidate group, construct a prompt containing:

```json
{
  "task": "deduplicate_facts",
  "fact_type": "WORKS_AT",
  "subject_id": "12345",
  "subject_name": "John Smith",
  "candidate_facts": [
    {
      "fact_id": 147,
      "object_label": "Google",
      "attributes": {
        "organization": "Google",
        "role": "Software Engineer",
        "location": "San Francisco"
      },
      "confidence": 0.85,
      "evidence": ["msg_001", "msg_045"],
      "timestamp": "2024-01-15T10:23:00Z",
      "notes": "Explicit statement about current role"
    },
    {
      "fact_id": 148,
      "object_label": "Google Inc.",
      "attributes": {
        "organization": "Google Inc.",
        "role": "SWE",
        "location": "SF"
      },
      "confidence": 0.78,
      "evidence": ["msg_102"],
      "timestamp": "2024-01-20T14:30:00Z",
      "notes": "Casual mention in conversation"
    }
  ]
}
```

#### LLM Prompt Template

```
You are a knowledge graph deduplication specialist. Your task is to analyze a group of potentially duplicate facts and output a consolidated list of canonical facts.

FACT TYPE: {fact_type}
SUBJECT: {subject_name} (ID: {subject_id})

CANDIDATE FACTS:
{json_formatted_facts}

RULES FOR MERGING:

1. Merge Criteria - Combine facts if they describe the same real-world assertion with:
   - Minor textual variations (abbreviations, typos, formatting)
   - Complementary attributes that don't conflict
   - Same time period (if temporal attributes exist)

2. Keep Separate - Do NOT merge facts if:
   - They describe different entities (e.g., two different companies)
   - Temporal attributes indicate different time periods
   - Core attributes fundamentally conflict
   - They represent distinct relationships

3. Confidence Score Rules:
   - If merging identical facts: use MAX(confidence_scores)
   - If merging facts with complementary info: use WEIGHTED_AVERAGE + 0.05 boost
   - If merged fact has 5+ evidence messages: add 0.1 confidence boost
   - Cap confidence at 0.95 (never claim absolute certainty)

4. Evidence Merging:
   - Take UNION of all evidence message IDs
   - Remove duplicates
   - Use earliest timestamp as canonical timestamp

5. Attribute Resolution:
   - Prefer more specific/complete attributes
   - Normalize variations (e.g., "Google Inc." → "Google")
   - If attributes conflict, keep both facts separate
   - Fill missing attributes from other facts when possible

OUTPUT FORMAT:
Return valid JSON with this exact structure:

{
  "canonical_facts": [
    {
      "type": "WORKS_AT",
      "subject_id": "12345",
      "object_label": "Normalized Organization Name",
      "attributes": {
        "organization": "...",
        "role": "...",
        "location": "..."
      },
      "confidence": 0.90,
      "evidence": ["msg_001", "msg_045", "msg_102"],
      "timestamp": "2024-01-15T10:23:00Z",
      "merged_from": [147, 148],
      "merge_reasoning": "Two facts about same Google employment. Combined specific role from fact 147 with casual mention in fact 148. Normalized 'Google Inc.' to 'Google' and 'SF' to 'San Francisco'."
    }
  ]
}

IMPORTANT:
- You may output 1 fact (full merge), N facts (no merge), or anything in between
- Always include merge_reasoning to explain your decisions
- Ensure all JSON is valid and parseable
- Do not invent information not present in the input facts
```

#### LLM Configuration

**Model**: Use a capable model with good reasoning (e.g., `claude-sonnet-4`, `gpt-4`, `gemini-pro`)

**Parameters**:
- `temperature=0.2`: Low temperature for consistent, deterministic output
- `max_tokens=4096`: Enough for groups with 10+ facts
- `response_format={"type": "json_object"}`: Enforce JSON output if supported

**Error Handling**:
- If LLM returns invalid JSON: retry up to 3 times with same prompt
- If still fails: log error, skip this candidate group, continue with next
- If LLM returns empty `canonical_facts`: treat as no merge, preserve originals

#### Output Validation

Before accepting LLM output, validate:
1. Valid JSON structure
2. All required fields present (`type`, `subject_id`, `attributes`, `confidence`, `evidence`)
3. Confidence scores between 0 and 1
4. All `merged_from` fact IDs exist in the input group
5. Evidence message IDs are valid (exist in database)
6. No duplicate canonical facts in output

---

### Stage 4: Database Update

**Purpose**: Persist the canonical facts and delete originals while maintaining audit trail.

#### Transaction Structure

Each candidate group update must be atomic (all-or-nothing):

```sql
BEGIN TRANSACTION;

-- 1. Insert canonical facts
INSERT INTO fact (ie_run_id, type, subject_id, object_id, object_type, attributes, ts, confidence)
VALUES (?, ?, ?, ?, ?, ?, ?, ?);
-- Get last_insert_rowid() as canonical_fact_id

-- 2. Insert evidence for canonical fact
INSERT INTO fact_evidence (fact_id, message_id)
VALUES (canonical_fact_id, 'msg_001'),
       (canonical_fact_id, 'msg_045'),
       (canonical_fact_id, 'msg_102');

-- 3. Record audit trail
INSERT INTO fact_deduplication_audit (
    canonical_fact_id, 
    original_fact_ids, 
    merge_reasoning,
    similarity_scores
) VALUES (?, ?, ?, ?);

-- 4. Delete original facts (cascade deletes evidence)
DELETE FROM fact WHERE id IN (147, 148);

COMMIT;
```

#### Audit Schema

```sql
CREATE TABLE IF NOT EXISTS fact_deduplication_audit (
    id INTEGER PRIMARY KEY,
    canonical_fact_id INTEGER REFERENCES fact(id) ON DELETE CASCADE,
    original_fact_ids TEXT NOT NULL,  -- JSON array: [147, 148]
    merge_reasoning TEXT,
    similarity_scores TEXT,  -- JSON: {"minhash": 0.85, "embedding": 0.92}
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dedup_audit_canonical 
    ON fact_deduplication_audit(canonical_fact_id);
```

#### Progress Tracking Schema

```sql
CREATE TABLE IF NOT EXISTS deduplication_run (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'paused')),
    total_partitions INTEGER,
    processed_partitions INTEGER DEFAULT 0,
    facts_processed INTEGER DEFAULT 0,
    facts_merged INTEGER DEFAULT 0,
    candidate_groups_processed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS deduplication_partition_progress (
    run_id INTEGER NOT NULL REFERENCES deduplication_run(id) ON DELETE CASCADE,
    fact_type TEXT NOT NULL,
    subject_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'in_progress', 'completed', 'failed')),
    processed_at TEXT,
    PRIMARY KEY (run_id, fact_type, subject_id)
);
```

---

## Implementation Plan

### Module Structure

```
deduplicate/
├── __init__.py
├── core.py                 # Main orchestration logic
├── partitioning.py         # Stage 1: Partition facts
├── similarity/
│   ├── __init__.py
│   ├── minhash_lsh.py     # MinHash + LSH implementation
│   ├── embeddings.py      # Embedding generation and similarity
│   └── grouping.py        # Union-Find and candidate grouping
├── llm/
│   ├── __init__.py
│   ├── client.py          # LLM client (reuse ie/client.py pattern)
│   ├── prompts.py         # Prompt templates
│   └── parser.py          # Parse and validate LLM output
├── persistence.py          # Stage 4: Database updates
└── progress.py            # Progress tracking and resumability

deduplicate_facts.py        # CLI entry point
```

### Phase 1: Foundation (Week 1)

**Goal**: Set up project structure, schemas, and basic partitioning

**Tasks**:
1. Create `deduplicate/` module structure
2. Add audit and progress tracking tables to `schema.sql`
3. Implement `partitioning.py`:
   ```python
   class FactPartitioner:
       def __init__(self, conn: sqlite3.Connection):
           self.conn = conn
       
       def get_partitions(self) -> Iterator[Partition]:
           """Yield (fact_type, subject_id, fact_ids) tuples."""
           ...
       
       def count_partitions(self) -> int:
           """Count total partitions for progress tracking."""
           ...
   ```
4. Implement `progress.py`:
   ```python
   class DeduplicationProgress:
       def start_run(self) -> int:
           """Create new deduplication_run, return run_id."""
           ...
       
       def mark_partition_completed(self, run_id, fact_type, subject_id):
           ...
       
       def is_partition_completed(self, run_id, fact_type, subject_id) -> bool:
           ...
   ```

**Deliverables**:
- Working partition iteration
- Progress tracking infrastructure
- Unit tests for partitioning logic

### Phase 2: Similarity Detection (Week 2)

**Goal**: Implement MinHash+LSH and embedding-based similarity

**Tasks**:

1. Implement `minhash_lsh.py`:
   ```python
   class MinHashLSHDetector:
       def __init__(self, threshold: float = 0.7, num_perm: int = 128):
           self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
           self.num_perm = num_perm
       
       def extract_text(self, fact: FactRecord) -> str:
           """Extract key attributes as concatenated text."""
           ...
       
       def generate_minhash(self, text: str) -> MinHash:
           """Generate MinHash signature from text."""
           ...
       
       def find_candidates(self, facts: List[FactRecord]) -> Set[Tuple[int, int]]:
           """Return set of (fact_id1, fact_id2) pairs."""
           ...
   ```

2. Implement `embeddings.py`:
   ```python
   class EmbeddingSimilarityDetector:
       def __init__(self, 
                    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                    threshold: float = 0.85,
                    use_gpu: bool = False):
           self.model = SentenceTransformer(model_name)
           self.threshold = threshold
           if use_gpu:
               self.model = self.model.to('cuda')
       
       def format_fact_text(self, fact: FactRecord) -> str:
           """Create structured text representation."""
           ...
       
       def generate_embeddings(self, facts: List[FactRecord]) -> np.ndarray:
           """Batch generate embeddings."""
           ...
       
       def find_candidates(self, facts: List[FactRecord]) -> Set[Tuple[int, int]]:
           """Return set of (fact_id1, fact_id2) pairs using cosine similarity."""
           ...
   ```

3. Implement `grouping.py`:
   ```python
   class CandidateGrouper:
       def __init__(self):
           self.uf = UnionFind()
       
       def merge_candidate_sets(self, 
                               minhash_pairs: Set[Tuple[int, int]],
                               embedding_pairs: Set[Tuple[int, int]]) -> List[Set[int]]:
           """Combine both methods and return connected components."""
           ...
   ```

4. Add embedding caching:
   ```python
   # Cache embeddings to disk to avoid recomputation
   @lru_cache(maxsize=10000)
   def get_cached_embedding(fact_id: int, fact_text: str) -> np.ndarray:
       cache_path = Path(f".cache/embeddings/{fact_id}.npy")
       if cache_path.exists():
           return np.load(cache_path)
       embedding = model.encode(fact_text)
       cache_path.parent.mkdir(exist_ok=True)
       np.save(cache_path, embedding)
       return embedding
   ```

**Deliverables**:
- Working MinHash+LSH detector
- Working embedding similarity detector
- Candidate group generation
- Unit tests with known duplicate examples

### Phase 3: LLM Integration (Week 3)

**Goal**: Implement LLM-based canonical fact generation

**Tasks**:

1. Implement `llm/client.py` (adapt from `ie/client.py`):
   ```python
   class DeduplicationLLMClient:
       def __init__(self, config: LLMConfig):
           self.config = config
           self.client = httpx.Client(...)
       
       def generate_canonical_facts(self, 
                                   candidate_group: List[FactRecord]) -> CanonicalFactsResult:
           """Send candidate group to LLM, return parsed canonical facts."""
           ...
   ```

2. Implement `llm/prompts.py`:
   ```python
   def build_deduplication_prompt(
       fact_type: FactType,
       subject_id: str,
       subject_name: str,
       candidates: List[FactRecord]
   ) -> List[Dict[str, str]]:
       """Build messages array for LLM."""
       ...
   ```

3. Implement `llm/parser.py`:
   ```python
   class CanonicalFactsParser:
       def parse(self, llm_response: str) -> CanonicalFactsResult:
           """Parse and validate LLM JSON output."""
           ...
       
       def validate(self, result: CanonicalFactsResult, 
                   input_facts: List[FactRecord]) -> bool:
           """Validate output references valid input fact IDs, etc."""
           ...
   ```

4. Add retry logic with exponential backoff:
   ```python
   MAX_RETRIES = 3
   for attempt in range(MAX_RETRIES):
       try:
           response = client.generate_canonical_facts(candidates)
           if validate(response):
               return response
       except (JSONDecodeError, ValidationError) as e:
           if attempt == MAX_RETRIES - 1:
               raise
           time.sleep(2 ** attempt)  # exponential backoff
   ```

**Deliverables**:
- Working LLM client with retry logic
- Validated prompt templates
- Response parser with validation
- Integration tests with mock LLM responses

### Phase 4: Persistence Layer (Week 4)

**Goal**: Implement database updates with audit trail

**Tasks**:

1. Implement `persistence.py`:
   ```python
   class DeduplicationPersistence:
       def __init__(self, conn: sqlite3.Connection):
           self.conn = conn
       
       def apply_canonical_facts(self,
                                 run_id: int,
                                 canonical_facts: List[CanonicalFact],
                                 original_fact_ids: List[int],
                                 similarity_scores: Dict[str, float]):
           """
           Atomically:
           1. Insert canonical facts
           2. Insert evidence
           3. Record audit trail
           4. Delete originals
           """
           with self.conn:  # transaction
               for canonical in canonical_facts:
                   fact_id = self._insert_fact(canonical)
                   self._insert_evidence(fact_id, canonical.evidence)
                   self._insert_audit(fact_id, canonical.merged_from, 
                                    canonical.merge_reasoning, similarity_scores)
               self._delete_facts(original_fact_ids)
   ```

2. Add safety checks:
   ```python
   def validate_evidence_exists(self, message_ids: List[str]) -> bool:
       """Ensure all message IDs exist before persisting."""
       placeholders = ','.join('?' * len(message_ids))
       result = self.conn.execute(
           f"SELECT COUNT(*) FROM message WHERE id IN ({placeholders})",
           message_ids
       )
       return result.fetchone()[0] == len(message_ids)
   ```

3. Add rollback on error:
   ```python
   try:
       apply_canonical_facts(...)
   except Exception as e:
       logger.error(f"Failed to apply canonical facts: {e}")
       # Transaction automatically rolled back
       raise
   ```

**Deliverables**:
- Atomic database updates
- Complete audit trail
- Error handling and rollback
- Integration tests with real database

### Phase 5: CLI and Orchestration (Week 5)

**Goal**: Tie everything together with a CLI tool

**Tasks**:

1. Implement `core.py`:
   ```python
   class DeduplicationOrchestrator:
       def __init__(self, 
                    conn: sqlite3.Connection,
                    minhash_detector: MinHashLSHDetector,
                    embedding_detector: EmbeddingSimilarityDetector,
                    llm_client: DeduplicationLLMClient,
                    persistence: DeduplicationPersistence):
           self.conn = conn
           self.minhash = minhash_detector
           self.embeddings = embedding_detector
           self.llm = llm_client
           self.persistence = persistence
       
       def run(self, resume: bool = False) -> DeduplicationStats:
           """Main entry point for deduplication."""
           run_id = self._setup_run(resume)
           
           for partition in self.partitioner.get_partitions():
               if self._is_completed(run_id, partition):
                   continue
               
               candidate_groups = self._find_candidates(partition)
               
               for group in candidate_groups:
                   canonical = self.llm.generate_canonical_facts(group)
                   self.persistence.apply_canonical_facts(...)
               
               self._mark_completed(run_id, partition)
           
           return self._finalize_run(run_id)
   ```

2. Implement `deduplicate_facts.py` CLI:
   ```python
   def parse_args():
       parser = argparse.ArgumentParser()
       parser.add_argument('--sqlite', default='./discord.db')
       parser.add_argument('--minhash-threshold', type=float, default=0.7)
       parser.add_argument('--embedding-threshold', type=float, default=0.85)
       parser.add_argument('--embedding-model', 
                          default='sentence-transformers/all-MiniLM-L6-v2')
       parser.add_argument('--llm-url', default='http://localhost:8080/...')
       parser.add_argument('--llm-model', default='claude-sonnet-4')
       parser.add_argument('--batch-size', type=int, default=32)
       parser.add_argument('--resume', action='store_true')
       parser.add_argument('--max-partitions', type=int, 
                          help='Process only N partitions (for testing)')
       return parser.parse_args()
   ```

3. Add progress reporting:
   ```python
   def print_progress(stats: DeduplicationStats):
       print(f"Processed: {stats.processed_partitions}/{stats.total_partitions} partitions")
       print(f"Facts merged: {stats.facts_merged}")
       print(f"Candidate groups: {stats.candidate_groups_processed}")
       print(f"Time elapsed: {stats.elapsed_time:.1f}s")
   ```

**Deliverables**:
- Complete CLI tool
- Progress reporting
- Resumability
- End-to-end tests

### Phase 6: Testing and Optimization (Week 6)

**Goal**: Validate correctness and optimize performance

**Tasks**:

1. Create test fixtures:
   ```python
   # tests/fixtures/duplicate_facts.json
   [
       {
           "fact_type": "WORKS_AT",
           "subject_id": "test_001",
           "duplicates": [
               {"organization": "Google", "role": "SWE"},
               {"organization": "Google Inc.", "role": "Software Engineer"},
               {"organization": "Google", "role": "Software Engineer", "location": "SF"}
           ],
           "expected_canonical": {
               "organization": "Google",
               "role": "Software Engineer", 
               "location": "San Francisco"
           }
       }
   ]
   ```

2. Add performance benchmarks:
   ```python
   def benchmark_similarity_detection():
       facts = generate_test_facts(n=10000)
       
       start = time.time()
       minhash_pairs = minhash_detector.find_candidates(facts)
       minhash_time = time.time() - start
       
       start = time.time()
       embedding_pairs = embedding_detector.find_candidates(facts)
       embedding_time = time.time() - start
       
       print(f"MinHash: {minhash_time:.2f}s ({len(minhash_pairs)} pairs)")
       print(f"Embeddings: {embedding_time:.2f}s ({len(embedding_pairs)} pairs)")
   ```

3. Optimize bottlenecks:
   - Batch embedding generation (32-64 facts at a time)
   - Use FAISS for large partitions (1000+ facts)
   - Parallelize partition processing
   - Cache embeddings across runs

4. Add dry-run mode:
   ```python
   parser.add_argument('--dry-run', action='store_true',
                      help='Show what would be merged without applying changes')
   ```

**Deliverables**:
- Comprehensive test suite
- Performance benchmarks
- Optimizations applied
- Documentation updates

---

## Configuration

### Default Configuration File

```yaml
# deduplicate_config.yaml

database:
  path: "./discord.db"

similarity:
  minhash:
    threshold: 0.7
    num_perm: 128
    ngram_size: 3
  
  embeddings:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    threshold: 0.85
    batch_size: 32
    use_gpu: false
    cache_dir: ".cache/embeddings"

llm:
  base_url: "http://localhost:8080/v1/chat/completions"
  model: "claude-sonnet-4"
  temperature: 0.2
  max_tokens: 4096
  timeout: 120.0
  max_retries: 3

processing:
  max_concurrent_llm_calls: 5
  checkpoint_every: 100  # Save progress every N partitions
  min_confidence_to_deduplicate: 0.5  # Only deduplicate facts above this confidence

logging:
  level: "INFO"
  file: "deduplication.log"
```

---

## Testing Strategy

### Unit Tests

1. **Partitioning** (`test_partitioning.py`)
   - Correct grouping by (fact_type, subject_id)
   - Empty database handling
   - Single-fact partitions skipped

2. **MinHash/LSH** (`test_minhash_lsh.py`)
   - Identical strings → similarity 1.0
   - Minor variations detected (typos, abbreviations)
   - Different strings → low similarity

3. **Embeddings** (`test_embeddings.py`)
   - Semantic equivalence detected ("Google" ≈ "Alphabet")
   - Unrelated concepts → low similarity
   - Batch processing correctness

4. **Candidate Grouping** (`test_grouping.py`)
   - Union of MinHash and embedding results
   - Connected components correctly formed
   - Singleton groups handled

5. **LLM Parsing** (`test_llm_parser.py`)
   - Valid JSON parsed correctly
   - Invalid JSON rejected with clear errors
   - Confidence score validation
   - Evidence validation

6. **Persistence** (`test_persistence.py`)
   - Atomic transactions (all-or-nothing)
   - Audit trail completeness
   - Evidence correctly merged
   - Rollback on error

### Integration Tests

1. **End-to-End** (`test_e2e.py`)
   - Load test fixtures with known duplicates
   - Run full deduplication pipeline
   - Assert expected canonical facts created
   - Assert original facts deleted
   - Assert audit trail recorded

2. **Resumability** (`test_resumability.py`)
   - Start run, process 50% of partitions, interrupt
   - Resume run, verify remaining partitions processed
   - No duplicate processing

3. **Error Recovery** (`test_error_recovery.py`)
   - LLM timeout → partition marked failed, continue
   - Invalid LLM response → logged, skipped
   - Database error → transaction rolled back

### Performance Tests

1. **Scalability** (`test_performance.py`)
   - 1,000 facts: < 30 seconds
   - 10,000 facts: < 5 minutes
   - 100,000 facts: < 1 hour

2. **Memory Usage**
   - Monitor memory during large partition processing
   - Ensure embeddings don't cause OOM

---

## Deployment Checklist

- [ ] Schema migrations applied (`deduplication_run`, `deduplication_partition_progress`, `fact_deduplication_audit`)
- [ ] Dependencies installed (`datasketch`, `sentence-transformers`, `faiss-cpu` optional)
- [ ] Embedding model downloaded and cached
- [ ] LLM endpoint accessible and tested
- [ ] Configuration file created and validated
- [ ] Test run on small dataset (100 facts) successful
- [ ] Dry-run mode tested and output reviewed
- [ ] Backup database before first production run
- [ ] Monitoring/logging configured
- [ ] Documentation updated with usage examples

---

## Usage Examples

### Basic Deduplication

```bash
# First run (process all facts)
python deduplicate_facts.py \
    --sqlite ./discord.db \
    --llm-url http://localhost:8080/v1/chat/completions \
    --llm-model claude-sonnet-4

# Resume interrupted run
python deduplicate_facts.py --resume

# Dry run (show what would happen without changes)
python deduplicate_facts.py --dry-run
```

### Custom Configuration

```bash
python deduplicate_facts.py \
    --minhash-threshold 0.8 \
    --embedding-threshold 0.9 \
    --embedding-model google/gemma-2-9b-it-embedding \
    --batch-size 64 \
    --max-partitions 10  # Process only 10 partitions for testing
```

### Integration with Pipeline

```bash
# Run after IE and fact materialization
python run_pipeline.py --neo4j-password test  # Includes IE and facts_to_graph
python deduplicate_facts.py  # Now deduplicate
python facts_to_graph.py --password test  # Re-sync canonical facts to Neo4j
```

---

## Monitoring and Observability

### Metrics to Track

1. **Processing Metrics**
   - Partitions processed per minute
   - Facts processed per minute
   - Candidate groups identified per partition
   - Average group size

2. **Merge Metrics**
   - Total facts merged
   - Merge ratio (original facts / canonical facts)
   - Average confidence score change
   - Facts with confidence boost vs. decrease

3. **Quality Metrics**
   - LLM retry rate
   - Failed partitions (errors)
   - Singleton groups (no duplicates found)
   - Large groups (10+ facts, may need review)

4. **Performance Metrics**
   - MinHash processing time per partition
   - Embedding generation time per partition
   - LLM call duration (P50, P95, P99)
   - Database transaction time

### Logging

```python
import logging

logger = logging.getLogger('deduplicate')
logger.setLevel(logging.INFO)

# Log key events
logger.info(f"Starting partition {fact_type}/{subject_id} with {len(facts)} facts")
logger.info(f"Found {len(candidate_groups)} candidate groups via similarity detection")
logger.warning(f"LLM retry #{attempt} for group {group_id}")
logger.error(f"Failed to process partition: {error}")
```

---

## Future Enhancements

### Phase 7+ (Optional)

1. **Active Learning**
   - Track LLM merge decisions
   - Build training dataset for supervised model
   - Train smaller, faster model for common cases
   - Fall back to LLM for edge cases

2. **Cross-Subject Deduplication**
   - Detect when two subject_ids refer to same person
   - Merge person profiles (requires Neo4j updates)

3. **Temporal Modeling**
   - Track fact changes over time (WORKS_AT with start/end dates)
   - Build fact history rather than single snapshots

4. **Interactive Review**
   - Web UI to review large candidate groups
   - Human-in-the-loop for low-confidence merges

5. **Incremental Deduplication**
   - Run after each IE batch
   - Only process new facts
   - Maintain "stable" canonical facts

---

## Appendix: Data Structures

### FactRecord

```python
@dataclass
class FactRecord:
    id: int
    type: FactType
    subject_id: str
    subject_name: str | None
    object_label: str | None
    object_id: str | None
    attributes: dict[str, Any]
    confidence: float
    evidence: list[str]
    timestamp: str
    notes: str | None
```

### Partition

```python
@dataclass
class Partition:
    fact_type: FactType
    subject_id: str
    subject_name: str | None
    fact_ids: list[int]
```

### CandidateGroup

```python
@dataclass
class CandidateGroup:
    partition: Partition
    fact_ids: set[int]
    similarity_scores: dict[str, float]  # {"minhash": 0.85, "embedding": 0.92}
```

### CanonicalFact

```python
@dataclass
class CanonicalFact:
    type: FactType
    subject_id: str
    object_label: str | None
    object_id: str | None
    attributes: dict[str, Any]
    confidence: float
    evidence: list[str]
    timestamp: str
    merged_from: list[int]
    merge_reasoning: str
```

### DeduplicationStats

```python
@dataclass
class DeduplicationStats:
    run_id: int
    total_partitions: int
    processed_partitions: int
    facts_processed: int
    facts_merged: int
    candidate_groups_processed: int
    average_group_size: float
    elapsed_time: float
```

---

## References

- MinHash/LSH: Broder, Andrei Z. "On the resemblance and containment of documents." (1997)
- `datasketch` library: https://github.com/ekzhu/datasketch
- Sentence Transformers: https://www.sbert.net/
- EmbeddingGemma: https://huggingface.co/google/gemma-2-9b-it-embedding
- FAISS (approximate nearest neighbors): https://github.com/facebookresearch/faiss
