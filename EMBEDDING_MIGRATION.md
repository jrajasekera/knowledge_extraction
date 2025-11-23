# Embedding Model Migration Guide

## Migration: google/embeddinggemma-300m → BAAI/bge-large-en-v1.5

This guide walks you through migrating from the EmbeddingGemma model to BGE-large-en-v1.5.

### Key Changes
- **Model**: `google/embeddinggemma-300m` → `BAAI/bge-large-en-v1.5`
- **Dimensions**: 768 → 1024
- **Parameters**: 308M → 335M

### Why BGE?
- State-of-the-art retrieval performance on MTEB benchmarks
- No instruction prefixes needed (simplified usage)
- Better similarity distribution for semantic search
- Widely adopted in production RAG systems

---

## Migration Steps

### Step 1: Drop Existing Neo4j Vector Indices

Connect to Neo4j and drop the old indices:

```bash
# Using cypher-shell
cypher-shell -a bolt://localhost:7687 -u neo4j -p 'test'
```

Run these commands in cypher-shell:

```cypher
// Drop fact embeddings index
DROP INDEX fact_embeddings IF EXISTS;

// Drop message embeddings index
DROP INDEX message_embeddings IF EXISTS;

// Drop fulltext indices (will be recreated automatically)
DROP INDEX fact_fulltext IF EXISTS;
DROP INDEX message_fulltext IF EXISTS;
```

**Alternative**: Use Neo4j Browser at http://localhost:7474 and run the same commands.

### Step 2: Verify Indices are Dropped

```cypher
SHOW INDEXES;
```

You should NOT see `fact_embeddings`, `message_embeddings`, `fact_fulltext`, or `message_fulltext` in the output.

### Step 3: Re-generate Fact Embeddings

The new model will be downloaded automatically on first use.

```bash
# Generate fact embeddings with the new model
uv run python scripts/embed_facts.py --cleanup --workers 4

# Expected output:
# - Downloads BAAI/bge-large-en-v1.5 (~1.3GB)
# - Creates new vector index with 1024 dimensions
# - Generates embeddings for all facts
```

**Note**: The first run will download the model, which may take a few minutes depending on your connection.

### Step 4: Re-generate Message Embeddings

```bash
# Generate message embeddings with the new model
uv run python scripts/embed_messages.py --cleanup --workers 4

# Expected output:
# - Uses cached BGE model from Step 3
# - Creates new vector index with 1024 dimensions
# - Generates embeddings for all messages
```

### Step 5: Verify New Indices

```cypher
SHOW INDEXES;
```

You should see:
- `fact_embeddings` - VECTOR index with 1024 dimensions
- `message_embeddings` - VECTOR index with 1024 dimensions
- `fact_fulltext` - FULLTEXT index
- `message_fulltext` - FULLTEXT index

### Step 6: Test Semantic Search

Try a test query to verify embeddings are working:

```cypher
// Test fact semantic search
CALL db.index.vector.queryNodes('fact_embeddings', 5, [/* 1024-dim vector */])
YIELD node, score
RETURN node.text AS text, score
LIMIT 5;
```

Or test via the Memory Agent API:

```bash
curl -X POST http://localhost:8000/api/memory/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "confidence": "medium"}'
```

---

## Performance Considerations

### Model Size
- **EmbeddingGemma**: ~300MB
- **BGE-large-en-v1.5**: ~1.3GB

The BGE model is larger but offers better retrieval quality.

### Embedding Speed
Both models have similar inference speed on CPU. For faster embedding generation:

```bash
# Use more workers (adjust based on CPU cores)
uv run python scripts/embed_facts.py --workers 8 --batch-size 64
```

### Storage Impact
- **1024-dim vs 768-dim**: ~33% more storage per embedding
- For 10,000 facts: ~40MB vs ~30MB (minimal impact)

---

## Rollback (if needed)

If you need to roll back to EmbeddingGemma:

1. Edit `constants.py`:
   ```python
   DEFAULT_EMBEDDING_MODEL = "google/embeddinggemma-300m"
   EMBEDDING_VECTOR_DIMENSIONS = 768
   ```

2. Drop BGE indices (same as Step 1)

3. Re-run embedding scripts (Steps 3-4)

---

## Troubleshooting

### Issue: "Model not found" error
**Solution**: Check internet connection. The model downloads from HuggingFace on first use.

### Issue: "Dimension mismatch" error
**Solution**: Ensure you dropped the old indices (Step 1). The old 768-dim indices are incompatible.

### Issue: Out of memory during embedding
**Solution**: Reduce batch size and workers:
```bash
uv run python scripts/embed_facts.py --batch-size 16 --workers 2
```

### Issue: Slow embedding generation
**Solution**: Increase workers (if you have CPU cores available):
```bash
uv run python scripts/embed_facts.py --workers 8
```

---

## Optional: Update .env File

If you want to override the default model via environment variable:

```bash
# Add to .env file
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
```

This is optional since the default is now set in `constants.py`.

---

## Timeline Estimate

For a typical database with 10,000 messages and 5,000 facts:

- **Step 1-2** (Drop indices): < 1 minute
- **Step 3** (Embed facts): 5-15 minutes (includes model download)
- **Step 4** (Embed messages): 10-30 minutes
- **Step 5-6** (Verify): < 1 minute

**Total**: ~20-45 minutes

---

## Questions?

See the model documentation:
- [BAAI/bge-large-en-v1.5 on HuggingFace](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [BGE Official Documentation](https://bge-model.com/)
- [Sentence Transformers Docs](https://www.sbert.net/)
