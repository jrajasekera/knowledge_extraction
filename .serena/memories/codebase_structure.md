# Codebase Structure

## Root Directory
```
knowledge_extraction/
├── run_pipeline.py          # Full orchestration entry point
├── import_discord_json.py   # Discord JSON → SQLite importer
├── loader.py                # SQLite → Neo4j transformer
├── facts_to_graph.py        # Fact materialization to Neo4j
├── run_ie.py                # Information extraction entry point
├── deduplicate_facts.py     # Fact deduplication CLI
├── main.py                  # Alternative entry point
├── db_utils.py              # SQLite connection helpers (WAL mode)
├── constants.py             # Shared constants (embedding model, dimensions)
├── schema.sql               # SQLite schema (~20 tables)
├── ingest.cql               # Neo4j constraints and projections
├── create_fulltext_index.cql        # Fact fulltext index
├── create_message_fulltext_index.cql # Message fulltext index
└── pyproject.toml           # Dependencies and tool config
```

## ie/ - Information Extraction
```
ie/
├── __init__.py
├── runner.py           # IE job orchestration
├── windowing.py        # Message window generation
├── client.py           # llama-server API client
├── config.py           # 20+ fact type definitions
├── models.py           # Pydantic models (ExtractionFact, ExtractionResult)
├── types.py            # Type definitions
├── prompts.py          # Basic prompt templates
├── advanced_prompts.py # Enhanced prompt scaffolding
└── prompt_assets.json  # Few-shot examples (edit without code changes)
```

## memory_agent/ - FastAPI Microservice
```
memory_agent/
├── __init__.py
├── app.py                  # FastAPI application
├── agent.py                # LangGraph agentic workflow
├── config.py               # Environment-based config
├── models.py               # API request/response models
├── request_logger.py       # Database audit logging
├── llm.py                  # LLM integration
├── state.py                # Agent state management
├── embeddings.py           # Embedding generation
├── embedding_utils.py      # Embedding utilities
├── embedding_pipeline.py   # Fact embedding pipeline
├── message_embedding_pipeline.py # Message embedding pipeline
├── fact_formatter.py       # Fact formatting for responses
├── message_formatter.py    # Message formatting
├── normalization.py        # Data normalization
├── serialization.py        # JSON serialization helpers
├── conversation.py         # Conversation handling
├── context_summarizer.py   # Context summarization
└── tools/                  # Neo4j-backed retrieval tools
    ├── __init__.py
    └── *.py                # Person profile, timeline, semantic search, etc.
```

## deduplicate/ - Fact Deduplication
```
deduplicate/
├── __init__.py
├── core.py              # Orchestration and merge logic
├── models.py            # Deduplication models
├── partitioning.py      # Partition facts by type
├── persistence.py       # Database persistence
├── progress.py          # Progress tracking
├── display.py           # Display utilities
├── logging_support.py   # Logging helpers
├── similarity/          # Similarity detection
│   └── *.py            # MinHash LSH, embedding similarity
└── llm/                 # LLM verification
    └── *.py            # Merge decision making
```

## scripts/ - Utility Scripts
```
scripts/
├── enable_wal_mode.py      # Enable SQLite WAL mode
├── check_wal_status.py     # Check WAL mode status
├── embed_facts.py          # Generate fact embeddings
├── embed_messages.py       # Generate message embeddings
└── graph_snapshot.py       # Graph stats and validation
```

## tests/ - Test Suite
```
tests/
├── conftest.py             # Shared fixtures
├── test_ie_*.py            # IE module tests
├── test_memory_agent_*.py  # Memory agent tests
├── test_deduplicate_*.py   # Deduplication tests
└── test_*.py               # Other module tests
```

## Key Data Files
- `discord.db` - SQLite database (not in git)
- `data/` or `exports/` - Discord JSON exports (not in git)
