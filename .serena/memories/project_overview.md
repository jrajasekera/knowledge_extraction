# Project Overview: Knowledge Extraction

## Purpose
A knowledge extraction system that transforms Discord message exports into a multi-layer knowledge graph. The system uses LLM-powered information extraction to discover relationships, skills, organizations, and other structured facts from unstructured Discord conversations.

## Architecture: Three-Layer Data Flow

1. **SQLite (Staging & Provenance)**
   - Lossless Discord data storage (~20 tables)
   - IE provenance tracking: `ie_run`, `fact`, `fact_evidence`
   - API audit logging: `memory_agent_request_log`

2. **Neo4j (Graph Database)**
   - Core entities: `Person`, `Guild`, `Channel`, `Message`, `Role`
   - Derived entities: `Topic`, `Org`, `Place`, `Project`, `Skill`, `Event`
   - Relationships: `SENT`, `REPLIES_TO`, `MENTIONS`, `INTERACTED_WITH`, `WORKS_AT`, `LIVES_IN`, `HAS_SKILL`, etc.

3. **LLM Information Extraction**
   - Uses local llama-server (OpenAI-compatible API)
   - Windowed extraction: processes conversations in sliding windows
   - 20+ fact types with confidence thresholding

## Tech Stack
- **Python 3.13+** (via pyenv)
- **Package manager**: uv
- **Databases**: SQLite (staging), Neo4j 5.22+ (graph)
- **LLM**: llama-server with GLM-4.5-Air model
- **Embeddings**: sentence-transformers (BAAI/bge-large-en-v1.5)
- **Web framework**: FastAPI + uvicorn
- **Agent framework**: LangGraph + LangChain
- **Data models**: Pydantic v2
- **JSON parsing**: orjson

## Key Modules
- `run_pipeline.py` - Full orchestration: ingest → IE → fact materialization
- `import_discord_json.py` - Discord JSON to SQLite importer
- `loader.py` - SQLite to Neo4j transformer
- `facts_to_graph.py` - High-confidence fact materialization
- `ie/` - Information extraction subsystem
- `memory_agent/` - FastAPI microservice for fact retrieval
- `deduplicate/` - Fact deduplication with MinHash + embeddings + LLM
- `scripts/` - Utility scripts (embeddings, graph inspection)
