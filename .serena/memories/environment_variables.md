# Environment Variables

## Neo4j
- `NEO4J_URI`: Connection URI (default: `bolt://localhost:7687`)
- `NEO4J_USER`: Username (default: `neo4j`)
- `NEO4J_PASSWORD`: Password (default: `test`)
- `NEO4J_DATABASE`: Optional database name

## LLM (llama-server)
- `LLAMA_MODEL`: Model name (default: `GLM-4.5-Air`)
- `LLAMA_BASE_URL`: API endpoint (default: `http://localhost:8080/v1/chat/completions`)
- `LLAMA_API_KEY`: Optional API key
- `LLAMA_TEMPERATURE`: Sampling temperature (default: `0.3`)
- `LLAMA_TOP_P`: Nucleus sampling (default: `0.95`)
- `LLAMA_MAX_TOKENS`: Max completion tokens (default: `4096`)
- `LLAMA_TIMEOUT`: Request timeout in seconds (default: `1200`)

## Embeddings
- `EMBEDDING_MODEL`: Model name (default: `BAAI/bge-large-en-v1.5`)
- `EMBEDDING_DEVICE`: Device (default: `cpu`)
- `EMBEDDING_CACHE_DIR`: Optional cache directory

## Message Embeddings Job
- `MESSAGE_EMBEDDING_MODEL`: Override model for message embeddings
- `MESSAGE_EMBEDDING_DEVICE`: Override device for message embeddings
- `MESSAGE_EMBEDDING_CACHE_DIR`: Optional cache dir
- `MESSAGE_EMBEDDING_BATCH_SIZE`: Batch size (default: `128`)

## Memory Agent Service
- `SQLITE_DB_PATH`: Path to SQLite database (default: `./discord.db`)
- `API_HOST`: FastAPI host (default: `0.0.0.0`)
- `API_PORT`: FastAPI port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `ENABLE_CORS`: Enable CORS (default: `false`)
- `CORS_ALLOW_ORIGINS`: Comma-separated allowed origins
- `ENABLE_DEBUG_ENDPOINT`: Enable debug endpoint (default: `false`)
- `MAX_ITERATIONS`: Agent max iterations (default: `10`)
- `MAX_FACTS`: Max facts to return (default: `30`)

## Example .env File
See `.env.example` in project root for a template.
