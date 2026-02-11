"""Runtime configuration helpers for the memory agent service."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from constants import DEFAULT_EMBEDDING_MODEL

# Load environment variables from a .env file if present.
load_dotenv()


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
ConfidenceLevel = Literal["high", "medium", "low"]


@dataclass(frozen=True, slots=True)
class Neo4jConfig:
    """Connection details for the Neo4j database."""

    uri: str
    user: str
    password: str
    database: str | None = None
    max_connection_lifetime: int = 60
    encrypted: bool = False


@dataclass(frozen=True, slots=True)
class SQLiteConfig:
    """Configuration for the SQLite database."""

    db_path: Path


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """Configuration for the backing LLM provider."""

    model: str
    temperature: float = 0.3
    api_key: str | None = None
    base_url: str | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for the semantic search embedding model."""

    model: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cpu"
    cache_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class MessageEmbeddingConfig:
    """Configuration for the message embedding maintenance job."""

    model: str = DEFAULT_EMBEDDING_MODEL
    device: str = "cpu"
    cache_dir: Path | None = None
    batch_size: int = 128


@dataclass(frozen=True, slots=True)
class AgentConfig:
    """General agent runtime configuration."""

    max_iterations: int = 10
    max_facts: int = 30
    tool_timeout_seconds: int = 10
    tool_max_retries: int = 1
    reasoning_trace_limit: int = 20
    early_stop_min_iterations: int = 2
    novelty_min_new_facts: int = 1
    novelty_patience: int = 2
    stop_confidence_required: str = "high"
    fallback_max_queries: int = 12
    fallback_max_query_length: int = 120


@dataclass(frozen=True, slots=True)
class APIConfig:
    """FastAPI application configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = "INFO"
    enable_cors: bool = False
    cors_origins: tuple[str, ...] = ()
    enable_debug_endpoint: bool = False
    max_concurrent_requests: int = 2

    def __post_init__(self) -> None:
        if self.max_concurrent_requests < 1:
            raise ValueError(
                f"max_concurrent_requests must be >= 1, got {self.max_concurrent_requests}"
            )


@dataclass(frozen=True, slots=True)
class RateLimitConfig:
    """Basic request throttling configuration."""

    requests: int = 10
    window_seconds: int = 60


@dataclass(frozen=True, slots=True)
class Settings:
    """Aggregate application settings."""

    neo4j: Neo4jConfig
    llm: LLMConfig
    sqlite: SQLiteConfig
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    message_embeddings: MessageEmbeddingConfig = field(default_factory=MessageEmbeddingConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    api: APIConfig = field(default_factory=APIConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    @staticmethod
    def _bool_env(key: str, default: bool = False) -> bool:
        raw = os.getenv(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _tuple_env(key: str) -> tuple[str, ...]:
        raw = os.getenv(key)
        if not raw:
            return ()
        return tuple(item.strip() for item in raw.split(",") if item.strip())

    @staticmethod
    def _str_env(key: str) -> str | None:
        raw = os.getenv(key)
        if raw is None:
            return None
        stripped = raw.strip()
        return stripped or None

    @classmethod
    def from_env(cls) -> Settings:
        """Load configuration values from the environment."""
        neo4j = Neo4jConfig(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "test"),
            database=os.getenv("NEO4J_DATABASE") or None,
            max_connection_lifetime=int(os.getenv("NEO4J_MAX_CONN_LIFETIME", "60")),
            encrypted=cls._bool_env("NEO4J_ENCRYPTED", False),
        )

        sqlite = SQLiteConfig(
            db_path=Path(os.getenv("SQLITE_DB_PATH", "./discord.db")).expanduser(),
        )

        model = os.getenv("LLAMA_MODEL", "GLM-4.5-Air")
        temperature = float(os.getenv("LLAMA_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.3")))
        api_key = os.getenv("LLAMA_API_KEY")
        base_url = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1/chat/completions")
        top_p = float(os.getenv("LLAMA_TOP_P", "0.95"))
        max_tokens = int(os.getenv("LLAMA_MAX_TOKENS", "4096"))
        timeout = float(os.getenv("LLAMA_TIMEOUT", "1200"))

        llm = LLMConfig(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        embedding_model = cls._str_env("EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL
        embedding_device = cls._str_env("EMBEDDING_DEVICE") or "cpu"
        embedding_cache = cls._str_env("EMBEDDING_CACHE_DIR")
        embeddings = EmbeddingConfig(
            model=embedding_model,
            device=embedding_device,
            cache_dir=Path(embedding_cache).expanduser() if embedding_cache else None,
        )

        default_message_batch = MessageEmbeddingConfig().batch_size
        message_model = cls._str_env("MESSAGE_EMBEDDING_MODEL") or embeddings.model
        message_device = cls._str_env("MESSAGE_EMBEDDING_DEVICE") or embeddings.device
        message_cache_raw = cls._str_env("MESSAGE_EMBEDDING_CACHE_DIR")
        message_embeddings = MessageEmbeddingConfig(
            model=message_model,
            device=message_device,
            cache_dir=Path(message_cache_raw).expanduser()
            if message_cache_raw
            else embeddings.cache_dir,
            batch_size=int(os.getenv("MESSAGE_EMBEDDING_BATCH_SIZE", str(default_message_batch))),
        )

        agent = AgentConfig(
            max_iterations=int(os.getenv("MAX_ITERATIONS", "10")),
            max_facts=int(os.getenv("MAX_FACTS", "30")),
            tool_timeout_seconds=int(os.getenv("TOOL_TIMEOUT_SECONDS", "10")),
            tool_max_retries=int(os.getenv("TOOL_MAX_RETRIES", "2")),
            reasoning_trace_limit=int(os.getenv("REASONING_TRACE_LIMIT", "20")),
            early_stop_min_iterations=int(os.getenv("EARLY_STOP_MIN_ITERATIONS", "2")),
            novelty_min_new_facts=int(os.getenv("NOVELTY_MIN_NEW_FACTS", "1")),
            novelty_patience=int(os.getenv("NOVELTY_PATIENCE", "2")),
            stop_confidence_required=os.getenv("STOP_CONFIDENCE_REQUIRED", "high"),
            fallback_max_queries=int(os.getenv("FALLBACK_MAX_QUERIES", "12")),
            fallback_max_query_length=int(os.getenv("FALLBACK_MAX_QUERY_LENGTH", "120")),
        )

        # Validate early-stop config
        if agent.early_stop_min_iterations < 0:
            raise ValueError("EARLY_STOP_MIN_ITERATIONS must be >= 0")
        if agent.novelty_patience < 1:
            raise ValueError("NOVELTY_PATIENCE must be >= 1")
        if agent.novelty_min_new_facts < 0:
            raise ValueError("NOVELTY_MIN_NEW_FACTS must be >= 0")
        if agent.stop_confidence_required not in ("low", "medium", "high"):
            raise ValueError(
                f"STOP_CONFIDENCE_REQUIRED must be low/medium/high, got {agent.stop_confidence_required}"
            )
        if agent.fallback_max_queries < 1:
            raise ValueError("FALLBACK_MAX_QUERIES must be >= 1")
        if agent.fallback_max_query_length < 10:
            raise ValueError("FALLBACK_MAX_QUERY_LENGTH must be >= 10")

        api = APIConfig(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),  # type: ignore[arg-type]
            enable_cors=cls._bool_env("ENABLE_CORS", False),
            cors_origins=cls._tuple_env("CORS_ALLOW_ORIGINS"),
            enable_debug_endpoint=cls._bool_env("ENABLE_DEBUG_ENDPOINT", False),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "2")),
        )

        rate_limit = RateLimitConfig(
            requests=int(os.getenv("RATE_LIMIT_REQUESTS", "10")),
            window_seconds=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        )

        return cls(
            neo4j=neo4j,
            llm=llm,
            sqlite=sqlite,
            embeddings=embeddings,
            message_embeddings=message_embeddings,
            agent=agent,
            api=api,
            rate_limit=rate_limit,
        )
