"""FastAPI application factory for the memory agent service."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from neo4j import GraphDatabase

from .agent import MemoryAgent
from .config import Settings
from .embeddings import EmbeddingProvider
from .llm import LLMClient
from .models import HealthResponse, RetrievalRequest, RetrievalResponse
from .request_logger import RequestLogger
from .tools import ToolContext, build_toolkit

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple in-memory fixed window rate limiter."""

    def __init__(self, requests: int, window_seconds: int) -> None:
        self.requests = requests
        self.window = window_seconds
        self._buckets: dict[str, tuple[int, int]] = {}

    def allow(self, key: str) -> bool:
        now = int(time.time())
        window_start = now - (now % self.window)
        count, current_window = self._buckets.get(key, (0, window_start))
        if current_window != window_start:
            count = 0
            current_window = window_start
        if count >= self.requests:
            self._buckets[key] = (count, current_window)
            return False
        self._buckets[key] = (count + 1, current_window)
        return True


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="Memory Agent Service", version="0.1.0")

    if settings.api.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.api.cors_origins) or ["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.state.settings = settings
    app.state.driver = None
    app.state.embedding_provider = None
    app.state.tools = None
    app.state.agent = None
    app.state.rate_limiter = RateLimiter(
        settings.rate_limit.requests, settings.rate_limit.window_seconds
    )
    app.state.request_logger = RequestLogger(settings.sqlite.db_path)
    app.state.concurrency_semaphore = asyncio.Semaphore(settings.api.max_concurrent_requests)

    @app.on_event("startup")
    async def on_startup() -> None:
        cfg = app.state.settings

        # Configure logging for memory_agent modules.
        import logging

        log_level_name = (cfg.api.log_level or "INFO").upper()
        log_level = getattr(logging, log_level_name, logging.INFO)

        # Ensure the root logger has at least one handler so child loggers (like
        # memory_agent.tools.semantic_search_messages) emit records when running under
        # uvicorn's default logging config, which otherwise leaves the root handlerless.
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

        # Set log level for all memory_agent loggers and make sure they propagate to root.
        for logger_name in [
            "memory_agent",
            "memory_agent.agent",
            "memory_agent.tools",
            "memory_agent.tools.semantic_search_messages",
            "memory_agent.tools.semantic_search",
            "memory_agent.llm",
            "memory_agent.context_summarizer",
        ]:
            module_logger = logging.getLogger(logger_name)
            module_logger.setLevel(log_level)
            module_logger.propagate = True

        logger.info("Starting memory agent service with log level: %s", cfg.api.log_level)

        driver_kwargs: dict[str, Any] = {
            "max_connection_lifetime": cfg.neo4j.max_connection_lifetime
        }
        if cfg.neo4j.encrypted:
            driver_kwargs["encrypted"] = True
        driver = GraphDatabase.driver(
            cfg.neo4j.uri,
            auth=(cfg.neo4j.user, cfg.neo4j.password),
            **driver_kwargs,
        )
        app.state.driver = driver
        try:
            driver.verify_connectivity()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j connectivity check failed: %s", exc)
        embedding_provider = EmbeddingProvider(
            model_name=cfg.embeddings.model,
            device=cfg.embeddings.device,
            cache_dir=cfg.embeddings.cache_dir,
        )
        tool_context = ToolContext(driver=driver, embeddings_model=embedding_provider)
        toolkit = build_toolkit(tool_context)
        llm_client = LLMClient(
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
            api_key=cfg.llm.api_key,
            base_url=cfg.llm.base_url,
            top_p=cfg.llm.top_p,
            max_tokens=cfg.llm.max_tokens,
            timeout=cfg.llm.timeout,
        )
        agent = MemoryAgent(toolkit, cfg.agent, llm=llm_client)
        app.state.embedding_provider = embedding_provider
        app.state.tools = toolkit
        app.state.agent = agent

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        logger.info("Shutting down memory agent service")
        driver = app.state.driver
        if driver:
            driver.close()

    async def get_agent(request: Request) -> MemoryAgent:
        if app.state.agent is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Agent not ready"
            )
        limiter: RateLimiter = app.state.rate_limiter
        client_id = request.client.host if request.client else "unknown"
        if not limiter.allow(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded"
            )
        return app.state.agent

    @app.post("/api/memory/retrieve", response_model=RetrievalResponse)
    async def retrieve_memories(
        request: Request,
        request_body: RetrievalRequest,
        agent: MemoryAgent = Depends(get_agent),  # noqa: B008
    ) -> RetrievalResponse:
        request_id = uuid4().hex
        client_ip = request.client.host if request.client else None
        request_logger: RequestLogger = app.state.request_logger

        # Log request start
        request_logger.log_request_start(request_id, request_body, client_ip)

        semaphore: asyncio.Semaphore = app.state.concurrency_semaphore
        start_time = time.perf_counter()
        try:
            async with semaphore:
                result = await agent.run(request_body)
            response = RetrievalResponse.model_validate(result)
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            request_logger.log_request_complete(request_id, response, duration_ms)
            return response
        except HTTPException as exc:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            request_logger.log_request_error(
                request_id, exc.status_code, str(exc.detail), duration_ms
            )
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Memory retrieval failed: %s", exc)
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            request_logger.log_request_error(
                request_id, status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc), duration_ms
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc

    if settings.api.enable_debug_endpoint:

        @app.post("/api/memory/retrieve/debug")
        async def retrieve_memories_debug(
            request_body: RetrievalRequest,
            agent: MemoryAgent = Depends(get_agent),  # noqa: B008
        ) -> dict:
            semaphore: asyncio.Semaphore = app.state.concurrency_semaphore
            try:
                async with semaphore:
                    result = await agent.run(request_body, debug_mode=True)
                response = RetrievalResponse.model_validate(result)
                debug_info = result.get("debug", {})
                return {
                    "facts": response.facts,
                    "confidence": response.confidence,
                    "metadata": response.metadata,
                    "debug_info": debug_info,
                }
            except HTTPException:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("Memory retrieval debug failed: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
                ) from exc

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        driver = app.state.driver
        neo4j_connected = False
        if driver is not None:
            try:
                session_kwargs = {}
                if app.state.settings.neo4j.database:
                    session_kwargs["database"] = app.state.settings.neo4j.database
                with driver.session(**session_kwargs) as session:
                    session.run("RETURN 1").consume()
                neo4j_connected = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("Health check failed to reach Neo4j: %s", exc)
        model_loaded = app.state.embedding_provider is not None
        status_value = "healthy" if neo4j_connected and model_loaded else "degraded"
        return HealthResponse(
            status=status_value,
            neo4j_connected=neo4j_connected,
            model_loaded=model_loaded,
            version="0.1.0",
            additional={"agent_ready": app.state.agent is not None},
        )

    return app
