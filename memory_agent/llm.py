"""Lightweight LLM abstraction for the memory agent."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from ie.client import LlamaServerClient, LlamaServerConfig


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMClient:
    """Client that routes planning prompts to llama-server with fallback heuristics."""

    model: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None

    _llama_client: LlamaServerClient | None = None

    def __post_init__(self) -> None:
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            config = LlamaServerConfig(
                base_url=self.base_url or "http://localhost:8080/v1/chat/completions",
                model=self.model,
                timeout=self.timeout or 1200.0,
                temperature=self.temperature,
                top_p=self.top_p or 0.95,
                max_tokens=int(self.max_tokens or 4096),
                api_key=self.api_key,
            )
            self._llama_client = LlamaServerClient(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize LLM client: %s", exc)
            self._llama_client = None

    async def apredict(self, prompt: str) -> str:
        """Asynchronously generate a response to the prompt."""
        if self._llama_client is not None:
            return await asyncio.to_thread(self._llama_predict, prompt)
        return self._fallback(prompt)

    def predict(self, prompt: str) -> str:
        """Synchronously generate a response."""
        if self._llama_client is not None:
            return self._llama_predict(prompt)
        return self._fallback(prompt)

    def _llama_predict(self, prompt: str) -> str:
        if self._llama_client is None:
            return self._fallback(prompt)
        messages = [
            {
                "role": "system",
                "content": (
                    "You help coordinate which knowledge graph tool should run next. "
                    "Return only the tool name that best advances the stated goal."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llama_client.complete(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Llama call failed, falling back: %s", exc)
            return self._fallback(prompt)
        if response is None:
            return self._fallback(prompt)
        return response

    def _fallback(self, prompt: str) -> str:
        """Fallback heuristic when no LLM provider is configured."""
        truncated = (prompt[:200] + "...") if len(prompt) > 200 else prompt
        return (
            "LLM unavailable. "
            "Consider exploring skills, organizations, and relationships related to this request. "
            f"Prompt excerpt: {truncated}"
        )

    @property
    def is_available(self) -> bool:
        """Return True if an underlying chat model is ready."""
        return self._llama_client is not None

    async def aplan_tool_usage(self, goal: str, options: list[str]) -> str:
        """Return the name of a tool that best matches the current goal."""
        prompt = (
            "You are planning which knowledge graph tool to run next. "
            f"Goal: {goal}. Available tools: {', '.join(options)}. "
            "Return only the tool name that best advances the goal."
        )
        response = await self.apredict(prompt)
        for option in options:
            if option in response:
                return option
        return options[0] if options else ""
