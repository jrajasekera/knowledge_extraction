"""Lightweight LLM abstraction for the memory agent."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LLMClient:
    """Wrapper around LangChain chat models with graceful degradation."""

    provider: str
    model: str
    temperature: float
    api_key: str | None = None

    _chat_model: Any | None = None

    def __post_init__(self) -> None:
        if self.api_key:
            self._initialize_chat_model()

    def _initialize_chat_model(self) -> None:
        try:
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI

                self._chat_model = ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=self.api_key,
                )
            elif self.provider == "anthropic":
                from langchain_anthropic import ChatAnthropic

                self._chat_model = ChatAnthropic(
                    model=self.model,
                    temperature=self.temperature,
                    api_key=self.api_key,
                )
            else:
                logger.warning("LLM provider %s is not supported", self.provider)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize LLM client: %s", exc)
            self._chat_model = None

    async def apredict(self, prompt: str) -> str:
        """Asynchronously generate a response to the prompt."""
        if self._chat_model is None:
            return self._fallback(prompt)
        try:
            result = await self._chat_model.ainvoke(prompt)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed, falling back: %s", exc)
            return self._fallback(prompt)

    def predict(self, prompt: str) -> str:
        """Synchronously generate a response."""
        if self._chat_model is None:
            return self._fallback(prompt)
        try:
            result = self._chat_model.invoke(prompt)
            if hasattr(result, "content"):
                return str(result.content)
            return str(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM call failed, falling back: %s", exc)
            return self._fallback(prompt)

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
        return self._chat_model is not None

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
