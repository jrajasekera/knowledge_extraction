from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(slots=True)
class LlamaServerConfig:
    base_url: str = "http://localhost:8080/v1/chat/completions"
    model: str = "GLM-4.5-Air"
    timeout: float = 12000.0
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 4096
    api_key: str | None = None


class LlamaServerClient:
    def __init__(self, config: LlamaServerConfig) -> None:
        self.config = config
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._client = httpx.Client(timeout=config.timeout, headers=headers)

    def close(self) -> None:
        self._client.close()

    def complete(self, messages: Iterable[dict[str, Any]], json_mode: bool = True) -> str | None:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "high"},
            "cache_prompt": True,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        response = self._client.post(self.config.base_url, content=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")

    def __enter__(self) -> LlamaServerClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
