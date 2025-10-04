from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

import httpx


@dataclass(slots=True)
class LlamaServerConfig:
    base_url: str = "http://localhost:8080/v1/chat/completions"
    model: str = "GLM-4.5-Air"
    timeout: float = 600.0
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 512
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

    def complete(self, messages: Iterable[dict[str, Any]]) -> str | None:
        payload = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
            "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": "low"},
            "cache_prompt": True
        }
        response = self._client.post(self.config.base_url, content=json.dumps(payload))
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message", {}).get("content")

    def __enter__(self) -> "LlamaServerClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.close()
