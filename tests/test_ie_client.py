"""Tests for ie/client.py."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from ie.client import LlamaServerClient, LlamaServerConfig


def test_client_init_without_api_key() -> None:
    """Client should not set Authorization header when api_key is None."""
    config = LlamaServerConfig(api_key=None)

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        LlamaServerClient(config)

        call_args = mock_client_class.call_args
        headers = call_args.kwargs.get("headers", {})
        assert "Authorization" not in headers


def test_client_init_with_api_key() -> None:
    """Client should set Authorization header when api_key is provided."""
    config = LlamaServerConfig(api_key="test-key-123")

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        LlamaServerClient(config)

        call_args = mock_client_class.call_args
        headers = call_args.kwargs.get("headers", {})
        assert headers.get("Authorization") == "Bearer test-key-123"


def test_client_close_delegates_to_httpx() -> None:
    """close() should delegate to the underlying httpx client."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        client.close()

        mock_http_client.close.assert_called_once()


def test_client_complete_constructs_payload() -> None:
    """complete() should construct the correct payload."""
    config = LlamaServerConfig(
        model="test-model",
        temperature=0.5,
        top_p=0.9,
        max_tokens=2048,
    )

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "test response"}}]}
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        messages = [{"role": "user", "content": "hello"}]
        client.complete(messages)

        # Verify the post was called with correct payload
        call_args = mock_http_client.post.call_args
        payload = json.loads(call_args.kwargs.get("content") or call_args.args[1])

        assert payload["model"] == "test-model"
        assert payload["temperature"] == 0.5
        assert payload["top_p"] == 0.9
        assert payload["max_tokens"] == 2048
        assert payload["messages"] == messages
        assert payload["response_format"] == {"type": "json_object"}


def test_client_complete_without_json_mode() -> None:
    """complete() with json_mode=False should not include response_format."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "plain text"}}]}
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        result = client.complete([{"role": "user", "content": "hi"}], json_mode=False)

        call_args = mock_http_client.post.call_args
        payload = json.loads(call_args.kwargs.get("content") or call_args.args[1])

        assert "response_format" not in payload
        assert result == "plain text"


def test_client_complete_returns_none_on_empty_choices() -> None:
    """complete() should return None when choices is empty."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        result = client.complete([{"role": "user", "content": "test"}])

        assert result is None


def test_client_complete_returns_none_when_choices_missing() -> None:
    """complete() should return None when choices key is missing."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {}
        mock_http_client.post.return_value = mock_response
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        result = client.complete([{"role": "user", "content": "test"}])

        assert result is None


def test_client_context_manager_enter() -> None:
    """__enter__ should return the client instance."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        result = client.__enter__()

        assert result is client


def test_client_context_manager_exit_closes_client() -> None:
    """__exit__ should call close()."""
    config = LlamaServerConfig()

    with patch("ie.client.httpx.Client") as mock_client_class:
        mock_http_client = MagicMock()
        mock_client_class.return_value = mock_http_client

        client = LlamaServerClient(config)
        client.__exit__(None, None, None)

        mock_http_client.close.assert_called_once()
