"""Tests for fact search query extraction functionality."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memory_agent.llm import LLMClient
from memory_agent.models import MessageModel


class TestExtractFactSearchQueries:
    """Tests for extract_fact_search_queries method."""

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_returns_empty_when_unavailable(self):
        """Test that extraction returns empty list when LLM is unavailable."""
        llm = LLMClient(model="test", temperature=0.3)
        llm._llama_client = None  # Simulate unavailable LLM

        messages = [
            MessageModel(
                author_name="Alice",
                content="Who has experience with Python?",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)
        assert queries == []

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_returns_empty_for_no_messages(self):
        """Test that extraction returns empty list when no messages provided."""
        llm = LLMClient(model="test", temperature=0.3)

        queries = await llm.extract_fact_search_queries([])
        assert queries == []

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_parses_llm_response(self):
        """Test that extraction parses LLM response correctly."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps(
                    {
                        "queries": [
                            "Python",
                            "Python developer",
                            "programming experience",
                            "software engineer with Python skills",
                            "backend development Python",
                            "Python frameworks",
                            "Django Flask experience",
                            "Python web development",
                            "data science Python",
                            "machine learning Python",
                            "Python automation",
                            "Python scripting",
                        ]
                    }
                )

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Who has experience with Python?",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages, max_queries=15)

        assert len(queries) == 12
        assert "Python" in queries
        assert "Python developer" in queries
        assert "software engineer with Python skills" in queries

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_respects_max_queries(self):
        """Test that extraction respects max_queries parameter."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                queries_list = [f"query{i}" for i in range(25)]
                return json.dumps({"queries": queries_list})

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test message",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages, max_queries=10)
        assert len(queries) <= 10

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_handles_llm_error(self):
        """Test that extraction handles LLM errors gracefully."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                raise Exception("LLM error")

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test message",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)
        assert queries == []

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_handles_invalid_json(self):
        """Test that extraction handles invalid JSON gracefully."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return "invalid json {"

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test message",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)
        assert queries == []

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_handles_non_list_response(self):
        """Test that extraction handles non-list queries field."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps({"queries": "not a list"})

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test message",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)
        assert queries == []

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_cleans_query_prefix(self):
        """Test that extraction removes 'query:' prefix if present."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps(
                    {
                        "queries": [
                            "query: Python",
                            "Query: machine learning",
                            "normal query without prefix",
                        ]
                    }
                )

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)

        assert "Python" in queries
        assert "machine learning" in queries
        assert "normal query without prefix" in queries
        assert not any(q.lower().startswith("query:") for q in queries)

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_deduplicates(self):
        """Test that extraction removes duplicate queries."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps(
                    {
                        "queries": [
                            "Python",
                            "Python",  # Duplicate
                            "machine learning",
                            "Python",  # Another duplicate
                        ]
                    }
                )

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)

        assert len(queries) == 2
        assert "Python" in queries
        assert "machine learning" in queries

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_truncates_long_queries(self):
        """Test that extraction truncates queries to 120 characters."""
        llm = LLMClient(model="test", temperature=0.3)

        long_query = "a" * 150

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps({"queries": [long_query]})

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)

        assert len(queries) == 1
        assert len(queries[0]) == 120

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_skips_non_string_entries(self):
        """Test that extraction skips non-string entries in queries list."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps(
                    {
                        "queries": [
                            "valid query",
                            123,  # Not a string
                            "another valid query",
                            None,  # Not a string
                            {"key": "value"},  # Not a string
                        ]
                    }
                )

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)

        assert len(queries) == 2
        assert "valid query" in queries
        assert "another valid query" in queries

    @pytest.mark.asyncio
    async def test_extract_fact_search_queries_skips_empty_strings(self):
        """Test that extraction skips empty or whitespace-only queries."""
        llm = LLMClient(model="test", temperature=0.3)

        class DummyLlama:
            def complete(self, messages, json_mode: bool = False):
                return json.dumps(
                    {
                        "queries": [
                            "valid query",
                            "",  # Empty
                            "   ",  # Whitespace only
                            "another valid",
                        ]
                    }
                )

        llm._llama_client = DummyLlama()

        messages = [
            MessageModel(
                author_name="Alice",
                content="Test",
                author_id="1",
                channel_id="test",
                guild_id="test",
                message_id="1",
                timestamp=datetime.now(timezone.utc),
            ),
        ]

        queries = await llm.extract_fact_search_queries(messages)

        assert len(queries) == 2
        assert "valid query" in queries
        assert "another valid" in queries
