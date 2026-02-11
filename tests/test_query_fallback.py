"""Tests for deterministic fallback query expansion."""

from __future__ import annotations

import pytest

from memory_agent.query_fallback import (
    _SCHEMA_MAX_QUERIES,
    FallbackQuery,
    expand_fallback_queries,
)

# ---------------------------------------------------------------------------
# Unit tests for expand_fallback_queries
# ---------------------------------------------------------------------------


class TestSkillLookup:
    """Skill-related queries produce entity/keyword variants."""

    def test_produces_queries_for_skills(self) -> None:
        result = expand_fallback_queries("Who knows Python and machine learning?")
        texts = [q.text.lower() for q in result]
        assert any("python" in t for t in texts)
        assert any("machine" in t or "learning" in t for t in texts)
        assert len(result) > 1


class TestOrganizationLookup:
    """Organization names are extracted as entities."""

    def test_extracts_org_entity(self) -> None:
        result = expand_fallback_queries("Who works at Google?")
        texts = [q.text for q in result]
        assert any("Google" in t for t in texts)

    def test_multi_word_org(self) -> None:
        result = expand_fallback_queries("She works at Acme Corporation")
        texts = [q.text for q in result]
        assert any("Acme Corporation" in t for t in texts)


class TestPronounHeavyFollowUps:
    """Pronoun-heavy messages use recent context for entities."""

    def test_uses_recent_context_entities(self) -> None:
        result = expand_fallback_queries(
            last_message="What about their experience?",
            recent_messages=[
                "Tell me about John Smith",
                "He works at Microsoft",
            ],
        )
        texts = [q.text for q in result]
        assert any("John Smith" in t for t in texts) or any("Microsoft" in t for t in texts)
        assert len(result) > 1


class TestDeduplication:
    """No duplicate queries in output."""

    def test_no_duplicates(self) -> None:
        result = expand_fallback_queries("Tell me about Python programming and Python development")
        lower_texts = [q.text.strip().lower() for q in result]
        assert len(lower_texts) == len(set(lower_texts))


class TestPreviousQueryFiltering:
    """Previously-tried queries are excluded."""

    def test_excludes_previous_queries(self) -> None:
        result = expand_fallback_queries(
            "Who knows Python?",
            previous_queries=["python"],
        )
        texts = [q.text.lower() for q in result]
        # "python" as an exact match should be filtered out
        assert "python" not in texts

    def test_substring_match_filtering(self) -> None:
        result = expand_fallback_queries(
            "Who works at Google?",
            previous_queries=["works at google"],
        )
        texts = [q.text.lower() for q in result]
        # Substring match should filter queries containing "works at google"
        assert not any(t == "works at google" for t in texts)


class TestEmptyInput:
    """Graceful handling of empty input."""

    def test_empty_message_no_goal(self) -> None:
        result = expand_fallback_queries("")
        assert result == []

    def test_none_goal_empty_message(self) -> None:
        result = expand_fallback_queries("", goal=None)
        assert result == []

    def test_whitespace_only(self) -> None:
        result = expand_fallback_queries("   ", goal="   ")
        # May or may not produce queries, but should not crash
        assert isinstance(result, list)


class TestMaxQueryCap:
    """Respects max_queries and never exceeds _SCHEMA_MAX_QUERIES."""

    def test_respects_max_queries(self) -> None:
        # Long message should produce many candidates, but be capped
        long_msg = " ".join(f"Word{i}" for i in range(100))
        result = expand_fallback_queries(long_msg, max_queries=3)
        assert len(result) <= 3

    def test_never_exceeds_schema_max(self) -> None:
        long_msg = " ".join(f"Word{i}" for i in range(100))
        result = expand_fallback_queries(long_msg, max_queries=50)
        assert len(result) <= _SCHEMA_MAX_QUERIES

    def test_schema_max_is_20(self) -> None:
        assert _SCHEMA_MAX_QUERIES == 20


class TestMaxQueryLength:
    """All queries truncated to max_query_length."""

    def test_truncates_long_queries(self) -> None:
        long_msg = "A " * 200  # Very long message
        result = expand_fallback_queries(long_msg, max_query_length=50)
        for q in result:
            assert len(q.text) <= 50

    def test_custom_max_length(self) -> None:
        msg = "Tell me about Artificial Intelligence and Machine Learning applications"
        result = expand_fallback_queries(msg, max_query_length=30)
        for q in result:
            assert len(q.text) <= 30


class TestSourceMetadata:
    """Each query has correct source field."""

    def test_valid_sources(self) -> None:
        result = expand_fallback_queries(
            "Who works at Google and knows Python?",
            goal="Find people with Python skills at Google",
        )
        valid_sources = {"entity", "keyword", "phrase", "question", "goal"}
        for q in result:
            assert q.source in valid_sources

    def test_question_source_on_question_mark(self) -> None:
        result = expand_fallback_queries("What skills does Alice have?")
        sources = {q.source for q in result}
        assert "question" in sources

    def test_goal_source_when_different(self) -> None:
        result = expand_fallback_queries(
            "Tell me more",
            goal="Find Python developers at Google",
        )
        sources = {q.source for q in result}
        assert "goal" in sources

    def test_no_goal_source_when_same(self) -> None:
        msg = "Find Python developers"
        result = expand_fallback_queries(msg, goal=msg)
        sources = {q.source for q in result}
        assert "goal" not in sources


class TestSourcePriority:
    """Queries are ordered by source priority: entity > keyword > phrase > question > goal."""

    def test_entities_before_keywords(self) -> None:
        result = expand_fallback_queries(
            "Alice works at Google with Python experience?",
            goal="Find skilled developers",
        )
        if len(result) < 2:
            pytest.skip("Not enough queries generated")
        sources = [q.source for q in result]
        # All entities should come before all keywords
        entity_indices = [i for i, s in enumerate(sources) if s == "entity"]
        keyword_indices = [i for i, s in enumerate(sources) if s == "keyword"]
        if entity_indices and keyword_indices:
            assert max(entity_indices) < min(keyword_indices)


class TestGoalFallback:
    """Goal is used when last_message is empty."""

    def test_goal_used_as_fallback(self) -> None:
        result = expand_fallback_queries("", goal="Find people who know Rust")
        assert len(result) > 0
        texts = [q.text.lower() for q in result]
        assert any("rust" in t for t in texts)


class TestAcronymExtraction:
    """Uppercase acronyms are extracted as entities."""

    def test_extracts_acronyms(self) -> None:
        result = expand_fallback_queries("Does anyone here work with AWS or GCP?")
        texts = [q.text for q in result]
        assert any("AWS" in t for t in texts) or any("GCP" in t for t in texts)


class TestFallbackQueryDataclass:
    """FallbackQuery dataclass basics."""

    def test_slots(self) -> None:
        q = FallbackQuery(text="test", source="keyword")
        assert q.text == "test"
        assert q.source == "keyword"

    def test_dataclass_equality(self) -> None:
        q1 = FallbackQuery(text="test", source="keyword")
        q2 = FallbackQuery(text="test", source="keyword")
        assert q1 == q2
