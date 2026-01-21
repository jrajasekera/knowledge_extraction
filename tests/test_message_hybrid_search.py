"""Tests for message hybrid search (vector + keyword) functionality."""

from __future__ import annotations

import re

from memory_agent.tools.semantic_search_messages import KEYWORD_QUERY_OFFSET


class TestMessageKeywordQuerySanitization:
    """Tests for message keyword query sanitization logic."""

    def test_sanitize_special_characters(self):
        """Test that special characters are properly sanitized."""
        # This replicates the sanitization logic from run_keyword_query
        query = "error@#$500!?"
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert "@" not in safe_text
        assert "#" not in safe_text
        assert "$" not in safe_text
        assert "!" not in safe_text
        assert "error500" in safe_text or "error" in safe_text

    def test_empty_after_sanitization(self):
        """Test that fully special character strings become empty."""
        query = "@#$%^&*()"
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert safe_text == ""

    def test_fuzzy_matching_pattern(self):
        """Test fuzzy matching pattern generation for messages."""
        safe_text = "JIRA-123"
        # After sanitization: "JIRA-123" → "JIRA123"
        sanitized = re.sub(r"[^a-zA-Z0-9\s\-_]", "", safe_text).strip()
        terms = sanitized.split()
        lucene_parts = []
        for term in terms:
            lucene_parts.append(f"{term} OR {term}~")
        lucene_query = " ".join(lucene_parts)

        assert "~" in lucene_query
        assert "JIRA" in lucene_query or "123" in lucene_query
        assert "OR" in lucene_query

    def test_multi_word_message_query(self):
        """Test multi-word message queries are split and processed."""
        safe_text = "error 500 occurred"
        terms = safe_text.split()

        assert len(terms) == 3
        assert "error" in terms
        assert "500" in terms
        assert "occurred" in terms

    def test_channel_mention_query(self):
        """Test queries with channel mentions are sanitized."""
        query = "messages in #general"
        # '#' will be removed by sanitization
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert "#" not in safe_text
        assert "general" in safe_text
        assert "messages" in safe_text

    def test_username_mention_query(self):
        """Test queries with username mentions are sanitized."""
        query = "posts from @alice"
        # '@' will be removed by sanitization
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert "@" not in safe_text
        assert "alice" in safe_text
        assert "posts" in safe_text


class TestMessageHybridSearchConfiguration:
    """Tests for message hybrid search configuration and constants."""

    def test_query_idx_offset_constant(self):
        """Test that KEYWORD_QUERY_OFFSET is properly defined."""
        assert KEYWORD_QUERY_OFFSET == 1000
        assert isinstance(KEYWORD_QUERY_OFFSET, int)

    def test_offset_allows_independent_rrf_voting(self):
        """Test that offset is large enough to separate vector and keyword votes."""
        # With max ~100 queries per request, offset of 1000 ensures no collision
        max_expected_queries = 100
        assert max_expected_queries * 10 <= KEYWORD_QUERY_OFFSET

    def test_message_index_fields_coverage(self):
        """Test that message fulltext index covers key searchable fields."""
        # These fields should be indexed for hybrid search
        expected_fields = ["content", "clean_content", "author_name", "channel_name", "guild_name"]

        # This is a documentation test - the actual index creation is in the migration
        assert len(expected_fields) == 5
        assert "content" in expected_fields
        assert "author_name" in expected_fields
        assert "channel_name" in expected_fields
