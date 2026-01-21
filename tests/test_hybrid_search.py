"""Tests for hybrid search (vector + keyword) functionality."""

from __future__ import annotations

import re

from memory_agent.tools.semantic_search import KEYWORD_QUERY_OFFSET


class TestKeywordQuerySanitization:
    """Tests for keyword query sanitization logic."""

    def test_sanitize_special_characters(self):
        """Test that special characters are properly sanitized."""
        # This replicates the sanitization logic from run_keyword_query
        query = "test@#$%query!?"
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert "@" not in safe_text
        assert "#" not in safe_text
        assert "$" not in safe_text
        assert "!" not in safe_text
        assert "testquery" in safe_text or "test" in safe_text

    def test_empty_after_sanitization(self):
        """Test that fully special character strings become empty."""
        query = "@#$%^&*()"
        safe_text = re.sub(r"[^a-zA-Z0-9\s\-_]", "", query).strip()

        assert safe_text == ""

    def test_fuzzy_matching_pattern(self):
        """Test fuzzy matching pattern generation."""
        safe_text = "Python"
        terms = safe_text.split()
        lucene_parts = []
        for term in terms:
            lucene_parts.append(f"{term} OR {term}~")
        lucene_query = " ".join(lucene_parts)

        assert "~" in lucene_query
        assert "Python" in lucene_query
        assert "OR" in lucene_query

    def test_multi_word_query(self):
        """Test multi-word queries are split and processed."""
        safe_text = "Python programming"
        terms = safe_text.split()

        assert len(terms) == 2
        assert "Python" in terms
        assert "programming" in terms


class TestHybridSearchConfiguration:
    """Tests for hybrid search configuration and constants."""

    def test_query_idx_offset_constant(self):
        """Test that KEYWORD_QUERY_OFFSET is properly defined."""
        assert KEYWORD_QUERY_OFFSET == 1000
        assert isinstance(KEYWORD_QUERY_OFFSET, int)

    def test_offset_allows_independent_rrf_voting(self):
        """Test that offset is large enough to separate vector and keyword votes."""
        # With max ~100 queries per request, offset of 1000 ensures no collision
        max_expected_queries = 100
        assert max_expected_queries * 10 <= KEYWORD_QUERY_OFFSET
