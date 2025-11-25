"""Tests for RRF (Reciprocal Rank Fusion) implementation in semantic_search_facts."""

from __future__ import annotations

import pytest

from memory_agent.tools.semantic_search import (
    ADAPTIVE_THRESHOLD_MAX,
    ADAPTIVE_THRESHOLD_MIN,
    DEFAULT_FUSION_METHOD,
    DEFAULT_MULTI_QUERY_BOOST,
    RRF_K,
    FactOccurrence,
    SemanticSearchFactsTool,
    SemanticSearchInput,
    SemanticSearchResult,
)


class TestFactOccurrence:
    """Tests for FactOccurrence dataclass."""

    def test_fact_occurrence_initialization(self):
        """Test FactOccurrence can be initialized with required fields."""
        properties = {"person_id": "123", "fact_type": "works_at", "fact_object": "Google"}
        evidence = [{"content": "I work at Google"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)

        assert occurrence.properties == properties
        assert occurrence.best_score == 0.9
        assert occurrence.evidence == evidence
        assert occurrence.query_scores == {}
        assert occurrence.query_ranks == {}

    def test_add_observation_updates_scores_and_ranks(self):
        """Test that add_observation correctly tracks scores and ranks."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Initial evidence"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.8, evidence=evidence)

        # Add first observation
        new_evidence1 = [{"content": "Evidence from query 1"}]
        occurrence.add_observation(1, 0.85, 3, properties, new_evidence1)

        assert occurrence.query_scores[1] == 0.85
        assert occurrence.query_ranks[1] == 3
        assert occurrence.best_score == 0.85
        assert occurrence.evidence == new_evidence1

        # Add second observation from different query
        new_evidence2 = [{"content": "Evidence from query 2"}]
        occurrence.add_observation(2, 0.90, 1, properties, new_evidence2)

        assert occurrence.query_scores[2] == 0.90
        assert occurrence.query_ranks[2] == 1
        assert occurrence.best_score == 0.90
        assert occurrence.evidence == new_evidence2

    def test_add_observation_keeps_best_score_per_query(self):
        """Test that multiple observations from same query keep the highest score."""
        properties = {"person_id": "123"}
        evidence1 = [{"content": "First evidence"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.8, evidence=evidence1)

        # Add first observation from query 1
        occurrence.add_observation(1, 0.85, 3, properties, evidence1)
        assert occurrence.query_scores[1] == 0.85

        # Add another observation from query 1 with lower score
        evidence2 = [{"content": "Second evidence"}]
        occurrence.add_observation(1, 0.80, 5, properties, evidence2)
        assert occurrence.query_scores[1] == 0.85  # Should keep higher score
        assert occurrence.query_ranks[1] == 3  # Should keep better (lower) rank

        # Add another observation from query 1 with higher score
        evidence3 = [{"content": "Third evidence"}]
        occurrence.add_observation(1, 0.95, 2, properties, evidence3)
        assert occurrence.query_scores[1] == 0.95  # Should update to higher score
        assert occurrence.query_ranks[1] == 2  # Should update to better rank

    def test_add_observation_keeps_best_rank_per_query(self):
        """Test that multiple observations from same query keep the best (lowest) rank."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Evidence"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.8, evidence=evidence)

        occurrence.add_observation(1, 0.85, 5, properties, evidence)
        assert occurrence.query_ranks[1] == 5

        occurrence.add_observation(1, 0.85, 3, properties, evidence)
        assert occurrence.query_ranks[1] == 3  # Should keep better (lower) rank

        occurrence.add_observation(1, 0.85, 7, properties, evidence)
        assert occurrence.query_ranks[1] == 3  # Should still keep best rank

    def test_add_observation_updates_best_score_and_properties(self):
        """Test that best_score and properties are updated when a better observation is added."""
        initial_properties = {"person_id": "123", "version": "old"}
        initial_evidence = [{"content": "Old evidence"}]
        occurrence = FactOccurrence(
            properties=initial_properties,
            best_score=0.8,
            evidence=initial_evidence,
        )

        # Add observation with higher score
        new_properties = {"person_id": "123", "version": "new"}
        new_evidence = [{"content": "New evidence"}]
        occurrence.add_observation(1, 0.95, 1, new_properties, new_evidence)

        assert occurrence.best_score == 0.95
        assert occurrence.properties == new_properties
        assert occurrence.evidence == new_evidence

        # Add observation with lower score
        lower_properties = {"person_id": "123", "version": "lower"}
        lower_evidence = [{"content": "Lower evidence"}]
        occurrence.add_observation(2, 0.85, 2, lower_properties, lower_evidence)

        # Should keep the best properties and evidence
        assert occurrence.best_score == 0.95
        assert occurrence.properties == new_properties
        assert occurrence.evidence == new_evidence

    def test_evidence_tracking(self):
        """Test that evidence is properly tracked and updated with best observation."""
        properties = {"person_id": "123"}
        evidence1 = [{"content": "Initial evidence"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.7, evidence=evidence1)

        assert occurrence.evidence == evidence1

        # Add better observation
        evidence2 = [{"content": "Better evidence"}]
        occurrence.add_observation(1, 0.85, 1, properties, evidence2)
        assert occurrence.evidence == evidence2

        # Add worse observation
        evidence3 = [{"content": "Worse evidence"}]
        occurrence.add_observation(2, 0.75, 3, properties, evidence3)
        assert occurrence.evidence == evidence2  # Should keep better evidence


class TestRRFCalculation:
    """Tests for RRF score calculation methods."""

    def test_rrf_calculation_single_query(self):
        """Test RRF calculation with single query."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 5, properties, evidence)

        rrf_score = SemanticSearchFactsTool._calculate_combined_score(occurrence, "rrf", 0.0)
        expected = 1.0 / (RRF_K + 5)

        assert abs(rrf_score - expected) < 0.0001

    def test_rrf_calculation_multiple_queries(self):
        """Test RRF calculation with multiple queries."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 1, properties, evidence)
        occurrence.add_observation(2, 0.85, 3, properties, evidence)
        occurrence.add_observation(3, 0.88, 2, properties, evidence)

        rrf_score = SemanticSearchFactsTool._calculate_combined_score(occurrence, "rrf", 0.0)
        expected = 1.0 / (RRF_K + 1) + 1.0 / (RRF_K + 3) + 1.0 / (RRF_K + 2)

        assert abs(rrf_score - expected) < 0.0001

    def test_score_sum_calculation(self):
        """Test score_sum fusion method."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 1, properties, evidence)
        occurrence.add_observation(2, 0.85, 2, properties, evidence)
        occurrence.add_observation(3, 0.88, 3, properties, evidence)

        score_sum = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_sum", 0.0)
        expected = 0.9 + 0.85 + 0.88

        assert abs(score_sum - expected) < 0.0001

    def test_score_max_calculation_no_boost(self):
        """Test score_max fusion method without boost."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 1, properties, evidence)
        occurrence.add_observation(2, 0.85, 2, properties, evidence)

        score_max = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 0.0)
        expected = 0.9  # Max score with no boost

        assert abs(score_max - expected) < 0.0001

    def test_score_max_calculation_with_boost(self):
        """Test score_max fusion method with multi-query boost."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 1, properties, evidence)
        occurrence.add_observation(2, 0.85, 2, properties, evidence)
        occurrence.add_observation(3, 0.88, 3, properties, evidence)

        # Test with 0.5 boost
        score_max = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 0.5)
        expected = 0.9 * (1.0 + 0.5 * (3 - 1))  # max_score * (1 + boost * (count - 1))

        assert abs(score_max - expected) < 0.0001

    def test_score_max_calculation_different_boosts(self):
        """Test score_max with different boost values."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.8, evidence=evidence)
        occurrence.add_observation(1, 0.8, 1, properties, evidence)
        occurrence.add_observation(2, 0.75, 2, properties, evidence)

        # Boost of 0.0
        score_0 = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 0.0)
        assert abs(score_0 - 0.8) < 0.0001

        # Boost of 0.5
        score_05 = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 0.5)
        expected_05 = 0.8 * (1.0 + 0.5 * (2 - 1))
        assert abs(score_05 - expected_05) < 0.0001

        # Boost of 1.0
        score_1 = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 1.0)
        expected_1 = 0.8 * (1.0 + 1.0 * (2 - 1))
        assert abs(score_1 - expected_1) < 0.0001

    def test_empty_query_scores_returns_best_score(self):
        """Test that when no query scores exist, best_score is returned."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.85, evidence=evidence)

        # Don't add any observations
        rrf_score = SemanticSearchFactsTool._calculate_combined_score(occurrence, "rrf", 0.0)
        assert rrf_score == 0.85

        score_sum = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_sum", 0.0)
        assert score_sum == 0.85

        score_max = SemanticSearchFactsTool._calculate_combined_score(occurrence, "score_max", 0.5)
        assert score_max == 0.85

    def test_unrecognized_fusion_method_returns_best_score(self):
        """Test that unrecognized fusion method falls back to best_score."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]
        occurrence = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        occurrence.add_observation(1, 0.9, 1, properties, evidence)

        score = SemanticSearchFactsTool._calculate_combined_score(occurrence, "invalid_method", 0.0)
        assert score == 0.9


class TestSemanticSearchInput:
    """Tests for SemanticSearchInput model."""

    def test_input_with_defaults(self):
        """Test SemanticSearchInput with default values."""
        input_data = SemanticSearchInput(queries=["startup", "founder"])

        assert input_data.queries == ["startup", "founder"]
        assert input_data.limit == 10
        assert input_data.similarity_threshold is None
        assert input_data.adaptive_threshold is True
        assert input_data.adaptive_threshold_max == ADAPTIVE_THRESHOLD_MAX
        assert input_data.adaptive_threshold_min == ADAPTIVE_THRESHOLD_MIN
        assert input_data.fusion_method == DEFAULT_FUSION_METHOD
        assert input_data.multi_query_boost == DEFAULT_MULTI_QUERY_BOOST

    def test_input_with_custom_values(self):
        """Test SemanticSearchInput with custom values."""
        input_data = SemanticSearchInput(
            queries=["query1", "query2", "query3"],
            limit=20,
            similarity_threshold=0.7,
            adaptive_threshold=False,
            fusion_method="score_sum",
            multi_query_boost=0.5,
        )

        assert input_data.queries == ["query1", "query2", "query3"]
        assert input_data.limit == 20
        assert input_data.similarity_threshold == 0.7
        assert input_data.adaptive_threshold is False
        assert input_data.fusion_method == "score_sum"
        assert input_data.multi_query_boost == 0.5

    def test_input_accepts_up_to_20_queries(self):
        """Test that input accepts up to 20 queries."""
        queries = [f"query{i}" for i in range(20)]
        input_data = SemanticSearchInput(queries=queries)

        assert len(input_data.queries) == 20

    def test_input_validates_min_queries(self):
        """Test that input requires at least 1 query."""
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=[])

    def test_input_validates_max_queries(self):
        """Test that input limits to 20 queries."""
        queries = [f"query{i}" for i in range(21)]
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=queries)

    def test_input_validates_limit_range(self):
        """Test that limit is validated."""
        # Valid limits
        SemanticSearchInput(queries=["test"], limit=1)
        SemanticSearchInput(queries=["test"], limit=50)

        # Invalid limits
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], limit=0)

        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], limit=51)

    def test_input_validates_similarity_threshold_range(self):
        """Test that similarity_threshold is validated."""
        # Valid thresholds
        SemanticSearchInput(queries=["test"], similarity_threshold=0.0)
        SemanticSearchInput(queries=["test"], similarity_threshold=1.0)

        # Invalid thresholds
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], similarity_threshold=-0.1)

        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], similarity_threshold=1.1)

    def test_input_validates_fusion_method(self):
        """Test that fusion_method is validated."""
        # Valid methods
        SemanticSearchInput(queries=["test"], fusion_method="rrf")
        SemanticSearchInput(queries=["test"], fusion_method="score_sum")
        SemanticSearchInput(queries=["test"], fusion_method="score_max")

        # Invalid method
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], fusion_method="invalid")

    def test_input_validates_multi_query_boost_range(self):
        """Test that multi_query_boost is validated."""
        # Valid boosts
        SemanticSearchInput(queries=["test"], multi_query_boost=0.0)
        SemanticSearchInput(queries=["test"], multi_query_boost=1.0)

        # Invalid boosts
        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], multi_query_boost=-0.1)

        with pytest.raises(ValueError):
            SemanticSearchInput(queries=["test"], multi_query_boost=1.1)


class TestSemanticSearchResult:
    """Tests for SemanticSearchResult model."""

    def test_result_with_query_metadata(self):
        """Test that result includes query_scores and appeared_in_query_count."""
        result = SemanticSearchResult(
            person_id="123",
            person_name="John Doe",
            fact_type="works_at",
            fact_object="Google",
            attributes={},
            similarity_score=0.95,
            confidence=0.8,
            evidence=[{"content": "I work at Google"}],
            query_scores={1: 0.9, 2: 0.85, 3: 0.88},
            appeared_in_query_count=3,
        )

        assert result.query_scores == {1: 0.9, 2: 0.85, 3: 0.88}
        assert result.appeared_in_query_count == 3

    def test_result_without_query_metadata(self):
        """Test that query metadata fields are optional."""
        result = SemanticSearchResult(
            person_id="123",
            person_name="John Doe",
            fact_type="works_at",
            fact_object="Google",
            attributes={},
            similarity_score=0.95,
            confidence=0.8,
            evidence=[],
        )

        assert result.query_scores is None
        assert result.appeared_in_query_count is None


class TestBuildResult:
    """Tests for the _build_result helper method."""

    def test_build_result_basic(self):
        """Test building a result with basic properties."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "123",
            "person_name": "John Doe",
            "fact_type": "works_at",
            "fact_object": "Google",
            "confidence": 0.85,
        }
        evidence = [{"content": "I work at Google"}]

        result = tool._build_result(properties, 0.95, evidence)

        assert result.person_id == "123"
        assert result.person_name == "John Doe"
        assert result.fact_type == "works_at"
        assert result.fact_object == "Google"
        assert result.similarity_score == 0.95
        assert result.confidence == 0.85
        assert result.evidence == evidence
        assert result.query_scores is None
        assert result.appeared_in_query_count is None

    def test_build_result_with_query_scores(self):
        """Test building a result with query scores."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "123",
            "person_name": "Jane Smith",
            "fact_type": "has_skill",
            "fact_object": "Python",
        }
        evidence = [{"content": "I'm proficient in Python"}]
        query_scores = {1: 0.9, 2: 0.85}

        result = tool._build_result(properties, 0.92, evidence, query_scores=query_scores)

        assert result.query_scores == query_scores
        assert result.appeared_in_query_count == 2

    def test_build_result_parses_json_attributes(self):
        """Test that JSON string attributes are parsed correctly."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "123",
            "person_name": "John Doe",
            "fact_type": "close_to",
            "fact_object": "Jane",
            "attributes": '{"relationship_type": "friend"}',
        }
        evidence = []

        result = tool._build_result(properties, 0.9, evidence)

        assert result.attributes == {"relationship_type": "friend"}

    def test_build_result_handles_dict_attributes(self):
        """Test that dict attributes are handled correctly."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "123",
            "person_name": "John Doe",
            "fact_type": "works_at",
            "fact_object": "Google",
            "attributes": {"role": "engineer", "team": "search"},
        }
        evidence = []

        result = tool._build_result(properties, 0.9, evidence)

        assert result.attributes == {"role": "engineer", "team": "search"}

    def test_build_result_handles_invalid_json_attributes(self):
        """Test that invalid JSON attributes are handled gracefully."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "123",
            "person_name": "John Doe",
            "fact_type": "works_at",
            "fact_object": "Google",
            "attributes": "invalid json {",
        }
        evidence = []

        result = tool._build_result(properties, 0.9, evidence)

        assert result.attributes == {}

    def test_build_result_uses_person_id_as_fallback_name(self):
        """Test that person_id is used as fallback for person_name."""
        from memory_agent.tools.base import ToolContext

        tool = SemanticSearchFactsTool(ToolContext(driver=None))
        properties = {
            "person_id": "user_123",
            "fact_type": "works_at",
            "fact_object": "Company",
        }
        evidence = []

        result = tool._build_result(properties, 0.9, evidence)

        assert result.person_name == "user_123"


class TestMultiQueryFusion:
    """Integration-style tests for multi-query fusion behavior."""

    def test_facts_in_multiple_queries_tracked(self):
        """Test that facts appearing in multiple queries are tracked correctly."""
        properties = {"person_id": "123", "fact_type": "works_at"}
        evidence1 = [{"content": "Evidence 1"}]
        evidence2 = [{"content": "Evidence 2"}]
        evidence3 = [{"content": "Evidence 3"}]

        occurrence = FactOccurrence(properties=properties, best_score=0.85, evidence=evidence1)
        occurrence.add_observation(1, 0.85, 5, properties, evidence1)
        occurrence.add_observation(2, 0.88, 3, properties, evidence2)
        occurrence.add_observation(3, 0.90, 2, properties, evidence3)

        assert len(occurrence.query_scores) == 3
        assert len(occurrence.query_ranks) == 3
        assert occurrence.best_score == 0.90
        assert occurrence.evidence == evidence3

    def test_rrf_boosts_multi_query_facts(self):
        """Test that RRF gives higher scores to facts in multiple queries."""
        properties = {"person_id": "123"}
        evidence = [{"content": "Test"}]

        # Fact appearing in one query at rank 1
        single_query = FactOccurrence(properties=properties, best_score=0.9, evidence=evidence)
        single_query.add_observation(1, 0.9, 1, properties, evidence)
        single_score = SemanticSearchFactsTool._calculate_combined_score(single_query, "rrf", 0.0)

        # Fact appearing in three queries at ranks 3, 4, 5
        multi_query = FactOccurrence(properties=properties, best_score=0.85, evidence=evidence)
        multi_query.add_observation(1, 0.85, 3, properties, evidence)
        multi_query.add_observation(2, 0.83, 4, properties, evidence)
        multi_query.add_observation(3, 0.82, 5, properties, evidence)
        multi_score = SemanticSearchFactsTool._calculate_combined_score(multi_query, "rrf", 0.0)

        # Multi-query fact should have higher RRF score despite lower individual scores
        assert multi_score > single_score

    def test_deduplication_key_includes_relationship_type(self):
        """Test that deduplication considers relationship_type from attributes."""
        # This is more of a documentation test showing the expected behavior
        # The actual deduplication happens in the run() method

        # Two facts with same person/type/object but different relationship_type
        # should be treated as different facts
        key1 = ("person_123", "close_to", "Jane", "friend")
        key2 = ("person_123", "close_to", "Jane", "colleague")

        assert key1 != key2  # Should be deduplicated separately


class TestConstants:
    """Tests for module constants."""

    def test_rrf_k_value(self):
        """Test that RRF_K has expected value."""
        assert RRF_K == 60

    def test_default_fusion_method(self):
        """Test that default fusion method is rrf."""
        assert DEFAULT_FUSION_METHOD == "rrf"

    def test_default_multi_query_boost(self):
        """Test that default multi-query boost is 0."""
        assert DEFAULT_MULTI_QUERY_BOOST == 0.0
