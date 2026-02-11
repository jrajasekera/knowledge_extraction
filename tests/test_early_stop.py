"""Tests for the novelty-aware early-stop behavior in the memory agent."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from memory_agent.agent import (
    MemoryAgent,
    _confidence_meets_threshold,
    compute_novelty,
    evaluate_next_step,
    fact_key,
    should_continue,
)
from memory_agent.config import AgentConfig, Settings
from memory_agent.models import MessageModel, RetrievalRequest, RetrievedFact

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fact(
    person_id: str = "p1",
    fact_type: str = "WORKS_AT",
    fact_object: str | None = "Acme",
    relationship_type: str | None = None,
    confidence: float = 0.8,
) -> RetrievedFact:
    attrs: dict = {}
    if relationship_type is not None:
        attrs["relationship_type"] = relationship_type
    return RetrievedFact(
        person_id=person_id,
        person_name=f"Person {person_id}",
        fact_type=fact_type,
        fact_object=fact_object,
        attributes=attrs,
        confidence=confidence,
    )


# ---------------------------------------------------------------------------
# 1. compute_novelty — new facts
# ---------------------------------------------------------------------------


class TestComputeNovelty:
    def test_new_facts(self) -> None:
        facts = [_make_fact("p1", "WORKS_AT", "Acme"), _make_fact("p2", "HAS_SKILL", "Rust")]
        new_count, streak, seen = compute_novelty(facts, [], novelty_min_new_facts=1, prev_streak=0)
        assert new_count == 2
        assert streak == 0
        assert len(seen) == 2

    # 2. compute_novelty — duplicate facts
    def test_duplicate_facts(self) -> None:
        existing = [["p1", "WORKS_AT", "Acme", None]]
        facts = [_make_fact("p1", "WORKS_AT", "Acme")]
        new_count, streak, seen = compute_novelty(
            facts, existing, novelty_min_new_facts=1, prev_streak=0
        )
        assert new_count == 0
        assert streak == 1
        assert len(seen) == 1  # no growth

    # 3. compute_novelty — within-batch dedup
    def test_within_batch_dedup(self) -> None:
        f = _make_fact("p1", "WORKS_AT", "Acme")
        facts = [f, f]
        new_count, streak, seen = compute_novelty(facts, [], novelty_min_new_facts=1, prev_streak=0)
        assert new_count == 1
        assert len(seen) == 1

    # 4. compute_novelty — None fact_object
    def test_none_fact_object(self) -> None:
        f1 = _make_fact("p1", "WORKS_AT", None)
        f2 = _make_fact("p1", "WORKS_AT", None)
        new_count, streak, seen = compute_novelty(
            [f1, f2], [], novelty_min_new_facts=1, prev_streak=0
        )
        assert new_count == 1
        assert len(seen) == 1

    # 5. compute_novelty — relationship_type distinguishes facts
    def test_relationship_type_distinguishes_facts(self) -> None:
        f1 = _make_fact("p1", "CLOSE_TO", "p2", relationship_type="friend")
        f2 = _make_fact("p1", "CLOSE_TO", "p2", relationship_type="colleague")
        new_count, streak, seen = compute_novelty(
            [f1, f2], [], novelty_min_new_facts=1, prev_streak=0
        )
        assert new_count == 2

    # 6. compute_novelty — streak resets
    def test_streak_resets(self) -> None:
        facts = [_make_fact("p1", "WORKS_AT", "Acme")]
        new_count, streak, _ = compute_novelty(facts, [], novelty_min_new_facts=1, prev_streak=3)
        assert new_count == 1
        assert streak == 0  # reset because new_count >= min


# ---------------------------------------------------------------------------
# 7. _confidence_meets_threshold
# ---------------------------------------------------------------------------


class TestConfidenceMeetsThreshold:
    def test_high_meets_high(self) -> None:
        assert _confidence_meets_threshold("high", "high") is True

    def test_medium_below_high(self) -> None:
        assert _confidence_meets_threshold("medium", "high") is False

    def test_high_exceeds_low(self) -> None:
        assert _confidence_meets_threshold("high", "low") is True

    def test_low_meets_low(self) -> None:
        assert _confidence_meets_threshold("low", "low") is True


# ---------------------------------------------------------------------------
# 8-10. evaluate_next_step
# ---------------------------------------------------------------------------


class TestEvaluateNextStep:
    def test_stops_when_goal_accomplished(self) -> None:
        state = {
            "goal_accomplished": True,
            "iteration": 3,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "tool_calls": [],
        }
        assert evaluate_next_step(state) == "finish"  # type: ignore[arg-type]

    def test_continues_when_not_accomplished(self) -> None:
        state = {
            "goal_accomplished": False,
            "iteration": 1,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "tool_calls": [],
        }
        assert evaluate_next_step(state) == "continue"  # type: ignore[arg-type]

    def test_stop_decision_requires_goal_accomplished(self) -> None:
        state = {
            "should_stop_evaluation": {"should_continue": False},
            "goal_accomplished": False,
            "iteration": 3,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "tool_calls": [],
        }
        assert evaluate_next_step(state) == "continue"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 11-12. should_continue
# ---------------------------------------------------------------------------


class TestShouldContinue:
    def test_floor_ignores_stop_decision(self) -> None:
        state = {
            "should_stop_evaluation": {"should_continue": False},
            "goal_accomplished": True,
            "iteration": 0,
            "early_stop_min_iterations": 2,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "pending_tool": {"name": "semantic_search_facts", "input": {}},
        }
        assert should_continue(state) == "continue"  # type: ignore[arg-type]

    def test_honors_stop_above_floor(self) -> None:
        state = {
            "should_stop_evaluation": {"should_continue": False},
            "goal_accomplished": True,
            "iteration": 3,
            "early_stop_min_iterations": 2,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "pending_tool": {"name": "semantic_search_facts", "input": {}},
        }
        assert should_continue(state) == "finish"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 13. Regression: stops on no novelty
# ---------------------------------------------------------------------------


class TestRegressionStopsOnNoNovelty:
    def test_stops_on_no_novelty(self) -> None:
        state = {
            "novelty_streak_without_gain": 2,
            "novelty_patience": 2,
            "iteration": 3,
            "early_stop_min_iterations": 2,
            "goal_accomplished": True,
            "max_iterations": 10,
            "retrieved_facts": [],
            "max_facts": 30,
            "tool_calls": [],
        }
        assert evaluate_next_step(state) == "finish"  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 14. Config validation
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_rejects_negative_patience(self) -> None:
        with (
            patch.dict(os.environ, {"NOVELTY_PATIENCE": "0"}, clear=False),
            pytest.raises(ValueError, match="NOVELTY_PATIENCE must be >= 1"),
        ):
            Settings.from_env()

    def test_rejects_invalid_confidence(self) -> None:
        with (
            patch.dict(os.environ, {"STOP_CONFIDENCE_REQUIRED": "ultra"}, clear=False),
            pytest.raises(ValueError, match="STOP_CONFIDENCE_REQUIRED must be low/medium/high"),
        ):
            Settings.from_env()

    def test_rejects_negative_min_iterations(self) -> None:
        with (
            patch.dict(os.environ, {"EARLY_STOP_MIN_ITERATIONS": "-1"}, clear=False),
            pytest.raises(ValueError, match="EARLY_STOP_MIN_ITERATIONS must be >= 0"),
        ):
            Settings.from_env()

    def test_rejects_negative_min_new_facts(self) -> None:
        with (
            patch.dict(os.environ, {"NOVELTY_MIN_NEW_FACTS": "-1"}, clear=False),
            pytest.raises(ValueError, match="NOVELTY_MIN_NEW_FACTS must be >= 0"),
        ):
            Settings.from_env()


# ---------------------------------------------------------------------------
# 15. Integration test: agent continues past first successful call
# ---------------------------------------------------------------------------


class DummyGraph:
    """Fake compiled graph that records initial state and returns it with simulated iterations."""

    def __init__(self) -> None:
        self.invocations: list[dict | None] = []

    async def ainvoke(self, state, config=None):
        self.invocations.append(config)
        # Simulate 2 iterations completed
        state["iteration"] = 2
        state["new_facts_last_iteration"] = 0
        state["novelty_streak_without_gain"] = 0
        state["seen_fact_keys"] = []
        return state


@pytest.mark.asyncio
async def test_agent_initializes_novelty_state(monkeypatch) -> None:
    """Verify MemoryAgent.run passes the novelty state fields into initial_state."""
    dummy_graph = DummyGraph()

    def fake_graph_builder(*_args, **_kwargs):
        return dummy_graph

    monkeypatch.setattr("memory_agent.agent.create_memory_agent_graph", fake_graph_builder)

    agent_config = AgentConfig(
        max_iterations=5,
        early_stop_min_iterations=3,
        novelty_min_new_facts=2,
        novelty_patience=4,
        stop_confidence_required="medium",
    )
    agent = MemoryAgent({}, agent_config)

    message = MessageModel(
        author_id="u1",
        author_name="Alice",
        content="Who knows Rust?",
        timestamp=datetime.now(tz=UTC),
        message_id="msg_1",
    )
    request = RetrievalRequest(messages=[message], channel_id="ch1")
    result = await agent.run(request)

    # Check that novelty metadata is surfaced
    meta = result["metadata"]
    assert "new_facts_last_iteration" in meta
    assert "novelty_streak_without_gain" in meta
    assert "unique_facts_seen" in meta


# ---------------------------------------------------------------------------
# fact_key helper
# ---------------------------------------------------------------------------


class TestFactKey:
    def test_basic_fact_key(self) -> None:
        f = _make_fact("p1", "WORKS_AT", "Acme")
        assert fact_key(f) == ("p1", "WORKS_AT", "Acme", None)

    def test_fact_key_with_relationship_type(self) -> None:
        f = _make_fact("p1", "CLOSE_TO", "p2", relationship_type="friend")
        assert fact_key(f) == ("p1", "CLOSE_TO", "p2", "friend")

    def test_fact_key_none_object(self) -> None:
        f = _make_fact("p1", "HAS_SKILL", None)
        assert fact_key(f) == ("p1", "HAS_SKILL", None, None)
