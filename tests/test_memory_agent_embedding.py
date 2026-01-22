from __future__ import annotations

from collections import deque

from memory_agent.embedding_pipeline import GraphFact, generate_embeddings
from memory_agent.fact_formatter import format_fact_for_embedding_text


def test_format_fact_for_embedding_text_includes_attributes():
    result = format_fact_for_embedding_text(
        person_name="Alice",
        fact_type="WORKS_AT",
        fact_object="OpenAI",
        attributes={"role": "Researcher", "location": "SF", "empty": ""},
    )
    assert result.startswith("Alice works at OpenAI")
    assert "role=Researcher" in result
    assert "location=SF" in result
    assert "empty" not in result


def test_format_fact_for_embedding_text_includes_evidence():
    result = format_fact_for_embedding_text(
        person_name="Bob",
        fact_type="HAS_SKILL",
        fact_object="Rust",
        attributes={"level": "expert"},
        evidence_text=[
            "I've been programming in Rust for 5 years",
            "Just shipped a major Rust project at work",
        ],
    )
    assert result.startswith("Bob has skill Rust")
    assert "level=expert" in result
    assert "Evidence:" in result
    assert "I've been programming in Rust for 5 years" in result
    assert "Just shipped a major Rust project at work" in result


class DummyProvider:
    def __init__(self) -> None:
        self.calls: deque[str] = deque()

    def embed(self, texts):  # noqa: ANN001 - simplified for tests
        self.calls.extend(texts)
        return [[0.1, 0.2] for _ in texts]


def test_generate_embeddings_invokes_provider():
    facts = [
        GraphFact(
            fact_id=1,
            person_id="user1",
            person_name="Alice",
            fact_type="HAS_SKILL",
            fact_object="Python",
            attributes={"proficiency": "advanced", "metadata": {"basis": "friends"}},
            confidence=0.9,
            evidence=["msg1"],
            evidence_text=["She mentioned being really good at Python programming"],
            target_labels=["Skill"],
        )
    ]
    provider = DummyProvider()
    rows = generate_embeddings(facts, provider)  # type: ignore[arg-type]

    assert len(rows) == 1
    assert provider.calls
    row = rows[0]
    assert row["embedding"] == [0.1, 0.2]
    assert row["fact_id"] == 1
    assert row["fact_type"] == "HAS_SKILL"
    assert row["attributes_json"]
    # Verify evidence is included in the embedded text
    assert "Evidence:" in row["text"]
    assert "She mentioned being really good at Python programming" in row["text"]
