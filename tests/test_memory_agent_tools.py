from __future__ import annotations

import pytest

from memory_agent.tools.base import ToolContext
from memory_agent.tools.people_by_skill import FindPeopleBySkillTool


class DummyDriver:
    def session(self):  # pragma: no cover - patched in tests
        raise AssertionError("session should not be called in this test")


def test_find_people_by_skill_tool(monkeypatch):
    context = ToolContext(driver=DummyDriver(), embeddings_model=None)

    def fake_query(_context, query, params):  # noqa: ARG001
        return [
            {
                "person_id": "user123",
                "name": "Alice",
                "proficiency": "advanced",
                "years_experience": 5,
                "confidence": 0.92,
                "evidence": ["msg_1"],
            }
        ]

    monkeypatch.setattr(
        "memory_agent.tools.people_by_skill.run_read_query",
        fake_query,
    )

    tool = FindPeopleBySkillTool(context)
    result = tool({"skill": "Python"})

    assert result.skill == "Python"
    assert len(result.people) == 1
    person = result.people[0]
    assert person.person_id == "user123"
    assert person.proficiency == "advanced"
    assert person.confidence == pytest.approx(0.92)
