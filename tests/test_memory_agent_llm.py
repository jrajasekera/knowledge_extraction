from __future__ import annotations

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from memory_agent.llm import LLMClient
from memory_agent.tools.base import ToolBase, ToolContext


class DummyInput(BaseModel):
    pass


class DummyOutput(BaseModel):
    pass


class DummyDriver:
    def session(self):  # pragma: no cover - unused in tests
        raise AssertionError("session should not be opened in tests")


@dataclass
class DummyTool(ToolBase[DummyInput, DummyOutput]):
    input_model = DummyInput
    output_model = DummyOutput

    def __init__(self, context: ToolContext) -> None:
        super().__init__(context)

    def run(self, input_data: DummyInput) -> DummyOutput:  # pragma: no cover - not used
        return DummyOutput()


def test_tool_catalog_generation_includes_known_tools():
    client = LLMClient(model="GLM-4.5-Air", temperature=0.3)
    context = ToolContext(driver=DummyDriver())
    catalog = client._build_tool_catalog(
        {
            "get_person_profile": DummyTool(context),
            "semantic_search_facts": DummyTool(context),
        }
    )

    assert "get_person_profile" in catalog
    assert "semantic_search_facts" in catalog
    assert "Description" in catalog
    assert "Use When" in catalog


@pytest.mark.asyncio
async def test_aplan_tool_usage_fallback_when_llm_unavailable():
    client = LLMClient(model="GLM-4.5-Air", temperature=0.3)
    # Force llama client to be None to exercise fallback behavior
    client._llama_client = None

    heuristic_candidate = {"name": "semantic_search_facts", "input": {"query": "test"}}
    state = {
        "conversation": [],
        "identified_entities": {},
        "tool_calls": [],
        "retrieved_facts": [],
    }
    tools = {"semantic_search_facts": DummyTool(ToolContext(driver=DummyDriver()))}

    result = await client.aplan_tool_usage("Who knows Go?", tools, state, heuristic_candidate)

    assert result["tool_name"] == "semantic_search_facts"
    assert result["parameters"] == {"query": "test"}
    assert result["should_stop"] is False
