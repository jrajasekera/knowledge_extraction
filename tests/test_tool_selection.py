"""Tests for tool selection filtering in the memory agent iterative loop."""

from __future__ import annotations

from memory_agent.agent import _LOOP_TOOL_NAMES
from memory_agent.llm import TOOL_PROMPT_INFO
from memory_agent.tools import build_toolkit
from memory_agent.tools.base import ToolContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyDriver:
    def session(self):  # pragma: no cover
        raise AssertionError("session should not be opened in tests")


def _toolkit_tool_names() -> set[str]:
    """Return the set of tool names registered by build_toolkit."""
    ctx = ToolContext(driver=_DummyDriver())  # type: ignore[arg-type]
    return set(build_toolkit(ctx).keys())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToolPromptInfoCoverage:
    def test_all_toolkit_tools_have_prompt_info(self) -> None:
        """Every tool in build_toolkit() should have a TOOL_PROMPT_INFO entry.

        Catches the original bug pattern where a new tool is added to the
        toolkit but not given prompt metadata, causing 'No description available'.
        """
        tool_names = _toolkit_tool_names()
        missing = tool_names - set(TOOL_PROMPT_INFO.keys())
        assert not missing, f"Tools missing from TOOL_PROMPT_INFO: {missing}"


class TestLoopToolNames:
    def test_is_subset_of_toolkit(self) -> None:
        """_LOOP_TOOL_NAMES should only contain tools that exist in the toolkit."""
        tool_names = _toolkit_tool_names()
        stale = _LOOP_TOOL_NAMES - tool_names
        assert not stale, f"_LOOP_TOOL_NAMES contains unknown tools: {stale}"

    def test_excludes_message_search(self) -> None:
        """semantic_search_messages must not be in _LOOP_TOOL_NAMES.

        Message retrieval is handled by synthesize, not the iterative loop.
        """
        assert "semantic_search_messages" not in _LOOP_TOOL_NAMES

    def test_includes_fact_search(self) -> None:
        """semantic_search_facts must be in _LOOP_TOOL_NAMES."""
        assert "semantic_search_facts" in _LOOP_TOOL_NAMES
