"""Tests for memory_agent/tools/base.py."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from memory_agent.tools.base import ToolBase, ToolContext, ToolError


class SampleInput(BaseModel):
    """Sample input model for testing."""

    query: str
    limit: int = 10


class SampleOutput(BaseModel):
    """Sample output model for testing."""

    results: list[str]


class SampleTool(ToolBase[SampleInput, SampleOutput]):
    """Concrete tool implementation for testing."""

    input_model = SampleInput
    output_model = SampleOutput

    def run(self, input_data: SampleInput) -> dict[str, Any]:
        """Return sample results."""
        return {"results": [f"result for: {input_data.query}"]}


class FailingTool(ToolBase[SampleInput, SampleOutput]):
    """Tool that raises an exception during run."""

    input_model = SampleInput
    output_model = SampleOutput

    def run(self, input_data: SampleInput) -> dict[str, Any]:
        """Always raises an exception."""
        raise ValueError("Something went wrong")


class ToolErrorTool(ToolBase[SampleInput, SampleOutput]):
    """Tool that raises ToolError directly."""

    input_model = SampleInput
    output_model = SampleOutput

    def run(self, input_data: SampleInput) -> dict[str, Any]:
        """Raises ToolError directly."""
        raise ToolError("Explicit tool error")


def test_tool_context_session_returns_driver_session() -> None:
    """ToolContext.session() should return a session from the driver."""
    mock_driver = MagicMock()
    mock_session = MagicMock()
    mock_driver.session.return_value = mock_session

    ctx = ToolContext(driver=mock_driver)
    result = ctx.session()

    assert result is mock_session
    mock_driver.session.assert_called_once()


def test_tool_context_with_embeddings_model() -> None:
    """ToolContext can be created with an embeddings model."""
    mock_driver = MagicMock()
    mock_embeddings = MagicMock()

    ctx = ToolContext(driver=mock_driver, embeddings_model=mock_embeddings)

    assert ctx.driver is mock_driver
    assert ctx.embeddings_model is mock_embeddings


def test_tool_base_init_sets_name_and_context() -> None:
    """ToolBase.__init__ should set the name and context."""
    mock_driver = MagicMock()
    ctx = ToolContext(driver=mock_driver)

    tool = SampleTool(ctx)

    assert tool.name == "SampleTool"
    assert tool.context is ctx


def test_tool_base_call_validates_input_and_returns_output() -> None:
    """ToolBase.__call__ should validate input and return validated output."""
    mock_driver = MagicMock()
    ctx = ToolContext(driver=mock_driver)
    tool = SampleTool(ctx)

    result = tool({"query": "test query", "limit": 5})

    assert isinstance(result, SampleOutput)
    assert result.results == ["result for: test query"]


def test_tool_base_call_wraps_exceptions_in_tool_error() -> None:
    """ToolBase.__call__ should wrap non-ToolError exceptions in ToolError."""
    mock_driver = MagicMock()
    ctx = ToolContext(driver=mock_driver)
    tool = FailingTool(ctx)

    with pytest.raises(ToolError) as exc_info:
        tool({"query": "test"})

    assert "Something went wrong" in str(exc_info.value)


def test_tool_base_call_re_raises_tool_error() -> None:
    """ToolBase.__call__ should re-raise ToolError without wrapping."""
    mock_driver = MagicMock()
    ctx = ToolContext(driver=mock_driver)
    tool = ToolErrorTool(ctx)

    with pytest.raises(ToolError) as exc_info:
        tool({"query": "test"})

    assert str(exc_info.value) == "Explicit tool error"


def test_tool_base_run_raises_not_implemented() -> None:
    """ToolBase.run() should raise NotImplementedError in base class."""

    class UnimplementedTool(ToolBase[SampleInput, SampleOutput]):
        input_model = SampleInput
        output_model = SampleOutput

    mock_driver = MagicMock()
    ctx = ToolContext(driver=mock_driver)
    tool = UnimplementedTool(ctx)

    with pytest.raises(NotImplementedError):
        tool.run(SampleInput(query="test"))
