from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memory_agent.agent import MemoryAgent, _estimate_recursion_limit
from memory_agent.config import AgentConfig
from memory_agent.models import MessageModel, RetrievalRequest
from memory_agent.tools import ToolContext, build_toolkit


class DummySession:
    def run(self, query, params=None):  # noqa: ARG002
        return []

    def close(self) -> None:
        pass


class DummyDriver:
    def session(self, **kwargs):  # noqa: ARG002
        return DummySession()

    def close(self) -> None:
        pass


class DummyGraph:
    def __init__(self) -> None:
        self.invocations: list[dict | None] = []

    async def ainvoke(self, state, config=None):  # noqa: ANN001, D401
        """Capture invocation config for assertions."""
        self.invocations.append(config)
        return state


@pytest.mark.asyncio
async def test_memory_agent_sets_recursion_limit(monkeypatch):
    dummy_graph = DummyGraph()

    def fake_graph_builder(*_args, **_kwargs):  # noqa: ANN001
        return dummy_graph

    monkeypatch.setattr("memory_agent.agent.create_memory_agent_graph", fake_graph_builder)

    agent = MemoryAgent({}, AgentConfig(max_iterations=12))
    message = MessageModel(
        author_id="user2",
        author_name="Eve",
        content="Ping",
        timestamp=datetime.now(tz=timezone.utc),
        message_id="msg_2",
    )

    request = RetrievalRequest(messages=[message], channel_id="channel_2", max_iterations=3)
    await agent.run(request)

    assert dummy_graph.invocations, "Graph was never invoked"
    assert dummy_graph.invocations[0] == {"recursion_limit": _estimate_recursion_limit(3)}
