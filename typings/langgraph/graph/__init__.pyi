"""Type stubs for langgraph.graph module."""

from collections.abc import Callable, Mapping
from typing import Any, TypeVar

StateType = TypeVar("StateType", bound=Mapping[str, Any])

# Message annotation helper
def add_messages(left: list[Any], right: list[Any]) -> list[Any]: ...

# Sentinel for graph end node
END: str

class StateGraph[StateType]:
    """LangGraph state graph for building workflows."""

    def __init__(self, state_schema: type[StateType]) -> None: ...
    def add_node(
        self,
        name: str,
        action: Callable[[StateType], StateType | dict[str, Any]] | Callable[[StateType], Any],
    ) -> None: ...
    def add_edge(self, start: str, end: str) -> None: ...
    def add_conditional_edges(
        self,
        source: str,
        condition: Callable[[StateType], str],
        branches: dict[str, str],
    ) -> None: ...
    def set_entry_point(self, node: str) -> None: ...
    def compile(self) -> CompiledGraph[StateType]: ...

class CompiledGraph[StateType]:
    """Compiled LangGraph workflow."""

    async def ainvoke(
        self, input: StateType, config: dict[str, Any] | None = None
    ) -> StateType: ...
    def invoke(self, input: StateType, config: dict[str, Any] | None = None) -> StateType: ...
