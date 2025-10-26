"""LangGraph agent state definitions."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph import add_messages

from .models import MessageModel, RetrievedFact
from .tools.semantic_search_messages import SemanticSearchMessageResult


class AgentState(TypedDict, total=False):
    """Mutable state tracked by the memory agent workflow."""

    # Static inputs
    conversation: list[MessageModel]
    channel_id: str
    max_facts: int
    max_messages: int
    max_iterations: int

    # LangGraph specific messages container
    messages: Annotated[list[Any], add_messages]

    # Accumulated results
    retrieved_facts: list[RetrievedFact]
    retrieved_messages: list[SemanticSearchMessageResult]
    tool_calls: list[dict[str, Any]]
    iteration: int

    # Reasoning metadata
    current_goal: str | None
    identified_entities: dict[str, Any]
    pending_tool: dict[str, Any] | None
    goal_accomplished: bool

    # Output data
    formatted_facts: list[str]
    formatted_messages: list[str]
    confidence: str
    metadata: dict[str, Any]
    reasoning_trace: list[str]

    # Enhanced LLM integration
    llm_reasoning: list[dict[str, Any]]
    tool_selection_confidence: str
    fact_assessments: dict[str, dict[str, Any]]
    entity_extraction_results: dict[str, Any]
    should_stop_evaluation: dict[str, Any]
