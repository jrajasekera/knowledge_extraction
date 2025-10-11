"""LangGraph agent orchestration for memory retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable

from langgraph.graph import StateGraph, END

from .config import AgentConfig
from .conversation import extract_insights
from .fact_formatter import deduplicate_facts, format_facts
from .llm import LLMClient
from .models import RetrievalRequest, RetrievedFact
from .normalization import normalize_to_facts
from .state import AgentState
from .tools import ToolBase


logger = logging.getLogger(__name__)


class MemoryAgent:
    """Coordinates LangGraph workflow execution."""

    def __init__(
        self,
        tools: Dict[str, ToolBase],
        config: AgentConfig,
        llm: LLMClient | None = None,
    ) -> None:
        self.tools = tools
        self.config = config
        self.llm = llm
        self.graph = create_memory_agent_graph(tools, llm, config)

    async def run(self, request: RetrievalRequest, debug_mode: bool = False) -> dict[str, Any]:
        """Run the memory agent against an incoming request."""
        start = time.perf_counter()
        max_facts = request.max_facts or self.config.max_facts
        max_iterations = request.max_iterations or self.config.max_iterations

        initial_state: AgentState = {
            "conversation": request.messages,
            "channel_id": request.channel_id,
            "max_facts": max_facts,
            "max_iterations": max_iterations,
            "messages": [],
            "retrieved_facts": [],
            "tool_calls": [],
            "iteration": 0,
            "current_goal": None,
            "identified_entities": {},
            "pending_tool": None,
            "goal_accomplished": False,
            "formatted_facts": [],
            "confidence": "low",
            "metadata": {},
            "reasoning_trace": [],
        }
        final_state: AgentState = await self.graph.ainvoke(initial_state)
        processing_time_ms = int((time.perf_counter() - start) * 1000)

        metadata = {
            "queries_executed": len(final_state.get("tool_calls", [])),
            "facts_retrieved": len(final_state.get("retrieved_facts", [])),
            "processing_time_ms": processing_time_ms,
            "iterations_used": final_state.get("iteration", 0),
            "tool_calls": final_state.get("tool_calls", []),
        }

        result: dict[str, Any] = {
            "facts": final_state.get("formatted_facts", []),
            "confidence": final_state.get("confidence", "low"),
            "metadata": metadata,
        }

        if debug_mode:
            result["debug"] = {
                "reasoning_trace": final_state.get("reasoning_trace", []),
                "retrieved_facts": [fact.model_dump() for fact in final_state.get("retrieved_facts", [])],
            }
        return result


def create_memory_agent_graph(
    tools: Dict[str, ToolBase],
    llm: LLMClient | None,
    config: AgentConfig,
):
    """Build the LangGraph workflow for the memory agent."""
    workflow = StateGraph(AgentState)

    def update_reasoning(state: AgentState, message: str) -> list[str]:
        trace = list(state.get("reasoning_trace", []))
        trace.append(message)
        return trace

    def analyze_conversation(state: AgentState) -> AgentState:
        logger.debug("Analyzing conversation for channel %s", state.get("channel_id"))
        insights = extract_insights(state["conversation"])
        goal = insights.questions[0] if insights.questions else "Collect relevant context."
        identified = {
            "people_ids": list(insights.people),
            "organizations": [org for org in insights.organizations if org],
            "topics": [topic for topic in insights.topics if topic],
        }
        return {
            "identified_entities": identified,
            "current_goal": goal,
            "retrieved_facts": list(state.get("retrieved_facts", [])),
            "tool_calls": list(state.get("tool_calls", [])),
            "reasoning_trace": update_reasoning(state, f"Set goal: {goal}"),
        }

    def determine_tool_from_goal(state: AgentState) -> dict[str, Any] | None:
        conversation_text = " ".join(msg.content for msg in state["conversation"]).lower()
        identified = state.get("identified_entities", {})
        retrieved = state.get("retrieved_facts", [])
        retrieved_people = {fact.person_id for fact in retrieved}

        # Prioritize explicit person mentions
        for person_id in identified.get("people_ids", []):
            if person_id not in retrieved_people and "get_person_profile" in tools:
                return {"name": "get_person_profile", "input": {"person_id": person_id}}

        # Organization queries
        organizations = identified.get("organizations") or []
        if organizations and "find_people_by_organization" in tools:
            return {
                "name": "find_people_by_organization",
                "input": {"organization": organizations[0]},
            }

        # Skill queries
        skill = extract_skill(conversation_text)
        if skill and "find_people_by_skill" in tools:
            return {"name": "find_people_by_skill", "input": {"skill": skill}}

        # Topic queries
        topic = extract_topic(conversation_text)
        if topic and "find_people_by_topic" in tools:
            return {"name": "find_people_by_topic", "input": {"topic": topic}}

        # Location queries
        location = extract_location(conversation_text)
        if location and "find_people_by_location" in tools:
            return {"name": "find_people_by_location", "input": {"location": location}}

        # Fallback to semantic search
        if "semantic_search_facts" in tools:
            return {
                "name": "semantic_search_facts",
                "input": {
                    "query": conversation_text[-500:],
                    "limit": config.max_facts,
                },
            }
        return None

    async def plan_queries(state: AgentState) -> AgentState:
        logger.debug("Planning next query, iteration %s", state.get("iteration"))
        candidate = determine_tool_from_goal(state)
        if candidate and llm:
            try:
                chosen = await llm.aplan_tool_usage(state.get("current_goal", ""), list(tools.keys()))
                if chosen and chosen in tools and chosen != candidate["name"]:
                    candidate = {"name": chosen, "input": candidate["input"]}
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM planning failed: %s", exc)
        reasoning_message = (
            f"Planned tool {candidate['name']}" if candidate else "No further tool required."
        )
        return {
            "pending_tool": candidate,
            "reasoning_trace": update_reasoning(state, reasoning_message),
        }

    def execute_tool(state: AgentState) -> AgentState:
        pending = state.get("pending_tool")
        if not pending:
            return state
        tool_name = pending["name"]
        tool_input = pending.get("input", {})
        tool = tools.get(tool_name)
        if tool is None:
            logger.warning("Tool %s not available", tool_name)
            return {
                "pending_tool": None,
                "reasoning_trace": update_reasoning(state, f"Tool {tool_name} unavailable."),
            }
        logger.info("Executing tool %s", tool_name)
        result = tool(tool_input)
        facts = normalize_to_facts(tool_name, result)
        retrieved = list(state.get("retrieved_facts", []))
        retrieved.extend(facts)
        tool_calls = list(state.get("tool_calls", []))
        tool_calls.append(
            {
                "name": tool_name,
                "input": tool_input,
                "result_count": len(facts),
            }
        )
        reasoning_msg = f"Tool {tool_name} returned {len(facts)} facts."
        return {
            "retrieved_facts": retrieved,
            "tool_calls": tool_calls,
            "pending_tool": None,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": update_reasoning(state, reasoning_msg),
        }

    def evaluate_progress(state: AgentState) -> AgentState:
        calls = state.get("tool_calls", [])
        goal_accomplished = False
        if calls:
            last = calls[-1]
            goal_accomplished = last.get("result_count", 0) > 0
        reasoning_msg = (
            "Goal satisfied with new facts."
            if goal_accomplished
            else "Goal not yet satisfied."
        )
        return {
            "goal_accomplished": goal_accomplished,
            "reasoning_trace": update_reasoning(state, reasoning_msg),
        }

    def synthesize(state: AgentState) -> AgentState:
        facts = deduplicate_facts(state.get("retrieved_facts", []))
        formatted = format_facts(facts)
        confidence = compute_confidence(facts, state)
        reasoning_msg = f"Synthesized {len(formatted)} facts with confidence {confidence}."
        return {
            "formatted_facts": formatted[: state.get("max_facts", config.max_facts)],
            "confidence": confidence,
            "reasoning_trace": update_reasoning(state, reasoning_msg),
        }

    workflow.add_node("analyze_conversation", analyze_conversation)
    workflow.add_node("plan_queries", plan_queries)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("evaluate_progress", evaluate_progress)
    workflow.add_node("synthesize", synthesize)

    workflow.set_entry_point("analyze_conversation")
    workflow.add_edge("analyze_conversation", "plan_queries")
    workflow.add_conditional_edges("plan_queries", should_continue, {"continue": "execute_tool", "finish": "synthesize"})
    workflow.add_edge("execute_tool", "evaluate_progress")
    workflow.add_conditional_edges(
        "evaluate_progress",
        evaluate_next_step,
        {"continue": "plan_queries", "finish": "synthesize"},
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


def should_continue(state: AgentState) -> str:
    """Determine if the workflow should continue after planning."""
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        return "finish"
    if len(state.get("retrieved_facts", [])) >= state.get("max_facts", 1):
        return "finish"
    if not state.get("pending_tool"):
        return "finish"
    return "continue"


def evaluate_next_step(state: AgentState) -> str:
    """Decide whether to continue iterating after a tool execution."""
    if state.get("iteration", 0) >= state.get("max_iterations", 1):
        return "finish"
    if len(state.get("retrieved_facts", [])) >= state.get("max_facts", 1):
        return "finish"
    if detect_tool_loop(state.get("tool_calls", [])):
        return "finish"
    if not state.get("goal_accomplished"):
        return "continue"
    return "finish"


def detect_tool_loop(tool_calls: Iterable[dict[str, Any]]) -> bool:
    """Detect if the same tool has been called repeatedly without progress."""
    calls = list(tool_calls)
    if len(calls) < 3:
        return False
    recent = calls[-3:]
    names = {call.get("name") for call in recent}
    if len(names) == 1 and all(call.get("result_count", 0) == 0 for call in recent):
        return True
    return False


def compute_confidence(facts: list[RetrievedFact], state: AgentState) -> str:
    """Assess overall confidence."""
    if not facts:
        return "low"
    confidences = [fact.confidence for fact in facts if fact.confidence is not None]
    high_conf = sum(1 for value in confidences if value >= 0.8)
    if high_conf >= 3 or len(facts) >= 5:
        return "high"
    if high_conf >= 1 or len(facts) >= 2:
        return "medium"
    return "low"


def extract_skill(text: str) -> str | None:
    """Very lightweight heuristic to detect skill queries."""
    if "knows" in text:
        fragment = text.split("knows", 1)[1].strip()
        if fragment:
            candidate = fragment.split()[0].strip("?.!,")
            if len(candidate) > 2:
                return candidate
    if "expert in" in text:
        fragment = text.split("expert in", 1)[1].strip()
        candidate = fragment.split()[0].strip("?.!,")
        if len(candidate) > 2:
            return candidate
    return None


def extract_topic(text: str) -> str | None:
    """Detect possible topic interest."""
    markers = ["topic", "talk about", "interested in", "care about"]
    for marker in markers:
        if marker in text:
            return marker
    return None


def extract_location(text: str) -> str | None:
    """Detect possible location target."""
    markers = ["in ", "at ", "from "]
    for marker in markers:
        if marker in text:
            start = text.index(marker) + len(marker)
            token = text[start:].split()[0]
            if token:
                return token.strip("?.!,")
    return None
