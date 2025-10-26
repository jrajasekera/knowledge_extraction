"""LangGraph agent orchestration for memory retrieval."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Iterable

from langgraph.graph import StateGraph, END

from .config import AgentConfig
from .conversation import extract_insights
from .fact_formatter import format_facts
from .llm import LLMClient
from .models import RetrievalRequest, RetrievedFact
from .normalization import normalize_to_facts
from .serialization import to_serializable
from .state import AgentState
from .tools import ToolBase
from .tools.base import ToolError


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
        max_messages = request.max_messages or 10
        max_iterations = request.max_iterations or self.config.max_iterations

        initial_state: AgentState = {
            "conversation": request.messages,
            "channel_id": request.channel_id,
            "max_facts": max_facts,
            "max_messages": max_messages,
            "max_iterations": max_iterations,
            "messages": [],
            "retrieved_facts": [],
            "retrieved_messages": [],
            "formatted_messages": [],
            "context_summary": None,
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
            "llm_reasoning": [],
            "tool_selection_confidence": "low",
            "entity_extraction_results": {},
            "should_stop_evaluation": {},
        }
        recursion_limit = _estimate_recursion_limit(max_iterations)
        final_state: AgentState = await self.graph.ainvoke(
            initial_state,
            config={"recursion_limit": recursion_limit},
        )
        processing_time_ms = int((time.perf_counter() - start) * 1000)

        metadata = {
            "queries_executed": len(final_state.get("tool_calls", [])),
            "facts_retrieved": len(final_state.get("retrieved_facts", [])),
            "messages_retrieved": len(final_state.get("formatted_messages", [])),
            "processing_time_ms": processing_time_ms,
            "iterations_used": final_state.get("iteration", 0),
            "tool_calls": final_state.get("tool_calls", []),
        }

        result: dict[str, Any] = {
            "facts": final_state.get("formatted_facts", []),
            "messages": final_state.get("formatted_messages", []),
            "context_summary": final_state.get("context_summary", ""),
            "confidence": final_state.get("confidence", "low"),
            "metadata": metadata,
        }

        if debug_mode:
            result["debug"] = {
                "reasoning_trace": final_state.get("reasoning_trace", []),
                "retrieved_facts": [fact.model_dump() for fact in final_state.get("retrieved_facts", [])],
                "retrieved_messages": [msg.model_dump() for msg in final_state.get("retrieved_messages", [])],
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

    async def analyze_conversation(state: AgentState) -> AgentState:
        logger.debug("Analyzing conversation for channel %s", state.get("channel_id"))
        insights = extract_insights(state["conversation"])
        goal = insights.questions[0] if insights.questions else "Collect relevant context."
        llm_entities: dict[str, Any] = {}
        if llm and llm.is_available:
            try:
                llm_entities = await llm.extract_entities_from_conversation(state["conversation"])
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM entity extraction failed: %s", exc)

        people_ids = set(insights.people)
        organizations = set(filter(None, insights.organizations))
        topics = set(filter(None, insights.topics))

        if llm_entities:
            organizations.update(
                entity for entity in llm_entities.get("organizations", []) if isinstance(entity, str) and entity
            )
            topics.update(
                entity for entity in llm_entities.get("topics", []) if isinstance(entity, str) and entity
            )

        identified = {
            "people_ids": list(people_ids),
            "organizations": [org for org in organizations if org],
            "topics": [topic for topic in topics if topic],
        }
        if llm_entities.get("people"):
            identified["people_mentions"] = llm_entities["people"]
        if llm_entities.get("locations"):
            identified["locations"] = llm_entities["locations"]

        logger.debug("Identified entities: %s", identified)
        logger.debug("Initial goal: %s", goal)
        trace = update_reasoning(state, f"Set goal: {goal}")
        if llm_entities:
            trace = update_reasoning({"reasoning_trace": trace}, "Captured entities via LLM analysis")
        return {
            "identified_entities": identified,
            "current_goal": goal,
            "retrieved_facts": list(state.get("retrieved_facts", [])),
            "tool_calls": list(state.get("tool_calls", [])),
            "reasoning_trace": trace,
            "entity_extraction_results": llm_entities,
        }

    def determine_tool_from_goal(state: AgentState, preferred_tool: str | None = None) -> dict[str, Any] | None:
        conversation_text = " ".join(msg.content for msg in state["conversation"]).lower()
        identified = state.get("identified_entities", {})
        retrieved = state.get("retrieved_facts", [])
        retrieved_people = {fact.person_id for fact in retrieved}

        def build_topic_query() -> dict[str, Any] | None:
            topics = identified.get("topics") or []
            if topics:
                return {"topic": topics[0]}
            topic = extract_topic(conversation_text)
            if topic:
                return {"topic": topic}
            return None

        def build_semantic_search() -> dict[str, Any] | None:
            if conversation_text:
                retrieval_limit = state.get("max_facts", config.max_facts)
                return {"queries": [conversation_text[-500:]], "limit": retrieval_limit}
            return None

        def build_person_profile_query() -> dict[str, Any] | None:
            people_ids = identified.get("people_ids") or []
            if people_ids:
                return {"person_id": people_ids[0]}
            mentions = identified.get("people_mentions") or []
            if mentions:
                first = mentions[0]
                if isinstance(first, dict) and first.get("id"):
                    return {"person_id": first["id"]}
            return None

        heuristics: list[tuple[str, Callable[[], dict[str, Any] | None]]] = [
            ("get_person_profile", build_person_profile_query),
            ("find_people_by_topic", build_topic_query),
            ("semantic_search_facts", build_semantic_search),
        ]

        def fallback_payload(tool_name: str) -> dict[str, Any] | None:
            goal_text = state.get("current_goal") or conversation_text
            if not goal_text:
                return None
            if tool_name == "find_people_by_topic":
                return {"topic": goal_text}
            if tool_name == "get_person_profile":
                people_ids = identified.get("people_ids") or []
                if people_ids:
                    return {"person_id": people_ids[0]}
                mentions = identified.get("people_mentions") or []
                if mentions:
                    first = mentions[0]
                    if isinstance(first, dict) and first.get("id"):
                        return {"person_id": first["id"]}
                return None
            if tool_name == "semantic_search_facts":
                retrieval_limit = state.get("max_facts", config.max_facts)
                return {"queries": [goal_text], "limit": retrieval_limit}
            return None

        if preferred_tool:
            for name, builder in heuristics:
                if name == preferred_tool and name in tools:
                    payload = builder()
                    if not payload:
                        payload = fallback_payload(name)
                        if payload:
                            logger.debug("Using fallback payload for LLM-selected tool %s: %s", name, payload)
                    if payload:
                        return {"name": name, "input": payload}
                    logger.debug("No payload available for LLM-selected tool %s", name)
                    return None
            return None

        for name, builder in heuristics:
            if name not in tools:
                continue
            payload = builder()
            if payload:
                logger.debug("Heuristic selected tool %s with payload %s", name, payload)
                return {"name": name, "input": payload}
        return None

    async def plan_queries(state: AgentState) -> AgentState:
        logger.debug("Planning next query, iteration %s", state.get("iteration"))
        llm_reasoning_updates = list(state.get("llm_reasoning", []))
        candidate = determine_tool_from_goal(state)
        if llm and llm.is_available:
            try:
                goal = state.get("current_goal", "")
                llm_result = await llm.aplan_tool_usage(goal, tools, state, candidate)
                llm_reasoning_updates.append(
                    {
                        "iteration": state.get("iteration", 0),
                        "decision": llm_result,
                        "timestamp": time.time(),
                    }
                )

                if llm_result.get("should_stop"):
                    stop_reason = llm_result.get("stop_reason") or llm_result.get("reasoning")
                    logger.debug("LLM recommended stopping: %s", stop_reason)
                    return {
                        "pending_tool": None,
                        "goal_accomplished": True,
                        "llm_reasoning": llm_reasoning_updates,
                        "tool_selection_confidence": llm_result.get("confidence", "low"),
                        "reasoning_trace": update_reasoning(
                            state,
                            f"LLM suggests stopping: {stop_reason}",
                        ),
                    }

                tool_name = llm_result.get("tool_name")
                if tool_name and tool_name in tools:
                    parameters = llm_result.get("parameters") or {}
                    if not parameters:
                        alternate = determine_tool_from_goal(state, preferred_tool=tool_name)
                        if alternate:
                            parameters = alternate.get("input", {})

                    should_refine = any(
                        call.get("name") == tool_name and call.get("result_count", 0) == 0
                        for call in state.get("tool_calls", [])
                    )
                    if should_refine:
                        conversation_context = "\n".join(
                            f"{message.author_name}: {message.content}"
                            for message in state.get("conversation", [])[-5:]
                        )
                        try:
                            refinement = await llm.refine_tool_parameters(
                                tool_name,
                                parameters,
                                conversation_context,
                                [call for call in state.get("tool_calls", []) if call.get("name") == tool_name],
                            )
                            refined_parameters = refinement.get("refined_parameters")
                            if isinstance(refined_parameters, dict) and refined_parameters:
                                parameters.update(refined_parameters)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("Tool parameter refinement failed: %s", exc)

                    # Ensure semantic_search_facts uses max_facts limit
                    if tool_name == "semantic_search_facts" and isinstance(parameters, dict):
                        if "limit" not in parameters:
                            parameters["limit"] = state.get("max_facts", config.max_facts)

                    candidate = {"name": tool_name, "input": parameters}
                    reasoning_msg = (
                        f"LLM selected {tool_name} (confidence: {llm_result.get('confidence', 'low')}): "
                        f"{llm_result.get('reasoning', '')}"
                    )
                    return {
                        "pending_tool": candidate,
                        "tool_selection_confidence": llm_result.get("confidence", "low"),
                        "llm_reasoning": llm_reasoning_updates,
                        "reasoning_trace": update_reasoning(state, reasoning_msg),
                    }

            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM planning failed: %s", exc)

        reasoning_message = (
            f"Planned tool {candidate['name']}" if candidate else "No further tool required."
        )
        return {
            "pending_tool": candidate,
            "llm_reasoning": llm_reasoning_updates,
            "reasoning_trace": update_reasoning(state, reasoning_message),
            "tool_selection_confidence": "low" if candidate else state.get("tool_selection_confidence", "low"),
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
        logger.info("Executing tool %s with input %s", tool_name, tool_input)
        start = time.perf_counter()
        try:
            result = tool(tool_input)
        except ToolError as exc:
            logger.warning("Tool %s failed: %s", tool_name, exc)
            tool_calls = list(state.get("tool_calls", []))
            tool_calls.append(
                {
                    "name": tool_name,
                    "input": to_serializable(tool_input),
                    "result_count": 0,
                    "error": str(exc),
                    "success": False,
                    "duration_ms": int((time.perf_counter() - start) * 1000),
                    "timestamp": time.time(),
                }
            )
            reasoning_msg = f"Tool {tool_name} failed; continuing with fallback."
            return {
                "tool_calls": tool_calls,
                "pending_tool": None,
                "iteration": state.get("iteration", 0) + 1,
                "reasoning_trace": update_reasoning(state, reasoning_msg),
            }
        facts = normalize_to_facts(tool_name, result)
        logger.debug("Tool %s returned %d facts", tool_name, len(facts))

        meaningful_facts = [fact for fact in facts if fact.fact_type != "CONVERSATION_MENTION"]

        retrieved = list(state.get("retrieved_facts", []))
        retrieved.extend(meaningful_facts)
        tool_calls = list(state.get("tool_calls", []))
        duration_ms = int((time.perf_counter() - start) * 1000)
        tool_calls.append(
            {
                "name": tool_name,
                "input": to_serializable(tool_input),
                "result_count": len(meaningful_facts),
                "success": len(meaningful_facts) > 0,
                "duration_ms": duration_ms,
                "timestamp": time.time(),
            }
        )
        reasoning_msg = f"Tool {tool_name} returned {len(meaningful_facts)} actionable facts." if facts else f"Tool {tool_name} returned 0 facts."
        return {
            "retrieved_facts": retrieved,
            "tool_calls": tool_calls,
            "pending_tool": None,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": update_reasoning(state, reasoning_msg),
        }

    async def evaluate_progress(state: AgentState) -> AgentState:
        calls = state.get("tool_calls", [])
        goal_accomplished = False
        if calls:
            last = calls[-1]
            goal_accomplished = last.get("result_count", 0) > 0
        stop_decision: dict[str, Any] = state.get("should_stop_evaluation", {})
        if llm and llm.is_available:
            try:
                stop_decision = await llm.should_continue_searching(
                    state.get("current_goal", ""),
                    state.get("retrieved_facts", []),
                    calls,
                    state.get("max_iterations", 1),
                    state.get("iteration", 0),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("LLM stop evaluation failed: %s", exc)
        if stop_decision and stop_decision.get("should_continue") is False:
            goal_accomplished = True
        reasoning_msg = (
            "Goal satisfied with new facts."
            if goal_accomplished
            else "Goal not yet satisfied."
        )
        trace = update_reasoning(state, reasoning_msg)
        if stop_decision and stop_decision.get("should_continue") is False:
            trace = update_reasoning({"reasoning_trace": trace}, f"LLM advised stopping: {stop_decision.get('reasoning', 'no reasoning provided')}")
        logger.debug(
            "Goal accomplished: %s after tool call %s",
            goal_accomplished,
            calls[-1] if calls else None,
        )
        return {
            "goal_accomplished": goal_accomplished,
            "reasoning_trace": trace,
            "should_stop_evaluation": stop_decision,
        }

    async def synthesize(state: AgentState) -> AgentState:
        from .context_summarizer import generate_context_summary
        from .message_formatter import format_messages
        from .tools.semantic_search_messages import SemanticSearchMessagesInput

        # Format facts for output
        facts = state.get("retrieved_facts", [])
        formatted = format_facts(facts)

        # Retrieve and format messages
        formatted_messages: list[str] = []
        max_messages = state.get("max_messages", 10)
        if max_messages > 0:
            tool = tools.get("semantic_search_messages")
            if tool:
                conversation = state.get("conversation", [])
                conversation_text = " ".join([msg.content for msg in conversation[-3:]])
                queries: list[str] = []

                if llm and conversation:
                    try:
                        queries = await llm.extract_message_search_queries(conversation, max_queries=4)
                        if queries:
                            logger.debug("LLM produced %d message search queries", len(queries))
                    except Exception as exc:  # noqa: BLE001
                        logger.debug("Message query extraction failed: %s", exc)

                if not queries and conversation_text:
                    fallback = conversation_text[-500:]
                    if fallback.strip():
                        queries = [fallback]
                        logger.debug("Falling back to raw conversation text for message search")

                if queries:
                    try:
                        channel_id = state.get("channel_id")
                        filters = [channel_id] if channel_id else None
                        logger.info(
                            "Calling semantic_search_messages with %d queries (limit=%d)",
                            len(queries),
                            max_messages,
                        )
                        search_input = SemanticSearchMessagesInput(
                            queries=queries,
                            limit=max_messages,
                            channel_ids=filters,
                        )
                        result = tool(search_input.model_dump())
                        formatted_messages = format_messages(result.results)
                        logger.info(
                            "Retrieved %d messages from semantic_search_messages", len(formatted_messages)
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Message retrieval failed: %s", exc, exc_info=True)
                else:
                    logger.debug("No suitable queries available for message search")
            else:
                logger.warning("semantic_search_messages tool not available")
        else:
            logger.debug("max_messages is 0, skipping message retrieval")

        # Generate context summary using LLM
        context_summary = await generate_context_summary(
            llm=llm,
            conversation=state.get("conversation", []),
            formatted_facts=formatted[: state.get("max_facts", config.max_facts)],
            formatted_messages=formatted_messages[:max_messages],
        )
        logger.info("Generated context summary: %d characters", len(context_summary))

        confidence = compute_confidence(facts, state)
        logger.info(
            "Synthesis produced %d formatted facts and %d messages (confidence=%s)",
            len(formatted),
            len(formatted_messages),
            confidence,
        )
        reasoning_msg = f"Synthesized {len(formatted)} facts and {len(formatted_messages)} messages with confidence {confidence}."
        trace = update_reasoning(state, reasoning_msg)
        return {
            "formatted_facts": formatted[: state.get("max_facts", config.max_facts)],
            "formatted_messages": formatted_messages[:max_messages],
            "context_summary": context_summary,
            "confidence": confidence,
            "reasoning_trace": trace,
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
    stop_decision = state.get("should_stop_evaluation", {})
    if stop_decision and stop_decision.get("should_continue") is False:
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
    stop_decision = state.get("should_stop_evaluation", {})
    if stop_decision and stop_decision.get("should_continue") is False:
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
    """Assess overall confidence based on fact confidence scores and tool success rate."""
    if not facts:
        return "low"
    confidences = [fact.confidence for fact in facts if fact.confidence is not None]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    high_conf = sum(1 for value in confidences if value >= 0.8)

    tool_calls = state.get("tool_calls", []) or []
    success_count = sum(1 for call in tool_calls if call.get("success"))
    success_rate = success_count / len(tool_calls) if tool_calls else 0.0

    if success_rate >= 0.7 and (high_conf >= 3 or len(facts) >= 5):
        return "high"
    if success_rate >= 0.4 and (high_conf >= 1 or len(facts) >= 2 or avg_confidence >= 0.6):
        return "medium"
    return "low"



def extract_topic(text: str) -> str | None:
    """Detect possible topic interest."""
    markers = ["topic", "talk about", "interested in", "care about"]
    for marker in markers:
        if marker in text:
            return marker
    return None


def _estimate_recursion_limit(max_iterations: int) -> int:
    """Compute a LangGraph recursion limit that comfortably covers all steps."""
    per_iteration_steps = 4  # plan, execute, evaluate, follow-up plan
    safety_margin = 10
    limit = max_iterations * per_iteration_steps + safety_margin
    return max(25, limit)
