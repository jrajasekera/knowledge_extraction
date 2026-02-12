"""LangGraph agent orchestration for memory retrieval."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from typing import Any

from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from .config import AgentConfig
from .conversation import extract_insights
from .fact_formatter import format_facts
from .llm import LLMClient
from .models import RetrievalRequest, RetrievedFact
from .normalization import normalize_to_facts
from .query_fallback import expand_fallback_queries
from .serialization import to_serializable
from .state import AgentState
from .tools import ToolBase
from .tools.base import ToolError

logger = logging.getLogger(__name__)


def fact_key(fact: RetrievedFact) -> tuple[str, str, str | None, str | None]:
    """Canonical identity of a fact for novelty detection.

    Matches the dedup key used in fact_formatter.deduplicate_facts and
    SemanticSearchFacts._search_neo4j: (person_id, fact_type, fact_object, relationship_type).
    """
    relationship_type = None
    if isinstance(fact.attributes, dict) and fact.attributes.get("relationship_type"):
        relationship_type = str(fact.attributes["relationship_type"])
    return (fact.person_id, fact.fact_type, fact.fact_object, relationship_type)


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# Tools eligible for the iterative retrieval loop.
# semantic_search_messages is excluded because synthesize already handles
# message retrieval with proper query generation and channel scoping.
_LOOP_TOOL_NAMES: frozenset[str] = frozenset({"semantic_search_facts"})


def _confidence_meets_threshold(actual: str, required: str) -> bool:
    """Check if actual confidence level meets or exceeds the required threshold."""
    return _CONFIDENCE_RANK.get(actual, 0) >= _CONFIDENCE_RANK.get(required, 2)


def compute_novelty(
    new_facts: list[RetrievedFact],
    seen_fact_keys_serialized: list[list[str | None]],
    novelty_min_new_facts: int,
    prev_streak: int,
) -> tuple[int, int, list[list[str | None]]]:
    """Compute novelty metrics for a set of new facts.

    Returns (new_count, updated_streak, updated_seen_keys_serialized).
    Deduplicates both against history AND within the current batch.
    """
    seen: set[tuple[str | None, ...]] = {tuple(k) for k in seen_fact_keys_serialized}
    new_keys: list[list[str | None]] = []
    new_count = 0

    for f in new_facts:
        fk = fact_key(f)
        if fk not in seen:
            seen.add(fk)
            new_keys.append(list(fk))
            new_count += 1

    updated_streak = prev_streak + 1 if new_count < novelty_min_new_facts else 0

    updated_seen = list(seen_fact_keys_serialized)
    updated_seen.extend(new_keys)

    return new_count, updated_streak, updated_seen


class MemoryAgent:
    """Coordinates LangGraph workflow execution."""

    def __init__(
        self,
        tools: dict[str, ToolBase],
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
            "tool_max_retries": self.config.tool_max_retries,
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
            "tried_queries": [],
            "query_results_map": {},
            "new_facts_last_iteration": 0,
            "novelty_streak_without_gain": 0,
            "seen_fact_keys": [],
            "early_stop_min_iterations": self.config.early_stop_min_iterations,
            "novelty_min_new_facts": self.config.novelty_min_new_facts,
            "novelty_patience": self.config.novelty_patience,
            "stop_confidence_required": self.config.stop_confidence_required,
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
            "new_facts_last_iteration": final_state.get("new_facts_last_iteration", 0),
            "novelty_streak_without_gain": final_state.get("novelty_streak_without_gain", 0),
            "unique_facts_seen": len(final_state.get("seen_fact_keys", [])),
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
                "retrieved_facts": [
                    fact.model_dump() for fact in final_state.get("retrieved_facts", [])
                ],
                "retrieved_messages": [
                    msg.model_dump() for msg in final_state.get("retrieved_messages", [])
                ],
            }
        return result


def create_memory_agent_graph(
    tools: dict[str, ToolBase],
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
        logger.info("Analyzing conversation for channel %s", state.get("channel_id"))

        # Use LLM to extract goal from conversation (focusing on last message)
        conversation = state.get("conversation", [])
        if llm and llm.is_available:
            goal = await llm.extract_goal_from_conversation(conversation)
        else:
            # Fallback to simple question detection if LLM unavailable
            insights = extract_insights(conversation)
            goal = insights.questions[-1] if insights.questions else "Collect relevant context."

        logger.info("Initial goal: %s", goal)
        trace = update_reasoning(state, f"Set goal: {goal}")
        return {
            "current_goal": goal,
            "retrieved_facts": list(state.get("retrieved_facts", [])),
            "tool_calls": list(state.get("tool_calls", [])),
            "reasoning_trace": trace,
        }

    def determine_tool_from_goal(
        state: AgentState, preferred_tool: str | None = None
    ) -> dict[str, Any] | None:
        conversation = state.get("conversation", [])
        conversation_text = " ".join(msg.content for msg in conversation).lower()

        def _expand_queries() -> list[str]:
            """Use deterministic fallback expansion on original-case messages."""
            last_msg = conversation[-1].content if conversation else ""
            recent = [m.content for m in conversation[-5:]] if len(conversation) > 1 else None
            goal_text = state.get("current_goal") or ""
            prev = list(state.get("tried_queries", []))
            results = expand_fallback_queries(
                last_message=last_msg,
                recent_messages=recent,
                goal=goal_text,
                previous_queries=prev,
                max_queries=config.fallback_max_queries,
                max_query_length=config.fallback_max_query_length,
            )
            return [fq.text for fq in results]

        def build_semantic_search() -> dict[str, Any] | None:
            retrieval_limit = state.get("max_facts", config.max_facts)
            expanded = _expand_queries()
            if expanded:
                return {"queries": expanded, "limit": retrieval_limit}
            if conversation_text:
                return {"queries": [conversation_text[-500:]], "limit": retrieval_limit}
            return None

        heuristics: list[tuple[str, Callable[[], dict[str, Any] | None]]] = [
            ("semantic_search_facts", build_semantic_search),
        ]

        def fallback_payload(tool_name: str) -> dict[str, Any] | None:
            if tool_name != "semantic_search_facts":
                return None
            retrieval_limit = state.get("max_facts", config.max_facts)
            expanded = _expand_queries()
            if expanded:
                return {"queries": expanded, "limit": retrieval_limit}
            goal_text = state.get("current_goal") or conversation_text
            if not goal_text:
                return None
            return {"queries": [goal_text], "limit": retrieval_limit}

        if preferred_tool:
            for name, builder in heuristics:
                if name == preferred_tool and name in tools and name in _LOOP_TOOL_NAMES:
                    payload = builder()
                    if not payload:
                        payload = fallback_payload(name)
                        if payload:
                            logger.info(
                                "Using fallback payload for LLM-selected tool %s: %s", name, payload
                            )
                    if payload:
                        return {"name": name, "input": payload}
                    logger.info("No payload available for LLM-selected tool %s", name)
                    return None
            return None

        for name, builder in heuristics:
            if name not in tools:
                continue
            payload = builder()
            if payload:
                logger.info("Heuristic selected tool %s with payload %s", name, payload)
                return {"name": name, "input": payload}
        return None

    def should_refine_search(
        tool_calls: list[dict[str, Any]],
        tool_name: str,
        retrieved_facts: list[RetrievedFact],
    ) -> bool:
        """Determine if search needs refinement based on quality signals.

        Triggers refinement when:
        1. Previous call returned zero results
        2. Diminishing returns (70%+ drop in results)
        3. Low confidence facts (avg confidence < 0.4)
        """
        relevant_calls = [c for c in tool_calls if c.get("name") == tool_name]

        if not relevant_calls:
            return False

        last_call = relevant_calls[-1]

        # 1. Zero results - definitely refine
        if last_call.get("result_count", 0) == 0:
            return True

        # 2. Diminishing returns - refine if getting significantly fewer results
        if len(relevant_calls) >= 2:
            prev_count = relevant_calls[-2].get("result_count", 0)
            curr_count = last_call.get("result_count", 0)
            if prev_count > 0 and curr_count < prev_count * 0.3:  # 70% drop
                return True

        # 3. Low confidence facts - refine if recent facts are low quality
        if retrieved_facts:
            result_count = last_call.get("result_count", 0)
            if result_count > 0:
                recent_facts = retrieved_facts[-result_count:]
                confidences = [f.confidence for f in recent_facts if f.confidence is not None]
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    if avg_confidence < 0.4:  # Low average confidence
                        return True

        return False

    async def plan_queries(state: AgentState) -> AgentState:
        logger.info("Planning next query, iteration %s", state.get("iteration"))
        llm_reasoning_updates = list(state.get("llm_reasoning", []))
        candidate = determine_tool_from_goal(state)
        if llm and llm.is_available:
            try:
                goal = state.get("current_goal") or ""
                loop_tools = {k: v for k, v in tools.items() if k in _LOOP_TOOL_NAMES}
                llm_result = await llm.aplan_tool_usage(goal, loop_tools, state, candidate)
                llm_reasoning_updates.append(
                    {
                        "iteration": state.get("iteration", 0),
                        "decision": llm_result,
                        "timestamp": time.time(),
                    }
                )

                if llm_result.get("should_stop"):
                    stop_reason = llm_result.get("stop_reason") or llm_result.get("reasoning")
                    iteration = state.get("iteration", 0)
                    min_iterations = state.get("early_stop_min_iterations", 2)
                    llm_confidence = llm_result.get("confidence", "low")
                    required_confidence = state.get("stop_confidence_required", "high")

                    if iteration >= min_iterations and _confidence_meets_threshold(
                        llm_confidence, required_confidence
                    ):
                        logger.info(
                            "LLM recommended stopping: %s (confidence=%s)",
                            stop_reason,
                            llm_confidence,
                        )
                        return {
                            "pending_tool": None,
                            "goal_accomplished": True,
                            "llm_reasoning": llm_reasoning_updates,
                            "tool_selection_confidence": llm_confidence,
                            "reasoning_trace": update_reasoning(
                                state,
                                f"LLM suggests stopping: {stop_reason}",
                            ),
                        }
                    else:
                        logger.info(
                            "LLM recommended stopping (confidence=%s) at iteration %d "
                            "but floor=%d / required_confidence=%s not met; continuing",
                            llm_confidence,
                            iteration,
                            min_iterations,
                            required_confidence,
                        )

                tool_name = llm_result.get("tool_name")
                if tool_name and tool_name in tools and tool_name in _LOOP_TOOL_NAMES:
                    parameters = llm_result.get("parameters") or {}
                    if not parameters:
                        alternate = determine_tool_from_goal(state, preferred_tool=tool_name)
                        if alternate:
                            parameters = alternate.get("input", {})

                    # Quality-based refinement trigger
                    should_refine = should_refine_search(
                        state.get("tool_calls", []),
                        tool_name,
                        state.get("retrieved_facts", []),
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
                                [
                                    call
                                    for call in state.get("tool_calls", [])
                                    if call.get("name") == tool_name
                                ],
                                retrieved_facts=state.get("retrieved_facts", []),
                            )
                            refined_parameters = refinement.get("refined_parameters")
                            if isinstance(refined_parameters, dict) and refined_parameters:
                                parameters.update(refined_parameters)
                        except Exception as exc:  # noqa: BLE001
                            logger.info("Tool parameter refinement failed: %s", exc)

                    # Ensure semantic_search_facts uses max_facts limit and extract diverse queries
                    if tool_name == "semantic_search_facts" and isinstance(parameters, dict):
                        if "limit" not in parameters:
                            parameters["limit"] = state.get("max_facts", config.max_facts)

                        # Collect previous queries for context-aware generation
                        previous_queries: list[str] = []
                        for call in state.get("tool_calls", []):
                            if call.get("name") == "semantic_search_facts":
                                queries = call.get("input", {}).get("queries", [])
                                if queries:
                                    previous_queries.extend(queries)
                        # Also include queries from the dedicated state field
                        previous_queries.extend(state.get("tried_queries", []))

                        # Extract diverse fact search queries using LLM with context
                        conversation = state.get("conversation", [])
                        if llm and conversation:
                            try:
                                extracted_queries = await llm.extract_fact_search_queries(
                                    conversation,
                                    max_queries=15,
                                    previous_queries=previous_queries if previous_queries else None,
                                    retrieved_facts=state.get("retrieved_facts", []) or None,
                                )
                                if extracted_queries:
                                    parameters["queries"] = extracted_queries
                                    logger.info(
                                        "LLM produced %d fact search queries",
                                        len(extracted_queries),
                                    )
                            except Exception as exc:  # noqa: BLE001
                                logger.info("Fact query extraction failed: %s", exc)

                        # Fallback if no queries were extracted or LLM unavailable
                        if "queries" not in parameters or not parameters.get("queries"):
                            conversation_msgs = state.get("conversation", [])
                            last_msg = conversation_msgs[-1].content if conversation_msgs else ""
                            recent = (
                                [m.content for m in conversation_msgs[-5:]]
                                if len(conversation_msgs) > 1
                                else None
                            )
                            goal_text = state.get("current_goal") or ""

                            fallback_results = expand_fallback_queries(
                                last_message=last_msg,
                                recent_messages=recent,
                                goal=goal_text,
                                previous_queries=previous_queries,
                                max_queries=config.fallback_max_queries,
                                max_query_length=config.fallback_max_query_length,
                            )
                            if fallback_results:
                                parameters["queries"] = [fq.text for fq in fallback_results]
                                logger.info(
                                    "Fallback query expansion produced %d queries (sources: %s)",
                                    len(fallback_results),
                                    ", ".join(sorted({fq.source for fq in fallback_results})),
                                )
                            elif goal_text:
                                parameters["queries"] = [goal_text]
                                logger.info("Falling back to goal text for fact search")

                        if "adaptive_threshold" not in parameters:
                            parameters["adaptive_threshold"] = True

                        if (
                            parameters.get("adaptive_threshold")
                            and "similarity_threshold" in parameters
                        ):
                            parameters.pop("similarity_threshold", None)

                    # Validate required fields before dispatch
                    queries_value = parameters.get("queries")
                    has_valid_queries = isinstance(queries_value, list) and len(queries_value) > 0
                    if tool_name.startswith("semantic_search") and not has_valid_queries:
                        logger.warning(
                            "Tool %s selected but missing required 'queries'; skipping",
                            tool_name,
                        )
                        candidate = None
                    else:
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
                logger.info("LLM planning failed: %s", exc)

        reasoning_message = (
            f"Planned tool {candidate['name']}" if candidate else "No further tool required."
        )
        return {
            "pending_tool": candidate,
            "llm_reasoning": llm_reasoning_updates,
            "reasoning_trace": update_reasoning(state, reasoning_message),
            "tool_selection_confidence": "low"
            if candidate
            else state.get("tool_selection_confidence", "low"),
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
        max_retries = state.get("tool_max_retries", 0)
        attempts = 0
        last_error: str | None = None
        result = None
        duration_ms = 0
        logger.info(
            "Executing tool %s with input %s (max_retries=%d)",
            tool_name,
            tool_input,
            max_retries,
        )

        for attempt in range(1, max_retries + 2):
            attempts = attempt
            start = time.perf_counter()
            try:
                result = tool(tool_input)
                duration_ms = int((time.perf_counter() - start) * 1000)
                break
            except (ToolError, ValidationError, Exception) as exc:  # noqa: BLE001
                last_error = str(exc)
                duration_ms = int((time.perf_counter() - start) * 1000)
                retryable = not isinstance(exc, ValidationError) and attempt <= max_retries
                logger.warning(
                    "Tool %s failed on attempt %d/%d: %s",
                    tool_name,
                    attempt,
                    max_retries + 1,
                    exc,
                    exc_info=not isinstance(exc, ToolError),
                )
                if not retryable:
                    break

        if result is None:
            tool_calls = list(state.get("tool_calls", []))
            tool_calls.append(
                {
                    "name": tool_name,
                    "input": to_serializable(tool_input),
                    "result_count": 0,
                    "error": last_error,
                    "success": False,
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                }
            )
            reasoning_msg = (
                f"Tool {tool_name} failed after {attempts} attempt(s); continuing with fallback."
            )
            # Tool failure = 0 new facts, increment novelty streak
            prev_streak = state.get("novelty_streak_without_gain", 0)
            min_new = state.get("novelty_min_new_facts", 1)
            failure_streak = prev_streak + 1 if min_new > 0 else 0
            return {
                "tool_calls": tool_calls,
                "pending_tool": None,
                "iteration": state.get("iteration", 0) + 1,
                "reasoning_trace": update_reasoning(state, reasoning_msg),
                "new_facts_last_iteration": 0,
                "novelty_streak_without_gain": failure_streak,
            }

        facts = normalize_to_facts(tool_name, result)
        logger.info("Tool %s returned %d facts on attempt %d", tool_name, len(facts), attempts)

        meaningful_facts = [fact for fact in facts if fact.fact_type != "CONVERSATION_MENTION"]

        retrieved = list(state.get("retrieved_facts", []))
        retrieved.extend(meaningful_facts)
        tool_calls = list(state.get("tool_calls", []))
        tool_calls.append(
            {
                "name": tool_name,
                "input": to_serializable(tool_input),
                "result_count": len(meaningful_facts),
                "success": len(meaningful_facts) > 0,
                "duration_ms": duration_ms,
                "attempts": attempts,
                "timestamp": time.time(),
            }
        )

        # Track queries used in this execution for iterative refinement
        queries_used = tool_input.get("queries", [])
        tried_queries = list(state.get("tried_queries", []))
        tried_queries.extend(queries_used)

        # Track query -> result mapping for learning
        query_results_map = dict(state.get("query_results_map", {}))
        result_count = len(meaningful_facts)
        for q in queries_used:
            # Store best result count for each query
            query_results_map[q] = max(query_results_map.get(q, 0), result_count)

        # Compute novelty using extracted helper
        new_count, novelty_streak, updated_seen = compute_novelty(
            new_facts=meaningful_facts,
            seen_fact_keys_serialized=state.get("seen_fact_keys", []),
            novelty_min_new_facts=state.get("novelty_min_new_facts", 1),
            prev_streak=state.get("novelty_streak_without_gain", 0),
        )

        reasoning_msg = (
            f"Tool {tool_name} returned {len(meaningful_facts)} actionable facts."
            if facts
            else f"Tool {tool_name} returned 0 facts."
        )
        return {
            "retrieved_facts": retrieved,
            "tool_calls": tool_calls,
            "pending_tool": None,
            "iteration": state.get("iteration", 0) + 1,
            "reasoning_trace": update_reasoning(state, reasoning_msg),
            "tried_queries": tried_queries,
            "query_results_map": query_results_map,
            "new_facts_last_iteration": new_count,
            "novelty_streak_without_gain": novelty_streak,
            "seen_fact_keys": updated_seen,
        }

    async def evaluate_progress(state: AgentState) -> AgentState:
        calls = state.get("tool_calls", [])
        iteration = state.get("iteration", 0)
        min_iterations = state.get("early_stop_min_iterations", 2)
        patience = state.get("novelty_patience", 2)
        streak = state.get("novelty_streak_without_gain", 0)
        new_facts = state.get("new_facts_last_iteration", 0)
        required_confidence = state.get("stop_confidence_required", "high")

        goal_accomplished = False

        # Enforce minimum iteration floor: never stop early before this
        below_floor = iteration < min_iterations

        # Check novelty-based stopping: only if above floor
        if not below_floor and streak >= patience:
            goal_accomplished = True

        # Consult LLM for stop decision
        stop_decision: dict[str, Any] = state.get("should_stop_evaluation", {})
        if llm and llm.is_available:
            try:
                stop_decision = await llm.should_continue_searching(
                    state.get("current_goal") or "",
                    state.get("retrieved_facts", []),
                    calls,
                    state.get("max_iterations", 1),
                    iteration,
                    new_facts_last_iteration=new_facts,
                    novelty_streak=streak,
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("LLM stop evaluation failed: %s", exc)

        # LLM stop decision: only honor if above floor AND confidence meets threshold
        if not below_floor and stop_decision and stop_decision.get("should_continue") is False:
            llm_confidence = stop_decision.get("confidence", "low")
            if _confidence_meets_threshold(llm_confidence, required_confidence):
                goal_accomplished = True
            else:
                logger.info(
                    "LLM advised stopping with confidence %s, but %s required; continuing",
                    llm_confidence,
                    required_confidence,
                )

        reasoning_msg = (
            "Goal satisfied with new facts." if goal_accomplished else "Goal not yet satisfied."
        )
        trace = update_reasoning(state, reasoning_msg)
        if stop_decision and stop_decision.get("should_continue") is False:
            trace = update_reasoning(
                {"reasoning_trace": trace},
                f"LLM advised stopping: {stop_decision.get('reasoning', 'no reasoning provided')}",
            )
        logger.info(
            "Goal accomplished: %s after tool call %s (iteration=%d, new_facts=%d, streak=%d)",
            goal_accomplished,
            calls[-1] if calls else None,
            iteration,
            new_facts,
            streak,
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
                        queries = await llm.extract_message_search_queries(
                            conversation, max_queries=15
                        )
                        if queries:
                            logger.info("LLM produced %d message search queries", len(queries))
                    except Exception as exc:  # noqa: BLE001
                        logger.info("Message query extraction failed: %s", exc)

                if not queries and conversation_text:
                    fallback = conversation_text[-500:]
                    if fallback.strip():
                        queries = [fallback]
                        logger.info("Falling back to raw conversation text for message search")

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
                            "Retrieved %d messages from semantic_search_messages",
                            len(formatted_messages),
                        )
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Message retrieval failed: %s", exc, exc_info=True)
                else:
                    logger.info("No suitable queries available for message search")
            else:
                logger.warning("semantic_search_messages tool not available")
        else:
            logger.info("max_messages is 0, skipping message retrieval")

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
    workflow.add_conditional_edges(
        "plan_queries", should_continue, {"continue": "execute_tool", "finish": "synthesize"}
    )
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
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "finish"
    # Only enforce the max_facts hard cap once the iteration floor is met.
    # Before the floor, let the agent keep iterating so novelty tracking,
    # LLM stop evaluation, and diverse search strategies get a chance to run.
    early_stop_floor = state.get("early_stop_min_iterations", 2)
    if state.get("iteration", 0) >= early_stop_floor and len(
        state.get("retrieved_facts", [])
    ) >= state.get("max_facts", 30):
        return "finish"
    # LLM/metric stop decision: only honor if evaluate_progress agreed (goal_accomplished)
    # AND we're above the iteration floor
    stop_decision = state.get("should_stop_evaluation", {})
    if (
        stop_decision
        and stop_decision.get("should_continue") is False
        and state.get("goal_accomplished")
        and state.get("iteration", 0) >= early_stop_floor
    ):
        return "finish"
    if not state.get("pending_tool"):
        return "finish"
    return "continue"


def evaluate_next_step(state: AgentState) -> str:
    """Decide whether to continue iterating after a tool execution."""
    if state.get("iteration", 0) >= state.get("max_iterations", 10):
        return "finish"
    # Only enforce the max_facts hard cap once the iteration floor is met.
    early_stop_floor = state.get("early_stop_min_iterations", 2)
    if state.get("iteration", 0) >= early_stop_floor and len(
        state.get("retrieved_facts", [])
    ) >= state.get("max_facts", 30):
        return "finish"
    if detect_tool_loop(state.get("tool_calls", [])):
        return "finish"
    # stop_decision is now correctly gated in evaluate_progress;
    # only honor here if evaluate_progress also set goal_accomplished
    stop_decision = state.get("should_stop_evaluation", {})
    if (
        stop_decision
        and stop_decision.get("should_continue") is False
        and state.get("goal_accomplished")
    ):
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
    return len(names) == 1 and all(call.get("result_count", 0) == 0 for call in recent)


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
