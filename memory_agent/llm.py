"""LLM helpers for the memory agent."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from prometheus_client import Counter, Histogram

from ie.client import LlamaServerClient, LlamaServerConfig

from .fact_formatter import format_fact
from .models import MessageModel, RetrievedFact
from .serialization import json_dumps
from .state import AgentState
from .tools import ToolBase


logger = logging.getLogger(__name__)

llm_calls_total = Counter(
    "memory_agent_llm_calls_total",
    "Total LLM calls",
    ["method", "success"],
)

llm_latency_seconds = Histogram(
    "memory_agent_llm_latency_seconds",
    "LLM call latency in seconds",
    ["method"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
)

TOOL_PROMPT_INFO: dict[str, dict[str, str]] = {
    "get_relationships_between": {
        "description": "Show relationships and shared contexts between two people.",
        "use_when": "The goal compares or connects two specific people.",
        "inputs": "person_a_id (required), person_b_id (required)",
        "example": "Use when asked 'How do Alice and Bob know each other?'",
    },
    "find_people_by_topic": {
        "description": "Find people who talk about, care about, or are curious about a topic.",
        "use_when": "The goal is about interest or discussion of a subject.",
        "inputs": "topic (required), relationship_types, limit (optional)",
        "example": "Use when asked 'Who is into climate change research?'",
    },
    "get_person_timeline": {
        "description": "Return temporal facts about a person such as jobs, education, or events.",
        "use_when": "The goal asks for someone's history or timeline.",
        "inputs": "person_id (required), fact_types, start_date, end_date (optional)",
        "example": "Use when asked 'What's Bob's work history?'",
    },
    "find_people_by_location": {
        "description": "Find people associated with a location (lives, works, studied, visited).",
        "use_when": "The goal references a geographic location.",
        "inputs": "location (required), min_confidence, limit (optional)",
        "example": "Use when asked 'Who is based in New York?'",
    },
    "semantic_search_facts": {
        "description": "Perform semantic search across all fact embeddings for flexible matching.",
        "use_when": "Other tools are not specific enough or previous targeted queries returned nothing.",
        "inputs": "query (required), fact_types, limit, similarity_threshold (optional)",
        "example": "Use when asked nuanced questions like 'Who has experience with leadership in startups?'",
    },
}


class PromptCache:
    """Basic FIFO prompt cache for reusable sections."""

    def __init__(self, max_size: int = 100) -> None:
        self._cache: dict[str, str] = {}
        self._max_size = max_size

    def get_or_generate(self, key: str, generator: Callable[[], str]) -> str:
        if key not in self._cache:
            if len(self._cache) >= self._max_size:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = generator()
        return self._cache[key]


@dataclass(slots=True)
class LLMClient:
    """Client that routes planning prompts to llama-server with structured prompting."""

    model: str
    temperature: float
    api_key: str | None = None
    base_url: str | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    timeout: float | None = None
    prompt_cache: PromptCache = field(default_factory=PromptCache)

    _llama_client: LlamaServerClient | None = None

    def __post_init__(self) -> None:
        self._initialize_client()

    def _initialize_client(self) -> None:
        try:
            config = LlamaServerConfig(
                base_url=self.base_url or "http://localhost:8080/v1/chat/completions",
                model=self.model,
                timeout=self.timeout or 1200.0,
                temperature=self.temperature,
                top_p=self.top_p or 0.95,
                max_tokens=int(self.max_tokens or 4096),
                api_key=self.api_key,
            )
            self._llama_client = LlamaServerClient(config)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize LLM client: %s", exc)
            self._llama_client = None

    async def apredict(self, prompt: str) -> str:
        """Asynchronously generate a response to the prompt."""
        if self._llama_client is not None:
            return await asyncio.to_thread(self._llama_predict, prompt)
        return self._fallback(prompt)

    def predict(self, prompt: str) -> str:
        """Synchronously generate a response."""
        if self._llama_client is not None:
            return self._llama_predict(prompt)
        return self._fallback(prompt)

    def _llama_predict(self, prompt: str) -> str:
        if self._llama_client is None:
            return self._fallback(prompt)
        messages = [
            {
                "role": "system",
                "content": (
                    "You help coordinate which knowledge graph tool should run next. "
                    "Return only the tool name that best advances the stated goal."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._llama_client.complete(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Llama call failed, falling back: %s", exc)
            return self._fallback(prompt)
        if response is None:
            return self._fallback(prompt)
        return response

    def _fallback(self, prompt: str) -> str:
        """Fallback heuristic when no LLM provider is configured."""
        truncated = (prompt[:200] + "...") if len(prompt) > 200 else prompt
        return (
            "LLM unavailable. "
            "Consider exploring skills, organizations, and relationships related to this request. "
            f"Prompt excerpt: {truncated}"
        )

    @property
    def is_available(self) -> bool:
        """Return True if an underlying chat model is ready."""
        return self._llama_client is not None

    async def aplan_tool_usage(
        self,
        goal: str,
        available_tools: dict[str, ToolBase],
        state: AgentState,
        heuristic_candidate: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Enhanced tool selection that leverages llama-server output."""

        if not self.is_available:
            logger.debug("LLM unavailable, using heuristic candidate")
            return self._candidate_to_result(
                heuristic_candidate,
                reason="LLM unavailable",
                confidence="medium",
            )

        prompt = self._build_tool_selection_prompt(goal, available_tools, state)
        start = time.perf_counter()
        raw_response = ""
        try:
            raw_response = await asyncio.to_thread(self._llama_predict, prompt)
            latency = time.perf_counter() - start
            llm_calls_total.labels("tool_selection", "true").inc()
            llm_latency_seconds.labels("tool_selection").observe(latency)
            self._log_llm_interaction(
                method="tool_selection",
                prompt_summary=prompt[:500],
                response_summary=raw_response[:500],
                success=True,
                latency_ms=latency * 1000,
            )

            response_clean = self._normalize_json_response(raw_response)
            result = json.loads(response_clean)
            is_valid, error = self._validate_llm_response(
                result,
                required_fields=["tool_name", "reasoning", "confidence", "should_stop"],
                valid_values={"confidence": ["high", "medium", "low"]},
            )
            if not is_valid:
                raise ValueError(error)

            tool_name = result.get("tool_name")
            if tool_name and tool_name not in available_tools:
                raise ValueError(f"LLM proposed unknown tool: {tool_name}")

            parameters = result.get("parameters") or {}
            if not isinstance(parameters, dict):
                parameters = {}
            result["parameters"] = parameters
            return result
        except Exception as exc:  # noqa: BLE001
            latency = time.perf_counter() - start
            llm_calls_total.labels("tool_selection", "false").inc()
            llm_latency_seconds.labels("tool_selection").observe(latency)
            self._log_llm_interaction(
                method="tool_selection",
                prompt_summary=prompt[:500],
                response_summary=raw_response[:500],
                success=False,
                latency_ms=latency * 1000,
            )
            logger.warning("Tool planning failed: %s", exc)
            return self._fallback_tool_selection(available_tools, state, heuristic_candidate)

    async def extract_entities_from_conversation(self, messages: list[MessageModel]) -> dict[str, Any]:
        if not self.is_available:
            return {
                "people": [],
                "organizations": [],
                "topics": [],
                "locations": [],
                "skills": [],
                "implicit_references": [],
            }

        conversation_text = "\n".join(
            f"{message.author_name}: {message.content}" for message in messages[-5:]
        )
        prompt = (
            "Analyze the following Discord conversation and extract mentioned entities.\n\n"
            "## Conversation\n"
            f"{conversation_text or 'No recent messages.'}\n\n"
            "## Output Format\n"
            "{\n"
            '  "people": [],\n'
            '  "organizations": [],\n'
            '  "topics": [],\n'
            '  "locations": [],\n'
            '  "skills": [],\n'
            '  "implicit_references": []\n'
            "}\n\nReturn JSON only (no markdown)."
        )

        response = await asyncio.to_thread(self._llama_predict, prompt)
        response_clean = self._normalize_json_response(response)
        try:
            return json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("LLM entity extraction failed", exc_info=True)
            return {
                "people": [],
                "organizations": [],
                "topics": [],
                "locations": [],
                "skills": [],
                "implicit_references": [],
            }

    async def assess_fact_confidence(
        self,
        fact: RetrievedFact,
        conversation_context: str,
    ) -> dict[str, Any]:
        if not self.is_available:
            return {
                "relevance_score": 0.5,
                "relevance_explanation": "LLM unavailable",
                "reliability_score": fact.confidence or 0.5,
                "reliability_explanation": "Defaulted to fact confidence",
                "should_include": True,
                "caveats": [],
            }

        prompt = (
            "Evaluate the relevance and reliability of this fact for answering the user's query.\n\n"
            "## User Context\n"
            f"{conversation_context}\n\n"
            "## Retrieved Fact\n"
            f"- Type: {fact.fact_type}\n"
            f"- Person: {fact.person_name} ({fact.person_id})\n"
            f"- Object: {fact.fact_object}\n"
            f"- Attributes: {json_dumps(fact.attributes, indent=2)}\n"
            f"- Confidence Score: {fact.confidence}\n"
            f"- Evidence Count: {len(fact.evidence)}\n"
            f"- Timestamp: {fact.timestamp}\n\n"
            "## Output Format\n"
            "{\n"
            '  "relevance_score": 0.0,\n'
            '  "relevance_explanation": "",\n'
            '  "reliability_score": 0.0,\n'
            '  "reliability_explanation": "",\n'
            '  "should_include": true,\n'
            '  "caveats": []\n'
            "}\n\nRespond with JSON only."
        )

        response = await asyncio.to_thread(self._llama_predict, prompt)
        response_clean = self._normalize_json_response(response)
        try:
            return json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("Fact confidence assessment failed", exc_info=True)
            return {
                "relevance_score": 0.5,
                "relevance_explanation": "Unable to assess",
                "reliability_score": fact.confidence or 0.5,
                "reliability_explanation": "Defaulted to fact confidence",
                "should_include": True,
                "caveats": [],
            }

    async def should_continue_searching(
        self,
        goal: str,
        retrieved_facts: list[RetrievedFact],
        tool_calls: list[dict[str, Any]],
        max_iterations: int,
        current_iteration: int,
    ) -> dict[str, Any]:
        if not self.is_available:
            should_continue = current_iteration < max_iterations
            return {
                "should_continue": should_continue,
                "confidence": "low",
                "reasoning": "LLM unavailable; defaulting based on iteration limit.",
                "recommendations": [],
            }

        facts_summary = self._summarize_retrieved_facts(retrieved_facts)
        tool_history = self._format_tool_history(tool_calls)
        prompt = (
            "Determine whether the agent should continue searching for more information or stop.\n\n"
            f"## Goal\n{goal or 'Goal unspecified.'}\n\n"
            f"## Iteration\n{current_iteration}/{max_iterations}\n\n"
            f"## Retrieved Facts\n{facts_summary}\n\n"
            f"## Tool History\n{tool_history}\n\n"
            "## Output Format\n"
            "{\n"
            '  "should_continue": true,\n'
            '  "confidence": "high",\n'
            '  "reasoning": "",\n'
            '  "recommendations": []\n'
            "}\n\nRespond with JSON only."
        )

        response = await asyncio.to_thread(self._llama_predict, prompt)
        response_clean = self._normalize_json_response(response)
        try:
            return json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("Stopping evaluation failed", exc_info=True)
            should_continue = current_iteration < max_iterations
            return {
                "should_continue": should_continue,
                "confidence": "low",
                "reasoning": f"Fallback decision to {'continue' if should_continue else 'stop'}.",
                "recommendations": [],
            }

    async def refine_tool_parameters(
        self,
        tool_name: str,
        initial_parameters: dict[str, Any],
        conversation_context: str,
        previous_results: Sequence[Any],
    ) -> dict[str, Any]:
        if not self.is_available:
            return {
                "refined_parameters": initial_parameters,
                "changes_made": [],
                "reasoning": "LLM unavailable; using original parameters",
            }

        previous_repr = (
            "No previous results (first attempt)"
            if not previous_results
            else json_dumps(previous_results, indent=2)[:800]
        )
        prompt = (
            "Refine the parameters for a knowledge graph tool based on context and previous results.\n\n"
            f"## Tool\n{tool_name}\n\n"
            f"## Initial Parameters\n{json_dumps(initial_parameters, indent=2)}\n\n"
            f"## Conversation Context\n{conversation_context}\n\n"
            f"## Previous Results\n{previous_repr}\n\n"
            "## Output Format\n"
            "{\n"
            '  "refined_parameters": {},\n'
            '  "changes_made": [],\n'
            '  "reasoning": ""\n'
            "}\n\nRespond with JSON only."
        )

        response = await asyncio.to_thread(self._llama_predict, prompt)
        response_clean = self._normalize_json_response(response)
        try:
            return json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("Parameter refinement failed", exc_info=True)
            return {
                "refined_parameters": initial_parameters,
                "changes_made": [],
                "reasoning": "Refinement failed; using original parameters",
            }

    async def batch_assess_facts(
        self,
        facts: list[RetrievedFact],
        context: str,
        max_concurrent: int = 3,
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max_concurrent)

        async def assess_with_limit(fact: RetrievedFact) -> dict[str, Any]:
            async with semaphore:
                return await self.assess_fact_confidence(fact, context)

        return await asyncio.gather(*[assess_with_limit(fact) for fact in facts])

    def _normalize_json_response(self, response: str) -> str:
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _validate_llm_response(
        self,
        response: dict[str, Any],
        required_fields: Sequence[str],
        valid_values: dict[str, Sequence[Any]] | None = None,
    ) -> tuple[bool, str]:
        valid_values = valid_values or {}
        for field in required_fields:
            if field not in response:
                return False, f"Missing required field: {field}"
        for field, allowed in valid_values.items():
            if field in response and response[field] not in allowed:
                return False, f"Invalid value for {field}: {response[field]}"
        return True, ""

    def _build_tool_selection_prompt(
        self,
        goal: str,
        available_tools: dict[str, ToolBase],
        state: AgentState,
    ) -> str:
        conversation_summary = self._summarize_conversation(state.get("conversation", []))
        retrieved_facts_summary = self._summarize_retrieved_facts(state.get("retrieved_facts", []))
        tool_history = self._format_tool_history(state.get("tool_calls", []))
        entities = state.get("identified_entities", {})
        reasoning_trace = state.get("reasoning_trace", [])
        recent_reasoning = reasoning_trace[-5:]
        catalog_key = "tool_catalog::" + ",".join(sorted(available_tools.keys()))
        catalog = self.prompt_cache.get_or_generate(
            catalog_key,
            lambda: self._build_tool_catalog(available_tools),
        )
        goal_text = goal or self._goal_text(state, goal)
        prompt = (
            "You are an intelligent planner deciding which knowledge graph tool to call next.\n\n"
            "## Goal\n"
            f"{goal_text or 'Goal unclear.'}\n\n"
            "## Recent Conversation\n"
            f"{conversation_summary}\n\n"
            "## Identified Entities\n"
            f"- People: {', '.join(entities.get('people_ids', [])) or 'None'}\n"
            f"- Organizations: {', '.join(entities.get('organizations', [])) or 'None'}\n"
            f"- Topics: {', '.join(entities.get('topics', [])) or 'None'}\n\n"
            "## Retrieved Facts\n"
            f"{retrieved_facts_summary}\n\n"
            "## Tool Call History\n"
            f"{tool_history}\n\n"
            "## Available Tools\n"
            f"{catalog}\n\n"
            "## Recent Reasoning Steps\n"
            f"{chr(10).join('- ' + step for step in recent_reasoning) if recent_reasoning else 'No previous reasoning steps.'}\n\n"
            "Before you answer, think about what information is still missing, which tools have already been tried, and whether stopping is appropriate.\n\n"
            "## CRITICAL OUTPUT FORMAT\n"
            "Return ONLY a JSON object (no markdown) with this schema:\n"
            "{\n"
            '  "tool_name": "exact_tool_name",\n'
            '  "reasoning": "2-3 sentences explaining the decision",\n'
            '  "confidence": "high|medium|low",\n'
            '  "should_stop": false,\n'
            '  "stop_reason": null,\n'
            '  "parameters": {}\n'
            "}\n\n"
            "If you decide no tool should run, set should_stop to true and explain why in stop_reason.\n\n"
            "### Examples\n"
            "1. Goal: 'Goal already answered' -> should_stop true\n"
            "2. Goal: 'Who is based in New York?' -> find_people_by_location\n\n"
            "Respond now with JSON only."
        )
        return prompt

    def _goal_text(self, state: AgentState, goal: str) -> str:
        if goal:
            return goal
        messages = state.get("conversation", [])
        if messages:
            return messages[-1].content
        return ""

    def _build_tool_catalog(self, available_tools: dict[str, ToolBase]) -> str:
        entries: list[str] = []
        for name in available_tools:
            info = TOOL_PROMPT_INFO.get(
                name,
                {
                    "description": "No description available.",
                    "use_when": "Use when appropriate.",
                    "inputs": "",
                    "example": "",
                },
            )
            entry = (
                f"### {name}\n"
                f"- Description: {info['description']}\n"
                f"- Use When: {info['use_when']}\n"
                f"- Inputs: {info['inputs']}\n"
                f"- Example: {info['example']}\n"
            )
            entries.append(entry)
        return "\n".join(entries)

    def _summarize_conversation(self, messages: Sequence[MessageModel]) -> str:
        if not messages:
            return "No conversation context available."
        lines = []
        for message in messages[-5:]:
            content = message.content.strip()
            if len(content) > 160:
                content = content[:157] + "..."
            lines.append(f"- {message.author_name}: {content}")
        return "\n".join(lines)

    def _summarize_retrieved_facts(self, facts: Sequence[RetrievedFact]) -> str:
        if not facts:
            return "No facts retrieved yet."
        summaries = []
        for fact in facts[-5:]:
            summaries.append(f"- {format_fact(fact)}")
        return "\n".join(summaries)

    def _format_tool_history(self, tool_calls: Sequence[dict[str, Any]]) -> str:
        if not tool_calls:
            return "No tools have been called yet."
        lines = []
        for index, call in enumerate(tool_calls[-5:], start=max(0, len(tool_calls) - 5)):
            lines.append(
                f"- #{index + 1}: {call.get('name')} input={call.get('input')} -> {call.get('result_count', 0)} result(s)"
            )
        return "\n".join(lines)

    def _candidate_to_result(
        self,
        candidate: dict[str, Any] | None,
        *,
        reason: str,
        confidence: str,
    ) -> dict[str, Any]:
        if not candidate:
            return {
                "tool_name": None,
                "parameters": {},
                "reasoning": reason,
                "confidence": confidence,
                "should_stop": True,
                "stop_reason": reason,
            }
        return {
            "tool_name": candidate.get("name"),
            "parameters": candidate.get("input", {}),
            "reasoning": reason,
            "confidence": confidence,
            "should_stop": False,
            "stop_reason": None,
        }

    def _fallback_tool_selection(
        self,
        available_tools: dict[str, ToolBase],
        state: AgentState,
        heuristic_candidate: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if heuristic_candidate:
            return self._candidate_to_result(
                heuristic_candidate,
                reason="Fallback to heuristic selection",
                confidence="medium",
            )

        goal_text = self._goal_text(state, state.get("current_goal", ""))
        conversation_text = " ".join(msg.content for msg in state.get("conversation", []))

        if "semantic_search_facts" in available_tools:
            return {
                "tool_name": "semantic_search_facts",
                "parameters": {"query": goal_text or conversation_text, "limit": state.get("max_facts", 10)},
                "reasoning": "Fallback heuristic defaulted to semantic search.",
                "confidence": "low",
                "should_stop": False,
                "stop_reason": None,
            }

        return {
            "tool_name": None,
            "parameters": {},
            "reasoning": "No suitable fallback tool available.",
            "confidence": "low",
            "should_stop": True,
            "stop_reason": "No tools match the query.",
        }

    def _log_llm_interaction(
        self,
        *,
        method: str,
        prompt_summary: str,
        response_summary: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        logger.info(
            "LLM interaction",
            extra={
                "method": method,
                "prompt_length": len(prompt_summary),
                "response_length": len(response_summary),
                "success": success,
                "latency_ms": latency_ms,
                "model": self.model,
            },
        )
