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
    "semantic_search_facts": {
        "description": "Perform semantic search using multiple diverse queries (12-15) to find relevant facts across all types. Uses Reciprocal Rank Fusion (RRF) to intelligently combine results.",
        "use_when": "The goal requires broad discovery across fact types, or when searching for concepts using varied keywords and phrases. RRF will boost facts that appear in multiple query results.",
        "inputs": "queries (required, list of 12-20 diverse keywords/phrases of varying lengths)",
        "example": "Use when asked 'Who has startup experience?' - generate diverse queries: ['startup', 'founder', 'entrepreneur', 'early-stage company', 'venture-backed startup experience', 'building companies from scratch', 'startup leadership roles', 'validating scalable business models', 'minimal viable product (MVP) development', 'strategies for sustainable growth and customer acquisition']. Include concise keywords, medium phrases, and fuller descriptive sentences. Try not to search names more than once.",
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

    async def agenerate_text(self, prompt: str, system_message: str | None = None) -> str:
        """Generate text using the LLM without tool-specific formatting.
        
        This method is suitable for general text generation tasks like summarization,
        unlike apredict/predict which are optimized for tool selection.
        
        Args:
            prompt: The user prompt
            system_message: Optional custom system message. If None, uses a general assistant message.
            
        Returns:
            Generated text response
        """
        if self._llama_client is None:
            logger.debug("LLM client unavailable for text generation")
            return ""
        
        default_system = "You are a helpful AI assistant."
        messages = [
            {"role": "system", "content": system_message or default_system},
            {"role": "user", "content": prompt},
        ]
        
        try:
            response = self._llama_client.complete(messages, json_mode=False)
            return response if response else ""
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM text generation failed: %s", exc)
            return ""

    def generate_text(self, prompt: str, system_message: str | None = None) -> str:
        """Synchronous version of agenerate_text."""
        import asyncio
        return asyncio.run(self.agenerate_text(prompt, system_message))

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

    async def extract_goal_from_conversation(self, messages: list[MessageModel]) -> str:
        """Determine the user's information retrieval goal from the conversation.

        Analyzes the conversation (focusing on the last message) to determine what
        information the user is seeking. Always returns a retrieval goal, even if
        the message doesn't have a direct information need - in that case, generates
        a goal about gathering contextual information.

        Args:
            messages: List of conversation messages

        Returns:
            A clear goal string (e.g., "Find people with Kubernetes experience" or
            "Gather contextual information about recent conversation participants")
        """
        if not self.is_available or not messages:
            return "Collect relevant context about conversation participants and topics."

        # Separate last message from conversation history
        last_message = messages[-1]
        history_messages = messages[-6:-1] if len(messages) > 1 else []

        last_message_text = f"{last_message.author_name}: {last_message.content}"
        history_text = "\n".join(
            f"{msg.author_name}: {msg.content}" for msg in history_messages
        ) if history_messages else "(No previous messages)"

        prompt = (
            "Analyze a Discord conversation to determine what information should be retrieved.\n"
            "Focus EXCLUSIVELY on the LAST MESSAGE to understand what the user wants NOW.\n"
            "Use conversation history ONLY for context to understand references.\n\n"

            "## Previous Conversation (context only - for understanding references):\n"
            f"```\n{history_text}\n```\n\n"

            "## LAST MESSAGE (PRIMARY FOCUS - what does THIS message need?):\n"
            f"```\n{last_message_text}\n```\n\n"

            "## Your Task:\n"
            "Always generate a retrieval goal. Even if the message doesn't explicitly seek information,\n"
            "create a goal about gathering contextual information that could enrich the conversation.\n\n"

            "## Direct Information-Seeking (generate specific goals):\n"
            "- Questions about people: 'Who knows X?', 'Does anyone have Y experience?'\n"
            "- Requests for information: 'Tell me about...', 'I need to find...', 'Looking for...'\n"
            "- Implicit needs: 'We need someone with X skills', 'Anyone familiar with Y?'\n"
            "- Follow-ups referencing prior context: 'What about them?', 'Tell me more'\n\n"

            "## Indirect/Contextual Messages (generate context-gathering goals):\n"
            "- Casual chat: 'Hey', 'Thanks!', 'Sounds good' → Gather context about the person and recent topics\n"
            "- Rhetorical questions: 'Is it Friday yet?' → Gather context about work patterns and schedules\n"
            "- Statements of fact: 'I finished the project' → Gather context about the person's projects and skills\n"
            "- Commands/actions: 'Please update the docs' → Gather context about documentation contributors\n\n"

            "## Output Rules:\n"
            "ALWAYS write a clear retrieval goal (10-30 words):\n"
            "- For direct information needs: Focus on WHAT specific information to retrieve\n"
            "- For indirect messages: Focus on gathering contextual information about participants, topics, or activities\n"
            "- Use action verbs: 'Find', 'Identify', 'Locate', 'Gather', 'Retrieve', 'Collect'\n"
            "- Be specific about what's being sought (skills, people, experiences, context, etc.)\n\n"

            "## Examples:\n"
            "Last: 'Who knows Rust?'\n"
            "Goal: 'Find people with Rust programming experience'\n\n"

            "Last: 'Does anyone have experience with Kubernetes in production?'\n"
            "Goal: 'Identify people with production Kubernetes deployment experience'\n\n"

            "Last: 'What does he think about this?'\n"
            "Goal: 'Find out what [resolve person from history] thinks about [resolve topic from history]'\n\n"

            "Last: 'Tell me about the AI team members'\n"
            "Goal: 'Find people who work on or are associated with AI projects'\n\n"

            "Last: 'What about that person we discussed?'\n"
            "Goal: 'Retrieve information about [resolve reference from history]'\n\n"

            "Last: 'Thanks for the help!'\n"
            "Goal: 'Gather contextual information about recent conversation participants and their expertise'\n\n"

            "Last: 'Is it Friday yet?'\n"
            "Goal: 'Collect context about work schedules and weekly patterns discussed by participants'\n\n"

            "Last: 'I just deployed the new feature'\n"
            "Goal: 'Gather information about the person's deployment experience and related technical skills'\n\n"

            "Last: 'What are your favorite quotes by [person]?'\n"
            "Goal: 'Find notable quotes by [person]'\n\n"

            "Return ONLY the goal text (no JSON, no markdown, just the plain goal statement)."
        )

        try:
            response = await asyncio.to_thread(self._llama_predict, prompt)
            goal = response.strip()

            # Validate we got a reasonable response
            if not goal or len(goal) > 200:
                logger.warning("Invalid goal extraction response, using fallback")
                return "Collect relevant context."

            logger.info("Extracted goal: %s", goal)
            return goal

        except Exception:  # noqa: BLE001
            logger.warning("LLM goal extraction failed", exc_info=True)
            return "Collect relevant context."

    async def extract_message_search_queries(
            self,
            messages: list[MessageModel],
            *,
            max_queries: int = 15,
    ) -> list[str]:
        """Derive diverse search phrases for message retrieval from recent conversation."""

        if not self.is_available or not messages:
            return []

        capped = max(1, min(max_queries, 15))

        # Separate last message from conversation history
        last_message = messages[-1]
        history_messages = messages[-6:-1] if len(messages) > 1 else []

        last_message_text = f"{last_message.author_name}: {last_message.content}"
        history_text = "\n".join(
            f"{msg.author_name}: {msg.content}" for msg in history_messages
        ) if history_messages else "(No previous messages)"

        prompt = (
            "You assist with retrieving relevant historical Discord messages via semantic search.\n"
            "Your task: Generate diverse search queries based EXCLUSIVELY on the LAST MESSAGE below.\n"
            "Use the conversation history ONLY for context to understand references, but ALL queries must address the last message.\n\n"

            "## Previous Conversation (context only - for understanding references):\n"
            f"```\n{history_text}\n```\n\n"

            "## LAST MESSAGE (PRIMARY FOCUS - generate queries for THIS only):\n"
            f"```\n{last_message_text}\n```\n\n"

            "## Your Task:\n"
            "Generate search queries to find historical messages relevant to the LAST MESSAGE.\n"
            "Focus on distinct perspectives: core topic, sub-topics, follow-up actions, stakeholders, artifacts, locations, timelines, and synonyms.\n"
            "Produce a mix of lengths: include some concise 1-3 word keywords, some medium phrases (4-8 words), and several fuller descriptive sentences up to 20 words.\n"
            "Avoid filler like 'search for' and keep every query grounded in the LAST MESSAGE's context.\n\n"

            "Return JSON only in this format:\n"
            "{\n"
            '  "queries": ["keyword or phrase", "..."]\n'
            "}\n\n"
            f"Include between 12 and {capped} entries whenever possible; if the last message is sparse, return as many high-quality queries as you can."
        )

        try:
            response = await asyncio.to_thread(self._llama_predict, prompt)
            response_clean = self._normalize_json_response(response)
            parsed = json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("LLM message query extraction failed", exc_info=True)
            return []

        candidates = parsed.get("queries") if isinstance(parsed, dict) else None
        if not isinstance(candidates, list):
            return []

        cleaned: list[str] = []
        for raw in candidates:
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue
            if text.lower().startswith("query:"):
                text = text.split(":", 1)[1].strip()
            if text and text not in cleaned:
                cleaned.append(text[:120])
            if len(cleaned) >= capped:
                break

        return cleaned

    async def extract_fact_search_queries(
            self,
            messages: list[MessageModel],
            *,
            max_queries: int = 15,
            previous_queries: list[str] | None = None,
            retrieved_facts: list[RetrievedFact] | None = None,
    ) -> list[str]:
        """Derive diverse search queries for fact retrieval from recent conversation.

        Args:
            messages: Conversation messages to extract queries from.
            max_queries: Maximum number of queries to generate.
            previous_queries: Queries already tried in previous iterations (for deduplication).
            retrieved_facts: Facts already retrieved (to search for complementary info).
        """

        if not self.is_available or not messages:
            return []

        capped = max(1, min(max_queries, 20))

        # Separate last message from conversation history
        last_message = messages[-1]
        history_messages = messages[-6:-1] if len(messages) > 1 else []

        last_message_text = f"{last_message.author_name}: {last_message.content}"
        history_text = "\n".join(
            f"{msg.author_name}: {msg.content}" for msg in history_messages
        ) if history_messages else "(No previous messages)"

        # Build context about previous attempts for iterative refinement
        tried_context = ""
        if previous_queries:
            recent_tried = previous_queries[-15:]  # Show last 15 queries
            tried_context = (
                "## Previously Tried Queries (generate DIFFERENT queries, not these):\n"
                + "\n".join(f'- "{q}"' for q in recent_tried)
                + "\n\n"
            )

        found_context = ""
        if retrieved_facts:
            recent_facts = retrieved_facts[-10:]  # Show last 10 facts
            found_context = (
                "## Already Retrieved Facts (search for COMPLEMENTARY information):\n"
                + "\n".join(f"- {format_fact(f)}" for f in recent_facts)
                + "\n\n"
            )

        prompt = (
            "You assist with retrieving relevant facts about people via semantic search.\n"
            "Your task: Generate diverse search queries based EXCLUSIVELY on the LAST MESSAGE below.\n"
            "Use the conversation history ONLY for context to understand references, but ALL queries must address the last message.\n\n"

            "## Previous Conversation (context only - for understanding references):\n"
            f"```\n{history_text}\n```\n\n"

            "## LAST MESSAGE (PRIMARY FOCUS - generate queries for THIS only):\n"
            f"```\n{last_message_text}\n```\n\n"

            f"{tried_context}"
            f"{found_context}"

            "## Your Task:\n"
            "Generate 15-20 search queries that find facts relevant to answering or addressing the LAST MESSAGE.\n"
            "Facts include: employment, skills, education, relationships, interests, preferences, locations, and experiences.\n\n"

            "## CRITICAL: Generate queries from MULTIPLE ANGLES:\n"
            "1. **Direct keywords** from the LAST MESSAGE\n"
            "2. **Synonyms and related terms** (e.g., 'startup' → 'founder', 'entrepreneur', 'early-stage company')\n"
            "3. **Broader concepts** (e.g., 'React' → 'frontend development', 'JavaScript frameworks')\n"
            "4. **Narrower specifics** (e.g., 'AI' → 'machine learning', 'neural networks', 'LLMs')\n"
            "5. **Related skills/topics** (e.g., 'Python' → 'data science', 'backend development')\n"
            "6. **Job titles and roles** (e.g., 'engineering' → 'software engineer', 'tech lead', 'CTO')\n"
            "7. **Organizations and companies** mentioned or implied in the LAST MESSAGE\n"
            "8. **Technologies and tools** (specific versions, alternatives, competitors)\n"
            "9. **Domain areas** (industries, fields, specializations)\n"
            "10. **People mentioned** in the LAST MESSAGE by name or reference\n\n"

            "## IMPORTANT RULES:\n"
            "- ALL queries must be relevant to the LAST MESSAGE specifically\n"
            "- Generate 15-20 queries to maximize coverage\n"
            "- Mix lengths: 1-3 words (keywords), 4-8 words (phrases), up to 20 words (descriptive sentences)\n"
            "- Include both specific and general terms from the LAST MESSAGE\n"
            "- Think about what facts MIGHT be relevant to the LAST MESSAGE, not just obvious connections\n"
            "- Avoid duplicate or near-duplicate queries\n"
            "- Don't include 'search for' or other meta-language\n"
            "- If the last message references something from earlier (e.g., 'that project', 'the company we discussed'), use the history to understand WHAT is being referenced, then query for that thing\n"
            "- DO NOT repeat or rephrase any previously tried queries listed above\n"
            "- Search for DIFFERENT angles than what's already been found\n\n"

            "Return JSON only in this format:\n"
            "{\n"
            '  "queries": ["keyword or phrase", "..."]\n'
            "}\n\n"
            f"Generate {capped} diverse queries based on the LAST MESSAGE:"
        )

        try:
            response = await asyncio.to_thread(self._llama_predict, prompt)
            response_clean = self._normalize_json_response(response)
            parsed = json.loads(response_clean)
        except Exception:  # noqa: BLE001
            logger.warning("LLM fact query extraction failed", exc_info=True)
            return []

        candidates = parsed.get("queries") if isinstance(parsed, dict) else None
        if not isinstance(candidates, list):
            return []

        cleaned: list[str] = []
        for raw in candidates:
            if not isinstance(raw, str):
                continue
            text = raw.strip()
            if not text:
                continue
            if text.lower().startswith("query:"):
                text = text.split(":", 1)[1].strip()
            if text and text not in cleaned:
                cleaned.append(text[:120])
            if len(cleaned) >= capped:
                break

        return cleaned

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
            f"## Retrieved Facts ({len(retrieved_facts)} total)\n{facts_summary}\n\n"
            f"## Tool History\n{tool_history}\n\n"

            "## Decision Criteria\n"
            "**Continue searching IFF:**\n"
            "- Fewer than 5 iterations completed (need more exploration)\n"
            "- Recent tool calls are still finding new facts\n"
            "- The goal asks about multiple aspects and we've only covered some\n"
            "- Facts seem incomplete or partial\n"
            "- Different search terms might reveal more relevant information\n\n"

            "**Stop searching if:**\n"
            "- We've tried 10+ iterations with diminishing returns\n"
            "- Last 3 tool calls returned 0 results\n"
            "- Retrieved facts comprehensively address all aspects of the goal\n"
            "- We've exhausted different search strategies\n\n"

            "## Output Format\n"
            "{\n"
            '  "should_continue": true,\n'
            '  "confidence": "high",\n'
            '  "reasoning": "Explain why we should continue/stop",\n'
            '  "recommendations": ["suggest next search strategies if continuing"]\n'
            "}\n\n"

            "Be CONSERVATIVE about stopping - err on the side of searching more.\n"
            "Respond with JSON only."
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
        retrieved_facts: list[RetrievedFact] | None = None,
    ) -> dict[str, Any]:
        """Refine tool parameters based on previous results and retrieved facts.

        Args:
            tool_name: Name of the tool to refine parameters for.
            initial_parameters: Current parameters to refine.
            conversation_context: Recent conversation for context.
            previous_results: Previous tool call results (metadata).
            retrieved_facts: Facts already retrieved (to search for complementary info).
        """
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

        # Add context about what was found to inform refinement
        facts_context = ""
        if retrieved_facts:
            recent_facts = retrieved_facts[-10:]
            facts_context = (
                "## What We've Found So Far:\n"
                + "\n".join(f"- {format_fact(f)}" for f in recent_facts)
                + "\n\n"
                "Consider: What's MISSING? What related topics haven't been explored?\n"
                "Suggest parameters that find COMPLEMENTARY information, not duplicates.\n\n"
            )

        prompt = (
            "Refine the parameters for a knowledge graph tool based on context and previous results.\n\n"
            f"## Tool\n{tool_name}\n\n"
            f"## Initial Parameters\n{json_dumps(initial_parameters, indent=2)}\n\n"
            f"## Conversation Context\n{conversation_context}\n\n"
            f"## Previous Results\n{previous_repr}\n\n"
            f"{facts_context}"
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
        latest_message = self._latest_message(state.get("conversation", []))
        retrieved_facts_summary = self._summarize_retrieved_facts(state.get("retrieved_facts", []))
        tool_history = self._format_tool_history(state.get("tool_calls", []))
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
            "## Latest Message (primary intent)\n"
            f"{latest_message}\n\n"
            "## Recent Conversation (newest last)\n"
            f"{conversation_summary}\n\n"
            "Always prioritize the most recent message when determining the next action, using earlier turns only for additional context.\n\n"
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
            "2. Goal: 'Who has experience with startups?' -> semantic_search_facts\n\n"
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

    def _latest_message(self, messages: Sequence[MessageModel]) -> str:
        if not messages:
            return "No conversation context available."
        message = messages[-1]
        content = message.content.strip()
        if len(content) > 200:
            content = content[:197] + "..."
        return f"{message.author_name}: {content}"

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
            # Use goal text as a single query in fallback mode
            # Note: This fallback is only used when LLM planning fails entirely
            # Normal operation will use extract_fact_search_queries for diverse queries
            queries = [goal_text] if goal_text else [conversation_text]
            # Over-fetch for LLM quality filtering (3x multiplier, capped at tool maximum)
            retrieval_limit = min(state.get("max_facts", 10) * 3, 50)
            return {
                "tool_name": "semantic_search_facts",
                "parameters": {
                    "queries": queries,
                    "limit": retrieval_limit,
                },
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
