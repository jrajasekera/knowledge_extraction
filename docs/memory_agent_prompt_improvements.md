# Memory Agent LLM Prompt Improvements

## Executive Summary

The current `memory_agent/llm.py` implementation uses minimal prompting for tool selection, which limits the agent's ability to make intelligent decisions about which knowledge graph tools to invoke. This document provides detailed recommendations for improving prompt construction to enhance the agent's reasoning capabilities, tool selection accuracy, and overall performance.

## Current State Analysis

### Existing Implementation

The `LLMClient` class in `memory_agent/llm.py` currently has one main prompt construction method:

```python
async def aplan_tool_usage(self, goal: str, options: list[str]) -> str:
    prompt = (
        "You are planning which knowledge graph tool to run next. "
        f"Goal: {goal}. Available tools: {', '.join(options)}. "
        "Return only the tool name that best advances the goal."
    )
    response = await self.apredict(prompt)
    for option in options:
        if option in response:
            return option
    return options[0] if options else ""
```

### Critical Limitations

1. **Lack of Context**: The prompt doesn't include conversation content, identified entities, or previously retrieved facts
2. **No Tool Descriptions**: Tool names alone don't convey what each tool does or when to use it
3. **Minimal Reasoning**: No chain-of-thought or explanation of why a tool was chosen
4. **Brittle Parsing**: Simple substring matching can fail or match incorrectly
5. **No Few-Shot Examples**: Missing examples of good tool selection decisions
6. **Missing State Information**: Doesn't share what's already been tried or learned
7. **No Structured Output**: Unstructured text response makes parsing unreliable

---

## Recommended Improvements

### 1. Enhanced Tool Selection Prompt

#### 1.1 Add Comprehensive Tool Catalog

Create a method to generate detailed tool descriptions:

```python
def _build_tool_catalog(self, available_tools: dict[str, ToolBase]) -> str:
    """Generate a detailed catalog of available tools with descriptions."""
    catalog_entries = []
    
    tool_descriptions = {
        "get_person_profile": {
            "description": "Retrieves all known facts about a specific person by their Discord ID",
            "use_when": "User explicitly mentions or asks about a specific person you can identify",
            "inputs": "person_id (required)",
            "example": "When asked 'What do we know about Alice?' or user mentions '@123456'"
        },
        "find_people_by_organization": {
            "description": "Finds people who work or worked at a specific organization",
            "use_when": "User asks about employees/members of a company",
            "inputs": "organization (required), current_only (optional)",
            "example": "When asked 'Who works at Google?' or 'Anyone from Microsoft here?'"
        },
        "find_people_by_topic": {
            "description": "Finds people who discuss, care about, or are curious about a topic",
            "use_when": "User asks who is interested in or talks about something",
            "inputs": "topic (required), relationship_types (optional)",
            "example": "When asked 'Who cares about climate change?' or 'Who talks about crypto?'"
        },
        "find_people_by_location": {
            "description": "Finds people who live in, work in, or are associated with a location",
            "use_when": "User asks about people in a geographic area",
            "inputs": "location (required)",
            "example": "When asked 'Who lives in SF?' or 'Anyone in New York?'"
        },
        "get_person_timeline": {
            "description": "Retrieves temporal/historical facts about a person (jobs, education, events)",
            "use_when": "User asks about someone's history, career progression, or past",
            "inputs": "person_id (required), fact_types (optional), start_date/end_date (optional)",
            "example": "When asked 'What's Alice's work history?' or 'Where did Bob study?'"
        },
        "get_relationships_between": {
            "description": "Finds direct relationships and shared contexts between two people",
            "use_when": "User asks how two people know each other or what they have in common",
            "inputs": "person_a_id (required), person_b_id (required)",
            "example": "When asked 'How do Alice and Bob know each other?'"
        },
        "semantic_search_facts": {
            "description": "Searches for facts semantically similar to a natural language query",
            "use_when": "No specific tool matches or need broad exploratory search",
            "inputs": "query (required), fact_types (optional), limit (optional)",
            "example": "When asked 'What do we know about AI ethics?' or other broad queries"
        }
    }
    
    for tool_name in available_tools:
        if tool_name in tool_descriptions:
            info = tool_descriptions[tool_name]
            entry = f"""
**{tool_name}**
- Description: {info['description']}
- Use When: {info['use_when']}
- Inputs: {info['inputs']}
- Example: {info['example']}
"""
            catalog_entries.append(entry.strip())
    
    return "\n\n".join(catalog_entries)
```

#### 1.2 Include Conversation Context

```python
def _summarize_conversation(self, messages: list[MessageModel]) -> str:
    """Create a concise summary of the conversation for context."""
    if not messages:
        return "No messages provided."
    
    # Take last 3 messages for recency
    recent = messages[-3:]
    summary_parts = []
    
    for msg in recent:
        summary_parts.append(f"- {msg.author_name}: {msg.content[:150]}")
    
    return "\n".join(summary_parts)
```

#### 1.3 Share Retrieved Facts Context

```python
def _summarize_retrieved_facts(self, facts: list[RetrievedFact]) -> str:
    """Summarize what has already been retrieved."""
    if not facts:
        return "No facts retrieved yet."
    
    fact_types = {}
    for fact in facts:
        fact_type = fact.fact_type
        fact_types[fact_type] = fact_types.get(fact_type, 0) + 1
    
    people = set(fact.person_id for fact in facts)
    
    summary = f"Retrieved {len(facts)} facts about {len(people)} people:\n"
    for fact_type, count in sorted(fact_types.items(), key=lambda x: -x[1]):
        summary += f"- {count}x {fact_type}\n"
    
    return summary
```

#### 1.4 Include Tool Call History

```python
def _format_tool_history(self, tool_calls: list[dict]) -> str:
    """Format the history of tool calls to avoid repetition."""
    if not tool_calls:
        return "No tools have been called yet."
    
    history = []
    for call in tool_calls:
        name = call.get("name", "unknown")
        result_count = call.get("result_count", 0)
        history.append(f"- {name}: returned {result_count} results")
    
    return "\n".join(history)
```

#### 1.5 Complete Enhanced Tool Selection Method

```python
async def aplan_tool_usage_enhanced(
    self,
    goal: str,
    available_tools: dict[str, ToolBase],
    state: AgentState,
) -> dict[str, Any]:
    """
    Enhanced tool selection with comprehensive context and structured output.
    
    Returns:
        dict with keys: 'tool_name', 'reasoning', 'confidence', 'parameters'
    """
    
    # Build context sections
    tool_catalog = self._build_tool_catalog(available_tools)
    conversation_summary = self._summarize_conversation(state.get("conversation", []))
    retrieved_facts_summary = self._summarize_retrieved_facts(state.get("retrieved_facts", []))
    tool_history = self._format_tool_history(state.get("tool_calls", []))
    entities = state.get("identified_entities", {})
    reasoning_trace = state.get("reasoning_trace", [])
    
    # Build the prompt
    prompt = f"""You are an intelligent agent planning which knowledge graph tool to use next.

## Your Goal
{goal}

## Recent Conversation Context
{conversation_summary}

## Identified Entities
- People mentioned: {', '.join(entities.get('people_ids', [])) or 'None'}
- Organizations mentioned: {', '.join(entities.get('organizations', [])) or 'None'}
- Topics mentioned: {', '.join(entities.get('topics', [])) or 'None'}

## Previously Retrieved Information
{retrieved_facts_summary}

## Tool Call History
{tool_history}

## Available Tools
{tool_catalog}

## Recent Reasoning Steps
{chr(10).join('- ' + step for step in reasoning_trace[-5:]) if reasoning_trace else 'No previous reasoning steps.'}

## Your Task
Analyze the situation and select the BEST tool to use next. Consider:

1. **Relevance**: Which tool most directly addresses the goal?
2. **Efficiency**: Avoid calling tools that returned 0 results previously with similar inputs
3. **Completeness**: Have we already retrieved sufficient information?
4. **Specificity**: Prefer specific tools (like get_person_profile) over generic ones (like semantic_search) when possible
5. **Progression**: Are we making progress or stuck in a loop?

## Output Format
Respond with a JSON object (and ONLY a JSON object, no markdown formatting):

{{
  "tool_name": "exact_tool_name_from_list_above",
  "reasoning": "2-3 sentence explanation of why this tool is the best choice",
  "confidence": "high|medium|low",
  "should_stop": false,
  "stop_reason": null
}}

If you believe NO tool should be called (goal already met or impossible), set "should_stop" to true and explain why in "stop_reason".

## Examples

### Example 1: Specific Person Query
Goal: "Tell me about Alice's background"
Entities: people_ids=['12345']
Best Choice: get_person_profile (person_id='12345')
Reasoning: "User asks about a specific identified person. get_person_profile retrieves comprehensive facts."

### Example 2: Topic-Based Query  
Goal: "Who cares about climate change?"
Best Choice: find_people_by_topic (topic='climate change')
Reasoning: "The request is about interest in a subject, so find_people_by_topic can surface relevant people."

### Example 3: Avoiding Repetition
Goal: "Find people in Paris"
Tool History: find_people_by_location(Paris) returned 0 results
Best Choice: {{should_stop: true}}
Reasoning: "Location-specific tool already ran with no matches, so stopping avoids redundant calls."

### Example 4: Broad Question
Goal: "Who can help with distributed systems design?"
Best Choice: semantic_search_facts (query='distributed systems design expertise')
Reasoning: "No specific skill or org is mentioned, so semantic_search_facts can surface related experience across fact types."

Now analyze the current situation and select the best tool:"""

    try:
        response = await self.apredict(prompt)
        
        # Try to parse JSON response
        # Strip markdown code blocks if present
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:]
        if response_clean.startswith("```"):
            response_clean = response_clean[3:]
        if response_clean.endswith("```"):
            response_clean = response_clean[:-3]
        response_clean = response_clean.strip()
        
        result = json.loads(response_clean)
        
        # Validate the response
        if "tool_name" not in result and not result.get("should_stop"):
            logger.warning("LLM response missing tool_name and should_stop not set")
            return self._fallback_tool_selection(available_tools, state)
        
        return result
        
    except json.JSONDecodeError as exc:
        logger.warning(f"Failed to parse LLM response as JSON: {exc}")
        logger.debug(f"Raw response: {response}")
        return self._fallback_tool_selection(available_tools, state)
    except Exception as exc:
        logger.warning(f"Tool planning failed: {exc}")
        return self._fallback_tool_selection(available_tools, state)
```

#### 1.6 Fallback Tool Selection

```python
def _fallback_tool_selection(
    self, 
    available_tools: dict[str, ToolBase],
    state: AgentState
) -> dict[str, Any]:
    """
    Heuristic-based fallback when LLM planning fails.
    Uses the existing determine_tool_from_goal logic from agent.py.
    """
    # Import or replicate the heuristic logic from agent.py
    conversation_text = " ".join(
        msg.content for msg in state.get("conversation", [])
    ).lower()
    
    identified = state.get("identified_entities", {})
    
    # Try person profile first if person mentioned
    for person_id in identified.get("people_ids", []):
        if "get_person_profile" in available_tools:
            return {
                "tool_name": "get_person_profile",
                "reasoning": "Fallback heuristic: person ID identified in conversation",
                "confidence": "medium",
                "should_stop": False,
                "parameters": {"person_id": person_id}
            }
    
    # Default to semantic search as last resort
    if "semantic_search_facts" in available_tools:
        return {
            "tool_name": "semantic_search_facts",
            "reasoning": "Fallback heuristic: using semantic search as last resort",
            "confidence": "low",
            "should_stop": False
        }
    
    return {
        "tool_name": None,
        "reasoning": "Fallback: no suitable tool available",
        "confidence": "low",
        "should_stop": True,
        "stop_reason": "No tools match the query"
    }
```

---

### 2. Additional LLM-Assisted Methods

#### 2.1 Entity Extraction Enhancement

Add a method to extract entities with LLM assistance when heuristics fail:

```python
async def extract_entities_from_conversation(
    self,
    messages: list[MessageModel]
) -> dict[str, Any]:
    """
    Use LLM to extract entities from conversation when heuristic extraction is insufficient.
    """
    
    conversation_text = "\n".join(
        f"{msg.author_name}: {msg.content}" for msg in messages[-5:]
    )
    
    prompt = f"""Analyze this Discord conversation and extract mentioned entities.

## Conversation
{conversation_text}

## Task
Identify all entities mentioned in the conversation:

1. **People**: Discord user IDs (format: <@123456>), names, or pronouns referring to specific individuals
2. **Organizations**: Companies, institutions, groups
3. **Topics**: Subjects of discussion, technologies, domains
4. **Locations**: Cities, countries, regions
5. **Skills**: Expertise areas, technologies, competencies

## Output Format
Return a JSON object:

{{
  "people": [
    {{"id": "<@123456>", "name": "Alice", "mentions": 2}},
    {{"id": null, "name": "Bob's brother", "mentions": 1, "relationship": "family"}}
  ],
  "organizations": ["Google", "MIT"],
  "topics": ["machine learning", "Python"],
  "locations": ["San Francisco", "New York"],
  "skills": ["Kubernetes", "React"],
  "implicit_references": [
    {{"reference": "my brother", "likely_refers_to": "person", "context": "..."}}
  ]
}}

Extract entities now:"""

    try:
        response = await self.apredict(prompt)
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:-3].strip()
        
        return json.loads(response_clean)
    except Exception as exc:
        logger.warning(f"LLM entity extraction failed: {exc}")
        return {
            "people": [],
            "organizations": [],
            "topics": [],
            "locations": [],
            "skills": [],
            "implicit_references": []
        }
```

#### 2.2 Fact Confidence Assessment

```python
async def assess_fact_confidence(
    self,
    fact: RetrievedFact,
    conversation_context: str
) -> dict[str, Any]:
    """
    Use LLM to assess whether a retrieved fact is relevant and trustworthy for the query.
    """
    
    prompt = f"""Evaluate the relevance and reliability of this fact for answering the user's query.

## User's Query Context
{conversation_context}

## Retrieved Fact
- Type: {fact.fact_type}
- Person: {fact.person_name} ({fact.person_id})
- Object: {fact.fact_object}
- Attributes: {json.dumps(fact.attributes, indent=2)}
- Confidence Score: {fact.confidence}
- Evidence: {len(fact.evidence)} message(s)
- Timestamp: {fact.timestamp}

## Task
Assess this fact on two dimensions:

1. **Relevance**: Does this fact help answer the user's query?
2. **Reliability**: Based on evidence count, confidence score, and recency, how trustworthy is this fact?

## Output Format
{{
  "relevance_score": 0.0-1.0,
  "relevance_explanation": "brief explanation",
  "reliability_score": 0.0-1.0,
  "reliability_explanation": "brief explanation",
  "should_include": true|false,
  "caveats": ["list", "of", "caveats"]
}}

Assess now:"""

    try:
        response = await self.apredict(prompt)
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:-3].strip()
        
        return json.loads(response_clean)
    except Exception as exc:
        logger.warning(f"Fact confidence assessment failed: {exc}")
        return {
            "relevance_score": 0.5,
            "relevance_explanation": "Unable to assess",
            "reliability_score": fact.confidence or 0.5,
            "reliability_explanation": "Defaulted to fact's confidence score",
            "should_include": True,
            "caveats": []
        }
```

#### 2.3 Stopping Condition Evaluation

```python
async def should_continue_searching(
    self,
    goal: str,
    retrieved_facts: list[RetrievedFact],
    tool_calls: list[dict],
    max_iterations: int,
    current_iteration: int
) -> dict[str, Any]:
    """
    Determine if the agent should continue searching or stop.
    """
    
    facts_summary = self._summarize_retrieved_facts(retrieved_facts)
    tool_history = self._format_tool_history(tool_calls)
    
    prompt = f"""Determine whether the agent should continue searching for more information or stop.

## Goal
{goal}

## Current Situation
- Iteration: {current_iteration}/{max_iterations}
- Facts Retrieved: {len(retrieved_facts)}
- Tool Calls Made: {len(tool_calls)}

## Retrieved Facts Summary
{facts_summary}

## Tool Call History
{tool_history}

## Decision Criteria
1. **Sufficiency**: Have we retrieved enough information to address the goal?
2. **Diminishing Returns**: Are recent tool calls returning 0 or duplicate results?
3. **Iteration Limit**: Are we approaching the maximum iterations?
4. **Goal Satisfaction**: Can the goal be reasonably answered with current facts?

## Output Format
{{
  "should_continue": true|false,
  "confidence": "high|medium|low",
  "reasoning": "2-3 sentence explanation",
  "recommendations": ["what to do next"]
}}

Decide now:"""

    try:
        response = await self.apredict(prompt)
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:-3].strip()
        
        return json.loads(response_clean)
    except Exception as exc:
        logger.warning(f"Stopping condition evaluation failed: {exc}")
        # Default to continuing if LLM fails, but respect iteration limit
        should_continue = current_iteration < max_iterations
        return {
            "should_continue": should_continue,
            "confidence": "low",
            "reasoning": f"LLM evaluation failed. Defaulting to {'continue' if should_continue else 'stop'}.",
            "recommendations": []
        }
```

#### 2.4 Query Refinement

```python
async def refine_tool_parameters(
    self,
    tool_name: str,
    initial_parameters: dict,
    conversation_context: str,
    previous_results: list[Any]
) -> dict[str, Any]:
    """
    Refine tool parameters based on previous results and conversation context.
    """
    
    prompt = f"""Refine the parameters for a knowledge graph tool based on context and previous results.

## Tool
{tool_name}

## Initial Parameters
{json.dumps(initial_parameters, indent=2)}

## Conversation Context
{conversation_context}

## Previous Results
{'No previous results (first attempt)' if not previous_results else f'{len(previous_results)} results from previous attempt'}

## Task
Analyze if the parameters can be improved:
- Make queries more specific or broader as needed
- Adjust confidence thresholds
- Change limits
- Add optional filters

## Output Format
{{
  "refined_parameters": {{...}},
  "changes_made": ["list of changes"],
  "reasoning": "brief explanation"
}}

Refine now:"""

    try:
        response = await self.apredict(prompt)
        response_clean = response.strip()
        if response_clean.startswith("```json"):
            response_clean = response_clean[7:-3].strip()
        
        return json.loads(response_clean)
    except Exception as exc:
        logger.warning(f"Parameter refinement failed: {exc}")
        return {
            "refined_parameters": initial_parameters,
            "changes_made": [],
            "reasoning": "Refinement failed, using original parameters"
        }
```

---

### 3. Integration with Agent Workflow

#### 3.1 Update Agent State

Add new fields to `memory_agent/state.py`:

```python
class AgentState(TypedDict, total=False):
    # ... existing fields ...
    
    # New fields for enhanced LLM integration
    llm_reasoning: list[dict[str, Any]]  # Structured LLM reasoning steps
    tool_selection_confidence: str  # LLM's confidence in tool choice
    fact_assessments: dict[str, dict]  # LLM assessments of retrieved facts
    entity_extraction_results: dict[str, Any]  # LLM entity extraction
    should_stop_evaluation: dict[str, Any]  # LLM stopping condition evaluation
```

#### 3.2 Update Agent Planning Node

In `memory_agent/agent.py`, update the `plan_queries` node:

```python
async def plan_queries(state: AgentState) -> AgentState:
    logger.debug("Planning next query, iteration %s", state.get("iteration"))
    
    # First try heuristic selection
    candidate = determine_tool_from_goal(state)
    
    # Enhance with LLM if available
    if candidate and llm:
        try:
            llm_result = await llm.aplan_tool_usage_enhanced(
                state.get("current_goal", ""),
                tools,
                state
            )
            
            # Store LLM reasoning
            llm_reasoning = list(state.get("llm_reasoning", []))
            llm_reasoning.append({
                "iteration": state.get("iteration", 0),
                "decision": llm_result,
                "timestamp": time.time()
            })
            
            # Check if LLM suggests stopping
            if llm_result.get("should_stop"):
                return {
                    "pending_tool": None,
                    "goal_accomplished": True,
                    "llm_reasoning": llm_reasoning,
                    "reasoning_trace": update_reasoning(
                        state,
                        f"LLM suggests stopping: {llm_result.get('stop_reason')}"
                    )
                }
            
            # Use LLM's tool choice if high confidence
            tool_name = llm_result.get("tool_name")
            if tool_name and tool_name in tools:
                confidence = llm_result.get("confidence", "low")
                
                # Merge LLM parameters with heuristic parameters
                params = candidate.get("input", {})
                if "parameters" in llm_result:
                    params.update(llm_result["parameters"])
                
                candidate = {
                    "name": tool_name,
                    "input": params
                }
                
                reasoning_msg = (
                    f"LLM selected {tool_name} "
                    f"(confidence: {confidence}): "
                    f"{llm_result.get('reasoning', 'No reasoning provided')}"
                )
                
                return {
                    "pending_tool": candidate,
                    "tool_selection_confidence": confidence,
                    "llm_reasoning": llm_reasoning,
                    "reasoning_trace": update_reasoning(state, reasoning_msg)
                }
                
        except Exception as exc:
            logger.debug("LLM planning failed: %s", exc)
            # Fall back to heuristic candidate
    
    reasoning_message = (
        f"Planned tool {candidate['name']}" if candidate else "No further tool required."
    )
    return {
        "pending_tool": candidate,
        "reasoning_trace": update_reasoning(state, reasoning_message),
    }
```

---

### 4. Prompt Engineering Best Practices

#### 4.1 Use Clear Structure

**Always include:**
- Clear role definition
- Explicit task description  
- Relevant context sections
- Output format specification
- Few-shot examples
- Constraints and rules

#### 4.2 Request Reasoning

Include chain-of-thought prompting:
```
Before selecting a tool, think through:
1. What information does the user need?
2. What have we already retrieved?
3. Which tool most directly provides this?
4. What parameters should we use?
```

#### 4.3 Provide Examples

Include 3-5 examples covering:
- Typical cases
- Edge cases
- Error conditions
- When NOT to use certain tools

#### 4.4 Specify Output Format Clearly

```
## CRITICAL: Output Format
Your response must be ONLY a valid JSON object.
Do NOT include markdown code blocks.
Do NOT include explanatory text outside the JSON.
```

#### 4.5 Handle Ambiguity

```
If the situation is ambiguous:
- State your uncertainty in the reasoning field
- Set confidence to "low"
- Suggest alternative approaches in recommendations
```

---

### 5. Error Handling and Validation

#### 5.1 Response Validation

```python
def _validate_llm_response(
    self,
    response: dict,
    required_fields: list[str],
    valid_values: dict[str, list[Any]] = None
) -> tuple[bool, str]:
    """
    Validate that LLM response has required structure.
    
    Returns:
        (is_valid, error_message)
    """
    valid_values = valid_values or {}
    
    # Check required fields
    for field in required_fields:
        if field not in response:
            return False, f"Missing required field: {field}"
    
    # Check valid values
    for field, allowed in valid_values.items():
        if field in response and response[field] not in allowed:
            return False, f"Invalid value for {field}: {response[field]}"
    
    return True, ""
```

#### 5.2 Graceful Degradation

```python
def _handle_llm_failure(
    self,
    failure_type: str,
    context: dict
) -> dict[str, Any]:
    """
    Provide reasonable defaults when LLM fails.
    """
    logger.warning(f"LLM failure: {failure_type}")
    
    if failure_type == "json_parse_error":
        # Try to extract tool name with regex
        pass
    elif failure_type == "timeout":
        # Use fastest fallback
        pass
    elif failure_type == "rate_limit":
        # Queue for retry
        pass
    
    return self._fallback_tool_selection(context)
```

---

### 6. Performance Optimizations

#### 6.1 Prompt Caching

```python
class PromptCache:
    """Cache commonly used prompt sections."""
    
    def __init__(self, max_size: int = 100):
        self._cache: dict[str, str] = {}
        self._max_size = max_size
    
    def get_or_generate(
        self,
        key: str,
        generator: Callable[[], str]
    ) -> str:
        if key not in self._cache:
            if len(self._cache) >= self._max_size:
                # Simple FIFO eviction
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = generator()
        return self._cache[key]
```

#### 6.2 Parallel LLM Calls

```python
async def batch_assess_facts(
    self,
    facts: list[RetrievedFact],
    context: str,
    max_concurrent: int = 3
) -> list[dict]:
    """
    Assess multiple facts in parallel.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def assess_with_limit(fact):
        async with semaphore:
            return await self.assess_fact_confidence(fact, context)
    
    tasks = [assess_with_limit(fact) for fact in facts]
    return await asyncio.gather(*tasks)
```

---

### 7. Testing Recommendations

#### 7.1 Unit Tests for Prompt Generation

```python
def test_tool_catalog_generation():
    """Test that tool catalog includes all necessary information."""
    llm = LLMClient(...)
    catalog = llm._build_tool_catalog(mock_tools)
    
    # Check all tools present
    for tool_name in mock_tools:
        assert tool_name in catalog
    
    # Check required sections
    assert "Description:" in catalog
    assert "Use When:" in catalog
    assert "Example:" in catalog
```

#### 7.2 Integration Tests with Mock LLM

```python
def test_tool_selection_with_person_query():
    """Test that person queries correctly select get_person_profile."""
    
    mock_llm = MockLLM(return_value={
        "tool_name": "get_person_profile",
        "reasoning": "User asked about specific person",
        "confidence": "high",
        "should_stop": False
    })
    
    state = create_test_state(
        conversation=[
            MessageModel(
                author_id="123",
                author_name="Alice",
                content="What do we know about Bob?",
                timestamp=datetime.now()
            )
        ],
        identified_entities={"people_ids": ["456"]}
    )
    
    result = await mock_llm.aplan_tool_usage_enhanced("...", tools, state)
    
    assert result["tool_name"] == "get_person_profile"
    assert result["confidence"] == "high"
```

#### 7.3 Prompt Quality Metrics

Track and monitor:
- **Parse Success Rate**: % of responses that parse as valid JSON
- **Tool Selection Accuracy**: % of correct tool selections (vs human labels)
- **Confidence Calibration**: Correlation between confidence and actual success
- **Reasoning Quality**: Manual review sample for coherence

---

### 8. Monitoring and Observability

#### 8.1 Logging LLM Interactions

```python
def log_llm_interaction(
    self,
    method: str,
    prompt_summary: str,
    response_summary: str,
    success: bool,
    latency_ms: float
):
    """Log structured LLM interaction data."""
    logger.info(
        "LLM interaction",
        extra={
            "method": method,
            "prompt_length": len(prompt_summary),
            "response_length": len(response_summary),
            "success": success,
            "latency_ms": latency_ms,
            "provider": self.provider,
            "model": self.model
        }
    )
```

#### 8.2 Metrics Collection

```python
from prometheus_client import Counter, Histogram

llm_calls_total = Counter(
    'memory_agent_llm_calls_total',
    'Total LLM calls',
    ['method', 'success']
)

llm_latency = Histogram(
    'memory_agent_llm_latency_seconds',
    'LLM call latency',
    ['method']
)
```

---

## Implementation Roadmap

### Phase 1: Core Improvements (Week 1)
1. Implement enhanced tool selection prompt with tool catalog
2. Add conversation context and fact summaries
3. Implement structured JSON output with validation
4. Add fallback handling

### Phase 2: Additional Methods (Week 2)
1. Add entity extraction enhancement
2. Add fact confidence assessment  
3. Add stopping condition evaluation
4. Integrate with agent workflow

### Phase 3: Optimization (Week 3)
1. Add prompt caching
2. Implement parallel fact assessment
3. Add comprehensive logging
4. Set up monitoring metrics

### Phase 4: Testing and Refinement (Week 4)
1. Write comprehensive unit tests
2. Conduct integration testing
3. Collect prompt quality metrics
4. Iterate based on real-world performance

---

## Expected Impact

### Quantitative Improvements
- **Tool Selection Accuracy**: 60% → 85%+
- **Parse Success Rate**: 80% → 95%+
- **Unnecessary Tool Calls**: -40%
- **Average Query Completion Time**: -25%

### Qualitative Improvements
- More coherent reasoning traces
- Better handling of ambiguous queries
- Reduced stuck/loop situations
- More explainable decisions
- Improved user confidence in results

---

## Conclusion

These improvements transform the LLM integration from a minimal tool selector to an intelligent reasoning partner that:

1. **Understands context deeply** through comprehensive state sharing
2. **Makes informed decisions** using detailed tool catalogs and examples
3. **Explains its reasoning** with chain-of-thought prompting
4. **Degrades gracefully** with robust fallback mechanisms
5. **Learns from experience** by tracking tool call history
6. **Stops intelligently** by evaluating goal satisfaction

The enhanced prompting strategy will significantly improve the memory agent's ability to retrieve relevant information efficiently while providing users with transparent, explainable decision-making.
