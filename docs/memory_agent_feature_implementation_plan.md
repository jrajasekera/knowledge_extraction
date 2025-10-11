# Memory Agent Feature - Implementation Plan

## Executive Summary

This document provides a complete technical specification for implementing a memory agent service that retrieves relevant historical context from a Neo4j knowledge graph to enhance Discord chatbot responses. The agent uses LangGraph to orchestrate an iterative reasoning loop that explores the graph intelligently.

**Key Characteristics:**
- Separate microservice architecture
- LangGraph-based agent with semantic tools
- Returns list of formatted fact strings
- Target latency: ~30-60 seconds per request
- Small scale: 5 users, private Discord server

---

## 1. System Architecture

### 1.1 High-Level Flow

```
Discord User
    ↓ (sends message)
Discord Chatbot Service
    ↓ (POST /api/memory/retrieve with last N messages)
Memory Agent Service (this implementation)
    ↓ (queries)
Neo4j Knowledge Graph
    ↓ (returns facts)
Memory Agent Service
    ↓ (returns formatted fact strings)
Discord Chatbot Service
    ↓ (injects facts into LLM prompt)
LLM (GPT-4/Claude)
    ↓ (generates response with context)
Discord User
```

### 1.2 Service Responsibilities

**Memory Agent Service:**
- Accept conversation context via REST API
- Execute LangGraph agent loop
- Query Neo4j via high-level tools
- Perform semantic search on facts
- Return formatted fact strings

**Not Responsible For:**
- Discord message handling
- LLM prompt construction for chatbot
- Chatbot response generation
- User authentication (handled by chatbot service)

### 1.3 Technology Stack

- **Framework**: FastAPI (Python 3.13+)
- **Agent Framework**: LangGraph
- **LLM**: OpenAI GPT-4 or Anthropic Claude (configurable)
- **Database**: Neo4j (bolt connection)
- **Semantic Search**: sentence-transformers (same model as deduplication: `google/embeddinggemma-300m`)
- **Deployment**: Docker container

---

## 2. Tool Design

The agent has access to 10 high-level tools that abstract Neo4j queries. These tools provide semantic operations without requiring the agent to write Cypher.

### 2.1 Tool Catalog

#### Tool 1: `get_person_profile`
**Purpose**: Retrieve all facts about a specific person

**Input:**
```python
{
    "person_id": str,  # Discord member ID
    "fact_types": Optional[list[str]]  # Filter to specific types, e.g., ["WORKS_AT", "HAS_SKILL"]
}
```

**Output:**
```python
{
    "person_id": str,
    "name": str,
    "facts": [
        {
            "type": str,
            "object": str,
            "attributes": dict,
            "confidence": float,
            "evidence": list[str],
            "timestamp": str
        }
    ]
}
```

**Implementation Notes:**
- Query all relationships from `(:Person {id: $person_id})`
- Filter by `fact_types` if provided
- Return empty list if person not found

---

#### Tool 2: `find_people_by_skill`
**Purpose**: Find people who have a specific skill

**Input:**
```python
{
    "skill": str,  # Skill name (case-insensitive)
    "min_confidence": float = 0.5,
    "limit": int = 10
}
```

**Output:**
```python
{
    "skill": str,
    "people": [
        {
            "person_id": str,
            "name": str,
            "proficiency": str,
            "years_experience": Optional[int],
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Query: `MATCH (p:Person)-[r:HAS_SKILL]->(s:Skill) WHERE toLower(s.name) = toLower($skill)`
- Order by confidence DESC
- Apply semantic search if exact match fails (see Tool 10)

---

#### Tool 3: `find_people_by_organization`
**Purpose**: Find people who work or worked at an organization

**Input:**
```python
{
    "organization": str,
    "current_only": bool = False,
    "min_confidence": float = 0.5,
    "limit": int = 10
}
```

**Output:**
```python
{
    "organization": str,
    "people": [
        {
            "person_id": str,
            "name": str,
            "role": Optional[str],
            "start_date": Optional[str],
            "end_date": Optional[str],
            "location": Optional[str],
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Query both `WORKS_AT` and `PREVIOUSLY` relationships
- If `current_only=True`, filter for `WORKS_AT` only
- Support partial matching on organization name

---

#### Tool 4: `get_relationships_between`
**Purpose**: Find connections between two people

**Input:**
```python
{
    "person_a_id": str,
    "person_b_id": str
}
```

**Output:**
```python
{
    "relationships": [
        {
            "type": str,  # CLOSE_TO, RELATED_TO, etc.
            "attributes": dict,
            "confidence": float,
            "evidence": list[str]
        }
    ],
    "shared_contexts": [
        {
            "type": str,  # "same_organization", "same_project", "same_event"
            "context": str,
            "details": dict
        }
    ]
}
```

**Implementation Notes:**
- Direct relationships: `MATCH (a:Person {id: $a})-[r]-(b:Person {id: $b})`
- Shared contexts: Find common nodes (orgs, projects, events) both connect to
- Include interaction weight if `INTERACTED_WITH` relationship exists

---

#### Tool 5: `find_people_by_topic`
**Purpose**: Find people who discuss or care about a topic

**Input:**
```python
{
    "topic": str,
    "relationship_types": list[str] = ["TALKS_ABOUT", "CARES_ABOUT", "CURIOUS_ABOUT"],
    "min_confidence": float = 0.5,
    "limit": int = 10
}
```

**Output:**
```python
{
    "topic": str,
    "people": [
        {
            "person_id": str,
            "name": str,
            "relationship_type": str,
            "sentiment": Optional[str],
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Query multiple relationship types to topics
- Support semantic search for similar topics
- Aggregate if same person has multiple topic relationships

---

#### Tool 6: `get_person_timeline`
**Purpose**: Get temporal facts about a person (jobs, education, events)

**Input:**
```python
{
    "person_id": str,
    "fact_types": Optional[list[str]] = None,  # Default: temporal facts
    "start_date": Optional[str] = None,
    "end_date": Optional[str] = None
}
```

**Output:**
```python
{
    "person_id": str,
    "name": str,
    "timeline": [
        {
            "type": str,
            "object": str,
            "start": Optional[str],
            "end": Optional[str],
            "attributes": dict,
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Default fact types: WORKS_AT, STUDIED_AT, WORKING_ON, PREVIOUSLY, ATTENDED_EVENT, EXPERIENCED
- Sort by start date (or timestamp if no start date)
- Filter by date range if provided

---

#### Tool 7: `find_people_by_location`
**Purpose**: Find people who live in or have connection to a location

**Input:**
```python
{
    "location": str,
    "min_confidence": float = 0.5,
    "limit": int = 10
}
```

**Output:**
```python
{
    "location": str,
    "people": [
        {
            "person_id": str,
            "name": str,
            "relationship": str,  # "lives_in", "works_in", "studied_in"
            "details": dict,
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Query LIVES_IN relationships
- Also check work locations (WORKS_AT.location)
- Support fuzzy location matching (e.g., "SF" → "San Francisco")

---

#### Tool 8: `get_conversation_participants`
**Purpose**: Identify people mentioned or implied in conversation

**Input:**
```python
{
    "messages": list[dict],  # Conversation messages
}
```

**Output:**
```python
{
    "explicit_mentions": [
        {
            "name": str,
            "person_id": Optional[str],
            "mentioned_in_message": int  # index in messages array
        }
    ],
    "implicit_references": [
        {
            "reference": str,  # e.g., "my brother", "the new hire"
            "possible_matches": [
                {
                    "person_id": str,
                    "name": str,
                    "confidence": float,
                    "reason": str
                }
            ]
        }
    ]
}
```

**Implementation Notes:**
- Parse messages for names, pronouns, relationships
- Use member table + aliases for resolution
- Look for contextual clues (e.g., "my brother" + RELATED_TO relationships)

---

#### Tool 9: `find_experts`
**Purpose**: Find people best suited to answer a question or provide guidance

**Input:**
```python
{
    "query": str,  # Free-text query about what expertise is needed
    "limit": int = 5
}
```

**Output:**
```python
{
    "query": str,
    "experts": [
        {
            "person_id": str,
            "name": str,
            "relevance_score": float,
            "relevant_facts": [
                {
                    "type": str,
                    "description": str,
                    "confidence": float
                }
            ]
        }
    ]
}
```

**Implementation Notes:**
- Use semantic search to find relevant skills, topics, organizations
- Aggregate evidence across multiple fact types
- Weight recent facts higher than old ones

---

#### Tool 10: `semantic_search_facts`
**Purpose**: Find facts semantically similar to a query

**Input:**
```python
{
    "query": str,
    "fact_types": Optional[list[str]] = None,
    "limit": int = 10,
    "similarity_threshold": float = 0.7
}
```

**Output:**
```python
{
    "query": str,
    "results": [
        {
            "person_id": str,
            "person_name": str,
            "fact_type": str,
            "fact_object": str,
            "attributes": dict,
            "similarity_score": float,
            "confidence": float,
            "evidence": list[str]
        }
    ]
}
```

**Implementation Notes:**
- Embed query using sentence-transformers
- Query pre-computed fact embeddings (see Section 5)
- Return top K results above threshold
- This is the fallback for other tools when exact matches fail

---

### 2.2 Tool Implementation Pattern

All tools follow a consistent pattern:

```python
from typing import Any, Optional
from pydantic import BaseModel

class ToolInput(BaseModel):
    """Validated input schema"""
    pass

class ToolOutput(BaseModel):
    """Validated output schema"""
    pass

class ToolBase:
    def __init__(self, neo4j_driver, embeddings_model):
        self.driver = neo4j_driver
        self.embeddings = embeddings_model
    
    def execute(self, input: ToolInput) -> ToolOutput:
        """Execute tool logic with error handling"""
        try:
            result = self._query_neo4j(input)
            return self._format_output(result)
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return self._empty_response()
    
    def _query_neo4j(self, input: ToolInput) -> Any:
        """Override in subclass"""
        raise NotImplementedError
    
    def _format_output(self, result: Any) -> ToolOutput:
        """Override in subclass"""
        raise NotImplementedError
    
    def _empty_response(self) -> ToolOutput:
        """Return empty result on error"""
        raise NotImplementedError
```

---

## 3. LangGraph State Machine

### 3.1 State Definition

```python
from typing import TypedDict, Annotated
from langgraph.graph import add_messages

class AgentState(TypedDict):
    # Input (immutable)
    conversation: list[dict]  # Messages from chatbot
    channel_id: str
    max_facts: int
    max_iterations: int
    
    # Agent state (mutable)
    messages: Annotated[list, add_messages]  # LangGraph message history
    retrieved_facts: list[dict]  # Accumulated facts
    tool_calls: list[dict]  # History of tool invocations
    iteration: int
    
    # Reasoning state
    current_goal: str  # What the agent is trying to find
    identified_entities: dict  # People/topics/orgs mentioned
    
    # Output
    formatted_facts: list[str]  # Final output
    confidence: str  # "high", "medium", "low"
```

### 3.2 Node Definitions

#### Node: `analyze_conversation`
**Purpose**: Understand what information would be useful

**Logic:**
1. Parse conversation for entities (people, organizations, topics, locations)
2. Identify questions or information needs
3. Set initial `current_goal`
4. Update `identified_entities`

**Transition:**
- Always → `plan_queries`

---

#### Node: `plan_queries`
**Purpose**: Decide which tools to call next

**Logic:**
1. Review current goal and identified entities
2. Check what facts have been retrieved
3. Determine which tool would provide most relevant info
4. Update `current_goal` for next iteration

**Transition:**
- If `iteration >= max_iterations` → `synthesize`
- If `len(retrieved_facts) >= max_facts` → `synthesize`
- Else → `execute_tool`

---

#### Node: `execute_tool`
**Purpose**: Call selected tool and store results

**Logic:**
1. Execute tool with parameters from planning
2. Parse tool output
3. Add facts to `retrieved_facts`
4. Record tool call in `tool_calls`
5. Increment `iteration`

**Transition:**
- Always → `evaluate_progress`

---

#### Node: `evaluate_progress`
**Purpose**: Assess if we have sufficient information

**Logic:**
1. Check if current goal is satisfied
2. Determine if additional queries would be useful
3. Detect if stuck in loop (same tool called 3+ times)

**Transition:**
- If goal satisfied or stuck → `synthesize`
- Else → `plan_queries`

---

#### Node: `synthesize`
**Purpose**: Format facts into output strings

**Logic:**
1. Deduplicate facts (same person + fact type + object)
2. Sort by relevance and confidence
3. Format as natural language strings
4. Assess overall confidence based on:
   - Number of high-confidence facts found
   - Whether key entities were found
   - Tool success rate

**Output Format:**
```python
[
    "Alice works at Google as a Software Engineer in San Francisco (confidence: 0.95, evidence: msg_123)",
    "Charlie previously worked at Google from 2019-2022 as a Software Engineer (confidence: 0.92, evidence: msg_456)",
    "Alice and Charlie are close (basis: collaborate weekly, confidence: 0.85, evidence: msg_234)"
]
```

**Transition:**
- → END

---

### 3.3 Graph Construction

```python
from langgraph.graph import StateGraph, END

def create_memory_agent_graph():
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze_conversation", analyze_conversation)
    workflow.add_node("plan_queries", plan_queries)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("evaluate_progress", evaluate_progress)
    workflow.add_node("synthesize", synthesize)
    
    # Define edges
    workflow.set_entry_point("analyze_conversation")
    workflow.add_edge("analyze_conversation", "plan_queries")
    
    workflow.add_conditional_edges(
        "plan_queries",
        should_continue,
        {
            "continue": "execute_tool",
            "finish": "synthesize"
        }
    )
    
    workflow.add_edge("execute_tool", "evaluate_progress")
    
    workflow.add_conditional_edges(
        "evaluate_progress",
        evaluate_next_step,
        {
            "continue": "plan_queries",
            "finish": "synthesize"
        }
    )
    
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()
```

---

## 4. API Specification

### 4.1 Endpoint: POST /api/memory/retrieve

**Request:**
```json
{
    "messages": [
        {
            "author_id": "user123",
            "author_name": "Alice",
            "content": "Hey, I'm thinking about applying to Google",
            "timestamp": "2025-10-10T14:30:00Z"
        },
        {
            "author_id": "user456",
            "author_name": "Bob",
            "content": "Oh nice! Didn't Charlie work there?",
            "timestamp": "2025-10-10T14:30:15Z"
        }
    ],
    "channel_id": "channel_789",
    "max_facts": 30,
    "max_iterations": 10
}
```

**Response (Success):**
```json
{
    "facts": [
        "Charlie previously worked at Google from 2019-2022 as a Software Engineer in Mountain View (confidence: 0.95, evidence: msg_123, msg_456)",
        "Dana currently works at Google as a Product Manager in San Francisco (confidence: 0.92, evidence: msg_789)",
        "Alice is close to Charlie (basis: collaborate weekly, confidence: 0.85, evidence: msg_234, msg_567)"
    ],
    "confidence": "high",
    "metadata": {
        "queries_executed": 5,
        "facts_retrieved": 3,
        "processing_time_ms": 45000,
        "iterations_used": 5
    }
}
```

**Response (No Results):**
```json
{
    "facts": [],
    "confidence": "low",
    "metadata": {
        "queries_executed": 3,
        "facts_retrieved": 0,
        "processing_time_ms": 12000,
        "iterations_used": 3
    }
}
```

**Response (Error):**
```json
{
    "error": "Internal server error",
    "message": "Neo4j connection failed",
    "request_id": "req_abc123"
}
```

### 4.2 Endpoint: GET /health

**Response:**
```json
{
    "status": "healthy",
    "neo4j_connected": true,
    "model_loaded": true,
    "version": "0.1.0"
}
```

---

## 5. Semantic Search Implementation

### 5.1 Fact Embedding Strategy

**Pre-computation:**
- Generate embeddings for all facts during fact materialization
- Store embeddings separately in Neo4j or vector database

**Embedding Schema:**
```python
{
    "fact_id": int,
    "person_id": str,
    "fact_type": str,
    "text": str,  # Formatted fact description
    "embedding": list[float],  # 768-dimensional vector
    "confidence": float,
    "created_at": str
}
```

**Text Format for Embedding:**
```python
def format_fact_for_embedding(fact):
    person = fact.get("person_name", fact["person_id"])
    fact_type = fact["type"]
    obj = fact.get("object_label", "")
    
    # Attribute summary
    attr_str = ", ".join(f"{k}={v}" for k, v in fact.get("attributes", {}).items())
    
    return f"{person} {fact_type} {obj}. {attr_str}"

# Example output:
# "Alice WORKS_AT Google. role=Software Engineer, location=San Francisco, start_date=2022-01"
```

### 5.2 Vector Store Options

**Option A: Neo4j Vector Index (Recommended)**
- Native integration with existing graph
- Available in Neo4j 5.11+
- Query syntax:
```cypher
CALL db.index.vector.queryNodes('fact_embeddings', 10, $query_embedding)
YIELD node, score
```

**Option B: Separate Vector DB (Alternative)**
- Use Qdrant, Weaviate, or ChromaDB
- Better for high-volume queries
- Requires synchronization with Neo4j

**Implementation Choice**: Use Neo4j vector index for MVP

### 5.3 Semantic Search Tool Implementation

```python
from sentence_transformers import SentenceTransformer

class SemanticSearchTool(ToolBase):
    def __init__(self, neo4j_driver, model_name="google/embeddinggemma-300m"):
        super().__init__(neo4j_driver, None)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
    
    def execute(self, input: SemanticSearchInput) -> SemanticSearchOutput:
        # Generate query embedding
        query_embedding = self.model.encode(input.query, normalize_embeddings=True)
        
        # Query Neo4j vector index
        with self.driver.session() as session:
            results = session.run("""
                CALL db.index.vector.queryNodes('fact_embeddings', $k, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                MATCH (p:Person {id: node.person_id})
                RETURN p.id AS person_id, 
                       COALESCE(p.realName, p.name) AS person_name,
                       node.fact_type AS fact_type,
                       node.fact_data AS fact_data,
                       score AS similarity_score,
                       node.confidence AS confidence
                ORDER BY score DESC
            """, {
                "k": input.limit,
                "embedding": query_embedding.tolist(),
                "threshold": input.similarity_threshold
            })
            
            return self._format_results(results)
```

---

## 6. Data Models

### 6.1 Core Models

```python
from pydantic import BaseModel, Field
from typing import Optional

class Message(BaseModel):
    author_id: str
    author_name: str
    content: str
    timestamp: str

class RetrievalRequest(BaseModel):
    messages: list[Message]
    channel_id: str
    max_facts: int = Field(default=30, ge=1, le=100)
    max_iterations: int = Field(default=10, ge=1, le=20)

class Fact(BaseModel):
    person_id: str
    person_name: str
    fact_type: str
    object_label: str
    object_id: Optional[str]
    attributes: dict
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str]
    timestamp: Optional[str]

class RetrievalResponse(BaseModel):
    facts: list[str]
    confidence: str  # "high", "medium", "low"
    metadata: dict

class ToolCall(BaseModel):
    tool_name: str
    input_params: dict
    output: dict
    timestamp: str
    success: bool
```

### 6.2 Confidence Assessment

```python
def assess_confidence(state: AgentState) -> str:
    """Determine overall confidence level"""
    facts = state["retrieved_facts"]
    
    if not facts:
        return "low"
    
    high_conf_facts = sum(1 for f in facts if f["confidence"] >= 0.8)
    avg_confidence = sum(f["confidence"] for f in facts) / len(facts)
    
    tool_success_rate = sum(
        1 for call in state["tool_calls"] if call["success"]
    ) / len(state["tool_calls"]) if state["tool_calls"] else 0
    
    if high_conf_facts >= 3 and avg_confidence >= 0.75 and tool_success_rate >= 0.7:
        return "high"
    elif high_conf_facts >= 1 and avg_confidence >= 0.6:
        return "medium"
    else:
        return "low"
```

---

## 7. Implementation Phases

### Phase 1: Foundation (Week 1-2)

**Deliverables:**
- [ ] FastAPI service scaffolding
- [ ] Neo4j connection pooling
- [ ] Health check endpoint
- [ ] Configuration management (environment variables)
- [ ] Logging infrastructure
- [ ] Docker container + docker-compose setup

**Tasks:**
1. Set up project structure
   ```
   memory_agent/
   ├── src/
   │   ├── api/
   │   │   └── routes.py
   │   ├── agent/
   │   │   ├── graph.py
   │   │   ├── nodes.py
   │   │   └── state.py
   │   ├── tools/
   │   │   ├── base.py
   │   │   └── implementations/
   │   ├── models/
   │   │   └── schemas.py
   │   └── config.py
   ├── tests/
   ├── Dockerfile
   ├── docker-compose.yml
   └── requirements.txt
   ```

2. Implement configuration system
   ```python
   from pydantic_settings import BaseSettings
   
   class Settings(BaseSettings):
       neo4j_uri: str = "bolt://localhost:7687"
       neo4j_user: str = "neo4j"
       neo4j_password: str
       openai_api_key: str
       embedding_model: str = "google/embeddinggemma-300m"
       max_iterations: int = 10
       max_facts: int = 30
       
       class Config:
           env_file = ".env"
   ```

3. Create Neo4j connection manager
4. Set up structured logging with request tracing
5. Write integration tests for Neo4j connectivity

---

### Phase 2: Tool Implementation (Week 2-3)

**Deliverables:**
- [ ] All 10 tools implemented
- [ ] Tool unit tests
- [ ] Tool integration tests with test Neo4j data
- [ ] Tool documentation

**Tasks:**
1. Implement `ToolBase` abstract class
2. Implement each tool following the specification
3. Create test fixtures with sample Neo4j data
4. Write unit tests for each tool (80%+ coverage)
5. Create tool registry for agent access
6. Document tool usage with examples

**Testing Strategy:**
```python
# Example test
def test_get_person_profile_tool():
    # Arrange
    tool = GetPersonProfileTool(neo4j_driver)
    input_data = GetPersonProfileInput(
        person_id="user123",
        fact_types=["WORKS_AT", "HAS_SKILL"]
    )
    
    # Act
    result = tool.execute(input_data)
    
    # Assert
    assert result.person_id == "user123"
    assert len(result.facts) > 0
    assert all(f.type in ["WORKS_AT", "HAS_SKILL"] for f in result.facts)
```

---

### Phase 3: Semantic Search (Week 3-4)

**Deliverables:**
- [ ] Fact embedding pipeline
- [ ] Neo4j vector index setup
- [ ] Semantic search tool
- [ ] Embedding quality evaluation

**Tasks:**
1. Create script to embed existing facts
   ```python
   # scripts/embed_facts.py
   def embed_all_facts(driver, model):
       # Fetch all facts from Neo4j
       # Generate embeddings
       # Store embeddings back in Neo4j
       pass
   ```

2. Set up Neo4j vector index
   ```cypher
   CREATE VECTOR INDEX fact_embeddings IF NOT EXISTS
   FOR (f:Fact)
   ON f.embedding
   OPTIONS {indexConfig: {
     `vector.dimensions`: 768,
     `vector.similarity_function`: 'cosine'
   }}
   ```

3. Implement semantic search tool
4. Test semantic similarity quality
5. Add embedding generation to fact materialization pipeline

---

### Phase 4: LangGraph Agent (Week 4-5)

**Deliverables:**
- [ ] LangGraph state machine
- [ ] All node implementations
- [ ] Agent execution logic
- [ ] Agent testing framework

**Tasks:**
1. Implement `AgentState` TypedDict
2. Implement each node function:
   - `analyze_conversation`
   - `plan_queries`
   - `execute_tool`
   - `evaluate_progress`
   - `synthesize`
3. Create conditional edge logic
4. Compile LangGraph workflow
5. Test agent with mock tools
6. Test agent with real tools and data
7. Add observability (log each state transition)

**Testing Strategy:**
```python
def test_agent_end_to_end():
    # Arrange
    graph = create_memory_agent_graph()
    initial_state = {
        "conversation": [
            {"author_id": "user123", "content": "Who knows Python?"}
        ],
        "channel_id": "ch1",
        "max_facts": 10,
        "max_iterations": 5,
        "retrieved_facts": [],
        "tool_calls": [],
        "iteration": 0,
        "messages": []
    }
    
    # Act
    final_state = graph.invoke(initial_state)
    
    # Assert
    assert len(final_state["formatted_facts"]) > 0
    assert final_state["iteration"] <= 5
    assert final_state["confidence"] in ["high", "medium", "low"]
```

---

### Phase 5: API Integration (Week 5-6)

**Deliverables:**
- [ ] `/api/memory/retrieve` endpoint
- [ ] Request validation
- [ ] Response formatting
- [ ] Error handling
- [ ] Rate limiting
- [ ] API documentation

**Tasks:**
1. Implement FastAPI route
   ```python
   @app.post("/api/memory/retrieve")
   async def retrieve_memories(
       request: RetrievalRequest,
       agent: MemoryAgent = Depends(get_agent)
   ) -> RetrievalResponse:
       try:
           result = await agent.run(request)
           return RetrievalResponse(**result)
       except Exception as e:
           logger.error(f"Retrieval failed: {e}")
           raise HTTPException(status_code=500, detail=str(e))
   ```

2. Add request validation middleware
3. Implement response formatting
4. Add comprehensive error handling
5. Set up rate limiting (e.g., 10 requests/minute per user)
6. Generate OpenAPI documentation
7. Create integration tests for API endpoints

---

### Phase 6: Testing & Optimization (Week 6-7)

**Deliverables:**
- [ ] Full test suite (unit + integration + e2e)
- [ ] Performance benchmarks
- [ ] Load testing results
- [ ] Optimization recommendations

**Tasks:**
1. Achieve 80%+ code coverage
2. Write end-to-end tests simulating real conversations
3. Performance testing:
   - Measure average latency
   - Identify bottlenecks
   - Optimize slow Neo4j queries
4. Load testing:
   - Test with 10 concurrent requests
   - Verify no resource leaks
5. Create test data generators
6. Document test suite

**Performance Targets:**
- P50 latency: < 30 seconds
- P95 latency: < 60 seconds
- Memory usage: < 1GB per request
- Successful completion rate: > 95%

---

### Phase 7: Deployment (Week 7-8)

**Deliverables:**
- [ ] Production Docker image
- [ ] Kubernetes manifests (if applicable)
- [ ] Monitoring dashboards
- [ ] Alerting rules
- [ ] Deployment documentation
- [ ] Runbook for operations

**Tasks:**
1. Create production Dockerfile with multi-stage build
2. Set up CI/CD pipeline
3. Configure monitoring:
   - Prometheus metrics
   - Request latency histograms
   - Error rate tracking
   - Neo4j query performance
4. Set up alerting:
   - High error rate
   - High latency
   - Neo4j connection failures
5. Create deployment runbook
6. Deploy to staging environment
7. Run smoke tests
8. Deploy to production

---

## 8. Testing Strategy

### 8.1 Test Data Setup

Create a test Neo4j database with realistic data:

```python
# tests/fixtures/graph_data.py

TEST_PEOPLE = [
    {
        "id": "alice_123",
        "name": "Alice",
        "realName": "Alice Johnson"
    },
    {
        "id": "bob_456",
        "name": "Bob",
        "realName": "Bob Smith"
    }
]

TEST_FACTS = [
    {
        "person_id": "alice_123",
        "type": "WORKS_AT",
        "object": "Google",
        "attributes": {
            "role": "Software Engineer",
            "location": "San Francisco"
        },
        "confidence": 0.95,
        "evidence": ["msg_001", "msg_002"]
    },
    {
        "person_id": "alice_123",
        "type": "HAS_SKILL",
        "object": "Python",
        "attributes": {
            "proficiency": "expert",
            "years_experience": 5
        },
        "confidence": 0.9,
        "evidence": ["msg_003"]
    }
]

def populate_test_graph(driver):
    """Populate Neo4j with test data"""
    with driver.session() as session:
        # Create people
        for person in TEST_PEOPLE:
            session.run(
                "MERGE (p:Person {id: $id}) SET p.name = $name, p.realName = $realName",
                person
            )
        
        # Create facts
        for fact in TEST_FACTS:
            # Implementation details...
            pass
```

### 8.2 Test Categories

**Unit Tests (tools/):**
- Each tool in isolation
- Mock Neo4j responses
- Verify output format
- Handle edge cases (empty results, errors)

**Integration Tests (agent/):**
- LangGraph state transitions
- Tool execution from agent
- Multi-step reasoning flows

**End-to-End Tests (api/):**
- Full request/response cycle
- Real Neo4j test database
- Realistic conversation scenarios

**Performance Tests:**
- Measure query latency
- Stress test with concurrent requests
- Memory profiling

### 8.3 Example E2E Test

```python
def test_find_expert_scenario():
    """
    Scenario: User asks 'Who should I ask about TypeScript?'
    Expected: Agent finds people with TypeScript skill
    """
    # Arrange
    request = RetrievalRequest(
        messages=[
            {
                "author_id": "user_999",
                "author_name": "Test User",
                "content": "Who should I ask about TypeScript?",
                "timestamp": "2025-10-10T10:00:00Z"
            }
        ],
        channel_id": "test_channel",
        max_facts=10,
        max_iterations=5
    )
    
    # Act
    response = client.post("/api/memory/retrieve", json=request.dict())
    
    # Assert
    assert response.status_code == 200
    data = response.json()
    assert len(data["facts"]) > 0
    assert any("TypeScript" in fact for fact in data["facts"])
    assert data["confidence"] in ["high", "medium", "low"]
    assert data["metadata"]["processing_time_ms"] < 60000
```

---

## 9. Observability & Debugging

### 9.1 Logging Strategy

**Structured Logging Format:**
```json
{
    "timestamp": "2025-10-10T14:30:15Z",
    "level": "INFO",
    "request_id": "req_abc123",
    "component": "agent",
    "node": "plan_queries",
    "iteration": 2,
    "message": "Planning to call get_person_profile tool",
    "context": {
        "person_id": "alice_123",
        "current_goal": "Find info about Alice"
    }
}
```

**Log Levels:**
- `DEBUG`: State transitions, tool parameters
- `INFO`: Tool executions, major decisions
- `WARNING`: Tool failures, approaching limits
- `ERROR`: Unhandled exceptions, service failures

### 9.2 Metrics to Track

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
requests_total = Counter(
    'memory_agent_requests_total',
    'Total requests',
    ['status']
)

request_duration = Histogram(
    'memory_agent_request_duration_seconds',
    'Request duration',
    buckets=[1, 5, 10, 30, 60, 120]
)

# Tool metrics
tool_calls_total = Counter(
    'memory_agent_tool_calls_total',
    'Total tool calls',
    ['tool_name', 'success']
)

tool_duration = Histogram(
    'memory_agent_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

# Agent metrics
facts_retrieved = Histogram(
    'memory_agent_facts_retrieved',
    'Number of facts retrieved per request'
)

iterations_used = Histogram(
    'memory_agent_iterations',
    'Number of iterations per request'
)
```

### 9.3 Debug Endpoint

```python
@app.post("/api/memory/retrieve/debug")
async def retrieve_memories_debug(
    request: RetrievalRequest,
    agent: MemoryAgent = Depends(get_agent)
) -> dict:
    """
    Returns full agent trace for debugging
    """
    result = await agent.run(request, debug_mode=True)
    return {
        "facts": result["facts"],
        "confidence": result["confidence"],
        "metadata": result["metadata"],
        "debug_info": {
            "state_history": result["state_history"],
            "tool_calls": result["tool_calls"],
            "reasoning_trace": result["reasoning_trace"]
        }
    }
```

---

## 10. Configuration & Deployment

### 10.1 Environment Variables

```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_TEMPERATURE=0.3

# Embedding Model
EMBEDDING_MODEL=google/embeddinggemma-300m
EMBEDDING_DEVICE=cpu  # or 'cuda' for GPU

# Agent Configuration
MAX_ITERATIONS=10
MAX_FACTS=30
TOOL_TIMEOUT_SECONDS=10

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
ENABLE_CORS=true

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60  # seconds
```

### 10.2 Docker Compose

```yaml
version: '3.8'

services:
  memory-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - neo4j
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  neo4j:
    image: neo4j:5.22
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}
    volumes:
      - neo4j_data:/data
    restart: unless-stopped

volumes:
  neo4j_data:
```

### 10.3 Kubernetes Deployment (Optional)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: memory-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: memory-agent
  template:
    metadata:
      labels:
        app: memory-agent
    spec:
      containers:
      - name: memory-agent
        image: memory-agent:latest
        ports:
        - containerPort: 8000
        env:
        - name: NEO4J_URI
          value: "bolt://neo4j-service:7687"
        - name: NEO4J_PASSWORD
          valueFrom:
            secretKeyRef:
              name: neo4j-secret
              key: password
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
```

---

## 11. Success Criteria

### 11.1 Functional Requirements

- [ ] Agent successfully retrieves relevant facts for test conversations
- [ ] All 10 tools implemented and tested
- [ ] Semantic search returns relevant results
- [ ] API returns properly formatted fact strings
- [ ] Agent stops within iteration limits
- [ ] Error handling prevents crashes

### 11.2 Performance Requirements

- [ ] P50 latency < 30 seconds
- [ ] P95 latency < 60 seconds
- [ ] 95%+ successful completion rate
- [ ] Memory usage < 1GB per request
- [ ] Can handle 5 concurrent requests

### 11.3 Quality Requirements

- [ ] 80%+ code coverage
- [ ] All tools have unit tests
- [ ] E2E tests for common scenarios
- [ ] Documentation complete
- [ ] Code passes linting (ruff, mypy)

---

## 12. Future Enhancements

### Phase 2 Features (Post-MVP)

1. **Caching Layer**
   - Cache recent queries
   - Cache person profiles
   - Reduce Neo4j load

2. **Proactive Context**
   - Detect conversation start
   - Pre-fetch likely relevant facts
   - Reduce latency

3. **Feedback Loop**
   - Track which facts were useful
   - Use feedback to improve tool selection
   - Learn user-specific patterns

4. **Multi-Modal Facts**
   - Support image/attachment context
   - Link to Discord attachments
   - Summarize linked content

5. **Conversational Memory**
   - Remember previous queries in session
   - Avoid redundant fact retrieval
   - Build on previous context

6. **Advanced Analytics**
   - Track fact usage patterns
   - Identify knowledge gaps
   - Suggest data to add to graph

---

## 13. Risk Mitigation

### Risk 1: Agent Gets Stuck in Loops
**Mitigation:**
- Hard iteration limit
- Detect repeated tool calls (same tool 3+ times)
- Graceful degradation to partial results

### Risk 2: Neo4j Query Performance
**Mitigation:**
- Add query timeouts (10s per tool)
- Monitor slow queries
- Optimize indexes during Phase 6
- Cache frequently accessed data

### Risk 3: Semantic Search Quality
**Mitigation:**
- Extensive embedding quality testing
- Manual evaluation of top-K results
- Fallback to keyword search
- Tunable similarity threshold

### Risk 4: LLM API Failures
**Mitigation:**
- Retry logic with exponential backoff
- Fallback to simpler reasoning
- Circuit breaker pattern
- Error handling at every LLM call

### Risk 5: Context Overload (Too Many Facts)
**Mitigation:**
- Enforce max_facts limit
- Prioritize high-confidence facts
- Deduplicate aggressively
- Summarize where possible

---

## 14. Documentation Deliverables

1. **API Documentation**
   - OpenAPI/Swagger spec
   - Request/response examples
   - Error codes and meanings

2. **Developer Guide**
   - Setup instructions
   - How to add new tools
   - How to modify agent behavior
   - Testing guidelines

3. **Operations Runbook**
   - Deployment steps
   - Monitoring dashboards
   - Common issues and solutions
   - Scaling guidelines

4. **Architecture Diagrams**
   - System architecture
   - LangGraph state machine
   - Data flow diagrams

---

## 15. Team Responsibilities

### Backend Engineer 1: Tools & API
- Implement all 10 tools
- Tool testing
- FastAPI endpoints
- Request validation

### Backend Engineer 2: Agent & LangGraph
- LangGraph state machine
- Agent nodes
- State management
- Agent testing

### ML Engineer: Semantic Search
- Fact embedding pipeline
- Vector index setup
- Semantic search tool
- Embedding quality evaluation

### DevOps Engineer: Deployment
- Docker setup
- CI/CD pipeline
- Monitoring/alerting
- Production deployment

### QA Engineer: Testing
- Test plan creation
- E2E test implementation
- Performance testing
- Load testing

---

## Appendix A: Sample Conversations for Testing

### Test Case 1: Simple Skill Query
**Input:**
```json
{
    "messages": [
        {"author_id": "user1", "content": "Who knows Python?"}
    ]
}
```
**Expected Output:**
- List of people with Python skill
- Proficiency levels
- Evidence links

### Test Case 2: Organization Query
**Input:**
```json
{
    "messages": [
        {"author_id": "user1", "content": "Anyone work at Google?"}
    ]
}
```
**Expected Output:**
- Current and past Google employees
- Roles and locations
- Timeline information

### Test Case 3: Complex Multi-Entity Query
**Input:**
```json
{
    "messages": [
        {"author_id": "user1", "content": "I'm thinking about applying to Google"},
        {"author_id": "user2", "content": "Didn't Charlie work there?"},
        {"author_id": "user1", "content": "Yeah, I should ask him about it"}
    ]
}
```
**Expected Output:**
- Charlie's Google employment facts
- Other Google employees
- Charlie's relationship to user1
- Relevant skills/topics

### Test Case 4: No Results
**Input:**
```json
{
    "messages": [
        {"author_id": "user1", "content": "Does anyone speak Klingon?"}
    ]
}
```
**Expected Output:**
- Empty facts array
- Low confidence
- Graceful handling

---

## Appendix B: Tool Implementation Checklist

For each tool, complete:

- [ ] Input schema defined (Pydantic model)
- [ ] Output schema defined (Pydantic model)
- [ ] Cypher query written and tested
- [ ] Error handling implemented
- [ ] Empty result handling
- [ ] Unit tests written
- [ ] Integration test with test data
- [ ] Documentation with examples
- [ ] Performance validated (< 10s)
- [ ] Code review completed

---

**End of Implementation Plan**

This document should be treated as a living specification. Update as requirements change or new learnings emerge during implementation.
