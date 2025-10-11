# Memory Agent: LLM-Based Knowledge Retrieval for Discord Chatbot

## Context

This document outlines the design for an intelligent knowledge retrieval agent that serves as the memory system for our Discord chatbot. The agent analyzes ongoing conversations and fetches relevant historical context from our knowledge graph to provide the chatbot with long-term memory capabilities.

**Key Design Decisions:**
- Prioritizes thoroughness over speed (willing to wait ~1 minute for responses)
- Uses LLM-based reasoning to intelligently explore the knowledge graph
- Designed for small-scale usage (5 users in a private Discord server)
- Cost and latency are not primary concerns

## Architecture Overview

### Core Concept: Reasoning-Acting Loop

The agent operates in an iterative loop, reasoning about what information would be useful and executing database queries until it has sufficient context.

```
User sends conversation
    ↓
Agent analyzes conversation
    ↓
Agent plans what to retrieve
    ↓
Agent executes queries (Neo4j/SQLite)
    ↓
Agent evaluates: "Do I have enough context?"
    ↓
    ├─ No → Formulate more queries, loop back
    └─ Yes → Synthesize findings and return
```

## Agent Tools

The agent will have access to the following database tools:

### 1. `cypher_query(query: str) -> List[Dict]`
Execute arbitrary Cypher queries against Neo4j knowledge graph.
- Returns results as list of records
- Includes error handling for syntax errors
- Used for graph traversal and relationship queries

### 2. `sql_query(query: str) -> List[Dict]`
Execute read-only SQL against SQLite message database.
- Useful for accessing raw messages and evidence
- Enables full-text search on message content
- Provides access to metadata not in the graph

### 3. `get_person_summary(person_id: str) -> Dict`
Convenience function for common operation: fetch all facts about a person.
- Returns structured profile with all fact types
- Includes confidence scores and evidence
- Pre-optimized common query

### 4. `get_recent_messages(channel_id: str, limit: int, before_timestamp: str) -> List[Dict]`
Fetch message history to understand broader conversation context.
- Sometimes agent needs to see what was discussed earlier
- Helps understand conversation flow and references

### 5. `search_similar_facts(query: str, fact_types: List[str], limit: int) -> List[Dict]`
Semantic search using embeddings (future enhancement).
- For fuzzy matching: "machine learning" → ML, AI, data science facts
- Helps find related concepts not explicitly mentioned

## Agent System Prompt

The system prompt guides the agent's behavior and decision-making process:

```
You are a knowledge retrieval agent for a Discord chatbot. Your job is to 
analyze a conversation and fetch relevant information from a knowledge graph 
about the participants and topics discussed.

AVAILABLE TOOLS:
- cypher_query: Execute Cypher queries against Neo4j knowledge graph
- sql_query: Execute SQL queries against SQLite message database  
- get_person_summary: Get all facts about a specific person
- get_recent_messages: Fetch message history from a channel
- search_similar_facts: Semantic search for related facts

KNOWLEDGE GRAPH STRUCTURE:
- Nodes: Person, Org, Place, Topic, Skill, Project, Event, etc.
- Relationships: WORKS_AT, LIVES_IN, TALKS_ABOUT, CLOSE_TO, HAS_SKILL, etc.
- All relationships have: confidence (float), evidence (message IDs), timestamp

TASK:
1. Analyze the conversation to identify:
   - Who is participating
   - What people/places/things are mentioned
   - What topics are being discussed
   - What questions might benefit from historical context

2. Fetch relevant knowledge:
   - Profiles of mentioned people
   - Facts related to discussed topics
   - Relationship information between people
   - Historical context from similar past conversations

3. Decide when you have enough:
   - Don't over-fetch - focus on what's relevant
   - But do explore connections (friends of mentioned people, related topics)
   - Usually 15-30 facts is sufficient

4. Return findings as structured JSON:
   {
     "relevant_facts": [...],
     "person_profiles": {...},
     "reasoning": "why I fetched what I fetched",
     "confidence": "how confident I am this is useful"
   }

GUIDELINES:
- Always check facts about conversation participants first
- If someone mentions a company/place/project, look up who else is connected
- For questions like "who should I ask about X", query skills and interests
- Include evidence (message IDs) so the chatbot can cite sources
- Think step by step - show your reasoning
```

## Implementation Approaches

### Option 1: Simple ReAct Pattern (Recommended for MVP)

Use a straightforward reasoning-acting loop with GPT-4/Claude:

1. Pass conversation + available tools to LLM
2. Agent outputs reasoning + tool calls
3. Execute tools, return results to agent
4. Agent sees results, decides next action
5. Repeat until agent outputs "DONE" with findings

**Pros:**
- Simplest to implement
- Leverages native LLM function calling
- Easy to debug and iterate

**Cons:**
- Less control over agent behavior
- Harder to enforce constraints

### Option 2: LangGraph State Machine

Use LangGraph to define explicit states and transitions:

```
States:
- Analyze: Understand the conversation
- Plan: Decide what queries to make
- Query: Execute database operations
- Evaluate: Check if enough context retrieved
- Synthesize: Format findings

Edges:
Analyze → Plan → Query → Evaluate
                         ↓
                    enough? → Synthesize → End
                         ↓
                     not enough → Plan (loop)
```

**Pros:**
- More structured and predictable
- Easier to add observability
- Better control over flow

**Cons:**
- More upfront complexity
- Requires LangGraph dependency

### Option 3: Custom Agent Loop

Write a custom agent loop with full control:

```python
def knowledge_retrieval_agent(conversation, max_iterations=10):
    context = {
        "conversation": conversation,
        "retrieved_facts": [],
        "queries_executed": [],
    }
    
    for iteration in range(max_iterations):
        # Agent reasons about what to do next
        decision = llm_decide_next_action(context)
        
        if decision["action"] == "query_neo4j":
            results = execute_cypher(decision["query"])
            context["retrieved_facts"].extend(results)
            context["queries_executed"].append(decision)
            
        elif decision["action"] == "query_sqlite":
            results = execute_sql(decision["query"])
            context["retrieved_facts"].extend(results)
            context["queries_executed"].append(decision)
            
        elif decision["action"] == "done":
            break
            
    return llm_synthesize_findings(context)
```

**Pros:**
- Maximum control and flexibility
- No external dependencies
- Can optimize exactly as needed

**Cons:**
- More code to maintain
- Need to handle edge cases manually

## Implementation Roadmap

### Phase 1: Basic Tools (Week 1)

**Goals:**
- Create and test individual tool functions
- Ensure database connectivity works
- Validate query execution and error handling

**Tasks:**
1. Implement `cypher_query()` wrapper around Neo4j driver
2. Implement `sql_query()` wrapper with read-only enforcement
3. Implement `get_person_summary()` helper function
4. Create tool schemas for LLM function calling
5. Write unit tests for each tool
6. Test tools with sample queries

**Deliverable:** Working tool functions that can be called independently

### Phase 2: Simple Agent (Week 2)

**Goals:**
- Implement basic ReAct loop
- Get agent making successful queries
- Establish feedback loop

**Tasks:**
1. Set up LLM client (OpenAI/Claude with function calling)
2. Implement basic agent loop with max 10 iterations
3. Add guardrails:
   - Query timeouts (5 sec max per query)
   - Complexity limits (no full graph scans)
   - Token usage tracking
4. Create test conversations as fixtures
5. Manually evaluate retrieved context quality

**Deliverable:** Working agent that can fetch relevant facts for test conversations

### Phase 3: Context Synthesis (Week 3)

**Goals:**
- Improve output formatting
- Add provenance tracking
- Optimize for chatbot consumption

**Tasks:**
1. Teach agent to format results as structured insights
2. Group facts by person/topic for better organization
3. Add relevance scoring
4. Include evidence links for every fact
5. Flag contradictions in retrieved facts
6. Format output as concise summaries

**Deliverable:** Agent returns well-formatted, chatbot-ready context

### Phase 4: Iterative Improvement (Ongoing)

**Goals:**
- Continuously improve retrieval quality
- Build common patterns library
- Refine prompting

**Tasks:**
1. Log all agent sessions (conversation → queries → results)
2. Manual review: was the context helpful?
3. Identify common query patterns
4. Add helper tools for frequent operations
5. Build prompt examples library
6. A/B test prompt variations

**Deliverable:** Continuously improving agent performance

## Example Agent Session

### Input Conversation:
```
Alice: "Hey, I'm thinking about applying to Google"
Bob: "Oh nice! Didn't Charlie work there?"
Alice: "Did he? I should ask him about it"
```

### Agent Reasoning Trace:

**Iteration 1:**
```
Thought: Alice is considering Google. Bob mentions Charlie might have worked there.
I should:
1. Check if Charlie has a WORKS_AT or PREVIOUSLY relationship with Google
2. Get Charlie's contact info/profile
3. Check if anyone else has Google experience

Tool call: cypher_query(
  "MATCH (p:Person {id: 'charlie_id'})-[r:WORKS_AT|PREVIOUSLY]->(o:Org) 
   WHERE o.name CONTAINS 'Google' 
   RETURN p, r, o"
)

Result: Charlie has PREVIOUSLY -> Google (2019-2022, SWE role, confidence 0.95)
```

**Iteration 2:**
```
Thought: Great! Charlie did work at Google. Let me see if anyone else has Google 
connections and get more details about Charlie's experience.

Tool call: cypher_query(
  "MATCH (p:Person)-[r:WORKS_AT|PREVIOUSLY]->(o:Org {name: 'Google'})
   RETURN p.id, p.realName, r.role, r.startDate, r.endDate, r.location
   ORDER BY r.confidence DESC"
)

Results: 
- Charlie: SWE, 2019-2022, Mountain View
- Dana: PM, 2020-present, San Francisco
```

**Iteration 3:**
```
Thought: Both Charlie and Dana have Google experience. Let me check if Alice knows Dana,
and get any topics Charlie has discussed about working at Google.

Tool call: cypher_query(
  "MATCH (alice:Person {id: 'alice_id'})-[:CLOSE_TO|INTERACTED_WITH]-(p:Person)
   WHERE p.id IN ['charlie_id', 'dana_id']
   RETURN p.id, p.realName"
)

Result: Alice is CLOSE_TO Charlie (basis: "collaborate weekly")
```

**Iteration 4:**
```
Thought: Perfect. Let me check if there are any facts about Charlie's thoughts on Google.

Tool call: cypher_query(
  "MATCH (charlie:Person {id: 'charlie_id'})-[r:TALKS_ABOUT]->(t:Topic)
   WHERE t.name CONTAINS 'Google' OR t.name CONTAINS 'work'
   RETURN t.name, r.sentiment, r.evidence
   LIMIT 5"
)

Result: [no results]
```

**Iteration 5:**
```
Thought: No explicit "talks about" facts. Let me search message history for context.

Tool call: sql_query(
  "SELECT m.id, m.content, m.timestamp 
   FROM message m 
   WHERE m.author_id = 'charlie_id' 
   AND (m.content LIKE '%Google%' OR m.content LIKE '%work%')
   ORDER BY m.timestamp DESC 
   LIMIT 10"
)

Result: [finds several messages about Charlie's experience]

Decision: DONE - I have enough context
```

### Final Output:
```json
{
  "relevant_facts": [
    {
      "type": "PREVIOUSLY",
      "person": "Charlie",
      "organization": "Google",
      "role": "Software Engineer",
      "duration": "2019-2022",
      "location": "Mountain View",
      "confidence": 0.95,
      "evidence": ["msg_123", "msg_456"]
    },
    {
      "type": "WORKS_AT",
      "person": "Dana",
      "organization": "Google",
      "role": "Product Manager",
      "current": true,
      "location": "San Francisco",
      "confidence": 0.92,
      "evidence": ["msg_789"]
    },
    {
      "type": "CLOSE_TO",
      "person_a": "Alice",
      "person_b": "Charlie",
      "basis": "collaborate weekly",
      "confidence": 0.85,
      "evidence": ["msg_234", "msg_567"]
    }
  ],
  "reasoning": "Alice is considering applying to Google. Charlie worked there 2019-2022 as a Software Engineer in Mountain View. Dana currently works there as a Product Manager in San Francisco. Alice is close to Charlie (they collaborate weekly), so he would be an excellent person to ask about the application process.",
  "confidence": "high",
  "suggested_context": "Alice should know that Charlie worked at Google as a Software Engineer from 2019-2022 and would be a great person to ask about the application process. Dana also currently works there as a PM if Alice wants a current perspective.",
  "queries_executed": 5,
  "facts_retrieved": 3
}
```

## REST API Specification

### Endpoint: `POST /api/knowledge/retrieve`

**Request:**
```json
{
  "messages": [
    {
      "author_id": "user123",
      "content": "Hey, I'm thinking about applying to Google",
      "timestamp": "2025-10-10T14:30:00Z"
    },
    {
      "author_id": "user456",
      "content": "Oh nice! Didn't Charlie work there?",
      "timestamp": "2025-10-10T14:30:15Z"
    }
  ],
  "channel_id": "channel_789",
  "max_facts": 30,
  "max_iterations": 10
}
```

**Response:**
```json
{
  "relevant_facts": [...],
  "person_profiles": {
    "charlie_id": "Charlie is a software engineer with 3 years at Google..."
  },
  "reasoning": "Analyzed conversation about Google application. Retrieved Charlie's employment history...",
  "queries_executed": [
    {
      "iteration": 1,
      "tool": "cypher_query",
      "query": "MATCH (p:Person {id: 'charlie_id'})...",
      "result_count": 1
    }
  ],
  "confidence": "high",
  "processing_time_ms": 45000
}
```

**Implementation Sketch:**
```python
@app.post("/api/knowledge/retrieve")
def retrieve_knowledge(request: KnowledgeRequest):
    """
    Agent-based knowledge retrieval for chatbot context.
    """
    agent = KnowledgeRetrievalAgent(
        neo4j_driver=neo4j_driver,
        sqlite_conn=sqlite_conn,
        llm_client=llm_client
    )
    
    result = agent.run(
        conversation=request.messages,
        max_facts=request.max_facts,
        max_iterations=request.max_iterations
    )
    
    return result
```

## Success Criteria

The agent will be considered successful if it:

1. **Retrieves relevant context** - Facts returned are useful for the chatbot to provide informed responses
2. **Explores connections** - Doesn't just fetch direct matches, but finds related information
3. **Includes provenance** - Every fact links back to source messages
4. **Explains reasoning** - Clear rationale for what was retrieved and why
5. **Completes within budget** - Finishes within max iterations and time limits
6. **Handles edge cases** - Gracefully handles missing data, empty results, query errors

## Tips for Success

1. **Start simple**: Begin with just 2-3 tools. Add more as needed based on actual usage patterns.

2. **Log everything**: Save every agent session to a debug database. Review what worked and what didn't.

3. **Give examples**: In the system prompt, include 2-3 example retrieval sessions showing good behavior.

4. **Use chain-of-thought**: Make the agent explain its reasoning at each step. This helps with debugging and improves results.

5. **Set hard limits**: 
   - Max 10 iterations to prevent infinite loops
   - Max 30 facts retrieved to avoid context overload
   - 5-second timeout per query to prevent hangs

6. **Test incrementally**: 
   - Start with simple conversations ("Hi Alice")
   - Gradually add complexity ("Who should I ask about TypeScript?")
   - Build up a test suite of realistic scenarios

7. **Version the prompts**: As you improve the system prompt, keep old versions. Sometimes simpler is better.

## Open Questions for Discussion

1. **Which implementation approach should we use?** (ReAct vs LangGraph vs Custom)
   - Trade-offs between simplicity and control
   - Team familiarity with frameworks

2. **What's the right balance for query limits?**
   - How many iterations before forcing stop?
   - How many facts is "too many" for chatbot context?

3. **Should we implement caching?**
   - Cache frequent queries (e.g., person summaries)?
   - Cache recent agent results for similar conversations?

4. **How do we measure success?**
   - Manual evaluation of helpfulness?
   - Automated metrics (precision/recall)?
   - User feedback from chatbot interactions?

5. **What about privacy/safety?**
   - Should certain facts be excluded from retrieval?
   - Do we need permission checks before returning information?

6. **Future enhancements:**
   - Semantic search with embeddings?
   - Learning from successful retrievals?
   - Proactive context fetching before conversation starts?

## Next Steps

1. Review and discuss this proposal with the team
2. Choose implementation approach (recommend starting with Simple ReAct)
3. Set up development environment and database connections
4. Implement Phase 1: Basic Tools
5. Create initial test suite with example conversations
6. Begin iterative development and testing
