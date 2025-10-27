## Profile Generation Architecture

### **Stage 1: Fact Aggregation & Enrichment**

Query Neo4j to gather all facts about a person, grouped by category:

```
For person "Alice":
- Identity: name, roles, basic info
- Work: WORKS_AT, STUDIED_AT, HAS_SKILL, WORKING_ON
- Location: LIVES_IN
- Interests: TALKS_ABOUT, ENJOYS, CURIOUS_ABOUT
- Social: CLOSE_TO, RELATED_TO
- Preferences: PREFERS, AVOIDS, BELIEVES
- Recent activity: message frequency, last active date
- Interaction patterns: who they talk to most, typical topics
```

**Enrichment:**
- Add temporal signals: "current" vs "past" (based on end_date, timestamp)
- Add recency scores: facts from last 30 days >> facts from 6 months ago
- Add confidence: surface high-confidence facts prominently
- Add evidence counts: facts supported by many messages are stronger

### **Stage 2: Fact Prioritization**

Not everything goes in the profile. Use a scoring system:

**Relevance Scoring:**
- **Recency weight**: Recent facts score higher (exponential decay from last message date)
- **Confidence weight**: High-confidence facts (>0.8) score 2x
- **Evidence count**: More supporting messages = higher score
- **Fact type importance**: For chatbot context, WORKS_AT/STUDIED_AT/INTERESTS > minor preferences
- **Conversation frequency**: Facts about frequently discussed topics rank higher

**Selection Strategy:**
- Top 2-3 work/education facts (prioritize current over past)
- Top 3-5 skills (especially if mentioned recently)
- Top 3-5 interests/topics they care about
- 1-2 key relationships (close friends, family)
- Notable preferences if strongly stated
- Current projects if active

### **Stage 3: Profile Template Structure**

Create a hierarchical profile with sections:

```
## Alice Chen (@alice_dev)

**Core Identity:**
[Current role/status in 1-2 sentences]

**Professional:**
[Work, education, skills, projects]

**Interests & Activities:**
[What they talk about, enjoy, are curious about]

**Social Context:**
[Key relationships, communication style]

**Preferences & Beliefs:**
[Strong preferences, values, things to avoid]

**Recent Context:**
[Last active, recent topics, current focus]
```

### **Stage 4: Narrative Generation**

Use an LLM to turn structured facts into natural prose. Two approaches:

#### **Approach A: Templated with LLM Refinement**

1. Generate structured fact bullets first
2. Use LLM to make it flow naturally while preserving all information
3. Ensures factual accuracy, improves readability

#### **Approach B: Direct LLM Generation from Facts**

1. Provide all relevant facts to LLM as structured input
2. Ask LLM to write coherent profile
3. Validate output contains all key facts

**I'd recommend Approach A** for better control and reliability.

### **Stage 5: Evidence Attribution**

Critical for chatbot trust:

```
Alice works as a Senior Software Engineer at Google in San Francisco [msg_12345, msg_12890]. 
She's passionate about Rust and distributed systems [msg_13001, msg_13445, msg_14002].
```

Or use footnote-style:
```
Alice works as a Senior Software Engineer at Google in San Francisco. She's passionate 
about Rust and distributed systems, often helping others with debugging[1].

[1] Based on 15 messages between 2024-08-15 and 2025-01-20
```

## Implementation Patterns

### **Pattern 1: Tiered Profiles**

Generate multiple profile lengths for different use cases:

- **Micro** (50-100 tokens): Name, current role, 2-3 key traits - for every message
- **Standard** (200-300 tokens): Full professional + interests - for direct conversations
- **Extended** (500+ tokens): Everything including history - for complex queries

Your chatbot can dynamically choose which to inject based on:
- Is Alice being mentioned? → Micro
- Is Alice in the conversation? → Standard
- User asks "tell me about Alice" → Extended

### **Pattern 2: Dynamic Profile Updates**

Profiles should stay fresh:

- **Trigger regeneration when:**
  - New high-confidence facts extracted
  - Person hasn't been active in 30+ days (mark as less active)
  - Significant fact changes (job change, location move)
  
- **Incremental updates:**
  - Instead of full regeneration, update specific sections
  - Append recent facts to "Recent Context" section

### **Pattern 3: Context-Aware Profiles**

Tailor profiles to conversation context:

```python
def generate_profile(person_id, context=None):
    facts = fetch_all_facts(person_id)
    
    if context == "technical_discussion":
        # Emphasize skills, work, projects
        return technical_profile(facts)
    
    elif context == "social":
        # Emphasize interests, relationships, activities
        return social_profile(facts)
    
    else:
        # Balanced general profile
        return general_profile(facts)
```

### **Pattern 4: Comparative Context**

For group conversations, add relational context:

```
Alice works at Google. She and Bob both work in tech and often discuss 
distributed systems. She's close friends with Carol, who she met at 
Stanford.
```

This helps the chatbot understand group dynamics.

## Specific Implementation Sketch

Here's how I'd structure the code:

### **1. Profile Generator Class**

```python
class ProfileGenerator:
    def __init__(self, neo4j_driver, llm_client, config):
        self.neo4j = neo4j_driver
        self.llm = llm_client
        self.config = config  # token limits, scoring weights, etc.
    
    def generate_profile(self, person_id, profile_type="standard"):
        # Aggregate facts from Neo4j
        facts = self.fetch_facts(person_id)
        
        # Score and rank
        ranked_facts = self.rank_facts(facts)
        
        # Select facts based on profile type and token budget
        selected_facts = self.select_facts(ranked_facts, profile_type)
        
        # Generate narrative
        profile = self.create_narrative(person_id, selected_facts)
        
        # Add metadata
        return ProfileResult(
            person_id=person_id,
            profile_text=profile,
            facts_used=selected_facts,
            generated_at=datetime.now(),
            token_count=count_tokens(profile)
        )
```

### **2. Fact Fetching Query**

```cypher
// Get comprehensive person data
MATCH (p:Person {id: $person_id})
OPTIONAL MATCH (p)-[r]->(target)
WHERE r.factId IS NOT NULL
WITH p, type(r) as rel_type, r, target,
     CASE 
       WHEN target:Org THEN target.name
       WHEN target:Place THEN target.label
       WHEN target:Topic THEN target.name
       WHEN target:Person THEN COALESCE(target.realName, target.id)
       WHEN target:Skill THEN target.name
       ELSE null
     END as target_label
RETURN p, rel_type, properties(r) as rel_props, target_label
ORDER BY r.confidence DESC, r.lastUpdated DESC
```

### **3. Narrative Generation Prompt**

```python
def create_narrative(self, person_id, selected_facts):
    person_info = self.get_basic_info(person_id)
    
    prompt = f"""
    Create a natural, coherent profile for {person_info.name} based on these facts.
    Write in third person, present tense for current information.
    Be concise but informative. Target 200-250 tokens.
    
    Facts organized by category:
    
    PROFESSIONAL:
    {self.format_facts(selected_facts['professional'])}
    
    INTERESTS:
    {self.format_facts(selected_facts['interests'])}
    
    SOCIAL:
    {self.format_facts(selected_facts['social'])}
    
    PREFERENCES:
    {self.format_facts(selected_facts['preferences'])}
    
    Instructions:
    - Start with their current primary role/status
    - Flow naturally between topics
    - Use specific details (company names, skills, etc.)
    - Mention timeframes when relevant (e.g., "currently", "previously")
    - Don't include every fact - synthesize into key themes
    - End with their recent focus or active interests
    """
    
    return self.llm.complete(prompt)
```

### **4. Profile Caching Strategy**

```python
class ProfileCache:
    def get_or_generate(self, person_id, max_age_hours=24):
        cached = self.fetch_from_cache(person_id)
        
        if cached and not self.is_stale(cached, max_age_hours):
            # Check if facts changed since cache
            if not self.has_new_facts(person_id, cached.generated_at):
                return cached.profile_text
        
        # Generate fresh profile
        profile = self.generator.generate_profile(person_id)
        self.store_in_cache(person_id, profile)
        return profile.profile_text
```

## Integration with Your Chatbot

### **System Prompt Injection**

```python
def build_system_prompt(conversation_context):
    participants = conversation_context.participants
    
    profiles = []
    for person_id in participants:
        # Use micro profiles for all, standard for main participants
        profile_type = "standard" if is_active_participant(person_id) else "micro"
        profile = profile_cache.get_or_generate(person_id, profile_type)
        profiles.append(profile)
    
    system_prompt = f"""
    You are a helpful assistant in a Discord server with the following members:
    
    {chr(10).join(profiles)}
    
    Use this information to personalize your responses and maintain context
    about what people do, care about, and prefer. Reference their interests
    and expertise naturally when relevant.
    """
    
    return system_prompt
```

### **Dynamic Profile Loading**

For token efficiency, only load full profiles when needed:

```python
# Minimal context always present
base_context = get_micro_profiles(all_members)

# Load detailed profile only when person is directly involved
if message.mentions_user(user_id):
    detailed_profile = get_standard_profile(user_id)
    context = merge_contexts(base_context, detailed_profile)
```

## Quality Considerations

1. **Fact Verification**: Before generating, validate facts aren't contradictory
2. **Tone Calibration**: Match the formality to your Discord's culture (casual vs professional)
3. **Privacy Filtering**: Exclude sensitive fact types if configured
4. **Staleness Indicators**: Mark profiles as "last updated 3 months ago" if stale
5. **Uncertainty Handling**: Use hedging language for lower-confidence facts ("appears to work at", "mentioned interest in")

## Next Steps

Once you have profile generation working, you can extend to:
- **Relationship summaries**: "Alice and Bob frequently collaborate on Rust projects"
- **Group profiles**: "The backend team consists of Alice, Bob, and Carol..."
- **Temporal profiles**: "Alice in 2023 vs Alice in 2025" showing evolution
- **Topic expert identification**: "For questions about Kubernetes, Alice is most knowledgeable"
