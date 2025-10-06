# Fact Category Prompt Templates Specification

## Overview

This document specifies a system of composable prompt templates that adapt the deduplication prompt to the specific semantics of each fact type. Rather than using a one-size-fits-all approach, each fact category gets specialized guidance, examples, and rules.

---

## Motivation

Different fact types have fundamentally different merge semantics:

| Fact Type | Merge Behavior | Key Challenge |
|-----------|---------------|---------------|
| `WORKS_AT` | Temporal-sensitive | Past vs present jobs must stay separate |
| `TALKS_ABOUT` | Aggregation-friendly | Multiple mentions should merge |
| `LIVES_IN` | Temporal-sensitive | Moving between cities = separate facts |
| `HAS_SKILL` | Evolving | Skill level changes over time |
| `PREFERS` | Stable | Preferences rarely change |
| `ATTENDED_EVENT` | Instance-based | Same event attended multiple times |
| `RELATED_TO` | Identity-based | Relationships are unique by person pair |

**Current Problem**: Single generic prompt leads to:
- Over-merging temporal facts (losing history)
- Under-merging aggregable facts (creating clutter)
- Inconsistent confidence calibration across types
- Poor attribute prioritization

**Solution**: Category-specific templates that inject type-aware guidance.

---

## Architecture

### Template Composition Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Deduplication Prompt                 │
│  - System context (purpose, tradeoffs)                       │
│  - General merge principles                                  │
│  - Output schema definition                                  │
│  - Quality validation checklist                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Fact Category Template (Selected)               │
│  - Category-specific merge rules                             │
│  - Attribute priority hierarchy                              │
│  - Confidence calibration guidance                           │
│  - Edge case handling                                        │
│  - Type-specific few-shot examples (3-5)                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Input Facts (JSON)                        │
│  - Candidate facts for current partition                     │
└─────────────────────────────────────────────────────────────┘
```

### Template Selection Logic

```python
def select_template(fact_type: FactType) -> PromptTemplate:
    """Select the appropriate template based on fact category."""
    category = FACT_TYPE_CATEGORIES[fact_type]
    return CATEGORY_TEMPLATES[category]
```

---

## Fact Type Categorization

Group the 23 fact types into 6 categories based on merge semantics:

### Category 1: Temporal Employment/Education (4 types)
**Semantics**: Position in time matters; different periods = different facts

- `WORKS_AT`: Employment relationships
- `STUDIED_AT`: Educational history
- `WORKING_ON`: Active projects
- `PREVIOUSLY`: Historical facts with explicit end dates

**Key Principle**: Do NOT merge facts spanning different time periods

---

### Category 2: Temporal Location/Status (2 types)
**Semantics**: Location/status changes over time

- `LIVES_IN`: Residency
- `EXPERIENCED`: Life events with stages

**Key Principle**: Current vs past must remain separate

---

### Category 3: Aggregable Topics/Interests (5 types)
**Semantics**: Multiple mentions should consolidate; sentiment/frequency matters

- `TALKS_ABOUT`: Topic discussions
- `CURIOUS_ABOUT`: Learning interests  
- `CARES_ABOUT`: Values and causes
- `REMEMBERS`: Memories and reflections
- `WITNESSED`: Firsthand observations

**Key Principle**: Merge freely; union evidence to show frequency

---

### Category 4: Stable Skills/Capabilities (2 types)
**Semantics**: Proficiency evolves but entity is stable

- `HAS_SKILL`: Technical and professional skills
- `BELIEVES`: Stances and opinions

**Key Principle**: Update proficiency/conviction; keep skill identity

---

### Category 5: Preference/Sentiment (6 types)
**Semantics**: Subject-object-sentiment triples; sentiment matters

- `PREFERS`: Positive preferences
- `RECOMMENDS`: Endorsements
- `AVOIDS`: Things avoided
- `DISLIKES`: Negative sentiment
- `ENJOYS`: Activities enjoyed
- `PLANS_TO`: Future intentions

**Key Principle**: Same target with different sentiment = separate facts

---

### Category 6: Relationships/Events (4 types)
**Semantics**: Identity-based; unique by entity pair or event instance

- `CLOSE_TO`: Social connections
- `RELATED_TO`: Family/personal relationships
- `ATTENDED_EVENT`: Event participation
- `RECOMMENDS`: Entity recommendations

**Key Principle**: Each unique relationship/event instance is separate

---

## Template Structure

Each template is a Pydantic model with standardized sections:

```python
@dataclass
class CategoryTemplate:
    """Template for a specific fact category."""
    
    category_name: str
    fact_types: list[FactType]
    
    # Core merge semantics
    merge_philosophy: str  # 1-2 sentence principle
    temporal_sensitivity: Literal["high", "medium", "low", "none"]
    default_merge_strategy: Literal["conservative", "moderate", "aggressive"]
    
    # Attribute handling
    critical_attributes: list[str]  # Conflicts prevent merge
    important_attributes: list[str]  # Normalize/combine
    optional_attributes: list[str]   # Fill from best source
    attribute_normalization_rules: dict[str, str]
    
    # Confidence calibration
    identical_confidence_rule: str
    complementary_confidence_rule: str
    conflict_handling: str
    
    # Edge cases
    edge_cases: list[EdgeCaseGuidance]
    
    # Examples
    few_shot_examples: list[FewShotExample]
    
    def render(self) -> str:
        """Render template to markdown for inclusion in prompt."""
        ...
```

---

## Complete Template Examples

### Template 1: Temporal Employment/Education

```python
TEMPORAL_EMPLOYMENT_TEMPLATE = CategoryTemplate(
    category_name="Temporal Employment/Education",
    fact_types=[FactType.WORKS_AT, FactType.STUDIED_AT, FactType.WORKING_ON],
    
    merge_philosophy="""
    Employment and education facts are HIGHLY TEMPORAL. Different time periods 
    represent different facts. Only merge when describing the SAME position/program 
    during the SAME time period.
    """,
    
    temporal_sensitivity="high",
    default_merge_strategy="conservative",
    
    critical_attributes=[
        "organization/institution",
        "start_date", 
        "end_date"
    ],
    
    important_attributes=[
        "role/degree_type",
        "location/field_of_study"
    ],
    
    optional_attributes=[
        "employment_type",
        "graduation_year"
    ],
    
    attribute_normalization_rules={
        "organization": "Normalize company names: 'Google Inc.' → 'Google', 'FB' → 'Facebook'",
        "role": "Expand abbreviations: 'SWE' → 'Software Engineer', 'PM' → 'Product Manager'",
        "location": "Use full city names: 'SF' → 'San Francisco', 'NYC' → 'New York'",
        "institution": "Use official names: 'Stanford' → 'Stanford University'",
    },
    
    identical_confidence_rule="""
    Use MAX(confidences) when facts are truly identical:
    - Same organization AND role AND time period
    - Differences only in formatting/abbreviation
    """,
    
    complementary_confidence_rule="""
    Use WEIGHTED_AVG + 0.05 when facts add complementary details:
    - Same organization AND time period
    - One has role, other has location
    - Facts fill in each other's gaps
    """,
    
    conflict_handling="""
    DO NOT MERGE if:
    - Different organizations (even if same time period)
    - start_date/end_date differ by >3 months
    - Conflicting roles that can't be the same position
    
    When in doubt about temporal overlap, KEEP SEPARATE.
    """,
    
    edge_cases=[
        EdgeCaseGuidance(
            scenario="Promotion within same company",
            description="Same org, different roles, overlapping dates",
            resolution="""
            If role change represents promotion/transfer and dates are contiguous:
            - Create separate facts for each role period
            - Do NOT merge even though same organization
            Example: 'Engineer 2020-2022' + 'Senior Engineer 2022-2024' → 2 facts
            """
        ),
        EdgeCaseGuidance(
            scenario="Part-time during school",
            description="Concurrent WORKS_AT and STUDIED_AT",
            resolution="""
            These are DIFFERENT fact types, so will appear in different partitions.
            No merge conflict—each partition processes independently.
            """
        ),
        EdgeCaseGuidance(
            scenario="Company acquisition",
            description="'Startup X' acquired by 'BigCo', both mentioned",
            resolution="""
            If dates suggest same position through acquisition:
            - Use pre-acquisition name in object_label
            - Note acquisition in merge_reasoning
            - Keep as single fact with earliest start_date
            """
        ),
        EdgeCaseGuidance(
            scenario="Fuzzy dates",
            description="One fact has exact date, other has '~2020' or 'around 2020'",
            resolution="""
            If fuzzy date overlaps with exact date (within ±6 months):
            - Merge, using exact date
            - Note date resolution in merge_reasoning
            - Reduce confidence by 0.05 due to ambiguity
            """
        ),
    ],
    
    few_shot_examples=[
        FewShotExample(
            name="Clear duplicate with normalization",
            input_facts=[
                {"fact_id": 1, "object_label": "Google", "role": "SWE", "location": "SF", 
                 "start_date": "2022-03", "confidence": 0.82, "evidence": ["msg_045"]},
                {"fact_id": 2, "object_label": "Google Inc.", "role": "Software Engineer", 
                 "location": "San Francisco", "start_date": "2022-03", "confidence": 0.78, 
                 "evidence": ["msg_102"]},
            ],
            output_facts=[
                {
                    "object_label": "Google",
                    "attributes": {
                        "organization": "Google",
                        "role": "Software Engineer",
                        "location": "San Francisco",
                        "start_date": "2022-03"
                    },
                    "confidence": 0.85,
                    "evidence": ["msg_045", "msg_102"],
                    "merged_from": [1, 2],
                    "merge_reasoning": """Merged 2 facts about same Google employment starting March 2022. 
Normalized organization name ('Google Inc.' → 'Google'), expanded role abbreviation ('SWE' → 'Software Engineer'), 
and standardized location ('SF' → 'San Francisco'). Used weighted average confidence + 0.05 for complementary 
details = 0.85."""
                }
            ],
            explanation="Perfect duplicate with only formatting differences."
        ),
        
        FewShotExample(
            name="Temporal conflict - different periods",
            input_facts=[
                {"fact_id": 3, "object_label": "Microsoft", "role": "Engineer", 
                 "start_date": "2018-06", "end_date": "2020-08", "confidence": 0.88, 
                 "evidence": ["msg_203"]},
                {"fact_id": 4, "object_label": "Microsoft", "role": "Senior Engineer",
                 "start_date": "2020-09", "confidence": 0.85, "evidence": ["msg_301"]},
            ],
            output_facts=[
                {
                    "object_label": "Microsoft",
                    "attributes": {"organization": "Microsoft", "role": "Engineer", 
                                   "start_date": "2018-06", "end_date": "2020-08"},
                    "confidence": 0.88,
                    "evidence": ["msg_203"],
                    "merged_from": [3],
                    "merge_reasoning": "First Microsoft role (Engineer) from June 2018 to August 2020."
                },
                {
                    "object_label": "Microsoft",
                    "attributes": {"organization": "Microsoft", "role": "Senior Engineer",
                                   "start_date": "2020-09"},
                    "confidence": 0.85,
                    "evidence": ["msg_301"],
                    "merged_from": [4],
                    "merge_reasoning": "Second Microsoft role (Senior Engineer) starting September 2020, likely a promotion."
                }
            ],
            explanation="Different time periods + role change = separate facts (promotion/progression)."
        ),
        
        FewShotExample(
            name="Complementary details - same period",
            input_facts=[
                {"fact_id": 5, "object_label": "Stripe", "role": "Engineer", 
                 "start_date": "2021-01", "confidence": 0.75, "evidence": ["msg_150"]},
                {"fact_id": 6, "object_label": "Stripe", "location": "San Francisco",
                 "start_date": "2021-01", "confidence": 0.70, "evidence": ["msg_175"]},
            ],
            output_facts=[
                {
                    "object_label": "Stripe",
                    "attributes": {
                        "organization": "Stripe",
                        "role": "Engineer",
                        "location": "San Francisco",
                        "start_date": "2021-01"
                    },
                    "confidence": 0.78,
                    "evidence": ["msg_150", "msg_175"],
                    "merged_from": [5, 6],
                    "merge_reasoning": """Merged 2 facts about same Stripe employment starting January 2021. 
Combined role from Fact 5 with location from Fact 6. Confidence = weighted_avg(0.75, 0.70) + 0.05 = 0.78."""
                }
            ],
            explanation="Same org, same period, complementary non-conflicting attributes."
        ),
        
        FewShotExample(
            name="Ambiguous dates - conservative split",
            input_facts=[
                {"fact_id": 7, "object_label": "Amazon", "role": "SDE", 
                 "start_date": "2019-08", "confidence": 0.82, "evidence": ["msg_220"]},
                {"fact_id": 8, "object_label": "Amazon", "role": "Software Engineer",
                 "start_date": "2020-02", "confidence": 0.68, "evidence": ["msg_245"]},
            ],
            output_facts=[
                {
                    "object_label": "Amazon",
                    "attributes": {"organization": "Amazon", "role": "Software Engineer", 
                                   "start_date": "2019-08"},
                    "confidence": 0.82,
                    "evidence": ["msg_220"],
                    "merged_from": [7],
                    "merge_reasoning": "Amazon SDE role starting August 2019."
                },
                {
                    "object_label": "Amazon",
                    "attributes": {"organization": "Amazon", "role": "Software Engineer",
                                   "start_date": "2020-02"},
                    "confidence": 0.68,
                    "evidence": ["msg_245"],
                    "merged_from": [8],
                    "merge_reasoning": "Amazon Software Engineer role starting February 2020."
                }
            ],
            explanation="Start dates differ by 6 months. Could be same role with fuzzy recall, or could be re-hire/transfer. Conservative approach: keep separate."
        ),
    ],
)
```

---

### Template 2: Aggregable Topics/Interests

```python
AGGREGABLE_TOPICS_TEMPLATE = CategoryTemplate(
    category_name="Aggregable Topics/Interests",
    fact_types=[FactType.TALKS_ABOUT, FactType.CURIOUS_ABOUT, FactType.CARES_ABOUT],
    
    merge_philosophy="""
    Topic and interest facts are AGGREGATION-FRIENDLY. Multiple mentions of the same 
    topic/interest should generally merge to show frequency and sustained interest. 
    Evidence union is valuable here.
    """,
    
    temporal_sensitivity="low",
    default_merge_strategy="moderate",
    
    critical_attributes=["topic"],
    important_attributes=["sentiment", "intensity"],
    optional_attributes=["context", "related_actions"],
    
    attribute_normalization_rules={
        "topic": "Normalize topic names: 'ML' → 'Machine Learning', 'k8s' → 'Kubernetes'",
        "sentiment": "Aggregate sentiments: if all positive/negative, preserve; if mixed, mark 'mixed'",
    },
    
    identical_confidence_rule="""
    Use MAX(confidences) + 0.05 for repeated mentions:
    - Shows sustained interest through multiple messages
    - More evidence = higher confidence in true interest
    """,
    
    complementary_confidence_rule="""
    Use MAX(confidences) + 0.10 when facts provide different aspects:
    - One mentions topic, another provides sentiment
    - Evidence spans different time periods (sustained interest)
    """,
    
    conflict_handling="""
    Sentiment conflicts require careful handling:
    - If sentiments opposite (positive vs negative): KEEP SEPARATE
    - If sentiments compatible (neutral + positive): MERGE with combined sentiment
    - Document sentiment evolution in merge_reasoning
    """,
    
    edge_cases=[
        EdgeCaseGuidance(
            scenario="Related but distinct topics",
            description="'Python' vs 'Django' or 'AI' vs 'Machine Learning'",
            resolution="""
            General rule: Keep separate unless explicitly synonymous.
            - 'AI' and 'Artificial Intelligence' → MERGE
            - 'AI' and 'Machine Learning' → KEEP SEPARATE (ML is subset)
            - 'Python' and 'Django' → KEEP SEPARATE (different specificity)
            """
        ),
        EdgeCaseGuidance(
            scenario="Sentiment evolution",
            description="Earlier messages positive, later messages neutral/negative",
            resolution="""
            If sentiment clearly changed over time:
            - Keep as separate facts if evidence dates support distinct periods
            - If all within ~1 month, merge with 'mixed' or dominant sentiment
            - Note evolution in merge_reasoning
            """
        ),
        EdgeCaseGuidance(
            scenario="High frequency mention",
            description="Topic appears in 10+ messages",
            resolution="""
            This is GOOD—shows strong, sustained interest.
            - Merge all mentions into single canonical fact
            - Include all message IDs in evidence (show frequency)
            - Boost confidence: MAX(confidences) + 0.15 (capped at 0.95)
            - Note in reasoning: "Strong sustained interest across N messages"
            """
        ),
    ],
    
    few_shot_examples=[
        FewShotExample(
            name="Multiple mentions of same topic",
            input_facts=[
                {"fact_id": 10, "object_label": "Rust programming", "topic": "Rust", 
                 "sentiment": "positive", "confidence": 0.72, "evidence": ["msg_305"]},
                {"fact_id": 11, "object_label": "Rust", "topic": "Rust", 
                 "confidence": 0.68, "evidence": ["msg_401"]},
                {"fact_id": 12, "object_label": "Rust lang", "topic": "Rust", 
                 "sentiment": "enthusiastic", "confidence": 0.75, "evidence": ["msg_502"]},
            ],
            output_facts=[
                {
                    "object_label": "Rust",
                    "attributes": {"topic": "Rust", "sentiment": "positive"},
                    "confidence": 0.88,
                    "evidence": ["msg_305", "msg_401", "msg_502"],
                    "merged_from": [10, 11, 12],
                    "merge_reasoning": """Merged 3 mentions of Rust programming interest. 
Normalized topic labels ('Rust programming'/'Rust lang' → 'Rust'). Consistent positive sentiment 
across all mentions. High confidence (0.88) due to sustained interest across multiple messages 
(MAX(0.75) + 0.10 + evidence bonus 0.03 = 0.88)."""
                }
            ],
            explanation="Classic aggregation case—merge all mentions."
        ),
        
        FewShotExample(
            name="Sentiment conflict",
            input_facts=[
                {"fact_id": 13, "object_label": "JavaScript", "topic": "JavaScript",
                 "sentiment": "frustrated", "confidence": 0.80, "evidence": ["msg_601", "msg_605"]},
                {"fact_id": 14, "object_label": "JavaScript", "topic": "JavaScript",
                 "sentiment": "positive", "confidence": 0.70, "evidence": ["msg_720"]},
            ],
            output_facts=[
                {
                    "object_label": "JavaScript",
                    "attributes": {"topic": "JavaScript", "sentiment": "frustrated"},
                    "confidence": 0.80,
                    "evidence": ["msg_601", "msg_605"],
                    "merged_from": [13],
                    "merge_reasoning": "Frustrated sentiment about JavaScript (earlier mentions)."
                },
                {
                    "object_label": "JavaScript",
                    "attributes": {"topic": "JavaScript", "sentiment": "positive"},
                    "confidence": 0.70,
                    "evidence": ["msg_720"],
                    "merged_from": [14],
                    "merge_reasoning": "Positive sentiment about JavaScript (later mention, possible sentiment shift)."
                }
            ],
            explanation="Opposing sentiments + temporal separation = separate facts."
        ),
    ],
)
```

---

### Template 3: Relationships/Events

```python
RELATIONSHIPS_EVENTS_TEMPLATE = CategoryTemplate(
    category_name="Relationships/Events",
    fact_types=[FactType.CLOSE_TO, FactType.RELATED_TO, FactType.ATTENDED_EVENT],
    
    merge_philosophy="""
    Relationships and events are IDENTITY-BASED. Each unique relationship (person pair) 
    or event instance should remain separate. Merge only when describing the exact same 
    relationship/event with formatting variations.
    """,
    
    temporal_sensitivity="medium",
    default_merge_strategy="conservative",
    
    critical_attributes=["other_person_id", "event_name", "relationship_type"],
    important_attributes=["basis", "date", "role"],
    optional_attributes=["format", "location"],
    
    attribute_normalization_rules={
        "relationship_type": "Use canonical terms: 'bro' → 'brother', 'sis' → 'sister'",
        "event_name": "Use official event names when known",
    },
    
    identical_confidence_rule="""
    Use MAX(confidences) when describing same relationship/event:
    - Same person pair or event instance
    - Differences only in how it's described
    """,
    
    complementary_confidence_rule="""
    Use WEIGHTED_AVG + 0.05 when facts add relationship/event details:
    - One mentions relationship type, other provides basis
    - One has event name, other has date/location
    """,
    
    conflict_handling="""
    DO NOT MERGE if:
    - Different person pairs (even if same relationship type)
    - Different event instances (same conference different years)
    - Conflicting relationship types that can't both be true
    """,
    
    edge_cases=[
        EdgeCaseGuidance(
            scenario="Same event, multiple years",
            description="'PyCon 2023' vs 'PyCon 2024'",
            resolution="""
            These are DIFFERENT event instances:
            - Keep separate (even though same conference series)
            - Each attendance is a distinct fact
            """
        ),
        EdgeCaseGuidance(
            scenario="Multiple mentions of same relationship",
            description="'Alice is my sister' appears in 3 messages",
            resolution="""
            This is the SAME relationship mentioned multiple times:
            - Merge into single fact
            - Union all evidence
            - Higher confidence due to repeated confirmation
            """
        ),
        EdgeCaseGuidance(
            scenario="Relationship evolution",
            description="'Alice is my girlfriend' → 'Alice is my wife'",
            resolution="""
            Different relationship types with temporal progression:
            - If dates clearly indicate progression: Keep separate (status changed)
            - If concurrent mentions: Choose most authoritative based on confidence
            - Document relationship evolution in merge_reasoning
            """
        ),
    ],
    
    few_shot_examples=[
        FewShotExample(
            name="Same relationship, multiple mentions",
            input_facts=[
                {"fact_id": 20, "object_id": "person_456", "object_label": "Bob",
                 "relationship_type": "brother", "confidence": 0.90, "evidence": ["msg_801"]},
                {"fact_id": 21, "object_id": "person_456", "object_label": "Bob Smith",
                 "relationship_type": "sibling", "confidence": 0.75, "evidence": ["msg_850"]},
            ],
            output_facts=[
                {
                    "object_id": "person_456",
                    "object_label": "Bob Smith",
                    "attributes": {"relationship_type": "brother"},
                    "confidence": 0.92,
                    "evidence": ["msg_801", "msg_850"],
                    "merged_from": [20, 21],
                    "merge_reasoning": """Merged 2 mentions of brother relationship with Bob (person_456). 
Normalized relationship type ('sibling' → 'brother' for male sibling). Used full name 'Bob Smith' 
from more complete mention. High confidence (0.92) due to repeated confirmation (MAX(0.90) + 0.02)."""
                }
            ],
            explanation="Same person, same relationship, slight terminology variation."
        ),
        
        FewShotExample(
            name="Different people, same relationship type",
            input_facts=[
                {"fact_id": 22, "object_id": "person_789", "object_label": "Carol",
                 "relationship_type": "coworker", "confidence": 0.80, "evidence": ["msg_901"]},
                {"fact_id": 23, "object_id": "person_890", "object_label": "Dave",
                 "relationship_type": "coworker", "confidence": 0.78, "evidence": ["msg_925"]},
            ],
            output_facts=[
                {
                    "object_id": "person_789",
                    "object_label": "Carol",
                    "attributes": {"relationship_type": "coworker"},
                    "confidence": 0.80,
                    "evidence": ["msg_901"],
                    "merged_from": [22],
                    "merge_reasoning": "Coworker relationship with Carol (person_789)."
                },
                {
                    "object_id": "person_890",
                    "object_label": "Dave",
                    "attributes": {"relationship_type": "coworker"},
                    "confidence": 0.78,
                    "evidence": ["msg_925"],
                    "merged_from": [23],
                    "merge_reasoning": "Coworker relationship with Dave (person_890)."
                }
            ],
            explanation="Different people = different facts (even though same relationship type)."
        ),
    ],
)
```

---

## Implementation Guide

### Step 1: Define Template Registry

```python
# deduplicate/templates/__init__.py

from typing import Mapping
from ie.types import FactType
from .models import CategoryTemplate
from .temporal_employment import TEMPORAL_EMPLOYMENT_TEMPLATE
from .aggregable_topics import AGGREGABLE_TOPICS_TEMPLATE
from .relationships_events import RELATIONSHIPS_EVENTS_TEMPLATE
from .stable_skills import STABLE_SKILLS_TEMPLATE
from .preference_sentiment import PREFERENCE_SENTIMENT_TEMPLATE
from .temporal_location import TEMPORAL_LOCATION_TEMPLATE

# Map each fact type to its category template
FACT_TYPE_TO_TEMPLATE: Mapping[FactType, CategoryTemplate] = {
    # Temporal Employment/Education
    FactType.WORKS_AT: TEMPORAL_EMPLOYMENT_TEMPLATE,
    FactType.STUDIED_AT: TEMPORAL_EMPLOYMENT_TEMPLATE,
    FactType.WORKING_ON: TEMPORAL_EMPLOYMENT_TEMPLATE,
    FactType.PREVIOUSLY: TEMPORAL_EMPLOYMENT_TEMPLATE,
    
    # Temporal Location/Status
    FactType.LIVES_IN: TEMPORAL_LOCATION_TEMPLATE,
    FactType.EXPERIENCED: TEMPORAL_LOCATION_TEMPLATE,
    
    # Aggregable Topics/Interests
    FactType.TALKS_ABOUT: AGGREGABLE_TOPICS_TEMPLATE,
    FactType.CURIOUS_ABOUT: AGGREGABLE_TOPICS_TEMPLATE,
    FactType.CARES_ABOUT: AGGREGABLE_TOPICS_TEMPLATE,
    FactType.REMEMBERS: AGGREGABLE_TOPICS_TEMPLATE,
    FactType.WITNESSED: AGGREGABLE_TOPICS_TEMPLATE,
    
    # Stable Skills/Capabilities
    FactType.HAS_SKILL: STABLE_SKILLS_TEMPLATE,
    FactType.BELIEVES: STABLE_SKILLS_TEMPLATE,
    
    # Preference/Sentiment
    FactType.PREFERS: PREFERENCE_SENTIMENT_TEMPLATE,
    FactType.RECOMMENDS: PREFERENCE_SENTIMENT_TEMPLATE,
    FactType.AVOIDS: PREFERENCE_SENTIMENT_TEMPLATE,
    FactType.DISLIKES: PREFERENCE_SENTIMENT_TEMPLATE,
    FactType.ENJOYS: PREFERENCE_SENTIMENT_TEMPLATE,
    FactType.PLANS_TO: PREFERENCE_SENTIMENT_TEMPLATE,
    
    # Relationships/Events
    FactType.CLOSE_TO: RELATIONSHIPS_EVENTS_TEMPLATE,
    FactType.RELATED_TO: RELATIONSHIPS_EVENTS_TEMPLATE,
    FactType.ATTENDED_EVENT: RELATIONSHIPS_EVENTS_TEMPLATE,
}

def get_template(fact_type: FactType) -> CategoryTemplate:
    """Retrieve the appropriate template for a fact type."""
    return FACT_TYPE_TO_TEMPLATE[fact_type]
```

---

### Step 2: Template Rendering

```python
# deduplicate/templates/models.py

from dataclasses import dataclass
from typing import Literal

@dataclass
class EdgeCaseGuidance:
    scenario: str
    description: str
    resolution: str
    
    def render(self) -> str:
        return f"""
**{self.scenario}**
{self.description}

Resolution:
{self.resolution}
""".strip()


@dataclass
class FewShotExample:
    name: str
    input_facts: list[dict]
    output_facts: list[dict]
    explanation: str
    
    def render(self) -> str:
        import json
        input_json = json.dumps({"candidate_facts": self.input_facts}, indent=2)
        output_json = json.dumps({"canonical_facts": self.output_facts}, indent=2)
        
        return f"""
### Example: {self.name}

**Input:**
```json
{input_json}
```

**Output:**
```json
{output_json}
```

**Explanation:** {self.explanation}
""".strip()


@dataclass
class CategoryTemplate:
    category_name: str
    fact_types: list[FactType]
    merge_philosophy: str
    temporal_sensitivity: Literal["high", "medium", "low", "none"]
    default_merge_strategy: Literal["conservative", "moderate", "aggressive"]
    
    critical_attributes: list[str]
    important_attributes: list[str]
    optional_attributes: list[str]
    attribute_normalization_rules: dict[str, str]
    
    identical_confidence_rule: str
    complementary_confidence_rule: str
    conflict_handling: str
    
    edge_cases: list[EdgeCaseGuidance]
    few_shot_examples: list[FewShotExample]
    
    def render(self) -> str:
        """Render complete template section for prompt."""
        
        # Attribute priorities
        attr_section = f"""
## Attribute Priority

**Critical** (conflicts prevent merge):
{', '.join(f'`{attr}`' for attr in self.critical_attributes)}

**Important** (normalize/combine):
{', '.join(f'`{attr}`' for attr in self.important_attributes)}

**Optional** (use best source):
{', '.join(f'`{attr}`' for attr in self.optional_attributes)}

### Normalization Rules
{chr(10).join(f"- **{key}**: {value}" for key, value in self.attribute_normalization_rules.items())}
"""
        
        # Confidence rules
        confidence_section = f"""
## Confidence Calculation

**Identical Facts:**
{self.identical_confidence_rule}

**Complementary Facts:**
{self.complementary_confidence_rule}

**Conflicts:**
{self.conflict_handling}
"""
        
        # Edge cases
        edge_case_section = "\n\n".join(
            case.render() for case in self.edge_cases
        )
        
        # Examples
        examples_section = "\n\n".join(
            example.render() for example in self.few_shot_examples
        )
        
        return f"""
# {self.category_name} Deduplication Guide

## Merge Philosophy
{self.merge_philosophy.strip()}

**Temporal Sensitivity:** {self.temporal_sensitivity.upper()}
**Default Strategy:** {self.default_merge_strategy.capitalize()}

{attr_section}

{confidence_section}

## Edge Cases

{edge_case_section}

## Examples

{examples_section}
""".strip()
```

---

### Step 3: Integrate with Prompt Builder

```python
# deduplicate/llm/prompts.py (updated)

from deduplicate.templates import get_template

def build_messages(
    partition: Partition, 
    candidate_facts: Iterable[FactRecord]
) -> list[dict[str, str]]:
    """Build LLM prompt with category-specific template."""
    
    # Get appropriate template
    template = get_template(partition.fact_type)
    template_content = template.render()
    
    # Format input facts
    facts_payload = [
        {
            "fact_id": fact.id,
            "object_label": fact.object_label,
            "object_id": fact.object_id,
            "object_type": fact.object_type,
            "attributes": fact.attributes,
            "confidence": fact.confidence,
            "evidence": fact.evidence,
            "timestamp": fact.timestamp,
        }
        for fact in candidate_facts
    ]
    
    request = {
        "task": "deduplicate_facts",
        "fact_type": partition.fact_type.value,
        "subject_id": partition.subject_id,
        "subject_name": partition.subject_name,
        "candidate_facts": facts_payload,
    }
    
    facts_json = json.dumps(request, indent=2, sort_keys=True)
    
    user_prompt = f"""
{template_content}

---

## Your Task

Review the candidate facts below and return canonical facts following the guidance above.

**Input Facts:**
```json
{facts_json}
```

**Output Format:**
Return strict JSON with structure:
```json
{{
  "canonical_facts": [
    {{
      "type": "{partition.fact_type.value}",
      "subject_id": "{partition.subject_id}",
      "object_label": "...",
      "object_id": "...",
      "object_type": "...",
      "attributes": {{}},
      "confidence": 0.0,
      "evidence": [],
      "timestamp": "...",
      "merged_from": [],
      "merge_reasoning": "..."
    }}
  ]
}}
```

Remember: Follow the {template.category_name} guidance above. When uncertain, prefer the {template.default_merge_strategy} approach.
"""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_templates.py

def test_temporal_employment_template_renders():
    """Verify template renders without errors."""
    template = TEMPORAL_EMPLOYMENT_TEMPLATE
    rendered = template.render()
    
    assert "Temporal Employment/Education" in rendered
    assert "WORKS_AT" in str(template.fact_types)
    assert len(template.few_shot_examples) >= 3
    assert "start_date" in template.critical_attributes


def test_template_selection():
    """Verify correct template selected for each fact type."""
    assert get_template(FactType.WORKS_AT) == TEMPORAL_EMPLOYMENT_TEMPLATE
    assert get_template(FactType.TALKS_ABOUT) == AGGREGABLE_TOPICS_TEMPLATE
    assert get_template(FactType.CLOSE_TO) == RELATIONSHIPS_EVENTS_TEMPLATE
```

### Integration Tests

```python
# tests/test_deduplication_with_templates.py

def test_works_at_merge_with_template(sqlite_db_with_facts):
    """Test WORKS_AT deduplication uses temporal-aware template."""
    config = DeduplicationConfig(
        sqlite_path=sqlite_db_with_facts,
        neo4j_password="test",
        dry_run=True,
    )
    
    orchestrator = DeduplicationOrchestrator(config)
    stats = orchestrator.run()
    
    # Verify temporal facts not over-merged
    conn = sqlite3.connect(sqlite_db_with_facts)
    facts = conn.execute(
        "SELECT * FROM fact WHERE type='WORKS_AT' AND subject_id='test_person'"
    ).fetchall()
    
    # Should preserve separate jobs at different companies/times
    assert len(facts) >= 2  # Not collapsed into 1


def test_talks_about_merge_with_template(sqlite_db_with_facts):
    """Test TALKS_ABOUT deduplication aggregates mentions."""
    config = DeduplicationConfig(
        sqlite_path=sqlite_db_with_facts,
        neo4j_password="test",
        dry_run=True,
    )
    
    orchestrator = DeduplicationOrchestrator(config)
    stats = orchestrator.run()
    
    # Verify topic mentions merged
    conn = sqlite3.connect(sqlite_db_with_facts)
    facts = conn.execute(
        "SELECT * FROM fact WHERE type='TALKS_ABOUT' AND subject_id='test_person' "
        "AND json_extract(attributes, '$.topic')='Rust'"
    ).fetchall()
    
    # Multiple mentions should collapse to 1 fact
    assert len(facts) == 1
    
    # Evidence should union all sources
    fact = facts[0]
    evidence_count = conn.execute(
        "SELECT COUNT(*) FROM fact_evidence WHERE fact_id=?",
        (fact["id"],)
    ).fetchone()[0]
    assert evidence_count >= 3  # Multiple message sources
```

---

## Rollout Plan

### Phase 1: Core Templates (Week 1)
- [ ] Implement template models and rendering system
- [ ] Create 3 category templates:
  - Temporal Employment/Education
  - Aggregable Topics/Interests
  - Relationships/Events
- [ ] Integrate template selection into prompt builder
- [ ] Write unit tests for template rendering

### Phase 2: Remaining Templates (Week 2)
- [ ] Create 3 additional category templates:
  - Temporal Location/Status
  - Stable Skills/Capabilities  
  - Preference/Sentiment
- [ ] Add 5 few-shot examples per template
- [ ] Document edge cases for each category
- [ ] Write integration tests

### Phase 3: Validation (Week 3)
- [ ] Run A/B test: generic prompt vs templated prompts
- [ ] Measure precision/recall per fact type
- [ ] Collect merge reasoning quality scores
- [ ] Identify template gaps and refine

### Phase 4: Optimization (Week 4)
- [ ] Tune confidence formulas per category
- [ ] Add more edge cases based on production failures
- [ ] Create template authoring guide for new fact types
- [ ] Document template maintenance process

---

## Template Authoring Guide

When adding a new fact type, follow this checklist:

### 1. Categorize the Fact Type
- [ ] Determine merge semantics (temporal, aggregable, identity-based)
- [ ] Identify closest existing category
- [ ] If no good fit, consider creating new category

### 2. Define Attributes
- [ ] List critical attributes (conflicts prevent merge)
- [ ] List important attributes (normalize/combine)
- [ ] List optional attributes (use best source)
- [ ] Write normalization rules for each attribute

### 3. Specify Confidence Rules
- [ ] Define when facts are "identical"
- [ ] Define when facts are "complementary"
- [ ] Specify conflict handling strategy
- [ ] Provide worked examples

### 4. Document Edge Cases
- [ ] Identify 3-5 common edge cases
- [ ] Provide clear resolution for each
- [ ] Include real-world examples

### 5. Create Few-Shot Examples
- [ ] Minimum 3 examples:
  - Clear duplicate (merge)
  - Non-duplicate (split)
  - Complementary (merge with details)
- [ ] Use realistic fact data
- [ ] Show complete input/output
- [ ] Explain reasoning

### 6. Test and Validate
- [ ] Unit test template renders correctly
- [ ] Integration test with real facts
- [ ] Measure precision/recall
- [ ] Collect LLM reasoning samples

---

## Maintenance

### Updating Templates

Templates should evolve based on production experience:

**When to Update:**
- LLM consistently mishandles specific edge case → Add edge case guidance
- New fact type added → Assign to category or create new template
- Precision/recall drifts → Adjust confidence rules or examples
- Merge reasoning quality drops → Add reasoning examples

**Update Process:**
1. Identify issue (low precision, poor reasoning, parse errors)
2. Review LLM responses for patterns
3. Draft template enhancement
4. Test on held-out dataset
5. Deploy if metrics improve
6. Document change in template changelog

### Template Versioning

```python
@dataclass
class CategoryTemplate:
    # ... existing fields ...
    version: str = "1.0"
    last_updated: str = "2025-01-15"
    changelog: list[str] = field(default_factory=list)
```

Track template evolution:
```python
TEMPORAL_EMPLOYMENT_TEMPLATE = CategoryTemplate(
    # ...
    version="1.2",
    last_updated="2025-02-10",
    changelog=[
        "v1.2 (2025-02-10): Added company acquisition edge case",
        "v1.1 (2025-01-28): Refined promotion handling, added fuzzy date example",
        "v1.0 (2025-01-15): Initial template",
    ]
)
```

---

## Success Metrics

Track these metrics per template category:

| Metric | Target | Measurement |
|--------|--------|-------------|
| Parse Success Rate | >95% | % responses that parse without retry |
| Merge Precision | >90% | % merges that are correct (manual audit) |
| Merge Recall | >80% | % true duplicates that are merged |
| Reasoning Quality | >4.0/5 | Human rating of merge_reasoning |
| Category Accuracy | >85% | % facts in category merge according to template philosophy |

---

## Conclusion

Category-specific prompt templates provide:

1. **Type-aware merge logic** - Temporal vs aggregable vs identity-based
2. **Consistent attribute handling** - Clear priority hierarchies
3. **Calibrated confidence** - Category-specific formulas
4. **Rich examples** - 3-5 few-shots per category showing edge cases
5. **Maintainability** - Centralized template definitions

This approach enables the deduplication system to handle the diverse semantics of 23 fact types without compromising precision or creating overly complex prompts.

**Expected Impact:**
- 15-20% improvement in category-specific precision
- 30-40% reduction in temporal over-merging
- 50%+ better reasoning quality
- Clearer path for adding new fact types
