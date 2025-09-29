# Discord Knowledge Extraction - Improvement Proposals

This document details four major improvements to enhance the accuracy, reliability, and effectiveness of the Discord knowledge extraction pipeline.

---

## 1. Fact Deduplication & Conflict Resolution

### What Is This?

Fact deduplication and conflict resolution is a system for handling cases where the LLM extracts multiple facts about the same relationship, potentially with conflicting information. This commonly occurs when:

- The same information is mentioned in multiple conversation windows
- Different conversations provide slightly different details about the same fact
- The LLM extracts the same fact with varying confidence scores
- Contradictory information exists (e.g., "I work at Google" followed weeks later by "I just started at Amazon")

### Why This Improves the Application

**Current Problem:** Without deduplication, the system creates multiple relationship edges in Neo4j for the same semantic fact, leading to:
- Graph clutter with redundant edges
- Confusion about which fact is "correct"
- Difficulty aggregating relationship strengths
- No temporal tracking of how facts evolve

**Benefits of Implementation:**
1. **Cleaner Graph:** One canonical edge per relationship type between entities
2. **Confidence Tracking:** Keep the highest-confidence version of each fact
3. **Temporal Awareness:** Track when facts change (job transitions, relocations)
4. **Better Analytics:** Accurate counts and weights for graph algorithms
5. **Evidence Accumulation:** Combine evidence from multiple extractions

### Recommended Implementation

#### High-Level Architecture

```
┌─────────────────┐
│  Raw Facts      │
│  from IE Run    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Fact Grouping          │
│  (by semantic key)      │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Conflict Resolution    │
│  Strategy Selection     │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Merged Fact Output     │
└─────────────────────────┘
```

#### Implementation Strategy

**Phase 1: Define Semantic Keys**

Each fact type needs a "semantic key" that identifies when two facts refer to the same relationship:

```python
# ie/deduplication.py

from dataclasses import dataclass
from typing import Protocol
import json

class FactKey(Protocol):
    """Protocol for creating semantic keys from facts."""
    def key(self, fact: FactRecord) -> tuple:
        """Return hashable key for grouping."""
        ...

@dataclass
class WorksAtKey:
    """Semantic key for WORKS_AT facts."""
    
    @staticmethod
    def key(fact: FactRecord) -> tuple:
        """Group by (person, organization), ignoring role changes."""
        org = fact.attributes.get("organization", "").strip().lower()
        return (fact.type, fact.subject_id, org)

@dataclass
class LivesInKey:
    """Semantic key for LIVES_IN facts."""
    
    @staticmethod
    def key(fact: FactRecord) -> tuple:
        """Group by (person, location)."""
        location = fact.attributes.get("location", "").strip().lower()
        return (fact.type, fact.subject_id, location)

@dataclass
class TalksAboutKey:
    """Semantic key for TALKS_ABOUT facts."""
    
    @staticmethod
    def key(fact: FactRecord) -> tuple:
        """Group by (person, topic)."""
        topic = fact.attributes.get("topic", "").strip().lower()
        return (fact.type, fact.subject_id, topic)

@dataclass
class CloseToKey:
    """Semantic key for CLOSE_TO facts."""
    
    @staticmethod
    def key(fact: FactRecord) -> tuple:
        """Bidirectional relationship - normalize order."""
        a, b = sorted([fact.subject_id, fact.object_id or ""])
        return (fact.type, a, b)

FACT_KEYS = {
    FactType.WORKS_AT: WorksAtKey.key,
    FactType.LIVES_IN: LivesInKey.key,
    FactType.TALKS_ABOUT: TalksAboutKey.key,
    FactType.CLOSE_TO: CloseToKey.key,
}
```

**Phase 2: Group Facts by Semantic Key**

```python
from collections import defaultdict

def group_facts(facts: list[FactRecord]) -> dict[tuple, list[FactRecord]]:
    """Group facts by their semantic keys."""
    groups = defaultdict(list)
    
    for fact in facts:
        key_fn = FACT_KEYS.get(fact.type)
        if not key_fn:
            # No deduplication strategy; keep as-is
            groups[(fact.type, fact.id)].append(fact)
            continue
        
        key = key_fn(fact)
        groups[key].append(fact)
    
    return dict(groups)
```

**Phase 3: Implement Conflict Resolution Strategies**

```python
from datetime import datetime
from enum import Enum

class MergeStrategy(Enum):
    """How to resolve conflicts between facts."""
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RECENT = "most_recent"
    ACCUMULATE_EVIDENCE = "accumulate_evidence"
    TEMPORAL_TRANSITION = "temporal_transition"

def merge_facts(
    fact_group: list[FactRecord],
    strategy: MergeStrategy = MergeStrategy.HIGHEST_CONFIDENCE
) -> list[FactRecord]:
    """
    Merge a group of semantically identical facts.
    
    Returns list because some strategies (TEMPORAL_TRANSITION) 
    may preserve multiple facts.
    """
    if len(fact_group) == 1:
        return fact_group
    
    if strategy == MergeStrategy.HIGHEST_CONFIDENCE:
        return [_merge_by_confidence(fact_group)]
    elif strategy == MergeStrategy.MOST_RECENT:
        return [_merge_by_timestamp(fact_group)]
    elif strategy == MergeStrategy.ACCUMULATE_EVIDENCE:
        return [_merge_accumulate_evidence(fact_group)]
    elif strategy == MergeStrategy.TEMPORAL_TRANSITION:
        return _detect_transitions(fact_group)
    
    return fact_group

def _merge_by_confidence(facts: list[FactRecord]) -> FactRecord:
    """Keep the fact with highest confidence; combine evidence."""
    best = max(facts, key=lambda f: (f.confidence, f.timestamp))
    
    # Accumulate all evidence
    all_evidence = []
    for fact in facts:
        all_evidence.extend(fact.evidence)
    
    return FactRecord(
        id=best.id,
        type=best.type,
        subject_id=best.subject_id,
        object_id=best.object_id,
        object_type=best.object_type,
        attributes=best.attributes,
        timestamp=best.timestamp,
        confidence=best.confidence,
        evidence=list(dict.fromkeys(all_evidence))  # Deduplicate
    )

def _merge_by_timestamp(facts: list[FactRecord]) -> FactRecord:
    """Keep the most recent fact; useful for facts that change."""
    latest = max(facts, key=lambda f: f.timestamp or "")
    
    # Accumulate evidence
    all_evidence = []
    for fact in facts:
        all_evidence.extend(fact.evidence)
    
    return FactRecord(
        id=latest.id,
        type=latest.type,
        subject_id=latest.subject_id,
        object_id=latest.object_id,
        object_type=latest.object_type,
        attributes=latest.attributes,
        timestamp=latest.timestamp,
        confidence=latest.confidence,
        evidence=list(dict.fromkeys(all_evidence))
    )

def _merge_accumulate_evidence(facts: list[FactRecord]) -> FactRecord:
    """
    Boost confidence based on multiple mentions.
    Use highest-confidence as base, but increase confidence if 
    the same fact appears multiple times.
    """
    base = max(facts, key=lambda f: f.confidence)
    
    # Confidence boost: log-scale based on number of mentions
    mention_count = len(facts)
    confidence_boost = min(0.15, 0.05 * (mention_count - 1))
    new_confidence = min(1.0, base.confidence + confidence_boost)
    
    all_evidence = []
    for fact in facts:
        all_evidence.extend(fact.evidence)
    
    return FactRecord(
        id=base.id,
        type=base.type,
        subject_id=base.subject_id,
        object_id=base.object_id,
        object_type=base.object_type,
        attributes=base.attributes,
        timestamp=base.timestamp,
        confidence=new_confidence,
        evidence=list(dict.fromkeys(all_evidence))
    )

def _detect_transitions(facts: list[FactRecord]) -> list[FactRecord]:
    """
    Detect temporal transitions (e.g., job changes).
    Keep multiple facts if they represent a timeline.
    """
    # Sort by timestamp
    sorted_facts = sorted(facts, key=lambda f: f.timestamp or "")
    
    # For WORKS_AT: detect if end_date in earlier fact aligns with 
    # start_date in later fact
    transitions = []
    
    for i, fact in enumerate(sorted_facts):
        if i == 0:
            transitions.append(fact)
            continue
        
        prev_fact = sorted_facts[i - 1]
        
        # Check if attributes suggest a transition
        prev_org = prev_fact.attributes.get("organization", "").lower()
        curr_org = fact.attributes.get("organization", "").lower()
        
        if prev_org != curr_org:
            # Different org = transition
            # Potentially set end_date on previous
            transitions.append(fact)
        else:
            # Same org, probably duplicate - merge
            continue
    
    return transitions if len(transitions) > 1 else [sorted_facts[-1]]
```

**Phase 4: Apply During Fact Storage**

Modify `ie/runner.py` to deduplicate before storing:

```python
# In run_ie_job, after collecting all facts from a window:

from ie.deduplication import group_facts, merge_facts, MergeStrategy

# Group facts by semantic keys
fact_groups = group_facts(all_extracted_facts)

# Merge each group
merged_facts = []
for group in fact_groups.values():
    # Choose strategy based on fact type
    if group[0].type == FactType.WORKS_AT:
        strategy = MergeStrategy.TEMPORAL_TRANSITION
    elif group[0].type == FactType.TALKS_ABOUT:
        strategy = MergeStrategy.ACCUMULATE_EVIDENCE
    else:
        strategy = MergeStrategy.HIGHEST_CONFIDENCE
    
    merged = merge_facts(group, strategy=strategy)
    merged_facts.extend(merged)

# Store merged facts
for fact in merged_facts:
    _upsert_fact(conn, run_id=run_id, ...)
```

**Phase 5: Schema Additions**

Add tracking for fact history:

```sql
-- Add to schema.sql

CREATE TABLE IF NOT EXISTS fact_history (
  id              INTEGER PRIMARY KEY,
  fact_id         INTEGER NOT NULL REFERENCES fact(id) ON DELETE CASCADE,
  previous_value  TEXT NOT NULL,  -- JSON of old attributes
  changed_at      TEXT NOT NULL,
  change_reason   TEXT  -- 'confidence_update', 'transition', 'merge'
);

-- Track which facts were merged
CREATE TABLE IF NOT EXISTS fact_merge (
  merged_fact_id  INTEGER NOT NULL REFERENCES fact(id),
  source_fact_id  INTEGER NOT NULL,
  merged_at       TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (merged_fact_id, source_fact_id)
);
```

---

## 2. Entity Normalization & Validation

### What Is This?

Entity normalization is the process of standardizing how organizations, locations, topics, and people are represented in the knowledge graph. Without normalization:

- "Google", "Google Inc", "Google LLC", "google" are treated as separate organizations
- "SF", "San Francisco", "San Francisco, CA" are different places
- "AI", "artificial intelligence", "A.I." are different topics
- Invalid person IDs slip through, creating broken relationships

Validation ensures that extracted entities actually exist in the system and meet quality standards before being added to the graph.

### Why This Improves the Application

**Current Problems:**
1. **Graph Fragmentation:** Same entity appears as multiple nodes, splitting connections
2. **Difficult Queries:** Must search many variations to find all mentions
3. **Broken References:** Person IDs that don't exist in the member table
4. **Inconsistent Capitalization:** "microsoft" vs "Microsoft" vs "MICROSOFT"
5. **Abbreviated Forms:** "ML" vs "Machine Learning" treated differently

**Benefits of Implementation:**
1. **Unified Entity Graph:** One canonical node per real-world entity
2. **Better Analytics:** Accurate aggregation of relationships per entity
3. **Improved Search:** Find all mentions of "Google" regardless of variation
4. **Data Quality:** Catch extraction errors before they pollute the graph
5. **Alias Support:** Track known variations without duplicating nodes

### Recommended Implementation

#### High-Level Architecture

```
┌──────────────────┐
│  Extracted Fact  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────┐
│  Entity Extraction   │
│  (orgs, places, IDs) │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐     ┌────────────────┐
│  Normalization       │────>│  Alias Tables  │
│  Rules & Lookups     │     │  (SQLite)      │
└────────┬─────────────┘     └────────────────┘
         │
         ▼
┌──────────────────────┐
│  Validation Checks   │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Normalized Fact     │
└──────────────────────┘
```

#### Implementation Components

**Component 1: Normalization Tables in SQLite**

```sql
-- Add to schema.sql

-- Organization aliases
CREATE TABLE IF NOT EXISTS org_alias (
  alias           TEXT PRIMARY KEY,
  canonical_name  TEXT NOT NULL,
  verified        INTEGER DEFAULT 0 CHECK (verified IN (0,1))
);

-- Pre-populate common variations
INSERT INTO org_alias (alias, canonical_name, verified) VALUES
  ('google', 'Google', 1),
  ('google inc', 'Google', 1),
  ('google llc', 'Google', 1),
  ('alphabet', 'Alphabet', 1),
  ('meta', 'Meta', 1),
  ('facebook', 'Meta', 1),
  ('fb', 'Meta', 1),
  ('microsoft', 'Microsoft', 1),
  ('msft', 'Microsoft', 1),
  ('ms', 'Microsoft', 1);

-- Location aliases
CREATE TABLE IF NOT EXISTS place_alias (
  alias           TEXT PRIMARY KEY,
  canonical_name  TEXT NOT NULL,
  place_type      TEXT,  -- 'city', 'state', 'country'
  verified        INTEGER DEFAULT 0
);

INSERT INTO place_alias (alias, canonical_name, place_type, verified) VALUES
  ('sf', 'San Francisco', 'city', 1),
  ('san fran', 'San Francisco', 'city', 1),
  ('san francisco, ca', 'San Francisco', 'city', 1),
  ('nyc', 'New York City', 'city', 1),
  ('new york, ny', 'New York City', 'city', 1),
  ('la', 'Los Angeles', 'city', 1),
  ('los angeles, ca', 'Los Angeles', 'city', 1),
  ('bay area', 'San Francisco Bay Area', 'region', 1);

-- Topic aliases
CREATE TABLE IF NOT EXISTS topic_alias (
  alias           TEXT PRIMARY KEY,
  canonical_name  TEXT NOT NULL,
  category        TEXT,
  verified        INTEGER DEFAULT 0
);

INSERT INTO topic_alias (alias, canonical_name, category, verified) VALUES
  ('ai', 'Artificial Intelligence', 'technology', 1),
  ('a.i.', 'Artificial Intelligence', 'technology', 1),
  ('ml', 'Machine Learning', 'technology', 1),
  ('llm', 'Large Language Models', 'technology', 1),
  ('llms', 'Large Language Models', 'technology', 1),
  ('crypto', 'Cryptocurrency', 'technology', 1),
  ('bitcoin', 'Bitcoin', 'cryptocurrency', 1),
  ('btc', 'Bitcoin', 'cryptocurrency', 1);
```

**Component 2: Normalization Engine**

```python
# ie/normalization.py

import sqlite3
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class NormalizationResult:
    """Result of normalizing an entity."""
    canonical: str
    original: str
    confidence: float  # How confident we are in the normalization
    method: str  # 'exact', 'alias', 'fuzzy', 'none'

class EntityNormalizer:
    """Normalize and validate entities from extracted facts."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._load_caches()
    
    def _load_caches(self):
        """Load normalization lookups into memory for speed."""
        # Organization aliases
        self.org_aliases = {}
        for row in self.conn.execute("SELECT alias, canonical_name FROM org_alias"):
            self.org_aliases[row[0].lower()] = row[1]
        
        # Place aliases
        self.place_aliases = {}
        for row in self.conn.execute("SELECT alias, canonical_name FROM place_alias"):
            self.place_aliases[row[0].lower()] = row[1]
        
        # Topic aliases
        self.topic_aliases = {}
        for row in self.conn.execute("SELECT alias, canonical_name FROM topic_alias"):
            self.topic_aliases[row[0].lower()] = row[1]
        
        # Known Discord members
        self.known_members = set()
        for row in self.conn.execute("SELECT id FROM member"):
            self.known_members.add(row[0])
    
    def normalize_organization(self, org_name: str) -> NormalizationResult:
        """
        Normalize organization name to canonical form.
        
        Strategy:
        1. Check exact alias match (case-insensitive)
        2. Try fuzzy matching for typos
        3. Apply common transformations
        4. Return cleaned version if no match
        """
        if not org_name or not org_name.strip():
            return NormalizationResult("", "", 0.0, "none")
        
        original = org_name.strip()
        cleaned = self._clean_org_name(original)
        
        # Check alias table
        if cleaned.lower() in self.org_aliases:
            canonical = self.org_aliases[cleaned.lower()]
            return NormalizationResult(canonical, original, 0.95, "alias")
        
        # Try removing common suffixes
        for suffix in [' inc', ' llc', ' corp', ' corporation', ' ltd', ' co']:
            if cleaned.lower().endswith(suffix):
                base = cleaned[:-len(suffix)].strip()
                if base.lower() in self.org_aliases:
                    canonical = self.org_aliases[base.lower()]
                    return NormalizationResult(canonical, original, 0.85, "alias")
        
        # Fuzzy match (Levenshtein distance)
        fuzzy_match = self._fuzzy_match_org(cleaned)
        if fuzzy_match:
            return NormalizationResult(fuzzy_match, original, 0.7, "fuzzy")
        
        # No match - return cleaned version with proper capitalization
        canonical = self._title_case(cleaned)
        return NormalizationResult(canonical, original, 0.5, "none")
    
    def normalize_location(self, location: str) -> NormalizationResult:
        """Normalize location to canonical form."""
        if not location or not location.strip():
            return NormalizationResult("", "", 0.0, "none")
        
        original = location.strip()
        cleaned = self._clean_location(original)
        
        # Check alias table
        if cleaned.lower() in self.place_aliases:
            canonical = self.place_aliases[cleaned.lower()]
            return NormalizationResult(canonical, original, 0.95, "alias")
        
        # Try fuzzy matching
        fuzzy_match = self._fuzzy_match_place(cleaned)
        if fuzzy_match:
            return NormalizationResult(fuzzy_match, original, 0.7, "fuzzy")
        
        canonical = self._title_case(cleaned)
        return NormalizationResult(canonical, original, 0.5, "none")
    
    def normalize_topic(self, topic: str) -> NormalizationResult:
        """Normalize topic to canonical form."""
        if not topic or not topic.strip():
            return NormalizationResult("", "", 0.0, "none")
        
        original = topic.strip()
        cleaned = original.lower()
        
        # Check alias table
        if cleaned in self.topic_aliases:
            canonical = self.topic_aliases[cleaned]
            return NormalizationResult(canonical, original, 0.95, "alias")
        
        # Apply transformations
        canonical = self._normalize_topic_case(original)
        return NormalizationResult(canonical, original, 0.6, "transform")
    
    def validate_person_id(self, person_id: str) -> bool:
        """Check if person_id exists in the member table."""
        return person_id in self.known_members
    
    def _clean_org_name(self, name: str) -> str:
        """Clean organization name."""
        # Remove common artifacts
        name = re.sub(r'\s+', ' ', name)  # Normalize whitespace
        name = name.strip('.,!?')
        return name
    
    def _clean_location(self, location: str) -> str:
        """Clean location name."""
        # Remove trailing state/country codes in parentheses
        location = re.sub(r'\s*\([^)]*\)\s*$', '', location)
        location = re.sub(r'\s+', ' ', location)
        return location.strip()
    
    def _title_case(self, text: str) -> str:
        """Apply smart title casing."""
        # Special cases that shouldn't be title-cased
        lowercase_words = {'of', 'the', 'and', 'in', 'at', 'for', 'on'}
        
        words = text.split()
        result = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in lowercase_words:
                result.append(word.capitalize())
            else:
                result.append(word.lower())
        
        return ' '.join(result)
    
    def _normalize_topic_case(self, topic: str) -> str:
        """Normalize topic capitalization."""
        # Keep common acronyms uppercase
        if topic.upper() in ['AI', 'ML', 'NLP', 'API', 'UI', 'UX', 'LLM']:
            return topic.upper()
        
        # Otherwise title case
        return self._title_case(topic)
    
    def _fuzzy_match_org(self, name: str) -> Optional[str]:
        """Fuzzy match organization using Levenshtein distance."""
        # Simple implementation - in production use python-Levenshtein
        name_lower = name.lower()
        best_match = None
        best_distance = float('inf')
        
        for alias, canonical in self.org_aliases.items():
            distance = self._levenshtein_distance(name_lower, alias)
            # Allow up to 2 character differences for short names
            threshold = 2 if len(name) < 10 else 3
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_match = canonical
        
        return best_match
    
    def _fuzzy_match_place(self, location: str) -> Optional[str]:
        """Fuzzy match location."""
        location_lower = location.lower()
        best_match = None
        best_distance = float('inf')
        
        for alias, canonical in self.place_aliases.items():
            distance = self._levenshtein_distance(location_lower, alias)
            threshold = 2
            if distance <= threshold and distance < best_distance:
                best_distance = distance
                best_match = canonical
        
        return best_match
    
    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return EntityNormalizer._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
```

**Component 3: Validation Layer**

```python
# ie/validation.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class ValidationError:
    """Describes why a fact failed validation."""
    field: str
    message: str
    severity: str  # 'error' or 'warning'

@dataclass
class ValidationResult:
    """Result of validating a fact."""
    valid: bool
    errors: list[ValidationError]
    warnings: list[ValidationError]

class FactValidator:
    """Validate facts before storing."""
    
    def __init__(self, normalizer: EntityNormalizer):
        self.normalizer = normalizer
    
    def validate_fact(self, fact: ExtractionFact) -> ValidationResult:
        """Validate a fact and return errors/warnings."""
        errors = []
        warnings = []
        
        # Validate subject (must be known Discord member)
        if not self.normalizer.validate_person_id(fact.subject_id):
            errors.append(ValidationError(
                "subject_id",
                f"Person ID '{fact.subject_id}' not found in member table",
                "error"
            ))
        
        # Validate based on fact type
        if fact.type == FactType.WORKS_AT:
            errors.extend(self._validate_works_at(fact))
        elif fact.type == FactType.LIVES_IN:
            errors.extend(self._validate_lives_in(fact))
        elif fact.type == FactType.TALKS_ABOUT:
            errors.extend(self._validate_talks_about(fact))
        elif fact.type == FactType.CLOSE_TO:
            errors.extend(self._validate_close_to(fact))
        
        # Check confidence is reasonable
        if fact.confidence < 0.3:
            warnings.append(ValidationError(
                "confidence",
                f"Confidence {fact.confidence} is very low",
                "warning"
            ))
        
        # Check evidence exists
        if not fact.evidence:
            errors.append(ValidationError(
                "evidence",
                "No evidence message IDs provided",
                "error"
            ))
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def _validate_works_at(self, fact: ExtractionFact) -> list[ValidationError]:
        """Validate WORKS_AT specific fields."""
        errors = []
        
        org = fact.attributes.get("organization")
        if not org or not str(org).strip():
            errors.append(ValidationError(
                "organization",
                "Organization is required for WORKS_AT",
                "error"
            ))
        
        # Validate date formats if present
        start_date = fact.attributes.get("start_date")
        if start_date and not self._is_valid_date(start_date):
            errors.append(ValidationError(
                "start_date",
                f"Invalid date format: {start_date}",
                "error"
            ))
        
        return errors
    
    def _validate_lives_in(self, fact: ExtractionFact) -> list[ValidationError]:
        """Validate LIVES_IN specific fields."""
        errors = []
        
        location = fact.attributes.get("location")
        if not location or not str(location).strip():
            errors.append(ValidationError(
                "location",
                "Location is required for LIVES_IN",
                "error"
            ))
        
        return errors
    
    def _validate_talks_about(self, fact: ExtractionFact) -> list[ValidationError]:
        """Validate TALKS_ABOUT specific fields."""
        errors = []
        
        topic = fact.attributes.get("topic")
        if not topic or not str(topic).strip():
            errors.append(ValidationError(
                "topic",
                "Topic is required for TALKS_ABOUT",
                "error"
            ))
        
        return errors
    
    def _validate_close_to(self, fact: ExtractionFact) -> list[ValidationError]:
        """Validate CLOSE_TO specific fields."""
        errors = []
        
        if not fact.object_id:
            errors.append(ValidationError(
                "object_id",
                "Object person ID required for CLOSE_TO",
                "error"
            ))
        elif not self.normalizer.validate_person_id(fact.object_id):
            errors.append(ValidationError(
                "object_id",
                f"Person ID '{fact.object_id}' not found in member table",
                "error"
            ))
        
        return errors
    
    @staticmethod
    def _is_valid_date(date_str: str) -> bool:
        """Check if string is a valid ISO date."""
        try:
            from datetime import datetime
            datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            return True
        except (ValueError, AttributeError):
            return False
```

**Component 4: Integration into IE Pipeline**

```python
# Modify ie/runner.py to use normalization and validation:

from ie.normalization import EntityNormalizer, NormalizationResult
from ie.validation import FactValidator

def run_ie_job(...):
    # ... setup ...
    
    normalizer = EntityNormalizer(conn)
    validator = FactValidator(normalizer)
    
    for window in builder.iter_windows():
        # ... extraction ...
        
        if result:
            for fact in result.facts:
                # Validate first
                validation = validator.validate_fact(fact)
                
                if not validation.valid:
                    print(f"[IE] Skipping invalid fact: {validation.errors}")
                    continue
                
                # Log warnings
                for warning in validation.warnings:
                    print(f"[IE] Warning: {warning.message}")
                
                # Normalize entities
                if fact.type == FactType.WORKS_AT:
                    org_result = normalizer.normalize_organization(
                        fact.attributes.get("organization", "")
                    )
                    fact.attributes["organization"] = org_result.canonical
                    fact.attributes["organization_original"] = org_result.original
                    
                    if fact.attributes.get("location"):
                        loc_result = normalizer.normalize_location(
                            fact.attributes["location"]
                        )
                        fact.attributes["location"] = loc_result.canonical
                
                elif fact.type == FactType.LIVES_IN:
                    loc_result = normalizer.normalize_location(
                        fact.attributes.get("location", "")
                    )
                    fact.attributes["location"] = loc_result.canonical
                    fact.attributes["location_original"] = loc_result.original
                
                elif fact.type == FactType.TALKS_ABOUT:
                    topic_result = normalizer.normalize_topic(
                        fact.attributes.get("topic", "")
                    )
                    fact.attributes["topic"] = topic_result.canonical
                    fact.attributes["topic_original"] = topic_result.original
                
                # Now store the normalized, validated fact
                _upsert_fact(conn, ...)
```

**Component 5: Alias Learning**

Allow the system to learn new aliases over time:

```python
# ie/alias_learning.py

def suggest_new_aliases(conn: sqlite3.Connection, min_occurrences: int = 3):
    """
    Analyze stored facts to suggest new alias mappings.
    
    Strategy:
    - Find entity strings that appear frequently
    - Identify potential variations (similar strings)
    - Suggest as aliases for manual review
    """
    # Find organizations mentioned multiple times
    org_counts = conn.execute("""
        SELECT json_extract(attributes, '$.organization') as org, COUNT(*) as cnt
        FROM fact
        WHERE type = 'WORKS_AT'
        GROUP BY org
        HAVING cnt >= ?
        ORDER BY cnt DESC
    """, (min_occurrences,)).fetchall()
    
    suggestions = []
    for org, count in org_counts:
        # Check if already has canonical mapping
        existing = conn.execute(
            "SELECT canonical_name FROM org_alias WHERE alias = ?",
            (org.lower(),)
        ).fetchone()
        
        if not existing:
            # Check for similar existing organizations
            similar = _find_similar_orgs(conn, org)
            if similar:
                suggestions.append({
                    "alias": org,
                    "suggested_canonical": similar,
                    "occurrences": count,
                    "type": "organization"
                })
    
    return suggestions
```

---

## 3. Enhanced Windowing Strategy

### What Is This?

The current windowing system uses a simple sliding window approach: take N consecutive messages from a channel and extract facts. Enhanced windowing improves context quality by:

- **Including reply chains:** When a message replies to something outside the window, pull in the parent
- **Adaptive window sizing:** Vary window size based on conversation density
- **Topic coherence:** Keep windows within logical conversation boundaries
- **Cross-channel context:** Include relevant context from related channels when needed

### Why This Improves the Application

**Current Limitations:**
1. **Lost Context:** "yes" replies make no sense without the question
2. **Broken Chains:** Multi-message threads are split across windows
3. **Fixed Size Inefficiency:** Small windows miss context, large windows add noise
4. **Topic Mixing:** Windows can span unrelated conversations

**Benefits of Enhancement:**
1. **Better Context Quality:** LLM sees complete conversational units
2. **Higher Extraction Accuracy:** More context = better fact confidence
3. **Fewer Hallucinations:** Complete threads reduce ambiguous references
4. **Efficient Processing:** Adaptive sizing avoids processing irrelevant messages
5. **Temporal Awareness:** Preserve the logical flow of information

### Recommended Implementation

#### Strategy Overview

```
Current: [M1, M2, M3, M4] → extract facts
                ↓
Enhanced: Analyze conversation structure
                ↓
          Build context-aware windows
                ↓
          [M1, M2, M3, M4, M0(parent), M5(continuation)]
```

#### Implementation Approach

**Phase 1: Reply Chain Reconstruction**

```python
# ie/enhanced_windowing.py

import sqlite3
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ConversationThread:
    """Represents a coherent conversation thread."""
    messages: list[MessageRecord]
    root_message_id: str
    depth: int = 0
    
    def get_window_with_context(
        self,
        focus_index: int,
        max_size: int = 8
    ) -> MessageWindow:
        """
        Build window around focus message with full context.
        
        Strategy:
        - Include the focus message
        - Add parent messages up the reply chain
        - Add surrounding chronological messages
        - Stay within max_size
        """
        focus_msg = self.messages[focus_index]
        window_messages = [focus_msg]
        
        # Track messages we've included
        included_ids = {focus_msg.id}
        
        # Add parent messages (reply context)
        current = focus_msg
        parent_count = 0
        max_parents = max_size // 2  # Reserve half for parents
        
        while current.reply_to_id and parent_count < max_parents:
            parent = self._find_message_by_id(current.reply_to_id)
            if parent and parent.id not in included_ids:
                window_messages.insert(0, parent)  # Add at beginning
                included_ids.add(parent.id)
                current = parent
                parent_count += 1
            else:
                break
        
        # Add surrounding chronological messages
        remaining_space = max_size - len(window_messages)
        before_count = remaining_space // 2
        after_count = remaining_space - before_count
        
        # Messages before focus
        for i in range(focus_index - 1, max(0, focus_index - before_count - 1), -1):
            msg = self.messages[i]
            if msg.id not in included_ids:
                window_messages.insert(parent_count, msg)
                included_ids.add(msg.id)
        
        # Messages after focus
        for i in range(focus_index + 1, min(len(self.messages), focus_index + after_count + 1)):
            msg = self.messages[i]
            if msg.id not in included_ids:
                window_messages.append(msg)
                included_ids.add(msg.id)
        
        # Sort by timestamp to maintain chronological order
        window_messages.sort(key=lambda m: m.timestamp)
        
        # Find new focus index
        new_focus_index = next(
            i for i, m in enumerate(window_messages) if m.id == focus_msg.id
        )
        
        return MessageWindow(
            messages=tuple(window_messages),
            focus_index=new_focus_index
        )
    
    def _find_message_by_id(self, message_id: str) -> Optional[MessageRecord]:
        """Find message in thread by ID."""
        for msg in self.messages:
            if msg.id == message_id:
                return msg
        return None


class EnhancedWindowBuilder:
    """
    Build context-aware windows with reply chain reconstruction.
    """
    
    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        base_window_size: int = 6,
        max_window_size: int = 12,
        min_window_size: int = 3,
        **kwargs
    ):
        self.conn = conn
        self.base_window_size = base_window_size
        self.max_window_size = max_window_size
        self.min_window_size = min_window_size
        self.filter_kwargs = kwargs
    
    def iter_windows(self) -> Iterator[MessageWindow]:
        """
        Generate context-aware windows.
        
        Strategy:
        1. Load messages by channel
        2. Build conversation threads
        3. For each message, create optimal window with context
        """
        messages_by_channel = self._load_messages_grouped()
        
        for channel_id, messages in messages_by_channel.items():
            # Build thread structure
            threads = self._build_threads(messages)
            
            # Generate windows for each thread
            for thread in threads:
                for i, message in enumerate(thread.messages):
                    # Determine optimal window size
                    window_size = self._calculate_optimal_size(thread, i)
                    
                    # Build window with full context
                    window = thread.get_window_with_context(i, window_size)
                    
                    yield window
    
    def _load_messages_grouped(self) -> dict[str, list[MessageRecord]]:
        """Load messages grouped by channel."""
        from collections import defaultdict
        
        # Use same filtering as base WindowBuilder
        builder = WindowBuilder(
            self.conn,
            window_size=self.base_window_size,
            **self.filter_kwargs
        )
        
        messages_by_channel = defaultdict(list)
        for message in builder.iter_rows():
            messages_by_channel[message.channel_id].append(message)
        
        return dict(messages_by_channel)
    
    def _build_threads(self, messages: list[MessageRecord]) -> list[ConversationThread]:
        """
        Organize messages into conversation threads.
        
        A thread is a sequence of messages connected by replies.
        """
        # Build reply graph
        children_map = defaultdict(list)  # parent_id -> [child messages]
        message_map = {msg.id: msg for msg in messages}
        
        for msg in messages:
            if msg.reply_to_id:
                children_map[msg.reply_to_id].append(msg)
        
        # Find root messages (no parent or parent not in this channel)
        roots = []
        for msg in messages:
            if not msg.reply_to_id or msg.reply_to_id not in message_map:
                roots.append(msg)
        
        # Build threads from roots
        threads = []
        for root in roots:
            thread_messages = self._collect_thread(root, children_map, message_map)
            if thread_messages:
                threads.append(ConversationThread(
                    messages=thread_messages,
                    root_message_id=root.id
                ))
        
        return threads
    
    def _collect_thread(
        self,
        root: MessageRecord,
        children_map: dict,
        message_map: dict
    ) -> list[MessageRecord]:
        """Collect all messages in a thread using DFS."""
        thread = [root]
        visited = {root.id}
        
        # BFS to collect all descendants
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for child in children_map.get(current.id, []):
                if child.id not in visited:
                    thread.append(child)
                    visited.add(child.id)
                    queue.append(child)
        
        # Sort by timestamp
        thread.sort(key=lambda m: m.timestamp)
        return thread
    
    def _calculate_optimal_size(
        self,
        thread: ConversationThread,
        message_index: int
    ) -> int:
        """
        Calculate optimal window size for a message.
        
        Considerations:
        - Messages with replies need larger windows
        - Dense conversations can use smaller windows
        - Long gaps suggest topic change, use smaller window
        """
        focus_msg = thread.messages[message_index]
        
        # Base size
        size = self.base_window_size
        
        # Increase for messages with replies
        if focus_msg.reply_to_id:
            size += 2
        
        # Increase if this message is replied to
        has_children = any(
            m.reply_to_id == focus_msg.id 
            for m in thread.messages[message_index + 1:]
        )
        if has_children:
            size += 2
        
        # Decrease if there's a long time gap before/after
        if message_index > 0:
            time_gap = (
                focus_msg.timestamp - thread.messages[message_index - 1].timestamp
            ).total_seconds()
            if time_gap > 3600:  # 1 hour gap
                size -= 1
        
        # Clamp to bounds
        return max(self.min_window_size, min(self.max_window_size, size))
```

**Phase 2: Topic Boundary Detection**

```python
# ie/topic_detection.py

from datetime import timedelta

def detect_topic_boundaries(
    messages: list[MessageRecord],
    *,
    time_threshold: timedelta = timedelta(hours=1),
    content_similarity_threshold: float = 0.3
) -> list[int]:
    """
    Detect where topic changes occur in a message sequence.
    
    Returns indices where new topics likely begin.
    
    Heuristics:
    - Large time gaps (> 1 hour)
    - Change in participants
    - Content similarity drop (if embeddings available)
    """
    boundaries = [0]  # Always start at beginning
    
    for i in range(1, len(messages)):
        current = messages[i]
        previous = messages[i - 1]
        
        # Time gap heuristic
        time_gap = (current.timestamp - previous.timestamp).total_seconds()
        if time_gap > time_threshold.total_seconds():
            boundaries.append(i)
            continue
        
        # Participant change heuristic (weak signal)
        # If everyone changes, likely new topic
        if i >= 3:
            recent_authors = {messages[j].author_id for j in range(i - 3, i)}
            if current.author_id not in recent_authors:
                # New person might indicate topic shift
                # But don't trust this alone
                pass
        
        # Content markers (explicit)
        # Look for topic-changing phrases
        content_lower = current.content.lower()
        topic_markers = [
            'anyway', 'btw', 'by the way', 'changing topics',
            'different topic', 'on another note', 'quick question',
            'completely different', 'switching gears'
        ]
        
        if any(marker in content_lower for marker in topic_markers):
            boundaries.append(i)
            continue
    
    return boundaries


class TopicAwareWindowBuilder(EnhancedWindowBuilder):
    """
    Window builder that respects topic boundaries.
    """
    
    def iter_windows(self) -> Iterator[MessageWindow]:
        """Generate windows that don't cross topic boundaries."""
        messages_by_channel = self._load_messages_grouped()
        
        for channel_id, messages in messages_by_channel.items():
            # Detect topic boundaries
            boundaries = detect_topic_boundaries(messages)
            
            # Split into topic segments
            segments = []
            for i in range(len(boundaries)):
                start = boundaries[i]
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(messages)
                segments.append(messages[start:end])
            
            # Process each segment independently
            for segment in segments:
                threads = self._build_threads(segment)
                
                for thread in threads:
                    for i, message in enumerate(thread.messages):
                        window_size = self._calculate_optimal_size(thread, i)
                        window = thread.get_window_with_context(i, window_size)
                        yield window
```

**Phase 3: Integration**

Replace standard `WindowBuilder` with enhanced version:

```python
# In ie/runner.py:

def run_ie_job(...):
    # ... setup ...
    
    # Use enhanced windowing instead of basic
    from ie.enhanced_windowing import TopicAwareWindowBuilder
    
    builder = TopicAwareWindowBuilder(
        conn,
        base_window_size=config.window_size,
        max_window_size=config.window_size * 2,
        min_window_size=max(2, config.window_size // 2)
    )
    
    # Rest of processing remains the same
    for window in builder.iter_windows():
        # ... extraction ...
```

---

## 4. Improved Prompt Engineering

### What Is This?

Prompt engineering involves designing the instructions and examples given to the LLM to maximize extraction quality. Current prompts are functional but can be significantly enhanced with:

- **Few-shot examples:** Show the LLM what good extractions look like
- **Negative examples:** Show what NOT to extract (ambiguous cases)
- **Structured reasoning:** Ask LLM to explain its extractions
- **Confidence calibration:** Guide the LLM to score confidence accurately
- **Error recovery:** Handle edge cases explicitly

### Why This Improves the Application

**Current Prompt Weaknesses:**
1. **Zero-shot approach:** LLM has to guess what you want
2. **No confidence guidance:** Arbitrary scoring
3. **Ambiguity handling:** Unclear how to handle "maybe" statements
4. **Missing edge cases:** No guidance on handling jokes, hypotheticals
5. **No reasoning trace:** Can't debug why facts were extracted

**Benefits of Enhancement:**
1. **Higher Precision:** Fewer false positive extractions
2. **Better Confidence:** Scores actually reflect reliability
3. **Consistent Behavior:** Same input → same output
4. **Easier Debugging:** Can trace extraction reasoning
5. **Reduced Hallucination:** LLM stays grounded in text

### Recommended Implementation

#### Enhanced Prompt Structure

```
System: Role + capabilities + constraints
    ↓
User: {
    Task overview
    Fact catalog with examples
    Few-shot examples (good + bad)
    Confidence guidelines
    Special case handling
    Output schema
    Actual conversation window
}
    ↓
Assistant: Structured JSON response
```

#### Implementation

```python
# ie/advanced_prompts.py

from typing import Sequence
from ie.types import FactType, FactDefinition
from ie.config import FACT_DEFINITION_INDEX
from ie.windowing import MessageWindow

# Enhanced system prompt with clear role definition
SYSTEM_PROMPT_V2 = """You are a specialized information extraction system analyzing Discord conversations.

Your task is to identify FACTUAL, VERIFIABLE relationship statements and output them in structured JSON format.

Core principles:
1. ONLY extract facts explicitly stated or strongly implied by the speaker
2. Distinguish between facts (confident statements) and speculation (thinking about, might, maybe)
3. Provide confidence scores that reflect certainty
4. When in doubt, DON'T extract - precision over recall
5. Use exact Discord user IDs from the conversation for people

You will see a short conversation window. Extract facts that can be confidently determined from this context."""


def build_few_shot_examples() -> str:
    """Build comprehensive few-shot examples."""
    return """
## Few-Shot Examples

### Example 1: High Confidence Work Fact
**Conversation:**
[2024-01-15T10:23:00] Alice (ID: 12345): Just finished my first week at Google! Working as a SWE in the Cloud team
[2024-01-15T10:24:30] Bob (ID: 67890): Congrats! How's the SF office?
[2024-01-15T10:25:12] Alice (ID: 12345): It's great! Love the campus

**Extraction:**
```json
{
  "facts": [
    {
      "type": "WORKS_AT",
      "subject_id": "12345",
      "object_label": "Google",
      "attributes": {
        "organization": "Google",
        "role": "Software Engineer",
        "location": "San Francisco"
      },
      "confidence": 0.95,
      "evidence": ["msg_001", "msg_003"],
      "timestamp": "2024-01-15T10:23:00Z",
      "notes": "Explicitly stated current employment with role and location confirmed in context"
    }
  ]
}
```

**Why high confidence (0.95)?** Alice directly states she works at Google, mentions her role, and confirms SF location. This is a primary source statement.

---

### Example 2: Medium Confidence Location Inference
**Conversation:**
[2024-02-01T14:00:00] Carol (ID: 11111): Anyone near downtown want to grab lunch?
[2024-02-01T14:01:00] Dave (ID: 22222): I'm in SoMa, where you thinking?
[2024-02-01T14:02:00] Carol (ID: 11111): How about that new place on Market St?

**Extraction:**
```json
{
  "facts": [
    {
      "type": "LIVES_IN",
      "subject_id": "22222",
      "object_label": "San Francisco",
      "attributes": {
        "location": "San Francisco"
      },
      "confidence": 0.70,
      "evidence": ["msg_002"],
      "timestamp": "2024-02-01T14:01:00Z",
      "notes": "Implied by being in SoMa neighborhood; likely resident but could be visiting"
    }
  ]
}
```

**Why medium confidence (0.70)?** Dave is in SoMa (San Francisco neighborhood) but this could mean he's visiting, works there, or lives there. The inference is reasonable but not certain.

---

### Example 3: Topic Interest
**Conversation:**
[2024-03-10T16:00:00] Eve (ID: 33333): Been diving deep into Rust lately
[2024-03-10T16:01:00] Eve (ID: 33333): The ownership model is fascinating
[2024-03-10T16:02:00] Frank (ID: 44444): Same! I've been reading the async book
[2024-03-10T16:03:00] Eve (ID: 33333): Oh nice, I haven't gotten to async yet

**Extraction:**
```json
{
  "facts": [
    {
      "type": "TALKS_ABOUT",
      "subject_id": "33333",
      "object_label": "Rust Programming",
      "attributes": {
        "topic": "Rust Programming",
        "sentiment": "positive"
      },
      "confidence": 0.85,
      "evidence": ["msg_001", "msg_002", "msg_004"],
      "timestamp": "2024-03-10T16:00:00Z",
      "notes": "Multiple messages showing active interest and learning"
    },
    {
      "type": "TALKS_ABOUT",
      "subject_id": "44444",
      "object_label": "Rust Programming",
      "attributes": {
        "topic": "Rust Programming",
        "sentiment": "positive"
      },
      "confidence": 0.80,
      "evidence": ["msg_003"],
      "timestamp": "2024-03-10T16:02:00Z",
      "notes": "Single message but shows active engagement with advanced topic (async)"
    }
  ]
}
```

---

### Example 4: DO NOT EXTRACT - Speculation
**Conversation:**
[2024-04-01T09:00:00] Grace (ID: 55555): Thinking about applying to Amazon
[2024-04-01T09:01:00] Henry (ID: 66666): You should! I heard they're hiring
[2024-04-01T09:02:00] Grace (ID: 55555): Yeah, might send in my resume next week

**Extraction:**
```json
{
  "facts": []
}
```

**Why no extraction?** Grace is considering applying (speculation), not stating current employment. "Thinking about" and "might" indicate future possibility, not present fact.

---

### Example 5: DO NOT EXTRACT - Hypothetical
**Conversation:**
[2024-05-01T12:00:00] Iris (ID: 77777): If I worked at Meta, I'd probably be on the Reality Labs team
[2024-05-01T12:01:00] Jack (ID: 88888): lol that would be cool

**Extraction:**
```json
{
  "facts": []
}
```

**Why no extraction?** "If I worked at" is explicitly hypothetical. Iris is not stating current employment.

---

### Example 6: DO NOT EXTRACT - Joking/Sarcasm
**Conversation:**
[2024-06-01T15:00:00] Kate (ID: 99999): Ugh I'm basically living at the office these days
[2024-06-01T15:01:00] Leo (ID: 10101): Same, I should just move into the conference room lol

**Extraction:**
```json
{
  "facts": []
}
```

**Why no extraction?** These are hyperbolic expressions of being busy, not literal statements about residence. The "lol" and context indicate non-literal meaning.

---

### Example 7: Relationship Closeness
**Conversation:**
[2024-07-01T18:00:00] Mike (ID: 20202): @Nina (ID: 30303) want to grab dinner after work?
[2024-07-01T18:01:00] Nina (ID: 30303): Sure! Our usual spot?
[2024-07-01T18:02:00] Mike (ID: 20202): Yeah, see you at 7
[2024-07-01T18:30:00] Oscar (ID: 40404): You two are always hanging out lol

**Extraction:**
```json
{
  "facts": [
    {
      "type": "CLOSE_TO",
      "subject_id": "20202",
      "object_id": "30303",
      "attributes": {
        "closeness_basis": "Regular social activities together, noted by others"
      },
      "confidence": 0.75,
      "evidence": ["msg_001", "msg_002", "msg_004"],
      "timestamp": "2024-07-01T18:00:00Z",
      "notes": "Pattern of regular interaction confirmed by third party observation"
    }
  ]
}
```

**Why extract?** Multiple signals: they have a "usual spot", Oscar comments on frequent hangouts. This suggests a genuine social relationship beyond casual interaction.
"""


def build_confidence_guidelines() -> str:
    """Detailed confidence scoring guidelines."""
    return """
## Confidence Scoring Guidelines

Score facts on a 0.0 to 1.0 scale based on these criteria:

### 0.90 - 1.00: Very High Confidence
- **Primary source:** Person directly states fact about themselves
- **Explicit and specific:** Clear, unambiguous language
- **Recently confirmed:** Stated in present tense or very recent past
- Example: "I work at Google as a software engineer"

### 0.75 - 0.89: High Confidence
- **Strong implication:** Context makes fact very likely
- **Confirmed by multiple sources:** Multiple people corroborate
- **Specific details present:** Includes verifiable specifics
- Example: "Started my new role at Microsoft last month" (implies current employment)

### 0.60 - 0.74: Moderate-High Confidence
- **Reasonable inference:** Logical conclusion from context
- **Single strong indicator:** One clear signal with good context
- **Recent but not explicit:** Implied from recent conversation
- Example: "Been working with the AWS team on this" (implies AWS employment)

### 0.50 - 0.59: Moderate Confidence
- **Weak implication:** Possible but not certain
- **Limited context:** Not enough information to be sure
- **Temporal ambiguity:** Unclear if current or past
- Example: "Love the SF weather" (might live there, might be visiting)

### 0.30 - 0.49: Low Confidence
- **Speculative inference:** Requires assumptions
- **Ambiguous language:** Could mean multiple things
- **Minimal evidence:** Only one weak signal
- **Generally avoid extracting below 0.5 unless particularly valuable**

### Below 0.30: Do Not Extract
- Too speculative
- Insufficient evidence
- High chance of being wrong

### Special Modifiers

**Reduce confidence by 0.1-0.2 if:**
- Statement is indirect (third-party report)
- Contains hedge words ("maybe", "probably", "I think")
- Context is ambiguous
- Time reference is unclear

**Increase confidence by 0.1 if:**
- Multiple pieces of corroborating evidence
- Very recent timestamp
- Specific details provided
- No contradictory information in window

**Never extract if:**
- Explicitly hypothetical ("if", "would", "could")
- Clearly joking (sarcasm indicators, "lol", absurd content)
- About future intent only ("going to", "planning to")
- Question rather than statement
- Negation ("I don't work at...")
"""


def build_edge_case_guidance() -> str:
    """Handle tricky extraction scenarios."""
    return """
## Edge Case Handling

### Past vs. Present Employment
❌ "I used to work at Google" → Do NOT extract (past tense)
❌ "I worked at Google for 5 years" → Do NOT extract (past, no current indication)
✅ "I've been at Google for 5 years" → Extract (present perfect = ongoing)
✅ "Just celebrated my 5 year anniversary at Google" → Extract (recent, implies current)

### Company Transitions
✅ "Starting at Amazon next week" → Extract with low-moderate confidence (0.6), note as future
✅ "Last day at Google tomorrow, starting Meta on Monday" → Extract BOTH with proper dates
- Google with end_date
- Meta with start_date

### Multiple Roles
✅ "Working at Google during the day, freelancing at night" → Extract BOTH
- Primary org: Google (higher confidence)
- Secondary: freelance work (note in attributes)

### Ambiguous Locations
❌ "Love visiting SF" → Do NOT extract LIVES_IN
✅ "Moved to SF last month" → Extract LIVES_IN
✅ "SF native" → Extract LIVES_IN with high confidence
⚠️ "In SF this week" → Extract with LOW confidence (0.4), could be visiting

### Company Name Variations
Always normalize:
- "google" → "Google"
- "GOOG" → "Google"
- "Google Inc" → "Google"
- "Microsoft Corp" → "Microsoft"

### Pronouns and References
Track conversation context for pronoun resolution:
- "They offered me the job" → Need prior context to know who "they" are
- "The company" → Must refer back to find which company
- If ambiguous, lower confidence or skip extraction

### Negative Statements
❌ "I don't work at Google anymore" → Do NOT extract current WORKS_AT
⚠️ Can note as ENDED employment if very clear and recent

### Questions vs Statements
❌ "Does anyone here work at Meta?" → Do NOT extract
❌ "Should I apply to Amazon?" → Do NOT extract
✅ Response: "Yes, I work at Meta" → Extract
"""


def build_enhanced_prompt(
    window: MessageWindow,
    fact_types: Sequence[FactType]
) -> list[dict[str, str]]:
    """
    Build enhanced prompt with examples and guidelines.
    """
    # Build participant list
    participants = {}
    for record in window.messages:
        participants.setdefault(record.author_id, record.author_display)
    
    participant_lines = [
        f"- {name} (ID: {author_id})" 
        for author_id, name in participants.items()
    ]
    participant_text = "\n".join(participant_lines)
    
    # Build fact catalog
    catalog_lines = []
    for fact_type in fact_types:
        definition = FACT_DEFINITION_INDEX[fact_type]
        catalog_lines.append(f"**{fact_type.value}**")
        catalog_lines.append(f"  Subject: {definition.subject_description}")
        if definition.object_description:
            catalog_lines.append(f"  Object: {definition.object_description}")
        catalog_lines.append(f"  Purpose: {definition.rationale}")
        catalog_lines.append("")
    
    catalog_text = "\n".join(catalog_lines)
    
    # Build conversation text
    conversation_text = window.as_text()
    
    # Construct full user message
    user_message = f"""
# Extraction Task

You will analyze a Discord conversation window and extract structured facts.

## Supported Fact Types

{catalog_text}

{build_few_shot_examples()}

{build_confidence_guidelines()}

{build_edge_case_guidance()}

## Output Format

Return ONLY valid JSON matching this schema:

```json
{{
  "facts": [
    {{
      "type": "<FactType>",
      "subject_id": "<exact_discord_id>",
      "object_label": "<human_readable_name>",
      "object_id": "<optional_stable_id>",
      "attributes": {{ "key": "value" }},
      "confidence": 0.0,
      "evidence": ["<message_id>"],
      "timestamp": "<ISO8601>",
      "notes": "<optional_reasoning>"
    }}
  ]
}}
```

## Conversation to Analyze

**Participants:**
{participant_text}

**Focus Message:** {window.focus.id}

**Conversation:**
```
{conversation_text}
```

## Your Task

1. Read the conversation carefully
2. Identify factual statements matching supported types
3. For each potential fact:
   - Verify it meets confidence threshold (≥ 0.5)
   - Use EXACT participant IDs from the list above
   - Include ALL message IDs that support the fact
   - Add reasoning in "notes" field
4. Output valid JSON only

Remember: Precision over recall. When in doubt, don't extract.
""".strip()
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_message}
    ]


# Usage in ie/runner.py:

def run_ie_job_with_enhanced_prompts(...):
    # Replace the build_messages call:
    from ie.advanced_prompts import build_enhanced_prompt
    
    for window in builder.iter_windows():
        messages = build_enhanced_prompt(window, config.fact_types)
        
        # Rest of extraction logic remains the same
        content = client.complete(messages)
        # ...
```

#### Additional Enhancements

**Chain-of-Thought Reasoning:**

```python
# Ask LLM to show its reasoning before extraction

CHAIN_OF_THOUGHT_SUFFIX = """
Before outputting the final JSON, briefly analyze:
1. What facts are explicitly stated?
2. What can be confidently inferred?
3. What is ambiguous or speculative?

Then provide your final extraction.
"""
```

**Structured Output Validation:**

```python
# In ie/models.py, enhance validation:

class ExtractionFact(BaseModel):
    # ... existing fields ...
    
    @model_validator(mode='after')
    def validate_confidence_with_notes(self) -> 'ExtractionFact':
        """Ensure high-confidence facts have reasoning."""
        if self.confidence >= 0.8 and not self.notes:
            raise ValueError(
                f"Facts with confidence ≥ 0.8 must include reasoning in 'notes'"
            )
        return self
    
    @model_validator(mode='after')
    def validate_evidence_count(self) -> 'ExtractionFact':
        """Ensure sufficient evidence for confidence level."""
        evidence_count = len(self.evidence)
        if self.confidence >= 0.9 and evidence_count < 1:
            raise ValueError("Very high confidence requires evidence")
        return self
```

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
1. **Entity Normalization** - Critical for data quality
2. **Improved Prompts** - Immediate accuracy gains

### Phase 2: Enhancement (Week 2)
3. **Fact Deduplication** - Clean up existing data
4. **Enhanced Windowing** - Better context quality

### Phase 3: Integration (Week 3)
- Combine all improvements
- Run full pipeline tests
- Monitor quality metrics
- Iterate on thresholds

## Success Metrics

Track these metrics to measure improvement:

1. **Extraction Precision:** % of facts that are correct
2. **Confidence Calibration:** Do 0.9 confidence facts actually have 90% accuracy?
3. **Entity Consolidation:** How many duplicate entities were merged?
4. **Window Context Quality:** Average reply chain coverage
5. **User Feedback:** Manual review of sample extractions

## Conclusion

These four improvements work synergistically:

- **Normalization** ensures consistent entity representation
- **Deduplication** prevents redundant graph edges
- **Enhanced Windowing** provides better context for extraction
- **Improved Prompts** help the LLM make better use of that context

Together, they transform the pipeline from "works okay" to "production-quality knowledge extraction system."
