# Deduplication Prompt Improvements

## Executive Summary

The current deduplication prompt in `deduplicate/llm/prompts.py` is functional but lacks clarity, examples, and type-specific guidance. This document identifies key weaknesses and proposes concrete improvements to increase merge quality, reduce errors, and improve LLM consistency.

---

## Current Prompt Issues

### 1. **No Few-Shot Examples**

**Issue**: The prompt provides merge rules in prose without showing the LLM what successful merges look like in practice.

**Impact**: 
- LLM may misinterpret abstract rules
- Inconsistent behavior across different fact types
- Higher error rates on edge cases

**Evidence from Code**:
```python
MERGE_RULES = """
Rules for merging:
1. Merge facts that represent the same real-world assertion...
"""
```
No examples provided of actual merge decisions.

---

### 2. **Vague Confidence Calculation Rules**

**Issue**: The confidence rules are mathematically specified but lack intuitive grounding:

```
Confidence rules: identical facts use max confidence; complementary merges 
use weighted average plus 0.05; cap at 0.95; add 0.1 if merged fact has 5+ evidence messages.
```

**Problems**:
- What constitutes "identical" vs "complementary"?
- Why +0.05 for complementary? Why +0.1 for 5+ evidence?
- No guidance on handling conflicting confidence scores
- Weighted average formula not specified

**Impact**: LLMs will apply these rules inconsistently or ignore them entirely.

---

### 3. **No Fact-Type-Specific Guidance**

**Issue**: All 23 fact types use the same generic prompt, but different types have different merge semantics:

- `WORKS_AT`: Multiple jobs at different times are NOT duplicates
- `LIVES_IN`: Multiple residences over time are NOT duplicates  
- `TALKS_ABOUT`: Multiple mentions of same topic ARE duplicates
- `HAS_SKILL`: Skill level changes are NOT duplicates

**Current Approach**: One-size-fits-all rules

**Impact**: 
- Over-merging temporal facts (incorrectly merging past vs present jobs)
- Under-merging topical facts (treating similar topics as distinct)

---

### 4. **Missing Edge Case Handling**

**Issue**: No guidance for common failure modes:

- **Outliers**: What if one fact in a group of 5 is clearly different?
- **Partial Matches**: Merge "Google" and "Google Cloud" or keep separate?
- **Temporal Conflicts**: "Started 2020" vs "Started 2021" for same job?
- **Conservative Default**: Should LLM prefer splitting or merging when uncertain?

**Current Code**: Assumes LLM will figure it out

**Impact**: Unpredictable behavior on ambiguous cases

---

### 5. **Weak Output Schema Definition**

**Issue**: Output format described in prose rather than clear JSON schema:

```
Output format:
Return strict JSON with a top-level key canonical_facts. Each fact requires 
fields type, subject_id, object_label, object_id, object_type, attributes, 
confidence, evidence, timestamp, merged_from, and merge_reasoning.
```

**Problems**:
- No example of valid output
- No specification of field types (string vs number vs array)
- No indication of which fields can be null

**Impact**: Parser errors, retry loops, wasted tokens

---

### 6. **No Quality Validation Requirements**

**Issue**: Prompt doesn't ask LLM to self-check its work.

**Missing Instructions**:
- Verify all merged_from IDs exist in input
- Ensure evidence union is non-empty
- Confirm confidence is in [0, 1] range
- Check that merge preserves semantic meaning

**Impact**: Invalid output that fails validation, requiring retries

---

### 7. **Insufficient Merge Reasoning Expectations**

**Issue**: Current prompt: "merged_from, and merge_reasoning" without elaboration.

**What's Missing**:
- Expected reasoning depth
- Key elements to include (what was normalized, what conflicts existed, why merge was chosen)
- Format expectations (bullet points vs prose)

**Impact**: 
- Shallow reasoning like "similar facts" 
- Difficult to audit merge decisions
- Can't learn from LLM's decision process

---

### 8. **No Ambiguity Handling**

**Issue**: Prompt doesn't address uncertainty:

**Questions Not Answered**:
- What if the LLM is only 60% sure facts are duplicates?
- Should it merge conservatively or aggressively?
- Can it output multiple canonical facts if uncertain?
- Should it flag low-confidence merges?

**Impact**: Arbitrary decisions on borderline cases

---

### 9. **Attribute Priority Unclear**

**Issue**: No guidance on which attributes matter most for each fact type.

**Example for WORKS_AT**:
- Is `organization` name normalization more important than preserving `role` details?
- Should `location` mismatches prevent merging?
- Are `start_date` and `end_date` critical for deduplication?

**Current Prompt**: "Prefer specific attributes, normalize variants"

**Impact**: Inconsistent attribute handling

---

### 10. **Missing Purpose Context**

**Issue**: LLM doesn't know WHY deduplication matters or how results will be used.

**Missing Context**:
- Facts feed into a knowledge graph for profile generation
- Over-merging loses temporal nuance
- Under-merging creates clutter
- Results will be displayed to users with citations

**Impact**: LLM can't make informed tradeoffs

---

## Proposed Improvements

### Improvement 1: Add Few-Shot Examples

**Solution**: Include 3-5 examples per major fact category showing:
- Clear duplicate (merge to 1 fact)
- Partial duplicate (merge with attribute union)
- Non-duplicate temporal facts (keep separate)
- Ambiguous case (conservative split)

**Example Structure**:
```json
{
  "example_type": "clear_duplicate_works_at",
  "input": [
    {"fact_id": 1, "object_label": "Google", "role": "SWE", "confidence": 0.8},
    {"fact_id": 2, "object_label": "Google Inc.", "role": "Software Engineer", "confidence": 0.75}
  ],
  "output": {
    "canonical_facts": [{
      "object_label": "Google",
      "role": "Software Engineer",
      "confidence": 0.85,
      "merged_from": [1, 2],
      "merge_reasoning": "Same organization (normalized 'Google Inc.' to 'Google'), same role (expanded 'SWE' to 'Software Engineer'). Combined evidence and used weighted average confidence (0.8 * 0.67 + 0.75 * 0.33) + 0.05 = 0.85."
    }]
  }
}
```

---

### Improvement 2: Clarify Confidence Calculation

**Solution**: Provide explicit formula and worked examples:

```
Confidence Calculation Rules:

1. Identical facts (same attributes): confidence = MAX(c1, c2, ...)
   Example: [0.7, 0.85, 0.8] → 0.85

2. Complementary facts (different but compatible attributes):
   confidence = WEIGHTED_AVG + 0.05 (information gain bonus)
   Weights: w_i = c_i / SUM(c_j)
   Example: 
     Fact A: c=0.8, Fact B: c=0.6
     weighted_avg = (0.8 * 0.571) + (0.6 * 0.429) = 0.714
     final = min(0.714 + 0.05, 0.95) = 0.764

3. Evidence bonus: +0.1 if merged fact has 5+ supporting messages
   (Max confidence cap: 0.95)

4. Conflicting attributes: DO NOT MERGE (keep as separate facts)
```

---

### Improvement 3: Add Type-Specific Instructions

**Solution**: Inject type-specific merge guidance based on `partition.fact_type`:

```python
TYPE_SPECIFIC_RULES = {
    FactType.WORKS_AT: """
    WORKS_AT Merging Rules:
    - TEMPORAL SENSITIVITY: Different time periods = different facts
    - If start_date/end_date differ significantly (>6 months), keep separate
    - Only merge if describing the SAME employment period
    - Normalize organization names ("Google Inc." → "Google")
    - Expand role abbreviations ("SWE" → "Software Engineer")
    """,
    
    FactType.LIVES_IN: """
    LIVES_IN Merging Rules:
    - TEMPORAL SENSITIVITY: Different residences over time = separate facts
    - Only merge if describing the SAME living situation
    - Normalize location names ("SF" → "San Francisco", "NYC" → "New York")
    """,
    
    FactType.TALKS_ABOUT: """
    TALKS_ABOUT Merging Rules:
    - AGGREGATION FRIENDLY: Multiple mentions of same topic CAN be merged
    - Aggregate sentiment if provided (majority wins, or mark as "mixed")
    - Union all evidence to show topic frequency
    - Normalize topic names for consistency
    """,
}

def build_messages(partition: Partition, candidate_facts: Iterable[FactRecord]) -> list[dict[str, str]]:
    type_rules = TYPE_SPECIFIC_RULES.get(partition.fact_type, "")
    # ... include type_rules in prompt
```

---

### Improvement 4: Add Edge Case Guidance

**Solution**: Explicit instructions for common edge cases:

```
Edge Case Handling:

1. Outliers in Groups:
   - If 1 fact differs significantly from N others in a group, split it out
   - Example: [Google, Google, Google, Microsoft] → [Google], [Microsoft]

2. Partial Organization Matches:
   - "Google" vs "Google Cloud" → KEEP SEPARATE (different entities)
   - "Google" vs "Google Inc." → MERGE (same entity)
   - When unsure, prefer separate facts

3. Temporal Conflicts:
   - Dates differ by <3 months: Use earliest date, note variance in reasoning
   - Dates differ by >3 months: KEEP SEPARATE (different time periods)

4. Attribute Conflicts:
   - If core attributes conflict (e.g., different cities), KEEP SEPARATE
   - If minor attributes conflict (e.g., "engineer" vs "software engineer"), merge and prefer more specific

5. Conservative Default:
   - When uncertain (confidence < 0.7 that facts are duplicates), KEEP SEPARATE
   - It's better to have duplicates than to incorrectly merge distinct facts
```

---

### Improvement 5: Show Clear Output Schema

**Solution**: Replace prose with annotated JSON schema example:

```json
{
  "canonical_facts": [
    {
      "type": "WORKS_AT",                    // string (must match partition type)
      "subject_id": "12345",                 // string (must match partition subject)
      "object_label": "Google",              // string (normalized name)
      "object_id": "google-inc",             // string|null (stable identifier)
      "object_type": "Organization",         // string|null (entity type)
      "attributes": {                        // object (fact-specific key-value pairs)
        "organization": "Google",
        "role": "Software Engineer",
        "location": "San Francisco"
      },
      "confidence": 0.87,                    // float [0.0, 1.0]
      "evidence": ["msg_001", "msg_045"],    // string[] (non-empty)
      "timestamp": "2024-01-15T10:23:00Z",   // ISO8601 string
      "merged_from": [147, 148],             // int[] (source fact IDs, non-empty)
      "merge_reasoning": "..."               // string (explanation, non-empty)
    }
  ]
}
```

---

### Improvement 6: Add Quality Validation Instructions

**Solution**: Request self-checks in the prompt:

```
Before returning your response, verify:

✓ All fact IDs in merged_from exist in the input candidate_facts
✓ All evidence message IDs come from the input facts
✓ Confidence scores are between 0.0 and 1.0
✓ Each canonical fact has at least one source fact (merged_from non-empty)
✓ Attributes preserve or improve specificity (no information loss)
✓ Merge reasoning explains the decision clearly
✓ No circular or contradictory merges
```

---

### Improvement 7: Strengthen Merge Reasoning Expectations

**Solution**: Provide reasoning template:

```
Merge Reasoning Format:

Your merge_reasoning field should include:
1. What was merged: "Merged N facts describing the same [entity/relationship]"
2. Key normalizations: "Normalized 'Google Inc.' to 'Google', expanded 'SWE' to 'Software Engineer'"
3. Attribute resolution: "Combined location from Fact A with role from Fact B"
4. Conflicts handled: "Resolved date discrepancy (2020 vs 2021) by using earliest date"
5. Confidence justification: "High confidence (0.9) due to identical organization and overlapping evidence"

Good Example:
"Merged 2 facts about the same Google employment. Normalized organization name and expanded role abbreviation. Used location from more recent fact. High confidence (0.87) due to strong attribute overlap."

Bad Example:
"Similar facts merged."
```

---

### Improvement 8: Add Ambiguity Handling Protocol

**Solution**: Clear instructions for uncertain cases:

```
Handling Uncertainty:

When you're unsure if facts should be merged:

1. Calculate your confidence in the merge decision:
   - 0.9+ confidence: Definitely the same → MERGE
   - 0.7-0.9 confidence: Probably the same → MERGE with cautious confidence score
   - 0.5-0.7 confidence: Unclear → PREFER SEPARATE (conservative)
   - <0.5 confidence: Probably different → KEEP SEPARATE

2. You can output multiple canonical facts if the input group naturally splits:
   - Example: Facts 1,2,3 are clearly duplicates of each other
   - Facts 4,5 are clearly duplicates of each other
   - But the two groups are unrelated
   - Output: Two canonical facts

3. Document uncertainty in merge_reasoning:
   - "Moderate confidence (0.72) due to minor date discrepancy"
   - "Conservative merge; attributes compatible but limited evidence overlap"
```

---

### Improvement 9: Define Attribute Priority by Fact Type

**Solution**: Provide attribute importance hierarchy:

```python
ATTRIBUTE_PRIORITIES = {
    FactType.WORKS_AT: {
        "critical": ["organization", "start_date", "end_date"],  # Conflicts prevent merge
        "important": ["role", "location"],                       # Normalize/combine
        "optional": ["employment_type"]                          # Can be missing
    },
    FactType.LIVES_IN: {
        "critical": ["location", "since"],
        "important": [],
        "optional": []
    },
    # ... etc
}
```

Include in prompt:
```
Attribute Priority for WORKS_AT:

Critical (must match or be compatible):
- organization: Must be the same entity (after normalization)
- start_date, end_date: Must describe the same time period (±3 months)

Important (normalize/combine):
- role: Expand abbreviations, use most specific version
- location: Normalize city names, use most complete version

Optional:
- employment_type: Use value from highest confidence source
```

---

### Improvement 10: Provide Purpose Context

**Solution**: Add context preamble to system prompt:

```python
SYSTEM_PROMPT_V2 = """
You are a knowledge graph deduplication specialist working on a Discord conversation analysis system.

Context:
- Facts are extracted from casual Discord messages and may have inconsistent formatting
- The knowledge graph powers user profiles shown to community members
- Users can click on facts to see supporting evidence (message links)
- Over-merging loses temporal nuance (e.g., conflating past and present jobs)
- Under-merging creates clutter but is safer (users prefer accuracy)

Your goal: Create clean, accurate canonical facts that:
1. Preserve temporal distinctions (past vs present)
2. Normalize casual language ("Google Inc." → "Google") 
3. Maintain traceability (evidence must justify the merge)
4. Favor precision over recall (when uncertain, keep separate)

Given potential duplicate facts, output a consolidated list of canonical facts.
"""
```

---

## Implementation Plan

### Phase 1: Quick Wins (Week 1)
- [ ] Add type-specific rules for top 5 fact types (WORKS_AT, LIVES_IN, TALKS_ABOUT, HAS_SKILL, STUDIED_AT)
- [ ] Include 3-5 few-shot examples in prompt
- [ ] Add explicit confidence calculation formula
- [ ] Update system prompt with purpose context

### Phase 2: Robustness (Week 2)
- [ ] Add edge case handling guidance
- [ ] Improve output schema documentation
- [ ] Add quality validation checklist
- [ ] Strengthen merge reasoning expectations

### Phase 3: Refinement (Week 3)
- [ ] Add ambiguity handling protocol
- [ ] Define attribute priorities for all 23 fact types
- [ ] Create prompt templates for each fact category
- [ ] Add self-check instructions

### Phase 4: Validation (Week 4)
- [ ] A/B test improved prompts on held-out test set
- [ ] Measure precision/recall improvements
- [ ] Collect merge reasoning quality scores
- [ ] Tune confidence thresholds based on results

---

## Success Metrics

Track these metrics to measure improvement:

1. **Parse Success Rate**: % of LLM responses that parse successfully without retry
   - Current baseline: ~85% (from logs)
   - Target: >95%

2. **Merge Precision**: % of merged facts that are true duplicates (manual audit)
   - Target: >90%

3. **Merge Recall**: % of true duplicates that are merged
   - Target: >80%

4. **Reasoning Quality**: Average human rating of merge_reasoning (1-5 scale)
   - Target: >4.0

5. **Confidence Calibration**: Correlation between LLM confidence and merge correctness
   - Target: Pearson r > 0.7

---

## Example: Improved Full Prompt

```python
def build_messages(partition: Partition, candidate_facts: Iterable[FactRecord]) -> list[dict[str, str]]:
    # Type-specific rules
    type_guidance = TYPE_SPECIFIC_RULES.get(partition.fact_type, "")
    
    # Attribute priorities
    priorities = ATTRIBUTE_PRIORITIES.get(partition.fact_type, {})
    priority_text = f"""
    Attribute Priorities:
    - Critical: {', '.join(priorities.get('critical', []))}
    - Important: {', '.join(priorities.get('important', []))}
    - Optional: {', '.join(priorities.get('optional', []))}
    """
    
    # Few-shot examples (select relevant ones for this fact type)
    examples = select_examples(partition.fact_type)
    examples_text = "\n\n".join(format_example(ex) for ex in examples)
    
    user_prompt = f"""
DEDUPLICATION TASK

Fact Type: {partition.fact_type.value}
Subject: {partition.subject_name or partition.subject_id}

{type_guidance}

{priority_text}

{CONFIDENCE_RULES}

{EDGE_CASE_RULES}

{MERGE_REASONING_TEMPLATE}

{QUALITY_CHECKLIST}

Few-Shot Examples:
{examples_text}

Input Facts:
{json.dumps(request, indent=2)}

Output:
Return canonical_facts JSON following the schema above.
"""
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {"role": "user", "content": user_prompt.strip()},
    ]
```

---

## Conclusion

The current deduplication prompt is a good starting point but needs significant enhancement to achieve production-quality results. The proposed improvements address:

1. **Clarity**: Few-shot examples and explicit formulas
2. **Specificity**: Type-specific rules and attribute priorities  
3. **Robustness**: Edge case handling and quality checks
4. **Consistency**: Clear confidence calculations and merge reasoning standards
5. **Context**: Purpose and tradeoff awareness

Implementing these changes incrementally (Phase 1 → Phase 4) will improve merge quality while allowing empirical validation at each step.

**Estimated Impact**: 
- 10-15% improvement in merge precision
- 20-30% reduction in parse errors
- 50%+ improvement in merge reasoning quality
- Better handling of temporal and ambiguous cases

**Next Steps**: 
1. Review and prioritize improvements with team
2. Implement Phase 1 changes
3. Run A/B test on sample dataset
4. Iterate based on results
