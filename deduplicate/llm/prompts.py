from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence

from ie.config import FACT_DEFINITION_INDEX
from ie.types import FactDefinition, FactType

from ..models import FactRecord, Partition

SYSTEM_PROMPT = """
You are a knowledge graph deduplication specialist working on a Discord conversation analysis system.

Context:
- Facts originate from casual Discord conversations and may include slang, abbreviations, and missing context.
- The knowledge graph powers member profiles that surface jobs, projects, skills, and preferences with evidence links.
- Over-merging collapses temporal nuance (e.g., past vs. present jobs) and confuses users.
- Under-merging creates duplicate clutter but is safer; when uncertain, err toward keeping facts separate.

Goal: produce clean, auditable canonical facts that preserve meaning, cite evidence, and respect temporal distinctions.
""".strip()

CONFIDENCE_RULES = """
Confidence Calculation Rules:

1. Identical facts (all key attributes match): confidence = max(c1, c2, ...).
   Example: [0.70, 0.85, 0.80] → 0.85

2. Complementary facts (attributes combine without conflict):
   confidence = weighted_average + 0.05 information gain bonus (cap at 0.95)
   Weights: w_i = c_i / Σ(c_j)
   Example: c = {0.8, 0.6} → weighted average 0.714, final = min(0.714 + 0.05, 0.95) = 0.764

3. Evidence bonus: add 0.10 when the merged fact references ≥5 distinct evidence messages (still cap at 0.95).

4. Conflicting core attributes (entity, timeframe, location, role, etc.) → DO NOT MERGE. Split into separate canonical facts.
""".strip()

EDGE_CASE_RULES = """
Edge Case Handling:

1. Outliers: if one fact clearly differs from a tight cluster, split it out (e.g., [Google, Google, Google, Microsoft]).
2. Partial organization overlaps: "Google" vs. "Google Cloud" → keep separate; "Google" vs. "Google Inc." → merge after normalization.
3. Temporal conflicts: if dates differ by ≤3 months, prefer earliest date and note variance; if >3 months, treat as separate periods.
4. Attribute conflicts: conflicting critical attributes prevent merging; non-critical conflicts favor the most specific value.
5. Conservative default: when merge confidence would fall below 0.70, keep facts separate.
""".strip()

AMBIGUITY_PROTOCOL = """
Handling Uncertainty:

- Confidence ≥ 0.90 → merge confidently.
- 0.70 ≤ confidence < 0.90 → merge but acknowledge caution in merge_reasoning.
- 0.50 ≤ confidence < 0.70 → prefer separate canonical facts and explain ambiguity.
- Confidence < 0.50 → definitely separate facts.

You may emit multiple canonical facts when the input group naturally breaks into sub-clusters. Document any residual uncertainty in merge_reasoning.
""".strip()

MERGE_REASONING_TEMPLATE = """
Merge Reasoning Expectations:

Provide concise but specific reasoning covering:
1. What was merged or split (number of facts, shared subject/object).
2. Key normalizations (e.g., normalize "Google Inc." → "Google", expand "SWE" → "Software Engineer").
3. Attribute resolution (which fact supplied each attribute, how conflicts were handled).
4. Temporal or conflicting data decisions (e.g., used earliest start date to resolve 2020 vs. 2021).
5. Confidence justification tied to evidence overlap and attribute agreement.

Good Example: "Merged 2 facts about the same Google employment. Normalized organization name and expanded role abbreviation. Retained San Francisco location from higher-confidence fact. Confidence 0.87 due to identical evidence and overlapping dates."
Poor Example: "Similar facts merged."
""".strip()

QUALITY_CHECKLIST = """
Quality Validation Checklist (run mentally before responding):

✓ Every merged_from ID exists in the provided candidate_facts.
✓ Evidence list is the union of source evidence (non-empty, no duplicates).
✓ Confidence scores fall within [0.0, 1.0] after bonuses/caps.
✓ Attributes preserve specificity (no information loss versus sources).
✓ canonical_facts list is non-empty only if input facts justified it.
✓ merge_reasoning strings are non-empty and explain the decision.
✓ No contradictions across canonical facts for the subject.
""".strip()

OUTPUT_SCHEMA = """
Output Schema (strict JSON):
{
  "canonical_facts": [
    {
      "type": "WORKS_AT",                    // string, must equal partition fact type
      "subject_id": "12345",                 // string, must equal partition subject
      "object_label": "Google",              // string (normalized label)
      "object_id": "google-inc",             // string|null (stable identifier)
      "object_type": "Organization",         // string|null (entity type)
      "attributes": {                        // object with fact-specific keys
        "organization": "Google",
        "role": "Software Engineer",
        "location": "San Francisco"
      },
      "confidence": 0.87,                    // float within [0.0, 1.0]
      "evidence": ["msg_001", "msg_045"],    // non-empty list of strings
      "timestamp": "2024-01-15T10:23:00Z",   // ISO8601 string or ""
      "merged_from": [147, 148],             // non-empty list of source fact IDs
      "merge_reasoning": "..."               // non-empty explanatory string
    }
  ]
}
""".strip()

DEFAULT_TYPE_GUIDANCE = """
Generic Deduplication Guidance:
- Confirm the object refers to the same real-world entity or concept before merging.
- Respect temporal distinctions unless evidence shows identical timeframes.
- Normalize labels (case, punctuation, abbreviations) and prefer the most specific wording.
""".strip()

TYPE_SPECIFIC_RULES: dict[FactType, str] = {
    FactType.WORKS_AT: """
WORKS_AT Rules:
- Temporal sensitivity: distinct employment periods remain separate unless dates overlap within ~3 months.
- Normalize organization naming ("Google Inc." → "Google"), and expand role abbreviations.
- Merge only when describing the same employment stint with compatible start/end dates.
- Combine locations when one fact is more specific ("SF" + "San Francisco" → "San Francisco").
""".strip(),
    FactType.LIVES_IN: """
LIVES_IN Rules:
- Temporal sensitivity: separate moves and past residences; merge only if evidence indicates the same living situation.
- Normalize place names ("SF" → "San Francisco"), prefer the most precise granularity.
- If timeframe differs ("since 2020" vs. "since 2023"), treat as separate unless evidence reconciles.
""".strip(),
    FactType.TALKS_ABOUT: """
TALKS_ABOUT Rules:
- Aggregation friendly: merge repeated mentions of the same normalized topic.
- Combine sentiment evidence (majority sentiment wins, or mark as "mixed" when conflicting).
- Merge when topics are synonyms or identical after normalization; keep separate for distinct subtopics.
""".strip(),
    FactType.HAS_SKILL: """
HAS_SKILL Rules:
- Skill evolutions (different proficiency levels or years of experience) represent different states—merge only if they describe the same capability at the same level.
- Normalize skill names (e.g., "Py" → "Python") and prefer the most specific phrasing.
- Retain the highest fidelity attributes (e.g., keep numeric years_experience when available).
""".strip(),
    FactType.STUDIED_AT: """
STUDIED_AT Rules:
- Separate distinct programs, campuses, or degree types even at the same institution.
- Merge when evidence points to the same enrollment period and program details.
- Normalize institution names and align graduation year/status fields carefully.
""".strip(),
}


@dataclass(frozen=True)
class AttributePriority:
    critical: Sequence[str]
    important: Sequence[str]
    optional: Sequence[str]


ATTRIBUTE_PRIORITY_OVERRIDES: dict[FactType, AttributePriority] = {
    FactType.WORKS_AT: AttributePriority(
        critical=("organization", "start_date", "end_date"),
        important=("role", "location"),
        optional=(),
    ),
    FactType.LIVES_IN: AttributePriority(
        critical=("location", "since"),
        important=(),
        optional=(),
    ),
    FactType.HAS_SKILL: AttributePriority(
        critical=("skill",),
        important=("proficiency_level", "years_experience"),
        optional=("learning_status",),
    ),
    FactType.STUDIED_AT: AttributePriority(
        critical=("institution", "degree_type"),
        important=("field_of_study", "graduation_year"),
        optional=("status",),
    ),
    FactType.TALKS_ABOUT: AttributePriority(
        critical=("topic",),
        important=("sentiment",),
        optional=(),
    ),
}


def _derive_attribute_priority(definition: FactDefinition) -> AttributePriority:
    critical = tuple(attr.name for attr in definition.attributes if attr.required)
    important = tuple(attr.name for attr in definition.attributes if not attr.required)
    return AttributePriority(critical=critical, important=important, optional=())


ATTRIBUTE_PRIORITIES: dict[FactType, AttributePriority] = {
    fact_type: ATTRIBUTE_PRIORITY_OVERRIDES.get(fact_type, _derive_attribute_priority(definition))
    for fact_type, definition in FACT_DEFINITION_INDEX.items()
}


@dataclass(frozen=True)
class FewShotExample:
    name: str
    fact_types: tuple[FactType, ...]
    description: str
    input_facts: list[dict[str, object]]
    expected_output: dict[str, object]
    takeaways: tuple[str, ...] = ()

    def applies_to(self, fact_type: FactType) -> bool:
        return not self.fact_types or fact_type in self.fact_types


FEW_SHOT_EXAMPLES: tuple[FewShotExample, ...] = (
    FewShotExample(
        name="clear_duplicate_works_at",
        fact_types=(FactType.WORKS_AT,),
        description="Two facts describe the same Google role with compatible dates and complementary attributes.",
        input_facts=[
            {
                "fact_id": 1,
                "object_label": "Google",
                "attributes": {"organization": "Google", "role": "SWE", "start_date": "2022-01-01"},
                "confidence": 0.8,
                "evidence": ["m1", "m3"],
            },
            {
                "fact_id": 2,
                "object_label": "Google Inc.",
                "attributes": {
                    "organization": "Google Inc.",
                    "role": "SorrySoftware Engineer",
                    "location": "San Francisco",
                    "start_date": "2022-01-15",
                },
                "confidence": 0.75,
                "evidence": ["m2"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "WORKS_AT",
                    "subject_id": "user_42",
                    "object_label": "Google",
                    "attributes": {
                        "organization": "Google",
                        "role": "Software Engineer",
                        "location": "San Francisco",
                        "start_date": "2022-01-01",
                    },
                    "confidence": 0.85,
                    "evidence": ["m1", "m2", "m3"],
                    "merged_from": [1, 2],
                    "merge_reasoning": "Merged 2 facts about the same Google employment stint. Normalized organization name, expanded role abbreviation, retained earliest start date. Confidence 0.85 via weighted average + bonus and strong evidence overlap.",
                }
            ]
        },
        takeaways=(
            "Merge when entity and timeframe align after normalization.",
            "Combine complementary attributes and union evidence.",
        ),
    ),
    FewShotExample(
        name="temporal_split_works_at",
        fact_types=(FactType.WORKS_AT,),
        description="Two roles at the same company but non-overlapping dates should remain separate facts.",
        input_facts=[
            {
                "fact_id": 3,
                "object_label": "Acme Corp",
                "attributes": {
                    "organization": "Acme Corp",
                    "role": "Intern",
                    "start_date": "2019-06-01",
                    "end_date": "2019-08-31",
                },
                "confidence": 0.72,
                "evidence": ["m5"],
            },
            {
                "fact_id": 4,
                "object_label": "Acme Corp",
                "attributes": {
                    "organization": "Acme Corp",
                    "role": "Software Engineer",
                    "start_date": "2021-04-01",
                },
                "confidence": 0.78,
                "evidence": ["m6"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "WORKS_AT",
                    "subject_id": "user_42",
                    "object_label": "Acme Corp",
                    "attributes": {
                        "organization": "Acme Corp",
                        "role": "Intern",
                        "start_date": "2019-06-01",
                        "end_date": "2019-08-31",
                    },
                    "confidence": 0.72,
                    "evidence": ["m5"],
                    "merged_from": [3],
                    "merge_reasoning": "Kept internship separate from later role due to non-overlapping dates.",
                },
                {
                    "type": "WORKS_AT",
                    "subject_id": "user_42",
                    "object_label": "Acme Corp",
                    "attributes": {
                        "organization": "Acme Corp",
                        "role": "Software Engineer",
                        "start_date": "2021-04-01",
                    },
                    "confidence": 0.78,
                    "evidence": ["m6"],
                    "merged_from": [4],
                    "merge_reasoning": "Separate employment period; no evidence of overlap with internship.",
                },
            ]
        },
        takeaways=(
            "Temporal conflicts >3 months should be split.",
            "Individual facts remain when no compatible partner exists.",
        ),
    ),
    FewShotExample(
        name="aggregation_talks_about",
        fact_types=(FactType.TALKS_ABOUT,),
        description="Multiple messages mention the same topic with consistent sentiment; merge into one canonical fact.",
        input_facts=[
            {
                "fact_id": 10,
                "object_label": "Rust programming",
                "attributes": {"topic": "Rust", "sentiment": "positive"},
                "confidence": 0.68,
                "evidence": ["m21"],
            },
            {
                "fact_id": 11,
                "object_label": "Rustlang",
                "attributes": {"topic": "Rust", "sentiment": "positive"},
                "confidence": 0.74,
                "evidence": ["m22", "m23"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "TALKS_ABOUT",
                    "subject_id": "user_42",
                    "object_label": "Rust",
                    "attributes": {"topic": "Rust", "sentiment": "positive"},
                    "confidence": 0.74,
                    "evidence": ["m21", "m22", "m23"],
                    "merged_from": [10, 11],
                    "merge_reasoning": "Merged two positive mentions of Rust. Normalized topic name and combined evidence. Confidence boosted via compatible attributes and multiple messages.",
                }
            ]
        },
        takeaways=(
            "Normalize synonymous topic labels before deciding to merge.",
            "Sentiment alignment signals compatibility for TALKS_ABOUT merges.",
        ),
    ),
    FewShotExample(
        name="skill_level_split",
        fact_types=(FactType.HAS_SKILL,),
        description="Same skill mentioned with different proficiency levels should remain separate unless describing the same capability state.",
        input_facts=[
            {
                "fact_id": 30,
                "object_label": "Python",
                "attributes": {"skill": "Python", "proficiency_level": "beginner"},
                "confidence": 0.6,
                "evidence": ["m50"],
            },
            {
                "fact_id": 31,
                "object_label": "Python",
                "attributes": {"skill": "Python", "proficiency_level": "expert"},
                "confidence": 0.65,
                "evidence": ["m51"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "HAS_SKILL",
                    "subject_id": "user_42",
                    "object_label": "Python",
                    "attributes": {"skill": "Python", "proficiency_level": "beginner"},
                    "confidence": 0.60,
                    "evidence": ["m50"],
                    "merged_from": [30],
                    "merge_reasoning": "Kept separate due to conflicting proficiency levels (beginner vs. expert).",
                },
                {
                    "type": "HAS_SKILL",
                    "subject_id": "user_42",
                    "object_label": "Python",
                    "attributes": {"skill": "Python", "proficiency_level": "expert"},
                    "confidence": 0.65,
                    "evidence": ["m51"],
                    "merged_from": [31],
                    "merge_reasoning": "Separate capability state with incompatible proficiency levels; no merge performed.",
                },
            ]
        },
        takeaways=(
            "Conflicting critical attributes force separate canonical facts.",
        ),
    ),
    FewShotExample(
        name="education_merge",
        fact_types=(FactType.STUDIED_AT,),
        description="Two education facts refer to the same program with complementary details.",
        input_facts=[
            {
                "fact_id": 40,
                "object_label": "MIT",
                "attributes": {
                    "institution": "MIT",
                    "degree_type": "BS",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2018,
                },
                "confidence": 0.7,
                "evidence": ["m70"],
            },
            {
                "fact_id": 41,
                "object_label": "Massachusetts Institute of Technology",
                "attributes": {
                    "institution": "Massachusetts Institute of Technology",
                    "degree_type": "Bachelor of Science",
                    "graduation_year": 2018,
                    "status": "alumni",
                },
                "confidence": 0.65,
                "evidence": ["m71", "m72"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "STUDIED_AT",
                    "subject_id": "user_42",
                    "object_label": "Massachusetts Institute of Technology",
                    "attributes": {
                        "institution": "Massachusetts Institute of Technology",
                        "degree_type": "Bachelor of Science",
                        "field_of_study": "Computer Science",
                        "graduation_year": 2018,
                        "status": "alumni",
                    },
                    "confidence": 0.75,
                    "evidence": ["m70", "m71", "m72"],
                    "merged_from": [40, 41],
                    "merge_reasoning": "Merged education facts referencing the same MIT program. Normalized institution name and combined complementary degree/field/status details. Confidence boosted due to aligned graduation year and overlapping evidence.",
                }
            ]
        },
        takeaways=(
            "Normalize institution and degree terminology before merging.",
            "Complementary attributes justify confidence bonus when consistent.",
        ),
    ),
    FewShotExample(
        name="conflicting_core_attribute_split",
        fact_types=(),
        description="Demonstrates conservative splitting when core attributes conflict despite overlapping evidence.",
        input_facts=[
            {
                "fact_id": 90,
                "object_label": "Berlin",
                "attributes": {"location": "Berlin"},
                "confidence": 0.62,
                "evidence": ["m90"],
            },
            {
                "fact_id": 91,
                "object_label": "Munich",
                "attributes": {"location": "Munich"},
                "confidence": 0.64,
                "evidence": ["m91"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "LIVES_IN",
                    "subject_id": "user_42",
                    "object_label": "Berlin",
                    "attributes": {"location": "Berlin"},
                    "confidence": 0.62,
                    "evidence": ["m90"],
                    "merged_from": [90],
                    "merge_reasoning": "Location conflict (Berlin vs. Munich); kept separate to preserve accuracy.",
                },
                {
                    "type": "LIVES_IN",
                    "subject_id": "user_42",
                    "object_label": "Munich",
                    "attributes": {"location": "Munich"},
                    "confidence": 0.64,
                    "evidence": ["m91"],
                    "merged_from": [91],
                    "merge_reasoning": "Conflicting residence city; conservative split despite similar structure.",
                },
            ]
        },
        takeaways=(
            "Conflicting critical attributes (like location) require separate canonical facts.",
            "When doubtful, communicate the conservative decision in merge_reasoning.",
        ),
    ),
)


def select_examples(fact_type: FactType, *, limit: int = 4) -> list[FewShotExample]:
    specific = [example for example in FEW_SHOT_EXAMPLES if fact_type in example.fact_types]
    generic = [example for example in FEW_SHOT_EXAMPLES if not example.fact_types and example not in specific]
    combined = (specific + generic)[:limit]
    if combined:
        return combined
    return list(FEW_SHOT_EXAMPLES[:limit])


def format_example(example: FewShotExample) -> str:
    input_json = json.dumps(example.input_facts, indent=2, sort_keys=True)
    output_json = json.dumps(example.expected_output, indent=2, sort_keys=True)
    lines = [
        f"Example: {example.name}",
        f"Scenario: {example.description}",
        "Input Facts:",
        input_json,
        "Expected Output:",
        output_json,
    ]
    if example.takeaways:
        lines.append("Key Takeaways:")
        lines.extend(f"- {point}" for point in example.takeaways)
    return "\n".join(lines)


def _format_attribute_priorities(fact_type: FactType) -> str:
    priorities = ATTRIBUTE_PRIORITIES[fact_type]
    critical = ", ".join(priorities.critical) or "(none specified)"
    important = ", ".join(priorities.important) or "(none)"
    optional = ", ".join(priorities.optional) or "(none)"
    return (
        "Attribute Priorities:\n"
        f"- Critical: {critical}\n"
        f"- Important: {important}\n"
        f"- Optional: {optional}"
    )


def build_messages(partition: Partition, candidate_facts: Iterable[FactRecord]) -> list[dict[str, str]]:
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
    subject_display = partition.subject_name or partition.subject_id
    type_guidance = TYPE_SPECIFIC_RULES.get(partition.fact_type, DEFAULT_TYPE_GUIDANCE)
    priority_text = _format_attribute_priorities(partition.fact_type)
    examples = select_examples(partition.fact_type)
    examples_text = "\n\n".join(format_example(example) for example in examples)

    user_sections = [
        "DEDUPLICATION TASK",
        f"Fact Type: {partition.fact_type.value}",
        f"Subject: {subject_display}",
        "",
        type_guidance,
        "",
        priority_text,
        "",
        CONFIDENCE_RULES,
        "",
        EDGE_CASE_RULES,
        "",
        AMBIGUITY_PROTOCOL,
        "",
        MERGE_REASONING_TEMPLATE,
        "",
        QUALITY_CHECKLIST,
        "",
        OUTPUT_SCHEMA,
        "",
        "Few-Shot Examples:",
        examples_text,
        "",
        "Input Facts:",
        facts_json,
        "",
        "Return canonical_facts JSON exactly matching the schema above.",
    ]
    user_prompt = "\n".join(section for section in user_sections if section is not None)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt.strip()},
    ]
