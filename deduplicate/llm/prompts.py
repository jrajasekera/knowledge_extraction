from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum

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

BASE_CONFIDENCE_RULES = """
Base Confidence Calculation:

1. Identical facts (all key attributes match): confidence = max(c1, c2, ...).
2. Complementary facts (attributes combine without conflict): confidence = weighted_average + 0.05 (cap at 0.95).
   Weights: w_i = c_i / SUM(c_j)
3. Evidence bonus: add 0.10 when the merged fact references five or more distinct evidence messages (still cap at 0.95).
4. Conflicting core attributes (entity, timeframe, location, role, sentiment, etc.) -> DO NOT MERGE. Split into separate canonical facts.
""".strip()

BASE_EDGE_CASE_RULES = """
General Edge Case Handling:

- Outliers: if one fact clearly differs from a tight cluster, split it out (e.g., [Google, Google, Google, Microsoft]).
- Partial overlaps: distinguish between synonyms (merge) and different entities (keep separate).
- Temporal conflicts: if dates differ by <=3 months, prefer earliest date and note variance; if >3 months, treat as separate periods.
- Conservative default: when merge confidence would fall below 0.70, keep facts separate.
""".strip()

AMBIGUITY_PROTOCOL = """
Handling Uncertainty:

- Confidence >= 0.90 -> merge confidently.
- 0.70 <= confidence < 0.90 -> merge but acknowledge caution in merge_reasoning.
- 0.50 <= confidence < 0.70 -> prefer separate canonical facts and explain ambiguity.
- Confidence < 0.50 -> keep facts separate.

You may emit multiple canonical facts when the input group naturally breaks into sub-clusters. Document residual uncertainty in merge_reasoning.
""".strip()

MERGE_REASONING_TEMPLATE = """
Merge Reasoning Expectations:

Provide concise but specific reasoning covering:
1. What was merged or split (number of facts, shared subject/object).
2. Key normalizations (e.g., normalize "Google Inc." -> "Google", expand "SWE" -> "Software Engineer").
3. Attribute resolution (which fact supplied each attribute, how conflicts were handled).
4. Temporal or conflicting data decisions (e.g., used earliest start date to resolve 2020 vs. 2021).
5. Confidence justification tied to evidence overlap and attribute agreement.

Good Example: "Merged 2 facts about the same Google employment. Normalized organization name and expanded role abbreviation. Retained San Francisco location from higher-confidence fact. Confidence 0.87 due to identical evidence and overlapping dates."
Poor Example: "Similar facts merged."
""".strip()

QUALITY_CHECKLIST = """
Quality Validation Checklist (run mentally before responding):

- Ensure every merged_from ID exists in the provided candidate_facts.
- Ensure the evidence list is the union of source evidence (non-empty, no duplicates).
- Ensure confidence scores fall within [0.0, 1.0] after bonuses/caps.
- Confirm attributes preserve specificity (no information loss versus sources).
- Confirm canonical_facts is non-empty only when input facts justified it.
- Confirm merge_reasoning strings are non-empty and explain the decision.
- Confirm there are no contradictions across canonical facts for the subject.
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


class FactCategory(str, Enum):
    TEMPORAL_EMPLOYMENT = "Temporal Employment/Education"
    TEMPORAL_LOCATION = "Temporal Location/Status"
    AGGREGABLE_TOPICS = "Aggregable Topics/Interests"
    STABLE_SKILLS = "Stable Skills/Beliefs"
    PREFERENCE_SENTIMENT = "Preference/Sentiment"
    RELATIONSHIPS_EVENTS = "Relationships/Events"


@dataclass(frozen=True)
class EdgeCaseGuidance:
    scenario: str
    resolution: str
    description: str | None = None

    def render(self) -> str:
        if self.description:
            return f"- {self.scenario}: {self.description}\n  Resolution: {self.resolution.strip()}"
        return f"- {self.scenario}: {self.resolution.strip()}"


@dataclass(frozen=True)
class FewShotExample:
    name: str
    category: FactCategory
    fact_types: tuple[FactType, ...]
    description: str
    input_facts: list[dict[str, object]]
    expected_output: dict[str, object]
    explanation: str

    def applies_to(self, fact_type: FactType) -> bool:
        return not self.fact_types or fact_type in self.fact_types


@dataclass(frozen=True)
class CategoryTemplate:
    name: FactCategory
    fact_types: tuple[FactType, ...]
    merge_philosophy: str
    temporal_sensitivity: str
    default_merge_strategy: str
    attribute_normalization_rules: dict[str, str]
    identical_confidence_rule: str
    complementary_confidence_rule: str
    conflict_handling: str
    edge_cases: tuple[EdgeCaseGuidance, ...] = ()
    few_shot_names: tuple[str, ...] = ()
    fact_type_guidance: dict[FactType, str] = field(default_factory=dict)

    def overview_text(self) -> str:
        return "\n".join(
            [
                f"Category: {self.name.value}",
                f"Merge Philosophy: {self.merge_philosophy.strip()}",
                f"Temporal Sensitivity: {self.temporal_sensitivity}",
                f"Default Merge Strategy: {self.default_merge_strategy}",
            ]
        )

    def guidance_for_fact_type(self, fact_type: FactType) -> str:
        return self.fact_type_guidance.get(fact_type, "")

    def normalization_text(self) -> str:
        if not self.attribute_normalization_rules:
            return ""
        lines = ["Attribute Normalization:"]
        for attribute, rule in self.attribute_normalization_rules.items():
            lines.append(f"- {attribute}: {rule}")
        return "\n".join(lines)

    def confidence_text(self) -> str:
        return "\n".join(
            [
                "Category Confidence Guidance:",
                f"- Identical facts: {self.identical_confidence_rule.strip()}",
                f"- Complementary facts: {self.complementary_confidence_rule.strip()}",
                f"- Conflicts: {self.conflict_handling.strip()}",
            ]
        )

    def edge_case_text(self) -> str:
        if not self.edge_cases:
            return ""
        rendered = "\n".join(case.render() for case in self.edge_cases)
        return f"Category Edge Cases:\n{rendered}"

    def examples_for_fact_type(self, fact_type: FactType) -> list[FewShotExample]:
        selected: list[FewShotExample] = []
        for example_name in self.few_shot_names:
            example = FEW_SHOT_EXAMPLES.get(example_name)
            if example and example.applies_to(fact_type):
                selected.append(example)
        if selected:
            return selected
        # Fallback: any example in registry for this fact type and category
        fallback = [
            example
            for example in FEW_SHOT_EXAMPLES.values()
            if example.category == self.name and example.applies_to(fact_type)
        ]
        return fallback[:4]


FACT_CATEGORY_BY_TYPE: dict[FactType, FactCategory] = {
    FactType.WORKS_AT: FactCategory.TEMPORAL_EMPLOYMENT,
    FactType.STUDIED_AT: FactCategory.TEMPORAL_EMPLOYMENT,
    FactType.WORKING_ON: FactCategory.TEMPORAL_EMPLOYMENT,
    FactType.PREVIOUSLY: FactCategory.TEMPORAL_EMPLOYMENT,
    FactType.LIVES_IN: FactCategory.TEMPORAL_LOCATION,
    FactType.EXPERIENCED: FactCategory.TEMPORAL_LOCATION,
    FactType.TALKS_ABOUT: FactCategory.AGGREGABLE_TOPICS,
    FactType.CURIOUS_ABOUT: FactCategory.AGGREGABLE_TOPICS,
    FactType.CARES_ABOUT: FactCategory.AGGREGABLE_TOPICS,
    FactType.REMEMBERS: FactCategory.AGGREGABLE_TOPICS,
    FactType.WITNESSED: FactCategory.AGGREGABLE_TOPICS,
    FactType.HAS_SKILL: FactCategory.STABLE_SKILLS,
    FactType.BELIEVES: FactCategory.STABLE_SKILLS,
    FactType.PREFERS: FactCategory.PREFERENCE_SENTIMENT,
    FactType.RECOMMENDS: FactCategory.PREFERENCE_SENTIMENT,
    FactType.AVOIDS: FactCategory.PREFERENCE_SENTIMENT,
    FactType.DISLIKES: FactCategory.PREFERENCE_SENTIMENT,
    FactType.ENJOYS: FactCategory.PREFERENCE_SENTIMENT,
    FactType.PLANS_TO: FactCategory.PREFERENCE_SENTIMENT,
    FactType.CLOSE_TO: FactCategory.RELATIONSHIPS_EVENTS,
    FactType.RELATED_TO: FactCategory.RELATIONSHIPS_EVENTS,
    FactType.ATTENDED_EVENT: FactCategory.RELATIONSHIPS_EVENTS,
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
    FactType.PREFERS: AttributePriority(
        critical=("target",),
        important=("reason", "context"),
        optional=("intensity",),
    ),
    FactType.AVOIDS: AttributePriority(
        critical=("target",),
        important=("reason", "severity"),
        optional=("timeframe",),
    ),
    FactType.DISLIKES: AttributePriority(
        critical=("target",),
        important=("reason",),
        optional=(),
    ),
    FactType.RECOMMENDS: AttributePriority(
        critical=("target",),
        important=("recommendation_strength", "context"),
        optional=("reason",),
    ),
    FactType.ATTENDED_EVENT: AttributePriority(
        critical=("event_name", "date"),
        important=("role", "format", "location"),
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


CATEGORY_TEMPLATES: dict[FactCategory, CategoryTemplate] = {
    FactCategory.TEMPORAL_EMPLOYMENT: CategoryTemplate(
        name=FactCategory.TEMPORAL_EMPLOYMENT,
        fact_types=(
            FactType.WORKS_AT,
            FactType.STUDIED_AT,
            FactType.WORKING_ON,
            FactType.PREVIOUSLY,
        ),
        merge_philosophy=(
            "Employment, education, and project facts are highly temporal. Distinct time periods represent"
            " separate canonical facts. Only merge when the same position/program spans the same timeframe."
        ),
        temporal_sensitivity="high",
        default_merge_strategy="conservative",
        attribute_normalization_rules={
            "organization": "Normalize company names (e.g., 'Google Inc.' -> 'Google').",
            "role": "Expand abbreviations (e.g., 'SWE' -> 'Software Engineer').",
            "location": "Use full city names (e.g., 'SF' -> 'San Francisco').",
            "institution": "Use official school names (e.g., 'Stanford' -> 'Stanford University').",
        },
        identical_confidence_rule="Use max confidence when organization/institution, role/program, and timeframe match exactly.",
        complementary_confidence_rule="Use weighted average +0.05 when facts add compatible attributes for the same stint (e.g., one provides role, the other location).",
        conflict_handling="Do not merge when organizations differ or start/end dates differ by more than ~3 months.",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="Promotion vs. new stint",
                description="Same organization but non-overlapping dates or different seniority titles.",
                resolution="Treat as separate facts unless evidence states it is the same ongoing role.",
            ),
            EdgeCaseGuidance(
                scenario="Company rebrand/acquisition",
                description="Organization renamed during employment period.",
                resolution="Merge if dates overlap; note normalization and acquisition in reasoning.",
            ),
        ),
        few_shot_names=(
            "clear_duplicate_works_at",
            "temporal_split_works_at",
            "education_merge",
            "project_complementary",
        ),
        fact_type_guidance={
            FactType.WORKS_AT: (
                "WORKS_AT: Keep separate entries for distinct employment periods. Merge only when describing the same stint"
                " with overlapping dates and compatible roles."
            ),
            FactType.STUDIED_AT: (
                "STUDIED_AT: Do not merge different programs, campuses, or degree types even at the same institution."
                " Merge only when enrollment period and program details align."
            ),
            FactType.WORKING_ON: (
                "WORKING_ON: Separate projects when scope or collaborators differ. Merge when describing the same project"
                " with complementary attributes (role, timeframe, format)."
            ),
            FactType.PREVIOUSLY: (
                "PREVIOUSLY: Use to capture historical states. Merge only when evidence clearly references the same past event."
            ),
        },
    ),
    FactCategory.TEMPORAL_LOCATION: CategoryTemplate(
        name=FactCategory.TEMPORAL_LOCATION,
        fact_types=(FactType.LIVES_IN, FactType.EXPERIENCED),
        merge_philosophy=(
            "Location and experience facts change over time. Preserve chronological differences to reflect moves and life stages."
        ),
        temporal_sensitivity="high",
        default_merge_strategy="conservative",
        attribute_normalization_rules={
            "location": "Normalize to full city/region names (e.g., 'NYC' -> 'New York City').",
            "since": "Prefer ISO dates; keep textual phrases if no structured date exists.",
        },
        identical_confidence_rule="Use max confidence when location/stage and timeframe align exactly.",
        complementary_confidence_rule="Use weighted average +0.05 when facts add detail about the same ongoing location/status (e.g., one adds neighborhood).",
        conflict_handling="Do not merge conflicting cities or mutually exclusive stages; treat them as separate residence/experience periods.",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="Seasonal or dual residency",
                resolution="If evidence shows simultaneous residences, note dual locations but keep separate canonical facts.",
            ),
            EdgeCaseGuidance(
                scenario="Ambiguous timeframe",
                resolution="When dates conflict, choose conservative split and explain ambiguity.",
            ),
        ),
        few_shot_names=(
            "residence_split",
            "residence_merge_with_specificity",
            "experience_stage_split",
        ),
        fact_type_guidance={
            FactType.LIVES_IN: (
                "LIVES_IN: Merge only when describing the same living situation. Moves or relocations at different times stay separate."
            ),
            FactType.EXPERIENCED: (
                "EXPERIENCED: Treat different phases (e.g., planning, during, after) as distinct unless evidence shows they describe the same stage."
            ),
        },
    ),
    FactCategory.AGGREGABLE_TOPICS: CategoryTemplate(
        name=FactCategory.AGGREGABLE_TOPICS,
        fact_types=(
            FactType.TALKS_ABOUT,
            FactType.CURIOUS_ABOUT,
            FactType.CARES_ABOUT,
            FactType.REMEMBERS,
            FactType.WITNESSED,
        ),
        merge_philosophy=(
            "Topic and interest facts are aggregation-friendly. Merge repeated mentions to show frequency while respecting sentiment conflicts."
        ),
        temporal_sensitivity="low",
        default_merge_strategy="moderate",
        attribute_normalization_rules={
            "topic": "Normalize synonyms (e.g., 'Rustlang' -> 'Rust').",
            "sentiment": "Aggregate consistent sentiments; mark as 'mixed' when evidence conflicts.",
        },
        identical_confidence_rule="Use max confidence when topic and sentiment align across facts.",
        complementary_confidence_rule="Boost confidence (max +0.10) when additional facts add context or reinforce sustained interest.",
        conflict_handling="Split when topics differ in specificity (e.g., 'AI' vs. 'Machine Learning') or sentiment is contradictory.",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="High-frequency mentions",
                resolution="Merge all mentions, union evidence, and highlight count in reasoning.",
            ),
            EdgeCaseGuidance(
                scenario="Sentiment change",
                resolution="When sentiment shifts (positive -> negative), keep separate facts and describe timeline if evidence supports.",
            ),
        ),
        few_shot_names=(
            "aggregation_talks_about",
            "topic_sentiment_conflict",
            "memory_split",
        ),
        fact_type_guidance={
            FactType.TALKS_ABOUT: (
                "TALKS_ABOUT: Merge mentions of the same normalized topic; combine evidence to reflect frequency."
            ),
            FactType.CURIOUS_ABOUT: (
                "CURIOUS_ABOUT: Merge when curiosity targets align; note learning stage or intent."
            ),
            FactType.CARES_ABOUT: (
                "CARES_ABOUT: Merge values aligned around the same cause; explain intensity if provided."
            ),
            FactType.REMEMBERS: (
                "REMEMBERS: Distinguish between separate memories. Merge only when they refer to the same recalled event."
            ),
            FactType.WITNESSED: (
                "WITNESSED: Merge reports of the same incident; split distinct events even if related."
            ),
        },
    ),
    FactCategory.STABLE_SKILLS: CategoryTemplate(
        name=FactCategory.STABLE_SKILLS,
        fact_types=(FactType.HAS_SKILL, FactType.BELIEVES),
        merge_philosophy=(
            "Skills and beliefs evolve but remain anchored to the same subject. Merge only when proficiency/stance is consistent."
        ),
        temporal_sensitivity="medium",
        default_merge_strategy="moderate",
        attribute_normalization_rules={
            "skill": "Normalize skill names (e.g., 'Py' -> 'Python').",
            "proficiency_level": "Standardize levels (beginner/intermediate/expert).",
        },
        identical_confidence_rule="Use max confidence when proficiency/stance matches exactly.",
        complementary_confidence_rule="Use weighted average +0.05 when facts add context (e.g., years of experience) without conflicting proficiency.",
        conflict_handling="Keep separate when proficiency levels or belief polarity differ.",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="Skill progression",
                resolution="Treat significant proficiency changes as separate unless evidence shows old data is obsolete.",
            ),
            EdgeCaseGuidance(
                scenario="Belief nuance",
                resolution="If belief wording differs but sentiment aligns, merge and capture nuance in attributes or reasoning.",
            ),
        ),
        few_shot_names=(
            "skill_level_split",
            "skill_complementary_details",
            "belief_merge",
        ),
        fact_type_guidance={
            FactType.HAS_SKILL: (
                "HAS_SKILL: Keep separate when proficiency level or years of experience conflict. Merge complementary skill details when levels align."
            ),
            FactType.BELIEVES: (
                "BELIEVES: Merge only when statements express the same stance. Capture nuances (e.g., qualifiers) in attributes or reasoning."
            ),
        },
    ),
    FactCategory.PREFERENCE_SENTIMENT: CategoryTemplate(
        name=FactCategory.PREFERENCE_SENTIMENT,
        fact_types=(
            FactType.PREFERS,
            FactType.RECOMMENDS,
            FactType.AVOIDS,
            FactType.DISLIKES,
            FactType.ENJOYS,
            FactType.PLANS_TO,
        ),
        merge_philosophy=(
            "Preferences, recommendations, and plans typically persist but sentiment/polarity matters. Merge only when sentiment and target align."
        ),
        temporal_sensitivity="medium",
        default_merge_strategy="moderate",
        attribute_normalization_rules={
            "target": "Normalize product/place names; preserve most specific variant.",
            "sentiment": "Respect polarity; do not merge positive with negative.",
        },
        identical_confidence_rule="Use max confidence when subject expresses the same stance toward the same target.",
        complementary_confidence_rule="Use weighted average +0.05 when facts add context (reason, strength, timeframe) without changing sentiment.",
        conflict_handling="Keep separate when sentiment, intention, or timeframe conflicts (e.g., 'plans to travel' vs. 'no longer plans').",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="Preference evolution",
                resolution="If a preference flips (e.g., from love to dislike), keep separate facts and document timeline.",
            ),
            EdgeCaseGuidance(
                scenario="Conditional plans",
                resolution="If plans depend on conditions, merge only when conditions align; otherwise keep distinct entries.",
            ),
        ),
        few_shot_names=(
            "preference_merge",
            "preference_conflict_split",
            "plan_with_timeframe",
        ),
        fact_type_guidance={
            FactType.PREFERS: (
                "PREFERS: Merge endorsements of the same target when sentiment matches; differentiate intensity via attributes."
            ),
            FactType.RECOMMENDS: (
                "RECOMMENDS: Merge repeated recommendations for identical targets; preserve strongest context and reasoning."
            ),
            FactType.AVOIDS: (
                "AVOIDS/DISLIKES: Keep separate from positive sentiments; merge only when avoidance reasons align."
            ),
            FactType.DISLIKES: (
                "DISLIKES: Separate entries for distinct triggers; merge identical complaints."
            ),
            FactType.ENJOYS: (
                "ENJOYS: Merge repeated enjoyment of same activity; summarize frequency in reasoning."
            ),
            FactType.PLANS_TO: (
                "PLANS_TO: Merge plans only when the goal and timeframe remain consistent; otherwise keep separate."
            ),
        },
    ),
    FactCategory.RELATIONSHIPS_EVENTS: CategoryTemplate(
        name=FactCategory.RELATIONSHIPS_EVENTS,
        fact_types=(FactType.CLOSE_TO, FactType.RELATED_TO, FactType.ATTENDED_EVENT),
        merge_philosophy=(
            "Relationships and events are identity-based. Each unique person pair or event instance should remain distinct unless evidence shows exact duplication."
        ),
        temporal_sensitivity="medium",
        default_merge_strategy="conservative",
        attribute_normalization_rules={
            "event_name": "Normalize event titles; include year/location if provided.",
            "relationship_type": "Standardize relationship labels (e.g., 'sis' -> 'sister').",
        },
        identical_confidence_rule="Use max confidence when participants and event details align exactly.",
        complementary_confidence_rule="Use weighted average +0.05 when facts add complementary details (e.g., role or format) about the same event/relationship.",
        conflict_handling="Do not merge different people or events; conflicting locations/dates require separate canonical facts.",
        edge_cases=(
            EdgeCaseGuidance(
                scenario="Recurring event",
                resolution="Treat each year/instance as separate unless evidence states it is the same occurrence.",
            ),
            EdgeCaseGuidance(
                scenario="Relationship context",
                resolution="If relationship basis differs (family vs. coworker), clarify in reasoning or keep separate.",
            ),
        ),
        few_shot_names=(
            "relationship_duplicate",
            "relationship_conflict",
            "event_complementary",
        ),
        fact_type_guidance={
            FactType.CLOSE_TO: (
                "CLOSE_TO: Verify the same counterpart person. Merge when evidence repeats the same relationship justification."
            ),
            FactType.RELATED_TO: (
                "RELATED_TO: Relationship type must match; distinguish different family ties (e.g., cousin vs. sibling)."
            ),
            FactType.ATTENDED_EVENT: (
                "ATTENDED_EVENT: Merge attendance reports when event name, date, and role align. Separate distinct dates or formats."
            ),
        },
    ),
}


FEW_SHOT_EXAMPLES: dict[str, FewShotExample] = {
    "clear_duplicate_works_at": FewShotExample(
        name="clear_duplicate_works_at",
        category=FactCategory.TEMPORAL_EMPLOYMENT,
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
                    "role": "Software Engineer",
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
        explanation="Perfect duplicate with only formatting differences.",
    ),
    "temporal_split_works_at": FewShotExample(
        name="temporal_split_works_at",
        category=FactCategory.TEMPORAL_EMPLOYMENT,
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
        explanation="Non-overlapping dates signal distinct employment periods.",
    ),
    "education_merge": FewShotExample(
        name="education_merge",
        category=FactCategory.TEMPORAL_EMPLOYMENT,
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
                    "merge_reasoning": "Merged education facts referencing the same MIT program. Normalized institution name and combined complementary degree, field, and status details.",
                }
            ]
        },
        explanation="Complementary education attributes justify merge.",
    ),
    "project_complementary": FewShotExample(
        name="project_complementary",
        category=FactCategory.TEMPORAL_EMPLOYMENT,
        fact_types=(FactType.WORKING_ON,),
        description="Project facts combine role and collaboration details for the same timeframe.",
        input_facts=[
            {
                "fact_id": 60,
                "object_label": "Open Source Bot",
                "attributes": {
                    "project": "Open Source Bot",
                    "role": "Maintainer",
                    "start_date": "2024-04",
                },
                "confidence": 0.66,
                "evidence": ["m80"],
            },
            {
                "fact_id": 61,
                "object_label": "Open Source Bot",
                "attributes": {
                    "project": "Open Source Bot",
                    "collaboration_mode": "team",
                    "project_type": "open-source",
                    "start_date": "2024-04",
                },
                "confidence": 0.64,
                "evidence": ["m81"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "WORKING_ON",
                    "subject_id": "user_42",
                    "object_label": "Open Source Bot",
                    "attributes": {
                        "project": "Open Source Bot",
                        "role": "Maintainer",
                        "collaboration_mode": "team",
                        "project_type": "open-source",
                        "start_date": "2024-04",
                    },
                    "confidence": 0.71,
                    "evidence": ["m80", "m81"],
                    "merged_from": [60, 61],
                    "merge_reasoning": "Merged matching project facts describing the same Open Source Bot effort. Combined role and collaboration details; timeframe identical.",
                }
            ]
        },
        explanation="Project facts with matching timeframe and complementary details should merge.",
    ),
    "residence_split": FewShotExample(
        name="residence_split",
        category=FactCategory.TEMPORAL_LOCATION,
        fact_types=(FactType.LIVES_IN,),
        description="Residences in different cities stay separate despite similar evidence.",
        input_facts=[
            {
                "fact_id": 70,
                "object_label": "Berlin",
                "attributes": {"location": "Berlin", "since": "2023-01"},
                "confidence": 0.62,
                "evidence": ["m90"],
            },
            {
                "fact_id": 71,
                "object_label": "Munich",
                "attributes": {"location": "Munich", "since": "2024-02"},
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
                    "attributes": {"location": "Berlin", "since": "2023-01"},
                    "confidence": 0.62,
                    "evidence": ["m90"],
                    "merged_from": [70],
                    "merge_reasoning": "Location conflict (Berlin vs. Munich); kept separate to preserve accuracy.",
                },
                {
                    "type": "LIVES_IN",
                    "subject_id": "user_42",
                    "object_label": "Munich",
                    "attributes": {"location": "Munich", "since": "2024-02"},
                    "confidence": 0.64,
                    "evidence": ["m91"],
                    "merged_from": [71],
                    "merge_reasoning": "Conflicting residence city; conservative split despite similar structure.",
                },
            ]
        },
        explanation="Different locations imply different residences over time.",
    ),
    "residence_merge_with_specificity": FewShotExample(
        name="residence_merge_with_specificity",
        category=FactCategory.TEMPORAL_LOCATION,
        fact_types=(FactType.LIVES_IN,),
        description="Same residence with complementary location detail.",
        input_facts=[
            {
                "fact_id": 72,
                "object_label": "SF",
                "attributes": {"location": "SF", "since": "2022"},
                "confidence": 0.6,
                "evidence": ["m92"],
            },
            {
                "fact_id": 73,
                "object_label": "San Francisco",
                "attributes": {"location": "San Francisco", "since": "2022-05"},
                "confidence": 0.65,
                "evidence": ["m93"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "LIVES_IN",
                    "subject_id": "user_42",
                    "object_label": "San Francisco",
                    "attributes": {"location": "San Francisco", "since": "2022-05"},
                    "confidence": 0.7,
                    "evidence": ["m92", "m93"],
                    "merged_from": [72, 73],
                    "merge_reasoning": "Merged facts referencing same residence. Normalized 'SF' to 'San Francisco' and retained more precise 'since' date.",
                }
            ]
        },
        explanation="Complementary normalization for same residence.",
    ),
    "experience_stage_split": FewShotExample(
        name="experience_stage_split",
        category=FactCategory.TEMPORAL_LOCATION,
        fact_types=(FactType.EXPERIENCED,),
        description="Different experience stages should stay separate.",
        input_facts=[
            {
                "fact_id": 74,
                "object_label": "Chronic illness",
                "attributes": {"stage": "diagnosed", "timestamp": "2023-01-05"},
                "confidence": 0.65,
                "evidence": ["m94"],
            },
            {
                "fact_id": 75,
                "object_label": "Chronic illness",
                "attributes": {"stage": "in remission", "timestamp": "2023-12-10"},
                "confidence": 0.67,
                "evidence": ["m95"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "EXPERIENCED",
                    "subject_id": "user_42",
                    "object_label": "Chronic illness",
                    "attributes": {"stage": "diagnosed", "timestamp": "2023-01-05"},
                    "confidence": 0.65,
                    "evidence": ["m94"],
                    "merged_from": [74],
                    "merge_reasoning": "Diagnosis stage recorded separately from remission stage.",
                },
                {
                    "type": "EXPERIENCED",
                    "subject_id": "user_42",
                    "object_label": "Chronic illness",
                    "attributes": {"stage": "in remission", "timestamp": "2023-12-10"},
                    "confidence": 0.67,
                    "evidence": ["m95"],
                    "merged_from": [75],
                    "merge_reasoning": "Distinct stage (in remission) kept separate from diagnosis.",
                },
            ]
        },
        explanation="Experience stages represent different temporal states.",
    ),
    "aggregation_talks_about": FewShotExample(
        name="aggregation_talks_about",
        category=FactCategory.AGGREGABLE_TOPICS,
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
                    "confidence": 0.78,
                    "evidence": ["m21", "m22", "m23"],
                    "merged_from": [10, 11],
                    "merge_reasoning": "Merged two positive mentions of Rust. Normalized topic name and combined evidence. Confidence boosted via compatible attributes and multiple messages.",
                }
            ]
        },
        explanation="Compatible mentions aggregate into one canonical fact.",
    ),
    "topic_sentiment_conflict": FewShotExample(
        name="topic_sentiment_conflict",
        category=FactCategory.AGGREGABLE_TOPICS,
        fact_types=(FactType.TALKS_ABOUT, FactType.CURIOUS_ABOUT),
        description="Opposing sentiments about the same topic should remain separate.",
        input_facts=[
            {
                "fact_id": 110,
                "object_label": "JavaScript",
                "attributes": {"topic": "JavaScript", "sentiment": "positive"},
                "confidence": 0.62,
                "evidence": ["m400"],
            },
            {
                "fact_id": 111,
                "object_label": "JavaScript",
                "attributes": {"topic": "JavaScript", "sentiment": "frustrated"},
                "confidence": 0.6,
                "evidence": ["m401"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "TALKS_ABOUT",
                    "subject_id": "user_42",
                    "object_label": "JavaScript",
                    "attributes": {"topic": "JavaScript", "sentiment": "positive"},
                    "confidence": 0.62,
                    "evidence": ["m400"],
                    "merged_from": [110],
                    "merge_reasoning": "Positive sentiment kept separate from conflicting frustrated sentiment.",
                },
                {
                    "type": "TALKS_ABOUT",
                    "subject_id": "user_42",
                    "object_label": "JavaScript",
                    "attributes": {"topic": "JavaScript", "sentiment": "frustrated"},
                    "confidence": 0.6,
                    "evidence": ["m401"],
                    "merged_from": [111],
                    "merge_reasoning": "Conflicting sentiment indicates separate canonical fact.",
                },
            ]
        },
        explanation="Contrasting sentiments should not merge.",
    ),
    "memory_split": FewShotExample(
        name="memory_split",
        category=FactCategory.AGGREGABLE_TOPICS,
        fact_types=(FactType.REMEMBERS,),
        description="Distinct memories about different events should remain separate.",
        input_facts=[
            {
                "fact_id": 112,
                "object_label": "Hackathon win",
                "attributes": {"topic": "Hackathon", "timestamp": "2019-10"},
                "confidence": 0.58,
                "evidence": ["m402"],
            },
            {
                "fact_id": 113,
                "object_label": "Conference talk",
                "attributes": {"topic": "Conference Talk", "timestamp": "2021-07"},
                "confidence": 0.6,
                "evidence": ["m403"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "REMEMBERS",
                    "subject_id": "user_42",
                    "object_label": "Hackathon win",
                    "attributes": {"topic": "Hackathon", "timestamp": "2019-10"},
                    "confidence": 0.58,
                    "evidence": ["m402"],
                    "merged_from": [112],
                    "merge_reasoning": "Separate memory about hackathon win.",
                },
                {
                    "type": "REMEMBERS",
                    "subject_id": "user_42",
                    "object_label": "Conference talk",
                    "attributes": {"topic": "Conference Talk", "timestamp": "2021-07"},
                    "confidence": 0.6,
                    "evidence": ["m403"],
                    "merged_from": [113],
                    "merge_reasoning": "Distinct memory about giving a conference talk.",
                },
            ]
        },
        explanation="Different memories stay separate.",
    ),
    "skill_level_split": FewShotExample(
        name="skill_level_split",
        category=FactCategory.STABLE_SKILLS,
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
                    "confidence": 0.6,
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
        explanation="Conflicting proficiency states stay separate.",
    ),
    "skill_complementary_details": FewShotExample(
        name="skill_complementary_details",
        category=FactCategory.STABLE_SKILLS,
        fact_types=(FactType.HAS_SKILL,),
        description="Complementary skill facts merge when proficiency aligns.",
        input_facts=[
            {
                "fact_id": 120,
                "object_label": "Go",
                "attributes": {"skill": "Go", "proficiency_level": "intermediate"},
                "confidence": 0.63,
                "evidence": ["m520"],
            },
            {
                "fact_id": 121,
                "object_label": "Golang",
                "attributes": {
                    "skill": "Go",
                    "years_experience": 3,
                    "proficiency_level": "intermediate",
                },
                "confidence": 0.62,
                "evidence": ["m521"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "HAS_SKILL",
                    "subject_id": "user_42",
                    "object_label": "Go",
                    "attributes": {
                        "skill": "Go",
                        "proficiency_level": "intermediate",
                        "years_experience": 3,
                    },
                    "confidence": 0.69,
                    "evidence": ["m520", "m521"],
                    "merged_from": [120, 121],
                    "merge_reasoning": "Merged Go skill facts with consistent proficiency. Added years of experience from complementary fact.",
                }
            ]
        },
        explanation="Consistent proficiency allows combining complementary details.",
    ),
    "belief_merge": FewShotExample(
        name="belief_merge",
        category=FactCategory.STABLE_SKILLS,
        fact_types=(FactType.BELIEVES,),
        description="Beliefs with matching stance merge and capture supporting rationale.",
        input_facts=[
            {
                "fact_id": 130,
                "object_label": "Remote work effective",
                "attributes": {"belief": "Remote work is productive", "sentiment": "positive"},
                "confidence": 0.58,
                "evidence": ["m530"],
            },
            {
                "fact_id": 131,
                "object_label": "Remote work works",
                "attributes": {
                    "belief": "Remote work is productive",
                    "reason": "fewer interruptions",
                },
                "confidence": 0.57,
                "evidence": ["m531"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "BELIEVES",
                    "subject_id": "user_42",
                    "object_label": "Remote work is productive",
                    "attributes": {
                        "belief": "Remote work is productive",
                        "sentiment": "positive",
                        "reason": "fewer interruptions",
                    },
                    "confidence": 0.63,
                    "evidence": ["m530", "m531"],
                    "merged_from": [130, 131],
                    "merge_reasoning": "Merged consistent belief statements about remote work productivity; added rationale as complementary detail.",
                }
            ]
        },
        explanation="Aligned stance with complementary rationale merges.",
    ),
    "preference_merge": FewShotExample(
        name="preference_merge",
        category=FactCategory.PREFERENCE_SENTIMENT,
        fact_types=(FactType.PREFERS,),
        description="Same preference reiterated with added justification merges.",
        input_facts=[
            {
                "fact_id": 140,
                "object_label": "Coffee from Blue Bottle",
                "attributes": {"target": "Blue Bottle Coffee", "reason": "tastes smooth"},
                "confidence": 0.6,
                "evidence": ["m600"],
            },
            {
                "fact_id": 141,
                "object_label": "Blue Bottle",
                "attributes": {"target": "Blue Bottle Coffee", "context": "morning routine"},
                "confidence": 0.58,
                "evidence": ["m601"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "PREFERS",
                    "subject_id": "user_42",
                    "object_label": "Blue Bottle Coffee",
                    "attributes": {
                        "target": "Blue Bottle Coffee",
                        "reason": "tastes smooth",
                        "context": "morning routine",
                    },
                    "confidence": 0.65,
                    "evidence": ["m600", "m601"],
                    "merged_from": [140, 141],
                    "merge_reasoning": "Merged matching preference for Blue Bottle Coffee; added context from second fact.",
                }
            ]
        },
        explanation="Matching preference with complementary context merges.",
    ),
    "preference_conflict_split": FewShotExample(
        name="preference_conflict_split",
        category=FactCategory.PREFERENCE_SENTIMENT,
        fact_types=(FactType.PREFERS, FactType.DISLIKES),
        description="Conflicting sentiments about the same target remain separate.",
        input_facts=[
            {
                "fact_id": 142,
                "object_label": "Cats",
                "attributes": {"target": "Cats", "sentiment": "loves"},
                "confidence": 0.6,
                "evidence": ["m602"],
            },
            {
                "fact_id": 143,
                "object_label": "Cats",
                "attributes": {"target": "Cats", "sentiment": "allergic"},
                "confidence": 0.61,
                "evidence": ["m603"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "PREFERS",
                    "subject_id": "user_42",
                    "object_label": "Cats",
                    "attributes": {"target": "Cats", "sentiment": "loves"},
                    "confidence": 0.6,
                    "evidence": ["m602"],
                    "merged_from": [142],
                    "merge_reasoning": "Positive sentiment about cats kept separate from conflicting avoidance statement.",
                },
                {
                    "type": "DISLIKES",
                    "subject_id": "user_42",
                    "object_label": "Cats",
                    "attributes": {"target": "Cats", "reason": "allergic"},
                    "confidence": 0.61,
                    "evidence": ["m603"],
                    "merged_from": [143],
                    "merge_reasoning": "Allergy-related avoidance recorded separately from preference statement.",
                },
            ]
        },
        explanation="Opposing sentiments must remain distinct.",
    ),
    "plan_with_timeframe": FewShotExample(
        name="plan_with_timeframe",
        category=FactCategory.PREFERENCE_SENTIMENT,
        fact_types=(FactType.PLANS_TO,),
        description="Plans with matching target and timeframe merge while preserving strongest timeframe detail.",
        input_facts=[
            {
                "fact_id": 144,
                "object_label": "Run marathon",
                "attributes": {
                    "plan": "Run a marathon",
                    "timeframe": "2025",
                    "confidence_level": "medium",
                },
                "confidence": 0.6,
                "evidence": ["m604"],
            },
            {
                "fact_id": 145,
                "object_label": "Marathon",
                "attributes": {"plan": "Run a marathon", "timeframe": "October 2025"},
                "confidence": 0.59,
                "evidence": ["m605"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "PLANS_TO",
                    "subject_id": "user_42",
                    "object_label": "Run a marathon",
                    "attributes": {
                        "plan": "Run a marathon",
                        "timeframe": "October 2025",
                        "confidence_level": "medium",
                    },
                    "confidence": 0.64,
                    "evidence": ["m604", "m605"],
                    "merged_from": [144, 145],
                    "merge_reasoning": "Merged matching marathon plan; retained specific October 2025 timeframe.",
                }
            ]
        },
        explanation="Consistent plans combine and prefer more specific timeframe.",
    ),
    "relationship_duplicate": FewShotExample(
        name="relationship_duplicate",
        category=FactCategory.RELATIONSHIPS_EVENTS,
        fact_types=(FactType.CLOSE_TO,),
        description="Repeated relationship mention merges when counterpart and rationale align.",
        input_facts=[
            {
                "fact_id": 150,
                "object_label": "Jordan",
                "attributes": {"closeness_basis": "collaborate weekly"},
                "confidence": 0.62,
                "evidence": ["m700"],
            },
            {
                "fact_id": 151,
                "object_label": "Jordan",
                "attributes": {"closeness_basis": "work together on team"},
                "confidence": 0.6,
                "evidence": ["m701"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "CLOSE_TO",
                    "subject_id": "user_42",
                    "object_label": "Jordan",
                    "attributes": {"closeness_basis": "collaborate weekly"},
                    "confidence": 0.66,
                    "evidence": ["m700", "m701"],
                    "merged_from": [150, 151],
                    "merge_reasoning": "Merged relationship facts about Jordan; basis aligns around close collaboration.",
                }
            ]
        },
        explanation="Same counterpart and rationale justify merge.",
    ),
    "relationship_conflict": FewShotExample(
        name="relationship_conflict",
        category=FactCategory.RELATIONSHIPS_EVENTS,
        fact_types=(FactType.RELATED_TO,),
        description="Conflicting relationship types stay separate.",
        input_facts=[
            {
                "fact_id": 152,
                "object_label": "Taylor",
                "attributes": {"relationship_type": "sister"},
                "confidence": 0.68,
                "evidence": ["m702"],
            },
            {
                "fact_id": 153,
                "object_label": "Taylor",
                "attributes": {"relationship_type": "cousin"},
                "confidence": 0.66,
                "evidence": ["m703"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "RELATED_TO",
                    "subject_id": "user_42",
                    "object_label": "Taylor",
                    "attributes": {"relationship_type": "sister"},
                    "confidence": 0.68,
                    "evidence": ["m702"],
                    "merged_from": [152],
                    "merge_reasoning": "Conflicting relationship types; preserved sister relationship separately.",
                },
                {
                    "type": "RELATED_TO",
                    "subject_id": "user_42",
                    "object_label": "Taylor",
                    "attributes": {"relationship_type": "cousin"},
                    "confidence": 0.66,
                    "evidence": ["m703"],
                    "merged_from": [153],
                    "merge_reasoning": "Conflicting relationship claims; documented cousin relationship separately.",
                },
            ]
        },
        explanation="Conflicting relationship types require separate entries.",
    ),
    "event_complementary": FewShotExample(
        name="event_complementary",
        category=FactCategory.RELATIONSHIPS_EVENTS,
        fact_types=(FactType.ATTENDED_EVENT,),
        description="Event attendance facts merge when they describe the same event with complementary details.",
        input_facts=[
            {
                "fact_id": 154,
                "object_label": "GraphCon 2024",
                "attributes": {"event_name": "GraphCon", "date": "2024-05-12", "role": "speaker"},
                "confidence": 0.7,
                "evidence": ["m704"],
            },
            {
                "fact_id": 155,
                "object_label": "GraphCon",
                "attributes": {
                    "event_name": "GraphCon",
                    "date": "2024-05-12",
                    "format": "in-person",
                },
                "confidence": 0.68,
                "evidence": ["m705"],
            },
        ],
        expected_output={
            "canonical_facts": [
                {
                    "type": "ATTENDED_EVENT",
                    "subject_id": "user_42",
                    "object_label": "GraphCon",
                    "attributes": {
                        "event_name": "GraphCon",
                        "date": "2024-05-12",
                        "role": "speaker",
                        "format": "in-person",
                    },
                    "confidence": 0.75,
                    "evidence": ["m704", "m705"],
                    "merged_from": [154, 155],
                    "merge_reasoning": "Merged event attendance facts for GraphCon 2024. Combined role and format details with matching date.",
                }
            ]
        },
        explanation="Same event instance with complementary details merges.",
    ),
}


def select_template(fact_type: FactType) -> CategoryTemplate:
    category = FACT_CATEGORY_BY_TYPE.get(fact_type)
    if category is None:
        raise ValueError(f"No category template registered for fact type {fact_type}")
    return CATEGORY_TEMPLATES[category]


def format_example(example: FewShotExample) -> str:
    input_json = json.dumps(example.input_facts, indent=2, sort_keys=True)
    output_json = json.dumps(example.expected_output, indent=2, sort_keys=True)
    return "\n".join(
        [
            f"Example: {example.name}",
            f"Scenario: {example.description}",
            "Input Facts:",
            input_json,
            "Expected Output:",
            output_json,
            f"Explanation: {example.explanation}",
        ]
    )


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


def build_messages(
    partition: Partition, candidate_facts: Iterable[FactRecord]
) -> list[dict[str, str]]:
    template = select_template(partition.fact_type)
    examples = template.examples_for_fact_type(partition.fact_type)
    examples_text = (
        "\n\n".join(format_example(example) for example in examples)
        if examples
        else "(No examples available)"
    )

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

    category_sections = [
        template.overview_text(),
        template.guidance_for_fact_type(partition.fact_type),
        template.normalization_text(),
        template.confidence_text(),
        template.edge_case_text(),
    ]
    category_text = "\n\n".join(section for section in category_sections if section)

    user_sections = [
        "DEDUPLICATION TASK",
        f"Fact Type: {partition.fact_type.value}",
        f"Subject: {subject_display}",
        "",
        category_text,
        "",
        _format_attribute_priorities(partition.fact_type),
        "",
        BASE_CONFIDENCE_RULES,
        "",
        BASE_EDGE_CASE_RULES,
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
