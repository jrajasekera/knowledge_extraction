from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .types import FactAttribute, FactDefinition, FactType, build_fact_definition_index


IE_CACHE_VERSION = "v1"

FACT_DEFINITIONS: tuple[FactDefinition, ...] = (
    FactDefinition(
        type=FactType.WORKS_AT,
        subject_description="Discord member who holds or held the position",
        object_type="Organization",
        object_description="Organization or company referenced in the conversation",
        attributes=(
            FactAttribute("organization", "Normalized organization name", required=True),
            FactAttribute("role", "Role or title held by the subject"),
            FactAttribute("location", "Location associated with the role"),
            FactAttribute("start_date", "ISO date when the role started"),
            FactAttribute("end_date", "ISO date when the role ended"),
        ),
        rationale="Work relationships unlock org-level graph analytics and profile summaries.",
    ),
    FactDefinition(
        type=FactType.LIVES_IN,
        subject_description="Discord member the statement applies to",
        object_type="Place",
        object_description="Place or region where the person resides",
        attributes=(
            FactAttribute("location", "Normalized place name", required=True),
            FactAttribute("since", "ISO date or description of how long they have lived there"),
        ),
        rationale="Residency helps build geo-centric slices of the social graph.",
    ),
    FactDefinition(
        type=FactType.TALKS_ABOUT,
        subject_description="Discord member discussing a topic",
        object_type="Topic",
        object_description="Topic, project, or subject matter mentioned",
        attributes=(
            FactAttribute("topic", "Normalized topic or subject", required=True),
            FactAttribute("sentiment", "Optional sentiment or stance (positive/negative/neutral)"),
        ),
        rationale="Topic affinity powers interest clustering and recommendation use cases.",
    ),
    FactDefinition(
        type=FactType.CLOSE_TO,
        subject_description="Discord member with ties to another member",
        object_type="Person",
        object_description="Second member who is closely connected",
        attributes=(
            FactAttribute("closeness_basis", "Evidence summary for why they are close"),
        ),
        rationale="Augments interaction-based weights with explicit relationship assertions.",
    ),
    FactDefinition(
        type=FactType.STUDIED_AT,
        subject_description="Discord member with an educational history entry",
        object_type="Institution",
        object_description="School, university, bootcamp, or educational program",
        attributes=(
            FactAttribute("institution", "Normalized institution name", required=True),
            FactAttribute("degree_type", "Degree or certification type (BSc, MBA, Bootcamp)"),
            FactAttribute("field_of_study", "Primary field or major"),
            FactAttribute("graduation_year", "Four-digit graduation/completion year"),
            FactAttribute("status", "Enrollment status: current/past/deferred"),
        ),
        rationale="Education history helps expertise mapping and alumni clustering.",
    ),
    FactDefinition(
        type=FactType.HAS_SKILL,
        subject_description="Discord member who possesses a skill",
        object_type="Skill",
        object_description="Skill, technology, or domain the member is capable with",
        attributes=(
            FactAttribute("skill", "Canonical skill name", required=True),
            FactAttribute("proficiency_level", "Self-reported level (beginner/intermediate/expert)"),
            FactAttribute("years_experience", "Approximate years practicing the skill"),
            FactAttribute("learning_status", "learning/expert/rusty"),
        ),
        rationale="Captures actionable capabilities for matching collaborators.",
    ),
    FactDefinition(
        type=FactType.WORKING_ON,
        subject_description="Discord member actively working on something",
        object_type="Project",
        object_description="Named project, initiative, or product",
        attributes=(
            FactAttribute("project", "Project name or description", required=True),
            FactAttribute("role", "Role or contribution focus"),
            FactAttribute("start_date", "ISO8601 start date or month"),
            FactAttribute("project_type", "work/personal/open-source"),
            FactAttribute("collaboration_mode", "solo/pair/team/community"),
        ),
        rationale="Highlights active initiatives that open collaboration opportunities.",
    ),
    FactDefinition(
        type=FactType.RELATED_TO,
        subject_description="Discord member with a personal relationship",
        object_type="Person",
        object_description="The related individual (Discord member or external person)",
        attributes=(
            FactAttribute("relationship_type", "sibling/parent/spouse/partner/etc", required=True),
            FactAttribute("relationship_basis", "Brief description or evidence of the relationship"),
        ),
        rationale="Surface family and personal ties beyond interaction weights.",
    ),
    FactDefinition(
        type=FactType.ATTENDED_EVENT,
        subject_description="Discord member who participated in an event",
        object_type="Event",
        object_description="Conference, meetup, or gathering",
        attributes=(
            FactAttribute("event_name", "Event title", required=True),
            FactAttribute("event_type", "conference/meetup/webinar/etc"),
            FactAttribute("date", "ISO8601 date or range"),
            FactAttribute("role", "attendee/speaker/organizer"),
            FactAttribute("format", "in-person/virtual/hybrid"),
            FactAttribute("location", "City or venue if applicable"),
        ),
        rationale="Connects members through shared real-world or virtual events.",
    ),
    FactDefinition(
        type=FactType.RECOMMENDS,
        subject_description="Discord member endorsing something",
        object_type="RecommendationTarget",
        object_description="Product, tool, book, service, or place being recommended",
        attributes=(
            FactAttribute("target", "Normalized item being recommended", required=True),
            FactAttribute("recommendation_strength", "strong/medium/light"),
            FactAttribute("context", "Scenario or use-case for the recommendation"),
            FactAttribute("reason", "Short explanation for the endorsement"),
        ),
        rationale="Captures explicit endorsements useful for discovery and referrals.",
    ),
    FactDefinition(
        type=FactType.AVOIDS,
        subject_description="Discord member avoiding something",
        object_type="AvoidanceTarget",
        object_description="Tool, company, or practice the member avoids",
        attributes=(
            FactAttribute("target", "Normalized avoidance target", required=True),
            FactAttribute("reason", "Summary of why it is avoided"),
            FactAttribute("severity", "Strength of avoidance (hard/soft/conditional)"),
            FactAttribute("timeframe", "When the avoidance applies or started"),
        ),
        rationale="Negative signals prevent poor suggestions and clarify preferences.",
    ),
    FactDefinition(
        type=FactType.PLANS_TO,
        subject_description="Discord member expressing a future plan or goal",
        object_type="Plan",
        object_description="Goal, move, event, or action they intend to pursue",
        attributes=(
            FactAttribute("plan", "Short description of the plan", required=True),
            FactAttribute("goal_type", "move/job_change/learn/attend/etc"),
            FactAttribute("timeframe", "Expected timing or deadline"),
            FactAttribute("confidence_level", "Self-reported certainty (high/medium/low)"),
        ),
        rationale="Forward-looking statements help with coordination and follow-up.",
    ),
    FactDefinition(
        type=FactType.PREVIOUSLY,
        subject_description="Discord member describing a past experience no longer current",
        object_type="HistoricalFact",
        object_description="Organization, location, or role previously associated with the member",
        attributes=(
            FactAttribute("fact_type", "Referenced fact type (worked_at/lived_in/etc)", required=True),
            FactAttribute("object_label", "Human-readable target of the past fact", required=True),
            FactAttribute("start_date", "When it began"),
            FactAttribute("end_date", "When it ended"),
            FactAttribute("transition_reason", "Why it ended or changed"),
        ),
        rationale="Adds historic context that shapes profiles and trajectories.",
    ),
    FactDefinition(
        type=FactType.PREFERS,
        subject_description="Discord member stating a preference",
        object_type="Preference",
        object_description="Option or choice the member prefers",
        attributes=(
            FactAttribute("preference_target", "The thing they prefer", required=True),
            FactAttribute("preference_category", "food/workflow/style/etc"),
            FactAttribute("preference_strength", "strong/moderate/light"),
            FactAttribute("alternatives_considered", "What it is preferred over"),
            FactAttribute("reason", "Why they prefer it"),
        ),
        rationale="Captures decision-making patterns and taste profiles.",
    ),
    FactDefinition(
        type=FactType.BELIEVES,
        subject_description="Discord member expressing a stance or opinion",
        object_type="Topic",
        object_description="Topic, issue, or concept the stance applies to",
        attributes=(
            FactAttribute("stance", "support/oppose/nuanced/uncertain", required=True),
            FactAttribute("conviction_strength", "Confidence in the belief"),
            FactAttribute("reasoning", "Supporting reasoning or evidence"),
        ),
        rationale="Clarifies values and viewpoints for collaboration compatibility.",
    ),
    FactDefinition(
        type=FactType.DISLIKES,
        subject_description="Discord member expressing dislike",
        object_type="DislikeTarget",
        object_description="Thing, practice, or genre they dislike",
        attributes=(
            FactAttribute("target", "Item being disliked", required=True),
            FactAttribute("dislike_intensity", "low/medium/high"),
            FactAttribute("reason", "Why they dislike it"),
            FactAttribute("still_engages", "yes/no for whether they still engage"),
        ),
        rationale="Knowing aversions helps avoid friction and tailor recommendations.",
    ),
    FactDefinition(
        type=FactType.ENJOYS,
        subject_description="Discord member describing enjoyment of something",
        object_type="Activity",
        object_description="Activity, hobby, or experience providing joy",
        attributes=(
            FactAttribute("activity", "Activity enjoyed", required=True),
            FactAttribute("enjoyment_level", "low/moderate/high"),
            FactAttribute("frequency", "How often they do it"),
            FactAttribute("social_mode", "solo/with_friends/community"),
        ),
        rationale="Highlights leisure interests for bonding and recommendations.",
    ),
    FactDefinition(
        type=FactType.EXPERIENCED,
        subject_description="Discord member recounting a significant life event",
        object_type="LifeEvent",
        object_description="Event type such as birth, loss, illness, achievement, travel",
        attributes=(
            FactAttribute("event_type", "Category of life event", required=True),
            FactAttribute("event_date", "Date or approximate timeframe"),
            FactAttribute("impact_level", "low/moderate/high"),
            FactAttribute("current_stage", "ongoing/resolved/recent"),
        ),
        rationale="Contextualizes availability, support needs, and milestones.",
    ),
    FactDefinition(
        type=FactType.CARES_ABOUT,
        subject_description="Discord member stating a priority or value",
        object_type="CauseOrValue",
        object_description="Cause, value, or principle they prioritize",
        attributes=(
            FactAttribute("importance_level", "low/moderate/high", required=True),
            FactAttribute("how_it_manifests", "How this value shows up in actions"),
            FactAttribute("related_actions", "Concrete activities they take"),
        ),
        rationale="Connects members through shared motivations and causes.",
    ),
    FactDefinition(
        type=FactType.REMEMBERS,
        subject_description="Discord member recalling a notable memory",
        object_type="Memory",
        object_description="Event, experience, person, or place remembered",
        attributes=(
            FactAttribute("memory_type", "nostalgic/funny/significant/etc", required=True),
            FactAttribute("emotional_valence", "positive/negative/mixed"),
            FactAttribute("approximate_date", "When it occurred"),
            FactAttribute("significance", "Why it matters"),
        ),
        rationale="Shared memories deepen bonds and serve as narrative anchors.",
    ),
    FactDefinition(
        type=FactType.CURIOUS_ABOUT,
        subject_description="Discord member expressing curiosity",
        object_type="Topic",
        object_description="Topic, field, or question they want to explore",
        attributes=(
            FactAttribute("topic", "Subject they are curious about", required=True),
            FactAttribute("curiosity_level", "low/moderate/high", required=True),
            FactAttribute("what_sparked_it", "Trigger for curiosity"),
            FactAttribute("exploration_status", "researching/asking/experimenting"),
        ),
        rationale="Signals learning interests for matching with mentors or resources.",
    ),
    FactDefinition(
        type=FactType.WITNESSED,
        subject_description="Discord member reporting firsthand experience",
        object_type="WitnessedEvent",
        object_description="Event, phenomenon, or situation they saw or experienced",
        attributes=(
            FactAttribute("context", "Brief description of circumstances", required=True),
            FactAttribute("date", "When it was witnessed"),
            FactAttribute("their_role", "participant/observer/first_responder"),
            FactAttribute("impact", "Perceived impact or consequence"),
        ),
        rationale="Validates real-world perspective and expertise through firsthand accounts.",
    ),
)


FACT_DEFINITION_INDEX: Mapping[FactType, FactDefinition] = build_fact_definition_index(FACT_DEFINITIONS)

DEFAULT_FACT_TYPES: tuple[FactType, ...] = tuple(definition.type for definition in FACT_DEFINITIONS)


@dataclass(slots=True)
class IEConfig:
    fact_types: Sequence[FactType] = DEFAULT_FACT_TYPES
    window_size: int = 4
    confidence_threshold: float = 0.5
    max_windows: int | None = None
    max_concurrent_requests: int = 1
    prompt_version: Literal["legacy", "enhanced"] = "enhanced"

    def validate(self) -> "IEConfig":
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be >= 1")
        invalid = [fact for fact in self.fact_types if fact not in FACT_DEFINITION_INDEX]
        if invalid:
            raise ValueError(f"Unsupported fact types provided: {invalid}")
        if self.prompt_version not in {"legacy", "enhanced"}:
            raise ValueError("prompt_version must be 'legacy' or 'enhanced'")
        return self


def compute_config_hash(config: IEConfig) -> str:
    """Generate a stable fingerprint for cacheable IE runs."""
    fact_type_values = sorted({fact_type.value for fact_type in config.fact_types})
    payload = {
        "version": IE_CACHE_VERSION,
        "window_size": config.window_size,
        "prompt_version": config.prompt_version,
        "confidence_threshold": round(config.confidence_threshold, 4),
        "fact_types": fact_type_values,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return digest[:16]
