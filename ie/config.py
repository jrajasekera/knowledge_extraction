from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .types import FactAttribute, FactDefinition, FactType, build_fact_definition_index


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
)


FACT_DEFINITION_INDEX: Mapping[FactType, FactDefinition] = build_fact_definition_index(FACT_DEFINITIONS)

DEFAULT_FACT_TYPES: tuple[FactType, ...] = tuple(definition.type for definition in FACT_DEFINITIONS)


@dataclass(slots=True)
class IEConfig:
    fact_types: Sequence[FactType] = DEFAULT_FACT_TYPES
    window_size: int = 4
    confidence_threshold: float = 0.5
    max_windows: int | None = None

    def validate(self) -> "IEConfig":
        if self.window_size < 1:
            raise ValueError("window_size must be >= 1")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        invalid = [fact for fact in self.fact_types if fact not in FACT_DEFINITION_INDEX]
        if invalid:
            raise ValueError(f"Unsupported fact types provided: {invalid}")
        return self
