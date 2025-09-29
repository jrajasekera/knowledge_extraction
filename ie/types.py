from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping


class FactType(str, Enum):
    WORKS_AT = "WORKS_AT"
    LIVES_IN = "LIVES_IN"
    TALKS_ABOUT = "TALKS_ABOUT"
    CLOSE_TO = "CLOSE_TO"


@dataclass(frozen=True, slots=True)
class FactAttribute:
    name: str
    description: str
    required: bool = False


@dataclass(frozen=True, slots=True)
class FactDefinition:
    type: FactType
    subject_description: str
    object_type: str | None
    object_description: str | None
    attributes: tuple[FactAttribute, ...]
    rationale: str

    def attribute_names(self) -> tuple[str, ...]:
        return tuple(attr.name for attr in self.attributes)


def build_fact_definition_index(definitions: Iterable[FactDefinition]) -> Mapping[FactType, FactDefinition]:
    definition_list = tuple(definitions)
    index = {definition.type: definition for definition in definition_list}
    if len(index) != len(definition_list):
        raise ValueError("Duplicate fact definitions detected")
    return index
