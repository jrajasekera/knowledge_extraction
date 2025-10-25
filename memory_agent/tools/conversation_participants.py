"""Implementation for get_conversation_participants."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, Field

from ..conversation import MENTION_PATTERN
from ..models import MessageModel
from .base import ToolBase
from .utils import run_read_query


REFERENCE_KEYWORDS = {
    "brother": ["brother", "sibling"],
    "sister": ["sister", "sibling"],
    "mom": ["mother", "parent"],
    "mother": ["mother", "parent"],
    "dad": ["father", "parent"],
    "father": ["father", "parent"],
    "parent": ["parent"],
    "roommate": ["roommate", "lives_with"],
    "partner": ["partner", "significant_other"],
    "girlfriend": ["girlfriend", "partner"],
    "boyfriend": ["boyfriend", "partner"],
    "spouse": ["spouse", "partner"],
}


ALIAS_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_'`-]*(?:\s+[A-Za-z][A-Za-z0-9_'`-]*)*")


@dataclass(slots=True)
class AliasCandidate:
    person_id: str
    display_name: str
    source: str


class ConversationParticipantsInput(BaseModel):
    """Inputs for get_conversation_participants."""

    messages: list[MessageModel]


class ExplicitMentionModel(BaseModel):
    """Explicit mention output."""

    name: str
    person_id: str | None = None
    mentioned_in_message: int


class ImplicitReferenceGuess(BaseModel):
    """Possible match for an implicit reference."""

    person_id: str
    name: str
    confidence: float
    reason: str | None = None


class ImplicitReferenceModel(BaseModel):
    """Implicit reference entry."""

    reference: str
    possible_matches: list[ImplicitReferenceGuess] = Field(default_factory=list)


class ConversationParticipantsOutput(BaseModel):
    """Outputs for get_conversation_participants."""

    explicit_mentions: list[ExplicitMentionModel] = Field(default_factory=list)
    implicit_references: list[ImplicitReferenceModel] = Field(default_factory=list)


class GetConversationParticipantsTool(
    ToolBase[ConversationParticipantsInput, ConversationParticipantsOutput],
):
    """Identify people mentioned in conversation."""

    input_model = ConversationParticipantsInput
    output_model = ConversationParticipantsOutput

    def run(self, input_data: ConversationParticipantsInput) -> ConversationParticipantsOutput:
        people = self._fetch_people()
        alias_index = self._build_alias_index(people)
        person_lookup = {row["id"]: row for row in people}

        explicit_mentions: list[ExplicitMentionModel] = []
        explicit_seen: set[tuple[str, int]] = set()
        implicit_map: dict[str, ImplicitReferenceModel] = {}

        for idx, message in enumerate(input_data.messages):
            author_id = message.author_id
            author_record = person_lookup.get(author_id)
            if author_record and (author_id, idx) not in explicit_seen:
                explicit_seen.add((author_id, idx))
                explicit_mentions.append(
                    ExplicitMentionModel(
                        name=author_record.get("display_name") or author_record.get("name") or message.author_name,
                        person_id=author_id,
                        mentioned_in_message=idx,
                    )
                )

            # Handle Discord mention syntax
            for discord_id in MENTION_PATTERN.findall(message.content):
                display_name = None
                record = person_lookup.get(discord_id)
                if record:
                    display_name = record.get("display_name") or record.get("name")
                if (discord_id, idx) in explicit_seen:
                    continue
                explicit_seen.add((discord_id, idx))
                explicit_mentions.append(
                    ExplicitMentionModel(
                        name=display_name or message.author_name,
                        person_id=discord_id,
                        mentioned_in_message=idx,
                    )
                )

            lowered = message.content.lower()
            alias_matches = self._resolve_alias_mentions(lowered, alias_index)
            for alias, candidates in alias_matches.items():
                if len(candidates) == 1:
                    candidate = candidates[0]
                    key = (candidate.person_id, idx)
                    if key in explicit_seen:
                        continue
                    explicit_seen.add(key)
                    explicit_mentions.append(
                        ExplicitMentionModel(
                            name=candidate.display_name,
                            person_id=candidate.person_id,
                            mentioned_in_message=idx,
                        )
                    )
                else:
                    guesses = [
                        ImplicitReferenceGuess(
                            person_id=candidate.person_id,
                            name=candidate.display_name,
                            confidence=0.6,
                            reason=f"Matched alias '{alias}'",
                        )
                        for candidate in candidates
                    ]
                    self._append_implicit(implicit_map, alias, guesses)

            # Pronoun-style references (e.g., "my brother")
            if author_record:
                pronoun_matches = self._resolve_pronoun_references(author_record["id"], lowered)
                for reference_text, guesses in pronoun_matches:
                    self._append_implicit(implicit_map, reference_text, guesses)

        implicit_references = list(implicit_map.values())
        return ConversationParticipantsOutput(explicit_mentions=explicit_mentions, implicit_references=implicit_references)

    def _fetch_people(self) -> list[dict[str, Any]]:
        query = """
        MATCH (p:Person)
        RETURN p
        """
        rows = run_read_query(self.context, query)
        people: list[dict[str, Any]] = []
        for row in rows:
            node = row.get("p")
            if node is None:
                continue
            properties = dict(node)
            record = {
                "id": properties.get("id"),
                "name": properties.get("name"),
                "nickname": properties.get("nickname"),
                "real_name": properties.get("realName"),
                "discriminator": properties.get("discriminator"),
                "aliases": properties.get("aliases") or [],
            }
            record["display_name"] = (
                record.get("nickname")
                or record.get("name")
                or record.get("real_name")
                or record.get("id")
            )
            people.append(record)
        return people

    def _build_alias_index(self, rows: Iterable[dict[str, Any]]) -> dict[str, list[AliasCandidate]]:
        index: dict[str, list[AliasCandidate]] = defaultdict(list)
        for row in rows:
            person_id = row.get("id")
            if not person_id:
                continue
            display_name = row.get("display_name") or person_id
            aliases = set()
            for field in ("name", "nickname", "real_name"):
                value = row.get(field)
                if value:
                    aliases.add(value)
                    aliases.add(value.split("#")[0])
            discriminator = row.get("discriminator")
            if discriminator and row.get("name"):
                aliases.add(f"{row['name']}#{discriminator}")
            for alias_entry in row.get("aliases", []) or []:
                if isinstance(alias_entry, str):
                    aliases.add(alias_entry)

            for alias in aliases:
                normalized = alias.strip().lower()
                if not normalized or len(normalized) < 2:
                    continue
                index[normalized].append(AliasCandidate(person_id=person_id, display_name=display_name, source=alias))

            # Include individual words for multi-word names (length >= 3 to avoid stop words)
            for alias in list(aliases):
                for token in alias.split():
                    token_clean = token.strip().lower()
                    if len(token_clean) >= 3:
                        index[token_clean].append(
                            AliasCandidate(person_id=person_id, display_name=display_name, source=alias)
                        )

        return index

    def _resolve_alias_mentions(
        self,
        message_lower: str,
        alias_index: dict[str, list[AliasCandidate]],
    ) -> dict[str, list[AliasCandidate]]:
        matches: dict[str, list[AliasCandidate]] = {}
        for token in set(match.group(0).strip().lower() for match in ALIAS_TOKEN_PATTERN.finditer(message_lower)):
            if len(token) < 3:
                continue
            candidates = alias_index.get(token)
            if candidates:
                matches[token] = candidates
        return matches

    def _resolve_pronoun_references(
        self,
        author_id: str,
        message_lower: str,
    ) -> list[tuple[str, list[ImplicitReferenceGuess]]]:
        references: list[tuple[str, list[ImplicitReferenceGuess]]] = []
        for keyword in REFERENCE_KEYWORDS:
            pattern = rf"my\s+{re.escape(keyword)}"
            if re.search(pattern, message_lower):
                guesses = self._lookup_relationship_matches(author_id, keyword)
                references.append((f"my {keyword}", guesses))
        return references

    def _lookup_relationship_matches(self, author_id: str, keyword: str) -> list[ImplicitReferenceGuess]:
        relationship_types = [term.lower() for term in REFERENCE_KEYWORDS.get(keyword, [])]
        if not relationship_types:
            return []
        query = """
        MATCH (author:Person {id:$author_id})-[r:RELATED_TO]->(other)
        WHERE toLower(coalesce(r.relationshipType, '')) IN $relationship_types
        RETURN other.id AS person_id,
               coalesce(other.name, other.nickname, other.realName, other.id) AS name,
               coalesce(r.relationshipType, '') AS relationship_type,
               coalesce(r.relationshipBasis, '') AS basis,
               coalesce(r.confidence, 0.0) AS confidence
        LIMIT 10
        """
        rows = run_read_query(
            self.context,
            query,
            {"author_id": author_id, "relationship_types": relationship_types},
        )
        guesses: list[ImplicitReferenceGuess] = []
        for row in rows:
            person_id = row.get("person_id")
            name = row.get("name")
            if not person_id or not name:
                continue
            reason = row.get("relationship_type") or "related"
            basis = row.get("basis")
            if basis:
                reason = f"{reason} ({basis})"
            guesses.append(
                ImplicitReferenceGuess(
                    person_id=person_id,
                    name=name,
                    confidence=row.get("confidence"),
                    reason=reason,
                )
            )
        return guesses

    def _append_implicit(
        self,
        implicit_map: dict[str, ImplicitReferenceModel],
        reference: str,
        guesses: list[ImplicitReferenceGuess],
    ) -> None:
        if reference not in implicit_map:
            implicit_map[reference] = ImplicitReferenceModel(reference=reference, possible_matches=[])
        existing = implicit_map[reference].possible_matches
        seen_ids = {guess.person_id for guess in existing}
        for guess in guesses:
            if guess.person_id in seen_ids:
                continue
            existing.append(guess)
