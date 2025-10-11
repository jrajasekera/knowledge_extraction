"""Implementation for get_conversation_participants."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..conversation import MENTION_PATTERN
from ..models import MessageModel
from .base import ToolBase

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
        explicit_mentions: list[ExplicitMentionModel] = []
        implicit: list[ImplicitReferenceModel] = []
        for idx, message in enumerate(input_data.messages):
            for match in MENTION_PATTERN.findall(message.content):
                explicit_mentions.append(
                    ExplicitMentionModel(
                        name=message.author_name,
                        person_id=match,
                        mentioned_in_message=idx,
                    )
                )
            lowered = message.content.lower()
            if "my brother" in lowered:
                implicit.append(
                    ImplicitReferenceModel(
                        reference="my brother",
                        possible_matches=[],
                    )
                )
        return ConversationParticipantsOutput(explicit_mentions=explicit_mentions, implicit_references=implicit)
