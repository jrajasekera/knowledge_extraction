from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator, model_validator


def _parse_datetime(value: Any) -> datetime | None:
    """Parse ISO timestamps while tolerating timezone-naive values."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _format_datetime(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


class MessageType(str, Enum):
    DEFAULT = "Default"
    REPLY = "Reply"


class ChannelType(str, Enum):
    GUILD_TEXT_CHAT = "GuildTextChat"
    DM = "DM"
    GROUP_DM = "GroupDM"
    VOICE = "GuildVoice"
    STAGE = "GuildStageVoice"


class Role(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    color: str | None = None
    position: int | None = None

    @field_validator("color", mode="before")
    @classmethod
    def _normalize_color(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    @computed_field(return_type=int | None)
    def color_rgb(self) -> int | None:  # noqa: D401
        """Color expressed as an integer (0xRRGGBB)."""
        if not self.color:
            return None
        return int(self.color.lstrip("#"), 16)


class Member(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    discriminator: str
    nickname: str | None = None
    color: str | None = None
    is_bot: bool = Field(alias="isBot")
    roles: list[Role] = Field(default_factory=list)
    avatar_url: str | None = Field(default=None, alias="avatarUrl")

    @field_validator("color", mode="before")
    @classmethod
    def _normalize_color(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    @computed_field(return_type=str)
    def display_name(self) -> str:  # noqa: D401
        """Preferred display name (nickname fallback to username)."""
        return self.nickname or self.name

    @computed_field(return_type=int | None)
    def accent_color(self) -> int | None:  # noqa: D401
        """Accent color as integer if provided."""
        if not self.color:
            return None
        return int(self.color.lstrip("#"), 16)

    @computed_field(return_type=tuple[str, ...])
    def role_names(self) -> tuple[str, ...]:  # noqa: D401
        """Role names in server order (highest first)."""
        ordered = sorted(
            (role for role in self.roles),
            key=lambda role: (-1 if role.position is None else -role.position, role.name.lower()),
        )
        return tuple(role.name for role in ordered)


class Attachment(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    url: str
    file_name: str = Field(alias="fileName")
    file_size_bytes: int = Field(alias="fileSizeBytes")

    @computed_field(return_type=float)
    def file_size_kib(self) -> float:  # noqa: D401
        """Attachment size in kibibytes."""
        return self.file_size_bytes / 1024


class EmbedAuthor(BaseModel):
    name: str
    url: str | None = None


class EmbedMedia(BaseModel):
    url: str
    width: int | None = None
    height: int | None = None


class InlineEmoji(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = ""
    name: str
    code: str
    is_animated: bool = Field(alias="isAnimated")
    image_url: str | None = Field(default=None, alias="imageUrl")


class Embed(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    title: str | None = None
    url: str | None = None
    timestamp: datetime | None = None
    description: str | None = None
    color: str | None = None
    author: EmbedAuthor | None = None
    thumbnail: EmbedMedia | None = None
    video: EmbedMedia | None = None
    images: list[EmbedMedia] = Field(default_factory=list)
    fields: list[dict[str, Any]] = Field(default_factory=list)
    inline_emojis: list[InlineEmoji] = Field(default_factory=list, alias="inlineEmojis")

    @field_validator("timestamp", mode="before")
    @classmethod
    def _parse_timestamp(cls, value: Any) -> datetime | None:
        return _parse_datetime(value)

    @field_validator("color", mode="before")
    @classmethod
    def _normalize_color(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.upper()

    @computed_field(return_type=str | None)
    def timestamp_text(self) -> str | None:  # noqa: D401
        """Timestamp rendered as ISO-8601 string."""
        return _format_datetime(self.timestamp)

    @computed_field(return_type=int | None)
    def color_rgb(self) -> int | None:  # noqa: D401
        """Color expressed as an integer (0xRRGGBB)."""
        if not self.color:
            return None
        return int(self.color.lstrip("#"), 16)


class Emoji(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str | None = None
    name: str
    code: str
    is_animated: bool = Field(alias="isAnimated")
    image_url: str | None = Field(default=None, alias="imageUrl")


class Reaction(BaseModel):
    emoji: Emoji
    count: int
    users: list[Member] = Field(default_factory=list)

    @computed_field(return_type=tuple[str, ...])
    def user_ids(self) -> tuple[str, ...]:  # noqa: D401
        """IDs of the users who reacted."""
        return tuple(user.id for user in self.users)


class MessageReference(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message_id: str = Field(alias="messageId")
    channel_id: str = Field(alias="channelId")
    guild_id: str | None = Field(default=None, alias="guildId")


class Message(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: MessageType = MessageType.DEFAULT
    timestamp: datetime = Field(alias="timestamp")
    timestamp_edited: datetime | None = Field(default=None, alias="timestampEdited")
    call_ended_at: datetime | None = Field(default=None, alias="callEndedTimestamp")
    is_pinned: bool = Field(alias="isPinned")
    content: str
    author: Member
    attachments: list[Attachment] = Field(default_factory=list)
    embeds: list[Embed] = Field(default_factory=list)
    stickers: list[dict[str, Any]] = Field(default_factory=list)
    reactions: list[Reaction] = Field(default_factory=list)
    mentions: list[Member] = Field(default_factory=list)
    inline_emojis: list[InlineEmoji] = Field(default_factory=list, alias="inlineEmojis")
    reference: MessageReference | None = None

    @field_validator("timestamp", "timestamp_edited", "call_ended_at", mode="before")
    @classmethod
    def _parse_timestamps(cls, value: Any) -> datetime | None:
        return _parse_datetime(value)

    @computed_field(return_type=str)
    def content_stripped(self) -> str:  # noqa: D401
        """Message content with leading/trailing whitespace removed."""
        return self.content.strip()

    @computed_field(return_type=bool)
    def has_attachments(self) -> bool:  # noqa: D401
        """Whether the message carries at least one attachment."""
        return bool(self.attachments)

    @computed_field(return_type=bool)
    def is_reply(self) -> bool:  # noqa: D401
        """True when the message references another message."""
        return self.type == MessageType.REPLY or self.reference is not None

    @computed_field(return_type=str | None)
    def timestamp_text(self) -> str | None:  # noqa: D401
        """Timestamp rendered as ISO-8601 string."""
        return _format_datetime(self.timestamp)

    @computed_field(return_type=str | None)
    def edited_timestamp_text(self) -> str | None:  # noqa: D401
        """Edited timestamp rendered as ISO-8601 string."""
        return _format_datetime(self.timestamp_edited)


class DateRange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    after: datetime | None = None
    before: datetime | None = None

    @field_validator("after", "before", mode="before")
    @classmethod
    def _parse_timestamps(cls, value: Any) -> datetime | None:
        return _parse_datetime(value)


class Guild(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    name: str
    icon_url: str | None = Field(default=None, alias="iconUrl")


class Channel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    type: ChannelType
    category_id: str | None = Field(default=None, alias="categoryId")
    category: str | None = None
    name: str
    topic: str | None = None


class ConversationExport(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    guild: Guild
    channel: Channel
    date_range: DateRange = Field(alias="dateRange")
    exported_at: datetime = Field(alias="exportedAt")
    messages: list[Message]
    message_count: int = Field(alias="messageCount")

    @field_validator("exported_at", mode="before")
    @classmethod
    def _parse_exported_at(cls, value: Any) -> datetime | None:
        parsed = _parse_datetime(value)
        if parsed is None:
            msg = "exported_at is required"
            raise ValueError(msg)
        return parsed

    @computed_field(return_type=str)
    def exported_at_text(self) -> str:  # noqa: D401
        """Export timestamp rendered as ISO-8601 string."""
        return _format_datetime(self.exported_at) or ""

    @computed_field(return_type=int)
    def message_count_actual(self) -> int:  # noqa: D401
        """Actual number of messages loaded."""
        return len(self.messages)

    def iter_messages(self) -> Iterable[Message]:
        yield from self.messages

    def iter_authors(self) -> Iterable[Member]:
        seen: set[str] = set()
        for message in self.messages:
            if message.author.id not in seen:
                seen.add(message.author.id)
                yield message.author

    def messages_by_author(self, author_id: str) -> tuple[Message, ...]:
        return tuple(m for m in self.messages if m.author.id == author_id)

    def message_ids(self) -> tuple[str, ...]:
        return tuple(message.id for message in self.messages)

    @model_validator(mode="after")
    def _reconcile_message_count(self) -> "ConversationExport":
        if self.message_count != len(self.messages):
            # Keep the reported count for reference but do not treat as fatal.
            pass
        return self


def load_export(source: str | Path | bytes | bytearray | Sequence[bytes]) -> ConversationExport:
    """Load a Discord channel export into the domain model."""
    if isinstance(source, (str, Path)):
        raw = Path(source).read_text(encoding="utf-8")
    elif isinstance(source, (bytes, bytearray)):
        raw = source.decode("utf-8")
    else:
        raw = b"".join(source).decode("utf-8")
    import json

    payload = json.loads(raw)
    return ConversationExport.model_validate(payload)
