from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path

from data_structures.ingestion import ConversationExport, load_export


def _format_dt(value: datetime | None) -> str:
    return value.isoformat() if value else "n/a"


def _print_stats(bundle: ConversationExport) -> None:
    messages = bundle.messages
    message_count = len(messages)
    authors = {message.author.id: message.author for message in messages}
    first_message = min(messages, key=lambda msg: msg.timestamp, default=None)
    last_message = max(messages, key=lambda msg: msg.timestamp, default=None)

    attachment_total = sum(len(message.attachments) for message in messages)
    embed_total = sum(len(message.embeds) for message in messages)

    reaction_counts = Counter()
    for message in messages:
        for reaction in message.reactions:
            reaction_counts[reaction.emoji.name] += reaction.count

    print("Guild:", bundle.guild.name)
    print("Channel:", bundle.channel.name)
    print("Exported at:", bundle.exported_at.isoformat())
    print("Reported messages:", bundle.message_count)
    print("Loaded messages:", message_count)
    print("Unique authors:", len(authors))
    print(
        "Time span:",
        _format_dt(first_message.timestamp if first_message else None),
        "→",
        _format_dt(last_message.timestamp if last_message else None),
    )
    print("Total attachments:", attachment_total)
    print("Total embeds:", embed_total)
    if reaction_counts:
        top_reaction, total = reaction_counts.most_common(1)[0]
        print("Top reaction:", f"{top_reaction} × {total}")
    else:
        print("Top reaction: none")


def main() -> None:
    data_path = Path(__file__).resolve().parent / "data" / "sample.json"
    bundle = load_export(data_path)
    _print_stats(bundle)


if __name__ == "__main__":
    main()
