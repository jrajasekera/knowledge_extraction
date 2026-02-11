"""Deterministic fallback query expansion for when LLM query extraction is unavailable."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Hard cap matching SemanticSearchInput.queries max_length=20
_SCHEMA_MAX_QUERIES = 20

# Common English stopwords for filtering
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "i",
        "me",
        "my",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "it",
        "they",
        "them",
        "their",
        "his",
        "her",
        "its",
        "this",
        "that",
        "these",
        "those",
        "not",
        "no",
        "so",
        "if",
        "then",
        "than",
        "too",
        "very",
        "just",
        "about",
        "also",
        "more",
        "some",
        "any",
        "all",
        "each",
        "there",
        "here",
        "from",
        "up",
        "out",
        "into",
    }
)

# Question words to strip when decomposing questions
_QUESTION_WORDS: frozenset[str] = frozenset(
    {"who", "what", "where", "when", "why", "how", "does", "is", "are", "do", "did", "can", "tell"}
)

# Regex for sequences of capitalized words (2+ words starting with uppercase)
_CAPITALIZED_SEQ_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


@dataclass(slots=True)
class FallbackQuery:
    """A query produced by deterministic fallback expansion."""

    text: str
    source: str  # "entity" | "keyword" | "phrase" | "question" | "goal"


def _extract_entities(text: str) -> list[str]:
    """Extract proper nouns and named entities from text via capitalization patterns."""
    entities: list[str] = []

    # Multi-word capitalized sequences (e.g., "Machine Learning", "New York")
    for match in _CAPITALIZED_SEQ_RE.finditer(text):
        entities.append(match.group(1))

    # Single capitalized words not at sentence start
    words = text.split()
    for i, word in enumerate(words):
        cleaned = word.strip("?!.,;:'\"()[]")
        if not cleaned or not cleaned[0].isupper():
            continue
        # Skip sentence-initial words (first word, or after sentence-ending punctuation)
        if i == 0:
            continue
        prev = words[i - 1].rstrip()
        if prev and prev[-1] in ".?!":
            continue
        if (
            cleaned[0].isupper()
            and cleaned[1:].islower()
            and len(cleaned) >= 3
            and cleaned.lower() not in _STOPWORDS
        ):
            entities.append(cleaned)

    # Also grab words that are fully uppercase (acronyms) with 2+ chars
    for word in words:
        cleaned = word.strip("?!.,;:'\"()[]")
        if cleaned.isupper() and len(cleaned) >= 2 and cleaned.isalpha():
            entities.append(cleaned)

    return entities


def _extract_keywords(text: str) -> list[str]:
    """Extract significant 1-2 word keywords after stopword filtering."""
    words = re.findall(r"[a-zA-Z]+(?:'[a-z]+)?", text.lower())
    significant = [w for w in words if w not in _STOPWORDS and len(w) > 2]

    keywords: list[str] = []
    # Single keywords
    keywords.extend(significant)

    # Bigrams from significant words (adjacent in original text)
    for i in range(len(significant) - 1):
        keywords.append(f"{significant[i]} {significant[i + 1]}")

    return keywords


def _extract_phrases(text: str, top_n: int = 5) -> list[str]:
    """Extract 2-4 word phrases scored by information density."""
    words = text.split()
    if len(words) < 2:
        return []

    scored: list[tuple[float, str]] = []
    for window_size in (2, 3, 4):
        for i in range(len(words) - window_size + 1):
            chunk = words[i : i + window_size]
            phrase = " ".join(chunk)
            # Score by non-stopword ratio
            clean_words = [re.sub(r"[^a-zA-Z]", "", w).lower() for w in chunk]
            non_stop = sum(1 for w in clean_words if w and w not in _STOPWORDS)
            if non_stop == 0:
                continue
            density = non_stop / len(chunk)
            scored.append((density, phrase))

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    seen: set[str] = set()
    result: list[str] = []
    for _, phrase in scored:
        lower = phrase.lower()
        if lower not in seen:
            seen.add(lower)
            result.append(phrase)
        if len(result) >= top_n:
            break
    return result


def _decompose_question(text: str) -> list[str]:
    """Strip question words and extract content from questions."""
    queries: list[str] = []
    for sentence in re.split(r"[.!?]+", text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Only process question-like fragments
        words = sentence.split()
        content_words = [w for w in words if w.lower().strip("?") not in _QUESTION_WORDS]
        if content_words and len(content_words) < len(words):
            query = " ".join(content_words)
            # Clean punctuation
            query = re.sub(r"[?!]+$", "", query).strip()
            if query and len(query) > 2:
                queries.append(query)
    return queries


def _deduplicate(
    queries: list[FallbackQuery],
    previous: set[str],
) -> list[FallbackQuery]:
    """Remove duplicate and previously-tried queries (case-insensitive)."""
    seen: set[str] = set()
    result: list[FallbackQuery] = []
    for q in queries:
        key = q.text.strip().lower()
        if not key:
            continue
        if key in seen:
            continue
        # Check against previous queries (substring match)
        if any(key in prev or prev in key for prev in previous):
            continue
        seen.add(key)
        result.append(q)
    return result


# Source priority order for sorting
_SOURCE_PRIORITY = {"entity": 0, "keyword": 1, "phrase": 2, "question": 3, "goal": 4}


def expand_fallback_queries(
    last_message: str,
    recent_messages: list[str] | None = None,
    goal: str | None = None,
    previous_queries: list[str] | None = None,
    max_queries: int = 12,
    max_query_length: int = 120,
) -> list[FallbackQuery]:
    """Generate diverse fallback queries from conversation text without LLM.

    Args:
        last_message: The most recent message content (original casing).
        recent_messages: Recent conversation messages for context mining.
        goal: The current retrieval goal, if different from last_message.
        previous_queries: Queries already tried (will be excluded).
        max_queries: Maximum number of queries to return.
        max_query_length: Maximum character length per query.

    Returns:
        Ordered list of FallbackQuery objects, prioritized by source type.
    """
    if not last_message and not goal:
        return []

    effective_max = min(max_queries, _SCHEMA_MAX_QUERIES)
    prev_set = {q.strip().lower() for q in (previous_queries or []) if q.strip()}
    candidates: list[FallbackQuery] = []

    text = last_message or goal or ""

    # 1. Entity extraction from last message
    for entity in _extract_entities(text):
        candidates.append(FallbackQuery(text=entity, source="entity"))

    # 2. Keyword extraction from last message
    for kw in _extract_keywords(text):
        candidates.append(FallbackQuery(text=kw, source="keyword"))

    # 3. Phrase extraction from last message
    for phrase in _extract_phrases(text):
        candidates.append(FallbackQuery(text=phrase, source="phrase"))

    # 4. Question decomposition
    if "?" in text:
        for q in _decompose_question(text):
            candidates.append(FallbackQuery(text=q, source="question"))

    # 5. Goal-based query (if goal differs from last message)
    if goal and goal.strip().lower() != (last_message or "").strip().lower():
        candidates.append(FallbackQuery(text=goal.strip(), source="goal"))

    # 6. Recent context mining
    if recent_messages:
        recent_entities_seen = {c.text.lower() for c in candidates if c.source == "entity"}
        recent_keywords_seen = {c.text.lower() for c in candidates if c.source == "keyword"}

        for msg in recent_messages:
            for entity in _extract_entities(msg):
                if entity.lower() not in recent_entities_seen:
                    recent_entities_seen.add(entity.lower())
                    candidates.append(FallbackQuery(text=entity, source="entity"))

            for kw in _extract_keywords(msg):
                if kw.lower() not in recent_keywords_seen:
                    recent_keywords_seen.add(kw.lower())
                    candidates.append(FallbackQuery(text=kw, source="keyword"))

    # Truncate each query to max length
    for c in candidates:
        if len(c.text) > max_query_length:
            c.text = c.text[:max_query_length].rsplit(" ", 1)[0]

    # Deduplicate and filter previously-tried queries
    candidates = _deduplicate(candidates, prev_set)

    # Sort by source priority
    candidates.sort(key=lambda q: _SOURCE_PRIORITY.get(q.source, 99))

    # Cap to effective maximum
    return candidates[:effective_max]
