"""Pick the part of a retrieved transcript chunk that answers the question.

Atlandex embeds transcripts in chunks of up to 2,000 tokens, roughly 8,000
characters each. Returning whole chunks would put about 25,000 characters into
the agent's context for three hits. The server returns one passage per chunk
instead: the word window covering the most question terms, started a few words
before the first match. The opening words of that window are the anchor used to
look up its timestamp, so the cited link lands just before the relevant line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

WINDOW_WORDS = 110  # about 45-60 seconds of speech
# Context kept before the first matching word. Small, because the backend's seek
# already starts its result 10 seconds before the match.
LEAD_WORDS = 6
ANCHOR_WORDS = 14
MAX_PASSAGE_CHARS = 900
MIN_PREFIX_MATCH = 4

_TOKEN_RE = re.compile(r"[a-z0-9À-ɏ]{3,}")

# Short English and German function words, plus verbs that only frame a
# question ("what does she say about..."). Tokens under three characters are
# already dropped by _TOKEN_RE.
_STOPWORDS = frozenset(
    """
    about after again all also and any are because been before being both but can
    could did does doing down during each few for from further had has have having
    her here hers herself him himself his how into its itself just more most not now
    off once only other our ours out over own same she should some such than that the
    their theirs them then there these they this those through too under until very
    was were what when where which while who whom why will with would you your yours
    say says said tell talk talks talked think thinks mention mentions mentioned explain
    aber als auch auf aus bei bin bis das dass dem den der des die dies diese diesem
    diesen dieser dir doch dort durch ein eine einem einen einer eines für hat hatte
    ich ihr ihre im ist mit nach nicht noch nur oder sehr sein seine sich sie sind
    über und uns unter vom von vor war wie wir wird zum zur
    """.split()
)


@dataclass(frozen=True)
class SelectedPassage:
    text: str
    """Display excerpt with whitespace normalized, at most MAX_PASSAGE_CHARS long."""

    anchor: str
    """Opening words of the window, as spoken; used to find the passage's timestamp."""

    matched_terms: tuple[str, ...]
    """Question terms found in the window."""


def question_terms(question: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(question.lower()) if t not in _STOPWORDS}


def _term_hit(term: str, tokens: list[str]) -> bool:
    """Exact token match, or a shared prefix so 'embeddings' finds 'embedding'."""
    for tok in tokens:
        if tok == term:
            return True
        if (
            len(term) >= MIN_PREFIX_MATCH
            and len(tok) >= MIN_PREFIX_MATCH
            and (tok.startswith(term) or term.startswith(tok))
        ):
            return True
    return False


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    space = cut.rfind(" ")
    if space > max_chars * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:") + " …"


def _best_window(hits: list[frozenset[str]], window: int) -> int:
    """Start of the window with the most distinct terms, then the most hits, then the earliest."""
    n = len(hits)
    width = min(window, n)
    counts: dict[str, int] = {}
    total = 0

    def add(i: int, delta: int) -> None:
        nonlocal total
        for term in hits[i]:
            counts[term] = counts.get(term, 0) + delta
            if counts[term] == 0:
                del counts[term]
            total += delta

    for i in range(width):
        add(i, +1)
    best_start, best_key = 0, (len(counts), total)
    for start in range(1, n - width + 1):
        add(start - 1, -1)
        add(start + width - 1, +1)
        key = (len(counts), total)
        if key > best_key:
            best_start, best_key = start, key
    return best_start


def select_passage(
    chunk_text: str,
    question: str,
    *,
    window_words: int = WINDOW_WORDS,
    lead_words: int = LEAD_WORDS,
    anchor_words: int = ANCHOR_WORDS,
    max_chars: int = MAX_PASSAGE_CHARS,
) -> SelectedPassage:
    """Return the passage of `chunk_text` that covers the most distinct question terms.

    With no overlap at all, the chunk's opening window is returned, since dense
    retrieval still ranked the chunk as relevant to the question.
    """
    words = chunk_text.split()
    if not words:
        return SelectedPassage(text="", anchor="", matched_terms=())

    terms = question_terms(question)
    hits = [
        frozenset(t for t in terms if _term_hit(t, _TOKEN_RE.findall(w.lower()))) for w in words
    ]

    start = 0
    if any(hits):
        best = _best_window(hits, window_words)
        first_hit = next(i for i in range(best, len(words)) if hits[i])
        start = max(best, first_hit - lead_words)

    window = words[start : start + window_words]
    matched = sorted(set().union(*hits[start : start + window_words]))
    return SelectedPassage(
        text=_truncate(" ".join(window), max_chars),
        anchor=" ".join(window[:anchor_words]),
        matched_terms=tuple(matched),
    )
