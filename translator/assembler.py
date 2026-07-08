"""Sentence assembly for pause-cut transcript chunks.

The browser VAD cuts audio at ~700 ms pauses, so a single spoken sentence often
arrives as several chunks (「住む場所は」→「結構色々なところに」→「住んでいた…」).
Translating fragments in isolation produces fragment English — instead, chunks
that are visibly mid-sentence are buffered and joined with what follows, and the
completed sentence is translated once. Latency is added only in the exact case
where translating immediately would produce junk.
"""

from __future__ import annotations

import re

# A chunk is considered sentence-final when it ends with terminal punctuation or
# a terminal predicate. Deliberately small lists — a heuristic that catches most
# real fragments is the goal, and the assembler's caps bound any mistake.
_TERMINAL_PUNCT = "。？！?!.」』"

_TERMINAL_ENDINGS = (
    "です", "ます", "でした", "ました", "ません", "ですね", "ますね", "ですよ",
    "ますよ", "ください", "でしょう", "ましょう", "だ", "だよ", "だね", "よね",
)

# Endings that mark an explicitly *continuing* clause even after punctuation:
# 「欲しくて。でも」 ends with a conjunction — the thought is not finished.
_CONTINUATION_ENDINGS = (
    "でも", "けど", "けれど", "が、", "して", "くて", "って", "とか", "たり",
    "は", "が", "を", "に", "で", "と", "も", "の", "から", "ので",
)


def looks_complete(text: str) -> bool:
    """True when the transcript chunk looks like a finished sentence."""
    t = text.strip()
    if not t:
        return True
    if any(t.endswith(e) for e in _CONTINUATION_ENDINGS):
        return False
    if t[-1] in _TERMINAL_PUNCT:
        return True
    return any(t.endswith(e) for e in _TERMINAL_ENDINGS)


_MAX_PARTS = 4          # join at most this many chunks
_MAX_SECONDS = 30.0     # ... or this much audio
_MAX_CHARS = 200        # ... or this much text — then emit regardless


class ChunkAssembler:
    """Joins mid-sentence chunks until the sentence completes (or a cap hits).

    add() returns the text ready to translate, or None when the chunk was
    buffered. flush() force-emits whatever is pending (call it when the speaker
    has gone silent).
    """

    def __init__(self):
        self._parts: list[str] = []
        self._seconds = 0.0
        self.last_merged = 0  # how many chunks made up the last flushed text

    @property
    def pending(self) -> bool:
        return bool(self._parts)

    @property
    def parts(self) -> int:
        return len(self._parts)

    def add(self, text: str, dur_s: float = 0.0) -> str | None:
        text = text.strip()
        if not text:
            return self.flush()
        self._parts.append(text)
        self._seconds += dur_s
        capped = (len(self._parts) >= _MAX_PARTS
                  or self._seconds >= _MAX_SECONDS
                  or sum(len(p) for p in self._parts) >= _MAX_CHARS)
        if looks_complete(text) or capped:
            return self.flush()
        return None

    def flush(self) -> str | None:
        if not self._parts:
            return None
        # Japanese needs no space at joins; re.sub collapses any doubled 。
        joined = re.sub(r"。+", "。", "".join(self._parts))
        self.last_merged = len(self._parts)
        self._parts = []
        self._seconds = 0.0
        return joined
