"""Per-session terminology glossary for translation consistency.

Proper nouns, product names, and domain jargon otherwise drift between chunks
(live mode) and between the analyze/translate passes. A Glossary pins a source
term to a preferred English rendering and injects it into three places:

  - the STT prompt      → biases Whisper/gpt-4o toward the right spelling of names
  - the translator prompt → keeps the English rendering identical every time
  - the analyzer prompt   → the analysis is aware of fixed terms

It is intentionally tiny and per-session only (no persistence). Entries are
seeded from the user's "Key terms / names" field and can be accumulated during
a session via ``add``.
"""

from __future__ import annotations


class Glossary:
    """Ordered source-term → English-rendering map, insertion-preserving."""

    def __init__(self) -> None:
        # dict preserves insertion order; later adds win (user seed first).
        self._terms: dict[str, str] = {}

    # ── construction ─────────────────────────────────────────────────────────

    @classmethod
    def parse(cls, raw: str) -> "Glossary":
        """Build from a free-text field, one entry per line.

        Accepts ``term => rendering``, ``term: rendering``, ``term = rendering``,
        or ``term - rendering``. A line with no separator is treated as a term to
        preserve verbatim (rendering == term). Blank lines are ignored.
        """
        g = cls()
        for line in (raw or "").splitlines():
            line = line.strip()
            if not line:
                continue
            term, rendering = _split_entry(line)
            g.add(term, rendering)
        return g

    # ── mutation ─────────────────────────────────────────────────────────────

    def add(self, term: str, rendering: str = "") -> None:
        term = (term or "").strip()
        if not term:
            return
        self._terms[term] = (rendering or "").strip() or term

    def __bool__(self) -> bool:
        return bool(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    # ── rendering ────────────────────────────────────────────────────────────

    def format_for_prompt(self) -> str:
        """Human-readable block for the translator/analyzer system context.

        Returns an empty string when there are no terms so callers can cheaply
        skip the section.
        """
        if not self._terms:
            return ""
        lines = [f'  - "{term}" → "{rendering}"' for term, rendering in self._terms.items()]
        return (
            "Fixed terminology (render these exactly as given, every time):\n"
            + "\n".join(lines)
        )

    def format_for_stt(self) -> str:
        """Comma-separated source terms to bias speech recognition.

        Only the source-side terms are surfaced — the English renderings would
        just confuse a same-language STT prompt.
        """
        return ", ".join(self._terms.keys())


def _split_entry(line: str) -> tuple[str, str]:
    for sep in ("=>", "::", ":", "=", " - ", "→", "\t"):
        if sep in line:
            term, _, rendering = line.partition(sep)
            return term.strip(), rendering.strip()
    return line, line
