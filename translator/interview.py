"""Interview copilot: profile-grounded live answer hints, tuned for SPEED.

Each assembled turn is first passed through a cheap regex gate
(looks_like_question) — only question-shaped turns cost a model call at all.
Gated-in turns go to Claude with the user's profile (CV, bio, projects) and a
pre-warmed company brief as a cached system prefix; the model returns a
structured verdict through the submit_hints tool.

Latency levers, in order:
- the gate (non-questions never call the model),
- the pre-warmed company brief (build_company_brief at session start) makes
  live web searches rare — the common company questions answer from the brief,
- streamed partials: bullets are surfaced from the tool-input JSON deltas as
  they generate (on_partial callback), so the card fills in ~1s instead of
  appearing whole at the end,
- Haiku + max one live search when something truly isn't covered.

The profile+brief prefix clears the prompt-cache minimum, so every turn after
the first reads it at ~0.1×.
"""

from __future__ import annotations

import re
import time
from typing import Callable

import anthropic


def _clean_bullet(text: str) -> str:
    """Strip citation markup the model sometimes copies from search results."""
    return re.sub(r"</?cite[^>]*>", "", text).strip()


# ── question gate ────────────────────────────────────────────────────────────

# Cues that a turn asks the candidate something. Deliberately permissive —
# a false positive costs one cheap Haiku call; a false negative costs a hint.
_QUESTION_STARTS = (
    "what", "how", "why", "when", "where", "which", "who", "whose",
    "tell", "walk", "describe", "explain", "share", "give", "talk",
    "do you", "did you", "have you", "has ", "are you", "were you",
    "can you", "could you", "would you", "will you", "is there", "any ",
    "so tell", "so what", "so how", "let's",
)
_QUESTION_ANYWHERE = (
    "?", "tell me", "tell us", "walk me", "walk us", "curious", "hear about",
    "hear more", "talk about", "your experience", "your background",
    "introduce yourself", "know about",
)


def looks_like_question(text: str) -> bool:
    """Cheap pre-filter: does this turn plausibly ask the candidate something?"""
    t = text.strip().lower()
    if not t:
        return False
    return t.startswith(_QUESTION_STARTS) or any(c in t for c in _QUESTION_ANYWHERE)


# ── prompt ───────────────────────────────────────────────────────────────────

_MAX_SEARCHES = 1  # the company brief covers the common lookups; keep live search rare

# Haiku is the latency sweet spot for glanceable bullets; override via env
# INTERVIEW_MODEL if hint quality ever needs the bump.
DEFAULT_HINT_MODEL = "claude-haiku-4-5"


def _profile_system_blocks(profile: str, context: str = "",
                           company_brief: str = "") -> list[dict]:
    """Cached system prefix: role + grounding rules + profile + company brief."""
    context_line = f"\nInterview context (role/company): {context}\n" if context else ""
    brief_block = (f"\nCOMPANY BRIEF (pre-researched — answer company questions from "
                   f"this, no search needed):\n{company_brief}\n" if company_brief else "")
    text = f"""\
You are a real-time interview copilot for the candidate whose profile follows.
You hear the interviewer's words (speech-to-text; may contain small errors) and
produce short answer hints the candidate can glance at while speaking. SPEED
matters: answer from the profile and company brief; be brief.

You ALWAYS finish by calling the submit_hints tool — it is your only way to
respond. Never reply with plain text.
{context_line}
Rules:
- Set is_question_for_me=true when the turn asks the candidate something or \
invites them to speak. That includes: "tell me about...", "walk me through...", \
"do you know / have you heard about X?", and "what do you know about our \
company/us/X?" — anything ending in a question mark directed at the candidate \
is a question for them. Only small talk, the interviewer describing something \
without asking, or the candidate's own words get is_question_for_me=false \
(with empty bullets).
- If you ran a web search, the turn was a question — is_question_for_me must be \
true and the bullets must use what you found.
- Bullets are memory joggers, not scripts: 3–5 bullets, each ≤ 12 words, concrete.
- Ground bullets about the CANDIDATE in facts from the profile below — a project \
name, a real technology, a real situation. NEVER invent experience, employers, \
numbers, or skills that are not in the profile.
- Company/product questions: answer from the COMPANY BRIEF when present. Use \
web_search ONLY for facts that neither the profile nor the brief covers (at most \
{_MAX_SEARCHES} search, tight query) — every search costs seconds.
- If nothing can answer, say so honestly in one bullet ("no direct experience — \
bridge from <closest real thing>") instead of inventing.
- angle: one short line naming the strongest framing for this answer.
- For follow-up questions, use the recent conversation to stay consistent with \
what was already said.

CANDIDATE PROFILE:
{profile}
{brief_block}"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


_HINTS_TOOL = {
    "name": "submit_hints",
    "description": "Submit answer hints for the candidate. Always call this exactly "
                   "once, as your final action.",
    "input_schema": {
        "type": "object",
        "properties": {
            "is_question_for_me": {
                "type": "boolean",
                "description": "true only if this turn asks the candidate something "
                               "or invites them to speak",
            },
            "question_gist": {
                "type": "string",
                "description": "the question in ≤10 words (empty if not a question)",
            },
            "bullets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "3-5 short answer hints, each ≤12 words, grounded in "
                               "the profile or search results (empty if not a question)",
            },
            "angle": {
                "type": "string",
                "description": "one line: the strongest framing for this answer "
                               "(empty if not a question)",
            },
        },
        "required": ["is_question_for_me", "question_gist", "bullets", "angle"],
    },
}


def _web_search_tool(model: str) -> dict:
    """The dynamic-filtering search variant needs Sonnet 4.6+/Opus 4.6+;
    Haiku (and older models) use the basic variant."""
    newer = any(m in model for m in ("sonnet-4-6", "sonnet-5", "opus-4-6",
                                     "opus-4-7", "opus-4-8", "fable"))
    return {
        "type": "web_search_20260209" if newer else "web_search_20250305",
        "name": "web_search",
        "max_uses": _MAX_SEARCHES,
    }


_NO_HINT = {"is_question": False, "gist": "", "bullets": [], "angle": "", "searched": False}


# ── partial-JSON extraction (streamed tool input) ────────────────────────────

_GIST_RE = re.compile(r'"question_gist"\s*:\s*"((?:[^"\\]|\\.)*)"')
_BULLET_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _parse_partial(buf: str) -> dict | None:
    """Best-effort extraction of gist + COMPLETED bullets from a partial
    submit_hints JSON string. Never json.loads a partial; regex only."""
    gist_m = _GIST_RE.search(buf)
    if not gist_m:
        return None
    arr_start = buf.find('"bullets"')
    bullets: list[str] = []
    if arr_start != -1:
        open_bracket = buf.find("[", arr_start)
        if open_bracket != -1:
            segment = buf[open_bracket + 1:]
            end = segment.find("]")
            if end != -1:
                segment = segment[:end]
            else:
                # drop the trailing (possibly unterminated) element
                segment = segment.rsplit('"', 1)[0] + '"' if segment.count('"') % 2 else segment
            bullets = [_clean_bullet(m.group(1).encode().decode("unicode_escape"))
                       for m in _BULLET_RE.finditer(segment)]
    try:
        gist = gist_m.group(1).encode().decode("unicode_escape")
    except Exception:
        gist = gist_m.group(1)
    return {"gist": gist.strip(), "bullets": [b for b in bullets if b][:5]}


def generate_hints(
    turn_text: str,
    history: list[str],
    profile: str,
    client: anthropic.Anthropic,
    model: str = DEFAULT_HINT_MODEL,
    context: str = "",
    speaker: str = "",
    company_brief: str = "",
    on_partial: Callable[[dict], None] | None = None,
) -> dict:
    """Return {"is_question", "gist", "bullets", "angle", "searched"}.

    tool_choice stays "auto" so the model can run the web-search server tool
    before submitting hints (a forced tool would preempt search). pause_turn
    (server-tool iteration limit) is resumed; a reply that never calls
    submit_hints is treated as not-a-question rather than crashing the meeting.

    ``on_partial`` (optional) receives {"gist", "bullets"} dicts as the tool
    input streams — throttled here to ≥150ms and only when content grew.
    Partial-parse problems degrade silently to final-only delivery.
    """
    recent = "\n".join(history[-8:])
    who = f"[{speaker}] " if speaker else ""
    user_msg = (
        (f"Recent conversation:\n{recent}\n\n" if recent else "")
        + f"New turn (interviewer side unless marked otherwise):\n{who}{turn_text}"
    )
    messages = [{"role": "user", "content": user_msg}]
    searched = False

    for _ in range(3):  # initial call + pause_turn resumes, bounded
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=_profile_system_blocks(profile, context, company_brief),
            tools=[_web_search_tool(model), _HINTS_TOOL],
            messages=messages,
        ) as stream:
            if on_partial is None:
                msg = stream.get_final_message()
            else:
                buf, last_emit, last_len = "", 0.0, 0
                in_hints_block = False
                try:
                    for event in stream:
                        et = getattr(event, "type", "")
                        if et == "content_block_start":
                            blk = getattr(event, "content_block", None)
                            in_hints_block = (getattr(blk, "type", "") == "tool_use"
                                              and getattr(blk, "name", "") == "submit_hints")
                        elif et == "content_block_delta" and in_hints_block:
                            delta = getattr(event, "delta", None)
                            if getattr(delta, "type", "") == "input_json_delta":
                                buf += delta.partial_json
                                now = time.monotonic()
                                if now - last_emit >= 0.15:
                                    partial = _parse_partial(buf)
                                    if partial and (len(partial["bullets"]),
                                                    len(partial["gist"])) != (0, 0) \
                                            and len(buf) > last_len:
                                        on_partial(partial)
                                        last_emit, last_len = now, len(buf)
                except Exception:
                    pass  # partial delivery is best-effort; the final message wins
                msg = stream.get_final_message()

        searched = searched or any(
            getattr(b, "type", "") == "server_tool_use" for b in msg.content)

        block = next((b for b in msg.content
                      if b.type == "tool_use" and b.name == "submit_hints"), None)
        if block is not None:
            inp = block.input
            raw = inp.get("bullets", [])
            if isinstance(raw, str):
                # The model occasionally emits the array as one string — split on
                # newlines rather than iterating characters.
                raw = [line.strip("-• \t") for line in raw.splitlines()]
            return {
                "is_question": bool(inp.get("is_question_for_me")),
                "gist": (inp.get("question_gist") or "").strip(),
                "bullets": [_clean_bullet(b) for b in raw
                            if isinstance(b, str) and _clean_bullet(b)][:5],
                "angle": (inp.get("angle") or "").strip(),
                "searched": searched,
            }

        if msg.stop_reason == "pause_turn":
            # Server-tool loop hit its iteration cap — resume where it left off.
            messages = [{"role": "user", "content": user_msg},
                        {"role": "assistant", "content": msg.content}]
            continue

        break  # ended without submit_hints — fall through to the safe default

    return dict(_NO_HINT, searched=searched)


# ── company brief pre-warm ───────────────────────────────────────────────────

def build_company_brief(role_company: str, client: anthropic.Anthropic,
                        model: str = DEFAULT_HINT_MODEL) -> str:
    """One search+summary pass at session start so company questions never need
    a live search mid-interview. Returns "" on any failure (brief is optional)."""
    if not role_company.strip():
        return ""
    try:
        with client.messages.stream(
            model=model,
            max_tokens=1024,
            system=("Research the company/role below with web search (1-2 tight "
                    "queries) and write a compact brief: what the company does, "
                    "products, scale, recent news, tech stack if known. ≤10 short "
                    "lines, plain text, no citations markup."),
            tools=[dict(_web_search_tool(model), max_uses=2)],
            messages=[{"role": "user", "content": role_company}],
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text").strip()
        return re.sub(r"</?cite[^>]*>", "", text)[:2000]
    except Exception:
        return ""
