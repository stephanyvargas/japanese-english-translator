"""Interview copilot: profile-grounded live answer hints for live questions.

Each assembled turn from the interviewer is sent to Claude with the user's
profile (CV, bio, projects) as a cached system prefix. The model returns a
structured verdict through the submit_hints tool — is this a question aimed at
the candidate, and if so, 3–5 short bullets grounded in profile facts.

When the profile can't answer (company facts, technical definitions, recent
events), the model may use the **web search server tool** (runs on Anthropic's
side, no client loop) before submitting hints — bullets then cite what was
found instead of inventing. Search is bounded (max_uses) to protect latency.

The profile prefix is typically 2–6k tokens, which clears the prompt-cache
minimum, so every turn after the first reads the profile at ~0.1×.
"""

from __future__ import annotations

import anthropic


def _profile_system_blocks(profile: str, context: str = "") -> list[dict]:
    """Cached system prefix: role + grounding rules + the candidate's profile."""
    context_line = f"\nInterview context (role/company): {context}\n" if context else ""
    text = f"""\
You are a real-time interview copilot for the candidate whose profile follows.
You hear the interviewer's words (speech-to-text; may contain small errors) and
produce short answer hints the candidate can glance at while speaking.

You ALWAYS finish by calling the submit_hints tool — it is your only way to
respond. Never reply with plain text.
{context_line}
Rules:
- Set is_question_for_me=true ONLY when the turn asks the candidate something or \
invites them to speak (questions, "tell me about...", "walk me through..."). \
Small talk, the interviewer describing the company, or the candidate's own words \
→ call submit_hints immediately with is_question_for_me=false and empty bullets.
- Bullets are memory joggers, not scripts: 3–5 bullets, each ≤ 12 words, concrete.
- Ground bullets about the CANDIDATE in facts from the profile below — a project \
name, a real technology, a real situation. NEVER invent experience, employers, \
numbers, or skills that are not in the profile.
- If the question needs facts the profile does not have — the company, a product, \
a technical concept, a recent event — use web_search FIRST (at most {_MAX_SEARCHES} \
searches, keep queries tight), then ground those bullets in what you found. \
Do NOT search for things the profile already answers; speed matters.
- If neither the profile nor a search can answer, say so honestly in one bullet \
("no direct experience — bridge from <closest real thing>") instead of inventing.
- angle: one short line naming the strongest framing for this answer.
- For follow-up questions, use the recent conversation to stay consistent with \
what was already said.

CANDIDATE PROFILE:
{profile}
"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


_MAX_SEARCHES = 2  # bounds added latency per question

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

_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": _MAX_SEARCHES,
}

_NO_HINT = {"is_question": False, "gist": "", "bullets": [], "angle": "", "searched": False}


def generate_hints(
    turn_text: str,
    history: list[str],
    profile: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    context: str = "",
    speaker: str = "",
) -> dict:
    """Return {"is_question", "gist", "bullets", "angle", "searched"}.

    tool_choice stays "auto" so the model can run the web-search server tool
    before submitting hints (a forced tool would preempt search). pause_turn
    (server-tool iteration limit) is resumed; a reply that never calls
    submit_hints is treated as not-a-question rather than crashing the meeting.
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
            max_tokens=2048,
            system=_profile_system_blocks(profile, context),
            tools=[_WEB_SEARCH_TOOL, _HINTS_TOOL],
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()

        searched = searched or any(
            getattr(b, "type", "") == "server_tool_use" for b in msg.content)

        block = next((b for b in msg.content
                      if b.type == "tool_use" and b.name == "submit_hints"), None)
        if block is not None:
            inp = block.input
            return {
                "is_question": bool(inp.get("is_question_for_me")),
                "gist": (inp.get("question_gist") or "").strip(),
                "bullets": [b.strip() for b in inp.get("bullets", []) if b.strip()][:5],
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
