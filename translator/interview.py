"""Interview copilot: profile-grounded answer hints for live questions.

Each assembled turn from the interviewer is sent to Claude with the user's
profile (CV, bio, projects) as a cached system prefix; the model returns a
structured verdict through a forced tool — is this a question aimed at the
candidate, and if so, 3–5 short bullets grounded ONLY in profile facts. The
bullets are glanceable memory joggers, never scripts.

The profile prefix is typically 2–6k tokens, which clears the prompt-cache
minimum, so every turn after the first reads the profile at ~0.1×.
"""

from __future__ import annotations

import anthropic

from .pipeline import _text_of  # noqa: F401  (kept import surface consistent)


def _profile_system_blocks(profile: str, context: str = "") -> list[dict]:
    """Cached system prefix: role + grounding rules + the candidate's profile."""
    context_line = f"\nInterview context: {context}\n" if context else ""
    text = f"""\
You are a real-time interview copilot for the candidate whose profile follows.
You hear the interviewer's words (speech-to-text; may contain small errors) and
produce short answer hints the candidate can glance at while speaking.
{context_line}
Rules:
- Set is_question_for_me=true ONLY when the turn asks the candidate something or \
invites them to speak (questions, "tell me about...", "walk me through..."). \
Small talk, the interviewer describing the company, or the candidate's own words \
→ is_question_for_me=false with empty bullets.
- Bullets are memory joggers, not scripts: 3–5 bullets, each ≤ 12 words, concrete.
- Ground every bullet in facts from the profile below — a project name, a real \
technology, a real situation. NEVER invent experience, employers, numbers, or \
skills that are not in the profile.
- If the profile has nothing relevant, say so honestly in one bullet \
("no direct experience — bridge from <closest real thing>") instead of inventing.
- angle: one short line naming the strongest framing for this answer.
- For follow-up questions, use the recent conversation to stay consistent with \
what was already said.

CANDIDATE PROFILE:
{profile}
"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


_HINTS_TOOL = {
    "name": "submit_hints",
    "description": "Submit answer hints for the candidate.",
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
                               "the profile (empty if not a question)",
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


def generate_hints(
    turn_text: str,
    history: list[str],
    profile: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    context: str = "",
    speaker: str = "",
) -> dict:
    """Return {"is_question": bool, "gist": str, "bullets": [str], "angle": str}.

    One pass, no self-repair — latency matters more here. History is the recent
    transcript (both voices) so follow-up questions resolve.
    """
    recent = "\n".join(history[-8:])
    who = f"[{speaker}] " if speaker else ""
    user_msg = (
        (f"Recent conversation:\n{recent}\n\n" if recent else "")
        + f"New turn (interviewer side unless marked otherwise):\n{who}{turn_text}"
    )
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=_profile_system_blocks(profile, context),
        tools=[_HINTS_TOOL],
        tool_choice={"type": "tool", "name": "submit_hints"},
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    block = next((b for b in msg.content if b.type == "tool_use"), None)
    if block is None:
        return {"is_question": False, "gist": "", "bullets": [], "angle": ""}
    inp = block.input
    return {
        "is_question": bool(inp.get("is_question_for_me")),
        "gist": (inp.get("question_gist") or "").strip(),
        "bullets": [b.strip() for b in inp.get("bullets", []) if b.strip()][:5],
        "angle": (inp.get("angle") or "").strip(),
    }
