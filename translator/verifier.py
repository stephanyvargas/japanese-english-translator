"""Back-translation meaning-drift check (opt-in, highest-quality runs only).

Round-trips the English back into the source language and asks the model to flag
any semantic drift versus the original. The findings are folded into the editor
critique so the refine pass can repair them. Off by default because it adds an
extra reasoning call per translation.
"""

import anthropic

from .models import DriftResult

_TOOL = {
    "name": "submit_drift",
    "description": "Report semantic differences between the original source and a back-translation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "has_drift": {
                "type": "boolean",
                "description": "True if the back-translation reveals any meaning change, omission, or addition",
            },
            "drift_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific meaning differences (empty if none)",
            },
        },
        "required": ["has_drift", "drift_notes"],
    },
}

_SYSTEM = """\
You are a bilingual meaning-verification checker. You are given an original source \
text and a back-translation of its English rendering (i.e. the English translated \
back into the source language). Compare them for MEANING only — ignore wording and \
style differences. Report any omissions, additions, or changes in meaning, referent, \
number, or nuance. Return your assessment with the submit_drift tool.\
"""


def check_drift(
    source_text: str,
    english_text: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
) -> DriftResult:
    """Back-translate the English and flag semantic drift against the source."""
    # Step 1: back-translate English → source language (literal, meaning-preserving).
    with client.messages.stream(
        model=model,
        max_tokens=2048,
        system=(
            f"Translate the following English text into {lang_name}. Be literal and "
            f"meaning-preserving. Output only the {lang_name} translation."
        ),
        messages=[{"role": "user", "content": english_text}],
    ) as stream:
        back = next((b.text for b in stream.get_final_message().content if b.type == "text"), "")

    # Step 2: compare the back-translation against the original source.
    user_message = (
        f"Original source ({lang_name}):\n{source_text}\n\n"
        f"Back-translation ({lang_name}):\n{back}"
    )
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_SYSTEM,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_drift"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        msg = stream.get_final_message()

    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return DriftResult(**tool_block.input)
