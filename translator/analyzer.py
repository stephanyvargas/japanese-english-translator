import json

import anthropic

from .models import AnalysisResult
from .prompts import ANALYZER_PROMPT

_TOOL = {
    "name": "submit_analysis",
    "description": "Submit the structured linguistic analysis of the Japanese text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Text domain: casual, business, literary, technical, news, or formal_document",
            },
            "formality_level": {
                "type": "string",
                "description": "Formality level: very_casual, casual, polite, formal, keigo_sonkeigo, keigo_kenjogo, keigo_teineigo, or archaic",
            },
            "has_keigo": {
                "type": "boolean",
                "description": "Whether the text uses any form of keigo",
            },
            "cultural_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cultural concepts, idioms, or references without direct English equivalents",
            },
            "implicit_subjects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subjects that are dropped but can be inferred from context",
            },
        },
        "required": ["domain", "formality_level", "has_keigo", "cultural_notes", "implicit_subjects"],
    },
}


def analyze(japanese_text: str, client: anthropic.Anthropic) -> AnalysisResult:
    """Run a linguistic analysis pass on the Japanese text."""
    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=1024,
        system=ANALYZER_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_analysis"},
        messages=[
            {"role": "user", "content": f"Analyze this Japanese text:\n\n{japanese_text}"}
        ],
    ) as stream:
        msg = stream.get_final_message()

    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return AnalysisResult(**tool_block.input)
