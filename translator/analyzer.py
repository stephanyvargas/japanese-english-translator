import anthropic

from .models import AnalysisResult
from .prompts import get_analyzer_prompt

_TOOL = {
    "name": "submit_analysis",
    "description": "Submit the structured linguistic analysis of the source text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "Text domain: casual, business, literary, technical, news, or formal_document",
            },
            "formality_level": {
                "type": "string",
                "description": "Formality level using the scale appropriate for the source language",
            },
            "has_honorifics": {
                "type": "boolean",
                "description": "Whether the text uses honorific, humble, or elevated language forms",
            },
            "cultural_notes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Cultural concepts, idioms, or references without direct English equivalents",
            },
            "implicit_subjects": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Subjects or referents that are omitted but can be inferred from context",
            },
        },
        "required": ["domain", "formality_level", "has_honorifics", "cultural_notes", "implicit_subjects"],
    },
}


def analyze(
    source_text: str,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
) -> AnalysisResult:
    """Run a linguistic analysis pass on the source text."""
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=get_analyzer_prompt(lang_name),
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_analysis"},
        messages=[
            {"role": "user", "content": f"Analyze this {lang_name} text:\n\n{source_text}"}
        ],
    ) as stream:
        msg = stream.get_final_message()

    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return AnalysisResult(**tool_block.input)
