import anthropic

from .models import AnalysisResult, ReviewResult, thinking_kwargs
from .prompts import REVIEWER_PROMPT

_TOOL = {
    "name": "submit_review",
    "description": "Submit the structured quality review of the translation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy_score": {
                "type": "integer",
                "description": "Accuracy score 1–10: faithfulness to source meaning, nuance, and intent",
            },
            "naturalness_score": {
                "type": "integer",
                "description": "Naturalness score 1–10: idiomatic, fluent English prose quality",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific translation problems found (empty if none)",
            },
            "suggestions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Concrete suggestions addressing each issue",
            },
        },
        "required": ["accuracy_score", "naturalness_score", "issues", "suggestions"],
    },
}


def review(
    source_text: str,
    english_text: str,
    analysis: AnalysisResult,
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
    context: str = "",
) -> ReviewResult:
    """Run a bilingual quality review on the translation."""
    setting_line = f"Setting: {context}\n" if context else ""
    user_message = (
        f"{setting_line}"
        f"Source ({lang_name}):\n{source_text}\n\n"
        f"Translation (English):\n{english_text}\n\n"
        f"Linguistic context: domain={analysis.domain}, formality={analysis.formality_level}, "
        f"has_honorifics={analysis.has_honorifics}"
    )
    if analysis.cultural_notes:
        user_message += "\nCultural notes: " + "; ".join(analysis.cultural_notes)

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        **thinking_kwargs(model),
        system=REVIEWER_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_review"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        msg = stream.get_final_message()

    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return ReviewResult(**tool_block.input)
