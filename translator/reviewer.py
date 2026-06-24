import anthropic

from .models import AnalysisResult, ReviewResult
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
    japanese_text: str,
    english_text: str,
    analysis: AnalysisResult,
    client: anthropic.Anthropic,
) -> ReviewResult:
    """Run a bilingual quality review on the translation."""
    user_message = (
        f"Source (Japanese):\n{japanese_text}\n\n"
        f"Translation (English):\n{english_text}\n\n"
        f"Context: domain={analysis.domain}, formality={analysis.formality_level}, "
        f"has_keigo={analysis.has_keigo}"
    )
    if analysis.cultural_notes:
        user_message += "\nCultural notes: " + "; ".join(analysis.cultural_notes)

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=REVIEWER_PROMPT,
        tools=[_TOOL],
        tool_choice={"type": "tool", "name": "submit_review"},
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        msg = stream.get_final_message()

    tool_block = next(b for b in msg.content if b.type == "tool_use")
    return ReviewResult(**tool_block.input)
