import anthropic

from .models import AnalysisResult
from .prompts import TRANSLATOR_PROMPT


def _build_context(analysis: AnalysisResult, critique: str | None) -> str:
    lines = [
        f"Domain: {analysis.domain}",
        f"Formality: {analysis.formality_level}",
        f"Uses keigo: {analysis.has_keigo}",
    ]
    if analysis.implicit_subjects:
        lines.append("Implicit subjects: " + ", ".join(analysis.implicit_subjects))
    if analysis.cultural_notes:
        lines.append("Cultural notes:")
        for note in analysis.cultural_notes:
            lines.append(f"  - {note}")
    if critique:
        lines.append(f"\nEditor critique to address:\n{critique}")
    return "\n".join(lines)


def translate(
    japanese_text: str,
    analysis: AnalysisResult,
    client: anthropic.Anthropic,
    critique: str | None = None,
) -> str:
    """Produce an English translation using adaptive thinking for deep reasoning."""
    context = _build_context(analysis, critique)
    user_message = (
        f"Linguistic context:\n{context}\n\n"
        f"Japanese source:\n{japanese_text}\n\n"
        "Translate to English:"
    )

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=TRANSLATOR_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        msg = stream.get_final_message()

    return next(b.text for b in msg.content if b.type == "text")
