import anthropic

from .models import AnalysisResult
from .prompts import get_translator_prompt


def _build_context(analysis: AnalysisResult, lang_name: str, critique: str | None) -> str:
    lines = [
        f"Domain: {analysis.domain}",
        f"Formality: {analysis.formality_level}",
        f"Uses honorifics: {analysis.has_honorifics}",
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
    source_text: str,
    analysis: AnalysisResult,
    client: anthropic.Anthropic,
    critique: str | None = None,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
) -> str:
    """Produce an English translation using adaptive thinking for deep reasoning."""
    context = _build_context(analysis, lang_name, critique)
    user_message = (
        f"Linguistic context:\n{context}\n\n"
        f"{lang_name} source:\n{source_text}\n\n"
        "Translate to English:"
    )

    with client.messages.stream(
        model=model,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        system=get_translator_prompt(lang_name),
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        msg = stream.get_final_message()

    return next((b.text for b in msg.content if b.type == "text"), "")
