import sys
import time
from typing import NamedTuple

import anthropic
from openai import OpenAI

from .analyzer import analyze
from .audio import AudioCapture, record_from_mic, stream_chunks
from .models import FinalOutput
from .reviewer import review
from .transcriber import transcribe
from .translator import translate
from .verifier import check_drift


class ConvResult(NamedTuple):
    """Result of a conversation-mode translation."""
    english: str
    repaired: bool


def _text_of(msg) -> str:
    """Concatenate the text blocks of a message (skips thinking blocks)."""
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def _get_conversation_system(lang_name: str, context: str = "", glossary: str = "") -> str:
    context_line = f"\nSetting: {context}\n" if context else ""
    glossary_line = f"\n{glossary}\n" if glossary else ""
    return f"""\
You are a real-time {lang_name}-to-English conversation interpreter.
{context_line}{glossary_line}
You receive the conversation transcript so far (in {lang_name}) plus a new chunk to translate.
Use the full history to resolve pronouns, implicit subjects, topic references, and incomplete
sentences — context that would be impossible to translate in isolation.

Rules:
- Output ONLY the English translation of the NEW chunk — nothing else, no labels, no commentary
- Short utterances and conversational fillers → natural English equivalents \
("Yeah.", "Right, right.", "So, um...", "Uh...")
- If the chunk is an incomplete clause (cut off at a pause), translate what is there \
and let the next chunk continue it — do not invent an ending
- Reproduce numbers, dates, quantities, units, and proper nouns exactly as spoken
- Convey the politeness/honorific register through natural English register, not literal honorifics
- Keep names and untranslatable loanwords consistent with how they appeared earlier
- Genuine noise or silence → output "(inaudible)"
"""


_REPAIR_SYSTEM = """\
You are a bilingual interpreter's quality checker. You are given a source-language \
chunk and a candidate English translation of it, plus recent conversation context.

If the translation is faithful and natural, output it UNCHANGED.
If it drops meaning, mistakes a referent/subject, mishandles a number or name, or reads \
awkwardly, output a corrected English translation.

Output ONLY the final English translation — no labels, no commentary, no explanation.\
"""


def _repair_translation(
    new_source: str,
    candidate: str,
    context_block: str,
    client: anthropic.Anthropic,
    model: str,
    lang_name: str,
    glossary: str = "",
) -> str:
    """One quick pass that returns a corrected translation, or the candidate unchanged."""
    glossary_line = f"{glossary}\n\n" if glossary else ""
    user_msg = (
        f"{glossary_line}{context_block}"
        f"{lang_name} chunk:\n{new_source}\n\n"
        f"Candidate English:\n{candidate}"
    )
    with client.messages.stream(
        model=model,
        max_tokens=512,
        system=_REPAIR_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    return _text_of(msg) or candidate


def _translate_with_context(
    new_source: str,
    history: list[str],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
    context: str = "",
    glossary: str = "",
    repair: bool = True,
) -> ConvResult:
    """Context-aware conversation translation with a guarded self-repair pass.

    Quality is prioritized over latency here (the frontend's backpressure bounds
    lag to ~1 chunk): the first pass uses adaptive thinking, then a cheap repair
    pass corrects the translation only when it actually needs it.
    """
    previous = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(history[:-1]))
    context_block = f"Conversation so far:\n{previous}\n\n" if previous else "(start of conversation)\n\n"

    user_msg = f"{context_block}NEW chunk to translate:\n{new_source}"

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_get_conversation_system(lang_name, context, glossary),
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()

    english = _text_of(msg)
    if not english:
        return ConvResult("", False)

    repaired = False
    if repair:
        fixed = _repair_translation(
            new_source, english, context_block, client, model, lang_name, glossary,
        )
        if fixed and fixed != english:
            english = fixed
            repaired = True

    return ConvResult(english, repaired)


def run_conversation(
    interval_seconds: int = 8,
    min_speech_ratio: float = 0.15,
    threshold: float = 0.02,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
    context: str = "",
) -> None:
    """Continuous conversation translation with threaded audio capture."""
    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()

    capture = AudioCapture()
    capture.start()

    source_history: list[str] = []
    chunk_num = 0
    lang_tag = source_lang.upper()

    print(
        f"Conversation mode — mic is always on, translating every ~{interval_seconds}s.\n"
        "Press Ctrl+C to stop.\n",
        flush=True,
    )

    try:
        while True:
            time.sleep(interval_seconds)

            frames = capture.drain()
            if not frames:
                continue

            ratio = capture.speech_ratio(frames, threshold)
            if ratio < min_speech_ratio:
                continue

            wav_bytes, duration = capture.to_wav(frames)
            prompt = source_history[-1] if source_history else ""

            print(f"[{duration:.1f}s audio → transcribing...]", flush=True)
            new_text = transcribe(wav_bytes, openai_client, prompt=prompt, source_lang=source_lang)

            if not new_text.strip():
                print("[silence/noise — skipped]\n", flush=True)
                continue

            source_history.append(new_text)
            chunk_num += 1

            print(f"[{lang_tag}] {new_text}", flush=True)
            english, _ = _translate_with_context(
                new_text, source_history, anthropic_client, model=model, lang_name=lang_name, context=context
            )

            if not english:
                print("[translation produced no output — skipping]\n", flush=True)
                continue

            print(f"[EN] {english}\n", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        capture.stop()


# ── Single-shot and quality pipeline ─────────────────────────────────────────

def _build_critique(rev) -> str:
    parts = [f"Accuracy: {rev.accuracy_score}/10, Naturalness: {rev.naturalness_score}/10"]
    if rev.issues:
        parts.append("Issues:\n" + "\n".join(f"  - {i}" for i in rev.issues))
    if rev.suggestions:
        parts.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in rev.suggestions))
    return "\n".join(parts)


def _make_notes(analysis, review_result, lang_name: str, refine_rounds: int,
                drift_notes: list[str] | None = None) -> list[str]:
    notes: list[str] = []
    if analysis.has_honorifics:
        notes.append(f"Honorific register detected ({analysis.formality_level}): conveyed through English register.")
    for note in analysis.cultural_notes:
        notes.append(f"Cultural: {note}")
    if analysis.implicit_subjects:
        notes.append("Implicit subjects inferred: " + ", ".join(analysis.implicit_subjects))
    for note in drift_notes or []:
        notes.append(f"Back-translation flagged (and addressed): {note}")
    if refine_rounds:
        plural = "s" if refine_rounds > 1 else ""
        notes.append(f"Translation was refined after editorial review ({refine_rounds} round{plural}).")
    if review_result.accuracy_score < 9:
        notes.append(f"Editor accuracy score: {review_result.accuracy_score}/10")
    return notes


# Refine while the editor still finds issues AND either score is below target.
_REFINE_MAX_ROUNDS = 2
_ACCURACY_TARGET = 9
_NATURALNESS_TARGET = 8


def _needs_refine(rev) -> bool:
    return bool(rev.issues) and (
        rev.accuracy_score < _ACCURACY_TARGET or rev.naturalness_score < _NATURALNESS_TARGET
    )


def run(
    source_text: str,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
    context: str = "",
    glossary: str = "",
    verify: bool = False,
) -> FinalOutput:
    """Full quality pipeline: analyze → translate → review → refine (≤2 rounds).

    ``glossary`` pins terminology across passes. ``verify`` enables an opt-in
    back-translation meaning-drift check that feeds the refine step.
    """
    anthropic_client = anthropic.Anthropic()

    print("[1/3] Analyzing text...", flush=True)
    analysis = analyze(source_text, anthropic_client, model=model, source_lang=source_lang,
                       lang_name=lang_name, context=context, glossary=glossary)

    print("[2/3] Translating...", flush=True)
    english = translate(source_text, analysis, anthropic_client, model=model, lang_name=lang_name,
                        context=context, glossary=glossary)

    print("[3/3] Reviewing quality...", flush=True)
    review_result = review(source_text, english, analysis, anthropic_client, model=model,
                           lang_name=lang_name, context=context)

    # Optional back-translation drift check — folded into the critique below.
    drift_notes: list[str] = []
    if verify:
        print("[+] Back-translation drift check...", flush=True)
        drift = check_drift(source_text, english, anthropic_client, model=model, lang_name=lang_name)
        if drift.has_drift:
            drift_notes = drift.drift_notes
    addressed_drift = list(drift_notes)  # recorded in notes even after the refine consumes it

    refine_rounds = 0
    while (_needs_refine(review_result) or drift_notes) and refine_rounds < _REFINE_MAX_ROUNDS:
        print(f"[+] Refining based on editorial critique (round {refine_rounds + 1})...", flush=True)
        critique = _build_critique(review_result)
        if drift_notes:
            critique += "\nBack-translation drift to fix:\n" + "\n".join(f"  - {n}" for n in drift_notes)
        english = translate(
            source_text, analysis, anthropic_client, critique=critique,
            model=model, lang_name=lang_name, context=context, glossary=glossary,
        )
        refine_rounds += 1
        drift_notes = []  # only fed into the first refine round

        # Re-review; stop when the editor is satisfied.
        review_result = review(source_text, english, analysis, anthropic_client, model=model,
                               lang_name=lang_name, context=context)
        if not _needs_refine(review_result):
            break

    return FinalOutput(
        source_text=source_text,
        english_text=english,
        translator_notes=_make_notes(analysis, review_result, lang_name, refine_rounds, addressed_drift),
        analysis=analysis,
    )


def run_from_mic(
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
    context: str = "",
) -> FinalOutput:
    """Record one utterance from mic and run the full quality pipeline."""
    openai_client = OpenAI()
    wav_bytes = record_from_mic(lang_name=lang_name)

    print("Transcribing...", flush=True)
    source_text = transcribe(wav_bytes, openai_client, source_lang=source_lang)

    if not source_text.strip():
        print("No speech transcribed. Please try again.", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribed: {source_text}\n", flush=True)
    return run(source_text, model=model, source_lang=source_lang, lang_name=lang_name, context=context)


def run_continuous(
    max_seconds: int = 20,
    silence_ms: int = 600,
    threshold: float = 0.02,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
) -> None:
    """Pause-triggered mode: translate each utterance as it finishes (no threading)."""
    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()
    previous_transcript = ""
    chunk_num = 0

    for wav_bytes in stream_chunks(max_seconds=max_seconds, silence_ms=silence_ms,
                                   threshold=threshold, lang_name=lang_name):
        print("Transcribing...", flush=True)
        source_text = transcribe(wav_bytes, openai_client, prompt=previous_transcript, source_lang=source_lang)

        if not source_text.strip():
            print("(no speech in chunk, listening again)\n", flush=True)
            continue

        print(f"Transcribed: {source_text}\n", flush=True)
        previous_transcript = source_text
        chunk_num += 1
        result = run(source_text, model=model, source_lang=source_lang, lang_name=lang_name)

        print(f"\n-- Chunk {chunk_num} --")
        print(result.english_text)
        print()
