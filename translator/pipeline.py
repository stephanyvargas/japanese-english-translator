import sys
import time

import anthropic
from openai import OpenAI

from .analyzer import analyze
from .audio import AudioCapture, record_from_mic, stream_chunks
from .models import FinalOutput
from .reviewer import review
from .transcriber import transcribe
from .translator import translate


def _get_conversation_system(lang_name: str) -> str:
    return f"""\
You are a real-time {lang_name}-to-English conversation interpreter.

You receive the conversation transcript so far (in {lang_name}) plus a new chunk to translate.
Use the full history to resolve pronouns, implicit subjects, topic references, and incomplete
sentences — context that would be impossible to translate in isolation.

Rules:
- Output ONLY the English translation of the NEW chunk — nothing else, no labels, no commentary
- Short utterances and conversational fillers → natural English equivalents \
("Yeah.", "Right, right.", "So, um...", "Uh...")
- Incomplete sentences → translate as-is, let the incompleteness show in English
- Names and untranslatable loanwords → keep or romanize
- Genuine noise or silence → output "(inaudible)"
"""


def _translate_with_context(
    new_source: str,
    history: list[str],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
) -> str:
    """Single-pass context-aware translation for conversation mode."""
    previous = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(history[:-1]))
    context_block = f"Conversation so far:\n{previous}\n\n" if previous else "(start of conversation)\n\n"

    user_msg = f"{context_block}NEW chunk to translate:\n{new_source}"

    with client.messages.stream(
        model=model,
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=_get_conversation_system(lang_name),
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()

    return next((b.text for b in msg.content if b.type == "text"), "").strip()


def run_conversation(
    interval_seconds: int = 8,
    min_speech_ratio: float = 0.15,
    threshold: float = 0.02,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
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
            english = _translate_with_context(
                new_text, source_history, anthropic_client, model=model, lang_name=lang_name
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


def _make_notes(analysis, review_result, lang_name: str, refined: bool) -> list[str]:
    notes: list[str] = []
    if analysis.has_honorifics:
        notes.append(f"Honorific register detected ({analysis.formality_level}): conveyed through English register.")
    for note in analysis.cultural_notes:
        notes.append(f"Cultural: {note}")
    if analysis.implicit_subjects:
        notes.append("Implicit subjects inferred: " + ", ".join(analysis.implicit_subjects))
    if refined:
        notes.append("Translation was refined after editorial review.")
    if review_result.accuracy_score < 9:
        notes.append(f"Editor accuracy score: {review_result.accuracy_score}/10")
    return notes


def run(
    source_text: str,
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
) -> FinalOutput:
    """Full quality pipeline: analyze → translate → review → (refine if needed)."""
    anthropic_client = anthropic.Anthropic()

    print("[1/3] Analyzing text...", flush=True)
    analysis = analyze(source_text, anthropic_client, model=model, source_lang=source_lang, lang_name=lang_name)

    print("[2/3] Translating...", flush=True)
    english = translate(source_text, analysis, anthropic_client, model=model, lang_name=lang_name)

    print("[3/3] Reviewing quality...", flush=True)
    review_result = review(source_text, english, analysis, anthropic_client, model=model, lang_name=lang_name)

    refined = False
    if review_result.issues and review_result.accuracy_score < 8:
        print("[+] Refining based on editorial critique...", flush=True)
        english = translate(
            source_text, analysis, anthropic_client,
            critique=_build_critique(review_result), model=model, lang_name=lang_name,
        )
        refined = True

    return FinalOutput(
        source_text=source_text,
        english_text=english,
        translator_notes=_make_notes(analysis, review_result, lang_name, refined),
        analysis=analysis,
    )


def run_from_mic(
    model: str = "claude-sonnet-4-6",
    source_lang: str = "ja",
    lang_name: str = "Japanese",
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
    return run(source_text, model=model, source_lang=source_lang, lang_name=lang_name)


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
