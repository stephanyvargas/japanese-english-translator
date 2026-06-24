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

_CONVERSATION_SYSTEM = """\
You are a real-time Japanese conversation interpreter.

You receive the conversation transcript so far (in Japanese) plus a new chunk to translate.
Use the full history to resolve pronouns, implicit subjects, topic references, and incomplete
sentences — context that would be impossible to translate in isolation.

Rules:
- Output ONLY the English translation of the NEW chunk — nothing else, no labels, no commentary
- Short utterances (うん, はいはい, あのー, えっと) → natural English equivalents: \
"Yeah.", "Right, right.", "So, um...", "Uh..."
- Incomplete sentences → translate as-is, let the incompleteness show in English
- Names and untranslatable loanwords → keep or romanize
- Genuine noise or silence → output "(inaudible)"
"""


def _translate_with_context(
    new_japanese: str,
    history: list[str],
    client: anthropic.Anthropic,
) -> str:
    """Single-pass context-aware translation for conversation mode."""
    previous = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(history[:-1]))
    context_block = f"Conversation so far:\n{previous}\n\n" if previous else "(start of conversation)\n\n"

    user_msg = f"{context_block}NEW chunk to translate:\n{new_japanese}"

    with client.messages.stream(
        model="claude-opus-4-8",
        max_tokens=512,
        thinking={"type": "adaptive"},
        system=_CONVERSATION_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()

    return next(b.text for b in msg.content if b.type == "text").strip()


def run_conversation(
    interval_seconds: int = 8,
    min_speech_ratio: float = 0.15,
    threshold: float = 0.02,
) -> None:
    """Continuous conversation translation with threaded audio capture.

    The mic runs on a background thread the entire time, so nothing is lost
    while Claude is translating. Every interval_seconds, accumulated audio is
    transcribed and translated in full conversational context.
    """
    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()

    capture = AudioCapture()
    capture.start()

    japanese_history: list[str] = []
    chunk_num = 0

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
            prompt = japanese_history[-1] if japanese_history else ""

            print(f"[{duration:.1f}s audio → transcribing...]", flush=True)
            new_text = transcribe(wav_bytes, openai_client, prompt=prompt)

            if not new_text.strip():
                print("[silence/noise — skipped]\n", flush=True)
                continue

            japanese_history.append(new_text)
            chunk_num += 1

            print(f"[JP] {new_text}", flush=True)
            english = _translate_with_context(new_text, japanese_history, anthropic_client)

            print(f"[EN] {english}\n", flush=True)

    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        capture.stop()


# ── Single-shot and legacy modes ──────────────────────────────────────────────

def _build_critique(rev) -> str:
    parts = [f"Accuracy: {rev.accuracy_score}/10, Naturalness: {rev.naturalness_score}/10"]
    if rev.issues:
        parts.append("Issues:\n" + "\n".join(f"  - {i}" for i in rev.issues))
    if rev.suggestions:
        parts.append("Suggestions:\n" + "\n".join(f"  - {s}" for s in rev.suggestions))
    return "\n".join(parts)


def _make_notes(analysis, review_result, refined: bool) -> list[str]:
    notes: list[str] = []
    if analysis.has_keigo:
        notes.append(f"Keigo detected ({analysis.formality_level}): honorific level reflected in register.")
    for note in analysis.cultural_notes:
        notes.append(f"Cultural: {note}")
    if analysis.implicit_subjects:
        notes.append("Implicit subjects inferred: " + ", ".join(analysis.implicit_subjects))
    if refined:
        notes.append("Translation was refined after editorial review.")
    if review_result.accuracy_score < 9:
        notes.append(f"Editor accuracy score: {review_result.accuracy_score}/10")
    return notes


def run(japanese_text: str) -> FinalOutput:
    """Full quality pipeline: analyze → translate → review → (refine if needed)."""
    anthropic_client = anthropic.Anthropic()

    print("[1/3] Analyzing text...", flush=True)
    analysis = analyze(japanese_text, anthropic_client)

    print("[2/3] Translating...", flush=True)
    english = translate(japanese_text, analysis, anthropic_client)

    print("[3/3] Reviewing quality...", flush=True)
    review_result = review(japanese_text, english, analysis, anthropic_client)

    refined = False
    if review_result.issues and review_result.accuracy_score < 8:
        print("[+] Refining based on editorial critique...", flush=True)
        english = translate(japanese_text, analysis, anthropic_client,
                            critique=_build_critique(review_result))
        refined = True

    return FinalOutput(
        japanese_source=japanese_text,
        english_text=english,
        translator_notes=_make_notes(analysis, review_result, refined),
        analysis=analysis,
    )


def run_from_mic() -> FinalOutput:
    """Record one utterance from mic and run the full quality pipeline."""
    openai_client = OpenAI()
    wav_bytes = record_from_mic()

    print("Transcribing...", flush=True)
    japanese_text = transcribe(wav_bytes, openai_client)

    if not japanese_text.strip():
        print("No speech transcribed. Please try again.", file=sys.stderr)
        sys.exit(1)

    print(f"Transcribed: {japanese_text}\n", flush=True)
    return run(japanese_text)


def run_continuous(max_seconds: int = 20, silence_ms: int = 600, threshold: float = 0.02) -> None:
    """Pause-triggered mode: translate each utterance as it finishes (no threading)."""
    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()
    previous_transcript = ""
    chunk_num = 0

    for wav_bytes in stream_chunks(max_seconds=max_seconds, silence_ms=silence_ms, threshold=threshold):
        print("Transcribing...", flush=True)
        japanese_text = transcribe(wav_bytes, openai_client, prompt=previous_transcript)

        if not japanese_text.strip():
            print("(no speech in chunk, listening again)\n", flush=True)
            continue

        print(f"Transcribed: {japanese_text}\n", flush=True)
        previous_transcript = japanese_text
        chunk_num += 1
        result = run(japanese_text)

        print(f"\n── Chunk {chunk_num} ──")
        print(result.english_text)
        print()
