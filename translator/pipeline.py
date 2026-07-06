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


def _session_lines(context: str, glossary: str, participants: str) -> tuple[str, str, str]:
    """Format the session-constant prompt fragments shared by both cached prefixes."""
    context_line = f"\nSetting: {context}\n" if context else ""
    glossary_line = f"\n{glossary}\n" if glossary else ""
    roster = ", ".join(n.strip() for n in participants.splitlines() if n.strip())
    roster_line = f"\nParticipants: {roster}\n" if roster else ""
    return context_line, glossary_line, roster_line


def _conversation_system_blocks(
    lang_name: str, context: str = "", glossary: str = "",
    participants: str = "", diarized: bool = False,
) -> list[dict]:
    """Build the conversation system prompt as cacheable blocks.

    Everything here is constant for the session (role, rules, glossary, meeting
    context, participants), so it is sent as a `system` prefix with a
    cache_control breakpoint — billed ~0.1x on every chunk after the first. The
    volatile part (summary + rolling history + the new chunk) goes in the user
    message.
    """
    context_line, glossary_line, roster_line = _session_lines(context, glossary, participants)

    speaker_rule = (
        "- Each line is tagged with its speaker in [brackets]. Use the speaker to "
        "resolve dropped subjects and make clear who is speaking, asking, or being "
        "addressed — but do NOT print the [speaker] tag in your output\n"
        if diarized else ""
    )

    text = f"""\
You are a real-time {lang_name}-to-English conversation interpreter.
{context_line}{glossary_line}{roster_line}
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
{speaker_rule}- Genuine noise or silence → output "(inaudible)"
"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _label(text: str, speaker: str) -> str:
    return f"[{speaker}] {text}" if speaker else text


def _repair_system_blocks(
    lang_name: str, context: str = "", glossary: str = "",
    participants: str = "", diarized: bool = False,
) -> list[dict]:
    """Cached system prefix for the repair pass.

    Same session-constant material as the translator prefix (setting, glossary,
    roster) so the checker judges against the same facts, plus an OK-sentinel
    protocol: a clean translation gets a bare "OK" instead of being re-echoed —
    no false "repaired" flags from paraphrase, and fewer output tokens.
    """
    context_line, glossary_line, roster_line = _session_lines(context, glossary, participants)
    speaker_rule = (
        "- Lines are tagged with their speaker in [brackets]; use them to check that "
        "dropped subjects and referents were resolved correctly. Never print a tag\n"
        if diarized else ""
    )
    text = f"""\
You are a bilingual quality checker for a real-time {lang_name}-to-English interpreter.
{context_line}{glossary_line}{roster_line}
You receive recent conversation context, a new {lang_name} chunk, and a candidate English
translation of that chunk.

Rules:
- If the candidate is faithful and natural, reply with exactly: OK
- Otherwise output ONLY the corrected English translation — no labels, no commentary
- Only correct real problems: dropped meaning, a wrong referent/subject, a mishandled \
number/date/name, or genuinely awkward English. Do not rephrase acceptable translations
- Keep glossary renderings and earlier name spellings
{speaker_rule}"""
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _repair_translation(
    new_source: str,
    candidate: str,
    context_block: str,
    client: anthropic.Anthropic,
    model: str,
    lang_name: str,
    context: str = "",
    glossary: str = "",
    participants: str = "",
    speaker: str = "",
    diarized: bool = False,
) -> tuple[str, bool]:
    """One quick pass; returns (translation, repaired). "OK" keeps the candidate."""
    user_msg = (
        f"{context_block}"
        f"{lang_name} chunk:\n{_label(new_source, speaker)}\n\n"
        f"Candidate English:\n{candidate}"
    )
    with client.messages.stream(
        model=model,
        max_tokens=512,
        system=_repair_system_blocks(lang_name, context, glossary, participants, diarized),
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    reply = _text_of(msg)
    if not reply or reply.strip().rstrip(".").upper() == "OK":
        return candidate, False
    return reply, True


def _translate_with_context(
    new_source: str,
    history: list[str],
    client: anthropic.Anthropic,
    model: str = "claude-sonnet-4-6",
    lang_name: str = "Japanese",
    context: str = "",
    glossary: str = "",
    repair: bool = True,
    speaker: str = "",
    speakers: list[str] | None = None,
    participants: str = "",
    summary: str = "",
) -> ConvResult:
    """Context-aware conversation translation with a guarded self-repair pass.

    Quality is prioritized over latency here (the frontend's backpressure bounds
    lag to ~1 chunk): the first pass uses adaptive thinking, then a cheap repair
    pass corrects the translation only when it actually needs it. When speaker
    labels are available (diarization), history and the new chunk are tagged so
    the model can resolve dropped subjects. ``summary`` is a rolling digest of
    turns older than the verbatim window (long-range context).
    """
    speakers = speakers or []
    diarized = bool(speaker) or any(speakers)

    prev_texts = history[:-1]
    prev_speakers = speakers[:-1]
    lines = []
    for i, t in enumerate(prev_texts):
        sp = prev_speakers[i] if i < len(prev_speakers) else ""
        lines.append(f"[{i+1}] {_label(t, sp)}")
    previous = "\n".join(lines)
    summary_block = f"Meeting so far (summary):\n{summary}\n\n" if summary else ""
    if previous:
        context_block = f"{summary_block}Recent turns:\n{previous}\n\n"
    else:
        context_block = summary_block or "(start of conversation)\n\n"

    user_msg = f"{context_block}NEW chunk to translate:\n{_label(new_source, speaker)}"

    with client.messages.stream(
        model=model,
        max_tokens=2048,
        thinking={"type": "adaptive"},
        system=_conversation_system_blocks(lang_name, context, glossary, participants, diarized),
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()

    english = _text_of(msg)
    if not english:
        return ConvResult("", False)

    repaired = False
    if repair:
        english, repaired = _repair_translation(
            new_source, english, context_block, client, model, lang_name,
            context=context, glossary=glossary, participants=participants,
            speaker=speaker, diarized=diarized,
        )

    return ConvResult(english, repaired)


# ── rolling meeting summary ──────────────────────────────────────────────────

_SUMMARY_MODEL = "claude-haiku-4-5"

_SUMMARY_SYSTEM = """\
You maintain a running summary of a live meeting for a simultaneous interpreter.
You receive the current summary (may be empty) and new transcript lines that are
about to scroll out of the interpreter's verbatim context window.

Fold the new lines into the summary. Keep at most 10 concise bullet points covering:
topics discussed, decisions made, open questions, and who said or asked what
(speaker tags appear in [brackets] when known). Write the summary in English.
Output ONLY the updated bullet list — no preamble, no commentary.\
"""


def _summarize_history(
    older_lines: list[str],
    prev_summary: str,
    client: anthropic.Anthropic,
    model: str = _SUMMARY_MODEL,
) -> str:
    """Fold transcript lines leaving the verbatim window into a running summary.

    Cheap (haiku) and called off the hot path — a stale-by-a-few-chunks summary
    is fine. Returns the previous summary unchanged on empty output.
    """
    if not older_lines:
        return prev_summary
    prev_block = f"Current summary:\n{prev_summary}\n\n" if prev_summary else "Current summary: (empty)\n\n"
    user_msg = prev_block + "New lines to fold in:\n" + "\n".join(older_lines)
    with client.messages.stream(
        model=model,
        max_tokens=1024,
        system=_SUMMARY_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    ) as stream:
        msg = stream.get_final_message()
    return _text_of(msg) or prev_summary


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
