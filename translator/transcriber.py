import io
import re

from openai import OpenAI

# whisper-1 verbose_json per-segment confidence thresholds (fallback path only).
_NO_SPEECH_THRESHOLD = 0.6
_AVG_LOGPROB_THRESHOLD = -1.0
_COMPRESSION_RATIO_THRESHOLD = 2.4

# Default STT model. gpt-4o-transcribe is markedly more accurate than whisper-1
# and hallucinates far less, at the cost of not exposing per-segment metrics.
DEFAULT_STT_MODEL = "gpt-4o-transcribe"

# Phrases the models emit from silence/music/outro noise rather than real speech.
# A transcript that is *only* one of these is dropped. Kept short and exact.
_HALLUCINATION_PHRASES = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "最後までご視聴いただきありがとうございます",
    "チャンネル登録をお願いします",
    "おやすみなさい",
    "ありがとうございました",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
}


def _build_prompt(prompt: str, glossary: str) -> str:
    """Combine the rolling transcript hint with glossary terms for STT biasing."""
    parts = [p for p in (prompt.strip(), glossary.strip()) if p]
    return "  ".join(parts)


def _collapse_repetition(text: str) -> str:
    """Collapse pathological verbatim repetition (a classic STT hallucination).

    Whisper/gpt-4o can loop a phrase dozens of times on noise. Reduce any run of
    the same token/phrase repeated 3+ times back to a single occurrence.
    """
    # Immediate word-level repeats: "はい はい はい ..." → "はい"
    text = re.sub(r"(\b\S+\b)(?:\s+\1){2,}", r"\1", text)
    # Repeated short clauses separated by punctuation.
    text = re.sub(r"(.{2,40}?)(?:\1){2,}", r"\1", text)
    return text.strip()


def _clean(text: str) -> str:
    """Text-level hallucination guard used for models without segment metrics."""
    text = _collapse_repetition(text.strip())
    if not text:
        return ""
    normalized = re.sub(r"[\s。、,.!！?？]", "", text).lower()
    if normalized in {re.sub(r"[\s。、,.!！?？]", "", p).lower() for p in _HALLUCINATION_PHRASES}:
        return ""
    return text


def transcribe(
    wav_bytes: bytes,
    client: OpenAI,
    prompt: str = "",
    source_lang: str = "ja",
    filename: str = "audio.wav",
    model: str = DEFAULT_STT_MODEL,
    glossary: str = "",
) -> str:
    """Transcribe WAV bytes, filtering hallucinated output.

    Two paths:
      - gpt-4o-transcribe / gpt-4o-mini-transcribe (default): response_format="text"
        with a text-level hallucination guard (repetition collapse + artifact
        denylist). These models expose no per-segment metrics.
      - whisper-1: verbose_json with per-segment confidence filtering (legacy
        fallback, retained so nothing is lost if the newer model misbehaves).

    ``prompt`` (previous transcript) and ``glossary`` (session terms) bias
    recognition of names and jargon. Returns "" if nothing real was detected.
    """
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = filename
    combined_prompt = _build_prompt(prompt, glossary)

    if model == "whisper-1":
        kwargs = dict(
            model="whisper-1",
            file=audio_file,
            language=source_lang,
            response_format="verbose_json",
        )
        if combined_prompt:
            kwargs["prompt"] = combined_prompt
        result = client.audio.transcriptions.create(**kwargs)
        good_segments = []
        for seg in result.segments:
            if seg.no_speech_prob > _NO_SPEECH_THRESHOLD:
                continue
            if seg.avg_logprob < _AVG_LOGPROB_THRESHOLD:
                continue
            if seg.compression_ratio > _COMPRESSION_RATIO_THRESHOLD:
                continue
            good_segments.append(seg.text.strip())
        return _clean(" ".join(good_segments))

    # gpt-4o-transcribe family: text response + text-level guard.
    kwargs = dict(
        model=model,
        file=audio_file,
        language=source_lang,
        response_format="text",
    )
    if combined_prompt:
        kwargs["prompt"] = combined_prompt
    result = client.audio.transcriptions.create(**kwargs)
    # response_format="text" returns a plain string.
    text = result if isinstance(result, str) else getattr(result, "text", "")
    return _clean(text)
