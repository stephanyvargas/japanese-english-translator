import io

from openai import OpenAI

_NO_SPEECH_THRESHOLD = 0.6
_AVG_LOGPROB_THRESHOLD = -1.0
_COMPRESSION_RATIO_THRESHOLD = 2.4


def transcribe(wav_bytes: bytes, client: OpenAI, prompt: str = "", source_lang: str = "ja", filename: str = "audio.wav") -> str:
    """Transcribe WAV bytes via Whisper, filtering hallucinated segments.

    Uses verbose_json to inspect per-segment confidence metrics and discard
    segments that Whisper generated from silence, music, or noise.
    Returns an empty string if nothing real was detected.
    """
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = filename

    kwargs = dict(
        model="whisper-1",
        file=audio_file,
        language=source_lang,
        response_format="verbose_json",
    )
    if prompt:
        kwargs["prompt"] = prompt

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

    return " ".join(good_segments)
