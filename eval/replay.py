"""Replay a real recording through the production translation pipeline.

Acts as an offline "browser": decodes a window of the audio file, segments it
with a Python port of the frontend VAD (same constants as frontend/app.js),
then runs each chunk through the exact functions server.py uses — transcribe →
diarize → context/summary-aware translate + self-repair — writing the server's
SAVE_CHUNKS_DIR format (NNNN.wav + session.jsonl) so judge.py/report.py work on
live-meeting captures too.

Interruption-safe: every chunk is on disk the moment it completes; billing
errors stop the run cleanly with a coverage summary and a resume command.

Usage:
  python3 -m eval.replay --audio meeting.mp3 --start 300 --minutes 5
  python3 -m eval.replay --resume eval/runs/20260708-141900
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic  # noqa: E402
from openai import OpenAI  # noqa: E402

from eval.common import (  # noqa: E402
    RunStopped, append_jsonl, call_with_budget_guard, load_jsonl,
    read_run_state, write_run_state,
)
from translator.diarizer import SpeakerBook, SpeakerEmbedder  # noqa: E402
from translator.glossary import Glossary  # noqa: E402
from translator.pipeline import _label, _summarize_history, _translate_with_context  # noqa: E402
from translator.transcriber import transcribe  # noqa: E402

SR = 16000

# Context windowing — mirrors server.py exactly.
MAX_HISTORY = 6
SUMMARY_EVERY = 8
MAX_VERBATIM = MAX_HISTORY + 2 * SUMMARY_EVERY
STT_PROMPT_CHARS = 400

# Frontend VAD constants, ported from frontend/app.js.
VAD_POLL_MS = 100
VAD_RMS_THRESHOLD = 0.015
VAD_SILENCE_MS = 700
VAD_MIN_SPEECH_MS = 300
VAD_MAX_MS = 14000

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}


# ── audio ────────────────────────────────────────────────────────────────────

def decode_window(audio_path: str, start_s: float, dur_s: float) -> np.ndarray:
    """Decode [start, start+dur] of any audio file to 16k mono float32 via ffmpeg."""
    cmd = [
        "ffmpeg", "-v", "error", "-ss", str(start_s), "-t", str(dur_s),
        "-i", audio_path, "-ac", "1", "-ar", str(SR),
        "-f", "s16le", "-acodec", "pcm_s16le", "-",
    ]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def vad_segment(samples: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Port of the browser VAD: cut on a sustained pause after speech, or at the
    hard cap. Windows that never contain speech are discarded (the browser never
    sends them). Returns [(offset_seconds, chunk_samples)]."""
    frame = SR * VAD_POLL_MS // 1000
    chunks: list[tuple[float, np.ndarray]] = []
    win_start = 0
    saw_speech = False
    speech_ms = 0
    silence_ms = 0
    pos = 0

    def cut(end: int) -> None:
        nonlocal win_start, saw_speech, speech_ms, silence_ms
        if saw_speech:
            chunks.append((win_start / SR, samples[win_start:end]))
        win_start = end
        saw_speech = False
        speech_ms = 0
        silence_ms = 0

    while pos + frame <= samples.shape[0]:
        rms = float(np.sqrt(np.mean(samples[pos:pos + frame] ** 2)))
        pos += frame
        if rms >= VAD_RMS_THRESHOLD:
            saw_speech = True
            speech_ms += VAD_POLL_MS
            silence_ms = 0
        else:
            silence_ms += VAD_POLL_MS
        pause_ended = saw_speech and speech_ms >= VAD_MIN_SPEECH_MS and silence_ms >= VAD_SILENCE_MS
        if pause_ended or (pos - win_start) >= SR * VAD_MAX_MS // 1000:
            cut(pos)
    cut(samples.shape[0])
    return chunks


def to_wav_bytes(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


# ── usage tracking (prompt-cache verification, no pipeline changes) ──────────

class TrackedClient:
    """Wraps an Anthropic client to record usage of every final message, so the
    replay can report cache_read/cache_creation tokens per chunk without
    touching translator/pipeline.py."""

    def __init__(self, client: anthropic.Anthropic):
        self._client = client
        self.usages: list = []
        self.messages = self  # pipeline calls client.messages.stream(...)

    def stream(self, **kwargs):
        outer = self

        class _Wrap:
            def __init__(self, cm):
                self._cm = cm

            def __enter__(self):
                s = self._cm.__enter__()
                orig = s.get_final_message

                def get_final_message():
                    msg = orig()
                    if getattr(msg, "usage", None) is not None:
                        outer.usages.append(msg.usage)
                    return msg

                s.get_final_message = get_final_message
                return s

            def __exit__(self, *a):
                return self._cm.__exit__(*a)

        return _Wrap(self._client.messages.stream(**kwargs))

    def drain_usage(self) -> tuple[int, int]:
        read = sum(getattr(u, "cache_read_input_tokens", 0) or 0 for u in self.usages)
        created = sum(getattr(u, "cache_creation_input_tokens", 0) or 0 for u in self.usages)
        self.usages = []
        return read, created


# ── replay ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Replay audio through the translation pipeline")
    p.add_argument("--audio", help="Path to the recording (any ffmpeg-readable format)")
    p.add_argument("--start", type=float, default=300, help="Window start, seconds (default 300)")
    p.add_argument("--minutes", type=float, default=5, help="Window length, minutes (default 5)")
    p.add_argument("--model", default="sonnet", help="Translator model alias or ID")
    p.add_argument("--context", default="Casual conversation between friends", help="Setting hint")
    p.add_argument("--glossary", default="", help="Path to a glossary file (term => rendering lines)")
    p.add_argument("--participants", default="", help="Comma-separated names in speaking order")
    p.add_argument("--resume", default="", help="Existing run dir to continue")
    args = p.parse_args()

    if args.resume:
        run_dir = args.resume.rstrip("/")
        cfg = read_run_state(run_dir).get("config")
        if not cfg:
            sys.exit(f"No run_state.json with config in {run_dir} — cannot resume.")
        print(f"Resuming {run_dir}")
    else:
        if not args.audio:
            sys.exit("--audio is required for a new run")
        run_dir = os.path.join("eval", "runs", time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        cfg = {
            "audio": args.audio, "start": args.start, "minutes": args.minutes,
            "model": MODEL_ALIASES.get(args.model.lower(), args.model),
            "context": args.context, "glossary": args.glossary,
            "participants": args.participants,
        }
        write_run_state(run_dir, config=cfg, stage="replay")

    session_path = os.path.join(run_dir, "session.jsonl")
    done = {r["seq"] for r in load_jsonl(session_path)}
    prior = load_jsonl(session_path)

    # Decode + segment (deterministic → resume sees identical chunks).
    print(f"Decoding {cfg['audio']} [{cfg['start']:.0f}s + {cfg['minutes']:.1f}min]…")
    samples = decode_window(cfg["audio"], cfg["start"], cfg["minutes"] * 60)
    chunks = vad_segment(samples)
    durs = [len(s) / SR for _, s in chunks]
    print(f"{len(chunks)} chunks (mean {np.mean(durs):.1f}s, max {np.max(durs):.1f}s)"
          + (f" — {len(done)} already done, resuming" if done else ""))

    glossary_raw = ""
    if cfg["glossary"] and os.path.exists(cfg["glossary"]):
        glossary_raw = Path(cfg["glossary"]).read_text()
    glossary = Glossary.parse(glossary_raw)
    glossary_stt = glossary.format_for_stt()
    glossary_prompt = glossary.format_for_prompt()
    participants = "\n".join(n.strip() for n in cfg["participants"].split(",") if n.strip())
    names = [n for n in participants.splitlines() if n]

    openai_client = OpenAI()
    client = TrackedClient(anthropic.Anthropic())
    embedder = SpeakerEmbedder()
    book = SpeakerBook(names=names)

    # Rebuild sequential state from prior records (resume): turns, summary
    # watermark, and speaker centroids (re-embedding saved WAVs is free/local).
    turns: list[tuple[str, str]] = [(r["source"], r.get("speaker", "")) for r in prior if "source" in r]
    state = read_run_state(run_dir)
    summary = state.get("summary", "")
    summary_upto = state.get("summary_upto", 0)
    for r in prior:
        wav_path = os.path.join(run_dir, f"{r['seq']:04d}.wav")
        if r.get("speaker") and os.path.exists(wav_path):
            emb = embedder.embed(Path(wav_path).read_bytes())
            if emb is not None:
                book.assign(emb)

    completed = len(done)
    stopped_reason = ""

    # Sentence assembly — same behavior as server.py: mid-sentence chunks are
    # buffered and joined; offline, the 6s "speaker went silent" flush becomes
    # "the next chunk starts more than 6s after this one ends".
    assembler = ChunkAssembler()

    for i, (offset, chunk) in enumerate(chunks):
        seq = i + 1
        if seq in done:
            continue
        wav_bytes = to_wav_bytes(chunk)
        dur_s = len(chunk) / SR
        t0 = time.perf_counter()
        try:
            def process():
                prompt = " ".join(t for t, _ in turns[-3:])[-STT_PROMPT_CHARS:]
                text = transcribe(wav_bytes, openai_client, prompt=prompt,
                                  source_lang="ja", model="gpt-4o-transcribe",
                                  glossary=glossary_stt)
                if not text.strip():
                    return None

                assembled = assembler.add(text, dur_s)
                if assembled is None:
                    end_t = offset + dur_s
                    next_off = chunks[i + 1][0] if i + 1 < len(chunks) else None
                    if next_off is not None and next_off - end_t <= 6.0:
                        return "buffered"
                    assembled = assembler.flush()

                speaker, sim = "", -1.0
                emb = embedder.embed(wav_bytes)
                if emb is not None:
                    _, speaker, sim = book.assign(emb)

                turns.append((assembled, speaker))
                start_idx = max(summary_upto, len(turns) - MAX_VERBATIM)
                start_idx = min(start_idx, max(0, len(turns) - MAX_HISTORY))
                window = turns[start_idx:]
                english, repaired = _translate_with_context(
                    assembled, [t for t, _ in window], client,
                    model=cfg["model"], lang_name="Japanese", context=cfg["context"],
                    glossary=glossary_prompt, speaker=speaker,
                    speakers=[sp for _, sp in window], participants=participants,
                    summary=summary, diarized=True,
                )
                return assembled, speaker, sim, english, repaired

            result = call_with_budget_guard(process)
        except RunStopped as stop:
            stopped_reason = stop.reason
            break

        ms = int((time.perf_counter() - t0) * 1000)
        cache_read, cache_created = client.drain_usage()

        if result == "buffered":
            record = {"seq": seq, "ts": time.time(), "buffered": True,
                      "offset_s": round(offset, 1), "dur_s": round(dur_s, 1)}
        elif result is None:
            record = {"seq": seq, "ts": time.time(), "skipped": True,
                      "offset_s": round(offset, 1), "dur_s": round(len(chunk) / SR, 1)}
        else:
            text, speaker, sim, english, repaired = result
            record = {
                "seq": seq, "ts": time.time(), "speaker": speaker, "sim": round(sim, 3),
                "source": text, "english": english or "", "repaired": repaired, "ms": ms,
                "merged": assembler.last_merged,
                "cache_read": cache_read, "cache_created": cache_created,
                "offset_s": round(offset, 1), "dur_s": round(len(chunk) / SR, 1),
            }
            print(f"#{seq}/{len(chunks)} [{offset:6.1f}s] {speaker or '-':10s} sim={sim:5.2f} "
                  f"merged={assembler.last_merged} {ms}ms  {text[:30]!r} → {(english or '')[:30]!r}", flush=True)

        with open(os.path.join(run_dir, f"{seq:04d}.wav"), "wb") as f:
            f.write(wav_bytes)
        append_jsonl(session_path, record)
        completed += 1

        # Rolling summary fold — synchronous offline (haiku call, budget-guarded).
        if len(turns) % SUMMARY_EVERY == 0 and len(turns) > MAX_HISTORY:
            try:
                def fold():
                    end = len(turns) - MAX_HISTORY
                    lines = [_label(t, sp) for t, sp in turns[summary_upto:end]]
                    return _summarize_history(lines, summary, client), end

                summary, summary_upto = call_with_budget_guard(fold)
                write_run_state(run_dir, summary=summary, summary_upto=summary_upto)
            except RunStopped as stop:
                stopped_reason = stop.reason
                break

    # ── coverage summary (always printed, success or stop) ──────────────────
    write_run_state(
        run_dir, stage="replay",
        completed_chunks=completed, total_chunks=len(chunks),
        stopped_reason=stopped_reason or "completed",
    )
    print("\n" + "─" * 60)
    if stopped_reason:
        print(f"STOPPED EARLY: {stopped_reason}")
        print(f"Completed {completed}/{len(chunks)} chunks — everything completed is saved.")
        print(f"Resume with:\n  python3 -m eval.replay --resume {run_dir}")
    else:
        print(f"Done: {completed}/{len(chunks)} chunks → {session_path}")
        print(f"Next:\n  python3 -m eval.judge {run_dir}")


if __name__ == "__main__":
    main()
