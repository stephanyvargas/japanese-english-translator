"""FastAPI backend for the translator web UI.

Exposes two surfaces:
  POST /translate          — full quality pipeline (text input)
  WS   /ws/conversation   — real-time conversation mode (audio from browser)
  GET  /health             — Cloud Run health probe
"""

import asyncio
import io
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import anthropic
import av
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("translator")

from translator.assembler import ChunkAssembler  # noqa: E402
from translator.diarizer import SpeakerBook, SpeakerEmbedder  # noqa: E402
from translator.glossary import Glossary  # noqa: E402
from translator.pipeline import _label, _summarize_history, _translate_with_context, run  # noqa: E402
from translator.transcriber import DEFAULT_STT_MODEL, transcribe  # noqa: E402

# Shared across connections — model/state is loaded once, embedding is stateless.
_embedder = SpeakerEmbedder()

app = FastAPI(title="Translator API")

# Comma-separated origins via ALLOWED_ORIGINS env; falls back to "*" for local dev.
# In production set e.g. ALLOWED_ORIGINS="https://PROJECT.web.app,https://PROJECT.firebaseapp.com"
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
_allow_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)

# Rolling-summary updates run on their own single worker so a blocking Haiku
# call can never occupy a hot-path worker and stall chunk processing.
_summary_executor = ThreadPoolExecutor(max_workers=1)

LANGUAGE_NAMES = {
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "es": "Spanish", "fr": "French", "de": "German",
    "pt": "Portuguese", "it": "Italian", "ru": "Russian", "ar": "Arabic",
}

MODEL_ALIASES = {
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-8",
    "haiku": "claude-haiku-4-5",
}

# At least the last MAX_HISTORY turns are sent verbatim to Claude per chunk —
# keeps latency flat as the conversation grows. Older turns are folded into a
# rolling summary (updated off the hot path every SUMMARY_EVERY chunks). The
# verbatim window always extends back to the summary's fold watermark so no
# turn is ever in neither the summary nor the verbatim block; MAX_VERBATIM
# caps it in case the summarizer stalls (e.g. persistent API errors).
MAX_HISTORY = 6
SUMMARY_EVERY = 8
MAX_VERBATIM = MAX_HISTORY + 2 * SUMMARY_EVERY

# STT continuity prompt: tail of the recent transcript (the STT prompt token
# budget is small; the tail is what biases continuation of names/topics).
STT_PROMPT_CHARS = 400

# Opt-in per-chunk recorder (local dev only — Cloud Run has no persistent disk).
# When set, each session dumps its 16k mono WAV chunks + a JSONL transcript,
# which doubles as eval-harness input collected from real meetings.
SAVE_CHUNKS_DIR = os.environ.get("SAVE_CHUNKS_DIR", "").strip()

# When REQUIRE_AUTH=1 (production), every WS connection and /translate request
# must carry a valid Firebase ID token — protects the public Cloud Run URL from
# anonymous use of the API keys. Off by default for local dev.
REQUIRE_AUTH = os.environ.get("REQUIRE_AUTH", "") == "1"
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "japanese-translator-501010")


def _verify_token(token: str) -> str | None:
    """Return the Firebase uid for a valid ID token, else None.

    Verified once per WS connection / REST request (not per chunk). Uses
    google-auth's Firebase verifier — no firebase-admin dependency.
    """
    if not token:
        return None
    try:
        import google.auth.transport.requests
        from google.oauth2 import id_token as google_id_token
        claims = google_id_token.verify_firebase_token(
            token, google.auth.transport.requests.Request(), audience=FIREBASE_PROJECT_ID)
        return claims.get("user_id") or claims.get("sub")
    except Exception as exc:
        log.info("token verification failed: %s", exc)
        return None


def _to_wav(audio_bytes: bytes) -> bytes:
    """Convert any browser audio (WebM/Ogg/MP4) to 16kHz mono s16 WAV via PyAV."""
    in_buf = io.BytesIO(audio_bytes)
    out_buf = io.BytesIO()
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
    with av.open(in_buf) as in_c:
        in_stream = in_c.streams.audio[0]
        with av.open(out_buf, "w", format="wav") as out_c:
            out_stream = out_c.add_stream("pcm_s16le", rate=16000, layout="mono")
            for frame in in_c.decode(in_stream):
                for rf in resampler.resample(frame):
                    for pkt in out_stream.encode(rf):
                        out_c.mux(pkt)
            for rf in resampler.resample(None):
                for pkt in out_stream.encode(rf):
                    out_c.mux(pkt)
            for pkt in out_stream.encode(None):
                out_c.mux(pkt)
    out_buf.seek(0)
    return out_buf.read()


# ── REST ─────────────────────────────────────────────────────────────────────

class TranslateRequest(BaseModel):
    text: str
    model: str = "sonnet"
    source_lang: str = "ja"
    context: str = ""
    glossary: str = ""      # free-text "term => rendering" lines, one per line
    verify: bool = False    # opt-in back-translation drift check


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/translate")
async def translate_text(req: TranslateRequest, authorization: str = Header(default="")):
    if REQUIRE_AUTH:
        token = authorization.removeprefix("Bearer ").strip()
        if not _verify_token(token):
            raise HTTPException(status_code=401, detail="Sign in to translate.")
    model = MODEL_ALIASES.get(req.model.lower(), req.model)
    lang_name = LANGUAGE_NAMES.get(req.source_lang.lower(), req.source_lang.upper())
    glossary = Glossary.parse(req.glossary).format_for_prompt()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: run(req.text.strip(), model=model, source_lang=req.source_lang,
                    lang_name=lang_name, context=req.context,
                    glossary=glossary, verify=req.verify),
    )
    return {
        "source_text": result.source_text,
        "english_text": result.english_text,
        "translator_notes": result.translator_notes,
        "analysis": {
            "domain": result.analysis.domain,
            "formality_level": result.analysis.formality_level,
            "has_honorifics": result.analysis.has_honorifics,
            "cultural_notes": result.analysis.cultural_notes,
            "implicit_subjects": result.analysis.implicit_subjects,
        },
    }


# ── WebSocket — conversation mode ────────────────────────────────────────────

@app.websocket("/ws/conversation")
async def ws_conversation(ws: WebSocket):
    await ws.accept()

    # First frame must be a JSON config text frame
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        cfg = json.loads(raw)
    except Exception:
        await ws.close(code=1008)
        return

    uid = None
    if REQUIRE_AUTH:
        loop_ = asyncio.get_event_loop()
        uid = await loop_.run_in_executor(
            _executor, _verify_token, cfg.get("id_token", ""))
        if not uid:
            await ws.close(code=4401)
            return

    model = MODEL_ALIASES.get(cfg.get("model", "sonnet").lower(), cfg.get("model", "sonnet"))
    source_lang = cfg.get("source_lang", "ja")
    lang_name = LANGUAGE_NAMES.get(source_lang.lower(), source_lang.upper())
    context = cfg.get("context", "")
    stt_model = cfg.get("stt_model", DEFAULT_STT_MODEL)

    glossary = Glossary.parse(cfg.get("glossary", ""))
    glossary_stt = glossary.format_for_stt()
    glossary_prompt = glossary.format_for_prompt()

    participants = cfg.get("participants", "")
    diarize = bool(cfg.get("diarize", True))
    names = [n.strip() for n in participants.splitlines() if n.strip()]
    speaker_book = SpeakerBook(names=names)

    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()
    # One (text, speaker) tuple per processed chunk — a single list so the
    # transcript and its speaker labels can never fall out of alignment.
    turns: list[tuple[str, str]] = []
    loop = asyncio.get_event_loop()
    seq = 0

    # Rolling summary of turns older than the verbatim window (long-range
    # context). Updated off the hot path; "upto" marks how far it has folded.
    summary_state = {"summary": "", "upto": 0, "busy": False}

    def update_summary() -> None:
        try:
            end = len(turns) - MAX_HISTORY
            if end <= summary_state["upto"]:
                return
            lines = [_label(t, sp) for t, sp in turns[summary_state["upto"]:end]]
            summary_state["summary"] = _summarize_history(
                lines, summary_state["summary"], anthropic_client)
            summary_state["upto"] = end
            log.info("summary updated (folded %d turns): %r",
                     len(lines), summary_state["summary"][:80])
        except Exception as exc:
            log.info("summary update failed (%s)", exc)
        finally:
            summary_state["busy"] = False

    # Opt-in chunk recorder (see SAVE_CHUNKS_DIR above).
    session_dir = ""
    if SAVE_CHUNKS_DIR:
        session_dir = os.path.join(SAVE_CHUNKS_DIR, time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(session_dir, exist_ok=True)
        log.info("recording chunks to %s", session_dir)

    def save_chunk(n: int, wav_bytes: bytes, record: dict) -> None:
        try:
            with open(os.path.join(session_dir, f"{n:04d}.wav"), "wb") as f:
                f.write(wav_bytes)
            with open(os.path.join(session_dir, "session.jsonl"), "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            log.info("#%d chunk save failed (%s)", n, exc)

    log.info("WS connected | uid=%s model=%s stt=%s lang=%s context=%r terms=%d diarize=%s participants=%d",
             uid or "-", model, stt_model, source_lang, context, len(glossary), diarize, len(names))

    # Sentence assembly: mid-sentence chunks are buffered and joined with what
    # follows, so fragments aren't translated in isolation. pending_wav keeps
    # the newest buffered chunk's audio for diarization at flush time.
    assembler = ChunkAssembler()
    pending_wav = {"wav": b""}

    def translate_turn(n: int, text: str, wav_bytes: bytes, t0: float,
                       stt_ms: int, merged: int) -> dict:
        """Diarize + translate an assembled sentence; log and record it."""
        speaker = ""
        sim = -1.0
        if diarize:
            try:
                emb = _embedder.embed(wav_bytes)
                if emb is not None:
                    _, speaker, sim = speaker_book.assign(emb)
            except Exception as exc:
                log.info("#%d diarize skipped (%s)", n, exc)

        turns.append((text, speaker))
        # Verbatim window: everything the summary hasn't folded yet (so
        # summary + verbatim always cover the whole conversation), at
        # least MAX_HISTORY turns, capped at MAX_VERBATIM.
        start = max(summary_state["upto"], len(turns) - MAX_VERBATIM)
        start = min(start, max(0, len(turns) - MAX_HISTORY))
        window = turns[start:]
        english, repaired = _translate_with_context(
            text, [t for t, _ in window], anthropic_client,
            model=model, lang_name=lang_name, context=context, glossary=glossary_prompt,
            speaker=speaker, speakers=[sp for _, sp in window], participants=participants,
            summary=summary_state["summary"], diarized=diarize,
        )
        t3 = time.perf_counter()

        total_ms = int((t3 - t0) * 1000)
        log.info(
            "#%d stt %dms %r | merged=%d | %s sim=%.2f | mt %dms%s %r | total %dms",
            n, stt_ms, text[:40], merged, speaker or "-", sim,
            (t3 - t0) * 1000 - stt_ms, " repaired" if repaired else "",
            (english or "")[:40], total_ms,
        )
        if session_dir:
            save_chunk(n, wav_bytes, {
                "seq": n, "ts": time.time(), "speaker": speaker,
                "sim": round(sim, 3), "source": text, "english": english or "",
                "repaired": repaired, "ms": total_ms, "merged": merged,
            })
        return {"source": text, "english": english or "", "speaker": speaker, "ms": total_ms}

    def flush_pending(n: int) -> dict:
        """Speaker went silent mid-buffer — translate what's pending as-is."""
        text = assembler.flush()
        if not text:
            return {"skipped": True}
        return translate_turn(n, text, pending_wav["wav"], time.perf_counter(),
                              0, assembler.last_merged)

    try:
        while True:
            # Receive binary audio frame (WAV bytes from browser MediaRecorder).
            # While a partial sentence is buffered, wait at most 6s — a silent
            # speaker means the sentence won't be continued, so flush it.
            if assembler.pending:
                try:
                    data = await asyncio.wait_for(ws.receive_bytes(), timeout=6)
                except asyncio.TimeoutError:
                    seq += 1
                    result = await loop.run_in_executor(_executor, flush_pending, seq)
                    if not result.get("skipped"):
                        await ws.send_text(json.dumps({
                            "source": result["source"],
                            "english": result["english"],
                            "speaker": result.get("speaker", ""),
                            "lang_tag": source_lang.upper(),
                            "seq": seq,
                            "ts": time.time(),
                            "ms": result["ms"],
                        }))
                    continue
            else:
                data = await ws.receive_bytes()
            seq += 1
            n = seq

            # Run transcription + translation in thread pool (blocking SDK calls)
            def process(raw_bytes: bytes) -> dict:
                t0 = time.perf_counter()
                if len(raw_bytes) < 1000:
                    log.info("#%d skipped (%d bytes, too short)", n, len(raw_bytes))
                    return {"skipped": True}
                try:
                    wav_bytes = _to_wav(raw_bytes)
                except Exception as exc:
                    log.info("#%d skipped (decode failed: %s)", n, exc)
                    return {"skipped": True}
                t1 = time.perf_counter()

                # Continuity prompt: tail of the last few turns, not just one.
                prompt = " ".join(t for t, _ in turns[-3:])[-STT_PROMPT_CHARS:]
                text = transcribe(wav_bytes, openai_client, prompt=prompt,
                                  source_lang=source_lang, model=stt_model, glossary=glossary_stt)
                t2 = time.perf_counter()
                stt_ms = int((t2 - t1) * 1000)
                if not text.strip():
                    log.info("#%d skipped (no speech) | wav %dms | stt %dms",
                             n, (t1 - t0) * 1000, stt_ms)
                    return {"skipped": True}

                # Sentence assembly: hold visibly mid-sentence chunks and join
                # them with what follows (or the 6s silence flush above).
                dur_s = max(0.0, (len(wav_bytes) - 44) / 2 / 16000)
                pending_wav["wav"] = wav_bytes
                assembled = assembler.add(text, dur_s)
                if assembled is None:
                    log.info("#%d buffered (mid-sentence) %r", n, text[:40])
                    return {"buffered": True}

                return translate_turn(n, assembled, wav_bytes, t0, stt_ms,
                                      assembler.last_merged)

            result = await loop.run_in_executor(_executor, process, data)

            if result.get("buffered"):
                await ws.send_text(json.dumps({"buffered": True, "skipped": True, "seq": n}))
            elif result.get("skipped"):
                await ws.send_text(json.dumps({"skipped": True, "seq": n}))
            else:
                await ws.send_text(json.dumps({
                    "source": result["source"],
                    "english": result["english"],
                    "speaker": result.get("speaker", ""),
                    "lang_tag": source_lang.upper(),
                    "seq": n,
                    "ts": time.time(),
                    "ms": result["ms"],
                }))
                # Fold aging turns into the rolling summary — after the reply is
                # sent, off the hot path, never overlapping itself.
                if (len(turns) % SUMMARY_EVERY == 0
                        and len(turns) > MAX_HISTORY
                        and not summary_state["busy"]):
                    summary_state["busy"] = True
                    _summary_executor.submit(update_summary)

    except WebSocketDisconnect:
        log.info("WS disconnected after %d chunks", seq)
    except Exception as exc:
        log.exception("WS error: %s", exc)
        try:
            await ws.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass


# ── Static frontend (dev convenience) ────────────────────────────────────────

_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
