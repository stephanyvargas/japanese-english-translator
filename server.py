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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
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

from translator.pipeline import run, _translate_with_context  # noqa: E402
from translator.transcriber import transcribe  # noqa: E402

app = FastAPI(title="Translator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)

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

# Only the last N turns are sent to Claude per chunk — keeps latency flat as
# the conversation grows (full history is still used for Whisper prompt continuity).
MAX_HISTORY = 6


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


@app.get("/health")
async def health():
    return {"ok": True}


@app.post("/translate")
async def translate_text(req: TranslateRequest):
    model = MODEL_ALIASES.get(req.model.lower(), req.model)
    lang_name = LANGUAGE_NAMES.get(req.source_lang.lower(), req.source_lang.upper())

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        _executor,
        lambda: run(req.text.strip(), model=model, source_lang=req.source_lang,
                    lang_name=lang_name, context=req.context),
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

    model = MODEL_ALIASES.get(cfg.get("model", "sonnet").lower(), cfg.get("model", "sonnet"))
    source_lang = cfg.get("source_lang", "ja")
    lang_name = LANGUAGE_NAMES.get(source_lang.lower(), source_lang.upper())
    context = cfg.get("context", "")

    anthropic_client = anthropic.Anthropic()
    openai_client = OpenAI()
    source_history: list[str] = []
    loop = asyncio.get_event_loop()
    seq = 0

    log.info("WS connected | model=%s lang=%s context=%r", model, source_lang, context)

    try:
        while True:
            # Receive binary audio frame (WAV bytes from browser MediaRecorder)
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

                prompt = source_history[-1] if source_history else ""
                text = transcribe(wav_bytes, openai_client, prompt=prompt, source_lang=source_lang)
                t2 = time.perf_counter()
                if not text.strip():
                    log.info("#%d skipped (no speech) | wav %dms | stt %dms",
                             n, (t1 - t0) * 1000, (t2 - t1) * 1000)
                    return {"skipped": True}

                source_history.append(text)
                english = _translate_with_context(
                    text, source_history[-MAX_HISTORY:], anthropic_client,
                    model=model, lang_name=lang_name, context=context,
                )
                t3 = time.perf_counter()

                total_ms = int((t3 - t0) * 1000)
                log.info(
                    "#%d recv %d bytes | wav %dms | stt %dms %r | mt %dms %r | total %dms",
                    n, len(raw_bytes), (t1 - t0) * 1000, (t2 - t1) * 1000, text[:40],
                    (t3 - t2) * 1000, (english or "")[:40], total_ms,
                )
                return {"source": text, "english": english or "", "ms": total_ms}

            result = await loop.run_in_executor(_executor, process, data)

            if result.get("skipped"):
                await ws.send_text(json.dumps({"skipped": True, "seq": n}))
            else:
                await ws.send_text(json.dumps({
                    "source": result["source"],
                    "english": result["english"],
                    "lang_tag": source_lang.upper(),
                    "seq": n,
                    "ts": time.time(),
                    "ms": result["ms"],
                }))

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
