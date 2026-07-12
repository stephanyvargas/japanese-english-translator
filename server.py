"""FastAPI backend for the translator web UI.

Exposes two surfaces:
  POST /translate          — full quality pipeline (text input)
  WS   /ws/conversation   — real-time conversation mode (audio from browser)
  GET  /health             — Cloud Run health probe
"""

import asyncio
import io
import secrets
import json
import threading
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import anthropic
import av
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
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
from translator.interview import build_company_brief, generate_hints, looks_like_question  # noqa: E402
from translator.profile_ingest import extract_document, summarize_repo  # noqa: E402
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

# Interview hints run off the reply path (transcript is sent immediately; the
# hint follows as its own WS message) — own workers so a slow web search never
# competes with transcription.
_hint_executor = ThreadPoolExecutor(max_workers=2)

LANGUAGE_NAMES = {
    "en": "English",
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


# ── Interview profile ingestion ──────────────────────────────────────────────
# The client stores the results on the user's Firestore profile; these routes
# only convert (document → text, repo → blurb). Same auth guard as /translate.

def _require_auth(authorization: str) -> None:
    if REQUIRE_AUTH:
        token = authorization.removeprefix("Bearer ").strip()
        if not _verify_token(token):
            raise HTTPException(status_code=401, detail="Sign in first.")


@app.post("/profile/ingest")
async def profile_ingest(file: UploadFile = File(...), note: str = Form(""),
                         authorization: str = Header(default="")):
    _require_auth(authorization)
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (10 MB max).")
    loop = asyncio.get_event_loop()
    try:
        text = await loop.run_in_executor(
            _executor,
            lambda: extract_document(file.filename or "document", data, anthropic.Anthropic()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    log.info("profile ingest %r (%d bytes -> %d chars) note=%r",
             file.filename, len(data), len(text), note[:40])
    return {"name": file.filename, "note": note, "text": text}


class RepoRequest(BaseModel):
    repo: str


@app.post("/profile/github")
async def profile_github(req: RepoRequest, authorization: str = Header(default="")):
    _require_auth(authorization)
    loop = asyncio.get_event_loop()
    try:
        summary = await loop.run_in_executor(
            _executor, lambda: summarize_repo(req.repo, anthropic.Anthropic()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    repo = req.repo.strip().removeprefix("https://github.com/").strip("/")
    log.info("profile github %r -> %d chars", repo, len(summary))
    return {"repo": repo, "summary": summary}


# ── Live sessions: server-owned state, typed events, resumability (P1) ──────
#
# A LiveSession owns everything that used to live in the WebSocket closure —
# transcript, assembler, diarization, summary, company brief — plus an event
# ring buffer. The WS is a dumb pipe: clients create a session over REST, then
# connect (and reconnect) with ?session_id=…&last_seq=N; missed events replay
# from the ring. Every server→client frame is a versioned envelope:
#   {v, sid, seq, ts, type, data}
# Event types: transcript.final · translation.final · hint.pending|partial|final
# · chunk.ack (transitional, dies with the batch-capture path in P2) ·
# session.status · error.

SESSION_RING = 500          # replayable events kept per session
SESSION_GRACE_S = 120       # how long a disconnected session survives


class LiveSession:
    def __init__(self, cfg: dict, uid: str | None, loop: asyncio.AbstractEventLoop):
        self.id = "sess_" + secrets.token_hex(8)
        self.uid = uid
        self.loop = loop

        self.mode = cfg.get("mode", "interpret")
        self.is_interview = self.mode == "interview"
        self.model = MODEL_ALIASES.get(cfg.get("model", "sonnet").lower(),
                                       cfg.get("model", "sonnet"))
        self.source_lang = cfg.get("source_lang", "ja")
        self.lang_name = LANGUAGE_NAMES.get(self.source_lang.lower(),
                                            self.source_lang.upper())
        self.stt_model = cfg.get("stt_model") or (
            "gpt-4o-mini-transcribe" if self.is_interview else DEFAULT_STT_MODEL)
        self.context = cfg.get("context", "")
        self.profile = cfg.get("profile", "")
        self.participants = cfg.get("participants", "")
        self.diarize = bool(cfg.get("diarize", True))
        self.flush_timeout = 2 if self.is_interview else 6

        g = Glossary.parse(cfg.get("glossary", ""))
        self.glossary_stt = g.format_for_stt()
        self.glossary_prompt = g.format_for_prompt()

        names = [n.strip() for n in self.participants.splitlines() if n.strip()]
        self.speaker_book = SpeakerBook(names=names)
        self.assembler = ChunkAssembler(max_parts=2 if self.is_interview else 4)
        self.pending_wav = b""
        self.turns: list[tuple[str, str]] = []
        self.summary_state = {"summary": "", "upto": 0, "busy": False}
        self.brief_state = {"brief": ""}
        self.chunk_n = 0

        self.anthropic_client = anthropic.Anthropic()
        self.openai_client = OpenAI()

        self.seq = 0
        self._lock = threading.Lock()
        self.ring: deque = deque(maxlen=SESSION_RING)
        self.ws: WebSocket | None = None
        self.disconnected_at: float | None = time.time()
        self.attached_before = False

        self.session_dir = ""
        if SAVE_CHUNKS_DIR:
            self.session_dir = os.path.join(SAVE_CHUNKS_DIR, time.strftime("%Y%m%d-%H%M%S"))
            os.makedirs(self.session_dir, exist_ok=True)

    def _make_event(self, type_: str, data: dict) -> dict:
        """Allocate a seq and append to the ring (thread-safe)."""
        with self._lock:
            self.seq += 1
            evt = {"v": 1, "sid": self.id, "seq": self.seq, "ts": time.time(),
                   "type": type_, "data": data}
            self.ring.append(evt)
        return evt

    async def emit(self, type_: str, data: dict) -> None:
        """Append to the ring and forward to the live socket (if any).

        Events emitted while disconnected are not lost — they wait in the ring
        for the next connection's last_seq replay.
        """
        evt = self._make_event(type_, data)
        if self.ws is not None:
            try:
                await self.ws.send_text(json.dumps(evt))
            except Exception:
                self.ws = None
                self.disconnected_at = time.time()

    def emit_threadsafe(self, type_: str, data: dict) -> None:
        """Emit from an executor thread. If no live event loop exists (client
        between connections), the event still lands in the ring for replay —
        results generated while disconnected must never be lost."""
        try:
            if self.loop is not None and not self.loop.is_closed():
                asyncio.run_coroutine_threadsafe(self.emit(type_, data), self.loop)
                return
        except Exception:
            pass
        self._make_event(type_, data)


_sessions: dict[str, LiveSession] = {}


def _reap_sessions() -> None:
    now = time.time()
    for sid, s in list(_sessions.items()):
        if s.ws is None and s.disconnected_at and now - s.disconnected_at > SESSION_GRACE_S:
            del _sessions[sid]
            log.info("session %s reaped after %ds idle", sid, SESSION_GRACE_S)


@app.post("/session")
async def create_session(cfg: dict, authorization: str = Header(default="")):
    """Create a live session; returns the id the WebSocket connects with."""
    uid = None
    if REQUIRE_AUTH:
        token = authorization.removeprefix("Bearer ").strip()
        uid = _verify_token(token)
        if not uid:
            raise HTTPException(status_code=401, detail="Sign in first.")
    _reap_sessions()
    s = LiveSession(cfg, uid, asyncio.get_event_loop())
    _sessions[s.id] = s

    if s.is_interview and s.context.strip():
        # Pre-research the company while the user is settling in.
        s.emit_threadsafe("session.status", {"state": "warming_brief", "detail": s.context})
        def _warm():
            s.brief_state["brief"] = build_company_brief(s.context, anthropic.Anthropic())
            log.info("session %s company brief ready (%d chars)", s.id, len(s.brief_state["brief"]))
        _hint_executor.submit(_warm)

    log.info("session %s created | uid=%s mode=%s model=%s stt=%s lang=%s",
             s.id, uid or "-", s.mode, s.model, s.stt_model, s.source_lang)
    return {"session_id": s.id}


# ── per-session pipeline (module-level; everything reads the session) ────────

def _save_chunk(s: LiveSession, n: int, wav_bytes: bytes, record: dict) -> None:
    try:
        with open(os.path.join(s.session_dir, f"{n:04d}.wav"), "wb") as f:
            f.write(wav_bytes)
        with open(os.path.join(s.session_dir, "session.jsonl"), "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.info("#%d chunk save failed (%s)", n, exc)


def _update_summary(s: LiveSession) -> None:
    try:
        end = len(s.turns) - MAX_HISTORY
        if end <= s.summary_state["upto"]:
            return
        lines = [_label(t, sp) for t, sp in s.turns[s.summary_state["upto"]:end]]
        s.summary_state["summary"] = _summarize_history(
            lines, s.summary_state["summary"], s.anthropic_client)
        s.summary_state["upto"] = end
        log.info("summary updated (folded %d turns): %r",
                 len(lines), s.summary_state["summary"][:80])
    except Exception as exc:
        log.info("summary update failed (%s)", exc)
    finally:
        s.summary_state["busy"] = False


def _translate_turn(s: LiveSession, n: int, text: str, wav_bytes: bytes,
                    t0: float, stt_ms: int, merged: int) -> dict:
    """Diarize + (interpret) translate an assembled sentence; log and record."""
    speaker = ""
    sim = -1.0
    if s.diarize:
        try:
            emb = _embedder.embed(wav_bytes)
            if emb is not None:
                _, speaker, sim = s.speaker_book.assign(emb)
        except Exception as exc:
            log.info("#%d diarize skipped (%s)", n, exc)

    s.turns.append((text, speaker))
    # Verbatim window extends back to the summary watermark (full coverage).
    start = max(s.summary_state["upto"], len(s.turns) - MAX_VERBATIM)
    start = min(start, max(0, len(s.turns) - MAX_HISTORY))
    window = s.turns[start:]

    english, repaired = "", False
    if s.is_interview:
        hint_ctx = [_label(t, sp) for t, sp in window[:-1]]
    else:
        english, repaired = _translate_with_context(
            text, [t for t, _ in window], s.anthropic_client,
            model=s.model, lang_name=s.lang_name, context=s.context,
            glossary=s.glossary_prompt, speaker=speaker,
            speakers=[sp for _, sp in window], participants=s.participants,
            summary=s.summary_state["summary"], diarized=s.diarize,
        )
    t3 = time.perf_counter()

    total_ms = int((t3 - t0) * 1000)
    log.info(
        "#%d stt %dms %r | merged=%d | %s sim=%.2f | mt %dms%s %r | total %dms",
        n, stt_ms, text[:40], merged, speaker or "-", sim,
        (t3 - t0) * 1000 - stt_ms, " repaired" if repaired else "",
        (english or "")[:40], total_ms,
    )
    if s.session_dir:
        _save_chunk(s, n, wav_bytes, {
            "seq": n, "ts": time.time(), "speaker": speaker,
            "sim": round(sim, 3), "source": text, "english": english or "",
            "repaired": repaired, "ms": total_ms, "merged": merged,
        })
    result = {"source": text, "english": english or "", "repaired": repaired,
              "speaker": speaker, "ms": total_ms}
    if s.is_interview:
        result["hint_ctx"] = hint_ctx
    return result


def _process_chunk(s: LiveSession, n: int, raw_bytes: bytes) -> dict:
    """Blocking pipeline step (runs on the executor): decode → STT → assemble."""
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

    prompt = " ".join(t for t, _ in s.turns[-3:])[-STT_PROMPT_CHARS:]
    text = transcribe(wav_bytes, s.openai_client, prompt=prompt,
                      source_lang=s.source_lang, model=s.stt_model,
                      glossary=s.glossary_stt)
    t2 = time.perf_counter()
    stt_ms = int((t2 - t1) * 1000)
    if not text.strip():
        log.info("#%d skipped (no speech) | wav %dms | stt %dms",
                 n, (t1 - t0) * 1000, stt_ms)
        return {"skipped": True}

    dur_s = max(0.0, (len(wav_bytes) - 44) / 2 / 16000)
    s.pending_wav = wav_bytes
    assembled = s.assembler.add(text, dur_s)
    if assembled is None:
        log.info("#%d buffered (mid-sentence) %r", n, text[:40])
        return {"buffered": True}

    return _translate_turn(s, n, assembled, wav_bytes, t0, stt_ms,
                           s.assembler.last_merged)


def _flush_pending(s: LiveSession, n: int) -> dict:
    text = s.assembler.flush()
    if not text:
        return {"skipped": True}
    return _translate_turn(s, n, text, s.pending_wav, time.perf_counter(),
                           0, s.assembler.last_merged)


def _hint_work(s: LiveSession, utt_id: str, text: str, speaker: str,
               hint_ctx: list[str]) -> None:
    """Generate interview hints on the hint executor — deliberately independent
    of any event loop or connection, so a hint that completes while the client
    is disconnected still lands in the ring and replays on reconnect."""
    t0 = time.perf_counter()

    def on_partial(partial: dict) -> None:
        s.emit_threadsafe("hint.partial", {"utt_id": utt_id, **partial})

    try:
        hint = generate_hints(text, hint_ctx, s.profile, s.anthropic_client,
                              context=s.context, speaker=speaker,
                              company_brief=s.brief_state["brief"],
                              on_partial=on_partial)
        hint["ms"] = int((time.perf_counter() - t0) * 1000)
        log.info("%s hint %dms | q=%s searched=%s %r", utt_id, hint["ms"],
                 hint["is_question"], hint["searched"], hint["gist"][:30])
        s.emit_threadsafe("hint.final", {"utt_id": utt_id, **hint})
    except Exception as exc:
        log.info("%s hint delivery failed (%s)", utt_id, exc)
        s.emit_threadsafe("hint.final", {"utt_id": utt_id, "is_question": False,
                                         "gist": "", "bullets": [], "angle": "",
                                         "searched": False, "ms": 0})


async def _emit_turn_events(s: LiveSession, n: int, result: dict) -> None:
    """transcript.final, then mode-specific follow-ups."""
    utt_id = f"u{n}"
    await s.emit("transcript.final", {
        "utt_id": utt_id, "channel": "mixed",
        "speaker": result.get("speaker", ""), "text": result["source"],
        "lang": s.source_lang,
    })
    if "hint_ctx" in result:
        if looks_like_question(result["source"]):
            await s.emit("hint.pending", {"utt_id": utt_id})
            _hint_executor.submit(_hint_work, s, utt_id, result["source"],
                                  result.get("speaker", ""), result["hint_ctx"])
        else:
            log.info("#%d hint gated out (not question-like)", n)
    else:
        await s.emit("translation.final", {
            "utt_id": utt_id, "english": result["english"],
            "repaired": result.get("repaired", False), "ms": result["ms"],
        })
    # Fold aging turns into the rolling summary — off the hot path.
    if (len(s.turns) % SUMMARY_EVERY == 0 and len(s.turns) > MAX_HISTORY
            and not s.summary_state["busy"]):
        s.summary_state["busy"] = True
        _summary_executor.submit(_update_summary, s)


# ── WebSocket — the dumb pipe ────────────────────────────────────────────────

@app.websocket("/ws/conversation")
async def ws_conversation(ws: WebSocket, session_id: str = Query(""),
                          last_seq: int = Query(0)):
    await ws.accept()
    s = _sessions.get(session_id)
    if s is None:
        await ws.close(code=4404)
        return

    # Start frame authenticates this connection: {"op": "start", "id_token": …}
    try:
        raw = await asyncio.wait_for(ws.receive_text(), timeout=10)
        start = json.loads(raw)
        assert start.get("op") == "start"
    except Exception:
        await ws.close(code=1008)
        return
    if REQUIRE_AUTH:
        loop_ = asyncio.get_event_loop()
        uid = await loop_.run_in_executor(_executor, _verify_token,
                                          start.get("id_token", ""))
        if not uid or uid != s.uid:
            await ws.close(code=4401)
            return

    # Attach + replay anything missed while disconnected. Refresh the loop
    # reference: emit_threadsafe must target the loop that owns THIS connection
    # (matters under test clients; harmless under uvicorn's single loop).
    s.loop = asyncio.get_running_loop()
    s.ws = ws
    s.disconnected_at = None
    replayed = 0
    for evt in list(s.ring):
        if evt["seq"] > last_seq:
            await ws.send_text(json.dumps(evt))
            replayed += 1
    await s.emit("session.status",
                 {"state": "resumed" if s.attached_before else "live",
                  "detail": f"replayed {replayed}" if replayed else ""})
    s.attached_before = True
    log.info("session %s attached | last_seq=%d replayed=%d", s.id, last_seq, replayed)

    loop = asyncio.get_event_loop()
    try:
        while True:
            # While a partial sentence is buffered, wait at most flush_timeout —
            # a silent speaker means the sentence won't be continued; flush it.
            if s.assembler.pending:
                try:
                    data = await asyncio.wait_for(ws.receive_bytes(),
                                                  timeout=s.flush_timeout)
                except asyncio.TimeoutError:
                    s.chunk_n += 1
                    n = s.chunk_n
                    try:
                        result = await loop.run_in_executor(_executor, _flush_pending, s, n)
                    except Exception as exc:
                        log.exception("#%d flush failed: %s", n, exc)
                        await s.emit("error", {"scope": "utterance",
                                               "message": str(exc), "retryable": True})
                        continue
                    if not result.get("skipped"):
                        await _emit_turn_events(s, n, result)
                    continue
            else:
                data = await ws.receive_bytes()
            s.chunk_n += 1
            n = s.chunk_n

            # A failing chunk reports its error and the session continues.
            try:
                result = await loop.run_in_executor(_executor, _process_chunk, s, n, data)
            except Exception as exc:
                log.exception("#%d chunk failed: %s", n, exc)
                await s.emit("error", {"scope": "utterance",
                                       "message": str(exc), "retryable": True})
                await s.emit("chunk.ack", {"chunk": n, "disposition": "error"})
                continue

            if not (result.get("skipped") or result.get("buffered")):
                await _emit_turn_events(s, n, result)
            # Transitional (P1-only): the batch-capture client sends one chunk at
            # a time and waits for this ack. Dies with the capture path in P2.
            disposition = ("buffered" if result.get("buffered")
                           else "skipped" if result.get("skipped") else "processed")
            await s.emit("chunk.ack", {"chunk": n, "disposition": disposition})

    except WebSocketDisconnect:
        s.ws = None
        s.disconnected_at = time.time()
        log.info("session %s detached after %d chunks (grace %ds)",
                 s.id, s.chunk_n, SESSION_GRACE_S)
    except Exception as exc:
        log.exception("session %s WS error: %s", s.id, exc)
        s.ws = None
        s.disconnected_at = time.time()


# ── Static frontend (dev convenience) ────────────────────────────────────────

_frontend = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend):
    app.mount("/", StaticFiles(directory=_frontend, html=True), name="frontend")
