"""Streaming transcription over the OpenAI Realtime API (architecture v2, P2).

One RealtimeTranscriber per live session: the browser streams raw PCM frames,
we forward them to a transcription-intent Realtime socket, and the API's
server-side VAD segments utterances for us. Callbacks fire on the owning
event loop:

    on_partial(utt_id, text_so_far)   — delta burst just before completion
    on_final(utt_id, text)            — punctuated utterance, ~1s after speech ends

Wire shapes were pinned by observation (scripts/p2_probe.py), not memory:
session.update with session.type="transcription", audio/pcm @ 24kHz,
input_audio_buffer.append with base64 PCM, and
conversation.item.input_audio_transcription.{delta,completed} events.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os

import websockets

log = logging.getLogger("translator")

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
PCM_RATE = 24000
DEFAULT_MODEL = "gpt-4o-mini-transcribe"
VAD_SILENCE_MS = 350   # server VAD: how much trailing silence ends an utterance


class RealtimeTranscriber:
    """Owns one Realtime transcription socket; restarts once on failure."""

    def __init__(self, on_partial, on_final, model: str = DEFAULT_MODEL,
                 language: str = "en", prompt: str = ""):
        self.on_partial = on_partial
        self.on_final = on_final
        self.model = model
        self.language = language
        self.prompt = prompt
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._partials: dict[str, str] = {}   # item_id -> accumulated text
        self._utt_ids: dict[str, str] = {}    # item_id -> our utterance id
        self._utt_n = 0
        self._closed = False
        self._restarts = 0

    async def _connect(self) -> None:
        self._ws = await websockets.connect(
            REALTIME_URL, max_size=None,
            additional_headers={
                "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        transcription = {"model": self.model, "language": self.language}
        if self.prompt:
            transcription["prompt"] = self.prompt
        await self._ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {"input": {
                    "format": {"type": "audio/pcm", "rate": PCM_RATE},
                    "transcription": transcription,
                    "turn_detection": {"type": "server_vad",
                                       "silence_duration_ms": VAD_SILENCE_MS},
                }},
            },
        }))
        self._reader = asyncio.create_task(self._read_loop())

    def _utt_for(self, item_id: str) -> str:
        if item_id not in self._utt_ids:
            self._utt_n += 1
            self._utt_ids[item_id] = f"s{self._utt_n}"
        return self._utt_ids[item_id]

    async def _read_loop(self) -> None:
        try:
            async for raw in self._ws:
                e = json.loads(raw)
                et = e.get("type", "")
                if et == "conversation.item.input_audio_transcription.delta":
                    item = e.get("item_id", "")
                    self._partials[item] = self._partials.get(item, "") + e.get("delta", "")
                    self.on_partial(self._utt_for(item), self._partials[item])
                elif et == "conversation.item.input_audio_transcription.completed":
                    item = e.get("item_id", "")
                    text = (e.get("transcript") or "").strip()
                    self._partials.pop(item, None)
                    usage = e.get("usage", {})
                    log.info("rt-stt final %s %r (audio_tokens=%s)",
                             self._utt_for(item), text[:40],
                             usage.get("input_token_details", {}).get("audio_tokens"))
                    if text:
                        self.on_final(self._utt_for(item), text)
                elif et == "error":
                    log.info("rt-stt error event: %s", json.dumps(e)[:300])
        except Exception as exc:
            if not self._closed:
                log.info("rt-stt reader ended (%s)", exc)

    async def send_audio(self, pcm: bytes) -> None:
        """Forward one PCM frame; transparently restart the socket once if it
        died (network blip on the OpenAI side must not kill the interview)."""
        if self._closed:
            return
        if self._ws is None:
            await self._connect()
        try:
            await self._ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode(),
            }))
        except Exception as exc:
            if self._restarts >= 3:
                raise
            self._restarts += 1
            log.info("rt-stt socket died (%s) — reconnecting (%d/3)", exc, self._restarts)
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            await self._connect()

    async def close(self) -> None:
        self._closed = True
        if self._reader:
            self._reader.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
