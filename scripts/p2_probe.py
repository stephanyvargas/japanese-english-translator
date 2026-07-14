"""P2 spike probe: pin down the OpenAI Realtime transcription API by observation.

Connects, configures a transcription session, streams a WAV at real-time pace,
and prints every event type with timing — the implementation in
translator/realtime_stt.py is built from what this prints, not from memory.

Usage: python3 scripts/p2_probe.py /tmp/ivtest/q1_profile.wav
"""

import asyncio
import base64
import json
import os
import sys
import time
import wave

import websockets
from dotenv import load_dotenv

load_dotenv(".env")

URL = "wss://api.openai.com/v1/realtime?intent=transcription"
RATE = 24000

# New GA shape (developers.openai.com, 2026); the probe falls back to the older
# transcription_session.update shape if this one is rejected.
SESSION_NEW = {
    "type": "session.update",
    "session": {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": RATE},
                "transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
                "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
            }
        },
    },
}

SESSION_OLD = {
    "type": "transcription_session.update",
    "session": {
        "input_audio_format": "pcm16",
        "input_audio_transcription": {"model": "gpt-4o-mini-transcribe", "language": "en"},
        "turn_detection": {"type": "server_vad", "silence_duration_ms": 400},
    },
}


def wav_pcm_at_rate(path: str, rate: int) -> bytes:
    import numpy as np
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if sr != rate:
        idx = (np.arange(int(len(pcm) * rate / sr)) * sr / rate).astype(int)
        pcm = pcm[np.clip(idx, 0, len(pcm) - 1)]
    return pcm.tobytes()


async def main(path: str) -> None:
    pcm = wav_pcm_at_rate(path, RATE)
    dur = len(pcm) / 2 / RATE
    print(f"streaming {path} ({dur:.1f}s) at 1x pace, rate={RATE}")

    async with websockets.connect(
        URL, max_size=None,
        additional_headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    ) as ws:
        t0 = time.monotonic()

        async def reader():
            speech_end = None
            async for raw in ws:
                e = json.loads(raw)
                t = time.monotonic() - t0
                et = e.get("type", "?")
                if et.endswith(".delta"):
                    print(f"[{t:6.2f}s] {et}  {e.get('delta', '')!r}")
                elif "completed" in et or "done" in et:
                    print(f"[{t:6.2f}s] {et}  {e.get('transcript', '')!r}")
                    if "usage" in json.dumps(e):
                        print("          usage:", {k: v for k, v in e.items() if "usage" in k} or e)
                elif "error" in et:
                    print(f"[{t:6.2f}s] {et}: {json.dumps(e)[:400]}")
                else:
                    print(f"[{t:6.2f}s] {et}")

        rtask = asyncio.create_task(reader())

        await ws.send(json.dumps(SESSION_NEW))
        await asyncio.sleep(1.0)

        # stream in 100ms frames at real-time pace
        frame = RATE * 2 // 10
        for i in range(0, len(pcm), frame):
            await ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm[i:i + frame]).decode(),
            }))
            await asyncio.sleep(0.1)
        print(f"[{time.monotonic()-t0:6.2f}s] -- audio fully sent --")
        await asyncio.sleep(4)
        rtask.cancel()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ivtest/q1_profile.wav"))
