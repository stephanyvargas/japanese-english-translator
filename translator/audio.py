import io
import queue
import sys
import threading
from collections.abc import Iterator

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECS = 0.1


def _lazy_imports():
    try:
        import numpy as np
        import scipy.io.wavfile as wavfile
        import sounddevice as sd
        return np, wavfile, sd
    except (ImportError, OSError) as e:
        print(
            f"Microphone input unavailable: {e}\n"
            "Install PortAudio (e.g. 'sudo apt install libportaudio2') and try again.\n"
            "You can also use --text to provide source text directly.",
            file=sys.stderr,
        )
        sys.exit(1)


def frames_to_wav(frames, np, wavfile) -> tuple[bytes, float]:
    audio = np.concatenate(frames, axis=0).squeeze()
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio_int16)
    return buf.getvalue(), len(audio_int16) / SAMPLE_RATE


class AudioCapture:
    """Continuously records from the mic into an internal queue on a background thread.

    The mic is always on — draining the queue while Claude is processing means
    no audio is ever lost between translation cycles.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._np = self._wavfile = self._sd = None

    def start(self) -> None:
        self._np, self._wavfile, self._sd = _lazy_imports()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def drain(self) -> list:
        frames = []
        while True:
            try:
                frames.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return frames

    def speech_ratio(self, frames, threshold: float) -> float:
        if not frames:
            return 0.0
        np = self._np
        ratios = [float(np.sqrt(np.mean(f ** 2))) > threshold for f in frames]
        return sum(ratios) / len(ratios)

    def to_wav(self, frames) -> tuple[bytes, float]:
        return frames_to_wav(frames, self._np, self._wavfile)

    def _loop(self) -> None:
        chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
        sd = self._sd
        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
                while not self._stop.is_set():
                    chunk, _ = stream.read(chunk_samples)
                    self._queue.put(chunk.copy())
        except Exception as e:
            print(f"\n[Audio thread error: {e}]", file=sys.stderr)


def record_from_mic(
    max_seconds: int = 30,
    silence_ms: int = 600,
    threshold: float = 0.02,
    lang_name: str = "source language",
) -> bytes:
    """Record a single utterance. Stops on silence or max_seconds."""
    np, wavfile, sd = _lazy_imports()

    chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
    silence_chunks_needed = int(silence_ms / (CHUNK_SECS * 1000))
    max_chunks = int(max_seconds / CHUNK_SECS)

    print(f"Listening... speak in {lang_name} now. (Recording stops after a pause)", flush=True)

    audio_frames: list = []
    silent_count = 0
    recording_started = False

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
            for _ in range(max_chunks):
                chunk, _ = stream.read(chunk_samples)
                rms = float(np.sqrt(np.mean(chunk ** 2)))

                if rms > threshold:
                    if not recording_started:
                        recording_started = True
                        print("Recording...", flush=True)
                    silent_count = 0
                    audio_frames.append(chunk.copy())
                elif recording_started:
                    audio_frames.append(chunk.copy())
                    silent_count += 1
                    if silent_count >= silence_chunks_needed:
                        break
    except KeyboardInterrupt:
        if not audio_frames:
            print("\nCancelled.", file=sys.stderr)
            sys.exit(0)

    if not audio_frames:
        raise RuntimeError("No speech detected. Check your microphone and try again.")

    wav_bytes, duration = frames_to_wav(audio_frames, np, wavfile)
    print(f"Captured {duration:.1f}s of audio.", flush=True)
    return wav_bytes


def stream_chunks(
    max_seconds: int = 20,
    silence_ms: int = 600,
    threshold: float = 0.02,
    lang_name: str = "source language",
) -> Iterator[bytes]:
    """Yield WAV chunks on natural pauses or hard cap. No threading — mic gaps during processing."""
    np, wavfile, sd = _lazy_imports()

    chunk_samples = int(SAMPLE_RATE * CHUNK_SECS)
    silence_chunks_needed = int(silence_ms / (CHUNK_SECS * 1000))
    max_chunks = int(max_seconds / CHUNK_SECS)

    print(f"Listening — speak in {lang_name}. Ctrl+C to stop.\n", flush=True)

    while True:
        audio_frames: list = []
        silent_count = 0
        recording_started = False

        try:
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
                print("Listening...", flush=True)
                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_samples)
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    if rms > threshold:
                        if not recording_started:
                            recording_started = True
                            print("Recording...", flush=True)
                        silent_count = 0
                        audio_frames.append(chunk.copy())
                    elif recording_started:
                        audio_frames.append(chunk.copy())
                        silent_count += 1
                        if silent_count >= silence_chunks_needed:
                            break
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
            return

        if not audio_frames:
            continue

        wav_bytes, duration = frames_to_wav(audio_frames, np, wavfile)
        reason = "pause" if silent_count >= silence_chunks_needed else f"{max_seconds}s cap"
        print(f"[{duration:.1f}s — {reason}]", flush=True)
        yield wav_bytes
