"""Lightweight, chunk-aligned speaker diarization for conversation mode.

Conversation mode already cuts audio on natural pauses, so each chunk is
essentially one speaker's turn. We therefore don't need continuous-stream
diarization (pyannote et al., which pull PyTorch and don't fit a 1Gi/1-vCPU
Cloud Run) — only "which speaker is this chunk?" once per chunk.

Default backend: an **MFCC-statistics speaker signature** (mean + std of MFCCs
over the chunk) using only scipy/numpy — no new dependency, no model file. A 3–8s
turn yields plenty of frames for a stable signature to cluster 2–4 speakers.

Optional upgrade: set SPEAKER_MODEL_PATH to a d-vector ONNX model (and install
onnxruntime) for stronger embeddings. See translator/models/README.md.

Speakers are assigned online: each chunk's embedding is compared (cosine) to the
running per-session centroids; the nearest above a threshold wins, otherwise a new
speaker is minted. Labels are "Speaker N", optionally mapped to provided names in
first-appearance order.
"""

from __future__ import annotations

import io
import os
import wave
from functools import lru_cache

import numpy as np

_SAMPLE_RATE = 16000
_N_MFCC = 13
_N_MELS = 26
_FRAME_MS = 25
_HOP_MS = 10
_FMIN = 20.0
_FMAX = 7600.0

# Minimum voiced length worth embedding (shorter → unreliable signature).
_MIN_SAMPLES = int(_SAMPLE_RATE * 0.4)

# 0.82: real-audio eval showed same-speaker re-match sims of 0.89–0.98 while a
# genuinely different voice scored 0.66; 0.72 under-split a real two-person
# session (everything labeled Speaker 1). Raise/lower via DIARIZE_THRESHOLD if
# the per-chunk sim= logs show merging/splitting.
DEFAULT_THRESHOLD = float(os.environ.get("DIARIZE_THRESHOLD", "0.82"))


# ── signal helpers ───────────────────────────────────────────────────────────

def _read_wav_mono_f32(wav_bytes: bytes) -> np.ndarray:
    """Decode 16-bit PCM WAV bytes to a mono float32 array in [-1, 1]."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        n_ch = w.getnchannels()
        sampwidth = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sampwidth != 2:
        # _to_wav always emits s16; bail defensively rather than misread.
        raise ValueError(f"expected 16-bit PCM, got sampwidth={sampwidth}")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_ch > 1:
        data = data.reshape(-1, n_ch).mean(axis=1)
    return data


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


@lru_cache(maxsize=4)
def _mel_filterbank(n_fft: int, sr: int) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2 + 1). Cached — the
    inputs are constants at runtime, so this is built once, not per chunk.
    Treat the returned array as read-only."""
    n_bins = n_fft // 2 + 1
    mel_pts = np.linspace(_hz_to_mel(np.array([_FMIN]))[0],
                          _hz_to_mel(np.array([_FMAX]))[0], _N_MELS + 2)
    hz_pts = _mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_bins - 1)
    fb = np.zeros((_N_MELS, n_bins), dtype=np.float32)
    for m in range(1, _N_MELS + 1):
        left, center, right = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        if center == left or right == center:
            continue
        fb[m - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        fb[m - 1, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


@lru_cache(maxsize=4)
def _dct_matrix(n_out: int, n_in: int) -> np.ndarray:
    """Orthonormal DCT-II matrix, shape (n_out, n_in). Cached (read-only)."""
    n = np.arange(n_in)
    k = np.arange(n_out)[:, None]
    d = np.cos(np.pi / n_in * (n + 0.5) * k) * np.sqrt(2.0 / n_in)
    d[0] *= 1.0 / np.sqrt(2.0)
    return d.astype(np.float32)


@lru_cache(maxsize=4)
def _hann_window(frame_len: int) -> np.ndarray:
    """Hanning window, cached (read-only)."""
    return np.hanning(frame_len).astype(np.float32)


def _mfcc(samples: np.ndarray, sr: int = _SAMPLE_RATE) -> np.ndarray:
    """Compute MFCCs, shape (n_frames, _N_MFCC). Returns empty if too short."""
    frame_len = int(sr * _FRAME_MS / 1000)
    hop = int(sr * _HOP_MS / 1000)
    if samples.shape[0] < frame_len:
        return np.zeros((0, _N_MFCC), dtype=np.float32)

    # Pre-emphasis + framing.
    emph = np.append(samples[0], samples[1:] - 0.97 * samples[:-1])
    n_frames = 1 + (emph.shape[0] - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = emph[idx] * _hann_window(frame_len)

    n_fft = 1
    while n_fft < frame_len:
        n_fft *= 2
    power = (np.abs(np.fft.rfft(frames, n=n_fft)) ** 2) / n_fft

    fb = _mel_filterbank(n_fft, sr)
    mel_energy = np.maximum(power @ fb.T, 1e-10)
    log_mel = np.log(mel_energy)
    mfcc = log_mel @ _dct_matrix(_N_MFCC, _N_MELS).T
    return mfcc.astype(np.float32)


# ── embedder ─────────────────────────────────────────────────────────────────

class SpeakerEmbedder:
    """Produce a fixed-length speaker signature for a chunk of WAV bytes.

    Default MFCC-statistics backend needs only numpy/scipy. If an ONNX d-vector
    model is configured and loadable, it is used instead (see models/README.md).
    """

    def __init__(self, model_path: str | None = None):
        self.available = True
        self._session = None
        model_path = model_path or os.environ.get("SPEAKER_MODEL_PATH")
        if model_path and os.path.isfile(model_path):
            try:
                import onnxruntime  # noqa: F401 — optional dependency
                self._session = onnxruntime.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"])
                self._input_name = self._session.get_inputs()[0].name
            except Exception:
                # Fall back to MFCC signature if the ONNX path can't load.
                self._session = None

    def embed(self, wav_bytes: bytes) -> np.ndarray | None:
        """Return an L2-normalized embedding, or None if the chunk is too short."""
        samples = _read_wav_mono_f32(wav_bytes)
        if samples.shape[0] < _MIN_SAMPLES:
            return None

        if self._session is not None:
            feats = _mfcc(samples)  # (frames, 13) — model expects fbank/mfcc features
            if feats.shape[0] == 0:
                return None
            out = self._session.run(None, {self._input_name: feats[None].astype(np.float32)})
            vec = np.asarray(out[0]).reshape(-1)
        else:
            mfcc = _mfcc(samples)
            if mfcc.shape[0] == 0:
                return None
            # Drop c0 (frame energy — not speaker-discriminative) and use the
            # mean+std of the remaining cepstra. Mean-centring the vector
            # (correlation distance) removes the large common offset that would
            # otherwise compress all cosine similarities near 1.0.
            stats = np.concatenate([mfcc[:, 1:].mean(axis=0), mfcc[:, 1:].std(axis=0)])
            vec = stats - stats.mean()

        norm = np.linalg.norm(vec)
        if norm == 0 or not np.isfinite(norm):
            return None
        return (vec / norm).astype(np.float32)


# ── online clustering ────────────────────────────────────────────────────────

class SpeakerBook:
    """Per-session online speaker clustering by cosine similarity."""

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, names: list[str] | None = None):
        self.threshold = threshold
        self.names = [n.strip() for n in (names or []) if n.strip()]
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []

    def label_for(self, speaker_id: int) -> str:
        if speaker_id < len(self.names):
            return self.names[speaker_id]
        return f"Speaker {speaker_id + 1}"

    def assign(self, embedding: np.ndarray) -> tuple[int, str, float]:
        """Return (speaker_id, label, best_sim) for an embedding, updating centroids.

        ``best_sim`` is the best cosine similarity *before* thresholding (-1.0 when
        the book was empty) — logged per chunk so DIARIZE_THRESHOLD can be tuned
        against real audio: raise it if distinct speakers merge, lower if one splits.
        """
        best_id, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            sim = float(np.dot(embedding, c))  # both are L2-normalized → cosine
            if sim > best_sim:
                best_id, best_sim = i, sim

        if best_id >= 0 and best_sim >= self.threshold:
            n = self._counts[best_id] + 1
            merged = self._centroids[best_id] * (n - 1) / n + embedding / n
            merged /= np.linalg.norm(merged) or 1.0
            self._centroids[best_id] = merged.astype(np.float32)
            self._counts[best_id] = n
            return best_id, self.label_for(best_id), best_sim

        self._centroids.append(embedding.astype(np.float32))
        self._counts.append(1)
        new_id = len(self._centroids) - 1
        return new_id, self.label_for(new_id), best_sim
