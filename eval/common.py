"""Shared plumbing for the eval harness: interruption-safe JSONL state and
error classification.

Everything is incremental — each completed unit of work is on disk before the
next starts — so a run that dies mid-way (credits exhausted, network, Ctrl+C)
loses nothing and can be resumed with --resume.
"""

from __future__ import annotations

import json
import os
import time


def append_jsonl(path: str, record: dict) -> None:
    """Append one record and flush — the line is durable before we move on."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_run_state(run_dir: str, **fields) -> None:
    """Merge fields into run_state.json (coverage, stop reason, resume info)."""
    path = os.path.join(run_dir, "run_state.json")
    state = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    state.update(fields, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def read_run_state(run_dir: str) -> dict:
    path = os.path.join(run_dir, "run_state.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── API error classification ─────────────────────────────────────────────────

_BILLING_MARKERS = ("credit", "billing", "quota", "insufficient", "payment")
_BILLING_CLASSES = ("AuthenticationError", "PermissionDeniedError")


def classify_error(exc: Exception) -> str:
    """Classify an API exception: 'billing' (stop now, no retry), 'ratelimit'
    (retry with backoff), or 'other' (stop, reason recorded)."""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in _BILLING_CLASSES or any(m in msg for m in _BILLING_MARKERS):
        return "billing"
    if name == "RateLimitError" or "rate limit" in msg or "429" in msg:
        return "ratelimit"
    return "other"


class RunStopped(Exception):
    """Raised to stop a run cleanly after state has been saved."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def call_with_budget_guard(fn, *, retries: int = 2, backoff_s: float = 15.0):
    """Run one API-bound unit of work with the budget policy:
    billing errors stop immediately; rate limits retry `retries` times with
    backoff then stop; anything else stops with the reason recorded."""
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 — classified below
            kind = classify_error(exc)
            reason = f"{type(exc).__name__}: {exc}"
            if kind == "ratelimit" and attempt < retries:
                attempt += 1
                print(f"  rate-limited, retry {attempt}/{retries} in {backoff_s:.0f}s…", flush=True)
                time.sleep(backoff_s)
                continue
            raise RunStopped(f"[{kind}] {reason}") from exc
