"""Aggregate a judged run into a Markdown quality report.

Purely local (no API calls). Works on partial data: the header states exactly
how much of the replay and judging completed, so a partial report is never
mistaken for a full one.

Usage:
  python3 -m eval.report eval/runs/20260708-141900
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from eval.common import load_jsonl, read_run_state

DIMS = ("accuracy", "completeness", "naturalness", "context_consistency")


def _dist(values: list[int]) -> str:
    return " ".join(f"{v}:{values.count(v)}" for v in (5, 4, 3, 2, 1) if values.count(v))


def main() -> None:
    p = argparse.ArgumentParser(description="Build the quality report for a run")
    p.add_argument("run_dir")
    args = p.parse_args()
    run_dir = args.run_dir.rstrip("/")

    session = load_jsonl(os.path.join(run_dir, "session.jsonl"))
    scores = load_jsonl(os.path.join(run_dir, "scores.jsonl"))
    state = read_run_state(run_dir)
    if not session:
        sys.exit(f"No session.jsonl in {run_dir}")

    turns = [r for r in session if r.get("source") and r.get("english")]
    skipped = [r for r in session if r.get("skipped")]
    by_seq = {r["seq"]: r for r in turns}
    scored = [s for s in scores if s["seq"] in by_seq]

    lines: list[str] = []
    w = lines.append
    cfg = state.get("config", {})

    w("# Translation quality report")
    w("")
    w(f"- **Audio**: `{os.path.basename(cfg.get('audio', '?'))}` "
      f"window {cfg.get('start', '?')}s + {cfg.get('minutes', '?')} min")
    w(f"- **Translator**: `{cfg.get('model', '?')}` · context: {cfg.get('context', '—')!r}")
    replay_reason = state.get("stopped_reason", "?")
    judge_reason = state.get("judge_stopped_reason", "not run")
    w(f"- **Coverage — replay**: {state.get('completed_chunks', len(session))}/"
      f"{state.get('total_chunks', '?')} chunks ({replay_reason})")
    w(f"- **Coverage — judge**: {len(scored)}/{len(turns)} scoreable turns ({judge_reason})")
    if replay_reason != "completed" or judge_reason not in ("completed", "not run"):
        w("")
        w("> ⚠️ **Partial run** — figures below cover only the completed portion.")
    w("")

    # ── judge scores ─────────────────────────────────────────────────────────
    if scored:
        w("## Judge scores (1–5, Opus)")
        w("")
        w("| Dimension | Mean | Distribution (score:count) |")
        w("|---|---|---|")
        for d in DIMS:
            vals = [s[d] for s in scored]
            w(f"| {d} | {np.mean(vals):.2f} | {_dist(vals)} |")
        all4 = sum(1 for s in scored if all(s[d] >= 4 for d in DIMS))
        w("")
        w(f"**Headline: {all4}/{len(scored)} turns ({100 * all4 / len(scored):.0f}%) "
          f"scored ≥4 on every dimension.**")
        stt_sus = [s for s in scored if s.get("stt_suspect")]
        w(f"STT-suspect turns (garbled Japanese input): {len(stt_sus)}/{len(scored)} "
          f"({100 * len(stt_sus) / len(scored):.0f}%).")
        w("")

        # Repair effectiveness.
        rep = [s for s in scored if by_seq[s["seq"]].get("repaired")]
        unrep = [s for s in scored if not by_seq[s["seq"]].get("repaired")]
        if rep:
            w(f"Self-repair fired on {len(rep)}/{len(scored)} turns; "
              f"mean accuracy {np.mean([s['accuracy'] for s in rep]):.2f} (repaired) vs "
              f"{np.mean([s['accuracy'] for s in unrep]):.2f} (untouched).")
            w("")

        # Worst turns.
        def worst_key(s):
            return (min(s[d] for d in DIMS), sum(s[d] for d in DIMS))

        w("## Worst 10 turns (human-review shortlist)")
        w("")
        w("| seq | spk | acc | cmp | nat | ctx | STT? | JA | EN | issues |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for s in sorted(scored, key=worst_key)[:10]:
            r = by_seq[s["seq"]]
            issues = "; ".join(s.get("issues", []))[:160]
            w(f"| {s['seq']} | {r.get('speaker', '')} | {s['accuracy']} | {s['completeness']} "
              f"| {s['naturalness']} | {s['context_consistency']} "
              f"| {'⚠' if s.get('stt_suspect') else ''} "
              f"| {r['source'][:60]} | {r['english'][:60]} | {issues} |")
        w("")
    else:
        w("## Judge scores")
        w("")
        w("_No scores yet — run `python3 -m eval.judge " + run_dir + "`._")
        w("")

    # ── mechanical metrics (no judge needed) ─────────────────────────────────
    w("## Pipeline mechanics")
    w("")
    sims = [r["sim"] for r in turns if r.get("sim", -1) >= 0]
    speakers: dict[str, int] = {}
    for r in turns:
        if r.get("speaker"):
            speakers[r["speaker"]] = speakers.get(r["speaker"], 0) + 1
    w(f"- **Chunks**: {len(turns)} translated, {len(skipped)} skipped (no speech). "
      f"Durations: mean {np.mean([r['dur_s'] for r in turns]):.1f}s, "
      f"max {np.max([r['dur_s'] for r in turns]):.1f}s, "
      f"{sum(1 for r in turns if r['dur_s'] >= 13.9)} cut at the 14s cap.")
    w(f"- **Speakers**: {', '.join(f'{k}: {v}' for k, v in sorted(speakers.items())) or 'none detected'}."
      + (f" Match sims: mean {np.mean(sims):.2f}, min {np.min(sims):.2f}, "
         f"{sum(1 for s in sims if s < 0.72)} below the 0.72 threshold (new-speaker mints)."
         if sims else ""))
    ms = [r["ms"] for r in turns if r.get("ms")]
    if ms:
        w(f"- **Latency** (STT+diarize+translate+repair, offline): "
          f"mean {np.mean(ms) / 1000:.1f}s, p90 {np.percentile(ms, 90) / 1000:.1f}s per chunk.")
    reads = [r.get("cache_read", 0) for r in turns]
    if len(reads) > 1:
        hit = sum(1 for c in reads[1:] if c > 0)
        w(f"- **Prompt cache**: {hit}/{len(reads) - 1} chunks after the first read cached tokens "
          f"(mean {np.mean(reads[1:]):.0f} tokens/chunk read at 0.1×).")
    rep_n = sum(1 for r in turns if r.get("repaired"))
    w(f"- **Self-repair rate**: {rep_n}/{len(turns)} ({100 * rep_n / len(turns):.0f}%).")
    w("")

    out = os.path.join(run_dir, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written → {out}")


if __name__ == "__main__":
    main()
