"""LLM-as-judge scoring for a replay (or live SAVE_CHUNKS_DIR) session.

Batches 10 consecutive turns per Opus call, giving each batch the previous
batch as read-only context so cross-turn consistency (referents, terminology,
speaker plausibility) is judged, not just per-sentence adequacy. Scores come
back through a forced tool. Appends per batch — interruption-safe, resumable.

Usage:
  python3 -m eval.judge eval/runs/20260708-141900 [--model claude-opus-4-8]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic  # noqa: E402

from eval.common import (  # noqa: E402
    RunStopped, append_jsonl, call_with_budget_guard, load_jsonl, write_run_state,
)

BATCH = 10

_SYSTEM = """\
You are a senior Japanese-English conference interpreter grading the output of a
real-time meeting translation system. You receive consecutive conversation turns:
for each, the Japanese transcript (from speech-to-text — it may contain
misrecognitions) and the system's English rendering, with speaker labels when
available. Earlier turns are provided as context.

Score each turn's English AS A RENDERING OF THE JAPANESE IN ITS CONVERSATIONAL
CONTEXT (not as isolated text), on four dimensions, each 1-5:

- accuracy: meaning fidelity. 5 = publishable minutes; 3 = core meaning preserved
  but nuance/register/detail flawed; 1 = meaning lost or inverted.
- completeness: 5 = nothing dropped or invented; 3 = minor omission/addition;
  1 = major content dropped or hallucinated.
- naturalness: 5 = fluent conversational English; 3 = understandable but stilted;
  1 = broken English.
- context_consistency: referents, terminology, and subject attribution consistent
  with the surrounding turns (dropped Japanese subjects resolved to the right
  person). 5 = fully consistent; 1 = contradicts context.

Also set stt_suspect=true when the JAPANESE side itself looks like a speech
recognition error (implausible, garbled, or contextually impossible Japanese) —
that separates transcription failures from translation failures. When
stt_suspect is true, still score the translation against the Japanese as given.

List concrete issues (in English, brief) for any turn scoring below 4 on any
dimension. Submit with the submit_scores tool.\
"""

_TOOL = {
    "name": "submit_scores",
    "description": "Submit per-turn quality scores for this batch.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "seq": {"type": "integer"},
                        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
                        "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "context_consistency": {"type": "integer", "minimum": 1, "maximum": 5},
                        "stt_suspect": {"type": "boolean"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["seq", "accuracy", "completeness", "naturalness",
                                 "context_consistency", "stt_suspect", "issues"],
                },
            },
        },
        "required": ["scores"],
    },
}


def _fmt_turn(r: dict) -> str:
    who = f" [{r['speaker']}]" if r.get("speaker") else ""
    return f"#{r['seq']}{who}\nJA: {r['source']}\nEN: {r['english']}"


def main() -> None:
    p = argparse.ArgumentParser(description="Judge a replay session with Opus")
    p.add_argument("run_dir")
    p.add_argument("--model", default="claude-opus-4-8")
    args = p.parse_args()
    run_dir = args.run_dir.rstrip("/")

    session = [r for r in load_jsonl(os.path.join(run_dir, "session.jsonl"))
               if r.get("source") and r.get("english")]
    if not session:
        sys.exit(f"No scoreable turns in {run_dir}/session.jsonl")

    scores_path = os.path.join(run_dir, "scores.jsonl")
    scored = {r["seq"] for r in load_jsonl(scores_path)}
    todo = [r for r in session if r["seq"] not in scored]
    print(f"{len(session)} turns, {len(scored)} already scored, {len(todo)} to judge "
          f"({(len(todo) + BATCH - 1) // BATCH} batches of ≤{BATCH})")

    client = anthropic.Anthropic()
    stopped_reason = ""
    judged = len(scored)

    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        first_idx = session.index(batch[0])
        context_turns = session[max(0, first_idx - BATCH):first_idx]
        ctx = ("Context (already judged, do not score):\n"
               + "\n\n".join(_fmt_turn(r) for r in context_turns) + "\n\n"
               if context_turns else "")
        user_msg = (f"{ctx}Score these {len(batch)} turns "
                    f"(seq {batch[0]['seq']}–{batch[-1]['seq']}):\n\n"
                    + "\n\n".join(_fmt_turn(r) for r in batch))

        try:
            def judge_batch():
                with client.messages.stream(
                    model=args.model,
                    max_tokens=4096,
                    thinking={"type": "adaptive"},
                    system=_SYSTEM,
                    tools=[_TOOL],
                    tool_choice={"type": "tool", "name": "submit_scores"},
                    messages=[{"role": "user", "content": user_msg}],
                ) as stream:
                    msg = stream.get_final_message()
                block = next(b for b in msg.content if b.type == "tool_use")
                return block.input["scores"]

            results = call_with_budget_guard(judge_batch)
        except RunStopped as stop:
            stopped_reason = stop.reason
            break

        valid = {r["seq"] for r in batch}
        for s in results:
            if s.get("seq") in valid:
                s["judged_at"] = time.time()
                append_jsonl(scores_path, s)
                judged += 1
        print(f"batch {b // BATCH + 1}: scored seq {batch[0]['seq']}–{batch[-1]['seq']}", flush=True)

    write_run_state(run_dir, stage="judge", judged_turns=judged,
                    scoreable_turns=len(session),
                    judge_stopped_reason=stopped_reason or "completed")
    print("\n" + "─" * 60)
    if stopped_reason:
        print(f"STOPPED EARLY: {stopped_reason}")
        print(f"Judged {judged}/{len(session)} turns — all completed batches are saved.")
        print(f"Resume with:\n  python3 -m eval.judge {run_dir}")
    else:
        print(f"Done: {judged}/{len(session)} turns scored → {scores_path}")
        print(f"Next:\n  python3 -m eval.report {run_dir}")


if __name__ == "__main__":
    main()
