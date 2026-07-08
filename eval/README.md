# Quality eval harness

Replays a real recording through the **production pipeline** (same functions the
server uses: gpt-4o-transcribe → diarization → cached-context translation +
self-repair), then scores every turn with an Opus judge and aggregates a report.

```bash
# 1. Replay a 5-minute window (≈ $1–2 in API calls, a few minutes)
python3 -m eval.replay --audio "recording.mp3" --start 300 --minutes 5

# 2. Judge (Opus, batched 10 turns/call, ≈ $1)
python3 -m eval.judge eval/runs/<timestamp>

# 3. Report (free, local)
python3 -m eval.report eval/runs/<timestamp>
cat eval/runs/<timestamp>/report.md
```

**Budget-safe**: every chunk/batch is saved the moment it completes. If credits
run out mid-run, the script stops cleanly, prints what completed vs. what
didn't, and `--resume <run_dir>` (replay) / re-running the same command (judge)
continues from the first missing item without re-paying for finished work.
`report.md` stamps its coverage, so a partial report is labeled as such.

**Live-meeting data**: sessions captured by the server with `SAVE_CHUNKS_DIR`
use the same format — point `eval.judge` / `eval.report` at those directories
directly.

Outputs live in `eval/runs/` (gitignored, like the source audio).
