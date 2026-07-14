# Architecture v2 — Redesign Proposal

**Status:** proposal · **Author:** system architect review · **Date:** 2026-07-12
**Scope:** the whole product (interpreter · interview copilot · notes), with the
redesign centered on the real-time audio front end, where architecture — not
tuning — is now the limiting factor.

---

## 1. Goals and non-goals

One user, three jobs, two opposing optimization targets:

| Mode | Job | Optimizes for | Today's verdict |
|---|---|---|---|
| Interpret | JA→EN live meeting translation + minutes | **Accuracy** | Working, measured (95–97% of turns ≥4/5 by LLM judge) |
| Interview | Glanceable answer hints during English interviews | **Speed** | Improved (4s cards) but structurally capped |
| Notes | English meeting summary/decisions/actions | Accuracy, async | Not built (designed) |

**Non-goals:** multi-tenant scale, mobile apps, additional cloud/ASR vendors
beyond the two already in use (Anthropic, OpenAI), offline operation.

**The architectural thesis:** the product's back half (LLM processing, knowledge,
persistence, UI) is sound and measured. The front half — how live audio becomes
text — was designed for the accuracy-first translator and then stretched to serve
the speed-first copilot. Every interview problem the user has felt (slow hints,
garbled questions, junk cards) traces to that front half. v2 replaces it once,
properly, and both modes get faster and simpler.

---

## 2. Current-state assessment

### 2.1 What is sound — keep unchanged

| Component | Evidence |
|---|---|
| Translator quality stack (context+glossary+summary+self-repair) | Eval on real audio: 95% of turns ≥4/5 pre-assembly, 97% post; naturalness 4.89→4.97 |
| Paper-&-ink UI + mode-first home | User-approved; views isolated per mode |
| Auth + data model (Firebase Auth, client-side Firestore writes, owner-only rules, `REQUIRE_AUTH` token check on the API) | Right-sized for a personal tool; backend keys protected |
| Profiles & knowledge layer (named profiles, PDF ingestion, GitHub blurbs, company-brief pre-warm) | Live-verified; company questions answer with **zero** mid-interview searches |
| Prompt-caching discipline | Profile prefix (>2048 tok) engages caching; known that thin prefixes silently don't |
| Eval harness (replay → Opus judge → report) | Produced every number in this document; budget-safe and resumable |
| Deployment (Cloud Run 1Gi + Firebase Hosting + Secret Manager) | Cheap, clean; the 1Gi constraint is a feature (forced the no-PyTorch diarizer) |

### 2.2 What is structurally fragile — the redesign target

The live-audio path is **five stacked workarounds** for one missing capability
(streaming ASR):

```
browser VAD chunking  →  WebM blobs  →  PyAV re-decode  →  batch STT
      →  sentence-assembly heuristics  →  one-in-flight backpressure
```

Each layer exists to patch the previous one, and each has produced a real,
observed failure:

1. **VAD chunking** waits 500–700ms of silence before anything is even sent, and
   cuts mid-sentence — which forced…
2. **Sentence assembly**, whose Japanese-tuned heuristic stalled unpunctuated
   English questions for up to **6 seconds** (root cause of "hints too slow");
   the English-aware fix is better but is still guesswork on fragments — which
   interacts with…
3. **One-in-flight backpressure**, which drops newest-wins chunks whenever the
   server is busy: in fast speech, *pieces of the interviewer's question are
   discarded* (observed as garbled/missing transcript).
4. **Mixed single-channel audio** makes "who is speaking" permanently heuristic:
   the MFCC diarizer labeled a real two-person conversation **54 turns / 1 turn**
   at threshold 0.72 and still under-split at 0.82. Question-gating by regex is
   the compensation — decent, but it misfires in both directions by design.
5. **Per-connection closure state** (assembler, speaker book, summary, brief) is
   unrecoverable: a dropped WebSocket mid-interview silently loses the session —
   the worst possible moment for it.

The message protocol also accreted eight ad-hoc shapes
(`config / audio-bytes / skipped / buffered / source+english / hint_pending /
hint_partial / hint_only / error`) with no envelope, no versioning, and no way
to resume.

**Conclusion:** further tuning of these layers has diminishing returns (we
measured the tuning ceiling: 4s hint cards). The layers should be *deleted*, not
tuned again.

---

## 3. Target architecture

```
   Browser                                   Cloud Run (session process)
┌───────────────────────┐            ┌──────────────────────────────────────┐
│ mic ────────┐         │            │        ┌─────────────────────┐       │
│             ├─ PCM ──►│── WS ─────►│ ASR    │  OpenAI Realtime    │       │
│ tab audio ──┘ frames  │  (binary,  │ gateway│  transcription      │       │
│  (interview: 2 chans) │   tagged   │        │  (1 socket/channel) │       │
│                       │   channel) │        └──────────┬──────────┘       │
│ ◄─ event stream ──────│◄─ WS ──────│    partial/final transcripts        │
│   (typed, versioned,  │  (JSON     │                  ▼                   │
│    resumable)         │   events)  │        SESSION ORCHESTRATOR          │
│                       │            │  transcript store · event ring      │
│ UI: transcript pane + │            │  mode processors:                   │
│ mode panel (hints /   │            │   interpret → translate+repair      │
│ translation / notes)  │            │   interview → gate → hint engine    │
└───────────────────────┘            │   notes     → rolling update_notes  │
                                     │  knowledge: profile · brief · gloss │
                                     └──────────────────┬──────────────────┘
                                                        ▼
                                     Anthropic API (Haiku hints / Sonnet
                                     translation / Haiku notes+summary)
                Firestore (client-written): sessions · profiles · history
```

### 3.A Capture — dual-channel, raw PCM

- Browser captures **mic** and (interview/notes) **tab audio** as *separate*
  streams; an AudioWorklet emits 16kHz mono PCM frames (~100ms), each prefixed
  with a 1-byte channel tag. No MediaRecorder, no WebM, no PyAV decode hop.
- **Channel identity replaces voice-guessing where it matters**: in interview
  mode, `tab` = interviewer (hint-eligible), `mic` = candidate (context only).
  The 54/1 diarization failure class disappears for this mode. Interpret mode
  (one shared mic, JA meeting) keeps the MFCC diarizer as today.
- Capture health is explicit: if the shared surface has no audio track, the UI
  blocks Start with a directive fix ("pick a *tab* and tick 'share tab audio'"),
  instead of silently degrading.

### 3.B ASR gateway — streaming transcription

- One OpenAI Realtime transcription socket per active channel
  (`gpt-4o-mini-transcribe` for interview/notes; model per mode — see §6 P4 for
  interpret). Server-side VAD and utterance segmentation happen at the ASR
  vendor; we receive **partial transcripts ~300–500ms behind speech** and
  punctuated finals at utterance end.
- Deleted outright: browser VAD tuning, chunk boundaries, `ChunkAssembler`,
  the 2s/6s silence flush, backpressure drops, `_to_wav`. The transcript can
  no longer lose words to a busy server because nothing queues behind an LLM
  call — audio flows continuously.
- Batch STT does not die: it remains the **eval-harness path** (deterministic
  replay) and the fallback flag if the Realtime API misbehaves.

### 3.C Session orchestrator — server-owned, resumable sessions

- A session is created by REST (`POST /session` → `session_id`) and *owns* all
  state currently trapped in the WS closure: transcript, mode processor state,
  speaker book, rolling summary, company brief, and an **event ring buffer**
  (last ~500 events).
- The WS becomes a dumb pipe: client connects with
  `?session_id=…&last_seq=N`; the orchestrator replays missed events from the
  ring and continues. A dropped connection mid-interview costs ~2s, not the
  session. Sessions survive 120s of disconnection before cleanup.
- Mode processors consume the transcript stream:
  - **interpret** — on utterance-final: translate (context window + glossary +
    summary) + self-repair, exactly today's measured pipeline.
  - **interview** — the question gate runs on *partials*: when the gate trips
    and the partial has been stable ~800ms, the hint engine **starts before the
    interviewer finishes the sentence** (it re-fires with the final text only if
    materially different). Hints stream token-partials as today.
  - **notes** — buffered transcript folded by `update_notes` (forced tool) every
    N utterances; final minutes pass at session end. (This unlocks the designed
    notes mode with no additional infrastructure.)

### 3.D Typed event protocol (replaces the 8 ad-hoc shapes)

Envelope — every server→client frame:

```json
{ "v": 1, "sid": "sess_9f2c", "seq": 417, "ts": 1786843201.442,
  "type": "hint.partial", "data": { …type-specific… } }
```

| `type` | `data` payload | Replaces |
|---|---|---|
| `transcript.partial` | `{utt_id, channel, text}` | *(new — live captions)* |
| `transcript.final` | `{utt_id, channel, speaker, text, lang}` | `{source,…}` |
| `translation.final` | `{utt_id, english, repaired, ms}` | `{english,…}` |
| `hint.pending` | `{utt_id}` | `hint_pending` |
| `hint.partial` | `{utt_id, gist, bullets}` | `hint_partial` |
| `hint.final` | `{utt_id, gist, bullets, angle, searched, ms}` | `hint_only` |
| `notes.updated` | `{summary[], decisions[], actions[], questions[]}` | *(new)* |
| `session.status` | `{state, detail}` (`warming_brief`, `live`, `resumed`…) | ad-hoc statuses |
| `error` | `{scope: "utterance"\|"session", message, retryable}` | `{error}` |

Client→server: binary PCM frames (channel-tagged) + a single JSON `control`
frame type (`{op: "start"|"end"|"set_config", …}`). All shapes versioned by `v`.

### 3.E Data & knowledge layer — unchanged

Firestore stays client-written under owner-only rules (`sessions`, `profiles`
with documents/repos, history). The orchestrator additionally posts a final
session record through the client at `session.end` (as today). Company-brief
pre-warm, profile compilation (24k cap), and glossary biasing carry over intact.

### 3.F Latency & cost budgets (the numbers v2 must hit)

Interview, question end → UI (targets, with measurement built into `hint.final.ms`
and event timestamps):

| Stage | v1 (measured 2026-07-11) | v2 target | Why it moves |
|---|---|---|---|
| Speech → transcript visible | 1.1–1.8s (after question end) | **live partials** during speech; final ≤0.5s after | streaming ASR |
| Question end → hint starts | ~1.9s (VAD+STT+gate) | **≤0s** (starts on stable partial) | gate on partials |
| Question end → first bullets | 2.7–3.2s | **≤1.5s** | generation already underway |
| Question end → complete card | 3.9–4.1s | **≤3.0s** | sum of above |
| Company question w/ search | 3.9s (brief) / ~8s (live search) | same; search stays rare | brief unchanged |

Interpret (accuracy-first — quality gates, not speed gates): utterance end →
translated line ≤4s p90; judge scores must not regress below the 97%/≥4-on-all
baseline when re-run through the eval harness.

Cost note: Realtime transcription bills continuously (~$0.003–0.006/audio-min
per channel, to be confirmed in the P2 spike) vs batch's per-chunk billing —
estimate **1.5–2×** STT cost for interview sessions (≈ cents/session either
way); LLM costs unchanged. The spike must produce the real number before P4
commits interpret mode to streaming.

---

## 4. What the output should look like

### 4.1 Storyboard — one interview question under v2

> Setup: profile "ML engineer" selected, role "ML engineer at Acme" pre-briefed,
> tab audio shared. Interviewer speaks over tab audio.

| t (s) | Interviewer audio | Transcript pane | Hints panel |
|---|---|---|---|
| 0.0 | "So I saw you have some open source…" | *(partial, grey, updating live)* "So I saw you have some open source" | — |
| 1.2 | "…work — tell me about your GitHub projects?" | partial extends | gate trips on "tell me about" → **card appears: "Thinking…"** (pulsing) |
| 1.8 | *(stops speaking)* | partial finalizes → ink-black line, tagged **Interviewer** | hint generation already ~1s in |
| 2.4 | | | gist replaces "Thinking…": *"Your GitHub projects"* + first two bullets fade in |
| 3.4 | | | card completes: 4 bullets + angle + `2.9s` meta. Bullets cite the *repo summaries from the profile* ("Real-time JA→EN translator — diarization under 1GiB, 95% eval") |
| 4+ | candidate answers (mic channel) | candidate's words transcribed, tagged **You**, grey rail | **no card** — mic channel is never hint-eligible |

The candidate starts answering ~2.5s after the question ends with bullets already
on screen — versus 4.1s (current) and ~11s (pre-overhaul).

### 4.2 Storyboard — one JA meeting turn under v2 (interpret)

| t (s) | Speaker audio | Transcript pane |
|---|---|---|
| 0.0–3.5 | 田中:「予算は承認されました。来月から新しい体制で進めます。」 | live JA partial (grey, indigo tint) grows word-by-word |
| 3.9 | *(pause)* | JA line finalizes under **Tanaka** rail |
| ~6.5 | | EN line appears beneath it: "The budget has been approved. We'll proceed under the new structure from next month." *(repair pass silently verified it)* |

Same paper-&-ink turn anatomy as today — the change is live partials instead of
dead air, and no possibility of a dropped fragment.

### 4.3 Event trace behind storyboard 4.1 (abridged)

```json
{"v":1,"sid":"sess_9f2c","seq":101,"ts":…,"type":"transcript.partial",
 "data":{"utt_id":"u17","channel":"tab","text":"So I saw you have some open source"}}
{"v":1,"sid":"sess_9f2c","seq":103,"ts":…,"type":"hint.pending","data":{"utt_id":"u17"}}
{"v":1,"sid":"sess_9f2c","seq":104,"ts":…,"type":"transcript.final",
 "data":{"utt_id":"u17","channel":"tab","speaker":"Interviewer",
         "text":"So I saw you have some open source work — tell me about your GitHub projects?","lang":"en"}}
{"v":1,"sid":"sess_9f2c","seq":106,"ts":…,"type":"hint.partial",
 "data":{"utt_id":"u17","gist":"Your GitHub projects",
         "bullets":["Real-time JA→EN translator — FastAPI + Claude"]}}
{"v":1,"sid":"sess_9f2c","seq":109,"ts":…,"type":"hint.final",
 "data":{"utt_id":"u17","gist":"Your GitHub projects",
         "bullets":["Real-time JA→EN translator — FastAPI + Claude",
                    "MFCC diarization under 1GiB Cloud Run",
                    "LLM-as-judge eval: 95% turns ≥4/5",
                    "Interview copilot: streamed hints, web-search grounding"],
         "angle":"Lead with the measurable quality result","searched":false,"ms":2870}}
```

### 4.4 Acceptance criteria

**Interview** — over a 20-question mock interview via tab audio:
- ≥80% of questions show a pending/partial card **before the interviewer
  finishes speaking**; 100% within 1s after.
- Complete cards ≤3.0s after question end (p90), `ms` recorded per card.
- Zero hint cards triggered by the candidate's own (mic) speech.
- Kill the network for 5s mid-interview: session resumes with full transcript,
  no card lost (`last_seq` replay).

**Interpret** — re-run the eval harness (replay path, batch STT) on the same
5-min window: judge scores ≥ the current 97%/≥4-on-all baseline; live-path spot
check shows JA partials rendering during speech.

**Protocol** — every frame validates against the v1 envelope schema; unknown
`type` is ignored by the client (forward compatibility).

---

## 5. Migration plan — four shippable phases, each behind a flag

| Phase | Delivers | Risk gate before next |
|---|---|---|
| **P1 — protocol + sessions** | Typed event envelope over the existing pipeline (adapter shims for old shapes); `POST /session`; ring buffer + `last_seq` resume | Reconnect test passes; zero behavior change otherwise |
| **P2 — streaming ASR spike → interview** | Realtime transcription behind `STREAMING_ASR=1`, interview mode only, single (mixed) channel; partials rendered; gate-on-partials | Latency table §3.F hit on the mock-interview script; real cost/min measured |
| **P3 — dual-channel capture** | AudioWorklet PCM + channel tags; tab=interviewer identity; capture-health gate in UI | Acceptance 4.4 interview criteria pass end-to-end |
| **P4 — interpret on streams + deletion** | Interpret evaluated on Realtime + `gpt-4o-transcribe`; if judge scores hold, migrate; then **delete** VAD chunking, assembler, backpressure, `_to_wav` live path | Eval harness shows no quality regression; batch path retained for eval/replay |

Rollback at every phase is a flag flip. The eval harness and the per-card `ms`
instrumentation are the referee throughout — same discipline that measured the
assembler win (55 fragments → 37 sentences) and the hint overhaul (11s → 4s).

---

## 6. Risks and open questions

| Risk | Mitigation |
|---|---|
| Realtime API cost surprises (continuous billing, 2 channels) | P2 spike measures real $/session before any commitment; batch fallback flag permanent |
| JA transcription quality on the streaming endpoint (interpret is accuracy-first) | P4 is *conditional on the eval harness* — if streaming JA scores worse, interpret simply stays on batch; the modes decouple cleanly |
| Chrome tab-audio capture variance (user picks a window; macOS quirks) | Explicit capture-health gate at Start (P3); mic-only degradation stays available for interpret |
| Cloud Run WS affinity/timeouts for long sessions | Already running 3600s timeout; session resume (P1) makes instance recycling survivable — this is *why* P1 comes first |
| Two ASR sockets + LLM calls within 1Gi | PCM passthrough is lighter than today's PyAV decode; validate memory headroom in P2/P3 load test |
| Eval-harness fidelity once live path streams | Harness keeps the batch replay path by design; add a recorded-PCM replay mode in P4 to eval the streaming path itself |

---

## 7. What deliberately does not change

The Claude processing core (Haiku hints + cached profile/brief prefix, Sonnet
translation + self-repair, Haiku summaries), the knowledge layer, the Firestore
model, auth, the visual design system, the eval methodology, and the deployment
topology. v2 is a front-half transplant: **one streaming audio path, one typed
protocol, one resumable session owner — everything downstream already works and
has the measurements to prove it.**
