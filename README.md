# Japanese-to-English Translator

Quality-focused Japanese-to-English translation system with microphone input. Speak Japanese, get English output — continuously, with full conversational context.

> **Why this design?** For a deep dive on the quality architecture — segmentation,
> terminology consistency, speaker diarization, prompt caching, and the self-repair /
> review loops — plus the research and papers behind it, see
> [`docs/quality-architecture.md`](docs/quality-architecture.md).

## How it works

**Conversation mode** (default, the meeting surface): audio is captured continuously and
segmented on natural pauses (voice-activity detection), so chunks break at clause
boundaries instead of mid-word. Each chunk is transcribed by OpenAI `gpt-4o-transcribe`,
tagged with a speaker (diarization), and translated by Claude with the recent history,
a session glossary, and the meeting context. A cheap self-repair pass fixes a chunk only
when it actually drops meaning.

**Single-shot mode** (`--once`, `--text`): a quality pipeline — linguistic analysis →
translation with adaptive thinking → bilingual review → up to two refinement rounds
(triggered when accuracy < 9 or naturalness < 8), with an optional back-translation
drift check.

```
Microphone → VAD segmentation → gpt-4o-transcribe → diarization
          → Claude (context + glossary + speaker → translate → self-repair)
```

### Transcription hallucination filtering

When speech-to-text receives silence, music, or noise it tends to fabricate common
phrases. `gpt-4o-transcribe` (the default) is filtered with a text-level guard —
verbatim-repetition collapse plus a denylist of unambiguous video-outro artifacts
(kept narrow so real speech like "ありがとうございました" is never dropped). The legacy
`whisper-1` backend remains selectable and keeps its per-segment confidence filter
(`no_speech_prob > 0.6`, `avg_logprob < -1.0`, `compression_ratio > 2.4`).

### Terminology consistency (glossary)

An optional **Key terms / names** field pins how names and jargon are rendered
(`田中 => Tanaka`). The glossary is injected into the transcription prompt (to bias
recognition) and into the analysis and translation prompts (to keep the English
rendering identical across chunks and passes).

## Requirements

- Python 3.10+
- PortAudio (for microphone input): `sudo apt install libportaudio2` on Debian/Ubuntu
- An Anthropic API key
- An OpenAI API key (for `gpt-4o-transcribe` speech-to-text)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

```bash
# Continuous conversation mode (default)
python3 main.py

# Adjust how often audio is processed (default: every 8 seconds)
python3 main.py --interval 12

# Translate a single spoken utterance, then exit
python3 main.py --once --notes

# Translate typed text, then exit
python3 main.py --text "おつかれさまでした。" --notes
```

### Model selection

```bash
python3 main.py --model sonnet   # claude-sonnet-4-6 (default) — balanced cost/quality
python3 main.py --model opus     # claude-opus-4-8 — highest quality
python3 main.py --model haiku    # claude-haiku-4-5 — fastest, cheapest
```

Full model IDs are also accepted: `--model claude-sonnet-4-6`.

## Web app

The same engine is exposed as a browser app: a FastAPI backend (`server.py`) and a
static frontend (`frontend/`). The browser captures the mic and streams audio to the
backend over a WebSocket; transcription and translation run server-side.

```bash
# Run the backend locally (serves the frontend at http://localhost:8000 too)
uvicorn server:app --reload --port 8000
```

- `POST /translate` — full quality pipeline for typed text
- `WS /ws/conversation` — real-time conversation mode (audio in, translations out)
- `GET /health` — health probe

### Speaker diarization (conversation mode)

Because conversation mode cuts audio on natural pauses, each chunk is essentially one
speaker's turn. The backend labels each chunk with a speaker (`Speaker 1/2/…`, or the
names from the **Participants** field in first-appearance order) so the interpreter can
resolve Japanese dropped subjects and the transcript reads as usable minutes. The
default backend is a lightweight MFCC-statistics speaker signature (numpy/scipy — no
extra dependency, no model file, fits the 1Gi Cloud Run); an optional ONNX d-vector
upgrade is documented in [`translator/models/README.md`](translator/models/README.md).
Toggle it off with the **Detect speakers** checkbox; overlapping speech on one shared
mic is the main failure mode.

### Rich context via prompt caching

The session-constant part of the conversation prompt — role, rules, glossary, meeting
context, and participant roster — is sent as a cached prefix (`cache_control: ephemeral`),
so it is billed at ~0.1× on every chunk after the first. That makes a *rich* standing
context affordable without adding per-chunk latency or cost; only the rolling history and
the new chunk are re-processed each time.

## Deployment

The app is deployed with the **frontend on Firebase Hosting** and the **backend on
Google Cloud Run** (region `asia-northeast1`). API keys are stored in Google Secret
Manager, not in the image.

| Surface | URL |
|---------|-----|
| App (frontend) | https://japanese-translator-501010.web.app |
| Backend API | https://translator-backend-1029193548741.asia-northeast1.run.app |

**Notes**

- The browser connects directly to the Cloud Run URL for the WebSocket (`wss://`).
  Firebase Hosting does not proxy WebSocket upgrades, so the frontend's *Backend URL*
  field points straight at Cloud Run.
- The backend's allowed CORS origins are set via the `ALLOWED_ORIGINS` env var
  (comma-separated); in production it is set to the Firebase Hosting domains.
- Cloud Run scales to zero, so the first request after idle has a few-second cold
  start. Add `--min-instances 1` to keep one warm.

### Redeploy

```bash
# Backend → Cloud Run (rebuilds the container from the Dockerfile)
gcloud run deploy translator-backend \
  --source . \
  --region asia-northeast1 \
  --allow-unauthenticated \
  --timeout 3600 \
  --memory 1Gi \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest,OPENAI_API_KEY=openai-api-key:latest \
  --set-env-vars "^|^ALLOWED_ORIGINS=https://japanese-translator-501010.web.app,https://japanese-translator-501010.firebaseapp.com"

# Frontend → Firebase Hosting
firebase deploy --only hosting

# Tail backend logs (timestamped per-chunk timing)
gcloud run services logs read translator-backend --region asia-northeast1 --limit 50
```

## Project structure

```
translator/
├── audio.py         # Microphone recording; AudioCapture for continuous threaded capture
├── transcriber.py   # gpt-4o-transcribe (default) + text-level guard; whisper-1 fallback
├── diarizer.py      # Per-chunk speaker embedding + online clustering (who spoke)
├── glossary.py      # Per-session terminology store (names/jargon consistency)
├── analyzer.py      # Claude: detect domain, formality level, keigo, cultural references
├── translator.py    # Claude: translation with adaptive thinking
├── reviewer.py      # Claude: bilingual quality review, structured critique
├── verifier.py      # Claude: opt-in back-translation drift check
├── pipeline.py      # Orchestrates all steps; run_conversation() and run()
├── prompts.py       # System prompts for each Claude role
├── models.py        # Pydantic models: AnalysisResult, ReviewResult, DriftResult, FinalOutput
└── models/          # Optional ONNX speaker-embedding weights (see models/README.md)
server.py            # FastAPI backend: /translate, WS /ws/conversation, /health
main.py              # CLI entry point
```

## API keys

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | All Claude calls (analysis, translation, review, self-repair) |
| `OPENAI_API_KEY` | `gpt-4o-transcribe` speech-to-text (and `whisper-1` fallback) |
