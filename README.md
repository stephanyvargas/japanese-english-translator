# Japanese-to-English Translator

Quality-focused Japanese-to-English translation system with microphone input. Speak Japanese, get English output — continuously, with full conversational context.

## How it works

**Conversation mode** (default): A background thread records audio continuously so nothing is lost while the model is translating. Every 8 seconds, accumulated audio is sent to OpenAI Whisper for transcription, then to Claude for translation. Each chunk receives the full conversation history so context carries across utterances.

**Single-shot mode** (`--once`, `--text`): A three-step quality pipeline — linguistic analysis, translation with adaptive thinking, bilingual review — with an optional refinement pass if the review score is below 8/10.

```
Microphone → Whisper (transcription) → Claude (analysis → translation → review)
```

### Whisper hallucination filtering

When Whisper receives silence, music, or noise it tends to fabricate common phrases from its training data. Each transcription response is inspected at the segment level and segments are discarded if:

- `no_speech_prob > 0.6` — likely silence or background noise
- `avg_logprob < -1.0` — model confidence is too low
- `compression_ratio > 2.4` — output is suspiciously repetitive

## Requirements

- Python 3.10+
- PortAudio (for microphone input): `sudo apt install libportaudio2` on Debian/Ubuntu
- An Anthropic API key
- An OpenAI API key (for Whisper transcription)

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

## Project structure

```
translator/
├── audio.py         # Microphone recording; AudioCapture for continuous threaded capture
├── transcriber.py   # OpenAI Whisper transcription with hallucination filtering
├── analyzer.py      # Claude: detect domain, formality level, keigo, cultural references
├── translator.py    # Claude: translation with adaptive thinking
├── reviewer.py      # Claude: bilingual quality review, structured critique
├── pipeline.py      # Orchestrates all steps; run_conversation() and run()
├── prompts.py       # System prompts for each Claude role
└── models.py        # Pydantic models: AnalysisResult, ReviewResult, FinalOutput
main.py              # CLI entry point
```

## API keys

| Key | Used for |
|-----|----------|
| `ANTHROPIC_API_KEY` | All Claude calls (analysis, translation, review) |
| `OPENAI_API_KEY` | Whisper speech-to-text transcription |
