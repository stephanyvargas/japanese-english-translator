# The quality architecture — why this translator is good

Live Japanese → English translation for meetings is deceptively hard. The naive
recipe — "record a few seconds, send to a speech-to-text model, send the text to an
LLM, print the English" — fails in ways that compound: it mishears speech, slices
sentences in half, forgets who is talking, renders the same name three different ways,
and hedges on subjects that Japanese simply omits.

The approach here treats quality as a **stack of layers, each removing one class of
error**. No single model does the heavy lifting; the wins come from how the pieces are
arranged. This document explains each layer, what it fixes, and the research it's built
on. Full citations are at the bottom.

The pipeline, end to end:

```
Microphone
  → VAD segmentation        (cut on pauses, not a blind timer)
  → gpt-4o-transcribe        (accurate STT) + hallucination guard
  → diarization              (which speaker is this chunk?)
  → Claude translation       (cached context + glossary + speaker + history)
  → self-repair              (fix only the chunks that actually need it)
```

---

## 1. Segmentation: cut on meaning, not on a clock

**The problem.** The single biggest source of bad translations isn't the translator —
it's *garbage in*. If you cut audio every N seconds, you routinely slice a sentence
mid-word. The STT model then transcribes a fragment, and the LLM faithfully translates a
fragment. Fixed-interval chunking is the original sin of naive real-time translation.

**What we do.** Conversation mode uses **voice-activity detection (VAD)** in the browser:
a WebAudio meter watches loudness and ends a chunk on a sustained pause *after* speech
(≈700 ms), or at a hard cap (≈14 s) so one long monologue still gets sent. Chunks now break
at natural clause boundaries.

**The frontier.** Production voice systems go one step further with *semantic* (dynamic)
endpointing: a fast VAD flags a pause (~500 ms), then a check on the partial transcript
decides whether to actually end the turn — extend the wait if the sentence is
syntactically incomplete ("I was walking down the…"), cut immediately if it's a complete
question. 2025 models like **Phoenix-VAD** and Pipecat's Smart-Turn do this with a small
LLM. We use the acoustic half today; the semantic half is the natural next upgrade (it
needs streaming partial transcripts — see §7).

> Research: dynamic/semantic endpointing ([Silero VAD + dynamic endpointing][vad-guide]),
> [Phoenix-VAD][phoenix-vad], [endpoint detection in streaming ASR][endpoint-asr].

---

## 2. Speech-to-text: a better model, and a guard against its lies

**The problem.** `whisper-1` is aging, and every STT model *hallucinates* on silence,
music, or noise — it fabricates plausible phrases from its training data (notoriously,
YouTube outros like "ご視聴ありがとうございました").

**What we do.** The default backend is **`gpt-4o-transcribe`**, which is markedly more
accurate on independent 2026 benchmarks and hallucinates far less than Whisper. Because
it doesn't expose per-segment confidence, we filter its output with a **text-level guard**:
collapse pathological verbatim repetition, and drop a *narrow* denylist of unambiguous
video-outro artifacts — deliberately narrow, so genuinely spoken politeness like
"ありがとうございました" is never discarded. `whisper-1` stays selectable and keeps its
original per-segment confidence filter as a fallback.

> Research: 2026 STT benchmarks and provider comparisons
> ([Future AGI][stt-futureagi], [Deepgram][stt-deepgram]). For non-English including
> Japanese, the Whisper / gpt-4o family remains best-in-class on raw accuracy, which is
> why we kept OpenAI STT rather than switching to a streaming vendor.

---

## 3. Terminology consistency: a session glossary

**The problem.** Proper nouns and jargon drift. Across chunks — and across the analysis
and translation passes — the same company name or acronym gets rendered three different
ways, which reads as sloppy and, worse, can change meaning.

**What we do.** A per-session **glossary** (`田中 => Tanaka`) is injected in three places:
the STT prompt (to bias *recognition* of the name), and the analyzer and translator
prompts (to pin the English *rendering*). One term, one spelling, everywhere.

---

## 4. Speaker diarization: solving Japanese's dropped subjects

**The problem — and why it's special for Japanese.** Japanese is *pro-drop*: it routinely
omits the grammatical subject. "行きます" is just "(someone) will go." Without knowing
**who is speaking**, the translator has to guess "I / you / we / he" — and a wrong guess
silently corrupts the meaning. It also makes the transcript useless as minutes: a wall of
unattributed lines.

**What we do.** Knowing the speaker is the single biggest remaining lever, so each chunk is
labeled with a speaker. Crucially, we exploit a property of our own design: because §1 cuts
on pauses, **each chunk is essentially one speaker's turn**. That means we don't need
heavyweight continuous-stream diarization (pyannote and friends pull PyTorch and don't fit
a 1 GiB / 1-vCPU Cloud Run) — we only need to answer "which speaker is *this chunk*?"

So the default diarizer is deliberately lightweight: compute an **MFCC-statistics speaker
signature** (mean + std of the cepstral coefficients over the chunk, energy coefficient
dropped, mean-centered so the cosine margin isn't compressed), then do **online cosine
clustering** against per-session speaker centroids — nearest above a threshold wins,
otherwise a new speaker is minted. Labels are `Speaker 1/2/3`, mapped to real names from a
Participants field in first-appearance order. No new dependency, no model file, <100 ms per
chunk. A stronger ONNX d-vector model is a documented drop-in upgrade for noisier rooms.

The speaker then flows into the prompt: history lines and the new chunk are tagged
`[Tanaka] …`, with an instruction telling Claude to use the tags to resolve dropped
subjects (and *not* print them). "行きます" from a tagged speaker becomes "Tanaka will go,"
not a guessed pronoun.

> Research: modern diarization ([pyannote Community-1][pyannote], [SpeakerLM][speakerlm]).
> We chose a chunk-aligned lightweight signature over these specifically because our
> segmentation already isolates turns and our deploy target is memory-constrained.

---

## 5. Context-aware translation: the LLM as a simultaneous interpreter

**The problem.** A fragment translated in isolation is often wrong — pronouns,
topic references, and incomplete clauses can only be resolved with context.

**What we do.** Each chunk is translated *with* the recent history, the glossary, the
meeting context, and the speaker labels. The prompt explicitly instructs: if a chunk is an
incomplete clause cut off at a pause, translate what's there and let the next chunk
continue it — don't invent an ending. This mirrors the research finding that LLMs are
strong **zero-shot, context-aware simultaneous translators**, and that *adaptive*
read/write policies (translate once a semantic unit is complete) beat fixed "wait-k"
chunking that translates on a schedule.

> Research: [LLMs are zero-shot context-aware simultaneous translators][zeroshot],
> [LLMs achieve high-quality simultaneous MT][simt-acl], [syntax-aware chunking
> (SASST)][sasst], [human-parity SiMT via an LLM agent][human-parity].

---

## 6. Prompt caching: making rich context affordable

**The problem.** Everything above wants *more* standing context — full glossary, agenda,
participant roster, meeting notes. But if you resend and re-bill that on every chunk, cost
and latency explode, so the naive move is to keep context thin. That directly caps quality.

**What we do.** The session-constant part of the prompt (role, rules, glossary, meeting
context, participants) is sent as a **cached prefix** (`cache_control: ephemeral`). After
the first chunk it's billed at ~0.1×, so a *rich* context is essentially free per chunk;
only the rolling history and the new chunk are freshly processed. Caching and rich context
reinforce each other — the cache is what lets the context be big.

> Research: [Anthropic prompt caching announcement][cache-news] and
> [prompt caching docs][cache-docs].

---

## 7. Quality loops: repair, review, and back-translation

**Conversation mode** runs a cheap **self-repair** pass after each translation: a quick
check that rewrites the English *only* when it drops meaning or mishandles a
referent/number/name — clean chunks are left untouched, so we don't pay latency for
nothing.

**Text / single-shot mode** runs the fuller pipeline: linguistic **analysis** → translation
with adaptive thinking → bilingual **review** → up to two **refine** rounds (triggered on
accuracy < 9 *or* naturalness < 8), plus an opt-in **back-translation** drift check that
translates the English back to Japanese and flags any meaning that shifted.

---

## What we deliberately did *not* do (yet)

- **Streaming ASR vendor (Deepgram / AssemblyAI / Speechmatics).** These give partial
  transcripts (enabling §1's semantic endpointing), word timings, keyterm boosting, and
  built-in diarization — but add a vendor, cost, and a Japanese-accuracy bake-off. We kept
  OpenAI STT for accuracy on Japanese and simplicity.
- **End-to-end speech-to-speech** (Meta SeamlessStreaming, Google's real-time S2S). Lower
  latency, but a cascade (ASR → MT) gives us far more control over terminology, speaker
  attribution, and the quality loops above — which matter more for meetings than shaving a
  second.

> Landscape: [real-time speech translation architecture & trade-offs][forasoft],
> [Google real-time S2S][google-s2s], [Meta SeamlessStreaming][seamless].

---

## References

**Segmentation & endpointing**
- [Silero VAD & dynamic endpointing — implementation guide][vad-guide]
- [Phoenix-VAD: Streaming Semantic Endpoint Detection (arXiv 2509.20410)][phoenix-vad]
- [Improving endpoint detection in end-to-end streaming ASR (arXiv 2505.17070)][endpoint-asr]

**Speech-to-text**
- [Best Speech-to-Text APIs in 2026 — benchmarks (Future AGI)][stt-futureagi]
- [Best Speech-to-Text APIs in 2026 (Deepgram)][stt-deepgram]

**Diarization**
- [pyannote Community-1 — open-source diarization][pyannote]
- [SpeakerLM: End-to-End Diarization & Recognition with Multimodal LLMs (arXiv 2508.06372)][speakerlm]

**Simultaneous / context-aware translation**
- [LLMs Are Zero-Shot Context-Aware Simultaneous Translators (arXiv 2406.13476)][zeroshot]
- [LLMs Can Achieve High-quality Simultaneous MT as Efficiently as Offline (ACL Findings 2025)][simt-acl]
- [SASST: Syntax-Aware Chunking + LLMs for Simultaneous Speech Translation (arXiv 2508.07781)][sasst]
- [Towards Human Parity on End-to-end Simultaneous Speech Translation via LLM Agent (arXiv 2407.21646)][human-parity]

**Prompt caching**
- [Prompt caching with Claude — announcement][cache-news]
- [Prompt caching — Claude platform docs][cache-docs]

**Landscape / systems**
- [Real-Time Speech Translation: Architecture & Trade-Offs 2026 (Fora Soft)][forasoft]
- [Real-time speech-to-speech translation (Google Research)][google-s2s]
- [Meta SeamlessStreaming / seamless_communication][seamless]

[vad-guide]: https://rajatpandit.com/agentic-ai/real-time-audio-vad/
[phoenix-vad]: https://arxiv.org/pdf/2509.20410
[endpoint-asr]: https://arxiv.org/pdf/2505.17070
[stt-futureagi]: https://futureagi.com/blog/speech-to-text-apis-in-2026-benchmarks-pricing-developer-s-decision-guide/
[stt-deepgram]: https://deepgram.com/learn/best-speech-to-text-apis-2026
[pyannote]: https://www.pyannote.ai/blog/community-1
[speakerlm]: https://arxiv.org/pdf/2508.06372
[zeroshot]: https://arxiv.org/pdf/2406.13476
[simt-acl]: https://aclanthology.org/2025.findings-acl.1045.pdf
[sasst]: https://arxiv.org/html/2508.07781v1
[human-parity]: https://arxiv.org/pdf/2407.21646
[cache-news]: https://www.anthropic.com/news/prompt-caching
[cache-docs]: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
[forasoft]: https://www.forasoft.com/learn/real-time-speech-translation-live-video
[google-s2s]: https://research.google/blog/real-time-speech-to-speech-translation/
[seamless]: https://github.com/facebookresearch/seamless_communication
