# Speaker diarization models (optional)

Conversation mode labels **who is speaking** so the interpreter can resolve
Japanese dropped subjects and produce usable meeting minutes. See
`translator/diarizer.py`.

## Default backend — no model needed

By default the diarizer uses an **MFCC-statistics speaker signature** computed with
numpy/scipy (already in `requirements.txt`). It needs no model file and no extra
dependency, and clusters 2–4 speakers well on reasonably clean, one-mic audio
because each pause-delimited chunk is essentially one speaker's turn.

Tuning: `DIARIZE_THRESHOLD` (env, default `0.82`) — cosine similarity above which a
chunk joins an existing speaker. Raise it if distinct speakers get merged; lower it
if one speaker splits into several. (Was 0.72; raised after a real-audio eval showed
same-speaker re-match sims of 0.89–0.98 while 0.72 merged a genuine two-person
conversation into one label. Per-chunk `sim=` values are in the server logs.)

## Optional upgrade — ONNX d-vector

For stronger embeddings (noisier rooms, more speakers), drop in a CPU **ONNX**
speaker-embedding model and point the diarizer at it:

1. `pip install onnxruntime` (add it to `requirements.txt` for the deployed image).
2. Place the model here, e.g. `translator/models/speaker.onnx`.
3. Set `SPEAKER_MODEL_PATH=translator/models/speaker.onnx`.

`SpeakerEmbedder` will load it (CPU provider) and use it in place of the MFCC
signature; if the file or `onnxruntime` is missing it silently falls back to MFCC.

Candidate models (export to ONNX, CPU-friendly, no PyTorch at runtime):
- WeSpeaker ResNet / 3D-Speaker CAMPPlus (~25 MB) — expect 80-dim fbank input.

Note: the ONNX branch in `diarizer.py` feeds the MFCC feature matrix as a
`(1, frames, n)` tensor; adjust `_mfcc` / the feed dict to match your model's exact
input spec (feature type, dims, layout) before enabling it.

Weights are **not** committed — this directory only holds this README until a model
is added.
