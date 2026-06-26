// ── DOM refs ─────────────────────────────────────────────────────────────────

const sourceLang   = document.getElementById('sourceLang');
const modelSel     = document.getElementById('model');
const contextInput = document.getElementById('context');
const backendUrl   = document.getElementById('backendUrl');

const tabs         = document.querySelectorAll('.tab');
const panels       = document.querySelectorAll('.panel');

const startBtn     = document.getElementById('startBtn');
const stopBtn      = document.getElementById('stopBtn');
const convStatus   = document.getElementById('convStatus');

const textInput    = document.getElementById('textInput');
const translateBtn = document.getElementById('translateBtn');
const showNotes    = document.getElementById('showNotes');
const textStatus   = document.getElementById('textStatus');

const output       = document.getElementById('output');
const clearBtn     = document.getElementById('clearBtn');

// ── Tab switching ─────────────────────────────────────────────────────────────

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    panels.forEach(p => p.classList.add('hidden'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.remove('hidden');
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function apiBase() {
  return backendUrl.value.replace(/\/$/, '');
}

function appendChunk({ source, english, langTag, notes, error }) {
  const div = document.createElement('div');
  div.className = 'chunk';
  if (error) {
    div.innerHTML = `<span class="error">Error: ${escHtml(error)}</span>`;
  } else {
    if (source) div.innerHTML += `<span class="source">[${escHtml(langTag || '??')}] ${escHtml(source)}</span>`;
    if (english) div.innerHTML += `<span class="english">[EN] ${escHtml(english)}</span>`;
    if (notes && notes.length) {
      div.innerHTML += notes.map(n => `<span class="notes">* ${escHtml(n)}</span>`).join('');
    }
  }
  output.appendChild(div);
  output.scrollTop = output.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Text mode ─────────────────────────────────────────────────────────────────

translateBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  translateBtn.disabled = true;
  textStatus.textContent = 'Translating...';

  try {
    const res = await fetch(`${apiBase()}/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        model: modelSel.value,
        source_lang: sourceLang.value,
        context: contextInput.value.trim(),
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      appendChunk({ error: err });
      textStatus.textContent = 'Error';
      return;
    }

    const data = await res.json();
    appendChunk({
      source: data.source_text,
      english: data.english_text,
      langTag: sourceLang.value.toUpperCase(),
      notes: showNotes.checked ? data.translator_notes : [],
    });
    textStatus.textContent = '';
  } catch (err) {
    appendChunk({ error: err.message });
    textStatus.textContent = 'Error';
  } finally {
    translateBtn.disabled = false;
  }
});

// ── Audio → WAV conversion ────────────────────────────────────────────────────
// MediaRecorder produces WebM/Ogg/MP4 depending on browser. We decode with
// AudioContext (which handles all of them) then repack as WAV so Whisper always
// receives a format it can decode.

// Reuse a decode context. sampleRate is intentionally omitted here — we do
// the resampling explicitly with OfflineAudioContext so it is guaranteed.
let _decodeCtx = null;
function getDecodeCtx() {
  if (!_decodeCtx || _decodeCtx.state === 'closed') _decodeCtx = new AudioContext();
  return _decodeCtx;
}

// Returns an ArrayBuffer (16kHz mono WAV) or null if the chunk is too short.
async function blobToWav(blob) {
  const arrayBuffer = await blob.arrayBuffer();

  // Step 1: decode whatever format MediaRecorder produced
  const decoded = await getDecodeCtx().decodeAudioData(arrayBuffer);
  if (decoded.duration < 0.5) return null; // Whisper rejects sub-second clips

  // Step 2: resample to 16kHz mono via OfflineAudioContext
  const TARGET_RATE = 16000;
  const numFrames = Math.ceil(decoded.duration * TARGET_RATE);
  const offline = new OfflineAudioContext(1, numFrames, TARGET_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start(0);
  const resampled = await offline.startRendering();

  return audioBufferToWav(resampled);
}

function audioBufferToWav(buffer) {
  const sampleRate = buffer.sampleRate;
  const samples = buffer.getChannelData(0); // mono
  const int16 = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, samples[i] * 32768));
  }

  const wavBuffer = new ArrayBuffer(44 + int16.byteLength);
  const v = new DataView(wavBuffer);
  const s = (o, str) => { for (let i = 0; i < str.length; i++) v.setUint8(o + i, str.charCodeAt(i)); };

  s(0,  'RIFF');  v.setUint32(4,  36 + int16.byteLength, true);
  s(8,  'WAVE');  s(12, 'fmt ');
  v.setUint32(16, 16, true);          // chunk size
  v.setUint16(20, 1,  true);          // PCM
  v.setUint16(22, 1,  true);          // mono
  v.setUint32(24, sampleRate, true);
  v.setUint32(28, sampleRate * 2, true); // byte rate
  v.setUint16(32, 2,  true);          // block align
  v.setUint16(34, 16, true);          // bits per sample
  s(36, 'data'); v.setUint32(40, int16.byteLength, true);
  new Int16Array(wavBuffer, 44).set(int16);

  return wavBuffer;
}

// ── Conversation mode ─────────────────────────────────────────────────────────

let activeStream = null;
let ws = null;
let active = false;
const INTERVAL_MS = 8000;

startBtn.addEventListener('click', startConversation);
stopBtn.addEventListener('click', stopConversation);

async function startConversation() {
  try {
    activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    convStatus.textContent = 'Mic access denied';
    return;
  }

  const wsUrl = apiBase().replace(/^http/, 'ws') + '/ws/conversation';
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      model: modelSel.value,
      source_lang: sourceLang.value,
      lang_name: sourceLang.options[sourceLang.selectedIndex].text,
      context: contextInput.value.trim(),
    }));
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.skipped) return;
    if (msg.error) { appendChunk({ error: msg.error }); return; }
    appendChunk({ source: msg.source, english: msg.english, langTag: sourceLang.value.toUpperCase() });
  };

  ws.onerror = () => { convStatus.textContent = 'WebSocket error'; };
  ws.onclose = () => { if (active) stopConversation(); };

  active = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  convStatus.innerHTML = '<span class="recording-dot"></span>Recording...';

  // Cycle stop/start so each recording is a complete, self-contained WebM file.
  // Using timeslice produces headerless continuation chunks that Whisper rejects.
  recordCycle();
}

function recordCycle() {
  if (!active || !activeStream) return;

  const rec = new MediaRecorder(activeStream);

  rec.ondataavailable = async (e) => {
    if (!e.data || e.data.size === 0) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      const wav = await blobToWav(e.data);
      if (!wav) { convStatus.textContent = 'Chunk too short — skipped'; return; }
      convStatus.textContent = `Sent ${(wav.byteLength / 1024).toFixed(1)} KB WAV — translating...`;
      ws.send(wav);
    } catch (err) {
      console.warn('Audio conversion failed:', err);
      convStatus.textContent = 'Audio conversion error — skipping chunk';
    }
  };

  rec.onstop = () => { if (active) recordCycle(); };

  rec.start();
  setTimeout(() => { if (rec.state === 'recording') rec.stop(); }, INTERVAL_MS);
}

function stopConversation() {
  active = false;
  if (activeStream) {
    activeStream.getTracks().forEach(t => t.stop());
    activeStream = null;
  }
  if (ws) { ws.close(); ws = null; }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  convStatus.textContent = 'Stopped';
}

// ── Clear output ──────────────────────────────────────────────────────────────

clearBtn.addEventListener('click', () => { output.innerHTML = ''; });
