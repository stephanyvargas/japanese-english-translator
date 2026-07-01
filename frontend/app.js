// ── DOM refs ─────────────────────────────────────────────────────────────────

const sourceLang   = document.getElementById('sourceLang');
const modelSel     = document.getElementById('model');
const contextInput = document.getElementById('context');
const glossaryEl   = document.getElementById('glossary');
const verifyEl     = document.getElementById('verify');
const participantsEl = document.getElementById('participants');
const diarizeEl    = document.getElementById('diarize');
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

function nowStamp() {
  return new Date().toLocaleTimeString('en-GB'); // HH:MM:SS
}

// Stable per-speaker color, assigned in first-appearance order.
const speakerColors = ['#4a9dff', '#4ecb7a', '#e0a13a', '#c471d6', '#e0655a', '#3ec3c3'];
const speakerIndex = {};
function speakerTag(speaker) {
  if (!speaker) return '';
  if (!(speaker in speakerIndex)) speakerIndex[speaker] = Object.keys(speakerIndex).length;
  const color = speakerColors[speakerIndex[speaker] % speakerColors.length];
  return `<span class="speaker" style="color:${color}">${escHtml(speaker)}</span> `;
}

function appendChunk({ source, english, langTag, speaker, notes, error, lagMs }) {
  const div = document.createElement('div');
  div.className = 'chunk';
  const ts = nowStamp();
  const lag = lagMs != null ? ` (${(lagMs / 1000).toFixed(1)}s)` : '';
  if (error) {
    div.innerHTML = `<span class="error">[${ts}] Error: ${escHtml(error)}</span>`;
  } else {
    if (source) div.innerHTML += `<span class="source">[${ts}] ${speakerTag(speaker)}[${escHtml(langTag || '??')}] ${escHtml(source)}</span>`;
    if (english) div.innerHTML += `<span class="english">[EN]${lag} ${escHtml(english)}</span>`;
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
        glossary: glossaryEl.value.trim(),
        verify: verifyEl.checked,
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

// ── Conversation mode ─────────────────────────────────────────────────────────

let activeStream = null;
let ws = null;
let active = false;

// Voice-activity segmentation: end a chunk on a natural pause instead of a blind
// timer, so sentences aren't sliced mid-word (the main cause of misheard STT).
const VAD = {
  POLL_MS: 100,        // how often we sample loudness
  RMS_THRESHOLD: 0.015, // above this = speech
  SILENCE_MS: 700,     // sustained silence after speech ends a chunk
  MIN_SPEECH_MS: 300,  // require this much speech before a pause counts
  MAX_MS: 14000,       // hard cap so one long utterance still gets sent
};

// WebAudio nodes for loudness metering (set up in startConversation).
let audioCtx = null;
let analyser = null;
let vadBuf = null;

// Backpressure: only one chunk in flight at a time. The newest recorded chunk
// waits in pendingChunk; if it's overwritten before being sent we count a drop.
// This bounds lag to ~1 chunk so the conversation can never fall behind.
let inFlight = false;
let pendingChunk = null;
let sentAt = 0;
let dropped = 0;

startBtn.addEventListener('click', startConversation);
stopBtn.addEventListener('click', stopConversation);

function setStatus() {
  if (!active) { convStatus.textContent = 'Stopped'; return; }
  const drop = dropped > 0 ? ` · dropped ${dropped}` : '';
  if (inFlight) {
    convStatus.innerHTML = `<span class="recording-dot"></span>Processing…${drop}`;
  } else {
    convStatus.innerHTML = `<span class="recording-dot"></span>Live${drop}`;
  }
}

function trySend() {
  if (inFlight || !pendingChunk) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  const buf = pendingChunk;
  pendingChunk = null;
  inFlight = true;
  sentAt = Date.now();
  ws.send(buf);
  setStatus();
}

async function startConversation() {
  try {
    activeStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    convStatus.textContent = 'Mic access denied';
    return;
  }

  // Loudness meter for pause detection. The analyser only reads the signal — it
  // is not connected to the destination, so there's no audio feedback.
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(activeStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  vadBuf = new Uint8Array(analyser.fftSize);
  src.connect(analyser);

  const wsUrl = apiBase().replace(/^http/, 'ws') + '/ws/conversation';
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      model: modelSel.value,
      source_lang: sourceLang.value,
      lang_name: sourceLang.options[sourceLang.selectedIndex].text,
      context: contextInput.value.trim(),
      glossary: glossaryEl.value.trim(),
      participants: participantsEl.value.trim(),
      diarize: diarizeEl.checked,
    }));
  };

  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    // Every server reply (including skips) clears the in-flight slot so the
    // next pending chunk can go out.
    inFlight = false;
    const lagMs = sentAt ? Date.now() - sentAt : null;

    if (msg.error) {
      appendChunk({ error: msg.error });
    } else if (!msg.skipped) {
      appendChunk({
        source: msg.source,
        english: msg.english,
        speaker: msg.speaker,
        langTag: sourceLang.value.toUpperCase(),
        lagMs,
      });
    }
    setStatus();
    trySend();
  };

  ws.onerror = () => { convStatus.textContent = 'WebSocket error'; };
  ws.onclose = () => { if (active) stopConversation(); };

  active = true;
  inFlight = false;
  pendingChunk = null;
  dropped = 0;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus();

  // Cycle stop/start so each recording is a complete, self-contained WebM file.
  // Using timeslice produces headerless continuation chunks that Whisper rejects.
  recordCycle();
}

// Root-mean-square loudness of the current mic frame, 0..~1.
function currentRms() {
  if (!analyser) return 0;
  analyser.getByteTimeDomainData(vadBuf);
  let sum = 0;
  for (let i = 0; i < vadBuf.length; i++) {
    const v = (vadBuf[i] - 128) / 128;
    sum += v * v;
  }
  return Math.sqrt(sum / vadBuf.length);
}

function recordCycle() {
  if (!active || !activeStream) return;

  const rec = new MediaRecorder(activeStream);
  let sawSpeech = false;
  let speechMs = 0;
  let silenceMs = 0;
  let elapsed = 0;

  rec.ondataavailable = async (e) => {
    if (!e.data || e.data.size === 0) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    // Silence-only window — nothing worth transcribing, don't send.
    if (!sawSpeech) return;
    // Newest-wins: if a chunk is already waiting, it's stale — drop it.
    if (pendingChunk) dropped++;
    pendingChunk = await e.data.arrayBuffer();
    setStatus();
    trySend();
  };

  rec.onstop = () => { clearInterval(monitor); if (active) recordCycle(); };

  // Cut the recording on a sustained pause after speech, or at the hard cap.
  const monitor = setInterval(() => {
    if (rec.state !== 'recording') return;
    elapsed += VAD.POLL_MS;
    if (currentRms() >= VAD.RMS_THRESHOLD) {
      sawSpeech = true;
      speechMs += VAD.POLL_MS;
      silenceMs = 0;
    } else {
      silenceMs += VAD.POLL_MS;
    }
    const pauseEnded = sawSpeech && speechMs >= VAD.MIN_SPEECH_MS && silenceMs >= VAD.SILENCE_MS;
    if (pauseEnded || elapsed >= VAD.MAX_MS) rec.stop();
  }, VAD.POLL_MS);

  rec.start();
}

function stopConversation() {
  active = false;
  if (activeStream) {
    activeStream.getTracks().forEach(t => t.stop());
    activeStream = null;
  }
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; vadBuf = null; }
  if (ws) { ws.close(); ws = null; }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  convStatus.textContent = 'Stopped';
}

// ── Clear output ──────────────────────────────────────────────────────────────

clearBtn.addEventListener('click', () => { output.innerHTML = ''; });
