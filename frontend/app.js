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

// ── Conversation mode ─────────────────────────────────────────────────────────

let mediaRecorder = null;
let ws = null;
const INTERVAL_MS = 8000;

startBtn.addEventListener('click', startConversation);
stopBtn.addEventListener('click', stopConversation);

async function startConversation() {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    convStatus.textContent = 'Mic access denied';
    return;
  }

  // Open WebSocket
  const wsUrl = apiBase().replace(/^http/, 'ws') + '/ws/conversation';
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    // Send config frame first
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
    if (msg.error) {
      appendChunk({ error: msg.error });
      return;
    }
    appendChunk({
      source: msg.source,
      english: msg.english,
      langTag: (sourceLang.value).toUpperCase(),
    });
  };

  ws.onerror = () => { convStatus.textContent = 'WebSocket error'; };
  ws.onclose = () => { if (mediaRecorder) stopConversation(); };

  // Start MediaRecorder — sends a chunk every INTERVAL_MS
  mediaRecorder = new MediaRecorder(stream);

  mediaRecorder.ondataavailable = async (e) => {
    if (!e.data || e.data.size === 0) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;

    // Convert Blob → WAV-compatible ArrayBuffer and send as binary
    const buf = await e.data.arrayBuffer();
    ws.send(buf);
    convStatus.textContent = `Last chunk: ${(e.data.size / 1024).toFixed(1)} KB`;
  };

  mediaRecorder.start(INTERVAL_MS);

  startBtn.disabled = true;
  stopBtn.disabled = false;
  convStatus.innerHTML = '<span class="recording-dot"></span>Recording...';
}

function stopConversation() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
    mediaRecorder = null;
  }
  if (ws) {
    ws.close();
    ws = null;
  }
  startBtn.disabled = false;
  stopBtn.disabled = true;
  convStatus.textContent = 'Stopped';
}

// ── Clear output ──────────────────────────────────────────────────────────────

clearBtn.addEventListener('click', () => { output.innerHTML = ''; });
