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
const prepStatus   = document.getElementById('prepStatus');
const liveMeta     = document.getElementById('liveMeta');
const elapsedEl    = document.getElementById('elapsed');

const textInput    = document.getElementById('textInput');
const translateBtn = document.getElementById('translateBtn');
const showNotes    = document.getElementById('showNotes');
const textStatus   = document.getElementById('textStatus');

const output       = document.getElementById('output');
const clearBtn     = document.getElementById('clearBtn');
const copyBtn      = document.getElementById('copyBtn');
const jumpLatest   = document.getElementById('jumpLatest');

const interviewRole  = document.getElementById('interviewRole');
const interviewTerms = document.getElementById('interviewTerms');
const profileText  = document.getElementById('profileText');
const profileStatus = document.getElementById('profileStatus');
const saveProfileBtn = document.getElementById('saveProfileBtn');
const captureTabEl = document.getElementById('captureTab');
const startInterviewBtn = document.getElementById('startInterviewBtn');
const prepInterviewStatus = document.getElementById('prepInterviewStatus');
const hintsPanel   = document.getElementById('hintsPanel');
const hintsList    = document.getElementById('hintsList');

const gate         = document.getElementById('gate');
const gateStatus   = document.getElementById('gateStatus');
const signInBtn    = document.getElementById('signInBtn');
const signOutBtn   = document.getElementById('signOutBtn');
const accountChip  = document.getElementById('accountChip');
const accountName  = document.getElementById('accountName');
const historyList  = document.getElementById('historyList');
const viewingStrip = document.getElementById('viewingStrip');
const viewingLabel = document.getElementById('viewingLabel');
const viewingBack  = document.getElementById('viewingBack');

// ── Auth (login required: the gate covers the app until signed in) ──────────

let currentUser = null;

document.body.classList.add('auth-pending');

document.addEventListener('store-ready', () => {
  window.store.onUser(user => {
    currentUser = user;
    document.body.classList.remove('auth-pending');
    document.body.classList.toggle('signed-out', !user);
    gate.classList.toggle('hidden', !!user);
    accountChip.classList.toggle('hidden', !user);
    if (user) accountName.textContent = (user.displayName || user.email || '').split(' ')[0];
  });
});

signInBtn.addEventListener('click', async () => {
  gateStatus.textContent = '';
  try {
    await window.store.signIn();
  } catch (err) {
    gateStatus.textContent = 'Sign-in did not complete — try again.';
  }
});

signOutBtn.addEventListener('click', () => window.store.signOut());

// ── Tab switching ─────────────────────────────────────────────────────────────

tabs.forEach(tab => {
  tab.addEventListener('click', () => {
    tabs.forEach(t => t.classList.remove('active'));
    panels.forEach(p => p.classList.add('hidden'));
    tab.classList.add('active');
    document.getElementById(`panel-${tab.dataset.tab}`).classList.remove('hidden');
    if (tab.dataset.tab === 'history') renderHistoryList();
    if (tab.dataset.tab === 'interview') loadProfile();
  });
});

// ── Interview mode: profile + hints ──────────────────────────────────────────

let currentMode = 'interpret';   // 'interpret' | 'interview' — set at Start
let profileLoaded = false;

async function loadProfile() {
  if (profileLoaded || !window.store || !currentUser) return;
  try {
    const p = await window.store.getProfile();
    profileText.value = p.text || '';
    profileLoaded = true;
  } catch (err) {
    profileStatus.textContent = 'Could not load profile — check your connection.';
  }
}

saveProfileBtn.addEventListener('click', async () => {
  try {
    await window.store.saveProfile(profileText.value);
    profileStatus.textContent = 'Saved';
  } catch (err) {
    profileStatus.textContent = 'Save failed — try again.';
  }
  setTimeout(() => { profileStatus.textContent = ''; }, 1500);
});

function renderHint(hint, ts) {
  if (!hint || !hint.is_question) return;
  const card = document.createElement('div');
  card.className = 'hint-card';
  const meta = [ts || '', hint.searched ? 'web' : ''].filter(Boolean).join(' · ');
  card.innerHTML =
    `<div class="hint-q">${escHtml(hint.gist || 'Question')}<span class="hint-ts">${escHtml(meta)}</span></div>` +
    (hint.bullets || []).map(b => `<div class="hint-bullet">${escHtml(b)}</div>`).join('') +
    (hint.angle ? `<div class="hint-angle">${escHtml(hint.angle)}</div>` : '');
  hintsList.prepend(card);  // newest on top — the one you need right now
}

// ── Meeting setup fold ────────────────────────────────────────────────────────
// The setup form collapses to one line; the summary always shows what a meeting
// would run with. Open/closed is remembered across visits (default: collapsed).

const setupFold    = document.getElementById('setupFold');
const setupSummary = document.getElementById('setupSummary');

function updateSetupSummary() {
  const lang = sourceLang.options[sourceLang.selectedIndex].text;
  const model = modelSel.options[modelSel.selectedIndex].text.split(' ')[0];
  const parts = [`${lang} → English`, model];
  if (contextInput.value.trim()) parts.push(contextInput.value.trim());
  const terms = glossaryEl.value.split('\n').filter(l => l.trim()).length;
  if (terms) parts.push(`${terms} key terms`);
  setupSummary.textContent = parts.join(' · ');
}

[sourceLang, modelSel].forEach(el => el.addEventListener('change', updateSetupSummary));
[contextInput, glossaryEl].forEach(el => el.addEventListener('input', updateSetupSummary));
updateSetupSummary();

setupFold.open = localStorage.getItem('setupOpen') === '1';
setupFold.addEventListener('toggle', () => {
  localStorage.setItem('setupOpen', setupFold.open ? '1' : '');
});

// ── Helpers ───────────────────────────────────────────────────────────────────

function apiBase() {
  return backendUrl.value.replace(/\/$/, '');
}

function nowStamp() {
  return new Date().toLocaleTimeString('en-GB'); // HH:MM:SS
}

// Speaker rail palette (ink-adjacent), assigned in first-appearance order.
// Reset on each new conversation so colors track the new session's speakers.
const speakerColors = ['#2B4C7E', '#3E6B4F', '#7A4A6D', '#A8762C', '#2F6E75', '#6E4A32'];
let speakerIndex = {};
function railColor(speaker) {
  if (!speaker) return '';
  if (!(speaker in speakerIndex)) speakerIndex[speaker] = Object.keys(speakerIndex).length;
  return speakerColors[speakerIndex[speaker] % speakerColors.length];
}

// Everything rendered, kept for "Copy minutes".
let transcript = [];

// Only auto-scroll when the reader is already pinned to the bottom.
function isPinned() {
  return output.scrollTop >= output.scrollHeight - output.clientHeight - 80;
}

// Turn grouping: consecutive pairs from the same speaker within GROUP_GAP_MS
// merge into one block (one meta line, sentences flowing under one rail) —
// otherwise a talkative speaker becomes a wall of tiny timestamped blocks.
const GROUP_GAP_MS = 20000;
const GROUP_MAX_PAIRS = 6;
let lastTurn = null; // { speaker, wallMs, pairs, bodyEl }

function resetTurnGrouping() { lastTurn = null; }

function pairHtml(source, english, langTag, notes) {
  const noteHtml = (notes && notes.length)
    ? notes.map(n => `<span class="note">* ${escHtml(n)}</span>`).join('')
    : '';
  return (source ? `<span class="ja" lang="${escHtml((langTag || 'ja').toLowerCase())}">${escHtml(source)}</span>` : '')
    + (english ? `<span class="en">${escHtml(english)}</span>` : '')
    + noteHtml;
}

function appendChunk({ source, english, langTag, speaker, notes, error, lagMs, ts }) {
  const pinned = isPinned();
  ts = ts || nowStamp();

  if (error) {
    const div = document.createElement('div');
    div.className = 'turn-error';
    div.innerHTML = `[${ts}] ${escHtml(error)}`;
    output.appendChild(div);
    resetTurnGrouping();
  } else {
    transcript.push({ ts, speaker: speaker || '', source: source || '', english: english || '' });
    const now = Date.now();
    const canGroup = lastTurn
      && lastTurn.speaker === (speaker || '')
      && now - lastTurn.wallMs < GROUP_GAP_MS
      && lastTurn.pairs < GROUP_MAX_PAIRS;

    if (canGroup) {
      const pair = document.createElement('div');
      pair.className = 'turn-pair';
      pair.innerHTML = pairHtml(source, english, langTag, notes);
      lastTurn.bodyEl.appendChild(pair);
      lastTurn.wallMs = now;
      lastTurn.pairs += 1;
    } else {
      const div = document.createElement('div');
      div.className = 'turn';
      const color = railColor(speaker);
      if (color) div.style.setProperty('--rail', color);
      const lag = lagMs != null ? ` · ${(lagMs / 1000).toFixed(1)}s` : '';
      const speakerHtml = speaker ? `<span class="turn-speaker">${escHtml(speaker)}</span>` : '';
      div.innerHTML =
        `<div class="turn-meta">${ts}${speakerHtml ? ' ' : ''}${speakerHtml}` +
        `<span>${escHtml(langTag || '')}${lag}</span></div>` +
        `<div class="turn-body"><div class="turn-pair">${pairHtml(source, english, langTag, notes)}</div></div>`;
      output.appendChild(div);
      lastTurn = { speaker: speaker || '', wallMs: now, pairs: 1,
                   bodyEl: div.querySelector('.turn-body') };
    }
  }

  if (pinned) {
    output.scrollTop = output.scrollHeight;
  } else {
    jumpLatest.classList.remove('hidden');
  }
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Transcript actions ────────────────────────────────────────────────────────

output.addEventListener('scroll', () => {
  if (isPinned()) jumpLatest.classList.add('hidden');
});

jumpLatest.addEventListener('click', () => {
  output.scrollTop = output.scrollHeight;
  jumpLatest.classList.add('hidden');
});

copyBtn.addEventListener('click', async () => {
  if (!transcript.length) return;
  const text = transcript.map(t => {
    const head = `[${t.ts}]${t.speaker ? ' ' + t.speaker : ''}`;
    return [head, t.source, t.english].filter(Boolean).join('\n');
  }).join('\n\n');
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = 'Copied';
    setTimeout(() => { copyBtn.textContent = 'Copy minutes'; }, 1500);
  } catch {
    copyBtn.textContent = 'Copy failed';
    setTimeout(() => { copyBtn.textContent = 'Copy minutes'; }, 1500);
  }
});

clearBtn.addEventListener('click', () => {
  output.innerHTML = '';
  transcript = [];
  resetTurnGrouping();
  jumpLatest.classList.add('hidden');
  hintsList.innerHTML = '';
  if (!active) hintsPanel.classList.add('hidden');
});

// ── Saved sessions & history ─────────────────────────────────────────────────

let liveSessionId = '';   // current live meeting doc
let textSessionId = '';   // lazy per-page-load doc for typed translations
let viewingHistory = false;

function saveTurn(sessionId, turn) {
  if (window.store && sessionId && !viewingHistory) window.store.saveTurn(sessionId, turn);
}

async function renderHistoryList() {
  if (!window.store || !currentUser) return;
  historyList.innerHTML = '<p class="hint">Loading your meetings…</p>';
  let sessions = [];
  try {
    sessions = await window.store.listSessions();
  } catch (err) {
    historyList.innerHTML = '<p class="hint">Could not load history — check your connection and reopen this tab.</p>';
    return;
  }
  if (!sessions.length) {
    historyList.innerHTML = '<p class="hint">No saved meetings yet — run one and it will appear here.</p>';
    return;
  }
  historyList.innerHTML = '';
  sessions.forEach(s => historyList.appendChild(historyRow(s)));
}

function sessionName(s) {
  return s.title || s.context || s.langName || 'Meeting';
}

function historyRow(s) {
  const when = s.startedAt && s.startedAt.toDate
    ? s.startedAt.toDate().toLocaleString('en-GB', { dateStyle: 'medium', timeStyle: 'short' })
    : '—';
  const row = document.createElement('div');
  row.className = 'history-row';
  row.innerHTML =
    `<button class="history-open">` +
    `<span class="history-when">${escHtml(when)}</span>` +
    `<span class="history-desc"><strong>${escHtml(sessionName(s))}</strong>` +
    `${s.preview ? ' — ' + escHtml(s.preview) : ''}</span>` +
    `<span class="history-count">${(s.turns || []).length} turns</span>` +
    `</button>` +
    `<button class="history-action" data-act="rename" title="Rename">Rename</button>` +
    `<button class="history-action history-action-danger" data-act="delete" title="Delete">Delete</button>`;

  row.querySelector('.history-open').addEventListener('click', () => openSession(s.id));

  row.querySelector('[data-act="rename"]').addEventListener('click', () => {
    const desc = row.querySelector('.history-desc');
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'history-rename';
    input.value = s.title || '';
    input.placeholder = 'Meeting title, e.g. Sales meeting';
    desc.replaceWith(input);
    input.focus();
    input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      if (save && input.value.trim() && input.value.trim() !== s.title) {
        try {
          await window.store.renameSession(s.id, input.value);
          s.title = input.value.trim();
        } catch (err) {
          console.warn('rename failed:', err);
        }
      }
      row.replaceWith(historyRow(s));
    };
    input.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') finish(true);
      if (e.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(true));
  });

  row.querySelector('[data-act="delete"]').addEventListener('click', async () => {
    if (!window.confirm(`Delete “${sessionName(s)}” and its transcript? This cannot be undone.`)) return;
    try {
      await window.store.deleteSession(s.id);
      row.remove();
      if (!historyList.children.length) {
        historyList.innerHTML = '<p class="hint">No saved meetings yet — run one and it will appear here.</p>';
      }
    } catch (err) {
      console.warn('delete failed:', err);
    }
  });

  return row;
}

async function openSession(id) {
  const s = await window.store.getSession(id);
  if (!s) return;
  viewingHistory = true;
  output.innerHTML = '';
  transcript = [];
  resetTurnGrouping();
  speakerIndex = {};
  const when = s.startedAt && s.startedAt.toDate ? s.startedAt.toDate().toLocaleString('en-GB') : '';
  const terms = (s.glossary || '').split('\n').filter(l => l.trim()).length;
  const meta = [
    `${s.langName || 'Japanese'} → English`,
    s.model || '',
    s.context && s.title ? s.context : '',
    terms ? `${terms} key terms` : '',
    s.participants ? `participants: ${s.participants.split('\n').filter(Boolean).join(', ')}` : '',
  ].filter(Boolean).join(' · ');
  viewingLabel.innerHTML =
    `<strong>${escHtml(sessionName(s))}</strong>${when ? ' · ' + escHtml(when) : ''}` +
    `<span class="viewing-meta">${escHtml(meta)}</span>`;
  viewingStrip.classList.remove('hidden');
  hintsList.innerHTML = '';
  (s.turns || []).forEach(t => {
    appendChunk({ source: t.source, english: t.english, speaker: t.speaker,
                  langTag: t.langTag || 'JA', ts: t.ts });
    if (t.hint) renderHint(t.hint, t.ts);
  });
  hintsPanel.classList.toggle('hidden', !(s.turns || []).some(t => t.hint));
  output.scrollTop = 0;
}

viewingBack.addEventListener('click', () => {
  viewingHistory = false;
  viewingStrip.classList.add('hidden');
  output.innerHTML = '';
  transcript = [];
  resetTurnGrouping();
});

// ── Text mode ─────────────────────────────────────────────────────────────────

translateBtn.addEventListener('click', async () => {
  const text = textInput.value.trim();
  if (!text) return;

  translateBtn.disabled = true;
  textStatus.textContent = 'Translating…';

  try {
    const idToken = window.store ? await window.store.idToken() : '';
    const res = await fetch(`${apiBase()}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(idToken ? { 'Authorization': `Bearer ${idToken}` } : {}),
      },
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
    if (window.store && !textSessionId) {
      textSessionId = await window.store.startSession({
        langName: sourceLang.options[sourceLang.selectedIndex].text,
        sourceLang: sourceLang.value,
        model: modelSel.value, context: 'Typed text',
        glossary: glossaryEl.value.trim(), participants: '', diarize: false,
      });
    }
    saveTurn(textSessionId, {
      seq: transcript.length, ts: nowStamp(), speaker: '',
      source: data.source_text, english: data.english_text,
      langTag: sourceLang.value.toUpperCase(), first: transcript.length === 1,
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

// Elapsed-time ticker for the live status bar.
let meetingStart = 0;
let elapsedTimer = null;

startBtn.addEventListener('click', () => startConversation('interpret'));
startInterviewBtn.addEventListener('click', () => startConversation('interview'));
stopBtn.addEventListener('click', stopConversation);

function setStatus() {
  if (!active) { convStatus.textContent = ''; return; }
  const drop = dropped > 0 ? ` · dropped ${dropped}` : '';
  convStatus.textContent = (inFlight ? 'Translating…' : 'Listening…') + drop;
}

function tickElapsed() {
  const s = Math.floor((Date.now() - meetingStart) / 1000);
  const hh = String(Math.floor(s / 3600)).padStart(2, '0');
  const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
  const ss = String(s % 60).padStart(2, '0');
  elapsedEl.textContent = `${hh}:${mm}:${ss}`;
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

let micStream = null;
let displayStream = null;

async function startConversation(mode) {
  currentMode = mode || 'interpret';
  const statusEl = currentMode === 'interview' ? prepInterviewStatus : prepStatus;
  statusEl.textContent = '';
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = 'Microphone access was denied — allow the mic and try again.';
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)();

  // Interview mode: the interviewer usually comes through headphones, which the
  // mic never hears — mix in the meeting tab's audio via screen-share capture.
  activeStream = micStream;
  if (currentMode === 'interview' && captureTabEl.checked) {
    try {
      displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true, audio: true,
      });
      if (displayStream.getAudioTracks().length) {
        const dest = audioCtx.createMediaStreamDestination();
        audioCtx.createMediaStreamSource(micStream).connect(dest);
        audioCtx.createMediaStreamSource(displayStream).connect(dest);
        activeStream = dest.stream;
      } else {
        statusEl.textContent = 'That window has no audio — pick a tab and tick "share tab audio". Using mic only.';
      }
    } catch (err) {
      statusEl.textContent = 'Tab capture declined — using mic only.';
    }
  }

  // Loudness meter for pause detection. The analyser only reads the signal — it
  // is not connected to the destination, so there's no audio feedback.
  const src = audioCtx.createMediaStreamSource(activeStream);
  analyser = audioCtx.createAnalyser();
  analyser.fftSize = 1024;
  vadBuf = new Uint8Array(analyser.fftSize);
  src.connect(analyser);

  // Interview mode is self-contained: English, Sonnet, and its own fields
  // (role/company + names & terms + profile) — Meeting setup does not apply.
  const isInterview = currentMode === 'interview';
  const langCode = isInterview ? 'en' : sourceLang.value;
  const langName = isInterview ? 'English' : sourceLang.options[sourceLang.selectedIndex].text;
  const sessionModel = isInterview ? 'sonnet' : modelSel.value;
  const sessionContext = isInterview ? interviewRole.value.trim() : contextInput.value.trim();
  const sessionGlossary = isInterview ? interviewTerms.value.trim() : glossaryEl.value.trim();
  const sessionParticipants = isInterview ? '' : participantsEl.value.trim();
  const sessionDiarize = isInterview ? true : diarizeEl.checked;

  // Session doc + ID token before the socket opens (login is required).
  // The full setup is saved with the meeting so history shows how it was run.
  const idToken = window.store ? await window.store.idToken() : '';
  if (window.store) {
    liveSessionId = await window.store.startSession({
      mode: currentMode,
      langName,
      sourceLang: langCode,
      model: sessionModel,
      context: sessionContext,
      glossary: sessionGlossary,
      participants: sessionParticipants,
      diarize: sessionDiarize,
    });
  }
  viewingHistory = false;
  viewingStrip.classList.add('hidden');

  const wsUrl = apiBase().replace(/^http/, 'ws') + '/ws/conversation';
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      mode: currentMode,
      profile: isInterview ? profileText.value.trim() : '',
      model: sessionModel,
      source_lang: langCode,
      lang_name: langName,
      context: sessionContext,
      glossary: sessionGlossary,
      participants: sessionParticipants,
      diarize: sessionDiarize,
      id_token: idToken,
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
        langTag: (msg.lang_tag || sourceLang.value).toUpperCase(),
        lagMs,
      });
      if (msg.hint) renderHint(msg.hint, nowStamp());
      saveTurn(liveSessionId, {
        seq: transcript.length, ts: nowStamp(), speaker: msg.speaker || '',
        source: msg.source, english: msg.english,
        langTag: (msg.lang_tag || sourceLang.value).toUpperCase(),
        first: transcript.length === 1,
        ...(msg.hint && msg.hint.is_question ? { hint: msg.hint } : {}),
      });
    }
    setStatus();
    trySend();
  };

  ws.onerror = () => {
    appendChunk({ error: 'Connection lost — press “Start meeting” to reconnect.' });
  };
  ws.onclose = (evt) => {
    if (evt.code === 4401) {
      appendChunk({ error: 'Session expired — sign in again to continue.' });
    }
    if (active) stopConversation();
  };

  active = true;
  inFlight = false;
  pendingChunk = null;
  dropped = 0;
  speakerIndex = {};  // fresh session → fresh first-appearance color order
  startBtn.disabled = true;
  stopBtn.disabled = false;

  // Live view: setup collapses to the status bar, transcript becomes the hero.
  const modelName = modelSel.options[modelSel.selectedIndex].text.split(' ')[0];
  liveMeta.textContent = isInterview
    ? `Interview copilot${sessionContext ? ' · ' + sessionContext : ''}`
    : `${langName} → English · ${modelName}`;
  if (isInterview) {
    hintsList.innerHTML = '';
    hintsPanel.classList.remove('hidden');
    document.body.classList.add('interview');
  } else {
    hintsPanel.classList.add('hidden');
    document.body.classList.remove('interview');
  }
  document.body.classList.add('live');
  meetingStart = Date.now();
  tickElapsed();
  elapsedTimer = setInterval(tickElapsed, 1000);
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
  if (window.store && liveSessionId) {
    window.store.endSession(liveSessionId, transcript.length);
    liveSessionId = '';
  }
  [activeStream, micStream, displayStream].forEach(s => {
    if (s) s.getTracks().forEach(t => t.stop());
  });
  activeStream = micStream = displayStream = null;
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; vadBuf = null; }
  if (ws) { ws.close(); ws = null; }
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  document.body.classList.remove('live');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  convStatus.textContent = '';
}
