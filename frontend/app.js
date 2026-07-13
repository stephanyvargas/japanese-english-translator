// ── DOM refs ─────────────────────────────────────────────────────────────────

const sourceLang   = document.getElementById('sourceLang');
const modelSel     = document.getElementById('model');
const contextInput = document.getElementById('context');
const glossaryEl   = document.getElementById('glossary');
const verifyEl     = document.getElementById('verify');
const participantsEl = document.getElementById('participants');
const diarizeEl    = document.getElementById('diarize');
const backendUrl   = document.getElementById('backendUrl');

const views        = document.querySelectorAll('#setup .view');

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
const profileSelect = document.getElementById('profileSelect');
const profileSummary = document.getElementById('profileSummary');
const newProfileBtn = document.getElementById('newProfileBtn');
const renameProfileBtn = document.getElementById('renameProfileBtn');
const deleteProfileBtn = document.getElementById('deleteProfileBtn');
const docChips     = document.getElementById('docChips');
const docFile      = document.getElementById('docFile');
const docNote      = document.getElementById('docNote');
const addDocBtn    = document.getElementById('addDocBtn');
const repoChips    = document.getElementById('repoChips');
const repoInput    = document.getElementById('repoInput');
const addRepoBtn   = document.getElementById('addRepoBtn');
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

// ── View router: home (mode cards) → dedicated per-mode screens ──────────────

function showView(name) {
  views.forEach(v => v.classList.toggle('hidden', v.id !== `view-${name}`));
  document.body.classList.toggle('view-home', name === 'home');
  // The shared Meeting-setup fold serves both interpreter and typed-text —
  // move the single element into whichever view is active.
  if (name === 'interpret' || name === 'text') {
    const slot = document.querySelector(`#view-${name} .setup-slot`);
    const fold = document.getElementById('setupFold');
    if (slot && fold && fold.parentElement !== slot) slot.appendChild(fold);
  }
  if (name === 'history') renderHistoryList();
  if (name === 'interview') loadProfile();
}

document.querySelectorAll('.mode-card[data-view]').forEach(card =>
  card.addEventListener('click', () => showView(card.dataset.view)));
document.querySelectorAll('.back-home').forEach(btn =>
  btn.addEventListener('click', () => showView('home')));

// ── Interview mode: named profiles (bio + documents + repos) + hints ─────────

let currentMode = 'interpret';   // 'interpret' | 'interview' — set at Start
let profiles = [];
let activeProfile = null;
let profilesLoaded = false;

function savedFlash(msg = 'Saved ✓ — loads automatically next time') {
  profileStatus.textContent = msg;
  setTimeout(() => { profileStatus.textContent = ''; }, 2500);
}

async function persistActiveProfile(flash = true) {
  if (!activeProfile) return;
  try {
    await window.store.updateProfile(activeProfile.id, {
      name: activeProfile.name, bio: activeProfile.bio,
      documents: activeProfile.documents, repos: activeProfile.repos,
    });
    if (flash) savedFlash();
  } catch (err) {
    profileStatus.textContent = 'Save failed — check your connection.';
  }
}

async function loadProfile() {   // called when the Interview view opens
  if (profilesLoaded || !window.store || !currentUser) return;
  try {
    profiles = await window.store.listProfiles();
    const savedId = localStorage.getItem('profileId');
    activeProfile = profiles.find(p => p.id === savedId) || profiles[0];
    profilesLoaded = true;
    renderProfileUI();
  } catch (err) {
    profileStatus.textContent = 'Could not load profiles — check your connection.';
  }
}

function renderProfileUI() {
  profileSelect.innerHTML = '';
  profiles.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.id;
    opt.textContent = p.name;
    profileSelect.appendChild(opt);
  });
  if (!activeProfile) return;
  profileSelect.value = activeProfile.id;
  localStorage.setItem('profileId', activeProfile.id);
  profileText.value = activeProfile.bio || '';
  renderChips();
  updateProfileSummary();
}

function updateProfileSummary() {
  if (!activeProfile) return;
  const bits = [activeProfile.name];
  if ((activeProfile.bio || '').trim()) bits.push('bio');
  const d = (activeProfile.documents || []).length;
  const r = (activeProfile.repos || []).length;
  if (d) bits.push(`${d} doc${d > 1 ? 's' : ''}`);
  if (r) bits.push(`${r} repo${r > 1 ? 's' : ''}`);
  profileSummary.textContent = bits.join(' · ');
}

function chip(label, onRemove) {
  const el = document.createElement('span');
  el.className = 'chip';
  el.innerHTML = `<span class="chip-label">${escHtml(label)}</span>`;
  const x = document.createElement('button');
  x.className = 'chip-x';
  x.textContent = '×';
  x.title = 'Remove';
  x.addEventListener('click', onRemove);
  el.appendChild(x);
  return el;
}

function renderChips() {
  docChips.innerHTML = '';
  (activeProfile.documents || []).forEach((d, i) => {
    docChips.appendChild(chip(d.note ? `${d.name} — ${d.note}` : d.name, async () => {
      activeProfile.documents.splice(i, 1);
      renderChips(); updateProfileSummary();
      await persistActiveProfile();
    }));
  });
  repoChips.innerHTML = '';
  (activeProfile.repos || []).forEach((r, i) => {
    repoChips.appendChild(chip(r.repo, async () => {
      activeProfile.repos.splice(i, 1);
      renderChips(); updateProfileSummary();
      await persistActiveProfile();
    }));
  });
}

profileSelect.addEventListener('change', () => {
  activeProfile = profiles.find(p => p.id === profileSelect.value) || activeProfile;
  renderProfileUI();
});

newProfileBtn.addEventListener('click', async () => {
  const name = window.prompt('Name for the new profile (e.g. "ML engineer"):');
  if (!name || !name.trim()) return;
  const p = await window.store.createProfile(name.trim());
  profiles.unshift(p);
  activeProfile = p;
  renderProfileUI();
  savedFlash('Profile created ✓');
});

renameProfileBtn.addEventListener('click', async () => {
  if (!activeProfile) return;
  const name = window.prompt('Rename profile:', activeProfile.name);
  if (!name || !name.trim() || name.trim() === activeProfile.name) return;
  activeProfile.name = name.trim();
  renderProfileUI();
  await persistActiveProfile();
});

deleteProfileBtn.addEventListener('click', async () => {
  if (!activeProfile) return;
  if (!window.confirm(`Delete profile “${activeProfile.name}”? This cannot be undone.`)) return;
  await window.store.deleteProfile(activeProfile.id);
  profiles = profiles.filter(p => p.id !== activeProfile.id);
  if (!profiles.length) profiles = [await window.store.createProfile('Default')];
  activeProfile = profiles[0];
  renderProfileUI();
});

let bioTimer = null;
profileText.addEventListener('input', () => {
  if (!activeProfile) return;
  activeProfile.bio = profileText.value;
  updateProfileSummary();
  clearTimeout(bioTimer);
  bioTimer = setTimeout(() => persistActiveProfile(), 900);
});

addDocBtn.addEventListener('click', async () => {
  const file = docFile.files && docFile.files[0];
  if (!file || !activeProfile) { profileStatus.textContent = 'Choose a file first.'; return; }
  addDocBtn.disabled = true;
  profileStatus.textContent = 'Extracting…';
  try {
    const form = new FormData();
    form.append('file', file);
    form.append('note', docNote.value.trim());
    const idToken = await window.store.idToken();
    const res = await fetch(`${apiBase()}/profile/ingest`, {
      method: 'POST', body: form,
      headers: idToken ? { 'Authorization': `Bearer ${idToken}` } : {},
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Extraction failed.');
    const d = await res.json();
    activeProfile.documents = activeProfile.documents || [];
    activeProfile.documents.push({ name: d.name, note: d.note, text: d.text });
    docFile.value = ''; docNote.value = '';
    renderChips(); updateProfileSummary();
    await persistActiveProfile();
  } catch (err) {
    profileStatus.textContent = err.message;
  } finally {
    addDocBtn.disabled = false;
  }
});

addRepoBtn.addEventListener('click', async () => {
  const repo = repoInput.value.trim();
  if (!repo || !activeProfile) { profileStatus.textContent = 'Enter owner/repo first.'; return; }
  addRepoBtn.disabled = true;
  profileStatus.textContent = 'Summarizing repo…';
  try {
    const idToken = await window.store.idToken();
    const res = await fetch(`${apiBase()}/profile/github`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json',
                 ...(idToken ? { 'Authorization': `Bearer ${idToken}` } : {}) },
      body: JSON.stringify({ repo }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || 'Repo lookup failed.');
    const r = await res.json();
    activeProfile.repos = activeProfile.repos || [];
    activeProfile.repos.push({ repo: r.repo, summary: r.summary });
    repoInput.value = '';
    renderChips(); updateProfileSummary();
    await persistActiveProfile();
  } catch (err) {
    profileStatus.textContent = err.message;
  } finally {
    addRepoBtn.disabled = false;
  }
});

// The one string the hint engine receives — bio, then documents, then repos.
function compileProfile() {
  if (!activeProfile) return profileText.value.trim();
  const parts = [];
  if ((activeProfile.bio || '').trim()) parts.push(`BIO & NOTES:\n${activeProfile.bio.trim()}`);
  (activeProfile.documents || []).forEach(d =>
    parts.push(`DOCUMENT: ${d.name}${d.note ? ` — ${d.note}` : ''}\n${d.text}`));
  (activeProfile.repos || []).forEach(r =>
    parts.push(`PROJECT (github ${r.repo}):\n${r.summary}`));
  return parts.join('\n\n').slice(0, 24000);
}

// Hint cards fill in three stages: a pending skeleton the moment the server
// gates a question in, streamed partial bullets as they generate, and the
// authoritative final card. pendingCards maps seq → card element.
let pendingCards = {};

function hintCardHtml(hint, meta) {
  return `<div class="hint-q">${escHtml(hint.gist || 'Question')}<span class="hint-ts">${escHtml(meta)}</span></div>` +
    (hint.bullets || []).map(b => `<div class="hint-bullet">${escHtml(b)}</div>`).join('') +
    (hint.angle ? `<div class="hint-angle">${escHtml(hint.angle)}</div>` : '');
}

function renderHintPending(seq) {
  const card = document.createElement('div');
  card.className = 'hint-card hint-thinking';
  card.innerHTML = '<div class="hint-q">Thinking<span class="hint-dots">…</span></div>';
  hintsList.prepend(card);  // newest on top — the one you need right now
  hintsList.scrollTop = 0;  // and make sure the top is what's visible
  pendingCards[seq] = card;
}

function renderHintPartial(seq, partial) {
  const card = pendingCards[seq];
  if (!card) return;
  card.innerHTML = hintCardHtml({ gist: partial.gist, bullets: partial.bullets }, '…');
}

function renderHint(hint, ts, seq) {
  const card = seq != null ? pendingCards[seq] : null;
  if (seq != null) delete pendingCards[seq];
  if (!hint || !hint.is_question) {
    if (card) card.remove();  // gate said maybe, model said no — clear skeleton
    return;
  }
  const secs = hint.ms ? `${(hint.ms / 1000).toFixed(1)}s` : '';
  const meta = [ts || '', hint.searched ? 'web' : '', secs].filter(Boolean).join(' · ');
  if (card) {
    card.classList.remove('hint-thinking');
    card.innerHTML = hintCardHtml(hint, meta);
  } else {
    const el = document.createElement('div');
    el.className = 'hint-card';
    el.innerHTML = hintCardHtml(hint, meta);
    hintsList.prepend(el);
    hintsList.scrollTop = 0;
  }
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
  let pairEl = null;
  let entry = null;

  if (error) {
    const div = document.createElement('div');
    div.className = 'turn-error';
    div.innerHTML = `[${ts}] ${escHtml(error)}`;
    output.appendChild(div);
    resetTurnGrouping();
  } else {
    entry = { ts, speaker: speaker || '', source: source || '', english: english || '' };
    transcript.push(entry);
    const now = Date.now();
    const canGroup = lastTurn
      && lastTurn.speaker === (speaker || '')
      && now - lastTurn.wallMs < GROUP_GAP_MS
      && lastTurn.pairs < GROUP_MAX_PAIRS;

    if (canGroup) {
      pairEl = document.createElement('div');
      pairEl.className = 'turn-pair';
      pairEl.innerHTML = pairHtml(source, english, langTag, notes);
      lastTurn.bodyEl.appendChild(pairEl);
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
      pairEl = div.querySelector('.turn-pair');
    }
  }

  if (pinned) {
    output.scrollTop = output.scrollHeight;
  } else {
    jumpLatest.classList.remove('hidden');
  }
  // Live turns render source first; the translation attaches when it arrives.
  return { pairEl, entry };
}

// Live-session registry: utterance id → its rendered pair + transcript entry,
// so translation.final can attach the English line seconds after the source.
let uttTurns = {};

function attachTranslation(uttId, english) {
  const t = uttTurns[uttId];
  if (!t) return;
  if (t.entry) t.entry.english = english;
  if (t.pairEl && !t.pairEl.querySelector('.en')) {
    const span = document.createElement('span');
    span.className = 'en';
    span.textContent = english;
    const note = t.pairEl.querySelector('.note');
    t.pairEl.insertBefore(span, note || null);
    if (isPinned()) output.scrollTop = output.scrollHeight;
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
  pendingCards = {};
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
  pendingCards = {};
  (s.turns || []).forEach(t => {
    if (t.source || t.english) {
      appendChunk({ source: t.source, english: t.english, speaker: t.speaker,
                    langTag: t.langTag || 'JA', ts: t.ts });
    }
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
// Interview preset trims the pause wait — a question's hint should start
// generating ~0.5s after the interviewer stops, not 0.8s.
let activeVad = null; // set per-session in startConversation

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

// Server-owned session + resumable event stream (architecture v2, P1).
let wsSessionId = null;    // backend LiveSession id
let streamingSession = false;  // P2: server transcribes a continuous PCM stream
let workletNode = null;
let partialEls = {};       // utt_id → live partial-transcript line element
let lastSeq = 0;           // highest envelope seq seen — reconnects replay from here
let reconnectAttempts = 0;
const RECONNECT_MAX = 5;

// P2 streaming capture: an AudioWorklet forwards raw PCM (24kHz mono Int16,
// ~100ms frames) straight to the session socket. The server-side Realtime API
// does the voice-activity detection — no MediaRecorder, no chunk boundaries.
async function startPcmStream() {
  const code = "class F extends AudioWorkletProcessor{process(i){const c=i[0][0];if(c)this.port.postMessage(c.slice(0));return true;}}registerProcessor('pcm-fwd',F);";
  const url = URL.createObjectURL(new Blob([code], { type: 'application/javascript' }));
  await audioCtx.audioWorklet.addModule(url);
  workletNode = new AudioWorkletNode(audioCtx, 'pcm-fwd');
  audioCtx.createMediaStreamSource(activeStream).connect(workletNode);
  let buf = new Float32Array(0);
  workletNode.port.onmessage = (e) => {
    const merged = new Float32Array(buf.length + e.data.length);
    merged.set(buf); merged.set(e.data, buf.length);
    buf = merged;
    if (buf.length >= 2400) {   // 100ms at 24kHz
      const pcm = new Int16Array(buf.length);
      for (let i = 0; i < buf.length; i++) {
        pcm[i] = Math.max(-32768, Math.min(32767, Math.round(buf[i] * 32767)));
      }
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(pcm.buffer);
      buf = new Float32Array(0);
    }
  };
}

function connectSessionWs() {
  const wsUrl = apiBase().replace(/^http/, 'ws')
    + `/ws/conversation?session_id=${encodeURIComponent(wsSessionId)}&last_seq=${lastSeq}`;
  ws = new WebSocket(wsUrl);

  ws.onopen = async () => {
    const idToken = window.store ? await window.store.idToken() : '';
    ws.send(JSON.stringify({ op: 'start', id_token: idToken }));
    reconnectAttempts = 0;
    inFlight = false;   // any chunk that was in flight during the drop is gone
    setStatus();
    trySend();
  };

  ws.onmessage = (evt) => {
    const env = JSON.parse(evt.data);
    if (env.seq) lastSeq = Math.max(lastSeq, env.seq);
    handleSessionEvent(env);
  };

  ws.onclose = (evt) => {
    if (evt.code === 4401) {
      appendChunk({ error: 'Session expired — sign in again to continue.' });
      if (active) stopConversation();
      return;
    }
    if (evt.code === 4404) {
      appendChunk({ error: 'Session ended on the server — press Start to begin a new one.' });
      if (active) stopConversation();
      return;
    }
    if (active && wsSessionId && reconnectAttempts < RECONNECT_MAX) {
      reconnectAttempts += 1;
      convStatus.textContent = `Reconnecting… (${reconnectAttempts})`;
      setTimeout(() => { if (active) connectSessionWs(); }, 1000 * reconnectAttempts);
    } else if (active) {
      appendChunk({ error: 'Connection lost — press Start to begin a new session.' });
      stopConversation();
    }
  };

  ws.onerror = () => { /* onclose handles recovery */ };
}

function handleSessionEvent(env) {
  const d = env.data || {};
  switch (env.type) {
    case 'transcript.partial': {
      let el = partialEls[d.utt_id];
      if (!el) {
        el = document.createElement('div');
        el.className = 'turn-partial';
        output.appendChild(el);
        partialEls[d.utt_id] = el;
      }
      el.textContent = d.text;
      if (isPinned()) output.scrollTop = output.scrollHeight;
      break;
    }
    case 'transcript.final': {
      if (partialEls[d.utt_id]) {
        partialEls[d.utt_id].remove();
        delete partialEls[d.utt_id];
      }
      const rendered = appendChunk({
        source: d.text, english: '', speaker: d.speaker,
        langTag: (d.lang || sourceLang.value).toUpperCase(),
      });
      uttTurns[d.utt_id] = rendered;
      if (currentMode === 'interview') {
        saveTurn(liveSessionId, {
          seq: transcript.length, ts: nowStamp(), speaker: d.speaker || '',
          source: d.text, english: '', langTag: (d.lang || 'en').toUpperCase(),
          first: transcript.length === 1,
        });
      }
      break;
    }
    case 'translation.final': {
      attachTranslation(d.utt_id, d.english);
      const t = uttTurns[d.utt_id];
      saveTurn(liveSessionId, {
        seq: transcript.length, ts: nowStamp(),
        speaker: (t && t.entry && t.entry.speaker) || '',
        source: (t && t.entry && t.entry.source) || '',
        english: d.english, langTag: sourceLang.value.toUpperCase(),
        first: transcript.length === 1,
      });
      break;
    }
    case 'hint.pending':
      renderHintPending(d.utt_id);
      break;
    case 'hint.partial':
      renderHintPartial(d.utt_id, d);
      break;
    case 'hint.final': {
      const hint = { is_question: d.is_question, gist: d.gist, bullets: d.bullets,
                     angle: d.angle, searched: d.searched, ms: d.ms };
      renderHint(hint, nowStamp(), d.utt_id);
      if (hint.is_question) {
        saveTurn(liveSessionId, {
          seq: transcript.length, ts: nowStamp(), speaker: '',
          source: '', english: '', hint,
        });
      }
      break;
    }
    case 'chunk.ack':
      // Transitional (P1): the capture loop sends one chunk at a time and
      // waits for this before releasing the next. Removed in P2.
      inFlight = false;
      setStatus();
      trySend();
      break;
    case 'session.status':
      if (d.state === 'resumed') convStatus.textContent = 'Reconnected';
      break;
    case 'error':
      appendChunk({ error: d.message });
      break;
    default:
      break;  // forward compatibility: ignore unknown event types
  }
}

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
  // Interview trims the pause wait and the max turn length — speed over accuracy.
  activeVad = currentMode === 'interview'
    ? { ...VAD, SILENCE_MS: 500, MAX_MS: 10000 }
    : { ...VAD };
  const statusEl = currentMode === 'interview' ? prepInterviewStatus : prepStatus;
  statusEl.textContent = '';
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    statusEl.textContent = 'Microphone access was denied — allow the mic and try again.';
    return;
  }

  // 24kHz context serves all paths: VAD metering, tab+mic mixing, and the
  // streaming-ASR PCM pipeline (which requires this exact rate).
  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });

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

  // Create the server-owned session over REST, then attach the WebSocket to
  // it. A dropped connection re-attaches with last_seq and replays anything
  // missed — the session (transcript, brief, summary) lives on the server.
  let sessionRes;
  try {
    sessionRes = await fetch(`${apiBase()}/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json',
                 ...(idToken ? { 'Authorization': `Bearer ${idToken}` } : {}) },
      body: JSON.stringify({
        mode: currentMode,
        profile: isInterview ? compileProfile() : '',
        model: sessionModel,
        source_lang: langCode,
        lang_name: langName,
        context: sessionContext,
        glossary: sessionGlossary,
        participants: sessionParticipants,
        diarize: sessionDiarize,
      }),
    });
    if (!sessionRes.ok) throw new Error(`session create failed (${sessionRes.status})`);
  } catch (err) {
    statusEl.textContent = 'Could not reach the backend — check the connection and try again.';
    [micStream, displayStream].forEach(st => st && st.getTracks().forEach(t => t.stop()));
    micStream = displayStream = null;
    if (audioCtx) { audioCtx.close(); audioCtx = null; }
    return;
  }
  const sessionInfo = await sessionRes.json();
  wsSessionId = sessionInfo.session_id;
  streamingSession = !!sessionInfo.streaming;
  lastSeq = 0;
  reconnectAttempts = 0;
  uttTurns = {};
  partialEls = {};
  connectSessionWs();

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
  pendingCards = {};
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

  if (streamingSession) {
    // P2 streaming: continuous PCM to the server; the Realtime API's VAD
    // segments utterances. No MediaRecorder, no chunk cycle, no backpressure.
    convStatus.textContent = 'Listening…';
    startPcmStream();
  } else {
    // Cycle stop/start so each recording is a complete, self-contained WebM
    // file. Using timeslice produces headerless chunks that Whisper rejects.
    recordCycle();
  }
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
    elapsed += activeVad.POLL_MS;
    if (currentRms() >= activeVad.RMS_THRESHOLD) {
      sawSpeech = true;
      speechMs += activeVad.POLL_MS;
      silenceMs = 0;
    } else {
      silenceMs += activeVad.POLL_MS;
    }
    const pauseEnded = sawSpeech && speechMs >= activeVad.MIN_SPEECH_MS && silenceMs >= activeVad.SILENCE_MS;
    if (pauseEnded || elapsed >= activeVad.MAX_MS) rec.stop();
  }, activeVad.POLL_MS);

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
  if (workletNode) { try { workletNode.disconnect(); } catch (e) {} workletNode = null; }
  partialEls = {};
  streamingSession = false;
  if (audioCtx) { audioCtx.close(); audioCtx = null; analyser = null; vadBuf = null; }
  wsSessionId = null;   // before close, so onclose doesn't try to reconnect
  if (ws) { ws.close(); ws = null; }
  if (elapsedTimer) { clearInterval(elapsedTimer); elapsedTimer = null; }
  document.body.classList.remove('live');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  convStatus.textContent = '';
}
