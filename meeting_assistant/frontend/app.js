// ── DOM refs ─────────────────────────────────────────────────────────────────
const modelSel    = document.getElementById('model');
const backendUrl  = document.getElementById('backendUrl');
const contextEl   = document.getElementById('context');
const chat        = document.getElementById('chat');
const messageEl   = document.getElementById('message');
const sendBtn     = document.getElementById('sendBtn');
const clearBtn    = document.getElementById('clearBtn');
const statusEl    = document.getElementById('status');

// Conversation history sent back to the server for multi-turn context.
const history = [];

function apiBase() { return backendUrl.value.replace(/\/$/, ''); }

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function linkify(s) {
  return s.replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
}

// Render reply text, coloring JA:/Kana:/EN: question lines.
function renderBody(text) {
  return escHtml(text).split('\n').map(line => {
    const m = line.match(/^\s*(JA|Kana|EN)\s*[:：]\s*(.*)$/);
    if (m) {
      const cls = m[1].toLowerCase();
      return `<span class="${cls}"><strong>${m[1]}:</strong> ${linkify(m[2])}</span>`;
    }
    return linkify(line);
  }).join('\n');
}

function addBubble(role, html) {
  const div = document.createElement('div');
  div.className = `bubble ${role}`;
  div.innerHTML = `<div class="role">${role === 'user' ? 'You' : 'Assistant'}</div><div class="body">${html}</div>`;
  chat.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return div;
}

function sourcesHtml(sources, searched) {
  let html = '';
  if (searched && searched.length) {
    html += `<div class="searched">Searched: ${searched.map(escHtml).join(' · ')}</div>`;
  }
  if (sources && sources.length) {
    html += '<div class="sources"><h4>Sources</h4><ol>';
    for (const s of sources) {
      const title = escHtml(s.title || s.link);
      const link = escHtml(s.link);
      html += `<li><a href="${link}" target="_blank" rel="noopener">${title}</a>`;
      if (s.snippet) html += `<span class="snip">${escHtml(s.snippet)}</span>`;
      html += '</li>';
    }
    html += '</ol></div>';
  }
  return html;
}

async function send() {
  const message = messageEl.value.trim();
  if (!message) return;

  addBubble('user', escHtml(message));
  messageEl.value = '';
  sendBtn.disabled = true;
  statusEl.textContent = 'Thinking…';

  try {
    const res = await fetch(`${apiBase()}/assistant`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        context: contextEl.value.trim(),
        model: modelSel.value,
        history,
      }),
    });

    if (!res.ok) {
      const err = await res.text();
      addBubble('assistant', `<span style="color:#e05555">Error: ${escHtml(err)}</span>`);
      return;
    }

    const data = await res.json();
    const bubble = addBubble('assistant', renderBody(data.reply));
    const extra = sourcesHtml(data.sources, data.searched);
    if (extra) bubble.querySelector('.body').insertAdjacentHTML('beforeend', extra);

    history.push({ role: 'user', text: message });
    history.push({ role: 'assistant', text: data.reply });
  } catch (err) {
    addBubble('assistant', `<span style="color:#e05555">Error: ${escHtml(err.message)}</span>`);
  } finally {
    sendBtn.disabled = false;
    statusEl.textContent = '';
  }
}

sendBtn.addEventListener('click', send);

// Enter sends, Shift+Enter newlines.
messageEl.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

clearBtn.addEventListener('click', () => {
  chat.innerHTML = '';
  history.length = 0;
});

// Example chips
document.querySelectorAll('.chip').forEach(chip => {
  chip.addEventListener('click', () => {
    messageEl.value = chip.dataset.q;
    messageEl.focus();
  });
});
