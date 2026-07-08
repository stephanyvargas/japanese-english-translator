// Firebase auth + Firestore for the Meeting Interpreter.
// Loaded as a module; exposes a small API on window.store for app.js (a plain
// script) and fires a "store-ready" event once wired. The web config below is
// public by design — security lives in Firestore rules and backend token checks.

import { initializeApp } from 'https://www.gstatic.com/firebasejs/11.0.2/firebase-app.js';
import {
  getAuth, GoogleAuthProvider, onAuthStateChanged, signInWithPopup, signOut,
} from 'https://www.gstatic.com/firebasejs/11.0.2/firebase-auth.js';
import {
  addDoc, arrayUnion, collection, deleteDoc, doc, getDoc, getDocs, getFirestore,
  limit, orderBy, query, serverTimestamp, updateDoc,
} from 'https://www.gstatic.com/firebasejs/11.0.2/firebase-firestore.js';

const app = initializeApp({
  projectId: 'japanese-translator-501010',
  appId: '1:1029193548741:web:6d0f1dd23d8bf0c47901ac',
  storageBucket: 'japanese-translator-501010.firebasestorage.app',
  apiKey: 'AIzaSyBy64LXuUYKkS676WRZYxlmZHhbaY1El4Y',
  authDomain: 'japanese-translator-501010.firebaseapp.com',
  messagingSenderId: '1029193548741',
});

const auth = getAuth(app);
const db = getFirestore(app);

function sessionsCol() {
  return collection(db, 'users', auth.currentUser.uid, 'sessions');
}

window.store = {
  onUser(cb) { onAuthStateChanged(auth, cb); },

  async signIn() {
    await signInWithPopup(auth, new GoogleAuthProvider());
  },

  async signOut() { await signOut(auth); },

  async idToken() {
    return auth.currentUser ? auth.currentUser.getIdToken() : '';
  },

  // ── conversation persistence ──────────────────────────────────────────────
  // Write failures log to the console and never block translation.

  async startSession(meta) {
    try {
      const ref = await addDoc(sessionsCol(), {
        ...meta, startedAt: serverTimestamp(), turnCount: 0, turns: [], preview: '',
      });
      return ref.id;
    } catch (err) {
      console.warn('startSession failed:', err);
      return '';
    }
  },

  async saveTurn(sessionId, turn) {
    if (!sessionId) return;
    try {
      const update = { turns: arrayUnion(turn) };
      if (turn.seq === 1 || turn.first) update.preview = turn.english;
      await updateDoc(doc(sessionsCol(), sessionId), update);
    } catch (err) {
      console.warn('saveTurn failed:', err);
    }
  },

  async endSession(sessionId, turnCount) {
    if (!sessionId) return;
    try {
      await updateDoc(doc(sessionsCol(), sessionId), {
        endedAt: serverTimestamp(), turnCount,
      });
    } catch (err) {
      console.warn('endSession failed:', err);
    }
  },

  async renameSession(sessionId, title) {
    await updateDoc(doc(sessionsCol(), sessionId), { title: title.trim() });
  },

  async deleteSession(sessionId) {
    await deleteDoc(doc(sessionsCol(), sessionId));
  },

  // ── history retrieval ─────────────────────────────────────────────────────

  async listSessions(max = 20) {
    const snap = await getDocs(query(sessionsCol(), orderBy('startedAt', 'desc'), limit(max)));
    return snap.docs.map(d => ({ id: d.id, ...d.data() }));
  },

  async getSession(id) {
    const snap = await getDoc(doc(sessionsCol(), id));
    return snap.exists() ? { id: snap.id, ...snap.data() } : null;
  },
};

document.dispatchEvent(new Event('store-ready'));
