// Data-source mode: 'live' (real backend results) or 'simulated' (clearly labelled demonstration data).
// Live is used only when the user prefers it AND the backend health check succeeds.
import { apiFetch } from './apiClient';

const KEY = 'fsa-data-mode';
const listeners = new Set();
let preferred = 'live';
try { preferred = localStorage.getItem(KEY) === 'simulated' ? 'simulated' : 'live'; } catch { /* storage unavailable */ }
let state = { preferred, backendOnline: null, mode: 'simulated' };
let inFlight = null;

const emit = () => listeners.forEach(l => l());
const update = patch => {
  state = { ...state, ...patch };
  state.mode = state.preferred === 'live' && state.backendOnline ? 'live' : 'simulated';
  emit();
};

export const dataMode = {
  get: () => state,
  isLive: () => state.mode === 'live',
  subscribe(l) { listeners.add(l); return () => listeners.delete(l); },
  setPreferred(p) { try { localStorage.setItem(KEY, p); } catch { /* ignore */ } update({ preferred: p }); if (p === 'live') this.check(); },
  check() {
    // one in-flight health check shared by every caller
    inFlight ??= apiFetch('/health', { timeout: 2500 })
      .then(() => update({ backendOnline: true }), () => update({ backendOnline: false }))
      .then(() => { inFlight = null; return state; });
    return inFlight;
  },
};
