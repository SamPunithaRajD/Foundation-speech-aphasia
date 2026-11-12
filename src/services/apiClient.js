// Thin fetch wrapper for the FoundationSpeech-Aphasia research backend.
const BASE = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/, '');

export class ApiError extends Error {
  constructor(code, message, status, details) { super(message); this.code = code; this.status = status; this.details = details; }
}

export async function apiFetch(path, { method = 'GET', body, signal, raw = false, timeout } = {}) {
  const controller = new AbortController();
  const timer = timeout ? setTimeout(() => controller.abort(), timeout) : null;
  signal?.addEventListener('abort', () => controller.abort());
  let res;
  try {
    res = await fetch(`${BASE}${path}`, { method, body, signal: controller.signal });
  } catch (e) {
    throw new ApiError('BACKEND_UNREACHABLE', 'The research backend could not be reached.', 0);
  } finally { if (timer) clearTimeout(timer); }
  if (!res.ok) {
    let err = null;
    try { err = (await res.json()).error; } catch { /* non-JSON error body */ }
    throw new ApiError(err?.code || 'HTTP_ERROR', err?.message || `Request failed (${res.status}).`, res.status, err?.details);
  }
  return raw ? res : res.json();
}
