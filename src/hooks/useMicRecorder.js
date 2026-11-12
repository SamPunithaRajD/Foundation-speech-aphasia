import { useCallback, useEffect, useRef, useState } from 'react';

// Container formats the backend accepts, in order of preference; the browser picks the first it supports
// (Chrome/Edge: webm/opus, Firefox: ogg/opus or webm, Safari: mp4/aac).
const FORMATS = [['audio/webm;codecs=opus', 'webm'], ['audio/ogg;codecs=opus', 'ogg'], ['audio/webm', 'webm'], ['audio/mp4', 'm4a']];

// Browser speech DSP (noise suppression, auto gain, echo cancellation) is disabled so the recorded signal is
// the raw microphone input the research pipeline expects.
const CONSTRAINTS = { audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } };

function micError(e) {
  if (!navigator.mediaDevices?.getUserMedia) return 'This browser cannot access a microphone here. Use Chrome, Edge or Firefox over http://127.0.0.1 or https.';
  if (e?.name === 'NotAllowedError' || e?.name === 'SecurityError') return 'Microphone permission was denied. Allow microphone access in the browser address bar and try again.';
  if (e?.name === 'NotFoundError' || e?.name === 'OverconstrainedError') return 'No microphone was found. Connect or enable a microphone and try again.';
  if (e?.name === 'NotReadableError') return 'The microphone is in use by another application or blocked by the system.';
  return `Microphone could not be started (${e?.name || 'unknown error'}).`;
}

/** Real microphone capture via getUserMedia + MediaRecorder, with a live input-level meter. */
export function useMicRecorder() {
  const [status, setStatus] = useState('idle'); // idle | requesting | recording | paused
  const [seconds, setSeconds] = useState(0);
  const [level, setLevel] = useState(0); // 0..1 RMS-based input level
  const [error, setError] = useState('');
  const [heardSound, setHeardSound] = useState(false);
  const r = useRef({});

  const cleanup = useCallback(() => {
    const s = r.current;
    cancelAnimationFrame(s.raf); clearInterval(s.timer);
    s.stream?.getTracks().forEach(t => t.stop());
    s.ctx?.close().catch(() => {});
    r.current = {};
    setLevel(0);
  }, []);
  useEffect(() => cleanup, [cleanup]);

  const start = useCallback(async () => {
    setError(''); setSeconds(0); setHeardSound(false);
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') { setError(micError()); return false; }
    setStatus('requesting');
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia(CONSTRAINTS); }
    catch (e) { setStatus('idle'); setError(micError(e)); return false; }
    const [mimeType, ext] = FORMATS.find(([m]) => MediaRecorder.isTypeSupported(m)) || ['', 'webm'];
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };

    // live level meter
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser(); analyser.fftSize = 1024;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const buf = new Float32Array(analyser.fftSize);
    const tick = () => {
      analyser.getFloatTimeDomainData(buf);
      const rms = Math.sqrt(buf.reduce((a, v) => a + v * v, 0) / buf.length);
      const lvl = Math.min(1, rms * 8);
      const now = performance.now();
      if (now - (r.current.lastLevel || 0) > 100) { r.current.lastLevel = now; setLevel(lvl); } // ~10 re-renders/s
      if (lvl > 0.04) setHeardSound(true);
      r.current.raf = requestAnimationFrame(tick);
    };

    r.current = { stream, recorder, chunks, ctx, mimeType: recorder.mimeType || mimeType || 'audio/webm', ext };
    recorder.start(250);
    r.current.timer = setInterval(() => { if (r.current.recorder?.state === 'recording') setSeconds(v => v + 1); }, 1000);
    tick();
    setStatus('recording');
    return true;
  }, []);

  const pause = useCallback(() => { r.current.recorder?.pause(); setStatus('paused'); }, []);
  const resume = useCallback(() => { r.current.recorder?.resume(); setStatus('recording'); }, []);

  /** Stops recording and resolves to a File ready for upload (or null if nothing was captured). */
  const stop = useCallback(() => new Promise(resolve => {
    const { recorder, chunks, mimeType, ext } = r.current;
    if (!recorder || recorder.state === 'inactive') { cleanup(); setStatus('idle'); resolve(null); return; }
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
      cleanup(); setStatus('idle');
      resolve(blob.size ? new File([blob], `browser-recording.${ext}`, { type: blob.type }) : null);
    };
    recorder.stop();
  }), [cleanup]);

  return { status, recording: status === 'recording' || status === 'paused', paused: status === 'paused', seconds, level, heardSound, error, setError, start, pause, resume, stop };
}
