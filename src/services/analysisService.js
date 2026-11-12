import { historyRows, metrics, transcript } from '../mock/data';
import { apiFetch } from './apiClient';
import { dataMode } from './dataMode';
import { formatDuration, formatFeatureValue, formatTimestamp } from '../utils/format';

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const TERMINAL = ['COMPLETED', 'ATTENTION_NEEDED', 'FAILED', 'CANCELLED'];
const FILLERS = new Set(['um', 'umm', 'uh', 'uhh', 'uhm', 'er', 'erm', 'ah', 'hmm', 'mm', 'mhm']);
const MODEL_NAMES = { wav2vec2: 'wav2vec 2.0', hubert: 'HuBERT', whisper_encoder: 'Whisper Encoder', wavlm: 'WavLM' };
const STATUS_LABELS = { COMPLETED: 'Complete', ATTENTION_NEEDED: 'Attention needed', FAILED: 'Failed', CANCELLED: 'Cancelled' };

// ---------------------------------------------------------------- simulated (demonstration) data
const SIM_STAGES = ['Preparing audio','Extracting speech representations','Transcribing speech','Analyzing clinical language','Combining features','Estimating severity','Preparing explanation'];

export function simulatedResult(filename) {
  return {
    id: 'FS-DEMO-024', filename: filename ?? 'research-sample.wav', duration: '02:18', simulated: true, mode: 'simulated',
    createdAt: '2026-09-24T11:42:00Z', analyzedLabel: '24 Sep 2026 · 11:42', status: 'COMPLETED', statusLabel: 'Workflow complete',
    strategyLabel: 'Multi-model', attention: [],
    severity: { available: true, state: 'SIMULATED', label: 'Moderate', confidence: 82.4,
      probabilities: [['Mild', 12.8], ['Moderate', 82.4], ['Severe', 4.8]], message: 'Simulated output. Not a model prediction.' },
    transcript: { state: 'SIMULATED', lines: transcript, note: 'Word-level confidence shown where available' },
    metrics: metrics.map(([label, value, explanation, note]) => ({ label, value, explanation, note, computed: true })),
    featuresNote: 'Illustrative feature values for interface demonstration only.',
    xai: { available: true, method: 'SIMULATED', rows: [['Pause duration','+0.28','positive',83],['Speech disfluency','+0.21','positive',67],['Word repetition','+0.15','positive',53],['Lexical diversity','−0.12','negative',42],['Speech fluency','−0.08','negative',31]] },
    embeddings: [], fusion: null, preprocessing: null,
  };
}

// ---------------------------------------------------------------- live backend -> view model
function transcriptLines(tr) {
  return (tr.segments || []).map(s => {
    const toks = s.text.toLowerCase().match(/[a-z0-9']+/g) || [];
    const rep = toks.some((t, i) => i > 0 && t === toks[i - 1] && !FILLERS.has(t));
    const fill = toks.some(t => FILLERS.has(t));
    return {
      time: formatTimestamp(s.start), text: s.text,
      confidence: s.mean_word_probability == null ? null : Math.round(s.mean_word_probability * 100),
      flag: rep ? 'Repetition' : fill ? 'Filled pause' : undefined,
    };
  });
}

function xaiView(ex) {
  const method = (ex.methods || []).find(m => m.state === 'COMPUTED' && m.attributions?.length);
  if (!method) return { available: false, state: ex.state, message: ex.message, note: ex.interpretation_note };
  const max = Math.max(...method.attributions.map(a => Math.abs(a.contribution))) || 1;
  return {
    available: true, method: method.method.toUpperCase() + (method.reliable === false ? ' · UNRELIABLE' : ''), target: method.target, note: ex.interpretation_note,
    rows: method.attributions.slice(0, 6).map(a => [a.feature.replaceAll('_', ' '), `${a.contribution >= 0 ? '+' : '−'}${Math.abs(a.contribution).toFixed(3)}`, a.direction, Math.round(Math.abs(a.contribution) / max * 90)]),
  };
}

export function toViewModel(r) {
  const sv = r.severity || {};
  const models = r.config?.models || [];
  return {
    id: r.analysis_id, filename: r.audio?.filename ?? 'speech-sample', simulated: false, mode: 'live',
    duration: r.audio ? formatDuration(r.audio.duration_seconds) : '—', durationSeconds: r.audio?.duration_seconds,
    createdAt: r.created_at, analyzedLabel: new Date(r.created_at).toLocaleString(undefined, { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }),
    status: r.status, statusLabel: STATUS_LABELS[r.status] || r.status, attention: r.attention || [], error: r.error,
    strategyLabel: r.config?.strategy === 'multi' ? 'Multi-model' : `Single · ${models.map(m => MODEL_NAMES[m] || m).join(', ')}`,
    severity: {
      // WAB-R-derived severity score (regression). No class probabilities exist in this formulation.
      available: !!sv.prediction_available, state: sv.state, message: sv.message, label: sv.severity_band ?? null,
      score: sv.severity_score ?? null, predictedAq: sv.predicted_aq ?? null, uncertainty: sv.uncertainty ?? null,
      clinicalValidity: sv.clinical_validity ?? null, modelVersion: sv.model_version ?? null,
      targetDefinition: sv.target_definition ?? null, probabilities: null, confidence: null,
      pilot: sv.model_stage === 'PILOT', evaluation: sv.evaluation_summary ?? null, limitations: sv.limitations ?? [],
    },
    transcript: {
      state: r.transcript?.state, message: r.transcript?.message, model: r.transcript?.model, language: r.transcript?.language,
      lines: transcriptLines(r.transcript || {}),
      note: (r.transcript?.confidence ? 'Segment values: mean Whisper word token probability (uncalibrated)' : 'No confidence values reported by the ASR model')
        + (r.transcript?.quality?.flags?.length ? ` · ASR flagged: ${r.transcript.quality.flags.join('; ')}` : '')
        + (r.transcript?.state === 'COMPUTED' ? ' · unverified ASR output' : ''),
    },
    metrics: (r.features?.features || []).map(f => ({
      label: f.label, value: f.status === 'computed' ? formatFeatureValue(f.value, f.unit) : '—', computed: f.status === 'computed',
      explanation: `${f.definition}${f.reason ? ` — ${f.reason}` : ''}${f.note ? ` (${f.note})` : ''}`,
      note: f.status === 'computed' ? `${f.source} · computed` : f.status.replace('_', ' '),
    })),
    featuresNote: r.features?.disclaimer || 'Features not computed.',
    xai: xaiView(r.explainability || {}),
    embeddings: (r.embeddings?.results || []).map(e => ({ name: MODEL_NAMES[e.model_key] || e.model_key, state: e.state, dimension: e.dimension, message: e.message })),
    fusion: r.fusion, preprocessing: r.preprocessing,
  };
}

// ---------------------------------------------------------------- service
export const analysisService = {
  /** Run an analysis. onProgress receives {status, stages, percent, simulated}. */
  async analyze(file, onProgress, { strategy = 'single', models = [] } = {}) {
    if (!dataMode.isLive()) {
      for (const [i, stage] of SIM_STAGES.entries()) { onProgress?.({ simulated: true, step: i + 1, total: SIM_STAGES.length, label: stage }); await wait(640); }
      return { ...simulatedResult(file?.name), createdAt: new Date().toISOString() };
    }
    const fd = new FormData();
    fd.append('file', file);
    fd.append('strategy', strategy);
    if (models.length) fd.append('models', models.join(','));
    const created = await apiFetch('/analysis', { method: 'POST', body: fd });
    for (;;) {
      const st = await apiFetch(`/analysis/${created.analysis_id}/status`);
      onProgress?.({ simulated: false, status: st.status, stages: st.stages, percent: st.progress.percent });
      if (TERMINAL.includes(st.status)) break;
      await wait(700);
    }
    return this.get(created.analysis_id);
  },
  async get(id) { return toViewModel(await apiFetch(`/analysis/${encodeURIComponent(id)}`)); },
  async history() {
    if (!dataMode.isLive()) return historyRows.map(r => ({ ...r, simulated: true }));
    const { items } = await apiFetch('/history');
    return items.map(i => ({
      id: i.analysis_id, simulated: false,
      date: new Date(i.created_at).toLocaleDateString(undefined, { day: '2-digit', month: 'short', year: 'numeric' }),
      duration: i.duration_seconds == null ? '—' : formatDuration(i.duration_seconds),
      severity: i.severity_band ?? null, severityState: i.severity_state,
      confidence: i.severity_score == null ? '—' : `${i.severity_score.toFixed(1)} / 100`,
      model: i.strategy === 'multi' ? 'Multi-model' : i.models.map(m => MODEL_NAMES[m] || m).join(', '),
      status: STATUS_LABELS[i.status] || 'In progress',
    }));
  },
  async result() { return simulatedResult(); },
};
