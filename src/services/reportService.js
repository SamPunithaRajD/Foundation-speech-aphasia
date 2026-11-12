import { formalProjectTitle } from '../types/index.js';
import { apiFetch } from './apiClient';

const save = (blob, name) => { const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000); };

export const reportService = {
  async preview(result) { return { ...result, title: formalProjectTitle }; },
  /** Live results: backend-generated PDF. Simulated results: a labelled demonstration text summary. */
  async download(result) {
    if (result && !result.simulated) {
      const res = await apiFetch(`/reports/${encodeURIComponent(result.id)}`, { raw: true });
      save(await res.blob(), `foundationspeech-report-${result.id}.pdf`);
      return;
    }
    const sev = result?.severity;
    const body = `${formalProjectTitle}\nFoundationSpeech-Aphasia | Research Prototype\n${result?.id ?? 'FS-DEMO-024'}\nSIMULATED / DEMONSTRATION DATA — NOT A CLINICAL RESULT\nPredicted class: ${sev?.label ?? 'Moderate'}\nConfidence: ${sev?.confidence ?? 82.4}%\n\nPrototype only. Do not use for diagnosis or treatment decisions.`;
    save(new Blob([body], { type: 'text/plain' }), 'foundation-speech-report.txt');
  }
};
