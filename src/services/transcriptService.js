import { transcript } from '../mock/data';
import { apiFetch } from './apiClient';
import { dataMode } from './dataMode';

export const transcriptService = {
  async get(analysisId) {
    if (!dataMode.isLive() || !analysisId) return transcript;
    return apiFetch(`/analysis/${encodeURIComponent(analysisId)}/transcript`);
  },
};
