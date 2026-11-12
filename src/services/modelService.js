import { models } from '../mock/data';
import { apiFetch } from './apiClient';
import { dataMode } from './dataMode';

const STATUS_LABEL = { ready: 'Loaded', available: 'Installed', loading: 'Loading', unavailable: 'Unavailable', error: 'Error' };

export const modelService = {
  /** Embedding models. Live entries carry the backend's real availability status and reason. */
  async list() {
    if (!dataMode.isLive()) return models.map(m => ({ ...m, simulated: true, usable: true }));
    const body = await apiFetch('/models');
    return body.embedding_models.map(m => ({
      key: m.key, name: m.name, tag: m.tag, desc: m.description, status: STATUS_LABEL[m.status] || m.status,
      usable: m.status !== 'unavailable' && m.status !== 'error', reason: m.reason, modelId: m.model_id,
      device: m.device, dimension: m.embedding_dimension, dimensionExpected: m.embedding_dimension_source === 'expected', simulated: false,
    }));
  },
  /** Full backend status (hardware, ASR, fusion, severity classifier); null in simulated mode. */
  async system() { return dataMode.isLive() ? apiFetch('/models') : null; },
};
