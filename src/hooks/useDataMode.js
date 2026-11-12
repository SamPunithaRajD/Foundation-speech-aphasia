import { useEffect, useSyncExternalStore } from 'react';
import { dataMode } from '../services/dataMode';

export function useDataMode() {
  const state = useSyncExternalStore(dataMode.subscribe, dataMode.get);
  useEffect(() => { if (state.backendOnline === null) dataMode.check(); }, [state.backendOnline]);
  return { ...state, live: state.mode === 'live', setPreferred: p => dataMode.setPreferred(p) };
}
