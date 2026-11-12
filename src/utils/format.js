export function formatDuration(seconds) {
  const value=Math.max(0,Math.floor(seconds));
  return `${String(Math.floor(value/60)).padStart(2,'0')}:${String(value%60).padStart(2,'0')}`;
}

export function formatFileSize(bytes) {
  return bytes < 1024*1024 ? `${(bytes/1024).toFixed(0)} KB` : `${(bytes/1024/1024).toFixed(2)} MB`;
}

export function formatTimestamp(seconds) {
  const s=Math.max(0,seconds);
  return `${String(Math.floor(s/60)).padStart(2,'0')}:${(s%60).toFixed(1).padStart(4,'0')}`;
}

const UNIT_SUFFIX={'ratio':'','index':'','words':'','s':'s','words/min':'/min','words/s':'/s','pauses/min':'/min','per 100 words':'%','words/sentence':'','per sentence':''};
export function formatFeatureValue(value, unit) {
  if (value==null) return '—';
  const digits=Math.abs(value)>=100?0:Math.abs(value)>=10?1:2;
  return `${Number(value.toFixed(digits))}${UNIT_SUFFIX[unit]??` ${unit}`}`;
}
