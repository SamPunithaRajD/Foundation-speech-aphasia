"""Feature extraction + caching for training, reusing the application's own pipeline code
(decode → preprocess/VAD → Whisper ASR → clinical language features → foundation-model embedding).

Feature groups:
    acoustic          handcrafted acoustic features (app/features/acoustic.py)
    language          clinical language features (app/features/clinical.py)
    embedding:<key>   mean-pooled foundation-model embedding (app/models/registry.py)

Caching is per record AND per unit (preprocessing info, ASR transcript, each feature group), each
keyed by the settings that affect it. Experiments therefore share work: Models A, B and C never
recompute the same ASR transcript, acoustic features or embedding. Whisper only runs when a
whisper-based language group is requested.

Missing values stay NaN in the cache (with the reason recorded). Imputation happens later,
inside the model pipeline, fitted on the training split only.

``extraction_spec`` is the single description of every setting that affects feature values. It is
stored in the model bundle and compared with the runtime configuration before any prediction.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import Settings
from app.features.acoustic import PROVENANCE as ACOUSTIC_PROVENANCE
from app.features.acoustic import acoustic_feature_names, acoustic_features
from app.features.clinical import FEATURE_ORDER, extract_features
from app.models.registry import SPEC_BY_KEY, ModelRegistry
from app.preprocessing.ingest import DecodedAudio, decode_audio
from app.preprocessing.pipeline import preprocess
from app.schemas.analysis import PreprocessingResult, TranscriptResult, TranscriptSegment
from app.schemas.common import SectionState
from app.training.dataset import DatasetRecord

log = logging.getLogger(__name__)
FEATURE_CODE_VERSION = "features/2.0"
# syntactic_complexity is never computed (no parser) and is therefore excluded from model inputs
LANGUAGE_FEATURES = [n for n in FEATURE_ORDER if n != "syntactic_complexity"]


@dataclass
class FeatureConfig:
    groups: list[str]  # e.g. ["acoustic", "language", "embedding:wav2vec2"]
    language_source: str = "whisper"  # whisper (deployment-consistent) | human (CHAT transcript)
    audio_scope: str = "speaker_utterances"
    exclude_features: list[str] = field(default_factory=list)  # named features to leave out (e.g. length confounds)

    def embedding_keys(self) -> list[str]:
        keys = [g.split(":", 1)[1] for g in self.groups if g.startswith("embedding:")]
        unknown = [k for k in keys if k not in SPEC_BY_KEY]
        if unknown:
            raise ValueError(f"Unknown embedding model(s): {unknown}")
        return keys

    def validate(self) -> None:
        if not self.groups or len(set(self.groups)) != len(self.groups):
            raise ValueError("feature groups must be non-empty and unique")
        for g in self.groups:
            if g not in ("acoustic", "language") and not g.startswith("embedding:"):
                raise ValueError(f"Unknown feature group '{g}'")
        if self.language_source not in ("whisper", "human"):
            raise ValueError("language_source must be 'whisper' or 'human'")
        known = set(acoustic_feature_names()) | set(LANGUAGE_FEATURES)
        bad = [n for n in self.exclude_features if n not in known]
        if bad:
            raise ValueError(f"exclude_features contains unknown feature names: {bad}")
        self.embedding_keys()


def preprocessing_spec(s: Settings) -> dict:
    return {"sr": s.target_sample_rate, "normalize_peak_dbfs": s.normalize_peak_dbfs, "highpass_hz": s.highpass_hz,
            "vad": [s.vad_enabled, s.vad_aggressiveness, s.vad_frame_ms, s.vad_merge_gap_ms, s.vad_min_segment_ms]}


def extraction_spec(s: Settings, registry: ModelRegistry, groups: list[str], language_source: str) -> dict:
    """Everything that changes feature VALUES at inference time, for exactly the given groups."""
    spec: dict = {"code": FEATURE_CODE_VERSION, "preprocessing": preprocessing_spec(s)}
    if "acoustic" in groups:
        spec["acoustic"] = ACOUSTIC_PROVENANCE
    if "language" in groups:
        spec["language"] = {"source": language_source, "pause_threshold_ms": s.pause_threshold_ms,
                            "mattr_window": s.mattr_window, "features": LANGUAGE_FEATURES}
        if language_source == "whisper":
            spec["asr"] = {"model": s.asr_model, "language": s.asr_language, "word_timestamps": s.asr_word_timestamps}
    for g in groups:
        if g.startswith("embedding:"):
            key = g.split(":", 1)[1]
            spec[g] = {"model_id": registry.model_id(SPEC_BY_KEY[key]), "max_seconds": s.max_embedding_seconds,
                       "pooling": "mean_over_frames"}
    return spec


def spec_differences(trained: dict, runtime: dict) -> list[str]:
    keys = sorted(set(trained) | set(runtime))
    return [f"{k}: trained with {json.dumps(trained.get(k))}, running with {json.dumps(runtime.get(k))}"
            for k in keys if trained.get(k) != runtime.get(k)]


def group_names(group: str, dims: dict[str, int] | None = None) -> list[str]:
    if group == "acoustic":
        return acoustic_feature_names()
    if group == "language":
        return list(LANGUAGE_FEATURES)
    key = group.split(":", 1)[1]
    dim = (dims or {}).get(group)
    if dim is None:
        raise ValueError(f"embedding dimension for {group} unknown")
    return [f"{key}[{i}]" for i in range(dim)]


def _cut_speaker_audio(audio: DecodedAudio, segments_ms: list[tuple[int, int]]) -> DecodedAudio:
    sr = audio.sample_rate
    parts = [audio.samples[:, int(s * sr / 1000):int(e * sr / 1000)] for s, e in segments_ms if e > s]
    parts = [p for p in parts if p.shape[1] > 0]
    if not parts:
        raise ValueError("speaker segments contain no audio")
    return DecodedAudio(np.concatenate(parts, axis=1), sr, audio.decoder)


def _human_transcript_result(text: str) -> TranscriptResult:
    return TranscriptResult(state=SectionState.COMPUTED if text.strip() else SectionState.INSUFFICIENT_DATA,
                            model="human-CHAT", engine="CHAT transcript", source="human CHAT transcript", text=text,
                            segments=[TranscriptSegment(id=0, start=0.0, end=0.0, text=text)] if text.strip() else [],
                            word_count=len(text.split()))


def _h(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


class FeatureExtractor:
    def __init__(self, settings: Settings, cfg: FeatureConfig, cache_root: Path, dataset_version: str,
                 registry: ModelRegistry | None = None):
        cfg.validate()
        self.settings, self.cfg = settings, cfg
        self.registry = registry or ModelRegistry(settings)
        self.root = Path(cache_root) / dataset_version
        self.spec = extraction_spec(settings, self.registry, cfg.groups, cfg.language_source)
        base = {"code": FEATURE_CODE_VERSION, "pre": preprocessing_spec(settings), "audio_scope": cfg.audio_scope}
        s = settings
        asr = {**base, "asr": [s.asr_model, s.asr_language, s.asr_word_timestamps]}
        self._unit_specs: dict[str, dict] = {
            "pre": base,
            "acoustic": {**base, "acoustic": ACOUSTIC_PROVENANCE},
            "asr": asr,
            "language": {**(asr if cfg.language_source == "whisper" else base), "source": cfg.language_source,
                         "pause": s.pause_threshold_ms, "mattr": s.mattr_window, "features": LANGUAGE_FEATURES},
        }
        for key in cfg.embedding_keys():
            self._unit_specs[f"embedding:{key}"] = {**base, "id": self.registry.model_id(SPEC_BY_KEY[key]),
                                                    "max_seconds": s.max_embedding_seconds}
        self.stats = {"computed": 0, "cache_hits": 0}

    # ---------------------------------------------------------------- cache
    def _path(self, unit: str, rec: DatasetRecord) -> Path:
        d = self.root / f"{unit.replace(':', '-')}-{_h(self._unit_specs[unit])}"
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            (d / "unit_spec.json").write_text(json.dumps(self._unit_specs[unit], indent=1, default=str))
        return d / (hashlib.sha256(rec.record_id.encode()).hexdigest()[:20] + ".json")

    def _load(self, unit: str, rec: DatasetRecord):
        p = self._path(unit, rec)
        if p.exists():
            self.stats["cache_hits"] += 1
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    def _store(self, unit: str, rec: DatasetRecord, payload: dict) -> dict:
        self._path(unit, rec).write_text(json.dumps(payload), encoding="utf-8")
        self.stats["computed"] += 1
        return payload

    # ---------------------------------------------------------------- extraction
    def extract(self, rec: DatasetRecord) -> dict:
        cfg, s = self.cfg, self.settings
        state: dict = {}

        def audio():  # decoded + preprocessed lazily, at most once per record, only when a unit is not cached
            if "pre" not in state:
                a = decode_audio(Path(rec.audio_path))
                if cfg.audio_scope == "speaker_utterances":
                    a = _cut_speaker_audio(a, rec.speaker_segments_ms)
                state["pre"] = preprocess(a, s)
            return state["pre"]

        pre_info = self._load("pre", rec) or self._store("pre", rec, audio().result.model_dump(mode="json"))
        pre_result = PreprocessingResult.model_validate(pre_info)
        meta = {"record_id": rec.record_id, "duration_s": pre_result.duration_seconds,
                "speech_s": pre_result.speech_duration_seconds, "human_transcript": rec.human_transcript,
                "whisper_transcript": None, "missing": {}}
        out: dict[str, np.ndarray] = {}

        if "acoustic" in cfg.groups:
            c = self._load("acoustic", rec)
            if c is None:
                feats = acoustic_features(audio().waveform, audio().sample_rate)
                c = self._store("acoustic", rec, {"values": [f.value for f in feats],
                                                  "missing": {f.name: f.reason for f in feats if f.value is None}})
            out["acoustic"] = np.array([np.nan if v is None else v for v in c["values"]], dtype=np.float64)
            meta["missing"]["acoustic"] = c["missing"]

        if "language" in cfg.groups:
            c = self._load("language", rec)
            if c is None:
                if cfg.language_source == "whisper":
                    a = self._load("asr", rec)
                    if a is None:
                        from app.pipelines.asr import transcribe

                        a = self._store("asr", rec, transcribe(self.registry, audio().waveform, audio().sample_rate, s)
                                        .model_dump(mode="json"))
                    tr = TranscriptResult.model_validate(a)
                else:
                    tr = _human_transcript_result(rec.human_transcript or "")
                by = {f.name: f for f in extract_features(tr, pre_result, s).features}
                c = self._store("language", rec, {
                    "values": [by[n].value if n in by and by[n].status == "computed" else None for n in LANGUAGE_FEATURES],
                    "missing": {n: (by[n].reason if n in by else "not produced") for n in LANGUAGE_FEATURES
                                if n not in by or by[n].status != "computed"},
                    "transcript": tr.text if cfg.language_source == "whisper" else None,
                    "asr_quality": tr.quality if cfg.language_source == "whisper" else None})
            out["language"] = np.array([np.nan if v is None else v for v in c["values"]], dtype=np.float64)
            meta["missing"]["language"] = c["missing"]
            meta["whisper_transcript"] = c.get("transcript")
            meta["asr_quality"] = c.get("asr_quality")

        for key in cfg.embedding_keys():
            unit = f"embedding:{key}"
            c = self._load(unit, rec)
            if c is None:
                emb = self.registry.extract_embedding(key, audio().waveform, audio().sample_rate)
                vec = np.asarray(emb["vector"], dtype=np.float64)
                if not np.all(np.isfinite(vec)):
                    raise ValueError(f"non-finite embedding from {key}")
                c = self._store(unit, rec, {"vector": vec.tolist(), "frames": emb["frames"]})
            out[unit] = np.asarray(c["vector"], dtype=np.float64)
        return {"meta": meta, **out}


@dataclass
class FeatureTable:
    record_ids: list[str]
    participant_ids: list[str]
    y: np.ndarray  # WAB-R-derived severity score
    aq: np.ndarray
    groups: dict[str, np.ndarray]
    names: dict[str, list[str]]
    failures: dict[str, str] = field(default_factory=dict)
    whisper_transcripts: dict[str, str | None] = field(default_factory=dict)
    human_transcripts: dict[str, str | None] = field(default_factory=dict)

    def subset(self, participants: list[str]) -> "FeatureTable":
        ids = set(participants)
        idx = [i for i, p in enumerate(self.participant_ids) if p in ids]
        return FeatureTable([self.record_ids[i] for i in idx], [self.participant_ids[i] for i in idx],
                            self.y[idx], self.aq[idx], {g: m[idx] for g, m in self.groups.items()}, self.names)


def build_table(records: list[DatasetRecord], extractor: FeatureExtractor) -> FeatureTable:
    rows, failures = [], {}
    for rec in records:
        if not rec.trainable:
            continue
        try:
            rows.append((rec, extractor.extract(rec)))
        except Exception as exc:  # reported, never silently dropped
            failures[rec.record_id] = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("feature extraction failed for %s: %s", rec.record_id, failures[rec.record_id])
    if not rows:
        raise RuntimeError(f"No records produced features. Failures: {len(failures)}")
    groups, names = {}, {}
    exclude = set(extractor.cfg.exclude_features)
    for g in extractor.cfg.groups:
        mat = np.vstack([f[g] for _, f in rows])
        full = group_names(g, dims={g: mat.shape[1]})
        keep = [i for i, n in enumerate(full) if n not in exclude]
        groups[g], names[g] = mat[:, keep], [full[i] for i in keep]
    return FeatureTable(
        record_ids=[r.record_id for r, _ in rows], participant_ids=[r.participant_id for r, _ in rows],
        y=np.array([r.severity_score for r, _ in rows], dtype=np.float64),
        aq=np.array([r.wab_aq for r, _ in rows], dtype=np.float64), groups=groups, names=names, failures=failures,
        whisper_transcripts={r.record_id: f["meta"].get("whisper_transcript") for r, f in rows},
        human_transcripts={r.record_id: r.human_transcript for r, _ in rows})
