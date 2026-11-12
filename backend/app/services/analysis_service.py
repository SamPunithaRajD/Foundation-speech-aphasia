"""Analysis job lifecycle.

QUEUED -> PREPROCESSING -> MODEL_PROCESSING -> ASR -> FEATURE_EXTRACTION -> FUSION
       -> PREDICTION -> EXPLAINABILITY -> COMPLETED | ATTENTION_NEEDED | FAILED | CANCELLED

Uploaded audio is decoded synchronously (so validation errors surface as HTTP
errors), then the file is deleted immediately unless ``FSA_RETAIN_AUDIO=true``.
The decoded waveform only lives in memory for the duration of the job.
Logs contain analysis IDs and stage names only – never transcript text or audio.
"""
from __future__ import annotations

import logging
import secrets
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

import numpy as np

from app.config import Settings
from app.database.repository import AnalysisRepository
from app.explainability.explain import explain
from app.features.clinical import FEATURE_ORDER, extract_features, feature_vector
from app.models.registry import SPEC_BY_KEY, ModelRegistry, ModelUnavailable
from app.pipelines import asr as asr_pipeline
from app.pipelines.fusion import FusedRepresentation, fuse
from app.pipelines.severity import SeverityService
from app.preprocessing.ingest import DecodedAudio, decode_audio, stream_to_file, validate_decoded, validate_extension
from app.preprocessing.pipeline import preprocess
from app.schemas.analysis import (AnalysisConfig, AnalysisRecord, AudioMetadata, EmbeddingResult, EmbeddingsResult,
                                  ExplainabilityResult, FeaturesResult, TranscriptResult)
from app.schemas.common import PIPELINE_STAGES, TERMINAL_STATUSES, AnalysisStatus, SectionState, StageRecord, StageStatus
from app.utils.errors import ApiError, ErrorCode

log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_analysis_id() -> str:
    return f"FS-{datetime.now(timezone.utc):%y%m%d}-{secrets.token_hex(3).upper()}"


class Cancelled(Exception):
    pass


class AnalysisService:
    def __init__(self, settings: Settings, repo: AnalysisRepository, registry: ModelRegistry, severity: SeverityService):
        self.settings = settings
        self.repo = repo
        self.registry = registry
        self.severity = severity
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_jobs, thread_name_prefix="fsa-job")
        self._cancel: dict[str, threading.Event] = {}
        settings.temp_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ housekeeping
    def startup_housekeeping(self) -> None:
        """Remove stale temp workspaces, apply retention, and mark interrupted jobs as FAILED."""
        for child in self.settings.temp_dir.iterdir():
            if child.is_dir() and not self.settings.retain_audio:
                shutil.rmtree(child, ignore_errors=True)
        purged = self.repo.purge_older_than(self.settings.retention_days)
        if purged:
            log.info("retention: purged %d analyses older than %d days", purged, self.settings.retention_days)
        records, _ = self.repo.list(limit=10_000)
        for r in records:
            if r.status not in TERMINAL_STATUSES:
                r.status = AnalysisStatus.FAILED
                r.error = {"code": ErrorCode.INFERENCE_FAILED.value,
                           "message": "The analysis was interrupted by a server restart."}
                r.updated_at = now()
                self.repo.save(r)

    def shutdown(self) -> None:
        for ev in self._cancel.values():
            ev.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------ creation
    def resolve_models(self, strategy: str, models: list[str]) -> list[str]:
        if strategy not in ("single", "multi"):
            raise ApiError(ErrorCode.INVALID_REQUEST, "strategy must be 'single' or 'multi'.", 422)
        unknown = [m for m in models if m not in SPEC_BY_KEY]
        if unknown:
            raise ApiError(ErrorCode.INVALID_REQUEST, f"Unknown model key(s): {', '.join(unknown)}.", 422,
                           {"known": list(SPEC_BY_KEY)})
        if strategy == "multi":
            return models or list(SPEC_BY_KEY)
        chosen = models[:1] or self.settings.default_embedding_models[:1]
        return chosen

    def create(self, stream: BinaryIO, filename: str, strategy: str, models: list[str]) -> AnalysisRecord:
        ext = validate_extension(filename, self.settings)
        model_keys = self.resolve_models(strategy, models)
        analysis_id = new_analysis_id()
        workdir = self.settings.temp_dir / analysis_id
        workdir.mkdir(parents=True, exist_ok=True)
        upload_path = workdir / f"upload{ext}"
        try:
            size = stream_to_file(stream, upload_path, self.settings)
            audio = decode_audio(upload_path)
            validate_decoded(audio, self.settings)
        except BaseException:
            shutil.rmtree(workdir, ignore_errors=True)
            raise
        if not self.settings.retain_audio:
            shutil.rmtree(workdir, ignore_errors=True)

        ts = now()
        record = AnalysisRecord(
            analysis_id=analysis_id,
            status=AnalysisStatus.QUEUED,
            created_at=ts,
            updated_at=ts,
            config=AnalysisConfig(strategy=strategy, models=model_keys),
            stages=[StageRecord(name=s) for s in PIPELINE_STAGES],
            audio=AudioMetadata(
                filename=filename if self.settings.store_original_filename else f"speech-sample{ext}",
                format=ext.lstrip("."),
                size_bytes=size,
                original_sample_rate=audio.sample_rate,
                original_channels=audio.channels,
                duration_seconds=round(audio.duration_seconds, 3),
                decoder=audio.decoder,
            ),
        )
        self.repo.save(record)
        self._cancel[analysis_id] = threading.Event()
        # the worker owns its own copy; the returned record must not change underneath the caller
        self._executor.submit(self._run_safe, record.model_copy(deep=True), audio)
        log.info("analysis %s queued (%s, %s)", analysis_id, strategy, ",".join(model_keys))
        return record

    def cancel(self, analysis_id: str) -> AnalysisRecord:
        record = self.repo.require(analysis_id)
        if record.status in TERMINAL_STATUSES:
            return record
        ev = self._cancel.get(analysis_id)
        if ev:
            ev.set()
        if record.status == AnalysisStatus.QUEUED:
            record.status = AnalysisStatus.CANCELLED
            record.updated_at = now()
            self.repo.save(record)
        return record

    # ------------------------------------------------------------------ job execution
    def _run_safe(self, record: AnalysisRecord, audio: DecodedAudio) -> None:
        try:
            self._run(record, audio)
        except Cancelled:
            record.status = AnalysisStatus.CANCELLED
            record.current_stage = None
            for s in record.stages:
                if s.status == StageStatus.RUNNING:
                    s.status, s.message, s.finished_at = StageStatus.SKIPPED, "Cancelled", now()
            self._save(record)
            log.info("analysis %s cancelled", record.analysis_id)
        except Exception as exc:
            log.exception("analysis %s failed", record.analysis_id)
            record.status = AnalysisStatus.FAILED
            record.error = record.error or {"code": ErrorCode.INFERENCE_FAILED.value,
                                            "message": f"Unexpected pipeline error ({type(exc).__name__})."}
            self._save(record)
        finally:
            self._cancel.pop(record.analysis_id, None)
            del audio

    def _save(self, record: AnalysisRecord) -> None:
        record.updated_at = now()
        self.repo.save(record)

    def _stage(self, record: AnalysisRecord, name: AnalysisStatus) -> StageRecord:
        ev = self._cancel.get(record.analysis_id)
        if ev is not None and ev.is_set():
            raise Cancelled()
        stage = next(s for s in record.stages if s.name == name)
        stage.status, stage.started_at = StageStatus.RUNNING, now()
        record.status = record.current_stage = name
        self._save(record)
        return stage

    def _finish(self, record: AnalysisRecord, stage: StageRecord, status: StageStatus, message: str | None = None):
        stage.status, stage.message, stage.finished_at = status, message, now()
        self._save(record)

    def _model_inputs(self, bundle, pre, record: AnalysisRecord, vectors: dict[str, np.ndarray]) -> dict:
        """Compute exactly the feature groups the promoted model was trained on. A group that cannot be computed
        is passed as None, which makes the prediction INSUFFICIENT_DATA (never estimated)."""
        from app.features.acoustic import acoustic_features
        from app.training.features import LANGUAGE_FEATURES

        inputs: dict[str, np.ndarray | None] = {}
        fc = bundle.meta.get("feature_config", {})
        # a model fitted on human transcripts may only be fed ASR-derived features if its bundle says so explicitly
        asr_ok = fc.get("language_source", "whisper") == "whisper" or bool(fc.get("apply_to_asr_transcripts"))
        for g in bundle.groups:
            try:
                if g == "acoustic":
                    inputs[g] = np.array([np.nan if f.value is None else f.value
                                          for f in acoustic_features(pre.waveform, pre.sample_rate)])
                elif g == "language":
                    # models trained on human CHAT transcripts cannot be fed ASR-derived features
                    if not asr_ok or record.transcript.state != SectionState.COMPUTED:
                        inputs[g] = None
                        continue
                    by = {f.name: f for f in record.features.features}
                    inputs[g] = np.array([by[n].value if n in by and by[n].status == "computed" else np.nan
                                          for n in LANGUAGE_FEATURES])
                else:
                    key = g.split(":", 1)[1]
                    vec = vectors.get(key)
                    if vec is None:  # the model needs this encoder even if it was not selected for the analysis
                        vec = self.registry.extract_embedding(key, pre.waveform, pre.sample_rate)["vector"]
                    inputs[g] = np.asarray(vec, dtype=np.float64)
            except Exception as exc:
                log.warning("analysis %s: model input %s failed (%s)", record.analysis_id, g, type(exc).__name__)
                inputs[g] = None
        return inputs

    def _run(self, record: AnalysisRecord, audio: DecodedAudio) -> None:
        s = self.settings

        # 1. PREPROCESSING (fatal on failure)
        st = self._stage(record, AnalysisStatus.PREPROCESSING)
        try:
            pre = preprocess(audio, s)
        except Exception as exc:
            record.error = {"code": ErrorCode.INVALID_AUDIO.value, "message": f"Preprocessing failed ({type(exc).__name__})."}
            self._finish(record, st, StageStatus.FAILED, record.error["message"])
            raise
        record.preprocessing = pre.result
        no_speech = s.vad_enabled and not pre.segments
        if no_speech:
            record.attention.append("No speech was detected by voice activity detection.")
        self._finish(record, st, StageStatus.DEGRADED if no_speech else StageStatus.DONE, pre.result.message)

        # 2. MODEL_PROCESSING (speech foundation model embeddings)
        st = self._stage(record, AnalysisStatus.MODEL_PROCESSING)
        vectors: dict[str, np.ndarray] = {}
        results: list[EmbeddingResult] = []
        for key in record.config.models:
            spec = SPEC_BY_KEY[key]
            t0 = time.perf_counter()
            try:
                out = self.registry.extract_embedding(key, pre.waveform, pre.sample_rate)
                vectors[key] = out["vector"]
                results.append(EmbeddingResult(model_key=key, model_id=out["model_id"], state=SectionState.COMPUTED,
                                               dimension=out["dimension"], frames=out["frames"], pooling=out["pooling"],
                                               layer=out["layer"], device=out["device"],
                                               l2_norm=round(float(np.linalg.norm(out["vector"])), 4),
                                               seconds=round(time.perf_counter() - t0, 2)))
            except ModelUnavailable as exc:
                results.append(EmbeddingResult(model_key=key, model_id=self.registry.model_id(spec),
                                               state=SectionState.MODEL_UNAVAILABLE, message=exc.reason))
            except Exception as exc:
                log.warning("analysis %s: embedding %s failed (%s)", record.analysis_id, key, type(exc).__name__)
                results.append(EmbeddingResult(model_key=key, model_id=self.registry.model_id(spec),
                                               state=SectionState.INFERENCE_FAILED,
                                               message=f"Embedding extraction failed ({type(exc).__name__})."))
        ok = len(vectors)
        record.embeddings = EmbeddingsResult(
            state=SectionState.COMPUTED if ok == len(results) else SectionState.PARTIAL if ok else SectionState.MODEL_UNAVAILABLE,
            strategy=record.config.strategy, requested_models=record.config.models, results=results,
            message=None if ok else "No speech foundation model produced embeddings.")
        for r in results:
            if r.state != SectionState.COMPUTED:
                record.attention.append(f"Embedding model '{r.model_key}': {r.state.value}.")
        self._finish(record, st, StageStatus.DONE if ok == len(results) else StageStatus.DEGRADED,
                     f"{ok}/{len(results)} model(s) produced embeddings")

        # 3. ASR
        st = self._stage(record, AnalysisStatus.ASR)
        if not s.asr_enabled:
            record.transcript = TranscriptResult(state=SectionState.SKIPPED, message="ASR disabled (FSA_ASR_ENABLED=false).")
            self._finish(record, st, StageStatus.SKIPPED, record.transcript.message)
        elif no_speech:
            record.transcript = TranscriptResult(state=SectionState.INSUFFICIENT_DATA,
                                                 message="ASR skipped: no speech detected (avoids transcribing silence).")
            self._finish(record, st, StageStatus.SKIPPED, record.transcript.message)
        else:
            try:
                record.transcript = asr_pipeline.transcribe(self.registry, pre.waveform, pre.sample_rate, s)
                status = StageStatus.DONE if record.transcript.state == SectionState.COMPUTED else StageStatus.DEGRADED
                if record.transcript.state == SectionState.INSUFFICIENT_DATA:
                    # VAD can accept non-speech (e.g. amplified background noise); surface ASR's verdict at job level
                    record.attention.append("No speech detected: ASR found no transcribable speech.")
                self._finish(record, st, status, record.transcript.message)
            except ModelUnavailable as exc:
                record.transcript = TranscriptResult(state=SectionState.MODEL_UNAVAILABLE, model=f"whisper-{s.asr_model}",
                                                     message=exc.reason)
                record.attention.append("ASR model unavailable.")
                self._finish(record, st, StageStatus.DEGRADED, exc.reason)
            except Exception as exc:
                log.warning("analysis %s: ASR failed (%s)", record.analysis_id, type(exc).__name__)
                record.transcript = TranscriptResult(state=SectionState.INFERENCE_FAILED, model=f"whisper-{s.asr_model}",
                                                     message=f"{ErrorCode.ASR_ERROR.value}: {type(exc).__name__}")
                record.attention.append("ASR failed.")
                self._finish(record, st, StageStatus.FAILED, record.transcript.message)

        # 4. FEATURE_EXTRACTION
        st = self._stage(record, AnalysisStatus.FEATURE_EXTRACTION)
        try:
            record.features = extract_features(record.transcript, record.preprocessing, s)
            self._finish(record, st, StageStatus.DONE if record.features.state == SectionState.COMPUTED else StageStatus.DEGRADED,
                         record.features.message)
        except Exception as exc:
            log.warning("analysis %s: features failed (%s)", record.analysis_id, type(exc).__name__)
            record.features = FeaturesResult(state=SectionState.INFERENCE_FAILED,
                                             message=f"{ErrorCode.FEATURE_EXTRACTION_ERROR.value}: {type(exc).__name__}")
            record.attention.append("Feature extraction failed.")
            self._finish(record, st, StageStatus.FAILED, record.features.message)

        # 5. FUSION
        st = self._stage(record, AnalysisStatus.FUSION)
        blocks = {f"embedding:{k}": v for k, v in vectors.items()}
        computed_names = [f.name for f in record.features.features if f.status == "computed" and f.value is not None]
        names = [n for n in FEATURE_ORDER if n in computed_names]
        clinical = feature_vector(record.features, names) if names else None
        if clinical is not None:
            blocks["clinical_features"] = clinical
        fused = fuse(blocks, s.fusion_strategy, s.fusion_block_normalization, s.fusion_weights)
        record.fusion = fused.result if isinstance(fused, FusedRepresentation) else fused
        self._finish(record, st, StageStatus.DONE if record.fusion.state == SectionState.COMPUTED else StageStatus.DEGRADED,
                     record.fusion.message)

        # 6. PREDICTION
        st = self._stage(record, AnalysisStatus.PREDICTION)
        bundle = self.severity.bundle()
        raw_inputs = None
        if bundle is None:
            record.severity = self.severity.not_ready()
        else:
            record.severity, raw_inputs = self.severity.predict(self._model_inputs(bundle, pre, record, vectors))
            if not record.severity.prediction_available:  # a model exists but gave no valid score: make it visible
                record.attention.append(f"Severity model produced no valid score ({record.severity.state.value}).")
        self._finish(record, st, StageStatus.DONE if record.severity.prediction_available else StageStatus.DEGRADED,
                     record.severity.message)

        # 7. EXPLAINABILITY
        st = self._stage(record, AnalysisStatus.EXPLAINABILITY)
        try:
            record.explainability = explain(record.severity, bundle, raw_inputs)
        except Exception as exc:
            record.explainability = ExplainabilityResult(state=SectionState.INFERENCE_FAILED,
                                                         message=f"Explainability failed ({type(exc).__name__}).")
        self._finish(record, st, StageStatus.DONE if record.explainability.state == SectionState.COMPUTED
                     else StageStatus.DEGRADED, record.explainability.message)

        record.current_stage = None
        record.completed_at = now()
        record.status = AnalysisStatus.ATTENTION_NEEDED if record.attention else AnalysisStatus.COMPLETED
        self._save(record)
        log.info("analysis %s finished: %s", record.analysis_id, record.status.value)
