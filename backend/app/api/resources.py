"""Models, history, reports and health endpoints."""
from __future__ import annotations

import hashlib
import logging

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.config import API_VERSION, FORMAL_PROJECT_TITLE, PROJECT_NAME
from app.models.hardware import hardware_summary
from app.models.registry import whisper_checkpoint_path
from app.pipelines.fusion import STRATEGIES
from app.reports.pdf import build_report
from app.schemas.analysis import AnalysisRecord, HistoryItem, HistoryResponse
from app.schemas.common import TERMINAL_STATUSES, ErrorResponse
from app.services.analysis_service import now
from app.services.container import Services, get_services
from app.utils.errors import ApiError, ErrorCode

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["resources"])
ERRORS = {404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}


@router.get("/health")
def health(svc: Services = Depends(get_services)):
    return {"status": "ok", "project": PROJECT_NAME, "title": FORMAL_PROJECT_TITLE, "version": API_VERSION,
            "mode": "live", "simulated": False}


@router.get("/models")
def list_models(svc: Services = Depends(get_services)):
    s = svc.settings
    asr_path = whisper_checkpoint_path(s.asr_model, s)
    asr_installed = bool(asr_path and asr_path.exists())
    return {
        "hardware": hardware_summary(s.device),
        "allow_model_download": s.allow_model_download,
        "default_embedding_models": s.default_embedding_models,
        "embedding_models": svc.registry.list(),
        "asr": {
            "engine": "openai-whisper", "model": s.asr_model, "enabled": s.asr_enabled,
            "status": "available" if asr_installed or s.allow_model_download else "unavailable",
            "reason": None if asr_installed else ("Weights will be downloaded on first use." if s.allow_model_download
                                                  else f"Whisper '{s.asr_model}' weights are not installed."),
            "maturity": "implemented",
        },
        "fusion": {"configured_strategy": s.fusion_strategy, "block_normalization": s.fusion_block_normalization,
                   "strategies": STRATEGIES},
        "severity_classifier": svc.severity.status(),
        "explainability": {"methods": ["shap", "lime"],
                           "status": "available" if svc.severity.status()["prediction_available"] else "not_available",
                           "reason": None if svc.severity.status()["prediction_available"]
                           else "Explanations require a trained severity model."},
    }


def _history_item(r: AnalysisRecord) -> HistoryItem:
    return HistoryItem(analysis_id=r.analysis_id, created_at=r.created_at, status=r.status,
                       duration_seconds=r.audio.duration_seconds if r.audio else None, models=r.config.models,
                       strategy=r.config.strategy, severity_state=r.severity.state.value,
                       severity_score=r.severity.severity_score, severity_band=r.severity.severity_band,
                       transcript_state=r.transcript.state.value)


@router.get("/history", response_model=HistoryResponse)
def history(limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0), svc: Services = Depends(get_services)):
    records, total = svc.repo.list(limit, offset)
    return HistoryResponse(items=[_history_item(r) for r in records], total=total)


@router.get("/history/{analysis_id}", response_model=AnalysisRecord, responses=ERRORS)
def history_item(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id)


@router.get("/reports/{analysis_id}", responses={**ERRORS, 200: {"content": {"application/pdf": {}}}})
def report(analysis_id: str, svc: Services = Depends(get_services)):
    record = svc.repo.require(analysis_id)
    if record.status not in TERMINAL_STATUSES:
        raise ApiError(ErrorCode.ANALYSIS_NOT_FINISHED, "The analysis is still running; the report is not ready.", 409,
                       {"status": record.status.value})
    try:
        pdf = build_report(record)
    except Exception as exc:
        log.exception("report generation failed for %s", analysis_id)
        raise ApiError(ErrorCode.REPORT_GENERATION_ERROR, "The report could not be generated.", 500) from exc
    record.report = {"generated_at": now(), "format": "application/pdf", "size_bytes": len(pdf),
                     "sha256": hashlib.sha256(pdf).hexdigest()}
    svc.repo.save(record)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="foundationspeech-report-{analysis_id}.pdf"'})
