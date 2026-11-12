from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.schemas.analysis import (AnalysisCreated, AnalysisRecord, AnalysisStatusResponse, ExplainabilityResult,
                                  FeaturesResult, Progress, SeverityResult, TranscriptResult)
from app.schemas.common import PIPELINE_STAGES, TERMINAL_STATUSES, ErrorResponse, StageStatus
from app.services.container import Services, get_services

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
ERRORS = {400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
          415: {"model": ErrorResponse}, 422: {"model": ErrorResponse}}


@router.post("", status_code=202, response_model=AnalysisCreated, responses=ERRORS)
def create_analysis(
    file: UploadFile = File(..., description="Speech recording (WAV, MP3, M4A, FLAC, OGG, WEBM)"),
    strategy: str = Form("single", description="'single' or 'multi' model representation"),
    models: str = Form("", description="Comma-separated model keys, e.g. 'whisper_encoder,wav2vec2'"),
    svc: Services = Depends(get_services),
):
    keys = [m.strip() for m in models.split(",") if m.strip()]
    record = svc.analysis.create(file.file, file.filename or "", strategy, keys)
    aid = record.analysis_id
    return JSONResponse(
        status_code=202,
        content=AnalysisCreated(analysis_id=aid, status=record.status, status_url=f"/api/analysis/{aid}/status",
                                result_url=f"/api/analysis/{aid}").model_dump(mode="json"),
    )


@router.get("/{analysis_id}", response_model=AnalysisRecord, responses=ERRORS)
def get_analysis(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id)


@router.get("/{analysis_id}/status", response_model=AnalysisStatusResponse, responses=ERRORS)
def get_status(analysis_id: str, svc: Services = Depends(get_services)):
    r = svc.repo.require(analysis_id)
    done = sum(s.status not in (StageStatus.WAITING, StageStatus.RUNNING) for s in r.stages)
    total = len(PIPELINE_STAGES)
    percent = 100 if r.status in TERMINAL_STATUSES and r.status.value in ("COMPLETED", "ATTENTION_NEEDED") \
        else int(done / total * 100)
    return AnalysisStatusResponse(analysis_id=r.analysis_id, status=r.status, current_stage=r.current_stage,
                                  progress=Progress(completed=done, total=total, percent=percent), stages=r.stages,
                                  updated_at=r.updated_at, error=r.error, attention=r.attention)


@router.post("/{analysis_id}/cancel", response_model=AnalysisStatusResponse, responses=ERRORS)
def cancel_analysis(analysis_id: str, svc: Services = Depends(get_services)):
    svc.analysis.cancel(analysis_id)
    return get_status(analysis_id, svc)


@router.get("/{analysis_id}/transcript", response_model=TranscriptResult, responses=ERRORS)
def get_transcript(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id).transcript


@router.get("/{analysis_id}/features", response_model=FeaturesResult, responses=ERRORS)
def get_features(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id).features


@router.get("/{analysis_id}/severity", response_model=SeverityResult, responses=ERRORS)
def get_severity(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id).severity


@router.get("/{analysis_id}/explainability", response_model=ExplainabilityResult, responses=ERRORS)
def get_explainability(analysis_id: str, svc: Services = Depends(get_services)):
    return svc.repo.require(analysis_id).explainability
