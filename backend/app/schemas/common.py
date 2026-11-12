"""Shared enums and response models.

Research-discipline vocabulary used throughout the API:

* ``Maturity`` describes *how far* a component has progressed
  (implemented / prototype / simulated / proposed / validated).
* ``SectionState`` describes *what actually happened* for one part of one analysis.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    QUEUED = "QUEUED"
    PREPROCESSING = "PREPROCESSING"
    MODEL_PROCESSING = "MODEL_PROCESSING"
    ASR = "ASR"
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"
    FUSION = "FUSION"
    PREDICTION = "PREDICTION"
    EXPLAINABILITY = "EXPLAINABILITY"
    COMPLETED = "COMPLETED"
    ATTENTION_NEEDED = "ATTENTION_NEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


PIPELINE_STAGES: list[AnalysisStatus] = [
    AnalysisStatus.PREPROCESSING,
    AnalysisStatus.MODEL_PROCESSING,
    AnalysisStatus.ASR,
    AnalysisStatus.FEATURE_EXTRACTION,
    AnalysisStatus.FUSION,
    AnalysisStatus.PREDICTION,
    AnalysisStatus.EXPLAINABILITY,
]
TERMINAL_STATUSES = {
    AnalysisStatus.COMPLETED,
    AnalysisStatus.ATTENTION_NEEDED,
    AnalysisStatus.FAILED,
    AnalysisStatus.CANCELLED,
}


class SectionState(str, Enum):
    PENDING = "PENDING"
    COMPUTED = "COMPUTED"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_OUTPUT = "INVALID_OUTPUT"  # model produced an output that is not a valid estimate (raw value kept)
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"  # inputs far outside the training distribution; no score reported
    NOT_AVAILABLE = "NOT_AVAILABLE"


class Maturity(str, Enum):
    IMPLEMENTED = "implemented"  # runs end to end
    PROTOTYPE = "prototype"  # runs, not scientifically validated
    SIMULATED = "simulated"  # demonstration only
    PROPOSED = "proposed"  # described by the research design, not implemented
    VALIDATED = "validated"  # supported by real experiments


class StageStatus(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    DONE = "done"
    DEGRADED = "degraded"  # stage ran but a component was unavailable / not ready
    SKIPPED = "skipped"
    FAILED = "failed"


class StageRecord(BaseModel):
    name: AnalysisStatus
    status: StageStatus = StageStatus.WAITING
    message: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: ErrorBody
