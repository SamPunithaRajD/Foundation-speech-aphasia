"""Response contracts for analysis resources.

Every result section carries a ``state`` so the frontend can distinguish a real
computed value from an unavailable / not-ready component. No section ever
contains placeholder numbers.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .common import AnalysisStatus, Maturity, SectionState, StageRecord


class AudioMetadata(BaseModel):
    filename: str
    format: str
    size_bytes: int
    original_sample_rate: int
    original_channels: int
    duration_seconds: float
    decoder: str


class SpeechSegment(BaseModel):
    start: float
    end: float


class PreprocessingResult(BaseModel):
    state: SectionState = SectionState.PENDING
    steps: list[str] = Field(default_factory=list)
    steps_not_applied: dict[str, str] = Field(default_factory=dict)
    original_sample_rate: int | None = None
    final_sample_rate: int | None = None
    original_channels: int | None = None
    duration_seconds: float | None = None
    speech_duration_seconds: float | None = None
    speech_ratio: float | None = None
    segment_count: int | None = None
    segments: list[SpeechSegment] = Field(default_factory=list)
    peak_gain_db: float | None = None
    vad: dict[str, Any] | None = None
    message: str | None = None


class TranscriptWord(BaseModel):
    word: str
    start: float
    end: float
    probability: float | None = None


class TranscriptSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[TranscriptWord] = Field(default_factory=list)
    mean_word_probability: float | None = None
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class TranscriptResult(BaseModel):
    state: SectionState = SectionState.PENDING
    model: str | None = None
    engine: str | None = None
    language: str | None = None
    language_probability: float | None = None
    text: str | None = None
    segments: list[TranscriptSegment] = Field(default_factory=list)
    word_count: int | None = None
    confidence: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None  # ASR-reported quality indicators and flags (not accuracy)
    source: str = "automatic speech recognition (unverified)"
    message: str | None = None


class ClinicalFeature(BaseModel):
    name: str
    label: str
    value: float | None
    unit: str
    definition: str
    source: Literal["transcript", "asr_timing", "vad", "audio"]
    provenance: str
    status: Literal["computed", "not_computed", "insufficient_data"]
    reason: str | None = None
    note: str | None = None
    clinically_validated: bool = False


class FeaturesResult(BaseModel):
    state: SectionState = SectionState.PENDING
    features: list[ClinicalFeature] = Field(default_factory=list)
    disclaimer: str = (
        "Features are computed automatically from ASR output and voice-activity timing. "
        "They are research measurements and have not been clinically validated in this prototype."
    )
    message: str | None = None


class EmbeddingResult(BaseModel):
    model_key: str
    model_id: str
    state: SectionState
    dimension: int | None = None
    frames: int | None = None
    pooling: str | None = None
    layer: str | None = None
    l2_norm: float | None = None
    device: str | None = None
    seconds: float | None = None
    message: str | None = None


class EmbeddingsResult(BaseModel):
    state: SectionState = SectionState.PENDING
    strategy: Literal["single", "multi"] = "single"
    requested_models: list[str] = Field(default_factory=list)
    results: list[EmbeddingResult] = Field(default_factory=list)
    message: str | None = None


class FusionBlock(BaseModel):
    name: str
    dimension: int
    normalization: str
    weight: float = 1.0


class FusionResult(BaseModel):
    state: SectionState = SectionState.PENDING
    strategy: str | None = None
    blocks: list[FusionBlock] = Field(default_factory=list)
    unified_dimension: int | None = None
    maturity: Maturity = Maturity.PROTOTYPE
    message: str | None = None


class SeverityResult(BaseModel):
    """WAB-R-derived severity score (100 − AQ). All prediction fields are None unless a promoted,
    real-data model produced them; there are no class probabilities in this formulation."""

    state: SectionState = SectionState.PENDING
    prediction_available: bool = False
    severity_score: float | None = None  # 0–100, higher = more impaired (clipped only at the output boundary)
    predicted_aq: float | None = None  # 100 − severity_score
    severity_band: str | None = None  # published WAB AQ band applied to the predicted AQ
    raw_output: float | None = None  # unclipped model output on the severity scale
    clipped: bool | None = None
    score_direction: Literal["higher_worse"] = "higher_worse"
    target_definition: str | None = None
    target: dict[str, Any] | None = None
    model_version: str | None = None
    uncertainty: dict[str, Any] | None = None
    clinical_validity: str | None = None  # always NOT_VALID_FOR_CLINICAL_INFERENCE for uploaded recordings
    clinical_validity_reason: str | None = None
    input_checks: dict[str, Any] | None = None  # OOD scores/limits and inputs imputed at inference
    model_stage: str | None = None  # "PILOT" for a small-sample cross-validated model
    evaluation_summary: dict[str, Any] | None = None  # how (and how well) the model was evaluated
    limitations: list[str] = Field(default_factory=list)
    classifier: dict[str, Any] | None = None  # model metadata (name kept for API compatibility)
    message: str | None = None


class Attribution(BaseModel):
    feature: str
    contribution: float
    direction: Literal["positive", "negative"]


class ExplanationMethodResult(BaseModel):
    method: Literal["shap", "lime"]
    state: SectionState
    target: str | None = None  # what is explained, e.g. "WAB-R-derived severity score"
    base_value: float | None = None
    reliable: bool | None = None  # False -> shown with a warning; see quality
    quality: dict[str, Any] | None = None  # e.g. SHAP additivity error, LIME local R²
    attributions: list[Attribution] = Field(default_factory=list)
    message: str | None = None


class ExplainabilityResult(BaseModel):
    state: SectionState = SectionState.PENDING
    model_version: str | None = None  # the exact model whose prediction is explained
    explained_raw_output: float | None = None  # the raw model output the attributions decompose
    explained_score: float | None = None
    methods: list[ExplanationMethodResult] = Field(default_factory=list)
    interpretation_note: str = (
        "SHAP/LIME provide model-attribution explanations describing which inputs influenced "
        "the model prediction. They do not identify the medical cause of aphasia."
    )
    message: str | None = None


class AnalysisConfig(BaseModel):
    strategy: Literal["single", "multi"] = "single"
    models: list[str] = Field(default_factory=list)


class AnalysisRecord(BaseModel):
    analysis_id: str
    mode: Literal["live"] = "live"
    simulated: bool = False
    status: AnalysisStatus
    current_stage: AnalysisStatus | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None
    config: AnalysisConfig
    stages: list[StageRecord] = Field(default_factory=list)
    audio: AudioMetadata | None = None
    preprocessing: PreprocessingResult = Field(default_factory=PreprocessingResult)
    embeddings: EmbeddingsResult = Field(default_factory=EmbeddingsResult)
    transcript: TranscriptResult = Field(default_factory=TranscriptResult)
    features: FeaturesResult = Field(default_factory=FeaturesResult)
    fusion: FusionResult = Field(default_factory=FusionResult)
    severity: SeverityResult = Field(default_factory=SeverityResult)
    explainability: ExplainabilityResult = Field(default_factory=ExplainabilityResult)
    report: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    attention: list[str] = Field(default_factory=list)


class AnalysisCreated(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    status_url: str
    result_url: str


class Progress(BaseModel):
    completed: int
    total: int
    percent: int


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: AnalysisStatus
    current_stage: AnalysisStatus | None
    progress: Progress
    stages: list[StageRecord]
    updated_at: str
    error: dict[str, Any] | None = None
    attention: list[str] = Field(default_factory=list)


class HistoryItem(BaseModel):
    analysis_id: str
    created_at: str
    status: AnalysisStatus
    duration_seconds: float | None
    models: list[str]
    strategy: str
    severity_state: str
    severity_score: float | None = None
    severity_band: str | None = None
    transcript_state: str
    simulated: bool = False


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int
