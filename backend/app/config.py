"""Runtime configuration for the FoundationSpeech-Aphasia backend.

All settings are read from environment variables prefixed with ``FSA_`` (or a
``.env`` file in the backend directory). See ``.env.example`` for documentation.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent

PROJECT_NAME = "FoundationSpeech-Aphasia"
FORMAL_PROJECT_TITLE = (
    "FoundationSpeech-Aphasia: A Self-Supervised Multi-Modal Framework for Explainable "
    "Aphasia Severity Prediction Using Speech Foundation Models and Clinical Language Analysis"
)
API_VERSION = "0.1.0"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FSA_", env_file=BACKEND_DIR / ".env", extra="ignore")

    # --- API ---
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    log_level: str = "INFO"
    serve_frontend: bool = True  # serve the built React app (../dist) at / when it exists
    frontend_dist: Path = BACKEND_DIR.parent / "dist"

    # --- Storage ---
    data_dir: Path = BACKEND_DIR / "data"
    database_url: str = ""  # sqlite:///path/to.db ; defaults to <data_dir>/foundationspeech.db
    temp_dir: Path = BACKEND_DIR / "data" / "tmp"
    retention_days: int = 0  # 0 = keep analysis records indefinitely (default). Set FSA_RETENTION_DAYS>0 to purge older records at startup.
    retain_audio: bool = False  # uploaded speech is deleted after inference unless True
    store_original_filename: bool = False  # filenames may contain identifiers; off by default

    # --- Upload validation ---
    max_upload_mb: float = 100.0
    max_duration_seconds: float = 600.0
    min_duration_seconds: float = 0.5
    allowed_extensions: list[str] = Field(default_factory=lambda: [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"])

    # --- Preprocessing ---
    target_sample_rate: int = 16000
    normalize_peak_dbfs: float = -1.0
    highpass_hz: float = 0.0  # 0 disables the high-pass filter
    vad_enabled: bool = True
    vad_aggressiveness: int = 2  # webrtcvad 0..3
    vad_frame_ms: int = 30
    vad_min_segment_ms: int = 150
    vad_merge_gap_ms: int = 300

    # --- Models ---
    device: str = "auto"  # auto | cpu | cuda
    allow_model_download: bool = False  # never fetch weights unless explicitly enabled
    hf_cache_dir: Path | None = None
    whisper_cache_dir: Path | None = None
    wav2vec2_model_id: str = "facebook/wav2vec2-base"
    hubert_model_id: str = "facebook/hubert-base-ls960"
    wavlm_model_id: str = "microsoft/wavlm-base-plus"
    whisper_encoder_model: str = "tiny"  # openai-whisper checkpoint name
    default_embedding_models: list[str] = Field(default_factory=lambda: ["whisper_encoder"])
    # Longer audio is embedded in chunks of this length (mean-pooled across chunks). Self-attention memory is
    # quadratic in chunk length: measured on CPU, WavLM-base+ peaked at 9.0 GB RAM with 120 s chunks vs 2.5 GB at 30 s.
    max_embedding_seconds: float = 30.0

    # --- ASR ---
    asr_enabled: bool = True
    asr_model: str = "tiny"  # openai-whisper checkpoint name
    asr_language: str | None = None  # None = auto-detect
    asr_word_timestamps: bool = True

    # --- Clinical language features ---
    pause_threshold_ms: int = 250
    mattr_window: int = 25

    # --- Fusion ---
    fusion_strategy: str = "concatenate"  # concatenate | weighted_concatenate
    fusion_block_normalization: str = "l2"  # l2 | none | fitted (fitted requires a trained bundle)
    fusion_weights: dict[str, float] = Field(default_factory=dict)

    # --- Severity classifier ---
    severity_model_dir: Path = BACKEND_DIR / "artifacts" / "severity"
    # No score is reported when the inputs' mean |z| exceeds ood_factor × the largest value seen in training (0 = off)
    ood_factor: float = 1.5
    # No score is reported when a named feature lies outside its training range by more than this many training SDs
    range_margin_sd: float = 0.5
    # Allow a promoted PILOT_CROSS_VALIDATED bundle (small-sample, cross-validated only) to produce predictions.
    # Off by default; every pilot prediction is labelled as such.
    allow_pilot_model: bool = False

    # --- Workers ---
    max_concurrent_jobs: int = 1

    @field_validator("hf_cache_dir", "whisper_cache_dir", "asr_language", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("data_dir", "temp_dir", "severity_model_dir", "hf_cache_dir", "whisper_cache_dir")
    @classmethod
    def _anchor_relative(cls, v):
        """Relative paths are resolved against the backend directory, not the process cwd."""
        return v if v is None or v.is_absolute() else (BACKEND_DIR / v).resolve()

    @property
    def sqlite_path(self) -> Path:
        if self.database_url:
            if not self.database_url.startswith("sqlite:///"):
                raise ValueError("Only sqlite:/// database URLs are supported in this prototype")
            p = Path(self.database_url.removeprefix("sqlite:///"))
            return p if p.is_absolute() else (BACKEND_DIR / p).resolve()
        return self.data_dir / "foundationspeech.db"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)


@lru_cache
def get_settings() -> Settings:
    return Settings()
