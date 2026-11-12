"""Structured API errors. Messages are safe to show to users: they never contain
transcript text, audio content or filesystem paths."""
from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    INVALID_AUDIO = "INVALID_AUDIO"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    AUDIO_DECODE_ERROR = "AUDIO_DECODE_ERROR"
    AUDIO_TOO_SHORT = "AUDIO_TOO_SHORT"
    AUDIO_TOO_LONG = "AUDIO_TOO_LONG"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    INFERENCE_ERROR = "INFERENCE_ERROR"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    ASR_ERROR = "ASR_ERROR"
    FEATURE_EXTRACTION_ERROR = "FEATURE_EXTRACTION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    REPORT_GENERATION_ERROR = "REPORT_GENERATION_ERROR"
    ANALYSIS_NOT_FOUND = "ANALYSIS_NOT_FOUND"
    ANALYSIS_NOT_FINISHED = "ANALYSIS_NOT_FINISHED"
    INVALID_REQUEST = "INVALID_REQUEST"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ApiError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int = 400, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"error": {"code": self.code.value, "message": self.message, "details": self.details}}
