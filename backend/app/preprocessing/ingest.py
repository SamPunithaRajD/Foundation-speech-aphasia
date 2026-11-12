"""Audio ingestion: upload streaming, validation and decoding.

Decoding order: ``soundfile`` (libsndfile: WAV/FLAC/OGG/MP3) then ``PyAV``
(bundled FFmpeg: M4A/AAC, WEBM/Opus and anything libsndfile rejects).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np

from app.config import Settings
from app.utils.errors import ApiError, ErrorCode

log = logging.getLogger(__name__)
_CHUNK = 1024 * 1024


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, shape (channels, n_samples)
    sample_rate: int
    decoder: str

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[1]) / float(self.sample_rate)


def validate_extension(filename: str, settings: Settings) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ApiError(
            ErrorCode.UNSUPPORTED_FORMAT,
            f"Unsupported audio format '{ext or 'none'}'. Supported: {', '.join(settings.allowed_extensions)}.",
            415,
            {"allowed": settings.allowed_extensions},
        )
    return ext


def stream_to_file(src: BinaryIO, dest: Path, settings: Settings) -> int:
    """Copy an upload stream to ``dest`` while enforcing the size limit."""
    written = 0
    with open(dest, "wb") as out:
        while chunk := src.read(_CHUNK):
            written += len(chunk)
            if written > settings.max_upload_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise ApiError(
                    ErrorCode.FILE_TOO_LARGE,
                    f"Audio file exceeds the {settings.max_upload_mb:g} MB upload limit.",
                    413,
                    {"max_upload_mb": settings.max_upload_mb},
                )
            out.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise ApiError(ErrorCode.INVALID_AUDIO, "The uploaded file is empty.", 400)
    return written


def _decode_soundfile(path: Path) -> DecodedAudio:
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, channels)
    return DecodedAudio(np.ascontiguousarray(data.T), int(sr), "soundfile")


def _decode_pyav(path: Path) -> DecodedAudio:
    import av

    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("no audio stream")
        stream = container.streams.audio[0]
        channels = stream.codec_context.channels or 1
        rate = stream.codec_context.sample_rate or stream.rate
        resampler = av.AudioResampler(format="fltp", layout=stream.layout.name if stream.layout else None, rate=rate)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    if not chunks:
        raise ValueError("audio stream contained no decodable frames")
    data = np.concatenate(chunks, axis=1).astype(np.float32)
    if data.shape[0] != channels and data.shape[0] >= 1:
        channels = data.shape[0]
    return DecodedAudio(np.ascontiguousarray(data), int(rate), "pyav")


def decode_audio(path: Path) -> DecodedAudio:
    errors: list[str] = []
    for decoder in (_decode_soundfile, _decode_pyav):
        try:
            audio = decoder(path)
        except Exception as exc:  # decoders raise many library-specific types
            errors.append(f"{decoder.__name__}: {type(exc).__name__}")
            continue
        if audio.samples.size == 0 or audio.sample_rate <= 0:
            errors.append(f"{decoder.__name__}: empty stream")
            continue
        if not np.all(np.isfinite(audio.samples)):
            audio.samples = np.nan_to_num(audio.samples, nan=0.0, posinf=0.0, neginf=0.0)
        return audio
    log.info("audio decode failed (%s)", "; ".join(errors))
    raise ApiError(
        ErrorCode.AUDIO_DECODE_ERROR,
        "The file could not be decoded as audio. It may be corrupt or use an unsupported codec.",
        422,
        {"decoders_tried": errors},
    )


def validate_decoded(audio: DecodedAudio, settings: Settings) -> None:
    duration = audio.duration_seconds
    if duration < settings.min_duration_seconds:
        raise ApiError(
            ErrorCode.AUDIO_TOO_SHORT,
            f"Audio is {duration:.2f}s long; at least {settings.min_duration_seconds:g}s is required.",
            422,
        )
    if duration > settings.max_duration_seconds:
        raise ApiError(
            ErrorCode.AUDIO_TOO_LONG,
            f"Audio is {duration:.0f}s long; the maximum is {settings.max_duration_seconds:g}s.",
            422,
        )
    if float(np.max(np.abs(audio.samples))) == 0.0:
        raise ApiError(ErrorCode.INVALID_AUDIO, "The audio stream contains only digital silence.", 422)
