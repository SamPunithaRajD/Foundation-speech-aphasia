"""Audio preprocessing: mono -> resample -> (optional high-pass) -> peak normalisation -> VAD.

Only the steps listed in ``PreprocessedAudio.steps`` were applied. Noise
reduction is intentionally *not* implemented and is reported as such.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import gcd

import numpy as np
from scipy import signal

from app.config import Settings
from app.preprocessing.ingest import DecodedAudio
from app.schemas.analysis import PreprocessingResult, SpeechSegment
from app.schemas.common import SectionState


@dataclass
class PreprocessedAudio:
    waveform: np.ndarray  # float32 mono at sample_rate
    sample_rate: int
    segments: list[tuple[float, float]] = field(default_factory=list)
    result: PreprocessingResult = field(default_factory=PreprocessingResult)


def to_mono(samples: np.ndarray) -> np.ndarray:
    return samples.mean(axis=0) if samples.shape[0] > 1 else samples[0]


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    g = gcd(sr_in, sr_out)
    return signal.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def highpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    sos = signal.butter(4, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def peak_normalize(x: np.ndarray, target_dbfs: float) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        return x, 0.0
    gain = (10 ** (target_dbfs / 20.0)) / peak
    return (x * gain).astype(np.float32), float(20 * np.log10(gain))


def webrtc_vad_segments(x: np.ndarray, sr: int, settings: Settings) -> list[tuple[float, float]]:
    """Speech regions from WebRTC VAD, merged across short gaps and filtered by minimum length."""
    import webrtcvad

    if sr not in (8000, 16000, 32000, 48000):
        raise ValueError(f"webrtcvad does not support {sr} Hz")
    vad = webrtcvad.Vad(settings.vad_aggressiveness)
    frame_len = int(sr * settings.vad_frame_ms / 1000)
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)
    n_frames = len(pcm) // frame_len
    flags = [vad.is_speech(pcm[i * frame_len:(i + 1) * frame_len].tobytes(), sr) for i in range(n_frames)]

    raw: list[list[float]] = []
    frame_s = settings.vad_frame_ms / 1000
    for i, is_speech in enumerate(flags):
        if not is_speech:
            continue
        start, end = i * frame_s, (i + 1) * frame_s
        if raw and start - raw[-1][1] <= settings.vad_merge_gap_ms / 1000:
            raw[-1][1] = end
        else:
            raw.append([start, end])
    min_len = settings.vad_min_segment_ms / 1000
    return [(round(s, 3), round(e, 3)) for s, e in raw if e - s >= min_len]


def preprocess(audio: DecodedAudio, settings: Settings) -> PreprocessedAudio:
    steps: list[str] = ["decode"]
    not_applied: dict[str, str] = {"noise_reduction": "Not implemented in this prototype."}

    x = to_mono(audio.samples).astype(np.float32)
    steps.append("mono_downmix" if audio.channels > 1 else "mono_passthrough")

    sr = settings.target_sample_rate
    x = resample(x, audio.sample_rate, sr)
    steps.append(f"resample_{audio.sample_rate}_to_{sr}_polyphase" if audio.sample_rate != sr else "resample_not_needed")

    if settings.highpass_hz > 0:
        x = highpass(x, sr, settings.highpass_hz)
        steps.append(f"highpass_butterworth4_{settings.highpass_hz:g}hz")
    else:
        not_applied["highpass_filter"] = "Disabled (FSA_HIGHPASS_HZ=0)."

    x -= float(np.mean(x))  # DC offset removal
    steps.append("dc_offset_removal")
    x, gain_db = peak_normalize(x, settings.normalize_peak_dbfs)
    steps.append(f"peak_normalize_{settings.normalize_peak_dbfs:g}dbfs")

    duration = len(x) / sr
    segments: list[tuple[float, float]] = []
    vad_info: dict | None = None
    if settings.vad_enabled:
        segments = webrtc_vad_segments(x, sr, settings)
        steps.append("voice_activity_detection_webrtc")
        vad_info = {
            "engine": "webrtcvad",
            "aggressiveness": settings.vad_aggressiveness,
            "frame_ms": settings.vad_frame_ms,
            "merge_gap_ms": settings.vad_merge_gap_ms,
            "min_segment_ms": settings.vad_min_segment_ms,
        }
    else:
        not_applied["voice_activity_detection"] = "Disabled (FSA_VAD_ENABLED=false)."

    speech = sum(e - s for s, e in segments) if settings.vad_enabled else None
    result = PreprocessingResult(
        state=SectionState.COMPUTED,
        steps=steps,
        steps_not_applied=not_applied,
        original_sample_rate=audio.sample_rate,
        final_sample_rate=sr,
        original_channels=audio.channels,
        duration_seconds=round(duration, 3),
        speech_duration_seconds=round(speech, 3) if speech is not None else None,
        speech_ratio=round(speech / duration, 4) if speech is not None and duration > 0 else None,
        segment_count=len(segments) if settings.vad_enabled else None,
        segments=[SpeechSegment(start=s, end=e) for s, e in segments],
        peak_gain_db=round(gain_db, 2),
        vad=vad_info,
        message=("No speech detected by voice activity detection." if settings.vad_enabled and not segments else None),
    )
    return PreprocessedAudio(waveform=x, sample_rate=sr, segments=segments, result=result)
