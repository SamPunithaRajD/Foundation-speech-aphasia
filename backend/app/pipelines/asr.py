"""Automatic speech recognition with openai-whisper.

Confidence reporting: Whisper exposes per-word *token probabilities* (with
``word_timestamps=True``), per-segment ``avg_logprob`` and ``no_speech_prob``.
These are model-internal probabilities, **not calibrated accuracy estimates**,
and are reported under that name. Known limitation: Whisper frequently omits
filled pauses ("um", "uh") and may normalise repetitions, which biases
disfluency features derived from its transcripts downward.
"""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.models.registry import ModelRegistry
from app.schemas.analysis import TranscriptResult, TranscriptSegment, TranscriptWord
from app.schemas.common import SectionState

# Whisper's own decoding heuristics (openai-whisper defaults) for distrusting a segment
LOGPROB_THRESHOLD, NO_SPEECH_THRESHOLD, COMPRESSION_RATIO_THRESHOLD = -1.0, 0.6, 2.4
QUALITY_NOTE = ("Indicators reported by the ASR model itself. They flag segments Whisper would distrust; they do not "
                "measure transcription accuracy, which is unknown without a human reference transcript.")


def quality_summary(segments: list[TranscriptSegment]) -> dict | None:
    if not segments:
        return None
    lp = [s.avg_logprob for s in segments if s.avg_logprob is not None]
    ns = [s.no_speech_prob for s in segments if s.no_speech_prob is not None]
    cr = [s.compression_ratio for s in segments if s.compression_ratio is not None]
    flags = []
    n_low = sum(v < LOGPROB_THRESHOLD for v in lp)
    n_ns = sum(v > NO_SPEECH_THRESHOLD for v in ns)
    n_cr = sum(v > COMPRESSION_RATIO_THRESHOLD for v in cr)
    if n_low:
        flags.append(f"{n_low} segment(s) with low average log-probability (< {LOGPROB_THRESHOLD})")
    if n_ns:
        flags.append(f"{n_ns} segment(s) with high no-speech probability (> {NO_SPEECH_THRESHOLD})")
    if n_cr:
        flags.append(f"{n_cr} segment(s) with high compression ratio (> {COMPRESSION_RATIO_THRESHOLD}; possible repetition/hallucination)")
    return {"segments": len(segments),
            "mean_avg_logprob": round(float(np.mean(lp)), 4) if lp else None,
            "max_no_speech_prob": round(float(np.max(ns)), 4) if ns else None,
            "max_compression_ratio": round(float(np.max(cr)), 4) if cr else None,
            "flags": flags, "accuracy_verified": False, "note": QUALITY_NOTE}


CONFIDENCE_NOTE = (
    "Word values are Whisper token probabilities (uncalibrated model-internal scores), "
    "not verified transcription accuracy."
)


def transcribe(registry: ModelRegistry, waveform: np.ndarray, sr: int, settings: Settings) -> TranscriptResult:
    import torch
    import whisper

    assert sr == whisper.audio.SAMPLE_RATE, "Whisper requires 16 kHz input"
    model = registry.load_whisper(settings.asr_model)
    audio = waveform.astype(np.float32)

    language = settings.asr_language
    language_probability = None
    if language is None and model.is_multilingual:
        mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(audio)), n_mels=model.dims.n_mels)
        _, probs = model.detect_language(mel.to(model.device))
        language = max(probs, key=probs.get)
        language_probability = round(float(probs[language]), 4)
    elif language is None:
        language = "en"

    out = whisper.transcribe(
        model,
        audio,
        language=language,
        word_timestamps=settings.asr_word_timestamps,
        fp16=model.device.type == "cuda",
        verbose=None,
    )

    segments: list[TranscriptSegment] = []
    all_probs: list[float] = []
    for i, seg in enumerate(out.get("segments", [])):
        words = [
            TranscriptWord(
                word=w["word"].strip(),
                start=round(float(w["start"]), 3),
                end=round(float(w["end"]), 3),
                probability=round(float(w["probability"]), 4) if "probability" in w else None,
            )
            for w in seg.get("words", []) or []
            if w.get("word", "").strip()
        ]
        probs = [w.probability for w in words if w.probability is not None]
        all_probs.extend(probs)
        segments.append(
            TranscriptSegment(
                id=i,
                start=round(float(seg["start"]), 3),
                end=round(float(seg["end"]), 3),
                text=seg["text"].strip(),
                words=words,
                mean_word_probability=round(float(np.mean(probs)), 4) if probs else None,
                avg_logprob=round(float(seg["avg_logprob"]), 4) if "avg_logprob" in seg else None,
                no_speech_prob=round(float(seg["no_speech_prob"]), 4) if "no_speech_prob" in seg else None,
                compression_ratio=round(float(seg["compression_ratio"]), 4) if "compression_ratio" in seg else None,
            )
        )

    text = " ".join(s.text for s in segments).strip()
    word_count = sum(len(s.words) for s in segments) if settings.asr_word_timestamps else len(text.split())
    return TranscriptResult(
        state=SectionState.COMPUTED if segments else SectionState.INSUFFICIENT_DATA,
        model=f"whisper-{settings.asr_model}",
        engine="openai-whisper",
        language=language,
        language_probability=language_probability,
        text=text,
        segments=segments,
        word_count=word_count,
        confidence=(
            {
                "type": "whisper_word_token_probability",
                "calibrated": False,
                "mean_word_probability": round(float(np.mean(all_probs)), 4),
                "note": CONFIDENCE_NOTE,
            }
            if all_probs
            else None
        ),
        quality=quality_summary(segments),
        message=None if segments else "ASR produced no transcribable speech.",
    )
