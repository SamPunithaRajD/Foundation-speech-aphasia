"""Clinical language / fluency feature extraction.

Every feature is computed from real inputs (ASR transcript, ASR word timing,
WebRTC VAD segments, audio duration). Nothing is estimated or imputed: if an
input is missing the feature is returned with ``status != "computed"`` and a
reason. Computing a feature does not make it clinically validated.
"""
from __future__ import annotations

import re

import numpy as np

from app.config import Settings
from app.schemas.analysis import ClinicalFeature, FeaturesResult, PreprocessingResult, TranscriptResult
from app.schemas.common import SectionState

FILLED_PAUSES = {"um", "umm", "uh", "uhh", "uhm", "er", "erm", "ah", "hmm", "mm", "mhm"}
CLAUSE_MARKERS = {"because", "although", "though", "since", "while", "whereas", "if", "unless", "until",
                  "which", "who", "whom", "whose", "where", "when", "after", "before", "so", "that"}
SHORT_SAMPLE_WORDS = 50
_TOKEN = re.compile(r"[a-z0-9']+")

FEATURE_ORDER = [
    "speech_rate", "articulation_rate", "type_token_ratio", "moving_average_ttr", "brunet_index",
    "mean_sentence_length", "immediate_repetition_rate", "filled_pause_rate", "disfluency_rate",
    "pause_frequency", "mean_pause_duration", "pause_time_ratio", "speech_ratio", "word_count",
    "clause_marker_density", "clause_marker_rate", "syntactic_complexity",
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def _f(name, label, unit, value, definition, source, provenance, note=None) -> ClinicalFeature:
    return ClinicalFeature(name=name, label=label, value=None if value is None else round(float(value), 4), unit=unit,
                           definition=definition, source=source, provenance=provenance, status="computed", note=note)


def _missing(name, label, unit, definition, source, provenance, reason, status="not_computed") -> ClinicalFeature:
    return ClinicalFeature(name=name, label=label, value=None, unit=unit, definition=definition, source=source,
                           provenance=provenance, status=status, reason=reason)


def mattr(tokens: list[str], window: int) -> float:
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return float(np.mean(ratios))


def pause_features(pre: PreprocessingResult, settings: Settings) -> list[ClinicalFeature]:
    threshold = max(settings.pause_threshold_ms, settings.vad_merge_gap_ms) / 1000
    prov = f"webrtcvad inter-segment gaps >= {threshold * 1000:.0f} ms; leading/trailing silence excluded"
    defs = {
        "pause_frequency": ("Pause frequency", "pauses/min", "Number of silent pauses per minute of recording."),
        "mean_pause_duration": ("Pause duration", "s", "Mean duration of silent pauses between speech regions."),
        "pause_time_ratio": ("Pause time ratio", "ratio", "Total internal pause time divided by recording duration."),
    }
    if pre.segments is None or pre.vad is None or not pre.duration_seconds:
        return [_missing(k, *v[:2], v[2], "vad", prov, "Voice activity detection was not run.") for k, v in defs.items()]
    segs = [(s.start, s.end) for s in pre.segments]
    gaps = [b[0] - a[1] for a, b in zip(segs, segs[1:]) if b[0] - a[1] >= threshold]
    minutes = pre.duration_seconds / 60
    out = [
        _f("pause_frequency", *defs["pause_frequency"][:2], len(gaps) / minutes if minutes else None,
           defs["pause_frequency"][2], "vad", prov),
        (_f("mean_pause_duration", *defs["mean_pause_duration"][:2], float(np.mean(gaps)), defs["mean_pause_duration"][2],
            "vad", prov) if gaps else _missing("mean_pause_duration", *defs["mean_pause_duration"][:2],
                                               defs["mean_pause_duration"][2], "vad", prov,
                                               "No internal pauses detected.", "insufficient_data")),
        _f("pause_time_ratio", *defs["pause_time_ratio"][:2], sum(gaps) / pre.duration_seconds,
           defs["pause_time_ratio"][2], "vad", prov),
    ]
    if pre.speech_ratio is not None:
        out.append(_f("speech_ratio", "Speech ratio", "ratio", pre.speech_ratio,
                      "VAD-detected speech time divided by recording duration.", "vad", "webrtcvad"))
    return out


def transcript_features(tr: TranscriptResult, pre: PreprocessingResult, settings: Settings) -> list[ClinicalFeature]:
    words = [w.word for s in tr.segments for w in s.words] or (tr.text or "").split()
    tokens = [t for w in words for t in tokenize(w)]
    n = len(tokens)
    short = f"Short sample ({n} words): lexical ratios are unstable below ~{SHORT_SAMPLE_WORDS} words." if n < SHORT_SAMPLE_WORDS else None
    out: list[ClinicalFeature] = [
        _f("word_count", "Word count", "words", n, "Number of word tokens in the ASR transcript.", "transcript",
           "regex tokenisation of Whisper output"),
    ]
    if n == 0:
        return out
    duration = pre.duration_seconds or 0
    rate_def = ("Words per minute over the full recording (includes pauses). Used here as a fluency proxy; "
                "no composite fluency score is computed.")
    out.append(_f("speech_rate", "Speech rate", "words/min", n / duration * 60, rate_def, "transcript",
                  "word_count / audio duration") if duration > 0 else
               _missing("speech_rate", "Speech rate", "words/min", rate_def, "transcript", "word_count / audio duration",
                        "No audio duration available (transcript without a recording)."))
    if pre.speech_duration_seconds:
        out.append(_f("articulation_rate", "Articulation rate", "words/s", n / pre.speech_duration_seconds,
                      "Words per second of VAD-detected speech time (pauses excluded). Word-based, not syllable-based.",
                      "asr_timing", "word_count / VAD speech duration"))
    else:
        out.append(_missing("articulation_rate", "Articulation rate", "words/s", "Words per second of speech time.",
                            "asr_timing", "word_count / VAD speech duration", "No VAD speech duration available."))

    types = len(set(tokens))
    out.append(_f("type_token_ratio", "Lexical diversity", "ratio", types / n,
                  "Type-token ratio: unique words / total words. Strongly length-dependent.", "transcript",
                  "TTR on lower-cased tokens", note=short))
    w = settings.mattr_window
    out.append(
        _f("moving_average_ttr", "Lexical richness", "ratio", mattr(tokens, w),
           f"Moving-average type-token ratio (MATTR) with a {w}-word window; less length-sensitive than TTR.",
           "transcript", f"MATTR window={w}", note=short)
        if n >= w else
        _missing("moving_average_ttr", "Lexical richness", "ratio", "Moving-average type-token ratio (MATTR).",
                 "transcript", f"MATTR window={w}", f"Requires at least {w} words; transcript has {n}.",
                 "insufficient_data")
    )
    out.append(_f("brunet_index", "Brunét's index", "index", n ** (types ** -0.165),
                  "Brunét's W = N^(V^-0.165). Lower values indicate richer vocabulary.", "transcript",
                  "Brunét (1978)", note=short))

    sentences = [s for s in re.split(r"[.!?]+", tr.text or "") if tokenize(s)]
    out.append(_f("mean_sentence_length", "Sentence length", "words/sentence",
                  float(np.mean([len(tokenize(s)) for s in sentences])) if sentences else n,
                  "Mean words per sentence. Sentence boundaries come from Whisper's predicted punctuation.",
                  "transcript", "split on . ! ?"))

    reps = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b and a not in FILLED_PAUSES)
    fillers = sum(1 for t in tokens if t in FILLED_PAUSES)
    whisper_note = "Whisper often omits fillers and repetitions, so this likely under-counts."
    out.append(_f("immediate_repetition_rate", "Word repetition", "per 100 words", reps / n * 100,
                  "Immediate whole-word repetitions (e.g. 'the the') per 100 words.", "transcript",
                  "adjacent identical tokens", note=whisper_note))
    out.append(_f("filled_pause_rate", "Filled pauses", "per 100 words", fillers / n * 100,
                  "Filled pauses (um, uh, er, ...) per 100 words.", "transcript",
                  f"lexicon of {len(FILLED_PAUSES)} filler tokens", note=whisper_note))
    out.append(_f("disfluency_rate", "Speech disfluency", "per 100 words", (reps + fillers) / n * 100,
                  "Filled pauses plus immediate repetitions per 100 words.", "transcript",
                  "filled_pause_rate + immediate_repetition_rate", note=whisper_note))
    markers = sum(1 for t in tokens if t in CLAUSE_MARKERS)
    out.append(_f("clause_marker_density", "Clause-marker density", "per sentence",
                  markers / max(1, len(sentences)),
                  "Subordinating/relative clause markers per sentence. A lexical proxy only, not a parse.",
                  "transcript", "lexicon lookup", note="Proxy; ambiguous words such as 'that' and 'so' are counted."))
    out.append(_f("clause_marker_rate", "Clause-marker rate", "per 100 words", markers / n * 100,
                  "Subordinating/relative clause markers per 100 words. Word-based, so it does not depend on how the "
                  "transcript is split into sentences. A lexical proxy only, not a parse.",
                  "transcript", "lexicon lookup", note="Proxy; ambiguous words such as 'that' and 'so' are counted."))
    return out


def extract_features(tr: TranscriptResult | None, pre: PreprocessingResult, settings: Settings) -> FeaturesResult:
    features = pause_features(pre, settings)
    tr_ok = tr is not None and tr.state == SectionState.COMPUTED
    if tr_ok:
        features += transcript_features(tr, pre, settings)
    else:
        reason = "No transcript available: " + ((tr.message or tr.state.value) if tr else "ASR did not run.")
        for name, label, unit in [("speech_rate", "Speech rate", "words/min"), ("type_token_ratio", "Lexical diversity", "ratio"),
                                  ("moving_average_ttr", "Lexical richness", "ratio"),
                                  ("mean_sentence_length", "Sentence length", "words/sentence"),
                                  ("immediate_repetition_rate", "Word repetition", "per 100 words"),
                                  ("disfluency_rate", "Speech disfluency", "per 100 words")]:
            features.append(_missing(name, label, unit, "Transcript-derived feature.", "transcript", "-", reason))
    features.append(_missing("syntactic_complexity", "Syntactic complexity", "index",
                             "Parse-based syntactic complexity (e.g. clause depth, Yngve/Frazier scores).",
                             "transcript", "-", "Requires a syntactic parser, which is not implemented in this prototype."))
    order = {name: i for i, name in enumerate(FEATURE_ORDER)}
    features.sort(key=lambda f: order.get(f.name, len(order)))
    computed = sum(f.status == "computed" for f in features)
    state = SectionState.COMPUTED if tr_ok and computed else SectionState.PARTIAL if computed else SectionState.INSUFFICIENT_DATA
    return FeaturesResult(state=state, features=features,
                          message=None if tr_ok else "Only audio/VAD-derived features were computed (no transcript).")


def feature_vector(result: FeaturesResult, names: list[str]) -> np.ndarray | None:
    """Ordered numeric vector for fusion/classification; None if any requested feature is missing."""
    by_name = {f.name: f for f in result.features}
    values = []
    for n in names:
        f = by_name.get(n)
        if f is None or f.status != "computed" or f.value is None:
            return None
        values.append(f.value)
    return np.asarray(values, dtype=np.float32)
