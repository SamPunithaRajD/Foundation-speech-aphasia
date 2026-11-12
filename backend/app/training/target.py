"""WAB-R-derived research target (single source of truth).

Source instrument: Western Aphasia Battery-Revised (Kertesz, 2007), Aphasia Quotient (AQ).
    AQ ∈ [0, 100]; higher AQ = better language performance.

Research target:
    severity_score = 100 − AQ        (the "WAB-R-derived severity score")
    0   = lowest WAB-R-derived impairment score
    100 = highest WAB-R-derived impairment score

This is a transformation of a published instrument score, NOT a newly validated clinical
severity scale. The original AQ is always retained alongside the derived score.

Severity bands are the published WAB AQ severity ranges (Kertesz, 2007):
    very severe 0–25 · severe 26–50 · moderate 51–75 · mild ≥ 76
Published ranges are integers; for decimal AQ values this module uses half-open intervals
(very severe: AQ < 26; severe: 26 ≤ AQ < 51; moderate: 51 ≤ AQ < 76; mild: AQ ≥ 76), which
reproduces the published assignment for every integer AQ. This convention is a documented
implementation choice. No "no aphasia" band is produced: the WAB-R AQ aphasia cut-off is
reported inconsistently (93.7/93.8) and is contested for mild impairment, so it is not applied.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_NAME = "WAB-R-derived severity score"
TARGET_SOURCE = "Western Aphasia Battery-Revised (WAB-R) Aphasia Quotient (Kertesz, 2007)"
TARGET_VERSION = "wabr-aq-inverted/1.0"
TARGET_TRANSFORMATION = "severity_score = 100 - WAB-R AQ"
SCORE_DIRECTION = "higher_worse"
AQ_MIN, AQ_MAX = 0.0, 100.0

# (label, lower bound inclusive on AQ). Order: most to least impaired.
AQ_BANDS: list[tuple[str, float]] = [("Very severe", 0.0), ("Severe", 26.0), ("Moderate", 51.0), ("Mild", 76.0)]
BAND_SOURCE = "WAB AQ severity ranges (Kertesz, 2007); half-open intervals for decimal AQ"


class InvalidAQ(ValueError):
    pass


def validate_aq(value) -> float:
    """Return AQ as float or raise InvalidAQ. Never coerces missing/out-of-range values."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise InvalidAQ("missing")
    try:
        aq = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidAQ(f"not numeric: {value!r}") from exc
    if not math.isfinite(aq):
        raise InvalidAQ("not finite")
    if not AQ_MIN <= aq <= AQ_MAX:
        raise InvalidAQ(f"outside [0, 100]: {aq}")
    return aq


def severity_from_aq(aq: float) -> float:
    return AQ_MAX - validate_aq(aq)


def aq_from_severity(severity: float) -> float:
    """Inverse transformation for model outputs (not validated: callers decide about clipping)."""
    return AQ_MAX - float(severity)


def band_from_aq(aq: float) -> str:
    aq = validate_aq(aq)
    label = AQ_BANDS[0][0]
    for name, lower in AQ_BANDS:
        if aq >= lower:
            label = name
    return label


BAND_LABELS = [b[0] for b in AQ_BANDS]


# A raw model output may overshoot the target range slightly (regression has no bounded head). Overshoot up to
# OUTPUT_TOLERANCE points is treated as numerical overshoot and clipped *with a flag*. Anything further outside is
# NOT a severity estimate: it is reported as INVALID with the raw value, never converted into a valid-looking score.
OUTPUT_TOLERANCE = 10.0


@dataclass(frozen=True)
class FinalPrediction:
    raw_severity: float  # unclipped model output on the severity scale (always preserved)
    valid: bool  # False -> no score, AQ or band may be reported
    severity_score: float | None  # clipped to [0, 100] only when within tolerance
    predicted_aq: float | None  # 100 - severity_score
    clipped: bool
    band: str | None  # published WAB AQ band applied to the predicted AQ
    reason: str | None = None


def finalize_prediction(raw_severity: float, tolerance: float = OUTPUT_TOLERANCE) -> FinalPrediction:
    raw = float(raw_severity)
    if not math.isfinite(raw):
        return FinalPrediction(raw, False, None, None, False, None, "Model output is not finite.")
    if raw < AQ_MIN - tolerance or raw > AQ_MAX + tolerance:
        return FinalPrediction(raw, False, None, None, False, None,
                               f"Raw model output {raw:.1f} is more than {tolerance:g} points outside the 0–100 target "
                               "range; it is not a valid severity estimate.")
    score = min(max(raw, AQ_MIN), AQ_MAX)
    aq = AQ_MAX - score
    return FinalPrediction(raw_severity=raw, valid=True, severity_score=round(score, 1), predicted_aq=round(aq, 1),
                           clipped=score != raw, band=band_from_aq(aq))


def target_definition() -> dict:
    return {
        "name": TARGET_NAME, "source": TARGET_SOURCE, "version": TARGET_VERSION,
        "transformation": TARGET_TRANSFORMATION, "direction": SCORE_DIRECTION, "range": [AQ_MIN, AQ_MAX],
        "bands": {"source": BAND_SOURCE, "aq_lower_bounds": dict(AQ_BANDS)},
        "note": "Transformation of a published instrument score; not a newly validated clinical severity scale.",
    }


def wab_1982_target_definition() -> dict:
    """Target of the QAB text pilot. The QAB study (Wilson et al., PLOS ONE 2018) administered the ORIGINAL Western
    Aphasia Battery (Kertesz, 1982), not the WAB-R. The Aphasia Quotient has the same composition and 0-100 range in
    both editions (2 x [Spontaneous Speech + Comprehension + Repetition + Naming]), so the same transformation applies,
    but the instrument must be named correctly and must not be reported as WAB-R."""
    return {
        "name": "WAB-AQ-derived severity score",
        "source": "Western Aphasia Battery (Kertesz, 1982) Aphasia Quotient",
        "version": "wab1982-aq-inverted/1.0",
        "transformation": "severity_score = 100 - WAB AQ", "direction": SCORE_DIRECTION, "range": [AQ_MIN, AQ_MAX],
        "bands": {"source": "AQ severity ranges as published for the WAB-R (Kertesz, 2007), applied to WAB (1982) AQ. "
                            "The AQ is composed identically in both editions, but these ranges have not been separately "
                            "verified for the 1982 edition; the band is a convenience label only.",
                  "aq_lower_bounds": dict(AQ_BANDS)},
        "provenance": {"file": "QAB-dataset.xlsx", "sheet": "WAB-dataset", "column": "AQ",
                       "dataset": "Quick Aphasia Battery Corpus, AphasiaBank English/NonProtocol/QAB, doi:10.21415/T8P2-CN50",
                       "citation": "Wilson SM, Eriksson DK, Schneck SM, Lucanie JM. PLOS ONE 2018; 13(2): e0192773",
                       "verified": "AQ equals 2 x (SpontSp + AudVbl Comp + Repetition + Naming) for all 16 participants"},
        "note": "Transformation of a published instrument score; not a newly validated clinical severity scale.",
    }
