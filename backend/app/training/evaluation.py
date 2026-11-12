"""Evaluation for the WAB-R-derived severity regression (primary) and AQ-band classification
(secondary, derived from the same AQ — never independent labels).

Every metric is computed from real predictions on real held-out participants; functions return
``None`` with a reason when a metric is undefined (e.g. constant targets, n < 3).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_recall_fscore_support, r2_score)

from app.training import target as tgt

# Comparison framework (results intentionally empty until experiments are run on comparable data)
EXPERIMENTS = {
    "A_handcrafted": {"config": "configs/experiments/model_a_handcrafted.yaml", "status": "READY_FOR_DATA"},
    "B_wav2vec2": {"config": "configs/experiments/model_b_wav2vec2.yaml", "status": "READY_FOR_DATA"},
    "C_multimodal_wav2vec2": {"config": "configs/experiments/model_c_multimodal.yaml", "status": "READY_FOR_DATA"},
    "B_wav2vec2_ridge": {"config": "configs/experiments/ablation_b_wav2vec2_ridge.yaml", "status": "READY_FOR_DATA"},
    "C_concat_mlp": {"config": "configs/experiments/ablation_c_concat_mlp.yaml", "status": "READY_FOR_DATA"},
    "D_foundation_model_ablation": {"config": "configs/experiments/ablation_d_*.yaml", "status": "READY_FOR_DATA"},
    "base_paper_comparison": {"config": "configs/experiments/base_paper_comparison.yaml", "status": "NOT_AVAILABLE",
                              "reason": "Requires a reimplementation of the base paper pipeline evaluated on the same "
                                        "participants and target; not implemented."},
}


def _corr(fn, a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return {"value": None, "p": None, "reason": "undefined (n < 3 or constant values)"}
    r = fn(a, b)
    return {"value": float(r[0]), "p": float(r[1])}


def regression_metrics(y_true: np.ndarray, y_pred_raw: np.ndarray) -> dict:
    """Metrics on the 0–100 severity scale. Headline metrics use predictions clipped to [0, 100];
    unclipped errors and the number of invalid (out-of-tolerance) outputs are reported alongside."""
    y_true, raw = np.asarray(y_true, float), np.asarray(y_pred_raw, float)
    pred = np.clip(raw, 0, 100)
    n = len(y_true)
    out = {
        "n": n,
        "mae": float(mean_absolute_error(y_true, pred)) if n else None,
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))) if n else None,
        "r2": float(r2_score(y_true, pred)) if n >= 2 and np.std(y_true) > 0 else None,
        "pearson": _corr(stats.pearsonr, y_true, pred),
        "spearman": _corr(stats.spearmanr, y_true, pred),
        "mae_aq_scale": float(mean_absolute_error(100 - y_true, 100 - pred)) if n else None,
        "fraction_clipped": float(np.mean(pred != raw)) if n else None,
        # outputs beyond the tolerance are INVALID at inference (no score is shown); here they are scored clipped
        # but counted, and the unclipped error is reported so wild outputs cannot hide behind clipping
        "n_invalid_outputs": int(np.sum((raw < -tgt.OUTPUT_TOLERANCE) | (raw > 100 + tgt.OUTPUT_TOLERANCE))) if n else None,
        "mae_unclipped": float(mean_absolute_error(y_true, raw)) if n else None,
        "rmse_unclipped": float(np.sqrt(mean_squared_error(y_true, raw))) if n else None,
        "prediction_distribution": _distribution(pred),
        "target_distribution": _distribution(y_true),
        "raw_output_range": [float(raw.min()), float(raw.max())] if n else None,
    }
    return out


def _distribution(v: np.ndarray) -> dict | None:
    if not len(v):
        return None
    q = np.percentile(v, [5, 25, 50, 75, 95])
    return {"mean": float(v.mean()), "sd": float(v.std()), "min": float(v.min()), "max": float(v.max()),
            "p5": float(q[0]), "p25": float(q[1]), "median": float(q[2]), "p75": float(q[3]), "p95": float(q[4])}


def participant_level(pids: list[str], y_true, y_pred_raw) -> tuple[np.ndarray, np.ndarray, list[str]]:
    agg = defaultdict(lambda: [[], []])
    for p, t, pr in zip(pids, y_true, y_pred_raw):
        agg[p][0].append(t)
        agg[p][1].append(pr)
    keys = sorted(agg)
    return (np.array([np.mean(agg[k][0]) for k in keys]), np.array([np.mean(agg[k][1]) for k in keys]), keys)


def bootstrap_mae_ci(pids: list[str], y_true, y_pred_raw, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI for MAE by resampling participants (not recordings)."""
    yt, yp, keys = participant_level(pids, y_true, np.clip(y_pred_raw, 0, 100))
    if len(keys) < 5:
        return {"low": None, "high": None, "reason": "fewer than 5 participants"}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    maes = np.abs(yt[idx] - yp[idx]).mean(axis=1)
    return {"low": float(np.percentile(maes, 2.5)), "high": float(np.percentile(maes, 97.5)),
            "method": f"participant-level bootstrap, {n_boot} resamples"}


def band_metrics(y_true_severity, y_pred_raw) -> dict:
    """Secondary: WAB AQ severity bands derived from true and predicted AQ (not independent labels)."""
    true_b = [tgt.band_from_aq(100 - float(s)) for s in y_true_severity]
    pred_b = [tgt.band_from_aq(100 - float(np.clip(s, 0, 100))) for s in y_pred_raw]
    labels = tgt.BAND_LABELS
    p, r, f, s = precision_recall_fscore_support(true_b, pred_b, labels=labels, zero_division=0)
    return {
        "note": "Bands derived from WAB-R AQ (Kertesz, 2007) for true and predicted AQ; not independent labels.",
        "accuracy": float(accuracy_score(true_b, pred_b)),
        "macro_f1": float(f1_score(true_b, pred_b, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_b, pred_b, labels=labels, average="weighted", zero_division=0)),
        "per_class": {lab: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, lab in enumerate(labels)},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(true_b, pred_b, labels=labels).tolist()},
    }


def residual_interval(y_true, y_pred_raw, coverage: float = 0.9) -> dict:
    """Empirical interval from validation residuals: a split-conformal-style band applied uniformly to every
    prediction. It is NOT a per-sample uncertainty estimate and not a probability."""
    res = np.abs(np.asarray(y_true, float) - np.clip(np.asarray(y_pred_raw, float), 0, 100))
    if len(res) < 5:
        return {"half_width": None, "coverage": coverage, "reason": "fewer than 5 validation recordings"}
    n = len(res)
    q = min(1.0, np.ceil((n + 1) * coverage) / n)
    return {"half_width": float(np.quantile(res, q)), "coverage": coverage, "n_calibration": n,
            "method": "absolute validation residual quantile (split-conformal style); uniform, not per-sample"}


def evaluate_split(pids, y_true, y_pred_raw) -> dict:
    yt_p, yp_p, keys = participant_level(pids, y_true, y_pred_raw)
    return {
        "recording_level": regression_metrics(y_true, y_pred_raw),
        "participant_level": {**regression_metrics(yt_p, yp_p), "participants": len(keys)},
        "mae_95ci": bootstrap_mae_ci(pids, y_true, y_pred_raw),
        "derived_bands": band_metrics(y_true, y_pred_raw),
    }


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Word-level Levenshtein distance / reference length (lower-cased, punctuation stripped)."""
    import re

    ref, hyp = re.findall(r"[a-z0-9']+", reference.lower()), re.findall(r"[a-z0-9']+", hypothesis.lower())
    if not ref:
        return None
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / len(ref)


def transcript_agreement(human: dict[str, str | None], whisper: dict[str, str | None]) -> dict | None:
    """Whisper vs human CHAT transcript, where both exist. The human transcript is normalised CHAT text
    (fillers kept, retracings kept), so this measures agreement under that convention, not a benchmark WER."""
    wers = [w for rid, ref in human.items() if ref and whisper.get(rid) is not None
            for w in [word_error_rate(ref, whisper[rid])] if w is not None]
    if not wers:
        return None
    return {"n_recordings": len(wers), "wer_mean": float(np.mean(wers)), "wer_median": float(np.median(wers)),
            "wer_p90": float(np.percentile(wers, 90)),
            "note": "Word error rate of Whisper against the normalised human CHAT transcript of the same audio scope."}
