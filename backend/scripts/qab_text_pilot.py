"""PILOT (text only): language features from human QAB transcripts -> WAB-R-derived severity score.

    python scripts/qab_text_pilot.py --data D:/FoundationSpeech-data/QAB

Data: Quick Aphasia Battery dataset (Wilson et al., PLOS ONE 2018; AphasiaBank English/NonProtocol/QAB):
16 participants with chronic aphasia, 3 connected-speech transcripts each (48), one WAB-R AQ per participant.
There is NO audio, so no acoustic features, speech embeddings, speech rate or pause timing can be computed.

Design (fixed before any result was seen):
  * target: severity = 100 - WAB-R AQ (same target as the main project);
  * two pre-specified feature sets (both reported; neither chosen after the fact);
  * leave-one-PARTICIPANT-out cross-validation; a participant's 3 sessions are never split;
  * ridge regression; alpha chosen by participant-grouped inner CV on the training participants only;
  * imputation/standardisation fitted on the training participants only (app.training.models.InputPipeline);
  * reference: predicting the training participants' mean (a model must beat this to be worth anything);
  * permutation test: the participant->AQ assignment is shuffled and the whole procedure repeated.

This is an exploratory pilot on 16 people. It does not produce a deployable model, writes no model bundle,
and cannot change the application's MODEL_NOT_READY state.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import BACKEND_DIR, Settings  # noqa: E402
from app.features.clinical import extract_features  # noqa: E402
from app.schemas.analysis import PreprocessingResult  # noqa: E402
from app.training import evaluation as ev  # noqa: E402
from app.training import target as tgt  # noqa: E402
from app.training.features import _human_transcript_result  # noqa: E402
from app.training.models import InputPipeline, RidgeModel  # noqa: E402
from app.training.qab import NORMALIZATION_RULES, parse_qab_transcript  # noqa: E402

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LENGTH_ROBUST = ["moving_average_ttr", "mean_sentence_length", "immediate_repetition_rate", "filled_pause_rate",
                 "clause_marker_density", "marked_pause_rate", "fragment_rate"]
# Deployment candidate (added 2026-10-12, AFTER the two sets above were evaluated; chosen for compatibility, not for
# performance): only features that mean the same thing in a human transcript and in a Whisper transcript. Whisper does
# not write pauses, fragments, fillers or repetitions, and punctuates sentences differently, so every feature that
# depends on those is excluded. Both remaining features are word-based.
ASR_COMPATIBLE = ["moving_average_ttr", "clause_marker_rate"]
FEATURE_SETS = {"length_robust": LENGTH_ROBUST,
                "all_text": LENGTH_ROBUST + ["word_count", "type_token_ratio", "brunet_index"],
                "asr_compatible": ASR_COMPATIBLE}
ALL_FEATURES = FEATURE_SETS["all_text"] + ["clause_marker_rate"]
MIN_WORDS = 25  # MATTR (window 25) cannot be computed below this; applied to training rows AND at inference
DEPLOY_ALPHA = 0.05  # go/no-go: LOPO MAE below the mean baseline AND permutation p below this


def load(data: Path, settings: Settings):
    wab = pd.read_excel(data / "QAB-dataset.xlsx", sheet_name="WAB-dataset")
    aq = {str(int(p)): tgt.validate_aq(a) for p, a in zip(wab["PID"], wab["AQ"])}
    rows, skipped = [], []
    for path in sorted((data / "transcriptions").glob("*.txt")):
        t = parse_qab_transcript(path)
        if t.participant_id not in aq:
            skipped.append((path.name, "no WAB-R AQ for participant"))
            continue
        feats = {f.name: f.value for f in extract_features(_human_transcript_result(t.text), PreprocessingResult(), settings).features
                 if f.status == "computed"}
        n = feats.get("word_count") or 0
        feats["marked_pause_rate"] = t.marked_pauses / n * 100 if n else np.nan
        feats["fragment_rate"] = t.fragments / n * 100 if n else np.nan
        rows.append({"pid": t.participant_id, "session": t.session, "aq": aq[t.participant_id], "words": n,
                     **{k: feats.get(k, np.nan) for k in ALL_FEATURES}})
    return pd.DataFrame(rows), skipped


def _fit_predict(Xtr, ytr, gtr, Xte, names):
    """Ridge with alpha chosen by participant-grouped inner CV on the training participants only."""
    groups = np.unique(gtr)
    folds = np.array_split(np.random.default_rng(0).permutation(groups), min(5, len(groups)))
    score = {}
    for a in ALPHAS:
        errs = []
        for held in folds:
            tr, va = ~np.isin(gtr, held), np.isin(gtr, held)
            pipe = InputPipeline(["x"]).fit({"x": Xtr[tr]}, {"x": names})
            m = RidgeModel(["x"], alpha=a)
            m.fit(pipe.transform({"x": Xtr[tr]}, {"x": names}), ytr[tr])
            errs.append(np.abs(m.predict(pipe.transform({"x": Xtr[va]}, {"x": names})) - ytr[va]).mean())
        score[a] = float(np.mean(errs))
    alpha = min(score, key=score.get)
    pipe = InputPipeline(["x"]).fit({"x": Xtr}, {"x": names})
    m = RidgeModel(["x"], alpha=alpha)
    m.fit(pipe.transform({"x": Xtr}, {"x": names}), ytr)
    return m.predict(pipe.transform({"x": Xte}, {"x": names})), alpha


def lopo(df: pd.DataFrame, names: list[str], y_by_pid: dict[str, float]):
    X, g = df[names].to_numpy(float), df["pid"].to_numpy()
    y = np.array([y_by_pid[p] for p in g])
    pids = sorted(y_by_pid)
    pred, base, alphas = {}, {}, []
    for held in pids:
        tr, te = g != held, g == held
        p, a = _fit_predict(X[tr], y[tr], g[tr], X[te], names)
        pred[held] = float(np.mean(p))  # participant-level: mean over that participant's sessions
        base[held] = float(np.mean([y_by_pid[q] for q in pids if q != held]))  # training participants' mean
        alphas.append(a)
    yt = np.array([y_by_pid[p] for p in pids])
    return yt, np.array([pred[p] for p in pids]), np.array([base[p] for p in pids]), alphas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--permutations", type=int, default=500)
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "artifacts" / "pilots")
    ap.add_argument("--export-bundle", action="store_true",
                    help="write a PILOT bundle for the asr_compatible set if (and only if) it passes the go/no-go rule")
    args = ap.parse_args()
    df, skipped = load(args.data, Settings())
    y_by_pid = {p: 100 - a for p, a in df.groupby("pid")["aq"].first().items()}
    n_p = len(y_by_pid)
    print(f"participants {n_p} | transcripts {len(df)} | skipped {len(skipped)} | words per transcript: "
          f"median {df.words.median():.0f}, min {df.words.min():.0f}, max {df.words.max():.0f}")
    report = {"pilot": "QAB text-only pilot", "status": "EXPLORATORY_PILOT - NOT A DEPLOYABLE MODEL",
              "created": datetime.now(timezone.utc).isoformat(), "data": str(args.data),
              "source": "Wilson et al., PLOS ONE 2018; AphasiaBank English/NonProtocol/QAB", "target": tgt.target_definition(),
              "participants": n_p, "transcripts": int(len(df)), "skipped": skipped,
              "words_per_transcript": {"median": float(df.words.median()), "min": float(df.words.min()), "max": float(df.words.max())},
              "missing_feature_values": {k: int(df[k].isna().sum()) for k in ALL_FEATURES},
              "aq_range": [float(df.aq.min()), float(df.aq.max())], "normalization_rules": NORMALIZATION_RULES,
              "design": "leave-one-participant-out; ridge; alpha by grouped inner CV on training participants; "
                        "train-only imputation/standardisation", "feature_sets": {}}
    rng = np.random.default_rng(13)
    pids = sorted(y_by_pid)
    deploy = None
    for name, feats in FEATURE_SETS.items():
        use = df[df.words >= MIN_WORDS] if name == "asr_compatible" else df
        y_use = {p: y_by_pid[p] for p in sorted(set(use.pid))}
        pids = sorted(y_use)
        yt, pred, base, alphas = lopo(use, feats, y_use)
        m, b = ev.regression_metrics(yt, pred), ev.regression_metrics(yt, base)
        mae = m["mae"]
        null = []
        for _ in range(args.permutations):  # shuffle which participant has which AQ; redo everything
            shuffled = dict(zip(pids, rng.permutation([y_use[p] for p in pids])))
            yt_s, pred_s, _, _ = lopo(use, feats, shuffled)
            null.append(float(np.abs(np.clip(pred_s, 0, 100) - yt_s).mean()))
        p_value = (1 + sum(v <= mae for v in null)) / (1 + len(null))
        if name == "asr_compatible":
            deploy = dict(use=use, feats=feats, y=y_use, yt=yt, pred=pred, mae=mae, base_mae=b["mae"], p=p_value, metrics=m, baseline=b)
        report["feature_sets"][name] = {
            "participants": len(pids), "transcripts": int(len(use)),
            "excluded_transcripts_below_min_words": int(len(df) - len(use)) if name == "asr_compatible" else 0,
            "features": feats, "model": {k: m[k] for k in ("n", "mae", "rmse", "r2", "pearson", "spearman", "n_invalid_outputs", "mae_unclipped")},
            "mean_baseline": {k: b[k] for k in ("mae", "rmse", "r2")},
            "mae_95ci": ev.bootstrap_mae_ci(pids, yt, pred), "alphas_selected": alphas,
            "permutation_test": {"n": args.permutations, "p_value_mae": p_value, "null_mae_mean": float(np.mean(null))},
            "per_participant_abs_error": [round(float(e), 1) for e in np.abs(np.clip(pred, 0, 100) - yt)]}
        print(f"[{name}] n={len(pids)} LOPO participant-level: MAE {mae:.1f} (baseline {b['mae']:.1f}), RMSE {m['rmse']:.1f} "
              f"(baseline {b['rmse']:.1f}), R2 {m['r2']:.2f}, Spearman {m['spearman']['value']:.2f}, "
              f"permutation p = {p_value:.3f}")
    # exploratory univariate associations (participant means; 10 tests, uncorrected)
    means = df.groupby("pid")[ALL_FEATURES].mean()
    sev = pd.Series(y_by_pid)
    report["exploratory_spearman_with_severity"] = {
        k: round(float(means[k].corr(sev.loc[means.index], method="spearman")), 3) for k in ALL_FEATURES}
    out = args.out / f"qab_text_pilot_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    passed = deploy["mae"] < deploy["base_mae"] and deploy["p"] < DEPLOY_ALPHA
    report["deployment_decision"] = {"rule": f"LOPO MAE < mean-baseline MAE and permutation p < {DEPLOY_ALPHA}", "candidate": "asr_compatible",
                                     "passed": bool(passed), "mae": deploy["mae"], "baseline_mae": deploy["base_mae"], "p": deploy["p"]}
    print(f"deployment rule for asr_compatible: {'PASSED' if passed else 'FAILED'}")
    if args.export_bundle:
        print("This script no longer exports bundles: its p-values do not account for feature-set selection. "
              "Run scripts/qab_pilot_audit.py, then scripts/qab_pilot_audit.py --export-from <audit_report.json>.")
    (out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"report: {out / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
