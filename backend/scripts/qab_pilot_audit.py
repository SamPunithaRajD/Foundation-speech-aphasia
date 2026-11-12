"""AUDIT of the QAB text pilot: selection-corrected, participant-level evaluation.

    python scripts/qab_pilot_audit.py --data D:/FoundationSpeech-data/QAB [--permutations 2000]

Why this exists
---------------
The deployed 2-feature set ("asr_compatible") was defined AFTER the first two feature sets had been evaluated and
AFTER univariate feature/label correlations had been printed for all participants. Its originally reported
permutation p-value (0.010) therefore did not account for that selection. This script corrects for it:

  * one fixed cohort for every analysis (participants with at least one transcript of >= MIN_WORDS words);
  * a FAMILY of candidates = every feature set that was examined, the rule-derived Whisper-compatible pool, and
    automatic "top-k most correlated" selectors that reproduce choosing features by looking at correlations
    (the selection is redone inside every training fold);
  * family-wise adjusted p-values: in each permutation the BEST candidate of the family is taken (min-MAE),
    so "I picked the best-looking one" is part of the null distribution;
  * a NESTED estimate: candidate and alpha are both chosen by inner participant-grouped CV on the training
    participants only; the held-out participant never influences preprocessing, selection, tuning or fitting.

Label: WAB Aphasia Quotient (Western Aphasia Battery, Kertesz 1982), sheet 'WAB-dataset', column 'AQ' of
QAB-dataset.xlsx; target = 100 - AQ. One AQ per participant (administered at the first session).

Decision rule (fixed before this script was first run): the deployed candidate keeps PILOT status only if its
leave-one-participant-out MAE is below the mean baseline AND its family-wise adjusted p < 0.05.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qab_text_pilot import ALPHAS, ASR_COMPATIBLE, FEATURE_SETS, MIN_WORDS, load  # noqa: E402

from app.config import BACKEND_DIR, Settings  # noqa: E402
from app.training import evaluation as ev  # noqa: E402

ALL11 = FEATURE_SETS["all_text"] + ["clause_marker_rate"]
# Rule-derived pool: every word-level feature that needs neither transcriber mark-up (pauses, fragments, fillers,
# repetitions) nor sentence segmentation. Defined by that rule, not by results.
WHISPER_POOL = ["moving_average_ttr", "clause_marker_rate", "type_token_ratio", "brunet_index", "word_count"]
CANDIDATES = {  # name -> fixed feature list, or ("auto", k) = top-k |Spearman| chosen inside each training set
    "length_robust": FEATURE_SETS["length_robust"],
    "all_text": FEATURE_SETS["all_text"],
    "asr_compatible (deployed)": ASR_COMPATIBLE,
    "mattr_only": ["moving_average_ttr"],
    "whisper_pool_all5": WHISPER_POOL,
    "auto_top1_of_11": ("auto", 1),
    "auto_top2_of_11": ("auto", 2),
}
DEPLOYED = "asr_compatible (deployed)"


def ridge_fit_predict(Xtr, ytr, Xte, alpha):
    """Standardise with TRAIN statistics, ridge with unpenalised intercept (identical to InputPipeline + sklearn Ridge)."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    A, B = (Xtr - mu) / sd, (Xte - mu) / sd
    am, ym = A.mean(0), ytr.mean()
    Ac = A - am
    beta = np.linalg.solve(Ac.T @ Ac + alpha * np.eye(A.shape[1]), Ac.T @ (ytr - ym))
    return (B - am) @ beta + ym


def select_columns(spec, X, y, g, cols):
    if not isinstance(spec, tuple):
        return [cols.index(c) for c in spec]
    pids = np.unique(g)  # automatic selection uses TRAINING participants only (participant means)
    Xm = np.vstack([X[g == p].mean(0) for p in pids])
    ym = np.array([y[g == p][0] for p in pids])
    r = np.abs(np.corrcoef(np.column_stack([rankdata(Xm, axis=0), rankdata(ym)]), rowvar=False)[-1, :-1])
    return list(np.argsort(-np.nan_to_num(r))[: spec[1]])


def inner_scores(spec, X, y, g, cols):
    """Mean absolute error for each alpha, by participant-grouped inner CV (same folds as the original pilot)."""
    groups = np.unique(g)
    folds = np.array_split(np.random.default_rng(0).permutation(groups), min(5, len(groups)))
    out = np.zeros(len(ALPHAS))
    for held in folds:
        tr, va = ~np.isin(g, held), np.isin(g, held)
        idx = select_columns(spec, X[tr], y[tr], g[tr], cols)  # selection redone inside the inner training set
        for k, a in enumerate(ALPHAS):
            out[k] += np.abs(ridge_fit_predict(X[tr][:, idx], y[tr], X[va][:, idx], a) - y[va]).mean()
    return out / len(folds)


def run_all(X, g, y_by_pid, cols):
    """One full pass: LOPO for every candidate + the nested (candidate x alpha) selection. Returns row-level predictions."""
    pids = sorted(y_by_pid)
    y = np.array([y_by_pid[p] for p in g])
    preds = {name: np.zeros(len(y)) for name in list(CANDIDATES) + ["NESTED"]}
    base = np.zeros(len(y))
    chosen = []
    for held in pids:
        tr, te = g != held, g == held
        assert held not in set(g[tr]) and set(g[te]) == {held}  # all sessions of one participant are held out together
        Xtr, ytr, gtr = X[tr], y[tr], g[tr]
        base[te] = np.mean([y_by_pid[p] for p in pids if p != held])
        best = (np.inf, None, None)
        for name, spec in CANDIDATES.items():
            sc = inner_scores(spec, Xtr, ytr, gtr, cols)
            k = int(np.argmin(sc))
            idx = select_columns(spec, Xtr, ytr, gtr, cols)
            preds[name][te] = ridge_fit_predict(Xtr[:, idx], ytr, X[te][:, idx], ALPHAS[k])
            if sc[k] < best[0]:
                best = (sc[k], name, preds[name][te].copy())
        preds["NESTED"][te] = best[2]
        chosen.append(best[1])
    return y, preds, base, chosen


def participant_mae(y, pred, g, pids):
    p = np.clip(pred, 0, 100)
    return float(np.mean([abs(p[g == q].mean() - y[g == q][0]) for q in pids]))


def summarise(y, pred, g, pids):
    yt = np.array([y[g == q][0] for q in pids])
    pp = np.array([pred[g == q].mean() for q in pids])
    m = ev.regression_metrics(yt, pp)
    return {"participant_mae": m["mae"], "participant_rmse": m["rmse"], "r2": m["r2"], "pearson": m["pearson"]["value"],
            "spearman": m["spearman"]["value"], "recording_mae": float(np.abs(np.clip(pred, 0, 100) - y).mean()),
            "n_invalid_outputs": int(np.sum((pp < -10) | (pp > 110))), "max_abs_error": float(np.abs(np.clip(pp, 0, 100) - yt).max()),
            "abs_errors": [round(float(e), 1) for e in np.abs(np.clip(pp, 0, 100) - yt)],
            "mae_95ci_bootstrap": ev.bootstrap_mae_ci(list(pids), yt, pp)}


def export_from(report_path, use, pids, y_by_pid, y, preds, g, base_mae, n_all, n_transcripts) -> int:
    from export_pilot_bundle import export
    from qab_text_pilot import _fit_predict

    rep = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if not rep["decision"]["passed"]:
        print("No bundle written: the audit decision did not pass.")
        return 2
    if rep["decision"]["deployed"] != DEPLOYED:
        print(f"No bundle written: the audit report decided on '{rep['decision']['deployed']}', not '{DEPLOYED}'.")
        return 2
    r, nested = rep["results"][DEPLOYED], rep["results"]["NESTED"]
    now = summarise(y, preds[DEPLOYED], g, pids)
    assert abs(now["participant_mae"] - r["participant_mae"]) < 1e-9, "evaluation does not reproduce the audit report"
    rec_err = np.abs(np.clip(preds[DEPLOYED], 0, 100) - y)
    summary = {
        "method": "leave-one-participant-out cross-validation on human transcripts",
        "participants": len(pids), "participants_in_dataset": n_all, "transcripts": int(len(use)),
        "transcripts_in_dataset": n_transcripts, "excluded_transcripts": n_transcripts - int(len(use)), "min_words": MIN_WORDS,
        "mae": round(r["participant_mae"], 1), "mae_95ci": [round(r["mae_95ci_bootstrap"]["low"], 1), round(r["mae_95ci_bootstrap"]["high"], 1)],
        "mean_baseline_mae": round(rep["mean_baseline_participant_mae"], 1), "recording_level_mae": round(r["recording_mae"], 1),
        "rmse": round(r["participant_rmse"], 1), "r2": round(r["r2"], 2), "pearson": round(r["pearson"], 2),
        "spearman": round(r["spearman"], 2), "max_abs_error": round(r["max_abs_error"], 1),
        "p_unadjusted": round(r["p_unadjusted"], 4), "p_familywise": round(r["p_familywise"], 4),
        "candidates_in_family": len(rep["candidates"]), "permutations": rep["permutations"],
        "selection_inclusive_mae": round(nested["participant_mae"], 1), "selection_inclusive_r2": round(nested["r2"], 2),
        "selection_inclusive_p": round(nested["p_unadjusted"], 4),
        "evaluated_on": "human transcripts", "validated_on_whisper_transcripts": False,
        "dataset": "Quick Aphasia Battery Corpus (Wilson et al., PLOS ONE 2018), AphasiaBank English/NonProtocol/QAB",
        "audit_report": str(report_path)}
    uncertainty = {"half_width": float(rec_err.max()), "coverage": None, "n_participants": len(pids), "n_recordings": int(len(rec_err)),
                   "method": "largest leave-one-participant-out error over single recordings (maximum of "
                             f"{len(rec_err)} errors from {len(pids)} participants) on HUMAN transcripts; not a probability, and its "
                             "coverage is unknown for Whisper transcripts"}
    out = Path(report_path).parent / "bundle"
    export(out, use=use, feats=list(CANDIDATES[DEPLOYED]), y_by_pid=y_by_pid, pilot_id=Path(report_path).parent.name, min_words=MIN_WORDS,
           fit_predict=_fit_predict, summary=summary, uncertainty=uncertainty)
    print("pilot bundle:", out)
    return 0


def main() -> int:
    global DEPLOYED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--permutations", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=BACKEND_DIR / "artifacts" / "pilots")
    ap.add_argument("--export-from", type=Path, default=None, metavar="AUDIT_REPORT_JSON",
                    help="export a PILOT bundle using a finished audit report (only if its decision passed and the "
                         "evaluation reproduces); skips the permutations")
    ap.add_argument("--deploy", default=DEPLOYED, choices=[k for k, v in CANDIDATES.items() if not isinstance(v, tuple)],
                    help="candidate to deploy. All candidates are evaluated in the same family either way, so the "
                         "family-wise p-value already accounts for choosing any one of them")
    args = ap.parse_args()
    DEPLOYED = args.deploy
    df, skipped = load(args.data, Settings())
    all_pids = sorted(df.pid.unique())
    use = df[df.words >= MIN_WORDS].reset_index(drop=True)
    pids = sorted(use.pid.unique())
    excluded = sorted(set(all_pids) - set(pids))
    y_by_pid = {p: 100 - a for p, a in use.groupby("pid")["aq"].first().items()}
    assert use.groupby("pid")["aq"].nunique().max() == 1  # exactly one AQ per participant
    X, g = use[ALL11].to_numpy(float), use["pid"].to_numpy()
    assert not np.isnan(X).any(), "eligible cohort must have no missing feature values (no imputation involved)"
    print(f"cohort: {len(pids)} participants, {len(use)} transcripts (>= {MIN_WORDS} words) | excluded: "
          f"{len(df) - len(use)} transcripts, {len(excluded)} participant(s) with no eligible transcript")

    y, preds, base, chosen = run_all(X, g, y_by_pid, ALL11)
    base_mae = participant_mae(y, base, g, pids)
    if args.export_from:
        return export_from(args.export_from, use, pids, y_by_pid, y, preds, g, base_mae, len(all_pids), len(df))
    results = {name: summarise(y, p, g, pids) for name, p in preds.items()}
    obs = {name: results[name]["participant_mae"] for name in preds}
    family = list(CANDIDATES)

    rng = np.random.default_rng(2026)
    null = {name: [] for name in preds}
    null_min, null_base = [], []
    for i in range(args.permutations):  # shuffle which participant has which AQ; redo EVERYTHING incl. selection
        shuffled = dict(zip(pids, rng.permutation([y_by_pid[p] for p in pids])))
        ys, ps, bs, _ = run_all(X, g, shuffled, ALL11)
        maes = {name: participant_mae(ys, ps[name], g, pids) for name in preds}
        for name in preds:
            null[name].append(maes[name])
        null_min.append(min(maes[n] for n in family))
        null_base.append(participant_mae(ys, bs, g, pids))
    n = args.permutations
    for name in preds:
        results[name]["p_unadjusted"] = (1 + sum(v <= obs[name] for v in null[name])) / (1 + n)
        results[name]["null_mae_mean"] = float(np.mean(null[name]))
        if name in family:  # single-step min-MAE (Westfall-Young) adjustment over the whole family
            results[name]["p_familywise"] = (1 + sum(v <= obs[name] for v in null_min)) / (1 + n)
    dep = results[DEPLOYED]
    passed = dep["participant_mae"] < base_mae and dep["p_familywise"] < 0.05
    report = {
        "audit": "QAB text pilot - selection-corrected evaluation", "created": datetime.now(timezone.utc).isoformat(),
        "label": {"instrument": "Western Aphasia Battery (Kertesz, 1982) Aphasia Quotient", "file": "QAB-dataset.xlsx",
                  "sheet": "WAB-dataset", "column": "AQ", "transformation": "target = 100 - AQ",
                  "matching": "one AQ per participant (administered at session 1), applied to that participant's sessions"},
        "cohort": {"participants": len(pids), "transcripts": int(len(use)), "min_words": MIN_WORDS,
                   "participants_in_dataset": len(all_pids), "transcripts_in_dataset": int(len(df)),
                   "excluded_transcripts": int(len(df) - len(use)), "excluded_participants": len(excluded),
                   "transcripts_per_participant": use.groupby("pid").size().value_counts().sort_index().to_dict()},
        "mean_baseline_participant_mae": base_mae, "null_baseline_mae_mean": float(np.mean(null_base)),
        "permutations": n, "candidates": {k: (v if not isinstance(v, tuple) else f"top-{v[1]} |Spearman| of 11, chosen inside each training fold")
                                          for k, v in CANDIDATES.items()},
        "results": results, "nested_selection_counts": {c: chosen.count(c) for c in sorted(set(chosen))},
        "decision": {"rule": "deployed candidate: participant MAE < mean baseline AND family-wise p < 0.05",
                     "deployed": DEPLOYED, "passed": bool(passed)},
    }
    out = args.out / f"qab_pilot_audit_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"mean baseline: participant MAE {base_mae:.1f} | mean null baseline {np.mean(null_base):.1f}")
    for name, r in results.items():
        print(f"{name:28s} MAE {r['participant_mae']:5.1f} | rec-level MAE {r['recording_mae']:5.1f} | R2 {r['r2']:5.2f} | "
              f"p {r['p_unadjusted']:.4f} | family-wise p {r.get('p_familywise', float('nan')):.4f} | null MAE {r['null_mae_mean']:.1f}")
    print("nested selection chose:", report["nested_selection_counts"])
    print(f"DECISION for '{DEPLOYED}': {'PASSED' if passed else 'FAILED'}")
    print("report:", out / "audit_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
