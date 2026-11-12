"""AUDIT of a QAB pilot bundle against the source data.

    python scripts/qab_bundle_audit.py --data D:/FoundationSpeech-data/QAB --bundle artifacts/severity

Checks (each prints PASS/FAIL):
  1. target metadata names the instrument that was actually administered and carries its provenance;
  2. the bundle was fitted on exactly the eligible participants/transcripts;
  3. refitting from the source data reproduces the stored coefficients;
  4. the application's inference path (features -> SeverityService) reproduces the training-path prediction for
     every training transcript (same preprocessing, same feature order);
  5. the stored uncertainty half-width equals what the saved procedure produces, and what that number really is;
  6. extrapolation is rejected: a feature beyond the training range gives no score;
  7. SHAP attributions add up to the actual prediction.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qab_text_pilot import ALPHAS, MIN_WORDS, _fit_predict, load, lopo  # noqa: E402

from app.config import Settings  # noqa: E402
from app.explainability.explain import shap_explanation  # noqa: E402
from app.features.clinical import extract_features  # noqa: E402
from app.models.registry import ModelRegistry  # noqa: E402
from app.pipelines.severity import SeverityService  # noqa: E402
from app.schemas.analysis import PreprocessingResult  # noqa: E402
from app.training.bundle import load_bundle  # noqa: E402
from app.training.features import LANGUAGE_FEATURES, _human_transcript_result  # noqa: E402
from app.training.models import InputPipeline, RidgeModel  # noqa: E402
from app.training.qab import parse_qab_transcript  # noqa: E402

results: dict[str, bool] = {}


def check(name: str, ok: bool, detail: str = "") -> None:
    results[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    args = ap.parse_args()
    if not (args.bundle / "promotion.json").exists():  # audit BEFORE promotion: use a temporary promoted copy
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="fsa-bundle-audit-")) / "bundle"
        shutil.copytree(args.bundle, tmp)
        (tmp / "promotion.json").write_text(json.dumps({"reviewer": "temporary copy for audit", "pilot": True}))
        print(f"(bundle is not promoted; auditing a temporary promoted copy: {tmp})")
        args.bundle = tmp
    settings = Settings(_env_file=None, allow_pilot_model=True, severity_model_dir=args.bundle)
    b = load_bundle(args.bundle)
    meta, feats = b.meta, b.input_names("language")
    df, _ = load(args.data, settings)
    use = df[df.words >= MIN_WORDS].reset_index(drop=True)
    pids = sorted(use.pid.unique())
    y_by = {p: 100 - a for p, a in use.groupby("pid")["aq"].first().items()}

    # 1. target metadata
    t = meta["target"]
    check("target names the administered instrument (WAB 1982, not WAB-R)",
          "1982" in t.get("source", "") and "WAB-R" not in t.get("name", ""), f"name='{t.get('name')}', source='{t.get('source')}'")
    prov = t.get("provenance") or {}
    check("target provenance recorded (file, sheet, column)",
          (prov.get("file"), prov.get("sheet"), prov.get("column")) == ("QAB-dataset.xlsx", "WAB-dataset", "AQ"), str(prov)[:120])
    check("bundle is real-data pilot status, not synthetic", meta["data_origin"] == "REAL" and meta["status"] == "PILOT_CROSS_VALIDATED",
          f"{meta['data_origin']} / {meta['status']}")

    # 2. fitted on eligible rows only
    c = meta.get("complexity", {})
    check("fitted on exactly the eligible cohort", (c.get("training_participants"), c.get("training_recordings")) == (len(pids), len(use)),
          f"bundle {c.get('training_participants')} participants / {c.get('training_recordings')} transcripts; "
          f"source {len(pids)} / {len(use)} (dataset total {df.pid.nunique()} / {len(df)})")

    # 3. refit reproduces coefficients
    X, g = use[feats].to_numpy(float), use["pid"].to_numpy()
    y = np.array([y_by[p] for p in g])
    _, alpha = _fit_predict(X, y, g, X[:1], feats)
    pipe = InputPipeline(["language"]).fit({"language": X}, {"language": feats})
    m = RidgeModel(["language"], alpha=alpha)
    m.fit(pipe.transform({"language": X}, {"language": feats}), y)
    check("refit from source data reproduces stored coefficients",
          alpha == meta["hyperparameters"]["alpha"] and np.allclose(m.coef, b.model.coef, atol=1e-10) and abs(m.intercept - b.model.intercept) < 1e-10,
          f"alpha {alpha}; max coef diff {np.abs(m.coef - b.model.coef).max():.1e}")
    train_pred = b.model.predict(b.pipeline.transform({"language": X}, {"language": feats}))

    # 4. inference path == training path, per transcript
    svc = SeverityService(args.bundle, settings, ModelRegistry(settings))
    svc.ood_factor = 0  # compare raw outputs only; distribution checks are tested separately below
    assert svc.bundle() is not None, svc.status()["reason"]
    diffs = []
    for path in sorted((args.data / "transcriptions").glob("*.txt")):
        tr = parse_qab_transcript(path)
        fr = extract_features(_human_transcript_result(tr.text), PreprocessingResult(), settings)
        by = {f.name: f for f in fr.features}
        vec = np.array([by[n].value if n in by and by[n].status == "computed" else np.nan for n in LANGUAGE_FEATURES])
        res, _ = svc.predict({"language": vec})
        row = use[(use.pid == tr.participant_id) & (use.session == tr.session)]
        if len(row) == 0:  # not eligible: the service must refuse it for the same reason
            assert res.state.value == "INSUFFICIENT_DATA", res.state
            continue
        diffs.append(abs(res.raw_output - float(train_pred[row.index[0]])))
    check("application inference path reproduces training-path predictions", len(diffs) == len(use) and max(diffs) < 2e-3,
          f"{len(diffs)} transcripts, max difference {max(diffs):.1e} (raw output is rounded to 3 decimals)")
    check("ineligible (<25-word) transcripts are refused at inference", True, f"{len(df) - len(use)} transcripts -> INSUFFICIENT_DATA")

    # 5. uncertainty
    import qab_pilot_audit as A

    ya, preds, _, _ = A.run_all(use[A.ALL11].to_numpy(float), g, y_by, A.ALL11)
    cand = [k for k, v in A.CANDIDATES.items() if not isinstance(v, tuple) and list(v) == list(feats)]
    assert len(cand) == 1, f"bundle features {feats} match no single audited candidate"
    rec = np.abs(np.clip(preds[cand[0]], 0, 100) - ya)  # held-out error of every single recording (bundle's own candidate)
    yt, pred, _, _ = lopo(use, feats, y_by)
    part = np.abs(np.clip(pred, 0, 100) - yt)
    unc = meta.get("uncertainty") or {}
    hw = unc.get("half_width") or -1
    check("stored interval half-width equals the saved procedure's value", min(abs(hw - rec.max()), abs(hw - part.max())) < 1e-6,
          f"stored {hw:.2f}; largest held-out error per recording {rec.max():.2f} ({len(rec)} recordings), "
          f"per participant {part.max():.2f} ({len(part)} participants)")
    check("interval description states what it is", "largest" in json.dumps(unc).lower() or "maximum" in json.dumps(unc).lower(),
          str(unc.get("method"))[:110])

    # 6. extrapolation is rejected
    svc2 = SeverityService(args.bundle, settings, ModelRegistry(settings))
    lang = {n: float(use[n].median()) if n in use else np.nan for n in LANGUAGE_FEATURES}
    lang["word_count"] = 200.0
    lang["moving_average_ttr"] = float(use["moving_average_ttr"].max()) + 2 * float(use["moving_average_ttr"].std())
    res, _ = svc2.predict({"language": np.array([lang[n] for n in LANGUAGE_FEATURES])})
    check("input beyond the training range gives no score", res.state.value == "OUT_OF_DISTRIBUTION" and res.severity_score is None,
          f"MATTR set 2 SD above the training maximum -> {res.state.value}, score {res.severity_score}, raw kept {res.raw_output}")

    # 7. SHAP explains the actual prediction
    raw = b.pipeline.raw({"language": X[:1]}, {"language": feats})
    sh = shap_explanation(b, raw)
    check("SHAP attributions add up to the actual prediction", sh.quality["additivity_error"] < 0.01
          and abs(sh.quality["explained_output"] - float(train_pred[0])) < 1e-2, f"additivity error {sh.quality['additivity_error']}")

    failed = [k for k, v in results.items() if not v]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
