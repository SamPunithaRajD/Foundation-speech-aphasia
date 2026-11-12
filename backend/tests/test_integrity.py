"""Research-integrity tests: nothing is fabricated, unavailability is reported, errors are preserved.

The "promoted" bundle used below is a TEST FIXTURE: a model trained on SYNTHETIC_TEST_DATA whose
metadata is rewritten inside a temporary directory solely to exercise the production code path.
It demonstrates the API contract; its numbers mean nothing and it never leaves the test.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.explainability.explain import explain
from app.pipelines.fusion import fuse
from app.pipelines.severity import NOT_READY_MESSAGE, NOT_VALID, SeverityService
from app.schemas.analysis import SeverityResult
from app.schemas.common import AnalysisStatus, SectionState
from app.services.analysis_service import now
from conftest import voiced_bursts, wait_for_terminal, wav_bytes

SEVERITY_VALUE_FIELDS = ("severity_score", "predicted_aq", "severity_band", "raw_output", "uncertainty", "model_version")


def test_no_model_means_not_ready_and_no_values(tmp_path):
    svc = SeverityService(tmp_path / "missing")
    res, raw = svc.predict({"acoustic": np.ones(38)})
    assert res.state == SectionState.MODEL_NOT_READY and res.prediction_available is False and raw is None
    assert all(getattr(res, f) is None for f in SEVERITY_VALUE_FIELDS) and res.message == NOT_READY_MESSAGE
    st = svc.status()
    assert st["prediction_available"] is False and st["model_card"]["status"] == "MODEL_NOT_READY"


def test_legacy_classification_bundle_is_not_loaded(tmp_path):
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "model.joblib").write_bytes(b"not loaded")
    st = SeverityService(tmp_path / "b").status()
    assert st["state"] == "MODEL_NOT_READY" and "legacy" in st["reason"]


def test_no_shap_or_lime_without_prediction():
    res = explain(SeverityResult(state=SectionState.MODEL_NOT_READY, message=NOT_READY_MESSAGE), None, None)
    assert res.state == SectionState.NOT_AVAILABLE and res.methods == []
    assert "medical cause" in res.interpretation_note
    assert "after a trained prediction model is integrated" in res.message


def test_proposed_fusion_strategy_is_not_faked():
    res = fuse({"a": np.ones(3)}, "cross_attention_transformer", "l2")
    assert res.state == SectionState.NOT_AVAILABLE
    res = fuse({"a": np.ones(3)}, "concatenate", "fitted")
    assert res.state == SectionState.MODEL_NOT_READY


def test_api_never_marks_live_results_simulated(client):
    r = client.post("/api/analysis", files={"file": ("a.wav", wav_bytes(voiced_bursts()), "audio/wav")})
    aid = r.json()["analysis_id"]
    wait_for_terminal(client, aid)
    full = client.get(f"/api/analysis/{aid}").json()
    assert full["simulated"] is False and full["mode"] == "live"
    sev = full["severity"]
    assert sev["state"] == "MODEL_NOT_READY" and sev["prediction_available"] is False
    assert all(sev[f] is None for f in SEVERITY_VALUE_FIELDS) and "probabilities" not in sev
    assert full["explainability"]["methods"] == []
    models = client.get("/api/models").json()
    assert models["severity_classifier"]["model_card"]["status"] == "MODEL_NOT_READY"
    assert models["severity_classifier"]["target_definition"]["name"] == "WAB-R-derived severity score"


@pytest.fixture
def promoted_fixture_bundle(tmp_path, settings):
    """Train a tiny acoustic-only ridge model on SYNTHETIC_TEST_DATA, then (TEST ONLY) relabel it as a promoted
    bundle in a temp directory to exercise the production prediction path end to end."""
    from app.training import train
    from app.training.synthetic import generate

    root = generate(tmp_path / "ds", participants=8, seconds=2.0)
    cfg = {"experiment": "FIXTURE", "dataset_config": "configs/datasets/synthetic.yaml", "dataset_root": str(root),
           "split": {"val_fraction": 0.25, "test_fraction": 0.25, "seed": 1},
           # audio_scope full: an uploaded training file then yields exactly the features the model was fitted on
           "features": {"groups": ["acoustic"], "language_source": "human", "audio_scope": "full"},
           "model": {"type": "ridge", "alphas": [1.0, 10.0, 100.0]}}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(cfg))
    exp = train.run(tmp_path / "exp.yaml", evaluate_test=True, synthetic_test_run=True, settings=settings,
                    artifacts=tmp_path / "artifacts")
    shutil.copytree(exp / "bundle", settings.severity_model_dir)
    meta = json.loads((settings.severity_model_dir / "bundle.json").read_text())
    meta.update(data_origin="REAL", status="EVALUATED_ON_HELD_OUT_TEST")  # TEST FIXTURE ONLY
    (settings.severity_model_dir / "bundle.json").write_text(json.dumps(meta))
    (settings.severity_model_dir / "promotion.json").write_text(json.dumps({"reviewer": "pytest fixture"}))
    split = json.loads((exp / "split.json").read_text())
    settings.__dict__["_train_audio"] = (root / "audio" / f"{split['train'][0]}a.wav").read_bytes()
    return settings


def _service(settings):
    from app.models.registry import ModelRegistry

    return SeverityService(settings.severity_model_dir, settings, ModelRegistry(settings))


def test_unpromoted_bundle_is_not_served(promoted_fixture_bundle):
    (promoted_fixture_bundle.severity_model_dir / "promotion.json").unlink()
    assert _service(promoted_fixture_bundle).status()["state"] == "MODEL_NOT_READY"


def test_runtime_configuration_mismatch_keeps_model_not_ready(promoted_fixture_bundle):
    """A model must not run under a feature-extraction configuration it was not trained with."""
    s = promoted_fixture_bundle
    assert _service(s).status()["state"] == "READY"
    s.vad_aggressiveness = 3  # changes preprocessing -> different feature values
    st = _service(s).status()
    assert st["state"] == "MODEL_NOT_READY" and "differs from the one the model was trained with" in st["reason"]


def test_prediction_contract_and_personal_recording_protection(promoted_fixture_bundle):
    """Contract once a real model exists: 0-100 score, predicted AQ, band, raw output, no probabilities,
    and every uploaded / microphone recording is NOT_VALID_FOR_CLINICAL_INFERENCE."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    s = promoted_fixture_bundle
    with TestClient(create_app(s)) as c:
        aid = c.post("/api/analysis", files={"file": ("mic.wav", s.__dict__["_train_audio"], "audio/wav")}).json()["analysis_id"]
        wait_for_terminal(c, aid)
        sev = c.get(f"/api/analysis/{aid}/severity").json()
        xai = c.get(f"/api/analysis/{aid}/explainability").json()
        hist = c.get("/api/history").json()["items"][0]
        pdf = c.get(f"/api/reports/{aid}")
    assert sev["prediction_available"] is True and 0.0 <= sev["severity_score"] <= 100.0, sev["message"]
    assert sev["predicted_aq"] == pytest.approx(100 - sev["severity_score"], abs=0.051)
    assert round(sev["severity_score"], 1) == sev["severity_score"]  # reported to one decimal
    assert sev["score_direction"] == "higher_worse" and sev["target_definition"] == "WAB-R-derived severity score"
    assert sev["severity_band"] in ("Very severe", "Severe", "Moderate", "Mild") and sev["raw_output"] is not None
    assert sev["clinical_validity"] == NOT_VALID and "probabilities" not in sev and sev["limitations"]
    assert sev["input_checks"]["out_of_distribution_groups"] == []
    assert sev["model_version"].startswith("FIXTURE_")
    # the explanation belongs to exactly this prediction and model version
    assert xai["model_version"] == sev["model_version"] and xai["explained_raw_output"] == sev["raw_output"]
    assert {m["method"] for m in xai["methods"]} == {"shap", "lime"}
    shap = next(m for m in xai["methods"] if m["method"] == "shap")
    assert shap["target"] == "WAB-R-derived severity score" and shap["quality"]["additivity_error"] < 0.5
    assert shap["quality"]["explained_output"] == pytest.approx(sev["raw_output"], abs=0.01)
    lime = next(m for m in xai["methods"] if m["method"] == "lime")
    assert "local_r2" in lime["quality"] and lime["reliable"] in (True, False)
    assert hist["severity_score"] == sev["severity_score"] and pdf.status_code == 200


def test_invalid_or_out_of_distribution_output_is_never_shown_as_a_score(promoted_fixture_bundle):
    """A recording unlike the training data must not be clipped into a valid-looking score or explained."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(promoted_fixture_bundle)) as c:
        aid = c.post("/api/analysis", files={"file": ("mic.wav", wav_bytes(voiced_bursts()), "audio/wav")}).json()["analysis_id"]
        status = wait_for_terminal(c, aid)
        full = c.get(f"/api/analysis/{aid}").json()
        pdf = c.get(f"/api/reports/{aid}")
    sev, xai = full["severity"], full["explainability"]
    assert sev["state"] in ("OUT_OF_DISTRIBUTION", "INVALID_OUTPUT") and sev["prediction_available"] is False
    assert sev["severity_score"] is None and sev["predicted_aq"] is None and sev["severity_band"] is None
    assert sev["raw_output"] is not None and "no score is reported" in sev["message"].lower()  # raw value preserved
    assert xai["state"] == "NOT_AVAILABLE" and xai["methods"] == []
    assert status["status"] == "ATTENTION_NEEDED" and any("no valid score" in a for a in status["attention"])
    assert pdf.status_code == 200


def test_invalid_raw_output_and_dimension_mismatch(promoted_fixture_bundle):
    svc = _service(promoted_fixture_bundle)
    b = svc.bundle()
    row = b.background["acoustic"][0][: b.pipeline.input_dims["acoustic"]]
    svc.ood_factor = 0  # isolate the output-range check
    res, raw = svc.predict({"acoustic": row * 50})
    assert res.state == SectionState.INVALID_OUTPUT and res.severity_score is None and raw is None
    assert res.raw_output > 110 or res.raw_output < -10
    res, _ = svc.predict({"acoustic": row[:-3]})  # wrong width
    assert res.state == SectionState.INFERENCE_FAILED and "expected" in res.message and res.severity_score is None
    res, _ = svc.predict({"acoustic": None})
    assert res.state == SectionState.INSUFFICIENT_DATA and "No score is reported" in res.message


def test_inputs_missing_at_inference_are_flagged_not_silently_imputed(promoted_fixture_bundle):
    svc = _service(promoted_fixture_bundle)
    b = svc.bundle()
    row = b.background["acoustic"][0][: b.pipeline.input_dims["acoustic"]].copy()
    row[0] = np.nan  # mfcc1_mean was never missing in the fixture's training data
    res, _ = svc.predict({"acoustic": row})
    assert res.input_checks["imputed_but_never_missing_in_training"] == {"acoustic": ["mfcc1_mean"]}
    if res.prediction_available:
        assert "never missing in training" in res.message


def test_lime_is_not_degenerate_off_distribution_and_reports_fit(promoted_fixture_bundle):
    """Regression: LIME's default (sampling from the training distribution) gave all-zero weights for
    out-of-distribution recordings."""
    from app.explainability.explain import lime_explanation

    b = _service(promoted_fixture_bundle).bundle()
    k = b.pipeline.input_dims["acoustic"]
    raw = b.pipeline.raw({"acoustic": b.background["acoustic"][:1, :k] * 3}, {"acoustic": b.input_names("acoustic")})
    lime = lime_explanation(b, raw)
    assert lime.state == SectionState.COMPUTED and any(abs(a.contribution) > 0 for a in lime.attributions)
    assert "local R" in lime.message and lime.quality["local_r2"] is not None and lime.reliable in (True, False)


def test_asr_finding_no_speech_needs_attention(settings, monkeypatch):
    """VAD may accept amplified noise; ASR's 'no transcribable speech' must still surface at job level."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.pipelines import asr
    from app.schemas.analysis import TranscriptResult

    settings.asr_enabled = True
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: TranscriptResult(
        state=SectionState.INSUFFICIENT_DATA, message="ASR produced no transcribable speech."))
    with TestClient(create_app(settings)) as c:
        aid = c.post("/api/analysis", files={"file": ("a.wav", wav_bytes(voiced_bursts()), "audio/wav")}).json()["analysis_id"]
        status = wait_for_terminal(c, aid)
    assert status["status"] == "ATTENTION_NEEDED"
    assert any("No speech detected" in a for a in status["attention"])


def test_interrupted_jobs_are_marked_failed_on_restart(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.schemas.analysis import AnalysisConfig, AnalysisRecord

    with TestClient(create_app(settings)) as c:
        repo = c.app.state.services.repo
        repo.save(AnalysisRecord(analysis_id="FS-TEST-1", status=AnalysisStatus.ASR, created_at=now(), updated_at=now(),
                                 config=AnalysisConfig()))
    with TestClient(create_app(settings)) as c:
        body = c.get("/api/analysis/FS-TEST-1/status").json()
        assert body["status"] == "FAILED" and "restart" in body["error"]["message"]


def test_cancelled_state_is_preserved(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.schemas.analysis import AnalysisConfig, AnalysisRecord

    with TestClient(create_app(settings)) as c:
        svc = c.app.state.services
        svc.repo.save(AnalysisRecord(analysis_id="FS-TEST-2", status=AnalysisStatus.QUEUED, created_at=now(),
                                     updated_at=now(), config=AnalysisConfig()))
        assert c.post("/api/analysis/FS-TEST-2/cancel").json()["status"] == "CANCELLED"
        assert c.get("/api/analysis/FS-TEST-2").json()["status"] == "CANCELLED"


# ====================================================================== pilot (text-only, cross-validated) model gate
@pytest.fixture
def pilot_bundle(tmp_path, settings):
    """A PILOT bundle built by the real export code from made-up numbers (TEST FIXTURE: the values mean nothing)."""
    import sys

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from export_pilot_bundle import export
    from qab_text_pilot import ASR_COMPATIBLE, MIN_WORDS, _fit_predict

    rng = np.random.default_rng(0)
    pids = [f"t{i}" for i in range(10)]
    y = {p: float(v) for p, v in zip(pids, np.linspace(10, 80, 10))}
    rows = [{"pid": p, "words": 200, "moving_average_ttr": 0.9 - y[p] / 200 + rng.normal(0, 0.01),
             "clause_marker_rate": 6 - y[p] / 20 + rng.normal(0, 0.1)} for p in pids for _ in range(3)]
    use = pd.DataFrame(rows)
    summary = {"method": "leave-one-participant-out cross-validation on human transcripts", "participants": 10,
               "participants_in_dataset": 11, "transcripts": 30, "transcripts_in_dataset": 33, "excluded_transcripts": 3,
               "min_words": MIN_WORDS, "mae": 5.0, "mean_baseline_mae": 20.0, "r2": 0.9, "p_familywise": 0.01,
               "dataset": "TEST FIXTURE", "validated_on_whisper_transcripts": False}
    exp = tmp_path / "pilot_exp"
    export(exp / "bundle", use=use, feats=ASR_COMPATIBLE, y_by_pid=y, pilot_id="PILOT_FIXTURE", min_words=MIN_WORDS,
           fit_predict=_fit_predict, summary=summary,
           uncertainty={"half_width": 12.0, "coverage": None, "method": "largest leave-one-participant-out error (fixture)"})
    return exp, settings


def _whisper_like(words: int):
    """A fake ASR result with a controlled vocabulary size (stands in for Whisper in these tests)."""
    from app.schemas.analysis import TranscriptResult, TranscriptSegment, TranscriptWord

    vocab = ("the boy went to the store because he wanted some bread and then he walked home when it was dark so that "
             "his mother who was waiting could cook dinner before the others arrived").split()
    toks = [vocab[i % len(vocab)] for i in range(words)]
    seg = TranscriptSegment(id=0, start=0.0, end=5.0, text=" ".join(toks) + ".",
                            words=[TranscriptWord(word=w, start=i * 0.1, end=i * 0.1 + 0.05, probability=0.9)
                                   for i, w in enumerate(toks)])
    return TranscriptResult(state=SectionState.COMPUTED, model="whisper-test", engine="test", text=seg.text,
                            segments=[seg], word_count=words)


def test_pilot_bundle_needs_opt_in_and_pilot_promotion(pilot_bundle):
    from app.training.bundle import BundleRejected
    from app.training.promote import promote

    exp, settings = pilot_bundle
    assert json.loads((exp / "bundle" / "bundle.json").read_text())["status"] == "PILOT_CROSS_VALIDATED"
    with pytest.raises(BundleRejected, match="requires --pilot"):
        promote(exp, "tester", settings.severity_model_dir)
    promote(exp, "tester", settings.severity_model_dir, pilot=True)
    assert json.loads((settings.severity_model_dir / "promotion.json").read_text())["pilot"] is True
    st = _service(settings).status()  # promoted, but the operator has not enabled pilot models
    assert st["state"] == "MODEL_NOT_READY" and "FSA_ALLOW_PILOT_MODEL" in st["reason"]
    settings.allow_pilot_model = True
    st = _service(settings).status()
    assert st["state"] == "READY" and st["model_stage"] == "PILOT" and st["evaluation_summary"]["participants"] == 10
    (settings.severity_model_dir / "promotion.json").write_text(json.dumps({"reviewer": "x"}))  # promoted without --pilot
    assert _service(settings).status()["state"] == "MODEL_NOT_READY"


def test_pilot_prediction_from_asr_transcript_is_labelled_and_needs_enough_words(pilot_bundle, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from app.pipelines import asr
    from app.training.promote import promote

    exp, settings = pilot_bundle
    promote(exp, "tester", settings.severity_model_dir, pilot=True)
    settings.allow_pilot_model, settings.asr_enabled, settings.ood_factor = True, True, 0
    spoken = {"n": 120}
    monkeypatch.setattr(asr, "transcribe", lambda *a, **k: _whisper_like(spoken["n"]))
    with TestClient(create_app(settings)) as c:
        def analyse():
            aid = c.post("/api/analysis", files={"file": ("a.wav", wav_bytes(voiced_bursts()), "audio/wav")}).json()["analysis_id"]
            wait_for_terminal(c, aid)
            return c.get(f"/api/analysis/{aid}").json(), c.get(f"/api/reports/{aid}").status_code

        full, pdf = analyse()
        sev = full["severity"]
        assert sev["state"] in ("COMPUTED", "INVALID_OUTPUT"), sev["message"]
        assert sev["model_stage"] == "PILOT" and sev["evaluation_summary"]["method"].startswith("leave-one-participant-out")
        assert any("NOT VALIDATED ON WHISPER TRANSCRIPTS" in x for x in sev["limitations"]) and pdf == 200
        assert sev["target_definition"] == "WAB-AQ-derived severity score"  # WAB (1982), never labelled WAB-R
        assert "1982" in sev["target"]["source"] and sev["target"]["provenance"]["column"] == "AQ"
        if sev["prediction_available"]:
            assert sev["clinical_validity"] == NOT_VALID and sev["message"].startswith("PILOT model output")
            assert 0 <= sev["severity_score"] <= 100 and "probabilities" not in sev
            assert full["explainability"]["model_version"] == "PILOT_FIXTURE"
            names = {a["feature"] for m in full["explainability"]["methods"] for a in m["attributions"]}
            assert names <= {"moving_average_ttr", "clause_marker_rate"}
        spoken["n"] = 12  # too short for the model's features
        short, _ = analyse()
        assert short["severity"]["state"] == "INSUFFICIENT_DATA" and short["severity"]["severity_score"] is None
        assert "at least 25 transcribed words" in short["severity"]["message"]
        assert short["explainability"]["methods"] == []


def test_pilot_refuses_inputs_beyond_the_training_range(pilot_bundle):
    """A linear model extrapolates without limit: a feature outside the fitted range must give no score, even when the
    raw output happens to land inside 0-100 (regression: such inputs used to be clipped into a valid-looking score)."""
    from app.training.features import LANGUAGE_FEATURES
    from app.training.promote import promote

    exp, settings = pilot_bundle
    promote(exp, "tester", settings.severity_model_dir, pilot=True)
    settings.allow_pilot_model = True
    svc = _service(settings)
    ranges = svc.bundle().pipeline.feature_ranges["language"]
    lo, hi, sd = ranges["moving_average_ttr"]

    def predict(mattr):
        vec = {n: np.nan for n in LANGUAGE_FEATURES}
        vec.update(word_count=200.0, moving_average_ttr=mattr, clause_marker_rate=float(np.mean(ranges["clause_marker_rate"][:2])))
        return svc.predict({"language": np.array([vec[n] for n in LANGUAGE_FEATURES])})[0]

    inside = predict((lo + hi) / 2)
    assert inside.prediction_available and inside.input_checks["features_outside_training_range"] == {}
    for value in (hi + 0.6 * sd, lo - 0.6 * sd):  # just beyond the allowed 0.5 SD margin
        res = predict(value)
        assert res.state == SectionState.OUT_OF_DISTRIBUTION and res.severity_score is None and res.raw_output is not None
        assert "moving_average_ttr" in res.input_checks["features_outside_training_range"]["language"]
        assert "extrapolating" in res.message
    assert predict(hi + 0.4 * sd).prediction_available  # within the margin


def test_first_pilot_script_can_no_longer_export_a_bundle():
    text = (Path(__file__).resolve().parent.parent / "scripts" / "qab_text_pilot.py").read_text(encoding="utf-8")
    assert "from export_pilot_bundle import export" not in text and "no longer exports bundles" in text
