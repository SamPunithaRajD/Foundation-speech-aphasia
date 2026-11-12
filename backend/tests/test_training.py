"""Training infrastructure tests.

All data here is SYNTHETIC_TEST_DATA generated in a temporary directory. It exercises software
only; nothing trained here is, or can become, a clinical model.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from app.config import Settings
from app.training import target as tgt
from app.training.bundle import (STATUS_SYNTHETIC, STATUS_TESTED, BundleRejected, load_bundle)
from app.training.chat import normalize_utterance, parse_chat
from app.training.dataset import (SYNTHETIC, AphasiaDataset, DatasetConfig, DatasetConfigError, DatasetNotAvailable)
from app.training.evaluation import band_metrics, evaluate_split, regression_metrics, residual_interval
from app.training.features import LANGUAGE_FEATURES
from app.training.models import InputPipeline, RidgeModel, TorchRegressor
from app.training.splits import ParticipantLeakage, assert_no_leakage, split_participants
from app.training.synthetic import generate

BACKEND = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ target
@pytest.mark.parametrize("bad", [None, "", "abc", float("nan"), float("inf"), -0.1, 100.01])
def test_aq_validation_rejects(bad):
    with pytest.raises(tgt.InvalidAQ):
        tgt.validate_aq(bad)


def test_severity_transformation_and_direction():
    assert tgt.severity_from_aq(100) == 0 and tgt.severity_from_aq(0) == 100 and tgt.severity_from_aq("62.5") == 37.5
    assert tgt.aq_from_severity(37.5) == 62.5
    assert tgt.SCORE_DIRECTION == "higher_worse" and tgt.TARGET_NAME == "WAB-R-derived severity score"


@pytest.mark.parametrize("aq,band", [(0, "Very severe"), (25, "Very severe"), (25.9, "Very severe"), (26, "Severe"),
                                     (50, "Severe"), (51, "Moderate"), (75, "Moderate"), (76, "Mild"), (100, "Mild")])
def test_published_wab_bands(aq, band):
    assert tgt.band_from_aq(aq) == band


def test_prediction_range_clipped_only_at_output_with_raw_kept():
    p = tgt.finalize_prediction(-7.25)
    assert p.raw_severity == -7.25 and p.severity_score == 0.0 and p.predicted_aq == 100.0 and p.clipped
    q = tgt.finalize_prediction(63.428193)
    assert q.severity_score == 63.4 and q.predicted_aq == 36.6 and not q.clipped and q.band == "Severe"  # AQ 26–50


# ------------------------------------------------------------------ CHAT
def test_chat_normalisation_rules():
    text, unintel = normalize_utterance("&-um <I want> [/] I want (be)cause xxx the dog@n 0is [* p] &=laughs &+fr . \x15100_900\x15")
    assert text == "um i want i want because the dog" and unintel == 1
    assert normalize_utterance("&-um hello .", keep_fillers=False)[0] == "hello"


def test_chat_parser_preserves_original(tmp_path):
    p = tmp_path / "x01a.cha"
    content = ("@UTF8\n@Begin\n@Participants:\tPAR Participant, INV Investigator\n"
               "@ID:\teng|corpus|PAR|60;|male|Broca||Participant|||\n@Media:\tx01a, video\n@G:\tCinderella\n"
               "*PAR:\tonce upon\n\ta time . \x151000_2500\x15\n%mor:\tadv|once\n*INV:\tgood . \x152500_3000\x15\n@End\n")
    p.write_text(content, encoding="utf-8")
    c = parse_chat(p)
    assert c.original == content  # verbatim
    assert c.media == {"name": "x01a", "type": "video", "raw": "x01a, video"}
    assert c.speaker_ids("PAR")["group"] == "Broca"
    u = c.speaker_utterances("PAR")[0]
    assert (u.text, u.start_ms, u.end_ms, u.gem) == ("once upon a time", 1000, 2500, "Cinderella")
    assert "once upon" in u.raw and u.dependent_tiers["mor"] == "adv|once"
    assert c.normalized_text("PAR") == "once upon a time" and c.normalized_text("INV") == "good"


# ------------------------------------------------------------------ dataset
def synthetic_config(root: Path) -> DatasetConfig:
    cfg = yaml.safe_load((BACKEND / "configs/datasets/synthetic.yaml").read_text())
    cfg["root"] = str(root)
    return DatasetConfig.from_dict(cfg)


def test_dataset_not_available_without_root(monkeypatch):
    monkeypatch.delenv("APHASIA_DATA_ROOT", raising=False)
    monkeypatch.delenv("FSA_APHASIA_DATA_ROOT", raising=False)
    with pytest.raises(DatasetNotAvailable):
        AphasiaDataset(DatasetConfig(name="x")).load()


def test_aphasiabank_config_requires_real_column_names(tmp_path):
    for d in ("audio", "transcripts", "scores"):
        (tmp_path / d).mkdir()
    cfg = DatasetConfig.from_dict({**yaml.safe_load((BACKEND / "configs/datasets/aphasiabank.yaml").read_text()),
                                   "root": str(tmp_path)})
    with pytest.raises(DatasetConfigError, match="not guessed"):
        AphasiaDataset(cfg).load()


def test_dataset_schema_and_matching(tmp_path):
    root = generate(tmp_path / "ds", participants=4, seconds=2.0)
    records, report = AphasiaDataset(synthetic_config(root)).load()
    assert report.data_origin == SYNTHETIC and all(r.data_origin == SYNTHETIC for r in records)
    assert report.total_audio_files == report.total_transcripts == report.matched_records == 8
    assert report.trainable_records == 8 and report.trainable_participants == 4
    r = records[0]
    for f in ("participant_id", "session_id", "audio_path", "transcript_path", "task", "wab_aq", "severity_score",
              "target_source", "target_version", "human_transcript", "aphasia_metadata", "demographics", "provenance"):
        assert hasattr(r, f)
    assert r.severity_score == pytest.approx(100 - r.wab_aq) and r.target_version == tgt.TARGET_VERSION
    assert r.speaker_segments_ms and r.human_transcript


def test_missing_labels_are_reported_never_filled(tmp_path):
    root = generate(tmp_path / "ds", participants=5, seconds=2.0)
    rows = (root / "scores/synthetic_scores.csv").read_text().splitlines()
    rows[1] = rows[1].split(",")[0] + ","             # synth00: missing AQ
    rows[2] = rows[2].split(",")[0] + ",142"          # synth01: invalid AQ
    rows.append(rows[3].split(",")[0] + ",11.0")      # synth02: second, different row -> ambiguous
    rows.pop(4)                                        # synth03: no score row at all
    (root / "scores/synthetic_scores.csv").write_text("\n".join(rows) + "\n")
    (root / "audio/synth04a.wav").unlink()            # missing audio
    (root / "audio/orphan.wav").write_bytes((root / "audio/synth04b.wav").read_bytes())  # audio w/o transcript
    records, report = AphasiaDataset(synthetic_config(root)).load()
    assert len(records) == 10  # nothing discarded
    by = {r.participant_id: r for r in records}
    assert by["synth00"].wab_aq is None and not by["synth00"].trainable
    assert any(i.startswith("invalid WAB-R AQ") for i in by["synth01"].issues)
    assert any(i.startswith("ambiguous WAB-R AQ") for i in by["synth02"].issues)
    assert "missing WAB-R AQ" in by["synth03"].issues
    assert len(report.missing_aq) == 4 and len(report.invalid_aq) == 2 and len(report.ambiguous_aq) == 2
    assert report.missing_audio == ["synth04/a/all"]
    assert [p.replace("\\", "/") for p in report.missing_transcripts] == ["audio/orphan.wav"]
    assert report.trainable_participants == 1  # only synth04 session b


def test_duplicate_records_are_flagged(tmp_path):
    root = generate(tmp_path / "ds", participants=3, seconds=2.0)
    sub = root / "transcripts" / "copy"
    sub.mkdir()
    (sub / "synth00a.cha").write_text((root / "transcripts/synth00a.cha").read_text(encoding="utf-8"), encoding="utf-8")
    records, report = AphasiaDataset(synthetic_config(root)).load()
    assert len(report.duplicate_records) == 2 and all(not r.trainable for r in records if r.record_id == "synth00/a/all")


def test_personal_recording_directories_can_never_be_a_dataset(tmp_path):
    app_data = tmp_path / "app-data"
    generate(app_data / "uploads", participants=3, seconds=2.0)
    with pytest.raises(DatasetConfigError, match="never be used as training data"):
        AphasiaDataset(synthetic_config(app_data / "uploads"), protected_dirs=[app_data]).load()


# ------------------------------------------------------------------ splits / leakage
def test_leakage_detector_fails_on_overlap():
    with pytest.raises(ParticipantLeakage):
        assert_no_leakage({"train": ["p1", "p2"], "validation": ["p3"], "test": ["p2"]})
    with pytest.raises(ParticipantLeakage):
        assert_no_leakage({"train": ["p1", "p1"], "validation": [], "test": []})


def test_participant_level_split_has_no_leakage(tmp_path):
    root = generate(tmp_path / "ds", participants=20, sessions=2, seconds=1.0)
    records, _ = AphasiaDataset(synthetic_config(root)).load()
    split = split_participants(records, 0.2, 0.2, seed=3, cv_folds=4)
    sets = [set(split.train), set(split.validation), set(split.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert set().union(*sets) == {r.participant_id for r in records}
    assert all(len(s) >= 2 for s in sets[1:])
    for fold in split.folds:  # CV never touches the test participants
        assert not set(fold["train"]) & set(fold["validation"]) and not (set(fold["validation"]) & sets[2])
    # every recording of a participant lands in the same split
    owner = {p: name for name, ps in split.assignments().items() for p in ps}
    assert len({(r.participant_id, owner[r.participant_id]) for r in records}) == 20
    assert split_participants(records, 0.2, 0.2, seed=3).to_dict()["test"] == split.to_dict()["test"]  # reproducible


# ------------------------------------------------------------------ models
def _data(n=24, dims=None, seed=0):
    rng = np.random.default_rng(seed)
    dims = dims or {"acoustic": 5, "embedding:wav2vec2": 12}
    X = {g: rng.normal(size=(n, d)) for g, d in dims.items()}
    y = np.clip(50 + 10 * X[next(iter(dims))][:, 0], 0, 100)
    return X, y


@pytest.mark.parametrize("model", [
    lambda g: RidgeModel(g, alpha=1.0),
    lambda g: TorchRegressor("mlp", g, epochs=3, hidden=[8]),
    lambda g: TorchRegressor("fusion_transformer", g, epochs=3, d_model=16, n_heads=4, n_layers=1, ff=32),
])
def test_forward_pass_and_checkpoint_roundtrip(model, tmp_path):
    X, y = _data()
    m = model(list(X))
    m.fit(X, y, X, y)
    pred = m.predict(X)
    assert pred.shape == (24,) and np.all(np.isfinite(pred))
    m.save(tmp_path)
    m2 = model(list(X))
    m2.load(tmp_path)
    np.testing.assert_allclose(m2.predict(X), pred, rtol=1e-5, atol=1e-4)


def test_feature_dimensions():
    from app.features.acoustic import acoustic_feature_names, acoustic_features
    from conftest import voiced_bursts

    feats = acoustic_features(voiced_bursts(), 16000)
    assert len(feats) == len(acoustic_feature_names()) == 38
    assert all(f.status == "computed" for f in feats)
    assert "syntactic_complexity" not in LANGUAGE_FEATURES and len(LANGUAGE_FEATURES) == 16


# ------------------------------------------------------------------ evaluation
def test_regression_and_band_metrics():
    y = np.array([10.0, 30.0, 60.0, 90.0, 45.0])
    raw = np.array([12.0, 28.0, 65.0, 104.0, 40.0])
    m = regression_metrics(y, raw)
    assert m["mae"] == pytest.approx(np.mean([2, 2, 5, 10, 5])) and m["fraction_clipped"] == 0.2
    assert m["raw_output_range"] == [12.0, 104.0] and m["r2"] is not None and m["spearman"]["value"] == pytest.approx(1.0)
    b = band_metrics(y, raw)
    assert "derived from WAB-R AQ" in b["note"] and len(b["confusion_matrix"]["matrix"]) == 4
    assert regression_metrics(np.array([5.0, 5.0]), np.array([5.0, 6.0]))["pearson"]["value"] is None
    ev = evaluate_split(["a", "a", "b", "c", "d"], y, raw)
    assert ev["participant_level"]["participants"] == 4 and ev["mae_95ci"]["low"] is None  # < 5 participants
    assert residual_interval(y[:3], raw[:3])["half_width"] is None


# ------------------------------------------------------------------ end-to-end synthetic training
@pytest.fixture
def train_env(tmp_path):
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    tiny = tmp_path / "tiny-w2v"  # random-init local model: exercises the wav2vec2 path without any download
    cfg = Wav2Vec2Config(hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                         conv_dim=(8, 8), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2)
    Wav2Vec2Model(cfg).save_pretrained(tiny)
    Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(tiny)
    settings = Settings(_env_file=None, data_dir=tmp_path / "app-data", temp_dir=tmp_path / "app-data" / "tmp",
                        severity_model_dir=tmp_path / "prod-model", wav2vec2_model_id=str(tiny), asr_enabled=False)
    root = generate(tmp_path / "ds", participants=10, seconds=2.0)
    exp = {"experiment": "SYNTH_C", "dataset_config": str(BACKEND / "configs/datasets/synthetic.yaml"),
           "dataset_root": str(root), "seed": 7,
           "split": {"val_fraction": 0.2, "test_fraction": 0.2, "seed": 7, "cv_folds": 2},
           "features": {"groups": ["embedding:wav2vec2", "language", "acoustic"], "language_source": "human"},
           "model": {"type": "fusion_transformer", "d_model": 16, "n_heads": 2, "n_layers": 1, "ff": 32, "epochs": 5}}
    cfg_path = tmp_path / "exp.yaml"
    cfg_path.write_text(yaml.safe_dump(exp))
    return settings, cfg_path, tmp_path / "artifacts"


def test_synthetic_training_end_to_end_is_quarantined(train_env):
    from app.training import train

    settings, cfg_path, artifacts = train_env
    with pytest.raises(train.TrainingBlocked, match="synthetic-test-run"):
        train.run(cfg_path, settings=settings, artifacts=artifacts)
    exp = train.run(cfg_path, evaluate_test=True, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    assert exp.parent == artifacts / "synthetic_test"  # never under artifacts/experiments
    meta = json.loads((exp / "bundle" / "bundle.json").read_text())
    assert meta["data_origin"] == SYNTHETIC and meta["status"] == STATUS_SYNTHETIC
    card = json.loads((exp / "model_card.json").read_text())
    assert card["status"] == STATUS_SYNTHETIC and card["split"]["leakage_check"] == "passed"
    assert "SYNTHETIC_TEST_DATA" in (exp / "MODEL_CARD.md").read_text(encoding="utf-8")
    metrics = json.loads((exp / "metrics.json").read_text())
    assert {"train", "validation", "test", "cross_validation"} <= set(metrics)
    split = json.loads((exp / "split.json").read_text())
    assert not set(split["train"]) & set(split["test"]) and not set(split["validation"]) & set(split["test"])
    assert (artifacts / "synthetic_test" / "test_access.log").read_text().count("SYNTH_C") == 1
    # a synthetic bundle can never be loaded for production, promoted, or served
    with pytest.raises(BundleRejected):
        load_bundle(exp / "bundle")
    from app.training.promote import promote

    with pytest.raises(BundleRejected):
        promote(exp, "tester", settings.severity_model_dir)
    import shutil

    from app.pipelines.severity import SeverityService

    shutil.copytree(exp / "bundle", settings.severity_model_dir)
    (settings.severity_model_dir / "promotion.json").write_text("{}")
    svc = SeverityService(settings.severity_model_dir)
    assert svc.status()["state"] == "MODEL_NOT_READY" and "SYNTHETIC" in svc.status()["reason"]
    assert svc.predict({})[0].prediction_available is False


def test_synthetic_flag_cannot_relabel_real_data(train_env, tmp_path):
    from app.training import train

    settings, cfg_path, artifacts = train_env
    cfg = yaml.safe_load(cfg_path.read_text())
    os.remove(Path(cfg["dataset_root"]) / "SYNTHETIC_TEST_DATA")
    with pytest.raises(train.TrainingBlocked, match="not marked SYNTHETIC_TEST_DATA"):
        train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    with pytest.raises(train.TrainingBlocked, match="minimum 20"):  # too few participants for a real run
        train.run(cfg_path, settings=settings, artifacts=artifacts)


def test_validation_only_bundle_is_not_promotable(train_env):
    from app.training import train
    from app.training.promote import promote

    settings, cfg_path, artifacts = train_env
    exp = train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    meta = json.loads((exp / "bundle" / "bundle.json").read_text())
    assert "test" not in json.loads((exp / "metrics.json").read_text())
    meta.update(data_origin="REAL", status="TRAINED_VALIDATION_ONLY")  # even if the origin were real...
    (exp / "bundle" / "bundle.json").write_text(json.dumps(meta))
    with pytest.raises(BundleRejected, match=STATUS_TESTED):  # ...no held-out test evaluation -> refused
        promote(exp, "tester", settings.severity_model_dir)


# ====================================================================== audit regression tests
def test_out_of_range_output_is_invalid_not_clipped():
    p = tgt.finalize_prediction(1093.8)
    assert not p.valid and p.severity_score is None and p.predicted_aq is None and p.band is None
    assert p.raw_severity == 1093.8 and "not a valid severity estimate" in p.reason
    assert not tgt.finalize_prediction(float("nan")).valid and not tgt.finalize_prediction(-10.01).valid
    edge = tgt.finalize_prediction(104.0)  # small overshoot: clipped AND flagged
    assert edge.valid and edge.severity_score == 100.0 and edge.clipped


def test_missingness_is_represented_and_inference_gaps_are_reported(tmp_path):
    names = {"language": ["a", "b", "c"]}
    train = {"language": np.array([[1.0, np.nan, np.nan], [3.0, 4.0, np.nan], [5.0, 6.0, np.nan]])}
    p = InputPipeline(["language"]).fit(train, names)
    assert p.source_names["language"] == ["a", "b"] and p.dropped["language"] == ["c"]
    assert p.names["language"] == ["a", "b", "b__missing"] and p.indicators["language"] == ["b"]
    x = {"language": np.array([[np.nan, np.nan, 5.0]])}
    np.testing.assert_allclose(p.raw(x, names)["language"], [[3.0, 5.0, 1.0]])  # train medians + indicator set
    assert p.missing_report(x, names) == {"language": {"seen_in_training": ["b"], "never_missing_in_training": ["a"]}}
    with pytest.raises(ValueError, match="trained with 3"):
        p.raw({"language": np.zeros((1, 2))}, names)
    assert set(p.ood_reference["language"]) == {"train_max", "train_p95", "train_median"}
    far = p.ood_scores(p.raw({"language": np.array([[500.0, 500.0, 0.0]])}, names))["language"][0]
    assert far > 10 * p.ood_reference["language"]["train_max"]
    p.save(tmp_path)
    q = InputPipeline.load(tmp_path)
    np.testing.assert_allclose(q.transform(train, names)["language"], p.transform(train, names)["language"])
    assert q.indicators == p.indicators and q.input_dims == {"language": 3}


def test_fusion_transformer_uses_every_declared_group():
    X, y = _data(dims={"embedding:wav2vec2": 12, "language": 4, "acoustic": 5})
    m = TorchRegressor("fusion_transformer", list(X), epochs=2, d_model=16, n_heads=4)
    m.fit(X, y)
    base = m.predict(X)
    for g in X:  # changing any one group's input must change the output
        changed = {k: (v + 3.0 if k == g else v) for k, v in X.items()}
        assert np.abs(m.predict(changed) - base).max() > 1e-4, g
    assert m.n_params() > 0 and m.hp["grad_clip"] == 1.0
    with pytest.raises(ValueError, match="divisible"):
        TorchRegressor("fusion_transformer", list(X), d_model=30, n_heads=4)


def test_ridge_alpha_is_selected_on_validation_only():
    X, y = _data(n=40)
    tr = {g: v[:30] for g, v in X.items()}
    va = {g: v[30:] for g, v in X.items()}
    m = RidgeModel(list(X), alphas=[0.01, 1.0, 1000.0])
    info = m.fit(tr, y[:30], va, y[30:])
    assert m.alpha in (0.01, 1.0, 1000.0) and len(info["alpha_search"]) == 3
    assert m.alpha == min(info["alpha_search"], key=lambda r: r["val_mae"])["alpha"] and info["selection"] == "validation MAE"
    assert RidgeModel(list(X), alphas=[0.5, 5.0]).fit(tr, y[:30])["selection"].startswith("fixed")  # no val -> no search


def test_saved_split_is_not_silently_reused_when_dataset_changes(tmp_path):
    from app.training.splits import check_reusable, inner_split

    root = generate(tmp_path / "ds", participants=12, seconds=1.0)
    records, _ = AphasiaDataset(synthetic_config(root)).load()
    split = split_participants(records, 0.2, 0.2, seed=3)
    ids = {r.participant_id for r in records}
    assert check_reusable(split, ids, {"val_fraction": 0.2, "test_fraction": 0.2, "stratify": True}) == []
    assert any("silently" in p for p in check_reusable(split, ids | {"newcomer"}, {}))
    assert any("no longer trainable" in p for p in check_reusable(split, ids - {split.train[0]}, {}))
    assert any("test_fraction" in p for p in check_reusable(split, ids, {"test_fraction": 0.3}))
    fit, sel = inner_split(split.train, 0.25, seed=1)
    assert sel and not set(fit) & set(sel) and set(fit) | set(sel) == set(split.train)


def test_changed_dataset_blocks_training_instead_of_dropping_participants(train_env):
    from app.training import train

    settings, cfg_path, artifacts = train_env
    train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    cfg = yaml.safe_load(cfg_path.read_text())
    scores = Path(cfg["dataset_root"]) / "scores" / "synthetic_scores.csv"
    scores.write_text("\n".join(scores.read_text().splitlines()[:-1]) + "\n")  # one participant loses its label
    with pytest.raises(train.TrainingBlocked, match="does not match the current dataset"):
        train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)


def test_feature_cache_is_shared_across_experiments_and_asr_is_not_run_needlessly(train_env):
    from app.training import train

    settings, cfg_path, artifacts = train_env
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg.update(experiment="SYNTH_B", features={"groups": ["embedding:wav2vec2"], "language_source": "whisper"},
               model={"type": "mlp", "hidden": [8], "epochs": 3})
    b_path = cfg_path.parent / "b.yaml"
    b_path.write_text(yaml.safe_dump(cfg))
    exp_b = train.run(b_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    stats_b = json.loads((exp_b / "feature_cache.json").read_text())
    feature_root = artifacts / "synthetic_test" / "features" / "synthetic-v1"
    assert stats_b["computed"] > 0 and not list(feature_root.glob("asr-*"))  # embedding-only model: Whisper never ran
    exp_c = train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)  # emb + language + acoustic
    stats_c = json.loads((exp_c / "feature_cache.json").read_text())
    assert stats_c["cache_hits"] >= 2 * 20  # preprocessing info + wav2vec2 embedding reused for all 20 recordings
    assert len(list(feature_root.glob("embedding-wav2vec2-*"))) == 1  # one shared embedding cache, not one per experiment
    # both experiments used the identical participant split
    assert json.loads((exp_b / "split.json").read_text())["test"] == json.loads((exp_c / "split.json").read_text())["test"]
    card = json.loads((exp_c / "model_card.json").read_text())
    assert card["training"]["complexity"]["n_parameters"] > 0 and "extraction_spec" in card["inputs"]
    assert "inner participant-level split" in json.loads((exp_c / "metrics.json").read_text())["cross_validation"]["selection"]
    meta = json.loads((exp_c / "bundle" / "bundle.json").read_text())
    assert {"extraction_spec", "input_names", "input_dims", "ood_reference", "n_parameters"} <= set(meta)


def test_failed_experiment_is_recorded(train_env, monkeypatch):
    from app.training import train

    settings, cfg_path, artifacts = train_env

    def boom(*a, **k):
        raise FloatingPointError("non-finite training loss at epoch 1")

    monkeypatch.setattr(train, "build_model", boom)
    with pytest.raises(FloatingPointError):
        train.run(cfg_path, synthetic_test_run=True, settings=settings, artifacts=artifacts)
    failed = list((artifacts / "synthetic_test").glob("SYNTH_C_*/FAILED.json"))
    assert len(failed) == 1 and json.loads(failed[0].read_text())["status"] == "FAILED"
    assert not (failed[0].parent / "bundle").exists()


def test_placeholder_config_cannot_be_trained_and_all_configs_are_valid():
    from app.training import train
    from app.training.features import FeatureConfig
    from app.training.models import build_model

    with pytest.raises(train.TrainingBlocked, match="placeholder"):
        train.run(BACKEND / "configs/experiments/base_paper_comparison.yaml")
    seen = set()
    for path in sorted((BACKEND / "configs/experiments").glob("*.yaml")):
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        if cfg.get("status") == "NOT_AVAILABLE":
            continue
        FeatureConfig(**cfg["features"]).validate()
        build_model(cfg["model"], cfg["features"]["groups"], 13)
        assert cfg["split"] == {"val_fraction": 0.15, "test_fraction": 0.15, "seed": 13, "stratify": True, "cv_folds": 5}
        seen.add(cfg["experiment"])
    assert {"A_handcrafted", "B_wav2vec2", "C_multimodal_wav2vec2", "B_wav2vec2_ridge", "C_concat_mlp"} <= seen


def test_excluded_features_and_wer_and_asr_quality():
    from app.pipelines.asr import quality_summary
    from app.schemas.analysis import TranscriptSegment
    from app.training.evaluation import transcript_agreement, word_error_rate
    from app.training.features import FeatureConfig

    with pytest.raises(ValueError, match="unknown feature names"):
        FeatureConfig(groups=["language"], exclude_features=["nope"]).validate()
    FeatureConfig(groups=["language"], exclude_features=["word_count"]).validate()
    assert word_error_rate("the cat sat on the mat", "the cat sat on mat") == pytest.approx(1 / 6)
    assert word_error_rate("", "x") is None
    agree = transcript_agreement({"r1": "a b c", "r2": None, "r3": "x y"}, {"r1": "a b d", "r3": None})
    assert agree["n_recordings"] == 1 and agree["wer_mean"] == pytest.approx(1 / 3)
    assert transcript_agreement({"r1": "a"}, {"r1": None}) is None
    q = quality_summary([TranscriptSegment(id=0, start=0, end=1, text="x", avg_logprob=-1.4, no_speech_prob=0.7,
                                           compression_ratio=3.1)])
    assert len(q["flags"]) == 3 and q["accuracy_verified"] is False
    assert quality_summary([TranscriptSegment(id=0, start=0, end=1, text="x", avg_logprob=-0.2, no_speech_prob=0.01,
                                              compression_ratio=1.2)])["flags"] == []


def test_qab_transcript_reader(tmp_path):
    from app.training.qab import normalize_line, parse_qab_transcript

    assert normalize_line("and that it's <n- a normal> not a normal thing...") == ("and that it's a normal not a normal thing...", 0, 1)
    assert normalize_line("um (.) I (..) think so: [nods]") == ("um I think so", 2, 0)
    p = tmp_path / "0120_T2_SMS.txt"
    content = "I went <to> to the s- store. [points]\n[mhm.]\n[oh good.] [laughs]\n\nit was (.) closed.\n"
    p.write_text(content, encoding="utf-8")
    t = parse_qab_transcript(p)
    assert (t.participant_id, t.session) == ("120", 2) and t.original == content  # original preserved
    assert t.utterances == ["I went to to the store.", "it was closed."] and t.examiner_turns == 2
    assert t.marked_pauses == 1 and t.fragments == 1
    with pytest.raises(ValueError):
        parse_qab_transcript(tmp_path / "notes.txt")


def test_language_features_without_audio_do_not_crash_or_invent_rates():
    from app.features.clinical import extract_features
    from app.schemas.analysis import PreprocessingResult
    from app.training.features import _human_transcript_result

    res = extract_features(_human_transcript_result("the cat sat on the mat. it was happy."), PreprocessingResult(), Settings())
    by = {f.name: f for f in res.features}
    assert by["speech_rate"].status == "not_computed" and by["speech_rate"].value is None
    assert by["articulation_rate"].value is None and by["pause_frequency"].value is None
    assert by["type_token_ratio"].status == "computed" and by["word_count"].value == 9
