"""Reproducible training command for the WAB-R-derived severity score.

    python -m app.training.train --config configs/experiments/model_c_multimodal.yaml
    python -m app.training.train --config ... --validate-only        # dataset validation report only
    python -m app.training.train --config ... --evaluate-test        # final, single held-out test evaluation

Steps: load config → validate dataset → participant-level split (shared per dataset version + seed,
so every model is compared on identical participants) → extract/cache features → fit on train
(model selection on validation only) → evaluate validation → [participant-level CV on train+validation,
with an inner selection split so the scored fold is never used for early stopping]
→ [--evaluate-test: held-out test, logged] → save bundle, metrics, config, environment, model card.

Model selection must use validation / CV metrics. Run ``--evaluate-test`` once, for the selected
configuration. Every test access is appended to artifacts/test_access.log. A run that fails after the
dataset was validated leaves a FAILED.json record in its experiment directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from app.config import BACKEND_DIR, Settings, get_settings
from app.models.hardware import hardware_summary
from app.models.registry import SPEC_BY_KEY, ModelRegistry
from app.training import evaluation as ev
from app.training.bundle import STATUS_SYNTHETIC, STATUS_TESTED, STATUS_VALIDATION_ONLY, save_bundle
from app.training.dataset import (REAL, SYNTHETIC, AphasiaDataset, DatasetConfig, DatasetNotAvailable,
                                  save_validation)
from app.training.features import FeatureConfig, FeatureExtractor, FeatureTable, build_table
from app.training.model_card import build_card, write_card
from app.training.models import InputPipeline, build_model, set_seed
from app.training.splits import (Split, assert_no_leakage, check_reusable, group_kfold, inner_split, participant_bands,
                                 records_for, save_split, split_participants, split_summary)

log = logging.getLogger("training")
ARTIFACTS = BACKEND_DIR / "artifacts"
MIN_REAL_PARTICIPANTS = 20
INNER_SELECTION_FRACTION = 0.2


class TrainingBlocked(RuntimeError):
    pass


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def environment(settings: Settings, registry: ModelRegistry, embedding_keys: list[str]) -> dict:
    import sklearn
    import torch
    import transformers

    hw = hardware_summary(settings.device)
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "transformers": transformers.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
            "device": hw["device"], "cuda": hw["cuda_available"], "gpus": hw["gpus"], "cpu_count": hw["cpu_count"],
            "asr_model": f"openai-whisper {settings.asr_model}",
            "embedding_models": {k: registry.model_id(SPEC_BY_KEY[k]) for k in embedding_keys}}


def _fit(model_cfg: dict, groups: list[str], seed: int, train: FeatureTable, selection: FeatureTable | None):
    """Fit pipeline + model on ``train``; ``selection`` is used only for early stopping / α selection."""
    pipe = InputPipeline(groups).fit(train.groups, train.names)
    model = build_model(model_cfg, groups, seed)
    Xtr = pipe.transform(train.groups, train.names)
    has_sel = selection is not None and len(selection.y) > 0
    Xs = pipe.transform(selection.groups, selection.names) if has_sel else None
    info = model.fit(Xtr, train.y, Xs, selection.y if has_sel else None)
    return pipe, model, info


def _predict(pipe, model, table: FeatureTable) -> np.ndarray:
    return model.predict(pipe.transform(table.groups, table.names))


def run(config_path: Path, *, evaluate_test: bool = False, validate_only: bool = False, synthetic_test_run: bool = False,
        settings: Settings | None = None, artifacts: Path = ARTIFACTS) -> Path:
    settings = settings or get_settings()
    cfg = load_yaml(config_path)
    if cfg.get("status") == "NOT_AVAILABLE":
        raise TrainingBlocked(f"{config_path} is a comparison placeholder (status NOT_AVAILABLE); it cannot be trained.")
    seed = int(cfg.get("seed", 13))
    set_seed(seed)
    ds_cfg_path = Path(cfg["dataset_config"])
    ds_cfg_path = ds_cfg_path if ds_cfg_path.is_absolute() else (BACKEND_DIR / ds_cfg_path)
    ds_cfg = DatasetConfig.from_dict(load_yaml(ds_cfg_path))
    if cfg.get("dataset_root"):
        ds_cfg.root = cfg["dataset_root"]
    # one source of truth: the experiment's audio_scope decides both trainability and how audio is cut
    cfg.setdefault("features", {}).setdefault("audio_scope", ds_cfg.audio_scope)
    ds_cfg.audio_scope = cfg["features"]["audio_scope"]

    # 2. validate dataset
    records, report = AphasiaDataset(ds_cfg, protected_dirs=[settings.data_dir, settings.temp_dir]).load()
    if report.data_origin == SYNTHETIC and not synthetic_test_run:
        raise TrainingBlocked("Dataset is SYNTHETIC_TEST_DATA. Refusing to train without --synthetic-test-run.")
    if report.data_origin == REAL and synthetic_test_run:
        raise TrainingBlocked("--synthetic-test-run was given but the dataset is not marked SYNTHETIC_TEST_DATA.")
    synthetic = report.data_origin == SYNTHETIC
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    exp_id = f"{cfg['experiment']}_{stamp}"
    base = artifacts / "synthetic_test" if synthetic else artifacts
    exp_dir = base / ("" if synthetic else "experiments") / exp_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_validation(records, report, exp_dir / "dataset")
    print(report.summary())
    if validate_only:
        return exp_dir
    try:
        return _train(cfg, config_path, settings, records, report, exp_dir, exp_id, base, seed, synthetic, evaluate_test, stamp)
    except TrainingBlocked:
        raise
    except Exception as exc:  # failed experiments are recorded, not silently lost
        (exp_dir / "FAILED.json").write_text(json.dumps({
            "experiment": exp_id, "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(), "config": cfg, "failed_at": datetime.now(timezone.utc).isoformat()}, indent=2))
        raise


def _train(cfg, config_path, settings, records, report, exp_dir, exp_id, base, seed, synthetic, evaluate_test, stamp) -> Path:
    min_p = int(cfg.get("min_participants", 3 if synthetic else MIN_REAL_PARTICIPANTS))
    if report.trainable_participants < min_p:
        raise TrainingBlocked(f"Only {report.trainable_participants} trainable participants (minimum {min_p}). "
                              f"See {exp_dir / 'dataset' / 'validation_summary.txt'}.")

    # 3. participant-level split, shared across experiments for the same dataset version + seed
    sp = cfg.get("split", {})
    split_seed = int(sp.get("seed", seed))
    params = {"val_fraction": sp.get("val_fraction", 0.15), "test_fraction": sp.get("test_fraction", 0.15),
              "stratify": sp.get("stratify", True)}
    cv_folds = int(sp.get("cv_folds", 0))
    split_dir = base / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"{report.dataset_version}_seed{split_seed}.json"
    trainable_ids = {r.participant_id for r in records if r.trainable}
    if split_path.exists():
        split = Split.from_dict(json.loads(split_path.read_text()))
        problems = check_reusable(split, trainable_ids, params)
        if problems:
            raise TrainingBlocked(
                f"The saved split {split_path} does not match the current dataset/config: " + "; ".join(problems)
                + ". Models must be compared on one split: set a new dataset version (or seed) to create a new split, "
                  "or restore the dataset/config that the split was created with.")
    else:
        split = split_participants(records, params["val_fraction"], params["test_fraction"], split_seed,
                                   params["stratify"], 0)
        save_split(split, split_path)
    if cv_folds and len(split.folds) != cv_folds:  # folds are deterministic given the split, bands and seed
        bands = participant_bands([r for r in records if r.trainable])
        split.folds = group_kfold(split.train + split.validation, bands, cv_folds, split_seed)
        save_split(split, split_path)
    assert_no_leakage(split.assignments())
    for fold in split.folds:
        assert_no_leakage({"train": fold["train"], "validation": fold["validation"], "test": split.test})
    save_split(split, exp_dir / "split.json")
    print(split_summary(split, records))

    # 4. features (cached per record and per feature group; shared across experiments)
    fcfg = FeatureConfig(**cfg["features"])
    registry = ModelRegistry(settings)
    extractor = FeatureExtractor(settings, fcfg, base / "features", report.dataset_version, registry)
    table = build_table(records, extractor)
    tr, va, te = table.subset(split.train), table.subset(split.validation), table.subset(split.test)
    if not len(tr.y) or not len(va.y):
        raise TrainingBlocked("Training or validation split has no usable recordings after feature extraction "
                              f"({len(table.failures)} extraction failure(s); see feature_failures.json).")
    groups = list(fcfg.groups)
    (exp_dir / "feature_failures.json").write_text(json.dumps(table.failures, indent=2))

    # 5–6. train (selection on validation) + validation metrics
    pipe, model, info = _fit(cfg["model"], groups, seed, tr, va)
    val_pred = _predict(pipe, model, va)
    metrics = {"validation": ev.evaluate_split(va.participant_ids, va.y, val_pred),
               "train": ev.evaluate_split(tr.participant_ids, tr.y, _predict(pipe, model, tr)),
               "note": "Validation was used for model selection (early stopping / α), so validation metrics are "
                       "optimistic. Cross-validation uses an inner selection split; the test split is untouched "
                       "unless --evaluate-test was given."}
    uncertainty = ev.residual_interval(va.y, val_pred)

    if split.folds:
        fold_rows = []
        full = table.subset(split.train + split.validation)
        for i, fold in enumerate(split.folds):
            fit_ids, sel_ids = inner_split(fold["train"], INNER_SELECTION_FRACTION, split_seed + 100 + i)
            fp, fm, _ = _fit(cfg["model"], groups, seed + i, full.subset(fit_ids), full.subset(sel_ids) if sel_ids else None)
            fva = full.subset(fold["validation"])  # scored fold: never used for fitting or selection
            fold_rows.append(ev.evaluate_split(fva.participant_ids, fva.y, _predict(fp, fm, fva)))
        maes = [f["participant_level"]["mae"] for f in fold_rows]
        metrics["cross_validation"] = {"folds": len(fold_rows), "participant_mae_mean": float(np.mean(maes)),
                                       "participant_mae_sd": float(np.std(maes)),
                                       "selection": f"inner participant-level split ({INNER_SELECTION_FRACTION:.0%} of each "
                                                    "fold's training participants); scored fold unseen",
                                       "per_fold": fold_rows}

    # ASR vs human transcript (only where both exist): measured, not assumed
    wer = ev.transcript_agreement(table.human_transcripts, table.whisper_transcripts)
    if wer:
        metrics["asr_vs_human_transcript"] = wer

    # 7. held-out test (explicit, logged)
    if evaluate_test:
        metrics["test"] = ev.evaluate_split(te.participant_ids, te.y, _predict(pipe, model, te))
        with open(base / "test_access.log", "a", encoding="utf-8") as fh:
            fh.write(f"{datetime.now(timezone.utc).isoformat()}\t{exp_id}\t{report.dataset_version}\t{split_path.name}\n")

    # 8–11. save
    status = STATUS_SYNTHETIC if synthetic else (STATUS_TESTED if evaluate_test else STATUS_VALIDATION_ONLY)
    raw_train = pipe.raw(tr.groups, tr.names)
    background = {g: (raw_train[g].mean(axis=0, keepdims=True) if g.startswith("embedding:") else raw_train[g][:100])
                  for g in groups}
    emb_models = {k: registry.model_id(SPEC_BY_KEY[k]) for k in fcfg.embedding_keys()}
    n_params, n_train_p = model.n_params(), len(split.train)
    complexity = {"n_parameters": n_params, "training_participants": n_train_p, "training_recordings": int(len(tr.y)),
                  "parameters_per_training_participant": round(n_params / max(1, n_train_p), 1)}
    bundle_dir = exp_dir / "bundle"
    save_bundle(bundle_dir, model=model, pipeline=pipe, feature_config=cfg["features"], data_origin=report.data_origin,
                status=status, experiment_id=exp_id, embedding_models=emb_models, background=background,
                extraction_spec=extractor.spec, input_names=tr.names,
                extra={"uncertainty": uncertainty, "complexity": complexity, "dataset_version": report.dataset_version})
    env = environment(settings, registry, fcfg.embedding_keys())
    counts = {k: {"participants": len(v), "recordings": len(records_for(records, v))}
              for k, v in split.assignments().items()}
    dims = {g: len(pipe.names[g]) for g in groups}
    card = build_card(
        status=status, data_origin=report.data_origin,
        experiment={"id": exp_id, "config": str(config_path), "seed": seed, "created": stamp},
        dataset_report=json.loads((exp_dir / "dataset" / "validation_report.json").read_text()),
        split=split.to_dict(), split_counts=counts, feature_config=cfg["features"],
        input_summary={"dims_after_pipeline": dims, "dropped_columns": pipe.dropped,
                       "imputed_values_train": pipe.imputed_counts, "missing_indicator_columns": pipe.indicators,
                       "imputation": "per-column median fitted on the training split, plus a binary <name>__missing "
                                     "indicator for every column that had missing values in training",
                       "extraction_spec": extractor.spec, "feature_extraction_failures": len(table.failures)},
        architecture=model.architecture(dims), hyperparameters=model.hp,
        training={k: v for k, v in info.items() if k != "history"} | {
            "optimizer": "closed form" if cfg["model"]["type"] == "ridge" else "AdamW (gradient-norm clipping)",
            "test_evaluated": evaluate_test, "complexity": complexity},
        metrics=metrics, environment=env, embedding_models=emb_models,
        preprocessing={"sample_rate": settings.target_sample_rate, "audio_scope": fcfg.audio_scope,
                       "vad": "webrtcvad", "normalisation": f"peak {settings.normalize_peak_dbfs} dBFS",
                       "noise_reduction": "not applied"})
    for d in (exp_dir, bundle_dir):
        write_card(d, card)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (exp_dir / "history.json").write_text(json.dumps(info.get("history", []), indent=1))
    (exp_dir / "experiment_config.json").write_text(json.dumps(cfg, indent=2))
    shutil.copy(config_path, exp_dir / Path(config_path).name)
    (exp_dir / "environment.json").write_text(json.dumps(env, indent=2))
    (exp_dir / "feature_cache.json").write_text(json.dumps(extractor.stats, indent=1))
    v = metrics["validation"]["participant_level"]
    print(f"\n{'SYNTHETIC_TEST_DATA - metrics are meaningless. ' if synthetic else ''}"
          f"VALIDATION (participant level, n={v['n']}): MAE {v['mae']:.2f}, RMSE {v['rmse']:.2f} | "
          f"parameters {n_params} ({complexity['parameters_per_training_participant']} per training participant)")
    print(f"Saved experiment: {exp_dir}\nStatus: {status}")
    return exp_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--evaluate-test", action="store_true", help="evaluate the held-out test split (do this once)")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--synthetic-test-run", action="store_true", help="required for SYNTHETIC_TEST_DATA datasets")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        run(args.config, evaluate_test=args.evaluate_test, validate_only=args.validate_only,
            synthetic_test_run=args.synthetic_test_run)
    except (DatasetNotAvailable, TrainingBlocked) as exc:
        print(f"\nTRAINING BLOCKED: {exc}\nProduction state remains MODEL_NOT_READY.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
