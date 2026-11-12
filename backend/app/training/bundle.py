"""Severity model bundle: save / load / predict / interpretable explanation space.

Bundle layout (one directory):
    bundle.json                format, model kind, feature groups, data origin, status
    input_pipeline.json/.npz   train-fitted imputation + standardisation
    weights.npz | weights.pt   model parameters (no pickle; torch loaded with weights_only=True)
    explain_background.npz     training-split background for SHAP/LIME (named features: up to 100 imputed
                               training rows; embedding groups: the training-mean vector only — individual
                               participants' embeddings are not stored because they can identify speakers)
    model_card.json / MODEL_CARD.md
    metrics.json, split.json, experiment_config.json, environment.json

Status values (``bundle.json`` → ``status``):
    SYNTHETIC_TEST_ONLY          trained on SYNTHETIC_TEST_DATA — can never be loaded for production
    TRAINED_VALIDATION_ONLY      real data, held-out test not yet evaluated
    EVALUATED_ON_HELD_OUT_TEST   real data, evaluated once on held-out participants
Clinical validation is always "NOT_VALIDATED" — no software step can establish it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.training import target as tgt
from app.training.dataset import REAL, SYNTHETIC
from app.training.models import InputPipeline, RidgeModel, TorchRegressor

BUNDLE_FORMAT = "foundationspeech-severity-bundle/1"
STATUS_SYNTHETIC = "SYNTHETIC_TEST_ONLY"
STATUS_VALIDATION_ONLY = "TRAINED_VALIDATION_ONLY"
STATUS_TESTED = "EVALUATED_ON_HELD_OUT_TEST"
# Small-sample pilot evaluated by participant-level cross-validation only. Served only when the operator explicitly
# enables FSA_ALLOW_PILOT_MODEL and promotes it with --pilot; always labelled as a pilot in every output.
STATUS_PILOT = "PILOT_CROSS_VALIDATED"
CLINICAL_VALIDATION = "NOT_VALIDATED"


class BundleRejected(RuntimeError):
    pass


def save_bundle(d: Path, *, model, pipeline: InputPipeline, feature_config: dict, data_origin: str, status: str,
                experiment_id: str, embedding_models: dict[str, str], background: dict[str, np.ndarray],
                extraction_spec: dict, input_names: dict[str, list[str]], extra: dict | None = None,
                target: dict | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    model.save(d)
    pipeline.save(d)
    np.savez(d / "explain_background.npz", **{f"raw::{g}": v for g, v in background.items()})
    (d / "bundle.json").write_text(json.dumps({
        "format": BUNDLE_FORMAT, "model_kind": model.kind if hasattr(model, "kind") else "ridge",
        "groups": pipeline.groups, "feature_config": feature_config, "embedding_models": embedding_models,
        "data_origin": data_origin, "status": status, "clinical_validation": CLINICAL_VALIDATION,
        "experiment_id": experiment_id, "target": target or tgt.target_definition(), "hyperparameters": model.hp,
        # --- training/inference compatibility contract ---
        "extraction_spec": extraction_spec,  # every setting that changes feature values
        "input_names": {g: (input_names[g] if not g.startswith("embedding:") else None) for g in pipeline.groups},
        "input_dims": pipeline.input_dims, "ood_reference": pipeline.ood_reference,
        "n_parameters": model.n_params(), "output_tolerance": tgt.OUTPUT_TOLERANCE, **(extra or {}),
    }, indent=2), encoding="utf-8")


@dataclass
class SeverityBundle:
    path: Path
    meta: dict
    model: object
    pipeline: InputPipeline
    background: dict[str, np.ndarray]
    card: dict

    @property
    def groups(self) -> list[str]:
        return self.meta["groups"]

    @property
    def version(self) -> str:
        return self.meta["experiment_id"]

    @property
    def target(self) -> dict:
        """The target this model was trained on (instrument, transformation, provenance)."""
        return self.meta.get("target") or tgt.target_definition()

    def input_names(self, group: str) -> list[str]:
        """Exact input column names (and order) the model was trained with, for one group."""
        from app.training.features import group_names

        stored = (self.meta.get("input_names") or {}).get(group)
        return stored or group_names(group, dims={group: self.pipeline.input_dims[group]})

    def embedding_keys(self) -> list[str]:
        return [g.split(":", 1)[1] for g in self.groups if g.startswith("embedding:")]

    def predict_raw(self, groups: dict[str, np.ndarray], names: dict[str, list[str]]) -> np.ndarray:
        """groups: unimputed matrices (NaN allowed) in the column order given by ``names``."""
        return self.model.predict(self.pipeline.transform(groups, names))

    # ---- interpretable explanation space (named features + one column per embedding group) ----
    def interpretable_names(self) -> list[str]:
        cols = []
        for g in self.groups:
            cols += ([f"speech_embedding[{g.split(':', 1)[1]}]"] if g.startswith("embedding:") else self.pipeline.names[g])
        return cols

    def interpretable_row(self, raw: dict[str, np.ndarray]) -> np.ndarray:
        parts = [np.ones(1) if g.startswith("embedding:") else raw[g][0] for g in self.groups]
        return np.concatenate(parts)

    def interpretable_background(self, max_rows: int = 50) -> np.ndarray:
        named = [g for g in self.groups if not g.startswith("embedding:")]
        n = min(len(self.background[named[0]]), max_rows) if named else 1
        parts = [np.zeros((n, 1)) if g.startswith("embedding:") else self.background[g][:n] for g in self.groups]
        return np.hstack(parts)

    def interpretable_predict_fn(self, instance_raw: dict[str, np.ndarray]):
        """f(Z) -> raw severity. Embedding columns are 1 = this sample's embedding, 0 = training-mean embedding."""
        sizes = [1 if g.startswith("embedding:") else len(self.pipeline.names[g]) for g in self.groups]

        def f(Z: np.ndarray) -> np.ndarray:
            Z = np.atleast_2d(Z)
            raw, col = {}, 0
            for g, size in zip(self.groups, sizes):
                block = Z[:, col:col + size]
                if g.startswith("embedding:"):
                    mean = self.background[g].mean(axis=0)
                    w = (block[:, :1] >= 0.5).astype(float)
                    raw[g] = w * instance_raw[g][0] + (1 - w) * mean
                else:
                    raw[g] = block
                col += size
            return self.model.predict(self.pipeline.standardize(raw))
        return f


def load_bundle(d: Path, allow_synthetic: bool = False) -> SeverityBundle:
    d = Path(d)
    meta = json.loads((d / "bundle.json").read_text(encoding="utf-8"))
    if meta.get("format") != BUNDLE_FORMAT:
        raise BundleRejected(f"Unsupported bundle format {meta.get('format')!r}.")
    if (meta.get("data_origin") != REAL or meta.get("status") == STATUS_SYNTHETIC) and not allow_synthetic:
        raise BundleRejected("Bundle was trained on SYNTHETIC_TEST_DATA and cannot be used for predictions.")
    if meta.get("data_origin") not in (REAL, SYNTHETIC):
        raise BundleRejected("Bundle has no recognised data origin.")
    kind = meta["model_kind"]
    if kind == "ridge":
        model = RidgeModel(meta["groups"], alpha=meta["hyperparameters"].get("alpha", 1.0))
    else:
        hp = {k: v for k, v in meta["hyperparameters"].items()}
        model = TorchRegressor(kind, meta["groups"], **hp)
    model.load(d)
    with np.load(d / "explain_background.npz", allow_pickle=False) as z:
        background = {k.split("::", 1)[1]: z[k] for k in z.files}
    card_path = d / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.exists() else {}
    return SeverityBundle(d, meta, model, InputPipeline.load(d), background, card)
