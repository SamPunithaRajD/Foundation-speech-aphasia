"""Severity prediction gate (WAB-R-derived severity score).

A prediction is only produced when FSA_SEVERITY_MODEL_DIR contains a bundle that
  * uses the ``foundationspeech-severity-bundle/1`` format,
  * was trained on REAL data (synthetic bundles are rejected by ``load_bundle``),
  * was evaluated on held-out test participants (status EVALUATED_ON_HELD_OUT_TEST),
  * was promoted by a named reviewer (promotion.json, written by ``python -m app.training.promote``), and
  * was trained with the SAME feature-extraction configuration that is running now (ASR model, encoder IDs,
    preprocessing/VAD settings, feature code version). A mismatch keeps the state MODEL_NOT_READY.
Otherwise the API returns MODEL_NOT_READY with ``prediction_available: false``.

Even with a valid model, no score is reported when
  * a required input group cannot be computed            -> INSUFFICIENT_DATA
  * the inputs are far outside the training distribution -> OUT_OF_DISTRIBUTION (raw output kept)
  * the raw output is outside the target range           -> INVALID_OUTPUT (raw output kept)
An invalid output is never clipped into a valid-looking score.

Recordings analysed through the application (uploads, microphone) are never part of the labelled
evaluation data, so every prediction carries ``clinical_validity = NOT_VALID_FOR_CLINICAL_INFERENCE``.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from app.schemas.analysis import SeverityResult
from app.schemas.common import SectionState
from app.training import target as tgt
from app.training.bundle import STATUS_PILOT, STATUS_TESTED, BundleRejected, SeverityBundle, load_bundle
from app.training.model_card import empty_card

log = logging.getLogger(__name__)

NOT_READY_MESSAGE = "A trained severity model is not currently available in this research prototype."
NOT_VALID = "NOT_VALID_FOR_CLINICAL_INFERENCE"
NOT_VALID_REASON = ("This recording is not part of the labelled research dataset the model was evaluated on. "
                    "The score is a research model output, not a clinical assessment.")
LIMITATIONS = [
    "Research model output; not clinically validated.",
    "The target is a transformation of WAB-R AQ (100 − AQ), not a newly validated clinical severity scale.",
    "The model was trained on time-aligned participant speech from the research dataset; this recording is analysed "
    "as a whole file under different recording conditions.",
    "The uncertainty interval is derived from validation residuals and applied uniformly; it is not a per-recording "
    "probability.",
]


class SeverityService:
    def __init__(self, model_dir: Path, settings=None, registry=None, ood_factor: float = 1.5):
        self.model_dir = Path(model_dir)
        self.settings, self.registry = settings, registry
        self.ood_factor = getattr(settings, "ood_factor", ood_factor) if settings is not None else ood_factor
        self.range_margin_sd = getattr(settings, "range_margin_sd", 0.5) if settings is not None else 0.5
        self._lock = threading.Lock()
        self._bundle: SeverityBundle | None = None
        self._reason: str | None = None
        self._loaded = False

    def _load(self) -> None:
        d = self.model_dir
        if not (d / "bundle.json").exists():
            self._reason = NOT_READY_MESSAGE
            if (d / "model.joblib").exists():
                self._reason += " (A legacy classification bundle was found and is not supported.)"
            return
        try:
            bundle = load_bundle(d)
            status = bundle.meta.get("status")
            if status not in (STATUS_TESTED, STATUS_PILOT):
                raise BundleRejected(f"Bundle status {status} is not {STATUS_TESTED}.")
            if not (d / "promotion.json").exists():
                raise BundleRejected("Bundle was not promoted (missing promotion.json).")
            if status == STATUS_PILOT:
                if not getattr(self.settings, "allow_pilot_model", False):
                    raise BundleRejected("A pilot (cross-validated only) model is present, but pilot models are not "
                                         "enabled (FSA_ALLOW_PILOT_MODEL is false).")
                if not json.loads((d / "promotion.json").read_text()).get("pilot"):
                    raise BundleRejected("The pilot bundle was not promoted with --pilot.")
            for key in ("extraction_spec", "input_dims", "ood_reference"):
                if key not in bundle.meta:
                    raise BundleRejected(f"Bundle has no '{key}' (created by an older, incompatible training version).")
            if self.settings is not None and self.registry is not None:
                from app.training.features import extraction_spec, spec_differences

                runtime = extraction_spec(self.settings, self.registry, bundle.groups,
                                          bundle.meta["feature_config"].get("language_source", "whisper"))
                diffs = spec_differences(bundle.meta["extraction_spec"], runtime)
                if diffs:
                    raise BundleRejected("The running feature-extraction configuration differs from the one the model "
                                         "was trained with (" + "; ".join(diffs)[:600] + ").")
            self._bundle, self._reason = bundle, None
        except BundleRejected as exc:
            self._reason = f"{NOT_READY_MESSAGE} {exc}"
        except Exception as exc:
            log.warning("severity bundle failed to load: %s", type(exc).__name__)
            self._reason = f"{NOT_READY_MESSAGE} The model bundle could not be loaded ({type(exc).__name__})."

    def bundle(self) -> SeverityBundle | None:
        with self._lock:
            if not self._loaded:
                self._load()
                self._loaded = True
            return self._bundle

    def reload(self) -> None:
        with self._lock:
            self._loaded, self._bundle = False, None

    def status(self) -> dict[str, Any]:
        b = self.bundle()
        if b is None:
            return {"status": "not_ready", "state": SectionState.MODEL_NOT_READY.value, "prediction_available": False,
                    "reason": self._reason, "maturity": "ready_for_data", "target_definition": tgt.target_definition(),
                    "model_card": empty_card()}
        return {"status": "ready", "state": "READY", "prediction_available": True, "model_version": b.version,
                "model_kind": b.meta["model_kind"], "feature_groups": b.groups, "target_definition": b.target,
                "evaluation_status": b.meta["status"], "clinical_validation": "NOT_VALIDATED", "maturity": "prototype",
                "model_stage": b.meta.get("model_stage"), "evaluation_summary": b.meta.get("evaluation_summary"),
                "complexity": b.meta.get("complexity"), "uncertainty": b.meta.get("uncertainty"),
                "explainability_background": True, "uploaded_recordings": NOT_VALID,
                "limitations": b.meta.get("limitations") or LIMITATIONS}

    def not_ready(self) -> SeverityResult:
        return SeverityResult(state=SectionState.MODEL_NOT_READY, prediction_available=False, message=self._reason)

    # ------------------------------------------------------------------ prediction
    def _assemble(self, b: SeverityBundle, inputs: dict[str, np.ndarray]) -> tuple[dict, dict]:
        """Arrange runtime feature vectors into exactly the columns (and order) the model was trained with."""
        from app.training.features import group_names

        groups, names = {}, {}
        for g in b.groups:
            vec = np.asarray(inputs[g], dtype=np.float64).ravel()
            want = b.input_names(g)
            if g.startswith("embedding:"):
                if len(vec) != len(want):
                    raise ValueError(f"'{g}' produced {len(vec)} dimensions; the model was trained with {len(want)}")
                groups[g], names[g] = vec[None, :], want
                continue
            have = group_names(g)
            if len(vec) != len(have):
                raise ValueError(f"'{g}' produced {len(vec)} values; expected {len(have)}")
            missing = [n for n in want if n not in have]
            if missing:
                raise ValueError(f"'{g}' features required by the model are not produced by this version: {missing[:5]}")
            groups[g], names[g] = vec[[have.index(n) for n in want]][None, :], want
        return groups, names

    def predict(self, inputs: dict[str, np.ndarray | None] | None) -> tuple[SeverityResult, dict | None]:
        """inputs: feature group -> full runtime feature vector (NaN = not computable).
        Returns the result and the imputed raw model inputs (for explanations; None unless a valid score exists)."""
        b = self.bundle()
        if b is None:
            return self.not_ready(), None
        meta = {"model_version": b.version, "model": b.meta["model_kind"], "feature_groups": b.groups,
                "status": b.meta["status"], "clinical_validation": "NOT_VALIDATED"}
        common = dict(classifier=meta, model_version=b.version, target_definition=b.target["name"],
                      target=b.target, limitations=b.meta.get("limitations") or LIMITATIONS,
                      model_stage=b.meta.get("model_stage"), evaluation_summary=b.meta.get("evaluation_summary"))
        missing = [g for g in b.groups if inputs is None or inputs.get(g) is None]
        if missing:
            return SeverityResult(state=SectionState.INSUFFICIENT_DATA, prediction_available=False, **common,
                                  message=f"Required model inputs could not be computed: {', '.join(missing)}. "
                                          "No score is reported."), None
        min_words = b.meta.get("min_words")
        if min_words and "language" in b.groups:
            from app.training.features import LANGUAGE_FEATURES

            words = np.asarray(inputs["language"], dtype=np.float64).ravel()[LANGUAGE_FEATURES.index("word_count")]
            if not np.isfinite(words) or words < min_words:
                return SeverityResult(
                    state=SectionState.INSUFFICIENT_DATA, prediction_available=False, **common,
                    message=f"The transcript has {0 if not np.isfinite(words) else int(words)} words; this model needs at "
                            f"least {min_words} transcribed words. No score is reported."), None
        try:
            groups, names = self._assemble(b, inputs)
            report = b.pipeline.missing_report(groups, names)
            raw_inputs = b.pipeline.raw(groups, names)
            ood = {g: float(v[0]) for g, v in b.pipeline.ood_scores(raw_inputs).items()}
            beyond = b.pipeline.out_of_range(raw_inputs, self.range_margin_sd) if self.ood_factor else {}
            raw = float(b.model.predict(b.pipeline.standardize(raw_inputs))[0])
        except Exception as exc:
            log.warning("severity inference failed: %s", type(exc).__name__)
            return SeverityResult(state=SectionState.INFERENCE_FAILED, prediction_available=False, **common,
                                  message=f"Severity inference failed ({type(exc).__name__}: {str(exc)[:200]}). "
                                          "No score is reported."), None

        ref = b.pipeline.ood_reference
        limits = {g: self.ood_factor * ref[g]["train_max"] for g in ood} if self.ood_factor else {}
        ood_groups = [g for g in ood if limits and ood[g] > limits[g]]
        unprecedented = {g: r["never_missing_in_training"] for g, r in report.items() if r["never_missing_in_training"]}
        checks = {"ood_score": {g: round(v, 3) for g, v in ood.items()},
                  "ood_limit": {g: round(v, 3) for g, v in limits.items()},
                  "ood_definition": "mean absolute z-score of the inputs vs. training statistics; limit = "
                                    f"{self.ood_factor} × the largest value seen in training",
                  "out_of_distribution_groups": ood_groups,
                  "features_outside_training_range": beyond,
                  "range_rule": f"a named feature may exceed its training range by at most {self.range_margin_sd} training SD",
                  "imputed_inputs": {g: r["seen_in_training"] + r["never_missing_in_training"] for g, r in report.items()},
                  "imputed_but_never_missing_in_training": unprecedented}
        raw_r = round(raw, 3) if np.isfinite(raw) else None

        if beyond:
            listed = "; ".join(f"{n} = {d['value']} (training range {d['training_min']} to {d['training_max']})"
                               for feats in beyond.values() for n, d in feats.items())
            return SeverityResult(
                state=SectionState.OUT_OF_DISTRIBUTION, prediction_available=False, raw_output=raw_r,
                input_checks=checks, **common,
                message="This recording is outside the range of the data the model was fitted on (" + listed
                        + "). The model would be extrapolating, so no score is reported."), None
        if ood_groups:
            return SeverityResult(
                state=SectionState.OUT_OF_DISTRIBUTION, prediction_available=False, raw_output=raw_r,
                input_checks=checks, **common,
                message="This recording's features are far outside the range the model was trained on ("
                        + ", ".join(f"{g}: {ood[g]:.2f} > {limits[g]:.2f}" for g in ood_groups)
                        + "). The model output is not a meaningful estimate here, so no score is reported."), None
        p = tgt.finalize_prediction(raw, b.meta.get("output_tolerance", tgt.OUTPUT_TOLERANCE))
        if not p.valid:
            return SeverityResult(state=SectionState.INVALID_OUTPUT, prediction_available=False, raw_output=raw_r,
                                  input_checks=checks, **common, message=f"{p.reason} No score is reported."), None

        unc = b.meta.get("uncertainty") or {}
        hw = unc.get("half_width")
        notes = []
        if p.clipped:
            notes.append(f"The raw output ({p.raw_severity:.1f}) slightly exceeded the 0–100 range and was clipped.")
        if unprecedented:
            notes.append("Some inputs could not be computed and were replaced by training medians although they were "
                         "never missing in training: " + ", ".join(n for v in unprecedented.values() for n in v) + ".")
        return SeverityResult(
            state=SectionState.COMPUTED, prediction_available=True, severity_score=p.severity_score,
            predicted_aq=p.predicted_aq, severity_band=p.band, raw_output=raw_r, clipped=p.clipped,
            uncertainty=({"interval": [round(max(0.0, p.severity_score - hw), 1), round(min(100.0, p.severity_score + hw), 1)],
                          "half_width": round(hw, 1), "coverage": unc.get("coverage"), "method": unc.get("method"),
                          "calibrated_probability": False} if hw is not None else None),
            input_checks=checks, clinical_validity=NOT_VALID, clinical_validity_reason=NOT_VALID_REASON, **common,
            message=" ".join([("PILOT model output" if b.meta.get("model_stage") == "PILOT" else "Research model output")
                              + f" ({b.target['name']}). Not a clinical assessment."] + notes)
        ), raw_inputs
