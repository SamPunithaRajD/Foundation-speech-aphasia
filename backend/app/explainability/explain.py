"""Model-attribution explanations (SHAP, LIME) for the severity model's target (named in each bundle).

Runs only when a real, promoted severity model produced a prediction (never for synthetic
bundles, which cannot be loaded for production). What is explained:

    the model's predicted severity score, over an INTERPRETABLE feature layer:
      * each named acoustic / clinical-language feature individually (raw values; missing values
        are shown after training-median imputation), and
      * each speech-embedding group as ONE grouped input ("speech_embedding[wav2vec2]"), switched
        between this recording's embedding and the training-mean embedding.
    Individual embedding dimensions are not explained (they have no human-readable meaning).

SHAP/LIME provide model-attribution explanations describing which inputs contributed to the
model prediction. They do not identify the medical cause of aphasia and are not clinical evidence.
"""
from __future__ import annotations

import logging

import numpy as np

from app.schemas.analysis import Attribution, ExplainabilityResult, ExplanationMethodResult, SeverityResult
from app.schemas.common import SectionState
from app.training.bundle import SeverityBundle

log = logging.getLogger(__name__)
TOP_K = 8
SPACE_NOTE = ("Explained on the interpretable feature layer: named features individually; each speech embedding as a "
              "single grouped input (this recording's embedding vs. the training-mean embedding).")


def _attributions(names: list[str], values: np.ndarray) -> list[Attribution]:
    order = np.argsort(-np.abs(values))[:TOP_K]
    return [Attribution(feature=names[i], contribution=round(float(values[i]), 4),
                        direction="positive" if values[i] >= 0 else "negative") for i in order]


LIME_MIN_LOCAL_R2 = 0.5  # below this the local linear surrogate explains the model poorly -> flagged unreliable
MIN_BACKGROUND_ROWS = 10


def shap_explanation(bundle: SeverityBundle, raw_inputs: dict[str, np.ndarray]) -> ExplanationMethodResult:
    import shap

    f = bundle.interpretable_predict_fn(raw_inputs)
    x = bundle.interpretable_row(raw_inputs)
    background = bundle.interpretable_background()
    explainer = shap.KernelExplainer(f, background)
    values = np.asarray(explainer.shap_values(x.reshape(1, -1), nsamples=500, silent=True))[0]
    base = float(np.ravel(explainer.expected_value)[0])
    fx = float(f(x)[0])
    additivity_error = abs(base + float(values.sum()) - fx)
    reliable = additivity_error <= max(0.5, 0.01 * abs(fx)) and len(background) >= MIN_BACKGROUND_ROWS
    return ExplanationMethodResult(
        method="shap", state=SectionState.COMPUTED, target=bundle.target["name"], base_value=round(base, 3),
        attributions=_attributions(bundle.interpretable_names(), values), reliable=reliable,
        quality={"additivity_error": round(additivity_error, 4), "explained_output": round(fx, 3),
                 "background_rows": int(len(background)), "samples": 500},
        message="KernelSHAP (model-agnostic) on the interpretable feature layer; attributions sum to (raw model output − "
                "background mean). Positive = contributed to a higher (more impaired) score."
                + ("" if reliable else " UNRELIABLE: additivity check failed or the background sample is very small."))


def lime_explanation(bundle: SeverityBundle, raw_inputs: dict[str, np.ndarray]) -> ExplanationMethodResult:
    from lime.lime_tabular import LimeTabularExplainer

    names = bundle.interpretable_names()
    background = bundle.interpretable_background()
    categorical = [i for i, n in enumerate(names) if n.startswith("speech_embedding[") or n.endswith("__missing")]
    emb_cols = [i for i, n in enumerate(names) if n.startswith("speech_embedding[")]
    # embedding columns: provide both states (0 = training mean, 1 = instance) so LIME can perturb them
    if emb_cols:
        flipped = background.copy()
        flipped[:, emb_cols] = 1.0
        background = np.vstack([background, flipped])
    # sample_around_instance: perturb around THIS recording (scaled by training std). The default samples from the
    # training distribution, which for out-of-distribution recordings gives ~0 kernel weights and a degenerate
    # all-zero surrogate.
    explainer = LimeTabularExplainer(background, feature_names=names, mode="regression", random_state=0,
                                     categorical_features=categorical, discretize_continuous=False,
                                     sample_around_instance=True)
    exp = explainer.explain_instance(bundle.interpretable_row(raw_inputs), bundle.interpretable_predict_fn(raw_inputs),
                                     num_features=min(TOP_K, len(names)), num_samples=1000)
    values = np.zeros(len(names))
    for i, w in exp.as_map()[1]:
        values[i] = w
    fit = float(exp.score)
    quality = {"local_r2": round(fit, 3), "min_reliable_local_r2": LIME_MIN_LOCAL_R2, "perturbations": 1000}
    if not np.any(np.abs(values) > 1e-9):
        return ExplanationMethodResult(method="lime", state=SectionState.INFERENCE_FAILED, target=bundle.target["name"],
                                       reliable=False, quality=quality,
                                       message=f"LIME surrogate was degenerate (all weights zero, local R² {fit:.2f}); "
                                               "no LIME explanation is reported.")
    reliable = fit >= LIME_MIN_LOCAL_R2
    return ExplanationMethodResult(
        method="lime", state=SectionState.COMPUTED, target=bundle.target["name"], reliable=reliable, quality=quality,
        attributions=_attributions(names, values)[:len(exp.as_map()[1])],
        message=f"LIME tabular local linear surrogate (1000 perturbations around this recording; local R² {fit:.2f}). "
                + ("" if reliable else f"UNRELIABLE: local R² is below {LIME_MIN_LOCAL_R2}, so the surrogate explains the "
                                       "model poorly near this recording; treat these weights with caution. ")
                + "Limitations: unstable across seeds and sensitive to the perturbation scale.")


def explain(severity: SeverityResult, bundle: SeverityBundle | None, raw_inputs: dict | None) -> ExplainabilityResult:
    if not severity.prediction_available or bundle is None or raw_inputs is None:
        return ExplainabilityResult(
            state=SectionState.NOT_AVAILABLE,
            message="Explainability will become available after a trained prediction model is integrated. "
            if severity.state == SectionState.MODEL_NOT_READY else
            f"No explanation is generated because no valid severity score was produced ({severity.state.value}). "
            + (severity.message or ""))
    link = dict(model_version=bundle.version, explained_raw_output=severity.raw_output, explained_score=severity.severity_score)
    # the explanation must belong to exactly the prediction that is reported
    fx = float(bundle.interpretable_predict_fn(raw_inputs)(bundle.interpretable_row(raw_inputs))[0])
    if severity.raw_output is None or abs(fx - severity.raw_output) > 1e-2 or severity.model_version != bundle.version:
        return ExplainabilityResult(state=SectionState.INFERENCE_FAILED, **link,
                                    message="Explanation inputs do not reproduce the reported prediction; no explanation "
                                            "is shown.")
    methods: list[ExplanationMethodResult] = []
    for fn, name in ((shap_explanation, "shap"), (lime_explanation, "lime")):
        try:
            methods.append(fn(bundle, raw_inputs))
        except Exception as exc:
            log.warning("%s explanation failed: %s", name, type(exc).__name__)
            methods.append(ExplanationMethodResult(method=name, state=SectionState.INFERENCE_FAILED, reliable=False,
                                                   message=f"{name.upper()} failed ({type(exc).__name__})."))
    ok = sum(m.state == SectionState.COMPUTED for m in methods)
    unreliable = [m.method.upper() for m in methods if m.state == SectionState.COMPUTED and m.reliable is False]
    return ExplainabilityResult(state=SectionState.COMPUTED if ok == len(methods) else
                                SectionState.PARTIAL if ok else SectionState.INFERENCE_FAILED,
                                methods=methods, **link,
                                message=SPACE_NOTE + (f" Flagged unreliable: {', '.join(unreliable)}." if unreliable else ""))
