# Evaluation

| Item | Status |
|---|---|
| Metric code, participant-level aggregation, bootstrap CI, band metrics, uncertainty | **IMPLEMENTED** |
| Results | **NOT_AVAILABLE** (no labelled data); nothing is reported |
| Clinical validation | **NOT_VALIDATED** (always; no software step establishes it) |

Code: `backend/app/training/evaluation.py`.

## Primary: regression on the WAB-R-derived severity score (0–100)
Reported for validation, CV folds and (once) test, at **recording level** and **participant level** (mean prediction per participant):

- MAE, RMSE, R²
- Pearson and Spearman correlation (with p-values; `None` with a reason if n < 3 or constant)
- MAE on the AQ scale (same magnitude, reported for readability)
- fraction of predictions clipped, raw output range, prediction and target distributions
- **MAE 95% CI** by participant-level bootstrap (2,000 resamples; requires ≥ 5 participants)

## Secondary: derived AQ bands
- **Bands:** the published WAB AQ bands are applied to both true and predicted AQ.
- **Metrics:** accuracy, macro-F1, weighted-F1, per-class precision / recall / F1 / support, and a 4×4 confusion matrix.
- **Labelled as derived:** these are derived from the regression target, **not** independent labels.

## Uncertainty
- **What is reported:** a split-conformal-style interval, the absolute validation-residual quantile at 90% coverage, applied uniformly around every prediction.
- **What it is not:** a per-sample uncertainty or a probability. It is labelled as such in the API, UI and PDF.
- **When it's missing:** it is `null` when there are fewer than 5 validation recordings.

## Protocol (against overfitting)
1. **Select on validation only:** choose models and hyperparameters on validation and participant-level CV. The test set is untouched.
2. **Test once:** run `--evaluate-test` once, for the selected configuration. Each access is appended to `artifacts/test_access.log`, so repeated test use is visible.
3. **Promote deliberately:** `promote` requires a test-evaluated, real-data bundle and a named reviewer.
4. **No manual edits:** no manual prediction adjustment, label creation from model outputs, or undocumented sample removal. Exclusions are listed in the validation report.

## Comparison table

```
python -m app.training.compare artifacts/experiments/*
```

| Experiment | Status |
|---|---|
| A handcrafted (acoustic + language, ridge) | READY_FOR_DATA |
| A′ language only | READY_FOR_DATA |
| B wav2vec2 → MLP | READY_FOR_DATA |
| C multimodal wav2vec2 + language + acoustic → Transformer | READY_FOR_DATA |
| D foundation-model ablation (HuBERT, WavLM, Whisper encoder) | READY_FOR_DATA |
| Base paper (Conv-Transformer + Jaya Gazelle) | NOT_AVAILABLE — requires a reimplementation on the same participants, split and target |

No result is filled in, and no model is claimed superior, until a controlled comparison on the same split has been run.

## Explainability (after a real model exists)
- **Methods:** SHAP (KernelExplainer) and LIME (tabular regression) explain the predicted WAB-R-derived severity score.
- **What is explained:** an **interpretable layer**. Each named acoustic or language feature is explained individually. Each speech embedding is one grouped input (this recording's embedding vs. the training-mean embedding); individual embedding dimensions are not explained.
- **LIME specifics:** it samples around the instance and reports its local R². A degenerate surrogate (all-zero weights) is reported as failed, not shown.
- **Never on synthetic models:** neither method runs for synthetic bundles, which can never be loaded.
- **Wording:** "these inputs contributed to the model prediction", never "caused aphasia".

## Changes from the 2026-10 audit
- **Unclipped metrics.** Headline MAE/RMSE use predictions clipped to 0–100. `mae_unclipped`, `rmse_unclipped` and `n_invalid_outputs` (outputs more than 10 points outside the range) are now reported alongside, so wild outputs cannot hide behind clipping.
- **Validation is optimistic.** Validation is used for early stopping and α selection. Compare models on **cross-validation** (which uses an inner selection split) and report the final model on **test**.
- **ASR vs. human transcript.** Where a recording has both, the word error rate of Whisper against the normalised CHAT transcript is reported (`asr_vs_human_transcript`). ASR quality is measured, not assumed.
- **Explanations are bound to the prediction.** Each explanation records the model version and the raw output it explains. If the explanation inputs don't reproduce the reported prediction, no explanation is shown.
- **Reliability flags.** SHAP reports its additivity error. LIME reports its local R² and is flagged **UNRELIABLE** below 0.5 (a heuristic).
- **No explanation without a valid score.** `INVALID_OUTPUT` and `OUT_OF_DISTRIBUTION` results get no SHAP/LIME.
