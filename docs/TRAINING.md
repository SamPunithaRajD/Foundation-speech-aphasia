# Training

| Item | Status |
|---|---|
| Training command, models A/B/C, caching, bundles, model card, promotion | **IMPLEMENTED** (tested on SYNTHETIC_TEST_DATA only) |
| Real training | **BLOCKED** (no labelled dataset) |
| Production model | **MODEL_NOT_READY** |

## Command (run from `backend/`, venv active)

```
# 1. validate the dataset only
python -m app.training.train --config configs/experiments/model_c_multimodal.yaml --validate-only
# 2. develop / select models on validation + CV (test untouched)
python -m app.training.train --config configs/experiments/model_a_handcrafted.yaml
python -m app.training.train --config configs/experiments/model_b_wav2vec2.yaml
python -m app.training.train --config configs/experiments/model_c_multimodal.yaml
python -m app.training.compare artifacts/experiments/*
# 3. evaluate the SELECTED configuration once on the held-out test participants
python -m app.training.train --config configs/experiments/<selected>.yaml --evaluate-test
# 4. human review, then promote
python -m app.training.promote artifacts/experiments/<experiment_id> --reviewer "<name>"
```

`train` runs these steps in order:
1. load config;
2. validate the dataset;
3. participant-level split (shared per dataset version and seed);
4. extract and cache features;
5. train, with early stopping on validation only;
6. evaluate validation, plus optional participant-level CV;
7. `--evaluate-test`: evaluate the held-out test set and append to `artifacts/test_access.log`;
8. save the bundle;
9. save metrics;
10. save the experiment config and environment;
11. write the model card.

## Feature groups
| Group | Source | Notes |
|---|---|---|
| `acoustic` | `app/features/acoustic.py` | 38 handcrafted features: MFCC 1–13 mean/std, energy, ZCR, spectral, pYIN F0; uncomputable = NaN + reason |
| `language` | `app/features/clinical.py` | 15 clinical language features; `syntactic_complexity` excluded (no parser). `language_source: whisper` (deployment-consistent) or `human` (CHAT) |
| `embedding:<key>` | `app/models/registry.py` | wav2vec2 / hubert / wavlm / whisper_encoder, mean-pooled final layer, 30 s chunks |

- **Missing values:** NaN in the cache. Inside the model pipeline they are imputed with the training-split median; columns entirely missing in training are dropped. Imputation counts are recorded in the model card.

## Models (`app/training/models.py`)
| Config | Model | Architecture |
|---|---|---|
| `model_a_handcrafted.yaml` | **A — handcrafted baseline** | standardised acoustic + language features → ridge regression (α=1) |
| `model_b_wav2vec2.yaml` | **B — speech foundation model** | wav2vec 2.0 embedding (768) → MLP [128, GELU, dropout 0.2] → 1 |
| `model_c_multimodal.yaml` | **C — proposed multimodal** | three tokens (wav2vec2, language, acoustic), each Linear→64 + LayerNorm + modality embedding; [CLS]; 2-layer pre-norm TransformerEncoder (4 heads, ff 128, dropout 0.2); LayerNorm→Linear on [CLS] |
| `ablation_a_language_only.yaml` | A′ | language features only → ridge |
| `ablation_d_{hubert,wavlm,whisper_encoder}.yaml` | D | Model C with another foundation model |
| `base_paper_comparison.yaml` | — | **NOT_AVAILABLE**: framework only, results empty |

- **Torch models:** AdamW, Huber loss (δ = 0.1 on severity/100), batch 16, lr 1e-3 (B) / 5e-4 (C), weight decay 0.01, up to 300 epochs, early stopping on validation MAE (patience 30), seed 13.
- **Hyperparameters are defaults, not tuned values.** Tune on validation / CV only.
- **No superiority claim:** no model is claimed to be better than another until the comparison has been run.

## Artifacts

```
artifacts/
    experiments/<id>/     bundle/, metrics.json, split.json, dataset/, MODEL_CARD.md, environment.json, ...
    splits/               shared participant splits
    features/<dataset_version>/<config_hash>/   cached features
    test_access.log       every held-out test evaluation
    synthetic_test/       ALL synthetic runs (separate tree; never promotable)
```

`FSA_SEVERITY_MODEL_DIR` (default `artifacts/severity`) is the production location. It is written only by `promote`.

## Safeguards
- **Synthetic data stays separate.** A dataset root containing the `SYNTHETIC_TEST_DATA` marker requires `--synthetic-test-run`. Its bundles are stamped `SYNTHETIC_TEST_ONLY`, written under `artifacts/synthetic_test/`, rejected by `load_bundle`, refused by `promote`, and never served.
- **No relabelling:** `--synthetic-test-run` on unmarked data is refused.
- **Minimum size:** real runs require at least 20 trainable participants (`min_participants`).
- **Production loads only promoted bundles:** `format` ok, `data_origin: REAL`, `status: EVALUATED_ON_HELD_OUT_TEST`, and `promotion.json` present.
- **No pickle:** bundles store JSON / NPZ, and torch weights are loaded with `weights_only=True`.
- **No speaker embeddings in bundles:** only the training-mean embedding is stored, never individual participants' embeddings.

## Reproducibility
Each experiment records:
- **Environment:** Python, torch, transformers, sklearn and numpy versions; device and GPU list; ASR model and foundation-model IDs.
- **Data:** the dataset version or file fingerprint, and the participant IDs of every split.
- **Run settings:** seed, full config, and a feature-config hash.

## Changes from the 2026-10 audit
| Area | Before | Now | Why |
|---|---|---|---|
| Feature cache | one cache per experiment config | per record **and per unit** (preprocessing, ASR, acoustic, language, each embedding) | Models A/B/C no longer recompute identical ASR, acoustic features and embeddings; Whisper is not run for embedding-only models |
| Missing values | median imputation only | median imputation **plus a `<name>__missing` indicator** for columns that were missing in training | imputed values were indistinguishable from real ones |
| Ridge α | fixed (1.0) | selected from a grid on the validation split | an untuned baseline makes B/C look better than they are |
| Cross-validation | early stopping on the scored fold | **inner participant-level selection split**; the scored fold is never seen | torch models got an optimistic CV score that ridge could not get |
| Model C default | d_model 64, 2 layers (~120k parameters) | d_model 32, 1 layer, dropout 0.3, input dropout (~35k) | only 4 tokens and a few hundred participants at best; a deeper encoder is not justified before validation shows it helps |
| Model B default | hidden [128] | hidden [64], dropout 0.3 | same reasoning |
| Training stability | none | gradient-norm clipping (1.0), non-finite loss aborts the run | Transformer training on small batches |
| Complexity | not recorded | parameter count and parameters per training participant in the bundle, model card and comparison table | makes over-parameterisation visible |
| Fairness baselines | A, B, C only | + `ablation_b_wav2vec2_ridge` (linear probe), `ablation_c_concat_mlp` (same inputs, no Transformer), `ablation_c_ridge_all` | isolates what the MLP head and the Transformer fusion actually contribute |
| Failed runs | no trace | `FAILED.json` with error, traceback and config | failed experiments are recorded |
| Bundle | weights + pipeline | + `extraction_spec`, exact `input_names`, `input_dims`, `ood_reference`, `n_parameters`, `output_tolerance` | lets inference verify it runs under the configuration it was trained with |
| Optional | — | `features.exclude_features` (e.g. `word_count`) | lets length-confounded features be tested as an ablation |

**Known research risks (not solved by code):**
- **Session length confound.** `word_count`, type-token ratio and Brunét's index depend on how long a participant spoke. If sessions differ in length by severity or by protocol, a model can learn length instead of language. Run an ablation with `exclude_features: [word_count, type_token_ratio, brunet_index]`.
- **Training vs. deployment audio.** Training uses time-aligned participant speech; uploads are whole files.
- **Repetition markers.** CHAT `[x N]` repetition counts are dropped by normalisation, so the repeated word is counted once.
