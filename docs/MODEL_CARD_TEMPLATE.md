# FoundationSpeech-Aphasia — Severity Model Card (template)

**STATUS: MODEL_NOT_READY**  ·  Clinical validation: **NOT_VALIDATED**

> No real labelled aphasia dataset has been trained on. This template shows the fields a card will contain.
> Cards are generated automatically by `backend/app/training/model_card.py` from values produced by an actual run
> (`MODEL_CARD.md` + `model_card.json` in each experiment and bundle). Fields without data read `NOT_AVAILABLE`.

## Status values
| Status | Meaning |
|---|---|
| `MODEL_NOT_READY` | no trained model (current production state) |
| `SYNTHETIC_TEST_ONLY` | trained on SYNTHETIC_TEST_DATA; software test only; can never be served |
| `TRAINED_VALIDATION_ONLY` | real data; held-out test not yet evaluated; cannot be promoted |
| `EVALUATED_ON_HELD_OUT_TEST` | real data; evaluated once on held-out participants; promotable after human review |

Clinical validation is always `NOT_VALIDATED`.

## Fields
| Section | Field | Filled from |
|---|---|---|
| Dataset | name, version | dataset config / file fingerprint |
| | participant count, recording count | validation report (trainable) |
| | exclusions | missing audio / AQ, invalid / ambiguous AQ, duplicates |
| | derived band distribution | validation report |
| Target | definition, source, transformation, bands | `target.py` (`severity_score = 100 − WAB-R AQ`) |
| Split | method, seed, counts per split, participant IDs | `split.json` |
| | participant leakage check | `passed` (training aborts otherwise) |
| Model | foundation model(s) and IDs | registry |
| | preprocessing | sample rate, audio scope, VAD, normalisation, noise reduction (not applied) |
| | language features | source (whisper/human), excluded features |
| | fusion method, architecture, input dimensions | model config |
| | hyperparameters, training (epochs run, early stopping, optimizer) | training run |
| Metrics | validation / CV / test | `evaluation.py`, participant level first |
| Environment | Python, torch, transformers, sklearn, device, GPU | runtime |
| Use | intended use / not intended use | fixed text (below) |
| Limitations | fixed + run-specific | fixed text (below) |

## Intended use
Research on automatic estimation of a WAB-R-derived severity score from speech recordings of participants in the
labelled research dataset the model was evaluated on.

## Not intended for
- **Clinical decisions:** diagnosis, screening, treatment planning, or any decision about an individual.
- **Other populations:** inference on people outside the evaluated dataset, including personal or microphone recordings uploaded to the application. These are reported as `NOT_VALID_FOR_CLINICAL_INFERENCE`.
- **Causal claims:** claims of medical causation from SHAP/LIME attributions.

## Limitations (always included)
- **Target:** a transformation of WAB-R AQ (100 − AQ), not a newly validated clinical severity scale.
- **Evaluation vs. validation:** held-out evaluation is a research metric, not clinical validation.
- **ASR:** Whisper is not adapted to aphasic speech, and its errors propagate into the language features.
- **Generalisation:** performance depends on dataset size, site, task and recording conditions. Generalisation beyond the dataset population is unknown.
- **Bands:** derived bands apply the published WAB AQ ranges to predicted AQ. They are a convenience label, not an independent classification.
- **Training vs. deployment audio:** training uses time-aligned participant speech only, while uploaded recordings are analysed as whole files.
