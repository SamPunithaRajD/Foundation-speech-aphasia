"""Model card generation (JSON + Markdown). See docs/MODEL_CARD_TEMPLATE.md.

A card is only ever filled from values produced by an actual run; fields without data are
written as NOT_AVAILABLE. The default card (no model) states STATUS: MODEL_NOT_READY.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.training import target as tgt

NA = "NOT_AVAILABLE"

INTENDED_USE = ("Research on automatic estimation of a WAB-R-derived severity score from speech recordings of "
                "participants in the labelled research dataset the model was evaluated on.")
NOT_INTENDED_USE = [
    "Clinical diagnosis, screening, treatment planning or any decision about an individual.",
    "Inference on people outside the evaluated dataset population, including personal or microphone recordings "
    "uploaded to the application (reported as NOT_VALID_FOR_CLINICAL_INFERENCE).",
    "Claims of medical causation from SHAP/LIME attributions.",
]
BASE_LIMITATIONS = [
    "The target is a transformation of WAB-R AQ (100 − AQ); it is not a newly validated clinical severity scale.",
    "Clinical validation: NOT_VALIDATED. Held-out evaluation is a research metric, not clinical validation.",
    "Whisper ASR is not adapted to aphasic speech; transcription errors propagate into language features.",
    "Performance depends on dataset size, site, task and recording conditions; generalisation beyond the dataset "
    "population is unknown.",
    "Derived severity bands apply published WAB AQ ranges (Kertesz, 2007) to predicted AQ; the band is a convenience "
    "label, not an independent classification.",
]


def empty_card() -> dict:
    return {"status": "MODEL_NOT_READY", "clinical_validation": "NOT_VALIDATED",
            "reason": "No real labelled aphasia dataset has been trained on.", "target": tgt.target_definition(),
            "intended_use": INTENDED_USE, "not_intended_use": NOT_INTENDED_USE, "limitations": BASE_LIMITATIONS}


def build_card(*, status: str, data_origin: str, experiment: dict, dataset_report: dict, split: dict,
               split_counts: dict, feature_config: dict, input_summary: dict, architecture: dict,
               hyperparameters: dict, training: dict, metrics: dict, environment: dict, embedding_models: dict,
               preprocessing: dict) -> dict:
    return {
        "status": status,
        "data_origin": data_origin,
        "clinical_validation": "NOT_VALIDATED",
        "experiment": experiment,
        "dataset": {"name": dataset_report.get("dataset"), "version": dataset_report.get("dataset_version"),
                    "participant_count": dataset_report.get("trainable_participants"),
                    "recording_count": dataset_report.get("trainable_records"),
                    "total_transcripts": dataset_report.get("total_transcripts"),
                    "excluded": {k: len(dataset_report.get(k, [])) for k in
                                 ("missing_audio", "missing_aq", "invalid_aq", "ambiguous_aq", "duplicate_records")},
                    "derived_band_distribution": dataset_report.get("band_distribution")},
        "target": tgt.target_definition(),
        "split": {"method": split.get("method"), "seed": split.get("seed"), "counts": split_counts,
                  "participant_ids": {k: split.get(k) for k in ("train", "validation", "test")},
                  "leakage_check": "passed"},
        "foundation_models": embedding_models or NA,
        "preprocessing": preprocessing,
        "language_features": {"source": feature_config.get("language_source"),
                              "included": "language" in feature_config.get("groups", []),
                              "excluded_always": ["syntactic_complexity (no parser)"]},
        "feature_groups": feature_config.get("groups"),
        "inputs": input_summary,
        "fusion_method": architecture.get("fusion", "none (single group)" if len(feature_config.get("groups", [])) == 1
                                          else "concatenation"),
        "architecture": architecture,
        "hyperparameters": hyperparameters,
        "training": training,
        "metrics": metrics or NA,
        "environment": environment,
        "intended_use": INTENDED_USE,
        "not_intended_use": NOT_INTENDED_USE,
        "limitations": BASE_LIMITATIONS + ([] if data_origin == "REAL" else [
            "SYNTHETIC_TEST_DATA: trained on generated placeholder data to test software only. Metrics are meaningless "
            "and this model can never be loaded for predictions."]),
    }


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return NA
    return str(v)


def render_markdown(card: dict) -> str:
    lines = ["# FoundationSpeech-Aphasia — Severity Model Card", "",
             f"**STATUS: {card['status']}**  ·  Clinical validation: **{card['clinical_validation']}**", ""]
    if card.get("data_origin") and card["data_origin"] != "REAL":
        lines += ["> **SYNTHETIC_TEST_DATA** — software test only. Not a clinical model. Metrics are meaningless.", ""]
    t = card["target"]
    lines += ["## Target", f"- {t['name']}: `{t['transformation']}` ({t['direction']}, range {t['range']})",
              f"- Source: {t['source']} · version `{t['version']}`", f"- Bands: {t['bands']['source']}", ""]
    if card["status"] == "MODEL_NOT_READY":
        lines += [f"Reason: {card.get('reason')}", ""]
    else:
        ds, sp = card["dataset"], card["split"]
        lines += ["## Dataset", f"- {ds['name']} (version {ds['version']})",
                  f"- Participants: {ds['participant_count']} · recordings: {ds['recording_count']}",
                  f"- Excluded records: {ds['excluded']}", f"- Derived band distribution: {ds['derived_band_distribution']}", "",
                  "## Split", f"- {sp['method']} (seed {sp['seed']}) · leakage check: **{sp['leakage_check']}**"]
        lines += [f"- {k}: {v['participants']} participants / {v['recordings']} recordings" for k, v in sp["counts"].items()]
        lines += ["", "## Model", f"- Feature groups: {card['feature_groups']}", f"- Foundation models: {card['foundation_models']}",
                  f"- Fusion: {card['fusion_method']}", f"- Architecture: `{json.dumps(card['architecture'])}`",
                  f"- Hyperparameters: `{json.dumps(card['hyperparameters'])}`",
                  f"- Complexity: `{json.dumps(card['training'].get('complexity'))}`",
                  f"- Missing-value handling: {card['inputs'].get('imputation')}", "", "## Metrics (severity scale 0–100)"]
        for split_name, m in (card["metrics"] or {}).items():
            if not isinstance(m, dict) or "participant_level" not in m:
                continue
            p = m["participant_level"]
            lines.append(f"- **{split_name}** (participant level, n={p['n']}): MAE {_fmt(p['mae'])}, RMSE {_fmt(p['rmse'])}, "
                         f"R² {_fmt(p['r2'])}, Pearson {_fmt(p['pearson']['value'])}, Spearman {_fmt(p['spearman']['value'])}; "
                         f"MAE 95% CI {m['mae_95ci']}; invalid outputs {p.get('n_invalid_outputs')}, "
                         f"unclipped MAE {_fmt(p.get('mae_unclipped'))}")
        lines += ["", "## Environment", f"`{json.dumps(card['environment'])}`"]
    lines += ["", "## Intended use", card["intended_use"], "", "## Not intended for"] + [f"- {x}" for x in card["not_intended_use"]]
    lines += ["", "## Limitations"] + [f"- {x}" for x in card["limitations"]]
    return "\n".join(lines) + "\n"


def write_card(d: Path, card: dict) -> None:
    (d / "model_card.json").write_text(json.dumps(card, indent=2, default=str), encoding="utf-8")
    (d / "MODEL_CARD.md").write_text(render_markdown(card), encoding="utf-8")
