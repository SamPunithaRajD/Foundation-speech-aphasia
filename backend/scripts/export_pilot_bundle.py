"""Write a PILOT severity bundle for the QAB text pilot (called by qab_pilot_audit.py --export-bundle).

The bundle is fitted on all eligible participants AFTER the leave-one-participant-out evaluation; every evaluation
number stored in it comes from held-out participants, never from this final fit. It carries status
PILOT_CROSS_VALIDATED: it can only be served when the operator sets FSA_ALLOW_PILOT_MODEL=true and promotes it
with ``python -m app.training.promote <dir> --reviewer "<name>" --pilot``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.config import Settings
from app.models.registry import ModelRegistry
from app.training import target as tgt
from app.training.bundle import STATUS_PILOT, save_bundle
from app.training.dataset import REAL
from app.training.features import LANGUAGE_FEATURES, extraction_spec
from app.training.model_card import NOT_INTENDED_USE, render_markdown
from app.training.models import InputPipeline, RidgeModel


def limitations(summary: dict) -> list[str]:
    n, total = summary["participants"], summary["participants_in_dataset"]
    return [
        f"PILOT MODEL: fitted on {n} participants with chronic post-stroke aphasia from one study (Quick Aphasia Battery "
        "Corpus; Wilson et al., PLOS ONE 2018). Evaluated by leave-one-participant-out cross-validation only; there is no "
        "independent test set.",
        f"Recordings with fewer than {summary['min_words']} transcribed words cannot be scored. In the pilot data this "
        f"excluded {summary['excluded_transcripts']} of {summary['transcripts_in_dataset']} transcripts and removed "
        f"{total - n} of {total} participants entirely, so the model is untested for people who produce very little speech "
        "(typically the most severe cases).",
        "Text only: it uses two word-based transcript measures. It uses no acoustic features and no speech "
        "foundation-model embeddings, so it is not the proposed multimodal framework. In practice vocabulary variety "
        "(moving-average type-token ratio) carries the prediction.",
        "NOT VALIDATED ON WHISPER TRANSCRIPTS: the model was fitted and evaluated on human transcripts and is applied "
        "here to automatic Whisper transcripts. Whisper omits fillers and repetitions and makes recognition errors, which "
        "shifts these measures by an unknown amount. The reported accuracy does not transfer to Whisper input until it is "
        "re-evaluated on the same participants' recordings.",
        "The feature set was chosen after other feature sets and feature/label correlations had been examined. The "
        "reported significance is therefore the family-wise adjusted value over every candidate examined.",
        "The label is the Aphasia Quotient of the original Western Aphasia Battery (Kertesz, 1982), administered once at "
        "the first of three sessions; it is not the WAB-R. The target (100 − AQ) is a transformation of that score, not a "
        "validated clinical severity scale.",
        "The training speech is a transcribed sample of clinician-led conversation; other speaking situations differ.",
        "The interval is the largest leave-one-participant-out error on human transcripts. It is not a probability and "
        "its coverage is unknown for Whisper transcripts.",
    ]


def export(out: Path, *, use, feats: list[str], y_by_pid: dict, pilot_id: str, min_words: int, fit_predict,
           summary: dict, uncertainty: dict) -> Path:
    X, g = use[feats].to_numpy(float), use["pid"].to_numpy()
    y = np.array([y_by_pid[p] for p in g])
    assert set(g) == set(y_by_pid) and (use.words >= min_words).all()  # eligible participants/transcripts only
    _, alpha = fit_predict(X, y, g, X[:1], feats)  # alpha by participant-grouped CV over all eligible participants
    names = {"language": feats}
    pipe = InputPipeline(["language"]).fit({"language": X}, names)
    model = RidgeModel(["language"], alpha=alpha)
    model.fit(pipe.transform({"language": X}, names), y)
    settings = Settings(_env_file=None)
    lims = limitations(summary)
    target = tgt.wab_1982_target_definition()
    feature_config = {"groups": ["language"], "language_source": "human", "apply_to_asr_transcripts": True,
                      "exclude_features": [n for n in LANGUAGE_FEATURES if n not in feats]}
    raw = pipe.raw({"language": X}, names)
    save_bundle(out, model=model, pipeline=pipe, feature_config=feature_config, data_origin=REAL, status=STATUS_PILOT,
                experiment_id=pilot_id, embedding_models={}, background={"language": raw["language"][:100]},
                extraction_spec=extraction_spec(settings, ModelRegistry(settings), ["language"], "human"),
                input_names=names, target=target,
                extra={"uncertainty": uncertainty, "min_words": min_words, "model_stage": "PILOT",
                       "evaluation_summary": summary, "limitations": lims,
                       "whisper_transfer_validated": False,
                       "complexity": {"n_parameters": model.n_params(), "training_participants": int(len(y_by_pid)),
                                      "training_recordings": int(len(use))}})
    card = {"status": STATUS_PILOT, "data_origin": REAL, "clinical_validation": "NOT_VALIDATED", "target": target,
            "experiment": {"id": pilot_id},
            "dataset": {"name": summary["dataset"], "version": "QAB open release, doi:10.21415/T8P2-CN50 (downloaded 2026-10-12)",
                        "participant_count": summary["participants"], "recording_count": summary["transcripts"],
                        "excluded": {"transcripts_below_min_words": summary["excluded_transcripts"],
                                     "participants_without_eligible_transcript": summary["participants_in_dataset"] - summary["participants"]},
                        "derived_band_distribution": None},
            "split": {"method": "leave-one-participant-out cross-validation (all sessions of a participant held out together)",
                      "seed": None, "counts": {}, "leakage_check": "passed"},
            "feature_groups": ["language"], "foundation_models": "none (text only)", "fusion_method": "none (single group)",
            "architecture": model.architecture({"language": len(feats)}) | {"features": feats},
            "hyperparameters": model.hp, "training": {"complexity": {"n_parameters": model.n_params()}},
            "inputs": {"imputation": "none (transcripts below the minimum word count are excluded, so no value is missing)"},
            "metrics": {"leave_one_participant_out": summary}, "environment": {},
            "intended_use": "Exploratory research demonstration of the clinical-language branch of FoundationSpeech-Aphasia.",
            "not_intended_use": NOT_INTENDED_USE, "limitations": lims}
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
    md = render_markdown(card).replace("## Metrics (severity scale 0–100)",
                                       "## Metrics (severity scale 0–100)\n- " + json.dumps(summary, ensure_ascii=False))
    (out / "MODEL_CARD.md").write_text(md, encoding="utf-8")
    return out
