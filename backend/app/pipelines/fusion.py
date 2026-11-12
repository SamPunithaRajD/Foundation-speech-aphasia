"""Multimodal feature fusion infrastructure.

Implemented strategies (prototype, no claim of superiority):

* ``concatenate``          – early fusion: [norm(emb_1), ..., norm(emb_k), norm(clinical)]
* ``weighted_concatenate`` – same, each block scaled by ``FSA_FUSION_WEIGHTS``

Block normalisation:

* ``l2``     – per-block L2 normalisation (valid for a single sample)
* ``none``   – raw values
* ``fitted`` – z-score using statistics fitted on training data; requires a trained bundle

Proposed but *not implemented*: attention / cross-modal transformer fusion (needs training).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from app.schemas.analysis import FusionBlock, FusionResult
from app.schemas.common import Maturity, SectionState

STRATEGIES = {
    "concatenate": {"maturity": Maturity.PROTOTYPE.value, "description": "Early fusion by block concatenation."},
    "weighted_concatenate": {"maturity": Maturity.PROTOTYPE.value, "description": "Concatenation with per-block weights."},
    "cross_attention_transformer": {"maturity": Maturity.PROPOSED.value,
                                    "description": "Cross-modal attention fusion; requires training. Not implemented."},
}


@dataclass
class FusedRepresentation:
    vector: np.ndarray
    result: FusionResult


def _normalize(v: np.ndarray, method: str, stats: dict | None) -> np.ndarray:
    if method == "l2":
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v
    if method == "fitted":
        if not stats:
            raise ValueError("fitted normalisation requires training statistics")
        return (v - stats["mean"]) / np.where(stats["std"] > 0, stats["std"], 1.0)
    return v


def fuse(blocks: dict[str, np.ndarray], strategy: str, normalization: str, weights: dict[str, float] | None = None,
         fitted_stats: dict[str, dict] | None = None) -> FusedRepresentation | FusionResult:
    if strategy not in STRATEGIES or STRATEGIES[strategy]["maturity"] == Maturity.PROPOSED.value:
        return FusionResult(state=SectionState.NOT_AVAILABLE, strategy=strategy, maturity=Maturity.PROPOSED,
                            message=f"Fusion strategy '{strategy}' is not implemented.")
    if normalization == "fitted" and not fitted_stats:
        return FusionResult(state=SectionState.MODEL_NOT_READY, strategy=strategy,
                            message="Fitted normalisation requires statistics from a trained model bundle.")
    if not blocks:
        return FusionResult(state=SectionState.INSUFFICIENT_DATA, strategy=strategy,
                            message="No embedding or feature blocks were available to fuse.")
    parts, meta = [], []
    for name, vec in blocks.items():
        w = float((weights or {}).get(name, 1.0)) if strategy == "weighted_concatenate" else 1.0
        parts.append(_normalize(vec.astype(np.float32), normalization, (fitted_stats or {}).get(name)) * w)
        meta.append(FusionBlock(name=name, dimension=int(vec.shape[0]), normalization=normalization, weight=w))
    vector = np.concatenate(parts)
    return FusedRepresentation(
        vector=vector,
        result=FusionResult(state=SectionState.COMPUTED, strategy=strategy, blocks=meta,
                            unified_dimension=int(vector.shape[0]), maturity=Maturity.PROTOTYPE,
                            message="Prototype multimodal representation built for inspection. It is not the severity "
                                    "model's input: a trained model applies its own train-fitted feature pipeline."),
    )
