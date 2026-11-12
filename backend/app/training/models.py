"""Severity regression models.

All models predict the WAB-R-derived severity score (100 − AQ) from one or more feature groups.
Raw (unclipped) outputs are always returned; validity / clipping is decided only at the final
output boundary (``target.finalize_prediction``).

    ridge               Model A — standardised features → ridge regression (α selected on validation)
    mlp                 Model B — foundation-model embedding → MLP regression head
    fusion_transformer  Model C — one token per feature group (projection + modality embedding),
                        a [CLS] token, TransformerEncoder, regression head on [CLS]

Input preprocessing (``InputPipeline``) is fitted on the TRAINING split only:
  * columns entirely missing in training are dropped (listed);
  * remaining missing values are imputed with the training median;
  * for every named-feature column that had missing values in training, a binary
    ``<name>__missing`` indicator column is appended, so imputed values are distinguishable;
  * columns are standardised with training mean/std (indicators are left as 0/1).
Everything is saved as JSON/NPZ (no pickle); torch weights are loaded with ``weights_only=True``.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCALE = 100.0  # targets are trained on severity/100
MISSING_SUFFIX = "__missing"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class InputPipeline:
    groups: list[str]
    source_names: dict[str, list[str]] = field(default_factory=dict)  # kept input columns per group
    names: dict[str, list[str]] = field(default_factory=dict)  # output columns (kept + missing indicators)
    indicators: dict[str, list[str]] = field(default_factory=dict)  # source columns that get an indicator
    dropped: dict[str, list[str]] = field(default_factory=dict)
    input_dims: dict[str, int] = field(default_factory=dict)  # full input width per group seen at fit time
    medians: dict[str, np.ndarray] = field(default_factory=dict)
    means: dict[str, np.ndarray] = field(default_factory=dict)
    stds: dict[str, np.ndarray] = field(default_factory=dict)
    imputed_counts: dict[str, int] = field(default_factory=dict)
    ood_reference: dict[str, dict[str, float]] = field(default_factory=dict)
    # named-feature groups only: {feature: [train_min, train_max, train_std]} on the imputed training rows
    feature_ranges: dict[str, dict[str, list[float]]] = field(default_factory=dict)

    def fit(self, groups: dict[str, np.ndarray], names: dict[str, list[str]]) -> "InputPipeline":
        for g in self.groups:
            m = np.asarray(groups[g], dtype=np.float64)
            self.input_dims[g] = m.shape[1]
            keep = ~np.all(np.isnan(m), axis=0)
            self.source_names[g] = [n for n, k in zip(names[g], keep) if k]
            self.dropped[g] = [n for n, k in zip(names[g], keep) if not k]
            m = m[:, keep]
            nan = np.isnan(m)
            self.imputed_counts[g] = int(nan.sum())
            # embeddings are never partially missing; indicators only make sense for named features
            self.indicators[g] = [] if g.startswith("embedding:") else \
                [n for n, has in zip(self.source_names[g], nan.any(axis=0)) if has]
            self.names[g] = self.source_names[g] + [n + MISSING_SUFFIX for n in self.indicators[g]]
            self.medians[g] = np.nanmedian(m, axis=0)
            filled = np.where(nan, self.medians[g], m)
            std = filled.std(axis=0)
            k = len(self.indicators[g])
            self.means[g] = np.concatenate([filled.mean(axis=0), np.zeros(k)])
            self.stds[g] = np.concatenate([np.where(std > 1e-12, std, 1.0), np.ones(k)])
            if not g.startswith("embedding:"):
                self.feature_ranges[g] = {n: [float(filled[:, i].min()), float(filled[:, i].max()), float(std[i])]
                                          for i, n in enumerate(self.source_names[g])}
        raw = self.raw(groups, names)
        for g, score in self.ood_scores(raw).items():  # reference distribution of the OOD score on TRAINING rows
            self.ood_reference[g] = {"train_max": float(score.max()), "train_p95": float(np.percentile(score, 95)),
                                     "train_median": float(np.median(score))}
        return self

    def check_dims(self, groups: dict[str, np.ndarray]) -> None:
        for g in self.groups:
            if g not in groups:
                raise ValueError(f"missing feature group '{g}'")
            width = np.atleast_2d(groups[g]).shape[1]
            if width != self.input_dims[g]:
                raise ValueError(f"feature group '{g}' has {width} columns; the model was trained with {self.input_dims[g]}")

    def missing_report(self, groups: dict[str, np.ndarray], names: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
        """Per group: kept columns that are missing in the first row, split by whether the model saw them missing
        in training (has an indicator) or never did (silent training-median imputation would be unprecedented)."""
        out = {}
        for g in self.groups:
            row = np.atleast_2d(groups[g])[0]
            miss = [n for n in self.source_names[g] if np.isnan(row[names[g].index(n)])]
            if miss:
                out[g] = {"seen_in_training": [n for n in miss if n in self.indicators[g]],
                          "never_missing_in_training": [n for n in miss if n not in self.indicators[g]]}
        return out

    def raw(self, groups: dict[str, np.ndarray], names: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Select kept columns, impute, append missing indicators (unstandardised, interpretable scale)."""
        self.check_dims(groups)
        out = {}
        for g in self.groups:
            idx = [names[g].index(n) for n in self.source_names[g]]
            m = np.atleast_2d(groups[g])[:, idx].astype(np.float64)
            nan = np.isnan(m)
            ind = nan[:, [self.source_names[g].index(n) for n in self.indicators[g]]].astype(np.float64)
            out[g] = np.hstack([np.where(nan, self.medians[g], m), ind])
        return out

    def standardize(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {g: (raw[g] - self.means[g]) / self.stds[g] for g in self.groups}

    def transform(self, groups, names):
        return self.standardize(self.raw(groups, names))

    def out_of_range(self, raw: dict[str, np.ndarray], margin_sd: float) -> dict[str, dict[str, dict]]:
        """Named features of the first row that lie outside the training range by more than ``margin_sd`` training
        standard deviations. A linear model extrapolates without limit, so these inputs are unsupported."""
        out: dict[str, dict[str, dict]] = {}
        for g, ranges in self.feature_ranges.items():
            for i, n in enumerate(self.source_names[g]):
                lo, hi, sd = ranges[n]
                v, m = float(raw[g][0, i]), margin_sd * sd
                if v < lo - m or v > hi + m:
                    out.setdefault(g, {})[n] = {"value": round(v, 4), "training_min": round(lo, 4),
                                               "training_max": round(hi, 4), "allowed_margin": round(m, 4)}
        return out

    def ood_scores(self, raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Mean absolute z-score per row and group (indicator columns excluded). ~0.8 for in-distribution data."""
        out = {}
        for g in self.groups:
            k = len(self.source_names[g])
            out[g] = np.abs((raw[g][:, :k] - self.means[g][:k]) / self.stds[g][:k]).mean(axis=1)
        return out

    def save(self, d: Path) -> None:
        np.savez(d / "input_pipeline.npz", **{f"{k}::{g}": getattr(self, k)[g] for k in ("medians", "means", "stds")
                                              for g in self.groups})
        (d / "input_pipeline.json").write_text(json.dumps(
            {k: getattr(self, k) for k in ("groups", "source_names", "names", "indicators", "dropped", "input_dims",
                                           "imputed_counts", "ood_reference", "feature_ranges")}, indent=1))

    @classmethod
    def load(cls, d: Path) -> "InputPipeline":
        p = cls(**json.loads((d / "input_pipeline.json").read_text()))
        with np.load(d / "input_pipeline.npz", allow_pickle=False) as z:
            for k in ("medians", "means", "stds"):
                setattr(p, k, {g: z[f"{k}::{g}"] for g in p.groups})
        return p


# --------------------------------------------------------------------------- Model A
class RidgeModel:
    kind = "ridge"

    def __init__(self, groups: list[str], alpha: float = 1.0, alphas: list[float] | None = None, **_):
        self.groups = groups
        self.alphas = [float(a) for a in (alphas or [alpha])]
        self.alpha = self.alphas[0]
        self.coef = self.intercept = None

    @property
    def hp(self) -> dict:
        return {"alpha": self.alpha, "alphas": self.alphas}

    def _fit_one(self, X, y, alpha):
        from sklearn.linear_model import Ridge

        m = Ridge(alpha=alpha).fit(X, y / SCALE)
        return m.coef_.astype(np.float64), float(m.intercept_)

    def fit(self, X: dict[str, np.ndarray], y: np.ndarray, Xv=None, yv=None, **_) -> dict:
        Xm = np.hstack([X[g] for g in self.groups])
        table = []
        if len(self.alphas) > 1 and Xv is not None and len(yv):
            Xvm = np.hstack([Xv[g] for g in self.groups])
            for a in self.alphas:  # α is selected on the validation split only
                coef, b = self._fit_one(Xm, y, a)
                table.append({"alpha": a, "val_mae": float(np.mean(np.abs((Xvm @ coef + b) * SCALE - yv)))})
            self.alpha = min(table, key=lambda r: r["val_mae"])["alpha"]
        self.coef, self.intercept = self._fit_one(Xm, y, self.alpha)
        return {"history": [], "alpha_selected": self.alpha, "alpha_search": table,
                "selection": "validation MAE" if table else "fixed alpha (no search)"}

    def predict(self, X: dict[str, np.ndarray]) -> np.ndarray:
        return (np.hstack([X[g] for g in self.groups]) @ self.coef + self.intercept) * SCALE

    def n_params(self) -> int:
        return int(self.coef.size + 1)

    def save(self, d: Path) -> None:
        np.savez(d / "weights.npz", coef=self.coef, intercept=np.array([self.intercept]))

    def load(self, d: Path) -> None:
        with np.load(d / "weights.npz", allow_pickle=False) as z:
            self.coef, self.intercept = z["coef"], float(z["intercept"][0])

    def architecture(self, dims: dict[str, int]) -> dict:
        return {"type": "ridge", "inputs": dims, "alpha": self.alpha, "input_dim": sum(dims.values())}


# --------------------------------------------------------------------------- torch models
def _torch_modules():
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self, dims: dict[str, int], hidden: list[int], dropout: float):
            super().__init__()
            self.groups = list(dims)
            layers, d = [], sum(dims.values())
            for h in hidden:
                layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
                d = h
            layers.append(nn.Linear(d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, inputs: dict[str, "torch.Tensor"]):
            return self.net(torch.cat([inputs[g] for g in self.groups], dim=1)).squeeze(-1)

    class FusionTransformer(nn.Module):
        def __init__(self, dims: dict[str, int], d_model: int, n_heads: int, n_layers: int, ff: int, dropout: float):
            super().__init__()
            self.groups = list(dims)
            self.proj = nn.ModuleDict({self._k(g): nn.Sequential(nn.Dropout(dropout), nn.Linear(dims[g], d_model),
                                                                 nn.LayerNorm(d_model)) for g in self.groups})
            self.modality = nn.Parameter(torch.zeros(len(self.groups), d_model))
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            layer = nn.TransformerEncoderLayer(d_model, n_heads, ff, dropout, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
            nn.init.normal_(self.modality, std=0.02)
            nn.init.normal_(self.cls, std=0.02)

        @staticmethod
        def _k(g: str) -> str:
            return g.replace(":", "__")

        def forward(self, inputs):
            tokens = torch.stack([self.proj[self._k(g)](inputs[g]) for g in self.groups], dim=1) + self.modality
            x = torch.cat([self.cls.expand(tokens.shape[0], -1, -1), tokens], dim=1)
            return self.head(self.encoder(x)[:, 0]).squeeze(-1)

    return MLP, FusionTransformer


class TorchRegressor:
    def __init__(self, kind: str, groups: list[str], epochs: int = 200, lr: float = 1e-3, weight_decay: float = 1e-2,
                 batch_size: int = 16, patience: int = 25, loss: str = "huber", dropout: float = 0.2,
                 hidden: list[int] | None = None, d_model: int = 32, n_heads: int = 4, n_layers: int = 1,
                 ff: int = 64, seed: int = 13, grad_clip: float = 1.0):
        if kind == "fusion_transformer" and d_model % n_heads:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.kind, self.groups = kind, groups
        self.hp = dict(epochs=epochs, lr=lr, weight_decay=weight_decay, batch_size=batch_size, patience=patience,
                       loss=loss, dropout=dropout, hidden=hidden or [128], d_model=d_model, n_heads=n_heads,
                       n_layers=n_layers, ff=ff, seed=seed, grad_clip=grad_clip)
        self.net = None
        self.dims: dict[str, int] = {}

    def _build(self, dims: dict[str, int]):
        MLP, Fusion = _torch_modules()
        self.dims = dims
        hp = self.hp
        if self.kind == "mlp":
            return MLP(dims, hp["hidden"], hp["dropout"])
        return Fusion(dims, hp["d_model"], hp["n_heads"], hp["n_layers"], hp["ff"], hp["dropout"])

    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.net.parameters()))

    def fit(self, X, y, Xv=None, yv=None) -> dict:
        import torch

        set_seed(self.hp["seed"])
        self.net = self._build({g: X[g].shape[1] for g in self.groups})
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.hp["lr"], weight_decay=self.hp["weight_decay"])
        loss_fn = torch.nn.HuberLoss(delta=0.1) if self.hp["loss"] == "huber" else torch.nn.MSELoss()
        Xt = {g: torch.tensor(X[g], dtype=torch.float32) for g in self.groups}
        yt = torch.tensor(y / SCALE, dtype=torch.float32)
        gen = torch.Generator().manual_seed(self.hp["seed"])
        has_val = Xv is not None and yv is not None and len(yv) > 0
        best, best_state, best_epoch, bad, history = np.inf, None, None, 0, []
        for epoch in range(self.hp["epochs"]):
            self.net.train()
            perm = torch.randperm(len(yt), generator=gen)
            total = 0.0
            for i in range(0, len(yt), self.hp["batch_size"]):
                b = perm[i:i + self.hp["batch_size"]]
                opt.zero_grad()
                loss = loss_fn(self.net({g: Xt[g][b] for g in self.groups}), yt[b])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at epoch {epoch + 1}")
                loss.backward()
                if self.hp["grad_clip"]:
                    torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.hp["grad_clip"])
                opt.step()
                total += loss.item() * len(b)
            row = {"epoch": epoch + 1, "train_loss": total / len(yt)}
            if has_val:
                val_mae = float(np.mean(np.abs(self.predict(Xv) - yv)))
                row["val_mae"] = val_mae
                if val_mae < best - 1e-6:
                    best, bad, best_epoch = val_mae, 0, epoch + 1
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
            history.append(row)
            if has_val and bad >= self.hp["patience"]:
                break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return {"history": history, "epochs_run": len(history), "best_epoch": best_epoch,
                "best_val_mae": None if best == np.inf else best,
                "early_stopping": "validation MAE (selection split only; test never used)" if has_val
                else "none (no selection split; trained for all epochs)"}

    def predict(self, X) -> np.ndarray:
        import torch

        self.net.eval()
        with torch.no_grad():
            out = self.net({g: torch.tensor(X[g], dtype=torch.float32) for g in self.groups})
        return out.numpy().astype(np.float64) * SCALE

    def save(self, d: Path) -> None:
        import torch

        torch.save(self.net.state_dict(), d / "weights.pt")
        (d / "torch_dims.json").write_text(json.dumps(self.dims))

    def load(self, d: Path) -> None:
        import torch

        self.net = self._build(json.loads((d / "torch_dims.json").read_text()))
        self.net.load_state_dict(torch.load(d / "weights.pt", weights_only=True))
        self.net.eval()

    def architecture(self, dims: dict[str, int]) -> dict:
        hp = self.hp
        if self.kind == "mlp":
            return {"type": "mlp", "inputs": dims, "input_dim": sum(dims.values()), "hidden": hp["hidden"],
                    "activation": "GELU", "dropout": hp["dropout"], "output": "1 (severity/100)"}
        return {"type": "fusion_transformer", "inputs": dims,
                "fusion": "one token per feature group: Dropout→Linear(d_g→d_model)→LayerNorm, + learned modality "
                          "embedding; [CLS] token prepended",
                "encoder": {"d_model": hp["d_model"], "heads": hp["n_heads"], "layers": hp["n_layers"],
                            "ff": hp["ff"], "dropout": hp["dropout"], "norm_first": True},
                "head": "LayerNorm → Linear(d_model→1) on [CLS]", "output": "1 (severity/100)"}


def build_model(cfg: dict, groups: list[str], seed: int):
    kind = cfg["type"]
    params = {k: v for k, v in cfg.items() if k != "type"}
    if kind == "ridge":
        return RidgeModel(groups, **params)
    if kind in ("mlp", "fusion_transformer"):
        return TorchRegressor(kind, groups, seed=seed, **params)
    raise ValueError(f"Unknown model type '{kind}'")
