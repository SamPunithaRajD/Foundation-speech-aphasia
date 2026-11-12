"""Tabulate experiment results (validation / CV first; test only where it was evaluated).

    python -m app.training.compare artifacts/experiments/*

Rows for planned experiments without results (including the base-paper comparison) are listed
with empty cells. No result is ever filled in by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.training.evaluation import EXPERIMENTS


def _cell(metrics: dict, split: str, key: str) -> str:
    m = metrics.get(split, {}).get("participant_level", {})
    v = m.get(key)
    v = v.get("value") if isinstance(v, dict) else v
    return f"{v:.3f}" if isinstance(v, (int, float)) else ""


def table(exp_dirs: list[Path]) -> str:
    rows = ["| experiment | origin | status | params | val MAE | val RMSE | val R² | val Spearman | CV MAE (mean±sd) | invalid (val) | test MAE | test R² |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    seen = set()
    for d in exp_dirs:
        mp, bp = d / "metrics.json", d / "bundle" / "bundle.json"
        if not mp.exists() or not bp.exists():
            continue
        m, b = json.loads(mp.read_text()), json.loads(bp.read_text())
        cv = m.get("cross_validation")
        cv_s = f"{cv['participant_mae_mean']:.3f}±{cv['participant_mae_sd']:.3f}" if cv else ""
        rows.append(f"| {d.name} | {b['data_origin']} | {b['status']} | {b.get('n_parameters', '')} | {_cell(m, 'validation', 'mae')} | "
                    f"{_cell(m, 'validation', 'rmse')} | {_cell(m, 'validation', 'r2')} | {_cell(m, 'validation', 'spearman')} | "
                    f"{cv_s} | {m.get('validation', {}).get('participant_level', {}).get('n_invalid_outputs', '')} | "
                    f"{_cell(m, 'test', 'mae')} | {_cell(m, 'test', 'r2')} |")
        seen.add(d.name.rsplit("_", 1)[0])
    for name, info in EXPERIMENTS.items():
        done = name in seen or (name.startswith("D_") and any(x.startswith("D_") for x in seen))
        if not done:
            rows.append(f"| {name} (planned) |  | {info['status']} |  |  |  |  |  |  |  |  |  |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment_dirs", nargs="*", type=Path)
    print(table(ap.parse_args(argv).experiment_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
