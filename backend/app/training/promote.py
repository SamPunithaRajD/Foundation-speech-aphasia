"""Promote an evaluated experiment bundle to the production model directory.

    python -m app.training.promote artifacts/experiments/<experiment_id> --reviewer "<name>"

Promotion is a deliberate, human step. It refuses bundles that:
  * were trained on SYNTHETIC_TEST_DATA,
  * have not been evaluated on the held-out test participants,
  * lack a passed participant-leakage check.
The production state stays MODEL_NOT_READY until this command succeeds.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.config import get_settings
from app.training.bundle import STATUS_PILOT, STATUS_TESTED, BundleRejected, load_bundle


def promote(exp_dir: Path, reviewer: str, target_dir: Path, replace: bool = False, pilot: bool = False) -> Path:
    bundle_dir = Path(exp_dir) / "bundle"
    bundle = load_bundle(bundle_dir)  # raises BundleRejected for synthetic bundles
    status = bundle.meta.get("status")
    if status == STATUS_PILOT and not pilot:
        raise BundleRejected("This is a PILOT (cross-validated only) bundle. Promoting it requires --pilot, and the "
                             "backend must run with FSA_ALLOW_PILOT_MODEL=true.")
    if pilot and status != STATUS_PILOT:
        raise BundleRejected(f"--pilot only applies to {STATUS_PILOT} bundles; this one is {status}.")
    if status not in (STATUS_TESTED, STATUS_PILOT):
        raise BundleRejected(f"Bundle status is {status}; it must be {STATUS_TESTED}.")
    if bundle.card.get("split", {}).get("leakage_check") != "passed":
        raise BundleRejected("Model card does not record a passed participant-leakage check.")
    if not reviewer.strip():
        raise BundleRejected("A reviewer name is required.")
    target_dir = Path(target_dir)
    if target_dir.exists():
        if not replace:
            raise BundleRejected(f"{target_dir} already exists; pass --replace to overwrite it.")
        shutil.rmtree(target_dir)
    shutil.copytree(bundle_dir, target_dir)
    (target_dir / "promotion.json").write_text(json.dumps({
        "experiment": str(exp_dir), "reviewer": reviewer, "promoted_at": datetime.now(timezone.utc).isoformat(),
        "clinical_validation": "NOT_VALIDATED", "pilot": status == STATUS_PILOT}, indent=2))
    return target_dir


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment_dir", type=Path)
    ap.add_argument("--reviewer", required=True)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--pilot", action="store_true", help="promote a PILOT_CROSS_VALIDATED bundle (small-sample model)")
    args = ap.parse_args(argv)
    try:
        out = promote(args.experiment_dir, args.reviewer, get_settings().severity_model_dir, args.replace, args.pilot)
    except (BundleRejected, FileNotFoundError) as exc:
        print(f"PROMOTION REFUSED: {exc}", file=sys.stderr)
        return 2
    print(f"Promoted to {out}. Restart the backend to load it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
