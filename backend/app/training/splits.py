"""Participant-level data splitting and leakage protection.

Recordings are NEVER split individually: whole participants are assigned to train, validation
or test. Stratification (optional) uses each participant's WAB-R AQ band derived from their
mean AQ, so the band mix is similar across splits. ``assert_no_leakage`` raises if any
participant occurs in more than one split and is called on every split that is created or loaded.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.training import target as tgt
from app.training.dataset import DatasetRecord


class ParticipantLeakage(AssertionError):
    pass


@dataclass
class Split:
    train: list[str]
    validation: list[str]
    test: list[str]
    seed: int
    method: str
    folds: list[dict[str, list[str]]] = field(default_factory=list)  # optional CV folds over train+validation
    params: dict = field(default_factory=dict)  # fractions / stratification the split was created with

    def assignments(self) -> dict[str, list[str]]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    def to_dict(self) -> dict:
        return {"train": sorted(self.train), "validation": sorted(self.validation), "test": sorted(self.test),
                "seed": self.seed, "method": self.method, "folds": self.folds, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        s = cls(d["train"], d["validation"], d["test"], d["seed"], d["method"], d.get("folds", []), d.get("params", {}))
        assert_no_leakage(s.assignments())
        return s


def assert_no_leakage(assignments: dict[str, list[str]]) -> None:
    owner: dict[str, str] = {}
    for split, pids in assignments.items():
        if len(set(pids)) != len(pids):
            raise ParticipantLeakage(f"duplicate participant IDs inside split '{split}'")
        for pid in pids:
            if pid in owner:
                raise ParticipantLeakage(f"participant '{pid}' appears in both '{owner[pid]}' and '{split}'")
            owner[pid] = split


def participant_bands(records: list[DatasetRecord]) -> dict[str, str]:
    aqs: dict[str, list[float]] = defaultdict(list)
    for r in records:
        aqs[r.participant_id].append(r.wab_aq)
    return {p: tgt.band_from_aq(float(np.mean(v))) for p, v in aqs.items()}


def split_participants(records: list[DatasetRecord], val_fraction: float = 0.15, test_fraction: float = 0.15,
                       seed: int = 13, stratify: bool = True, cv_folds: int = 0) -> Split:
    trainable = [r for r in records if r.trainable]
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be in (0, 1) and sum to < 1")
    bands = participant_bands(trainable)
    participants = sorted(bands)
    if len(participants) < 3:
        raise ValueError(f"Need at least 3 labelled participants to split; have {len(participants)}.")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for p in participants:
        groups[bands[p] if stratify else "all"].append(p)
    train, val, test = [], [], []
    for key in sorted(groups):
        members = list(groups[key])
        rng.shuffle(members)
        n = len(members)
        n_test, n_val = int(round(n * test_fraction)), int(round(n * val_fraction))
        test += members[:n_test]
        val += members[n_test:n_test + n_val]
        train += members[n_test + n_val:]
    # guarantee non-empty validation/test when possible (small datasets)
    for bucket in (test, val):
        if not bucket and len(train) > 1:
            bucket.append(train.pop(int(rng.integers(len(train)))))
    split = Split(sorted(train), sorted(val), sorted(test), seed,
                  f"participant-level, {'stratified by participant WAB-R AQ band' if stratify else 'unstratified'}",
                  params={"val_fraction": val_fraction, "test_fraction": test_fraction, "stratify": stratify,
                          "cv_folds": cv_folds})
    if cv_folds:
        split.folds = group_kfold(split.train + split.validation, bands, cv_folds, seed)
    assert_no_leakage(split.assignments())
    for fold in split.folds:
        assert_no_leakage({"train": fold["train"], "validation": fold["validation"], "test": split.test})
    return split


def group_kfold(participants: list[str], bands: dict[str, str], k: int, seed: int) -> list[dict[str, list[str]]]:
    """Participant-level K-fold (band-stratified round-robin). The held-out test set is never included."""
    if k < 2 or k > len(participants):
        raise ValueError(f"cv_folds must be between 2 and the number of participants ({len(participants)})")
    rng = np.random.default_rng(seed + 1)
    by_band = defaultdict(list)
    for p in sorted(participants):
        by_band[bands[p]].append(p)
    fold_of: dict[str, int] = {}
    offset = 0
    for band in sorted(by_band):
        members = by_band[band]
        rng.shuffle(members)
        for i, p in enumerate(members):
            fold_of[p] = (i + offset) % k
        offset += len(members)
    return [{"train": sorted(p for p in participants if fold_of[p] != f),
             "validation": sorted(p for p in participants if fold_of[p] == f)} for f in range(k)]


def inner_split(participants: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Participant-level split of a training set into (fit, selection). Used inside CV folds so that early stopping /
    hyper-parameter selection never sees the fold that is being scored."""
    ps = sorted(participants)
    if len(ps) < 4:
        return ps, []
    rng = np.random.default_rng(seed)
    rng.shuffle(ps)
    n_sel = max(1, int(round(len(ps) * fraction)))
    return sorted(ps[n_sel:]), sorted(ps[:n_sel])


def check_reusable(split: Split, trainable_participants: set[str], params: dict) -> list[str]:
    """Problems that make a saved split unusable for the current dataset state (empty list = reusable)."""
    assigned = set(split.train) | set(split.validation) | set(split.test)
    problems = []
    gone = sorted(assigned - trainable_participants)
    new = sorted(trainable_participants - assigned)
    if gone:
        problems.append(f"{len(gone)} participant(s) in the saved split are no longer trainable (e.g. {gone[:3]})")
    if new:
        problems.append(f"{len(new)} trainable participant(s) are not in the saved split and would be silently "
                        f"left out (e.g. {new[:3]})")
    for k in ("val_fraction", "test_fraction", "stratify"):
        if k in split.params and k in params and split.params[k] != params[k]:
            problems.append(f"{k} is {params[k]} in the config but the saved split used {split.params[k]}")
    return problems


def records_for(records: list[DatasetRecord], participants: list[str]) -> list[DatasetRecord]:
    ids = set(participants)
    return [r for r in records if r.trainable and r.participant_id in ids]


def split_summary(split: Split, records: list[DatasetRecord]) -> str:
    lines = []
    for name, pids in split.assignments().items():
        recs = records_for(records, pids)
        bands = defaultdict(int)
        for r in recs:
            bands[r.severity_band] += 1
        lines += [name.upper(), f"  participants: {len(pids)}", f"  recordings:   {len(recs)}",
                  f"  derived bands (recordings): {dict(bands)}"]
    lines.append(f"LEAKAGE CHECK: passed (no participant in more than one split); method: {split.method}; seed {split.seed}")
    return "\n".join(lines)


def save_split(split: Split, path: Path) -> None:
    path.write_text(json.dumps(split.to_dict(), indent=2), encoding="utf-8")
