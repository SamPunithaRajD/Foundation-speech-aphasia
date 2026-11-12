"""Dataset ingestion for labelled aphasia corpora (designed for AphasiaBank / TalkBank).

Nothing is assumed to exist. The dataset root comes from the dataset config or the
``APHASIA_DATA_ROOT`` environment variable, with the layout:

    <root>/audio/          media files (wav, mp3, mp4, m4a, flac)
    <root>/transcripts/    CHAT .cha files (one per session)
    <root>/scores/         WAB-R spreadsheet(s) (.xlsx / .csv)
    <root>/demographics/   optional demographics spreadsheet
    <root>/metadata/       optional, kept for provenance

Spreadsheet column names are NOT hard-coded: they must be set in the dataset config after
inspecting the real files. A record only becomes trainable when audio, transcript, participant
and a valid WAB-R AQ all match. Non-trainable records are kept and reported with their issues;
nothing is silently discarded and no label is ever filled in.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.training import target as tgt
from app.training.chat import ChatTranscript, parse_chat

REAL = "REAL"
SYNTHETIC = "SYNTHETIC_TEST_DATA"
SYNTHETIC_MARKER = "SYNTHETIC_TEST_DATA"  # marker file placed in the root of generated test datasets
LAYOUT_DIRS = ["audio", "transcripts", "scores", "demographics", "metadata"]


class DatasetNotAvailable(RuntimeError):
    pass


class DatasetConfigError(ValueError):
    pass


@dataclass
class TableConfig:
    file: str | None = None  # glob relative to its layout directory, e.g. "*.xlsx"
    sheet: str | int | None = None
    participant_column: str | None = None
    session_column: str | None = None
    aq_column: str | None = None  # scores only


@dataclass
class DatasetConfig:
    name: str
    version: str | None = None
    root: str | None = None
    speaker: str = "PAR"
    keep_fillers: bool = True
    file_id_regex: str = r"^(?P<participant>.+?)(?P<session>[a-z]?)$"
    audio_extensions: list[str] = field(default_factory=lambda: [".wav", ".mp3", ".mp4", ".m4a", ".flac"])
    tasks: list[str] = field(default_factory=list)  # CHAT @G gem labels; empty = whole session
    audio_scope: str = "speaker_utterances"  # speaker_utterances | full
    scores: TableConfig = field(default_factory=TableConfig)
    demographics: TableConfig = field(default_factory=TableConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetConfig":
        d = dict(d)
        d["scores"] = TableConfig(**(d.get("scores") or {}))
        d["demographics"] = TableConfig(**(d.get("demographics") or {}))
        return cls(**d)

    def resolve_root(self) -> Path:
        raw = self.root or os.environ.get("APHASIA_DATA_ROOT") or os.environ.get("FSA_APHASIA_DATA_ROOT")
        if not raw:
            raise DatasetNotAvailable("No dataset root configured. Set APHASIA_DATA_ROOT or 'root' in the dataset config.")
        root = Path(os.path.expandvars(raw)).expanduser().resolve()
        if not root.is_dir():
            raise DatasetNotAvailable(f"Dataset root does not exist: {root}")
        return root


@dataclass
class DatasetRecord:
    record_id: str
    participant_id: str
    session_id: str
    task: str
    audio_path: str | None
    transcript_path: str | None
    wab_aq: float | None
    severity_score: float | None
    severity_band: str | None
    target_source: str
    target_version: str
    data_origin: str
    human_transcript: str | None = None
    speaker_segments_ms: list[tuple[int, int]] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    aphasia_metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    @property
    def trainable(self) -> bool:
        return not self.issues and self.wab_aq is not None and bool(self.audio_path) and bool(self.transcript_path)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trainable": self.trainable}


@dataclass
class ValidationReport:
    dataset: str
    dataset_version: str
    data_origin: str
    root: str
    total_audio_files: int = 0
    total_transcripts: int = 0
    matched_records: int = 0
    trainable_records: int = 0
    trainable_participants: int = 0
    missing_transcripts: list[str] = field(default_factory=list)  # audio without transcript
    missing_audio: list[str] = field(default_factory=list)  # transcript without audio
    missing_aq: list[str] = field(default_factory=list)
    invalid_aq: list[str] = field(default_factory=list)
    ambiguous_aq: list[str] = field(default_factory=list)
    duplicate_records: list[str] = field(default_factory=list)
    unmatched_records: list[str] = field(default_factory=list)  # file ids the id regex could not parse
    transcript_warnings: dict[str, list[str]] = field(default_factory=dict)
    band_distribution: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"DATASET {self.dataset}  version={self.dataset_version}  origin={self.data_origin}",
            f"  audio files ........... {self.total_audio_files}",
            f"  transcripts ........... {self.total_transcripts}",
            f"  matched records ....... {self.matched_records}",
            f"  trainable records ..... {self.trainable_records}  (participants: {self.trainable_participants})",
            f"  missing transcripts ... {len(self.missing_transcripts)}",
            f"  missing audio ......... {len(self.missing_audio)}",
            f"  missing AQ ............ {len(self.missing_aq)}",
            f"  invalid AQ ............ {len(self.invalid_aq)}",
            f"  ambiguous AQ .......... {len(self.ambiguous_aq)}",
            f"  duplicate records ..... {len(self.duplicate_records)}",
            f"  unmatched file ids .... {len(self.unmatched_records)}",
            f"  derived band counts ... {self.band_distribution}",
        ]
        return "\n".join(lines + [f"  NOTE: {n}" for n in self.notes])


def load_table(path: Path, sheet: str | int | None):
    import pandas as pd

    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0, dtype=str)
    return pd.read_csv(path, dtype=str)


def _norm_id(x: Any) -> str:
    return str(x).strip().lower()


def _find_one(directory: Path, pattern: str | None, what: str) -> Path:
    if not pattern:
        raise DatasetConfigError(f"{what}.file is not configured.")
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise DatasetConfigError(f"{what}.file pattern '{pattern}' matched {len(matches)} files in {directory}; need exactly 1.")
    return matches[0]


def _fingerprint(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    return h.hexdigest()[:12]


class AphasiaDataset:
    """Loads and validates a labelled corpus into canonical ``DatasetRecord`` objects."""

    def __init__(self, config: DatasetConfig, protected_dirs: list[Path] | None = None):
        self.config = config
        self.protected_dirs = [Path(p).resolve() for p in (protected_dirs or [])]

    def _guard_root(self, root: Path) -> None:
        # Uploaded / microphone recordings live in the application data directory; they are never a dataset.
        for p in self.protected_dirs:
            if root == p or p in root.parents or root in p.parents:
                raise DatasetConfigError(
                    f"Dataset root {root} overlaps the application data directory {p}. Uploaded and microphone "
                    "recordings are technical validation samples and must never be used as training data.")

    def load(self) -> tuple[list[DatasetRecord], ValidationReport]:
        cfg = self.config
        root = cfg.resolve_root()
        self._guard_root(root)
        origin = SYNTHETIC if (root / SYNTHETIC_MARKER).exists() else REAL
        dirs = {d: root / d for d in LAYOUT_DIRS}
        for d in ("audio", "transcripts", "scores"):
            if not dirs[d].is_dir():
                raise DatasetNotAvailable(f"Required directory missing: {dirs[d]}")

        audio_files = [p for p in dirs["audio"].rglob("*") if p.suffix.lower() in cfg.audio_extensions]
        transcripts = sorted(dirs["transcripts"].rglob("*.cha"))
        version = cfg.version or f"unversioned-{_fingerprint(audio_files + transcripts)}"
        report = ValidationReport(dataset=cfg.name, dataset_version=version, data_origin=origin, root=str(root),
                                  total_audio_files=len(audio_files), total_transcripts=len(transcripts))
        if not cfg.version:
            report.notes.append("Dataset version not set in config; a file fingerprint is used instead.")
        if origin == SYNTHETIC:
            report.notes.append("SYNTHETIC_TEST_DATA: this dataset exists only to test software. It is not clinical data.")

        audio_by_stem: dict[str, list[Path]] = defaultdict(list)
        for p in audio_files:
            audio_by_stem[p.stem.lower()].append(p)
        scores = self._load_scores(dirs["scores"], report)
        demographics = self._load_demographics(dirs["demographics"])
        id_re = re.compile(cfg.file_id_regex)

        records: list[DatasetRecord] = []
        used_audio: set[Path] = set()
        seen: Counter = Counter()
        for tpath in transcripts:
            m = id_re.match(tpath.stem)
            if not m:
                report.unmatched_records.append(tpath.name)
                continue
            participant = _norm_id(m.group("participant"))
            session = (m.groupdict().get("session") or "").lower() or "1"
            chat = parse_chat(tpath, keep_fillers=cfg.keep_fillers)
            if chat.warnings:
                report.transcript_warnings[tpath.name] = chat.warnings[:20]
            audio = self._match_audio(tpath, chat, audio_by_stem)
            if audio:
                used_audio.add(audio)
            for task in (cfg.tasks or [None]):
                rec = self._build_record(chat, tpath, participant, session, task, audio, scores, demographics, origin)
                seen[(rec.participant_id, rec.session_id, rec.task)] += 1
                records.append(rec)

        for rec in records:
            key = (rec.participant_id, rec.session_id, rec.task)
            if seen[key] > 1:
                rec.issues.append("duplicate participant/session/task")
                report.duplicate_records.append(rec.record_id)
            if not rec.audio_path:
                report.missing_audio.append(rec.record_id)
            for issue in rec.issues:
                if issue == "missing WAB-R AQ":
                    report.missing_aq.append(rec.record_id)
                elif issue.startswith("invalid WAB-R AQ"):
                    report.invalid_aq.append(rec.record_id)
                elif issue.startswith("ambiguous WAB-R AQ"):
                    report.ambiguous_aq.append(rec.record_id)
        report.missing_transcripts = sorted(str(p.relative_to(root)) for p in audio_files if p not in used_audio)
        report.matched_records = sum(1 for r in records if r.audio_path and r.transcript_path)
        trainable = [r for r in records if r.trainable]
        report.trainable_records = len(trainable)
        report.trainable_participants = len({r.participant_id for r in trainable})
        report.band_distribution = dict(Counter(r.severity_band for r in trainable))
        return records, report

    # ---------------------------------------------------------------- helpers
    def _match_audio(self, tpath: Path, chat: ChatTranscript, audio_by_stem) -> Path | None:
        candidates = []
        if chat.media and chat.media.get("name"):
            candidates.append(chat.media["name"].lower())
        candidates.append(tpath.stem.lower())
        for c in candidates:
            if audio_by_stem.get(c):
                return sorted(audio_by_stem[c])[0]
        return None

    def _load_scores(self, directory: Path, report: ValidationReport) -> dict[tuple[str, str | None], list[str]]:
        sc = self.config.scores
        missing = [k for k in ("file", "participant_column", "aq_column") if not getattr(sc, k)]
        if missing:
            raise DatasetConfigError(
                f"scores.{', scores.'.join(missing)} must be set in the dataset config after inspecting the real "
                "WAB-R spreadsheet. Column names are not guessed.")
        path = _find_one(directory, sc.file, "scores")
        table = load_table(path, sc.sheet)
        for col in [sc.participant_column, sc.aq_column] + ([sc.session_column] if sc.session_column else []):
            if col not in table.columns:
                raise DatasetConfigError(f"Column '{col}' not found in {path.name}. Available: {list(table.columns)}")
        out: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        for _, row in table.iterrows():
            pid = row[sc.participant_column]
            if pid is None or str(pid).strip() in ("", "nan"):
                continue
            session = _norm_id(row[sc.session_column]) if sc.session_column else None
            aq = row[sc.aq_column]
            if aq is None or str(aq).strip().lower() in ("", "nan", "none"):
                continue  # an empty cell is a MISSING score (reported as such), not an invalid one
            out[(_norm_id(pid), session)].append(aq)
        report.notes.append(f"WAB-R AQ read from {path.name} column '{sc.aq_column}'.")
        return out

    def _load_demographics(self, directory: Path) -> dict[str, dict[str, Any]]:
        dc = self.config.demographics
        if not dc.file or not directory.is_dir():
            return {}
        path = _find_one(directory, dc.file, "demographics")
        table = load_table(path, dc.sheet)
        if dc.participant_column not in table.columns:
            raise DatasetConfigError(f"Column '{dc.participant_column}' not found in {path.name}.")
        return {_norm_id(r[dc.participant_column]): {k: (None if str(v) == "nan" else v) for k, v in r.items()}
                for _, r in table.iterrows()}

    def _build_record(self, chat: ChatTranscript, tpath: Path, participant: str, session: str, task: str | None,
                      audio: Path | None, scores, demographics, origin: str) -> DatasetRecord:
        cfg = self.config
        utts = chat.speaker_utterances(cfg.speaker, task)
        issues: list[str] = []
        if not audio:
            issues.append("missing audio")
        if not utts:
            issues.append(f"no '{cfg.speaker}' utterances" + (f" in task '{task}'" if task else ""))
        segments = [(u.start_ms, u.end_ms) for u in utts if u.start_ms is not None and u.end_ms is not None]
        if cfg.audio_scope == "speaker_utterances" and utts and not segments:
            issues.append("transcript has no time alignment; cannot isolate participant speech")

        raw = scores.get((participant, session if cfg.scores.session_column else None), [])
        wab_aq = severity = band = None
        if not raw:
            issues.append("missing WAB-R AQ")
        elif len({str(v).strip() for v in raw}) > 1:
            issues.append(f"ambiguous WAB-R AQ: {len(raw)} differing rows for this participant/session")
        else:
            try:
                wab_aq = tgt.validate_aq(raw[0])
                severity, band = tgt.severity_from_aq(wab_aq), tgt.band_from_aq(wab_aq)
            except tgt.InvalidAQ as exc:
                issues.append(f"invalid WAB-R AQ ({exc})")

        spk = chat.speaker_ids(cfg.speaker) or {}
        return DatasetRecord(
            record_id=f"{participant}/{session}/{task or 'all'}", participant_id=participant, session_id=session,
            task=task or "all", audio_path=str(audio) if audio else None, transcript_path=str(tpath),
            wab_aq=wab_aq, severity_score=severity, severity_band=band,
            target_source=tgt.TARGET_SOURCE, target_version=tgt.TARGET_VERSION, data_origin=origin,
            human_transcript=chat.normalized_text(cfg.speaker, task), speaker_segments_ms=segments,
            demographics=demographics.get(participant, {}),
            aphasia_metadata={k: spk.get(k) for k in ("group", "age", "sex", "custom") if spk.get(k)},
            provenance={"transcript": tpath.name, "media_header": (chat.media or {}).get("raw"),
                        "speaker": cfg.speaker, "task_gem": task, "score_rows": len(raw)},
            issues=issues,
        )


def save_validation(records: list[DatasetRecord], report: ValidationReport, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(json.dumps(asdict(report), indent=2, default=str), encoding="utf-8")
    (out_dir / "records.json").write_text(json.dumps([r.to_dict() for r in records], indent=2, default=str),
                                          encoding="utf-8")
    (out_dir / "validation_summary.txt").write_text(report.summary(), encoding="utf-8")
