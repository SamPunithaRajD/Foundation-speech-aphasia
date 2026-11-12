"""SQLite persistence for analysis records.

Only de-identified prototype data is stored: an opaque analysis ID, timestamps,
status, audio technical metadata, model metadata and computed results. Raw audio
is never written to the database. Original filenames are only kept when
``FSA_STORE_ORIGINAL_FILENAME=true``.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.schemas.analysis import AnalysisRecord
from app.utils.errors import ApiError, ErrorCode

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id  TEXT PRIMARY KEY,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    status       TEXT NOT NULL,
    record_json  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
"""


class AnalysisRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def _execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        try:
            with self._lock:
                cur = self._conn.execute(sql, params)
                self._conn.commit()
                return cur
        except sqlite3.Error as exc:
            raise ApiError(ErrorCode.DATABASE_ERROR, "Database operation failed.", 500) from exc

    def save(self, record: AnalysisRecord) -> None:
        self._execute(
            "INSERT INTO analyses(analysis_id, created_at, updated_at, status, record_json) VALUES (?,?,?,?,?) "
            "ON CONFLICT(analysis_id) DO UPDATE SET updated_at=excluded.updated_at, status=excluded.status, "
            "record_json=excluded.record_json",
            (record.analysis_id, record.created_at, record.updated_at, record.status.value, record.model_dump_json()),
        )

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        with self._lock:
            row = self._execute("SELECT record_json FROM analyses WHERE analysis_id=?", (analysis_id,)).fetchone()
        return AnalysisRecord.model_validate(json.loads(row["record_json"])) if row else None

    def require(self, analysis_id: str) -> AnalysisRecord:
        record = self.get(analysis_id)
        if record is None:
            raise ApiError(ErrorCode.ANALYSIS_NOT_FOUND, "No analysis exists with this ID.", 404)
        return record

    def list(self, limit: int = 100, offset: int = 0) -> tuple[list[AnalysisRecord], int]:
        with self._lock:
            rows = self._execute(
                "SELECT record_json FROM analyses ORDER BY created_at DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()
            total = self._execute("SELECT COUNT(*) AS n FROM analyses").fetchone()["n"]
        return [AnalysisRecord.model_validate(json.loads(r["record_json"])) for r in rows], total

    def purge_older_than(self, days: int) -> int:
        if days <= 0:
            return 0
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        return self._execute("DELETE FROM analyses WHERE created_at < ?", (cutoff,)).rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
