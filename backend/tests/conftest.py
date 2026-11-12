"""Test fixtures. All audio is generated programmatically; no recordings of people are used."""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def voiced_bursts(sr: int = 16000, bursts: int = 4, burst_s: float = 0.8, gap_s: float = 0.6, lead_s: float = 0.3) -> np.ndarray:
    """Harmonic 'voiced-like' bursts (glottal-style harmonics + formant-ish AM) separated by silence."""
    rng = np.random.default_rng(0)
    parts = [np.zeros(int(lead_s * sr))]
    for i in range(bursts):
        t = np.arange(int(burst_s * sr)) / sr
        f0 = 120 + 15 * i
        tone = sum((1 / k) * np.sin(2 * np.pi * f0 * k * t) for k in range(1, 15))
        env = np.sin(np.pi * t / burst_s) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
        parts.append(0.3 * tone * env + 0.003 * rng.standard_normal(len(t)))
        parts.append(np.zeros(int(gap_s * sr)))
    return np.concatenate(parts).astype(np.float32)


def wav_bytes(x: np.ndarray, sr: int = 16000, channels: int = 1, fmt: str = "WAV", subtype: str | None = "PCM_16") -> bytes:
    data = np.stack([x] * channels, axis=1) if channels > 1 else x
    buf = io.BytesIO()
    sf.write(buf, data, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # tests never depend on the local backend/.env
        data_dir=tmp_path / "data",
        temp_dir=tmp_path / "tmp",
        severity_model_dir=tmp_path / "no-severity-model",
        # Fast defaults for the test-suite: models that are not installed must be reported honestly.
        asr_enabled=False,
        default_embedding_models=["wav2vec2"],
        wav2vec2_model_id="nonexistent-org/not-installed-model",
        allow_model_download=False,
        max_upload_mb=1,
        max_duration_seconds=30,
    )


@pytest.fixture
def client(settings: Settings):
    with TestClient(create_app(settings)) as c:
        yield c


def wait_for_terminal(client: TestClient, analysis_id: str, timeout: float = 120) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/analysis/{analysis_id}/status").json()
        if body["status"] in ("COMPLETED", "ATTENTION_NEEDED", "FAILED", "CANCELLED"):
            return body
        time.sleep(0.1)
    raise AssertionError("analysis did not finish in time")
