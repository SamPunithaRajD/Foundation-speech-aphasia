"""API tests: upload, validation, status, results, models, history, reports."""
from __future__ import annotations

import numpy as np

from conftest import voiced_bursts, wait_for_terminal, wav_bytes

from app.config import Settings


def upload(client, data: bytes, name: str = "sample.wav", **form):
    return client.post("/api/analysis", files={"file": (name, data, "application/octet-stream")}, data=form)


def run_analysis(client) -> str:
    r = upload(client, wav_bytes(voiced_bursts()))
    assert r.status_code == 202, r.text
    aid = r.json()["analysis_id"]
    wait_for_terminal(client, aid)
    return aid


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["project"] == "FoundationSpeech-Aphasia" and body["simulated"] is False


def test_models_report_honest_status(client):
    body = client.get("/api/models").json()
    by_key = {m["key"]: m for m in body["embedding_models"]}
    assert set(by_key) == {"wav2vec2", "hubert", "whisper_encoder", "wavlm"}
    wav2vec = by_key["wav2vec2"]
    assert wav2vec["status"] == "unavailable" and "not installed" in wav2vec["reason"]
    assert body["severity_classifier"]["prediction_available"] is False
    assert body["severity_classifier"]["state"] == "MODEL_NOT_READY"
    assert body["explainability"]["status"] == "not_available"
    assert "device" in body["hardware"]


def test_upload_and_full_lifecycle(client):
    r = upload(client, wav_bytes(voiced_bursts()))
    assert r.status_code == 202
    created = r.json()
    assert created["status"] == "QUEUED" and created["analysis_id"].startswith("FS-")
    status = wait_for_terminal(client, created["analysis_id"])
    # the configured embedding model is not installed -> job finishes but needs attention
    assert status["status"] == "ATTENTION_NEEDED"
    assert status["progress"]["percent"] == 100
    names = [s["name"] for s in status["stages"]]
    assert names == ["PREPROCESSING", "MODEL_PROCESSING", "ASR", "FEATURE_EXTRACTION", "FUSION", "PREDICTION",
                     "EXPLAINABILITY"]

    full = client.get(f"/api/analysis/{created['analysis_id']}").json()
    assert full["mode"] == "live" and full["simulated"] is False
    assert full["audio"]["filename"] == "speech-sample.wav"  # original filename not stored by default
    assert full["preprocessing"]["state"] == "COMPUTED" and full["preprocessing"]["final_sample_rate"] == 16000
    assert full["preprocessing"]["segment_count"] == 4
    assert full["embeddings"]["results"][0]["state"] == "MODEL_UNAVAILABLE"
    assert full["transcript"]["state"] == "SKIPPED"


def test_section_endpoints(client):
    aid = run_analysis(client)
    assert client.get(f"/api/analysis/{aid}/transcript").json()["state"] == "SKIPPED"
    features = client.get(f"/api/analysis/{aid}/features").json()
    assert features["state"] == "PARTIAL"
    for f in features["features"]:
        assert {"name", "value", "unit", "definition", "source", "provenance", "status"} <= set(f)
        assert f["clinically_validated"] is False
        assert (f["value"] is None) == (f["status"] != "computed")
    severity = client.get(f"/api/analysis/{aid}/severity").json()
    assert severity == {**severity, "state": "MODEL_NOT_READY", "prediction_available": False, "severity_score": None}
    assert "probabilities" not in severity
    assert client.get(f"/api/analysis/{aid}/explainability").json()["state"] == "NOT_AVAILABLE"


def test_history(client):
    aid = run_analysis(client)
    items = client.get("/api/history").json()
    assert items["total"] == 1 and items["items"][0]["analysis_id"] == aid
    assert items["items"][0]["severity_score"] is None and items["items"][0]["severity_state"] == "MODEL_NOT_READY"
    assert client.get(f"/api/history/{aid}").json()["analysis_id"] == aid


def test_report_pdf(client):
    aid = run_analysis(client)
    r = client.get(f"/api/reports/{aid}")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")
    assert client.get(f"/api/analysis/{aid}").json()["report"]["size_bytes"] == len(r.content)


def test_unknown_analysis_is_structured_404(client):
    for path in ("/api/analysis/FS-NOPE", "/api/analysis/FS-NOPE/status", "/api/reports/FS-NOPE", "/api/history/FS-NOPE"):
        r = client.get(path)
        assert r.status_code == 404 and r.json()["error"]["code"] == "ANALYSIS_NOT_FOUND"


def test_serves_built_frontend_same_origin(tmp_path, settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root></div>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    (tmp_path / "secret.txt").write_text("outside dist")
    settings.frontend_dist = dist
    with TestClient(create_app(settings)) as c:
        assert c.get("/").text == "<div id=root></div>"
        assert c.get("/assets/app.js").text == "console.log(1)"
        assert c.get("/some/client/route").text == "<div id=root></div>"  # SPA fallback
        assert "outside dist" not in c.get("/../secret.txt").text  # no traversal
        r = c.get("/api/does-not-exist")
        assert r.status_code == 404 and r.json()["error"]["message"] == "Unknown API endpoint."
        assert c.get("/api/health").json()["status"] == "ok"
        assert c.head("/").status_code == 200  # uptime checks / link previews probe with HEAD


def test_records_are_kept_by_default(tmp_path):
    """Retention must be opt-in: the default never deletes history at startup."""
    assert Settings(_env_file=None).retention_days == 0
    from app.database.repository import AnalysisRepository
    repo = AnalysisRepository(tmp_path / "r.sqlite")
    try:
        assert repo.purge_older_than(0) == 0
    finally:
        repo.close()


def test_unsupported_format(client):
    r = upload(client, b"hello", name="notes.txt")
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_FORMAT"


def test_oversized_file(client):
    big = wav_bytes(np.zeros(16000 * 40, dtype=np.float32) + 0.1)  # > 1 MB test limit
    r = upload(client, big)
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_corrupt_file(client):
    r = upload(client, b"RIFF\x00\x00\x00\x00WAVEgarbage" * 20)
    assert r.status_code == 422 and r.json()["error"]["code"] == "AUDIO_DECODE_ERROR"


def test_empty_file(client):
    r = upload(client, b"")
    assert r.status_code == 400 and r.json()["error"]["code"] == "INVALID_AUDIO"


def test_too_short_and_silent(client):
    assert upload(client, wav_bytes(voiced_bursts()[:1000])).json()["error"]["code"] == "AUDIO_TOO_SHORT"
    assert upload(client, wav_bytes(np.zeros(16000 * 2, dtype=np.float32))).json()["error"]["code"] == "INVALID_AUDIO"


def test_too_long(client, settings):
    settings.max_duration_seconds = 2
    r = upload(client, wav_bytes(voiced_bursts()))
    assert r.status_code == 422 and r.json()["error"]["code"] == "AUDIO_TOO_LONG"


def test_invalid_strategy_and_model(client):
    assert upload(client, wav_bytes(voiced_bursts()), strategy="all").status_code == 422
    r = upload(client, wav_bytes(voiced_bursts()), models="gpt")
    assert r.status_code == 422 and r.json()["error"]["code"] == "INVALID_REQUEST"


def test_no_temp_files_left(client, settings):
    run_analysis(client)
    assert list(settings.temp_dir.iterdir()) == []


def test_report_requires_finished_analysis(client, settings):
    from app.schemas.common import AnalysisStatus

    aid = run_analysis(client)
    repo = client.app.state.services.repo
    rec = repo.get(aid)
    rec.status = AnalysisStatus.ASR
    repo.save(rec)
    r = client.get(f"/api/reports/{aid}")
    assert r.status_code == 409 and r.json()["error"]["code"] == "ANALYSIS_NOT_FINISHED"


def test_cancel_finished_is_noop(client):
    aid = run_analysis(client)
    r = client.post(f"/api/analysis/{aid}/cancel")
    assert r.status_code == 200 and r.json()["status"] == "ATTENTION_NEEDED"
