"""ASR parsing, feature extraction, embedding extraction and unavailable-model behaviour."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.config import Settings
from app.features.clinical import extract_features, mattr
from app.models.registry import ModelRegistry, ModelUnavailable, whisper_checkpoint_path
from app.pipelines import asr
from app.schemas.analysis import PreprocessingResult, SpeechSegment, TranscriptResult, TranscriptSegment, TranscriptWord
from app.schemas.common import SectionState
from conftest import voiced_bursts


def make_transcript(words: list[str], text: str | None = None) -> TranscriptResult:
    ws = [TranscriptWord(word=w, start=i * 0.5, end=i * 0.5 + 0.4, probability=0.9) for i, w in enumerate(words)]
    return TranscriptResult(state=SectionState.COMPUTED, text=text or " ".join(words),
                            segments=[TranscriptSegment(id=0, start=0, end=len(words) * 0.5, text=text or " ".join(words),
                                                        words=ws)])


def make_pre(segments=((0.0, 2.0), (3.0, 5.0), (5.1, 6.0)), duration=6.0) -> PreprocessingResult:
    segs = [SpeechSegment(start=s, end=e) for s, e in segments]
    return PreprocessingResult(state=SectionState.COMPUTED, duration_seconds=duration, segments=segs,
                               speech_duration_seconds=sum(e - s for s, e in segments), speech_ratio=0.8,
                               vad={"engine": "webrtcvad"})


# ---------------------------------------------------------------- features
def test_transcript_features_values(settings):
    words = "the the cat um sat on the mat. it was uh happy".split()
    feats = {f.name: f for f in extract_features(make_transcript(words, " ".join(words)), make_pre(), settings).features}
    n = len(words)  # 11 tokens after tokenisation ("mat." -> "mat")
    assert feats["word_count"].value == n
    assert feats["type_token_ratio"].value == pytest.approx(len(set(w.strip(".") for w in words)) / n, abs=1e-4)
    assert feats["immediate_repetition_rate"].value == pytest.approx(1 / n * 100, abs=1e-3)
    assert feats["filled_pause_rate"].value == pytest.approx(2 / n * 100, abs=1e-3)
    assert feats["disfluency_rate"].value == pytest.approx(3 / n * 100, abs=1e-3)
    assert feats["speech_rate"].value == pytest.approx(n / 6.0 * 60, abs=1e-3)
    assert feats["articulation_rate"].value == pytest.approx(n / 4.9, abs=1e-3)
    assert feats["mean_sentence_length"].value == pytest.approx((8 + 4) / 2)
    assert feats["moving_average_ttr"].status == "insufficient_data"  # 11 < window 25
    assert feats["syntactic_complexity"].status == "not_computed" and feats["syntactic_complexity"].value is None
    assert all(not f.clinically_validated for f in feats.values())


def test_pause_features_from_vad(settings):
    feats = {f.name: f for f in extract_features(None, make_pre(), settings).features}
    # gaps: 1.0 s (counted) and 0.1 s (below 300 ms threshold)
    assert feats["pause_frequency"].value == pytest.approx(1 / (6 / 60))
    assert feats["mean_pause_duration"].value == pytest.approx(1.0)
    assert feats["type_token_ratio"].status == "not_computed"


def test_features_without_transcript_are_partial(settings):
    tr = TranscriptResult(state=SectionState.MODEL_UNAVAILABLE, message="weights missing")
    res = extract_features(tr, make_pre(), settings)
    assert res.state == SectionState.PARTIAL
    assert all(f.value is None for f in res.features if f.source == "transcript")


def test_mattr():
    assert mattr(["a", "b", "a", "b"], 2) == 1.0
    assert mattr(["a", "a", "a"], 2) == 0.5


# ---------------------------------------------------------------- ASR
class FakeWhisperModel:
    is_multilingual = False
    device = SimpleNamespace(type="cpu")


def test_asr_parsing_uses_real_word_probabilities(monkeypatch, settings):
    import whisper

    out = {"segments": [{"start": 0.0, "end": 1.2, "text": " Hello there.", "avg_logprob": -0.3, "no_speech_prob": 0.01,
                         "words": [{"word": " Hello", "start": 0.0, "end": 0.5, "probability": 0.8},
                                   {"word": " there.", "start": 0.6, "end": 1.2, "probability": 0.6}]}]}
    monkeypatch.setattr(whisper, "transcribe", lambda *a, **k: out)
    reg = SimpleNamespace(load_whisper=lambda name: FakeWhisperModel())
    res = asr.transcribe(reg, np.zeros(16000, dtype=np.float32), 16000, settings)
    assert res.state == SectionState.COMPUTED and res.text == "Hello there." and res.word_count == 2
    assert res.segments[0].mean_word_probability == pytest.approx(0.7)
    assert res.confidence["calibrated"] is False and res.confidence["mean_word_probability"] == pytest.approx(0.7)


def test_asr_without_probabilities_reports_no_confidence(monkeypatch, settings):
    import whisper

    out = {"segments": [{"start": 0.0, "end": 1.0, "text": " Hi", "words": []}]}
    monkeypatch.setattr(whisper, "transcribe", lambda *a, **k: out)
    reg = SimpleNamespace(load_whisper=lambda name: FakeWhisperModel())
    res = asr.transcribe(reg, np.zeros(16000, dtype=np.float32), 16000, settings)
    assert res.confidence is None and res.segments[0].mean_word_probability is None


def test_asr_unavailable_model_raises(settings):
    settings.asr_model = "tiny"
    settings.whisper_cache_dir = settings.data_dir / "empty-whisper-cache"
    with pytest.raises(ModelUnavailable):
        ModelRegistry(settings).load_whisper("tiny")


_TINY = whisper_checkpoint_path("tiny", Settings())
whisper_installed = pytest.mark.skipif(not (_TINY and _TINY.exists()), reason="Whisper tiny weights not installed")


@whisper_installed
def test_real_whisper_asr_runs(settings):
    settings.asr_enabled = True
    res = asr.transcribe(ModelRegistry(settings), voiced_bursts(), 16000, settings)
    assert res.state in (SectionState.COMPUTED, SectionState.INSUFFICIENT_DATA)
    assert res.engine == "openai-whisper" and res.language


# ---------------------------------------------------------------- embeddings
def test_unavailable_hf_model_reported(settings):
    reg = ModelRegistry(settings)
    info = next(m for m in reg.list() if m["key"] == "wav2vec2")
    assert info["status"] == "unavailable" and info["reason"]
    with pytest.raises(ModelUnavailable):
        reg.extract_embedding("wav2vec2", voiced_bursts(), 16000)


def test_hf_embedding_code_path_with_local_model(tmp_path, settings):
    """Exercise the HF extraction path with a tiny randomly initialised model saved locally (no download).
    This validates plumbing only; the values carry no meaning."""
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    cfg = Wav2Vec2Config(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64,
                         conv_dim=(16, 16), conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=16,
                         num_conv_pos_embedding_groups=2)
    d = tmp_path / "tiny-w2v"
    Wav2Vec2Model(cfg).save_pretrained(d)
    Wav2Vec2FeatureExtractor(sampling_rate=16000).save_pretrained(d)
    settings.wav2vec2_model_id = str(d)
    reg = ModelRegistry(settings)
    assert next(m for m in reg.list() if m["key"] == "wav2vec2")["status"] == "available"
    out = reg.extract_embedding("wav2vec2", voiced_bursts(), 16000)
    assert out["dimension"] == 32 and out["frames"] > 0 and out["vector"].shape == (32,)
    assert next(m for m in reg.list() if m["key"] == "wav2vec2")["status"] == "ready"


def test_registry_forces_hub_offline_when_downloads_disabled(settings):
    import os

    ModelRegistry(settings)
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"


@pytest.mark.parametrize("key,setting", [("wav2vec2", "wav2vec2_model_id"), ("hubert", "hubert_model_id"),
                                         ("wavlm", "wavlm_model_id")])
def test_real_installed_hf_models(key, setting, settings):
    """Runs only where the real pretrained weights are cached locally; checks plumbing, not quality."""
    from app.models.registry import hf_weights_cached

    mid = getattr(Settings(), setting)
    if not hf_weights_cached(mid, settings):
        pytest.skip(f"{mid} weights not installed")
    setattr(settings, setting, mid)
    out = ModelRegistry(settings).extract_embedding(key, voiced_bursts(), 16000)
    assert out["dimension"] == 768 and out["vector"].shape == (768,) and np.isfinite(out["vector"]).all()


@whisper_installed
def test_real_whisper_encoder_embedding(settings):
    out = ModelRegistry(settings).extract_embedding("whisper_encoder", voiced_bursts(), 16000)
    assert out["dimension"] == 384 and out["frames"] == int(np.ceil(len(voiced_bursts()) / 16000 * 50))
