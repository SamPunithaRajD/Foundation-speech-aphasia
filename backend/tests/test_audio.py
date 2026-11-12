"""Audio decoding, validation and preprocessing."""
from __future__ import annotations

import io

import av
import numpy as np
import pytest
import soundfile as sf

from app.preprocessing.ingest import decode_audio, validate_decoded, validate_extension
from app.preprocessing.pipeline import preprocess
from app.utils.errors import ApiError
from conftest import voiced_bursts, wav_bytes


def write(tmp_path, name, data: bytes):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_decode_stereo_44k_wav_and_preprocess(tmp_path, settings):
    x44 = voiced_bursts(sr=44100)
    audio = decode_audio(write(tmp_path, "a.wav", wav_bytes(x44, sr=44100, channels=2)))
    assert audio.channels == 2 and audio.sample_rate == 44100 and audio.decoder == "soundfile"
    pre = preprocess(audio, settings)
    r = pre.result
    assert r.original_sample_rate == 44100 and r.final_sample_rate == 16000 and r.original_channels == 2
    assert pre.waveform.ndim == 1 and abs(len(pre.waveform) / 16000 - r.duration_seconds) < 1e-3
    assert "mono_downmix" in r.steps and "resample_44100_to_16000_polyphase" in r.steps
    assert np.isclose(np.max(np.abs(pre.waveform)), 10 ** (-1 / 20), atol=1e-3)
    assert r.segment_count == 4 and 0 < r.speech_ratio < 1
    assert "noise_reduction" in r.steps_not_applied


def test_decode_flac_and_mp3(tmp_path):
    x = voiced_bursts()
    flac = write(tmp_path, "a.flac", wav_bytes(x, fmt="FLAC"))
    assert decode_audio(flac).sample_rate == 16000
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="MP3")
    mp3 = decode_audio(write(tmp_path, "a.mp3", buf.getvalue()))
    assert abs(mp3.duration_seconds - len(x) / 16000) < 0.15


def test_decode_m4a_via_pyav(tmp_path):
    x = voiced_bursts()
    path = tmp_path / "a.m4a"
    with av.open(str(path), "w", format="mp4") as out:
        stream = out.add_stream("aac", rate=16000)
        stream.layout = "mono"
        frame_len = 1024
        pcm = (x * 32767).astype(np.int16)
        for i in range(0, len(pcm), frame_len):
            frame = av.AudioFrame.from_ndarray(pcm[None, i:i + frame_len], format="s16", layout="mono")
            frame.sample_rate = 16000
            for pkt in stream.encode(frame):
                out.mux(pkt)
        for pkt in stream.encode(None):
            out.mux(pkt)
    audio = decode_audio(path)
    assert audio.decoder == "pyav" and abs(audio.duration_seconds - len(x) / 16000) < 0.2


def test_corrupt_audio_raises(tmp_path):
    with pytest.raises(ApiError) as e:
        decode_audio(write(tmp_path, "bad.wav", b"not audio at all" * 100))
    assert e.value.code.value == "AUDIO_DECODE_ERROR"


def test_extension_validation(settings):
    assert validate_extension("x.WAV", settings) == ".wav"
    with pytest.raises(ApiError):
        validate_extension("x.exe", settings)


def test_silence_rejected(tmp_path, settings):
    audio = decode_audio(write(tmp_path, "s.wav", wav_bytes(np.zeros(32000, dtype=np.float32))))
    with pytest.raises(ApiError) as e:
        validate_decoded(audio, settings)
    assert e.value.code.value == "INVALID_AUDIO"


def test_noise_only_has_no_confident_speech(tmp_path, settings):
    rng = np.random.default_rng(1)
    audio = decode_audio(write(tmp_path, "n.wav", wav_bytes((0.0005 * rng.standard_normal(32000)).astype(np.float32))))
    settings.normalize_peak_dbfs = -40  # keep low-level noise quiet
    pre = preprocess(audio, settings)
    assert pre.result.speech_duration_seconds < 0.5 * pre.result.duration_seconds


def test_highpass_optional(tmp_path, settings):
    audio = decode_audio(write(tmp_path, "a.wav", wav_bytes(voiced_bursts())))
    assert "highpass_filter" in preprocess(audio, settings).result.steps_not_applied
    settings.highpass_hz = 80
    assert any(s.startswith("highpass_") for s in preprocess(audio, settings).result.steps)
