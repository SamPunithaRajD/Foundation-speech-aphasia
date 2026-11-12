"""Handcrafted acoustic features (baseline feature group, comparable in spirit to the
handcrafted-feature front-ends used in the base paper; not a reimplementation of it).

Computed with librosa on the preprocessed 16 kHz mono waveform, 25 ms windows / 10 ms hop.
Values that cannot be computed (e.g. no voiced frames for F0) are returned with
``status="not_computed"`` and ``value=None`` — never estimated.
"""
from __future__ import annotations

import numpy as np

from app.schemas.analysis import ClinicalFeature

N_MFCC = 13
WIN, HOP = 400, 160  # 25 ms / 10 ms at 16 kHz
PROVENANCE = f"librosa, n_fft={WIN}, hop={HOP} (16 kHz)"


def _feat(name, label, unit, value, definition, provenance=PROVENANCE) -> ClinicalFeature:
    ok = value is not None and np.isfinite(value)
    return ClinicalFeature(name=name, label=label, unit=unit, value=round(float(value), 6) if ok else None,
                           definition=definition, source="audio", provenance=provenance,
                           status="computed" if ok else "not_computed",
                           reason=None if ok else "Not computable for this signal.")


def acoustic_feature_names() -> list[str]:
    names = [f"mfcc{i}_{s}" for i in range(1, N_MFCC + 1) for s in ("mean", "std")]
    return names + ["rms_db_mean", "rms_db_std", "zcr_mean", "zcr_std", "spectral_centroid_mean",
                    "spectral_centroid_std", "spectral_bandwidth_mean", "spectral_rolloff_mean",
                    "f0_mean_hz", "f0_std_semitones", "f0_range_semitones", "voiced_fraction"]


def acoustic_features(x: np.ndarray, sr: int) -> list[ClinicalFeature]:
    import librosa

    x = np.asarray(x, dtype=np.float32)
    out: list[ClinicalFeature] = []
    mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=N_MFCC, n_fft=WIN, hop_length=HOP)
    for i in range(N_MFCC):
        out.append(_feat(f"mfcc{i + 1}_mean", f"MFCC {i + 1} mean", "coef", mfcc[i].mean(), f"Mean of MFCC coefficient {i + 1}."))
        out.append(_feat(f"mfcc{i + 1}_std", f"MFCC {i + 1} std", "coef", mfcc[i].std(), f"Std of MFCC coefficient {i + 1}."))
    rms_db = librosa.amplitude_to_db(librosa.feature.rms(y=x, frame_length=WIN, hop_length=HOP)[0] + 1e-10)
    zcr = librosa.feature.zero_crossing_rate(x, frame_length=WIN, hop_length=HOP)[0]
    cent = librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=WIN, hop_length=HOP)[0]
    bw = librosa.feature.spectral_bandwidth(y=x, sr=sr, n_fft=WIN, hop_length=HOP)[0]
    roll = librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=WIN, hop_length=HOP)[0]
    out += [
        _feat("rms_db_mean", "Energy mean", "dB", rms_db.mean(), "Mean frame RMS energy."),
        _feat("rms_db_std", "Energy variability", "dB", rms_db.std(), "Std of frame RMS energy."),
        _feat("zcr_mean", "Zero-crossing rate mean", "ratio", zcr.mean(), "Mean zero-crossing rate."),
        _feat("zcr_std", "Zero-crossing rate std", "ratio", zcr.std(), "Std of zero-crossing rate."),
        _feat("spectral_centroid_mean", "Spectral centroid mean", "Hz", cent.mean(), "Mean spectral centroid."),
        _feat("spectral_centroid_std", "Spectral centroid std", "Hz", cent.std(), "Std of spectral centroid."),
        _feat("spectral_bandwidth_mean", "Spectral bandwidth mean", "Hz", bw.mean(), "Mean spectral bandwidth."),
        _feat("spectral_rolloff_mean", "Spectral roll-off mean", "Hz", roll.mean(), "Mean 85% spectral roll-off."),
    ]
    f0, voiced, _ = librosa.pyin(x, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=HOP)
    f0v = f0[voiced & np.isfinite(f0)] if f0 is not None else np.array([])
    pyin_prov = "librosa.pyin fmin=60 fmax=400 Hz, frame=1024, hop=160"
    if len(f0v) >= 3:
        st = 12 * np.log2(f0v / np.median(f0v))
        vals = (f0v.mean(), st.std(), np.percentile(st, 95) - np.percentile(st, 5))
    else:
        vals = (None, None, None)
    out += [
        _feat("f0_mean_hz", "F0 mean", "Hz", vals[0], "Mean fundamental frequency of voiced frames.", pyin_prov),
        _feat("f0_std_semitones", "F0 variability", "semitones", vals[1], "Std of F0 in semitones re. median.", pyin_prov),
        _feat("f0_range_semitones", "F0 range", "semitones", vals[2], "5th–95th percentile F0 range (semitones).", pyin_prov),
        _feat("voiced_fraction", "Voiced fraction", "ratio", float(voiced.mean()) if voiced is not None else None,
              "Fraction of frames classified as voiced by pYIN.", pyin_prov),
    ]
    assert [f.name for f in out] == acoustic_feature_names()
    return out
