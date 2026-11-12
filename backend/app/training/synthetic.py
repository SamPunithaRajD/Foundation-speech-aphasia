"""Generate a SYNTHETIC_TEST_DATA dataset for software tests only.

    python -m app.training.synthetic --out <dir> [--participants 12]

Audio is generated harmonic tones, transcripts are template sentences, and WAB-R AQ values are
random numbers. The root contains a SYNTHETIC_TEST_DATA marker file, which makes the loader tag
every record SYNTHETIC_TEST_DATA, forces training into artifacts/synthetic_test/, and makes the
resulting bundle unloadable for predictions. None of this is clinical data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from app.training.dataset import SYNTHETIC_MARKER

WORDS = "the boy went to the store and he bought some bread for his mother then he walked home".split()


def _tone_audio(seconds: float, f0: float, rng) -> np.ndarray:
    sr = 16000
    t = np.arange(int(seconds * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 10))
    env = (np.sin(2 * np.pi * 1.5 * t) > -0.3).astype(float)  # gaps for VAD pauses
    return (0.2 * x * env + 0.002 * rng.standard_normal(len(t))).astype(np.float32)


def generate(out: Path, participants: int = 12, sessions: int = 2, seed: int = 0, seconds: float = 4.0) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out)
    for d in ("audio", "transcripts", "scores", "demographics", "metadata"):
        (out / d).mkdir(parents=True, exist_ok=True)
    (out / SYNTHETIC_MARKER).write_text("SYNTHETIC_TEST_DATA — generated for software tests. Not clinical data.\n")
    rows = ["participant,wab_aq_synthetic"]
    for p in range(participants):
        pid = f"synth{p:02d}"
        aq = float(np.round(rng.uniform(10, 99), 1))
        rows.append(f"{pid},{aq}")
        for s in "ab"[:sessions]:
            stem = f"{pid}{s}"
            sf.write(out / "audio" / f"{stem}.wav", _tone_audio(seconds, 110 + 3 * p, rng), 16000)
            n = int(rng.integers(6, len(WORDS)))
            text = " ".join(WORDS[:n])
            half = int(seconds * 500)
            (out / "transcripts" / f"{stem}.cha").write_text(
                "@UTF8\n@Begin\n@Languages:\teng\n@Participants:\tPAR Participant, INV Investigator\n"
                f"@ID:\teng|SYNTHETIC|PAR|||SYNTHETIC||Participant|||\n@Media:\t{stem}, audio\n@G:\tStory\n"
                f"*PAR:\t{text} &-uh <the the> [/] end . \x150_{half}\x15\n%mor:\tsynthetic\n"
                f"*INV:\tokay . \x15{half}_{half + 300}\x15\n"
                f"*PAR:\txxx more words (be)cause . \x15{half + 300}_{int(seconds * 1000)}\x15\n@End\n",
                encoding="utf-8")
    (out / "scores" / "synthetic_scores.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--participants", type=int, default=12)
    a = ap.parse_args(argv)
    print(f"SYNTHETIC_TEST_DATA written to {generate(a.out, a.participants)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
