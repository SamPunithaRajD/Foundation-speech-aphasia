"""Explicit, opt-in download of model weights into the local caches.

    python scripts/download_models.py whisper:tiny wav2vec2

Keys: wav2vec2, hubert, wavlm (Hugging Face IDs from settings) or whisper:<name>.
The API itself never downloads unless FSA_ALLOW_MODEL_DOWNLOAD=true.
Approximate sizes: whisper tiny 72 MB, base 139 MB; wav2vec2-base ~380 MB; hubert-base ~380 MB;
wavlm-base-plus ~380 MB.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.models.registry import SPEC_BY_KEY  # noqa: E402


def main(keys: list[str]) -> int:
    if not keys:
        print(__doc__)
        return 1
    s = get_settings()
    for key in keys:
        if key.startswith("whisper:"):
            import whisper

            name = key.split(":", 1)[1]
            whisper._download(whisper._MODELS[name], str(s.whisper_cache_dir or Path.home() / ".cache" / "whisper"), False)
            print(f"whisper {name}: ok")
        elif key in SPEC_BY_KEY and SPEC_BY_KEY[key].family == "hf":
            from huggingface_hub import list_repo_files, snapshot_download

            mid = getattr(s, SPEC_BY_KEY[key].model_id_setting)
            # prefer safetensors; many speech checkpoints (e.g. wav2vec2-base, hubert, wavlm) only ship pytorch_model.bin
            weights = "model.safetensors" if "model.safetensors" in list_repo_files(mid) else "pytorch_model.bin"
            snapshot_download(mid, cache_dir=str(s.hf_cache_dir) if s.hf_cache_dir else None,
                              allow_patterns=["*.json", "*.txt", weights])
            print(f"{key} ({mid}): ok")
        else:
            print(f"unknown key: {key}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
