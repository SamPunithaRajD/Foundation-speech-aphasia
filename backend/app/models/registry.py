"""Speech foundation model registry.

Models are never downloaded unless ``FSA_ALLOW_MODEL_DOWNLOAD=true``. Availability
is determined by checking the local weight caches, and loading is lazy (first
use). Status values:

* ``unavailable`` – library missing, torch broken, or weights not cached (reason given)
* ``available``   – weights present locally, not loaded yet
* ``loading``     – currently loading into memory
* ``ready``       – loaded and usable
* ``error``       – loading failed (reason given)
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.config import Settings
from app.models.hardware import library_available, resolve_device, torch_status

log = logging.getLogger(__name__)

WHISPER_DIMS = {"tiny": 384, "tiny.en": 384, "base": 512, "base.en": 512, "small": 768, "small.en": 768,
                "medium": 1024, "medium.en": 1024, "large": 1280, "large-v1": 1280, "large-v2": 1280,
                "large-v3": 1280, "turbo": 1280, "large-v3-turbo": 1280}


@dataclass
class ModelSpec:
    key: str
    display_name: str
    family: str  # "hf" | "whisper"
    tag: str
    description: str
    model_id_setting: str
    expected_dimension: int | None


SPECS: list[ModelSpec] = [
    ModelSpec("wav2vec2", "wav2vec 2.0", "hf", "Self-supervised",
              "Contextual speech representations learned from unlabeled audio.", "wav2vec2_model_id", 768),
    ModelSpec("hubert", "HuBERT", "hf", "Masked prediction",
              "Hidden-unit discovery for robust acoustic feature learning.", "hubert_model_id", 768),
    ModelSpec("whisper_encoder", "Whisper Encoder", "whisper", "Multilingual",
              "Encoder representations from a large-scale weakly supervised speech model.", "whisper_encoder_model", None),
    ModelSpec("wavlm", "WavLM", "hf", "Speech understanding",
              "Representations designed for varied speech tasks and conditions.", "wavlm_model_id", 768),
]
SPEC_BY_KEY = {s.key: s for s in SPECS}


def whisper_checkpoint_path(name: str, settings: Settings) -> Path | None:
    """Local path of an openai-whisper checkpoint, or None if unknown name / not installed."""
    if not library_available("whisper"):
        return None
    import whisper

    if name not in whisper._MODELS:
        p = Path(name)
        return p if p.is_file() else None
    root = settings.whisper_cache_dir or Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    return Path(root) / os.path.basename(whisper._MODELS[name])


def hf_weights_cached(model_id: str, settings: Settings) -> bool:
    if Path(model_id).is_dir():
        return any((Path(model_id) / f).exists() for f in ("model.safetensors", "pytorch_model.bin"))
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    cache_dir = str(settings.hf_cache_dir) if settings.hf_cache_dir else None
    has_config = isinstance(try_to_load_from_cache(model_id, "config.json", cache_dir=cache_dir), str)
    has_weights = any(
        isinstance(try_to_load_from_cache(model_id, f, cache_dir=cache_dir), str)
        for f in ("model.safetensors", "pytorch_model.bin")
    )
    return has_config and has_weights


class ModelUnavailable(Exception):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class ModelRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings
        if not settings.allow_model_download:
            # local_files_only alone is not enough: transformers also starts a background Hub check
            # (safetensors auto-conversion) when loading .bin checkpoints. Keep inference fully offline.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                import huggingface_hub.constants as hf_constants

                hf_constants.HF_HUB_OFFLINE = True  # in case huggingface_hub was imported before this point
            except Exception:
                pass
        self._lock = threading.RLock()
        self._loaded: dict[str, Any] = {}  # cache key -> loaded object
        self._state: dict[str, tuple[str, str | None]] = {}  # cache key -> (status, reason)

    # ----- availability -------------------------------------------------
    def model_id(self, spec: ModelSpec) -> str:
        return str(getattr(self.settings, spec.model_id_setting))

    def _cache_key(self, spec: ModelSpec) -> str:
        return f"{spec.family}:{self.model_id(spec)}"

    def availability(self, spec: ModelSpec) -> tuple[str, str | None]:
        key = self._cache_key(spec)
        with self._lock:
            if key in self._state and self._state[key][0] in ("ready", "loading", "error"):
                return self._state[key]
        ok, reason = torch_status()
        if not ok:
            return "unavailable", reason
        mid = self.model_id(spec)
        if spec.family == "whisper":
            if not library_available("whisper"):
                return "unavailable", "The openai-whisper package is not installed."
            path = whisper_checkpoint_path(mid, self.settings)
            if path is None:
                return "unavailable", f"Unknown Whisper checkpoint '{mid}'."
            if not path.exists() and not self.settings.allow_model_download:
                return "unavailable", f"Whisper '{mid}' weights are not installed and downloads are disabled."
            return "available", None if path.exists() else "Weights will be downloaded on first use."
        if not library_available("transformers"):
            return "unavailable", "The transformers package is not installed."
        if hf_weights_cached(mid, self.settings):
            return "available", None
        if self.settings.allow_model_download:
            return "available", "Weights will be downloaded on first use."
        return "unavailable", f"Model weights for '{mid}' are not installed and downloads are disabled."

    def describe(self, spec: ModelSpec) -> dict[str, Any]:
        status, reason = self.availability(spec)
        mid = self.model_id(spec)
        dim = spec.expected_dimension if spec.family == "hf" else WHISPER_DIMS.get(mid)
        loaded = self._loaded.get(self._cache_key(spec))
        if loaded is not None and spec.family == "hf":
            dim = int(loaded[1].config.hidden_size)
        elif loaded is not None and spec.family == "whisper":
            dim = int(loaded.dims.n_audio_state)
        return {
            "key": spec.key,
            "name": spec.display_name,
            "model_id": mid,
            "family": "huggingface-transformers" if spec.family == "hf" else "openai-whisper",
            "tag": spec.tag,
            "description": spec.description,
            "status": status,
            "reason": reason,
            "device": resolve_device(self.settings.device) if status != "unavailable" else None,
            "embedding_dimension": dim,
            "embedding_dimension_source": "loaded_config" if loaded is not None else ("expected" if dim else None),
            "maturity": "implemented" if status in ("available", "ready") else "proposed",
        }

    def list(self) -> list[dict[str, Any]]:
        return [self.describe(s) for s in SPECS]

    # ----- loading --------------------------------------------------------
    def _load(self, cache_key: str, loader: Callable[[], Any]) -> Any:
        with self._lock:
            if cache_key in self._loaded:
                return self._loaded[cache_key]
            self._state[cache_key] = ("loading", None)
            try:
                t0 = time.perf_counter()
                obj = loader()
                log.info("loaded %s in %.1fs", cache_key, time.perf_counter() - t0)
            except Exception as exc:
                self._state[cache_key] = ("error", f"Loading failed: {type(exc).__name__}: {exc}"[:300])
                raise ModelUnavailable(self._state[cache_key][1]) from exc
            self._loaded[cache_key] = obj
            self._state[cache_key] = ("ready", None)
            return obj

    def load_whisper(self, name: str):
        """Shared openai-whisper model (used by both ASR and the Whisper encoder)."""
        path = whisper_checkpoint_path(name, self.settings)
        if path is None:
            raise ModelUnavailable(f"Whisper checkpoint '{name}' is unknown or openai-whisper is not installed.")
        if not path.exists() and not self.settings.allow_model_download:
            raise ModelUnavailable(f"Whisper '{name}' weights are not installed and downloads are disabled.")

        def loader():
            import whisper

            root = str(self.settings.whisper_cache_dir) if self.settings.whisper_cache_dir else None
            return whisper.load_model(name, device=resolve_device(self.settings.device), download_root=root)

        return self._load(f"whisper:{name}", loader)

    def load_hf(self, spec: ModelSpec):
        status, reason = self.availability(spec)
        if status == "unavailable":
            raise ModelUnavailable(reason or "Model unavailable.")
        mid = self.model_id(spec)

        def loader():
            from transformers import AutoFeatureExtractor, AutoModel

            kw: dict[str, Any] = {"local_files_only": not self.settings.allow_model_download}
            if self.settings.hf_cache_dir:
                kw["cache_dir"] = str(self.settings.hf_cache_dir)
            fe = AutoFeatureExtractor.from_pretrained(mid, **kw)
            model = AutoModel.from_pretrained(mid, **kw).to(resolve_device(self.settings.device)).eval()
            return fe, model

        return self._load(self._cache_key(spec), loader)

    # ----- embedding extraction ---------------------------------------------
    def extract_embedding(self, key: str, waveform: np.ndarray, sr: int) -> dict[str, Any]:
        """Mean-pooled final-layer embedding. Returns the vector plus provenance metadata."""
        spec = SPEC_BY_KEY.get(key)
        if spec is None:
            raise ModelUnavailable(f"Unknown model key '{key}'.")
        status, reason = self.availability(spec)
        if status == "unavailable":
            raise ModelUnavailable(reason or "Model unavailable.")
        import torch

        device = resolve_device(self.settings.device)
        chunk = int(self.settings.max_embedding_seconds * sr)
        frames: list[np.ndarray] = []
        with torch.inference_mode():
            if spec.family == "whisper":
                import whisper

                model = self.load_whisper(self.model_id(spec))
                chunk = min(chunk, whisper.audio.N_SAMPLES)  # encoder window is 30 s
                n_mels = model.dims.n_mels
                for start in range(0, len(waveform), chunk):
                    piece = waveform[start:start + chunk]
                    mel = whisper.log_mel_spectrogram(whisper.pad_or_trim(torch.from_numpy(piece)), n_mels=n_mels)
                    out = model.encoder(mel.unsqueeze(0).to(device))[0]  # (1500, D) @ 50 Hz
                    n_valid = max(1, int(np.ceil(len(piece) / sr * 50)))
                    frames.append(out[:n_valid].float().cpu().numpy())
                layer = "encoder.final"
            else:
                fe, model = self.load_hf(spec)
                for start in range(0, len(waveform), chunk):
                    piece = waveform[start:start + chunk]
                    if len(piece) < sr // 10:  # skip tail fragments shorter than 100 ms
                        continue
                    inputs = fe(piece, sampling_rate=sr, return_tensors="pt")
                    out = model(**{k: v.to(device) for k, v in inputs.items()}).last_hidden_state[0]
                    frames.append(out.float().cpu().numpy())
                layer = "last_hidden_state"
        if not frames:
            raise ValueError("Audio too short for embedding extraction.")
        stacked = np.concatenate(frames, axis=0)
        vector = stacked.mean(axis=0)
        return {
            "vector": vector,
            "dimension": int(vector.shape[0]),
            "frames": int(stacked.shape[0]),
            "pooling": "mean_over_frames",
            "layer": layer,
            "device": device,
            "model_id": self.model_id(spec),
        }
