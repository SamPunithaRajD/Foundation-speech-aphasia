"""Hardware / environment inspection used to decide device placement and to
report honest model availability."""
from __future__ import annotations

import importlib.util
import os
import platform
from functools import lru_cache
from typing import Any


@lru_cache
def torch_status() -> tuple[bool, str | None]:
    """(usable, reason). Importing torch can fail on broken installs; never crash the API."""
    try:
        import torch  # noqa: F401

        return True, None
    except Exception as exc:  # OSError on DLL problems, ImportError when missing
        return False, f"PyTorch could not be imported ({type(exc).__name__})."


def resolve_device(preference: str) -> str:
    ok, _ = torch_status()
    if not ok:
        return "none"
    import torch

    if preference == "cpu":
        return "cpu"
    if preference in ("auto", "cuda") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def library_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def hardware_summary(device_preference: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "device_preference": device_preference,
        "device": resolve_device(device_preference),
        "torch_available": torch_status()[0],
        "torch_error": torch_status()[1],
        "cuda_available": False,
        "gpus": [],
    }
    try:
        import psutil  # optional

        info["memory_gb"] = round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        info["memory_gb"] = None
    if info["torch_available"]:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        for i in range(torch.cuda.device_count() if info["cuda_available"] else 0):
            props = torch.cuda.get_device_properties(i)
            info["gpus"].append({"name": props.name, "vram_gb": round(props.total_memory / 1024**3, 1)})
    info["libraries"] = {
        lib: library_available(lib) for lib in ("transformers", "whisper", "librosa", "shap", "lime", "sklearn", "av")
    }
    return info
