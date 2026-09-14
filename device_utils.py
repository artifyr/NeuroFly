"""
Device detection and hardware acceleration utilities.

Configures PyTorch execution for AMD ROCm (via the 'cuda' device abstraction)
or NVIDIA CUDA, with graceful fallback and warnings when falling back to CPU.
"""

from __future__ import annotations

import warnings
import torch


def is_rocm_available() -> bool:
    """Return True if PyTorch was built with ROCm/HIP support and a device is ready."""
    return torch.cuda.is_available() and getattr(torch.version, "hip", None) is not None


def get_device(preferred: str = "auto") -> torch.device:
    """
    Resolve the optimal execution device for PyTorch.

    Automatically detects NVIDIA CUDA, AMD ROCm, Apple Silicon (MPS),
    Intel GPU/XPU, or gracefully falls back to CPU (e.g. Intel integrated graphics / CPU).

    Parameters
    ----------
    preferred : str
        Target device name ('auto', 'cuda', 'hip', 'mps', 'xpu', or 'cpu'). Default is 'auto'.

    Returns
    -------
    torch.device
        Resolved PyTorch device.
    """
    preferred_normalized = preferred.strip().lower()

    if preferred_normalized in ("auto", "cuda", "hip"):
        # 1. Check CUDA / ROCm (AMD & NVIDIA share the torch.cuda API)
        if torch.cuda.is_available():
            dev_idx = 0
            return torch.device(f"cuda:{dev_idx}")

        # 2. Check Intel XPU (Intel Arc / Data Center GPU)
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu:0")

        # 3. Check Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")

        # 4. Graceful CPU Fallback (Default for machines without discrete GPU or on Intel CPU)
        if preferred_normalized in ("cuda", "hip"):
            warnings.warn(
                f"Requested device '{preferred}' is unavailable (torch.cuda.is_available() is False). "
                "Defaulting to CPU. (If using NVIDIA/AMD GPU, verify GPU drivers and PyTorch CUDA/ROCm install).",
                UserWarning,
                stacklevel=2,
            )
        return torch.device("cpu")

    if preferred_normalized == "cpu":
        return torch.device("cpu")

    # Explicit device string (e.g. 'cuda:1', 'cpu')
    try:
        return torch.device(preferred_normalized)
    except Exception:
        warnings.warn(
            f"Unrecognized device '{preferred}'. Falling back to CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")


def get_device_info(device: torch.device | None = None) -> dict[str, str | bool | int]:
    """
    Return descriptive metadata about the current compute environment.
    """
    dev = device or get_device()
    cuda_avail = torch.cuda.is_available()
    hip_ver = getattr(torch.version, "hip", None)

    # Determine processor / accelerator type
    if dev.type == "cuda":
        accel_type = "AMD ROCm" if hip_ver is not None else "NVIDIA CUDA"
        hardware_name = torch.cuda.get_device_name(dev.index or 0) if cuda_avail and torch.cuda.device_count() > 0 else "CUDA Device"
    elif dev.type == "xpu":
        accel_type = "Intel XPU"
        hardware_name = getattr(torch.xpu, "get_device_name", lambda idx: "Intel GPU")(dev.index or 0)
    elif dev.type == "mps":
        accel_type = "Apple Silicon MPS"
        hardware_name = "Apple Neural Engine / GPU"
    else:
        accel_type = "CPU (Host Processor / Integrated Graphics)"
        import platform
        hardware_name = platform.processor() or "CPU"

    info: dict[str, str | bool | int] = {
        "selected_device": str(dev),
        "device_type": dev.type,
        "accelerator_type": accel_type,
        "device_name": hardware_name,
        "cuda_available": cuda_avail,
        "is_rocm": bool(hip_ver is not None and cuda_avail),
        "rocm_version": str(hip_ver) if hip_ver else "None",
        "pytorch_version": torch.__version__,
    }

    if cuda_avail and torch.cuda.device_count() > 0:
        info["device_count"] = torch.cuda.device_count()

    return info
