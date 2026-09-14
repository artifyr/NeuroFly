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


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Resolve the execution device for PyTorch.

    Parameters
    ----------
    preferred : str
        Target device name ('cuda', 'hip', or 'cpu'). Default is 'cuda'.

    Returns
    -------
    torch.device
        Resolved PyTorch device. If preferred is 'cuda' or 'hip' but unavailable,
        issues a UserWarning and falls back to 'cpu'.
    """
    preferred_normalized = preferred.strip().lower()

    if preferred_normalized in ("cuda", "hip"):
        if torch.cuda.is_available():
            # Check if this is an AMD ROCm build or CUDA
            rocm_ver = getattr(torch.version, "hip", None)
            dev_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
            if rocm_ver:
                # ROCm accelerated
                return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cuda")
            else:
                # CUDA or general GPU
                return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cuda")
        else:
            warnings.warn(
                f"Requested device '{preferred}' is unavailable (torch.cuda.is_available() is False). "
                "Defaulting to CPU. Ensure AMD ROCm or GPU drivers and PyTorch GPU wheel are properly installed.",
                UserWarning,
                stacklevel=2,
            )
            return torch.device("cpu")

    return torch.device(preferred_normalized)


def get_device_info(device: torch.device | None = None) -> dict[str, str | bool | int]:
    """
    Return descriptive metadata about the current compute environment.
    """
    dev = device or get_device()
    cuda_avail = torch.cuda.is_available()
    hip_ver = getattr(torch.version, "hip", None)

    info: dict[str, str | bool | int] = {
        "selected_device": str(dev),
        "cuda_available": cuda_avail,
        "is_rocm": bool(hip_ver is not None and cuda_avail),
        "rocm_version": str(hip_ver) if hip_ver else "None",
        "pytorch_version": torch.__version__,
    }

    if cuda_avail and torch.cuda.device_count() > 0:
        info["device_count"] = torch.cuda.device_count()
        info["device_name"] = torch.cuda.get_device_name(0)

    return info
