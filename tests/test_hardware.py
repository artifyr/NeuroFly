"""
Tests for Hardware and Device Acceleration (ROCm / CUDA / CPU Fallback).
"""

import warnings
import pytest
import torch

from device_utils import get_device, get_device_info, is_rocm_available


def test_device_info_metadata():
    """Verify device info dictionary contains expected keys and data types."""
    info = get_device_info()
    assert "selected_device" in info
    assert "cuda_available" in info
    assert "is_rocm" in info
    assert "pytorch_version" in info
    assert isinstance(info["cuda_available"], bool)
    assert isinstance(info["is_rocm"], bool)


def test_explicit_cpu_request():
    """Explicitly requesting 'cpu' should always return a CPU device without warnings."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        device = get_device(preferred="cpu")
        assert device.type == "cpu"
        # No fallback warnings should be issued for explicit cpu request
        fallback_warnings = [w for w in record if issubclass(w.category, UserWarning) and "Defaulting to CPU" in str(w.message)]
        assert len(fallback_warnings) == 0


def test_device_fallback_or_gpu_detection():
    """
    Verify that requesting 'cuda' either:
    1. Returns a CUDA/ROCm device if hardware is present and supported by the PyTorch build.
    2. Issues a descriptive UserWarning and falls back to CPU if GPU is not available.
    """
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        device = get_device(preferred="cuda")

        if torch.cuda.is_available():
            assert device.type == "cuda"
            # If ROCm is used, verify HIP detection
            hip_ver = getattr(torch.version, "hip", None)
            if hip_ver is not None:
                assert is_rocm_available() is True
        else:
            # Fallback must be triggered
            assert device.type == "cpu"
            fallback_warnings = [
                w for w in record
                if issubclass(w.category, UserWarning) and "Defaulting to CPU" in str(w.message)
            ]
            assert len(fallback_warnings) >= 1, "Expected UserWarning when falling back from CUDA to CPU."


def test_tensor_allocation_on_resolved_device():
    """Verify PyTorch can allocate and perform math on the resolved device."""
    device = get_device()
    x = torch.ones((10, 10), device=device)
    y = torch.ones((10, 10), device=device)
    z = torch.matmul(x, y)
    assert z.shape == (10, 10)
    assert torch.allclose(z, torch.full((10, 10), 10.0, device=device))
