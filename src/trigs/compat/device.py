# SPDX-License-Identifier: AGPL-3.0-only
"""Central device selection and backend-agnostic helpers (macOS port).

Selection order for ``"auto"``: CUDA -> MPS -> CPU. The environment variable
``TRIGS_DEVICE`` (e.g. ``cpu``, ``mps``, ``cuda:1``) overrides the automatic
choice. An explicit request for an unavailable backend falls back to the
automatic choice with a warning, mirroring the old ``cuda -> cpu`` fallback.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Union

import torch

log = logging.getLogger(__name__)

DeviceLike = Union[str, torch.device, None]

AUTO = "auto"
_warned: set[str] = set()


def mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def get_default_device() -> torch.device:
    """Best available device (``TRIGS_DEVICE`` overrides)."""
    env = os.environ.get("TRIGS_DEVICE", "").strip()
    if env and env.lower() != AUTO:
        return resolve_device(env)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        log.warning(msg, *args)


def resolve_device(requested: DeviceLike = AUTO) -> torch.device:
    """Turn a user/config device spec into an available ``torch.device``."""
    if requested is None:
        return get_default_device()
    if isinstance(requested, torch.device):
        dev = requested
    else:
        s = str(requested).strip()
        if s == "" or s.lower() == AUTO:
            return get_default_device()
        dev = torch.device(s)
    if dev.type == "cuda" and not torch.cuda.is_available():
        fb = _auto_without_env()
        _warn_once(f"cuda->{fb}", "CUDA requested (%s) but unavailable; using %s", dev, fb)
        return fb
    if dev.type == "mps" and not mps_available():
        _warn_once("mps->cpu", "MPS requested but unavailable; using cpu")
        return torch.device("cpu")
    return dev


def _auto_without_env() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _types(device: DeviceLike) -> set[str]:
    if device is None:
        out = set()
        if torch.cuda.is_available():
            out.add("cuda")
        if mps_available():
            out.add("mps")
        return out
    return {resolve_device(device).type}


def device_synchronize(device: DeviceLike = None) -> None:
    """Wait for pending GPU work (no-op on CPU). ``None`` = every active backend."""
    t = _types(device)
    if "cuda" in t:
        torch.cuda.synchronize()
    if "mps" in t:
        torch.mps.synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator memory (no-op on CPU)."""
    t = _types(device)
    if "cuda" in t:
        torch.cuda.empty_cache()
    if "mps" in t:
        torch.mps.empty_cache()


def manual_seed_all(seed: int) -> None:
    """Seed every available accelerator RNG (CPU RNG is seeded by ``torch.manual_seed``)."""
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if mps_available():
        torch.mps.manual_seed(seed)


def accelerator_rng_states() -> dict:
    """RNG states of available accelerators, for checkpoints."""
    return {
        "cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if mps_available() else None,
    }


def f64(x: torch.Tensor) -> torch.Tensor:
    """``x.double()``, but on the CPU when ``x`` lives on MPS (MPS has no float64).

    Use only where the float64 result is reduced to Python scalars or stays on the
    CPU; move results back explicitly where a device tensor is needed.
    """
    if x.device.type == "mps":
        return x.detach().cpu().double() if not x.requires_grad else x.cpu().double()
    return x.double()


def gpu_available() -> bool:
    return torch.cuda.is_available() or mps_available()


def is_oom_error(exc: BaseException) -> bool:
    """True for CUDA OOM and the MPS allocator's 'out of memory' RuntimeError."""
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


__all__ = [
    "AUTO", "accelerator_rng_states", "device_synchronize", "empty_cache",
    "f64", "get_default_device", "gpu_available", "is_oom_error", "manual_seed_all",
    "mps_available", "resolve_device",
]
