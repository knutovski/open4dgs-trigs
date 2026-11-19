# SPDX-License-Identifier: AGPL-3.0-only
"""Tests for the macOS-port compatibility ops (trigs.compat)."""
import math

import pytest
import torch
import torch.nn.functional as F

from trigs.compat.device import get_default_device, resolve_device
from trigs.compat.ops import ssim_fused_compatible


def _ssim_ref(a, b, valid=True):
    ws, sigma = 11, 1.5
    x = torch.arange(ws, dtype=a.dtype) - ws // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(a.shape[1], 1, ws, ws)
    conv = lambda t: F.conv2d(t, k, padding=ws // 2, groups=a.shape[1])
    m1, m2 = conv(a), conv(b)
    s1, s2, s12 = conv(a * a) - m1 ** 2, conv(b * b) - m2 ** 2, conv(a * b) - m1 * m2
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * m1 * m2 + c1) * (2 * s12 + c2)) / ((m1 ** 2 + m2 ** 2 + c1) * (s1 + s2 + c2))
    return s[..., 5:-5, 5:-5].mean() if valid else s.mean()


def test_ssim_matches_2d_reference_valid_padding():
    torch.manual_seed(0)
    a, b = torch.rand(2, 3, 40, 52, dtype=torch.float64).unbind(0)
    a, b = a[None], b[None]
    assert torch.allclose(ssim_fused_compatible(a, b), _ssim_ref(a, b), atol=1e-12)
    assert torch.allclose(ssim_fused_compatible(a, b, padding="same"), _ssim_ref(a, b, valid=False), atol=1e-12)


def test_ssim_identical_is_one_and_tiny_images_are_finite():
    a = torch.rand(1, 3, 32, 32)
    assert math.isclose(float(ssim_fused_compatible(a, a)), 1.0, abs_tol=1e-5)
    tiny = torch.rand(1, 3, 4, 4, requires_grad=True)
    v = ssim_fused_compatible(tiny, torch.rand(1, 3, 4, 4))
    v.backward()
    assert torch.isfinite(v) and torch.isfinite(tiny.grad).all()


def test_resolve_device_falls_back_and_honours_auto(monkeypatch):
    monkeypatch.delenv("TRIGS_DEVICE", raising=False)
    assert resolve_device("auto") == get_default_device()
    assert resolve_device("cpu").type == "cpu"
    if not torch.cuda.is_available():
        assert resolve_device("cuda").type in ("mps", "cpu")
    monkeypatch.setenv("TRIGS_DEVICE", "cpu")
    assert get_default_device().type == "cpu"


@pytest.mark.skipif(not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()), reason="needs MPS")
def test_ssim_mps_matches_cpu():
    torch.manual_seed(1)
    a, b = torch.rand(1, 3, 64, 80), torch.rand(1, 3, 64, 80)
    cpu = ssim_fused_compatible(a, b)
    mps = ssim_fused_compatible(a.to("mps"), b.to("mps")).cpu()
    assert torch.allclose(cpu, mps, atol=1e-5)
