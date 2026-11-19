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


# --- pytorch3d replacements -------------------------------------------------
from trigs.compat.ops import knn_points_torch, sample_farthest_points_torch  # noqa: E402

_DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("dev", _DEVICES)
@pytest.mark.parametrize("D", [3, 4])
def test_knn_matches_brute_force(dev, D, monkeypatch):
    monkeypatch.setattr("trigs.compat.ops._KNN_BUDGET", 700)  # force many chunks
    g = torch.Generator().manual_seed(D)
    p1 = torch.rand(2, 300, D, generator=g, dtype=torch.float64)
    p2 = torch.rand(2, 250, D, generator=g, dtype=torch.float64)
    ref_d = torch.cdist(p1, p2) ** 2
    ref_v, ref_i = ref_d.topk(7, dim=-1, largest=False, sorted=True)
    dt = torch.float32 if dev == "mps" else torch.float64
    res = knn_points_torch(p1.to(dev, dt), p2.to(dev, dt), K=7)
    assert torch.equal(res.idx.cpu(), ref_i)
    assert torch.allclose(res.dists.cpu().double(), ref_v, atol=1e-5 if dev == "mps" else 1e-10)


@pytest.mark.parametrize("dev", _DEVICES)
def test_knn_self_query_puts_self_first(dev):
    p = torch.rand(1, 500, 3).to(dev)
    res = knn_points_torch(p, p, K=4)
    assert torch.equal(res.idx[0, :, 0].cpu(), torch.arange(500))


def _fps_ref(pts, k):
    sel = [0]
    d = ((pts - pts[0]) ** 2).sum(-1)
    for _ in range(k - 1):
        j = int(torch.argmax(d)); sel.append(j)
        d = torch.minimum(d, ((pts - pts[j]) ** 2).sum(-1))
    return torch.tensor(sel)


@pytest.mark.parametrize("dev", _DEVICES)
def test_fps_matches_reference_with_lengths_and_per_batch_k(dev):
    g = torch.Generator().manual_seed(3)
    pts = torch.rand(3, 400, 3, generator=g, dtype=torch.float64)
    lengths = torch.tensor([400, 250, 31])
    K = torch.tensor([50, 20, 40])  # last one exceeds its length -> clamped to 31
    dt = torch.float32 if dev == "mps" else torch.float64
    sampled, idx = sample_farthest_points_torch(pts.to(dev, dt), lengths=lengths.to(dev), K=K.to(dev))
    idx, sampled = idx.cpu(), sampled.cpu().double()
    assert idx.shape == (3, 50)
    for b, (n, k) in enumerate([(400, 50), (250, 20), (31, 31)]):
        assert torch.equal(idx[b, :k], _fps_ref(pts[b, :n], k))
        assert bool((idx[b, k:] == -1).all())
        assert torch.allclose(sampled[b, :k], pts[b, idx[b, :k]], atol=1e-6)
        assert bool((sampled[b, k:] == 0).all())


def test_fps_keeps_small_disconnected_component():
    main = torch.randn(1000, 3) * 0.05
    far = torch.tensor([[5.0, 0.0, 0.0], [5.1, 0.0, 0.0]])
    _, idx = sample_farthest_points_torch(torch.cat([main, far])[None], K=8)
    assert bool((idx[0] >= 1000).any())
