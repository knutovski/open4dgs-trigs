# SPDX-License-Identifier: AGPL-3.0-only
"""macOS port, phase 3: gsplat-compatible rasterization (mpsplat) on MPS vs. CPU reference.

Fixed scene (500 splats, one camera, SH degree 3). Tolerances (float32):
image max |mps-cpu| < 1e-2, relative gradient error < 5e-3 per parameter
(measured on M5 Max: image <= 4.1e-3, gradients <= 1.3e-3).
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()), reason="needs MPS")


def _scene(dev):
    g = torch.Generator().manual_seed(0)
    N, W, H = 500, 160, 120
    means = (torch.rand(N, 3, generator=g) - 0.5) * 2.0
    means[:, 2] += 4.0
    quats = torch.nn.functional.normalize(torch.randn(N, 4, generator=g), dim=-1)
    scales = torch.rand(N, 3, generator=g) * 0.05 + 0.02
    opac = torch.rand(N, generator=g) * 0.8 + 0.1
    sh = torch.randn(N, 16, 3, generator=g) * 0.3
    K = torch.tensor([[[150.0, 0, W / 2], [0, 150.0, H / 2], [0, 0, 1]]])
    params = [x.to(dev).requires_grad_(True) for x in (means, quats, scales, opac, sh)]
    return params, torch.eye(4)[None].to(dev), K.to(dev), W, H


@pytest.mark.parametrize("mode", ["RGB", "RGB+D"])
def test_rasterization_mps_matches_cpu(mode):
    from gsplat import rasterization
    out = {}
    for dev in ("cpu", "mps"):
        (m, q, s, o, c), V, K, W, H = _scene(dev)
        img, _, info = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=V,
                                     Ks=K, width=W, height=H, sh_degree=3, packed=False, render_mode=mode)
        assert "means2d" in info
        img.square().mean().backward()
        out[dev] = (img.detach().cpu(), [x.grad.detach().cpu() for x in (m, q, s, o, c)])
    (ic, gc), (im, gm) = out["cpu"], out["mps"]
    assert (ic - im).abs().max() < 1e-2
    for a, b in zip(gc, gm):
        assert ((a - b).norm() / (a.norm() + 1e-12)) < 5e-3
