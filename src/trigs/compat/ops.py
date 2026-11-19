# SPDX-License-Identifier: AGPL-3.0-only
"""Device-agnostic replacements for CUDA-only ops (macOS port)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

_SSIM_WINDOW = 11
_SSIM_SIGMA = 1.5
_SSIM_C1 = 0.01 ** 2
_SSIM_C2 = 0.03 ** 2


def _gauss_1d(device: torch.device, dtype: torch.dtype) -> Tensor:
    x = torch.arange(_SSIM_WINDOW, device=device, dtype=dtype) - _SSIM_WINDOW // 2
    g = torch.exp(-(x ** 2) / (2.0 * _SSIM_SIGMA ** 2))
    return g / g.sum()


def _blur(x: Tensor, g: Tensor) -> Tensor:
    """Separable Gaussian blur with zero padding (== 2D conv with zero padding)."""
    c = x.shape[1]
    p = _SSIM_WINDOW // 2
    x = F.conv2d(x, g.view(1, 1, 1, -1).expand(c, 1, 1, _SSIM_WINDOW), padding=(0, p), groups=c)
    return F.conv2d(x, g.view(1, 1, -1, 1).expand(c, 1, _SSIM_WINDOW, 1), padding=(p, 0), groups=c)


def ssim_fused_compatible(pred: Tensor, target: Tensor, padding: str = "valid") -> Tensor:
    """SSIM matching ``fused_ssim(pred, target, padding=...)`` (11x11 Gaussian, sigma 1.5,
    zero-padded statistics, data range 1). ``"valid"`` averages only pixels whose window
    lies fully inside the image; images too small for that fall back to the full map
    (fused_ssim would return NaN there).

    Args: pred, target: [B, C, H, W].
    """
    g = _gauss_1d(pred.device, pred.dtype)
    mu1, mu2 = _blur(pred, g), _blur(target, g)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = _blur(pred * pred, g) - mu1_sq
    s2 = _blur(target * target, g) - mu2_sq
    s12 = _blur(pred * target, g) - mu12
    ssim_map = ((2 * mu12 + _SSIM_C1) * (2 * s12 + _SSIM_C2)) / (
        (mu1_sq + mu2_sq + _SSIM_C1) * (s1 + s2 + _SSIM_C2))
    if padding == "valid":
        p = _SSIM_WINDOW // 2
        h, w = ssim_map.shape[-2:]
        if h > 2 * p and w > 2 * p:
            ssim_map = ssim_map[..., p:h - p, p:w - p]
    return ssim_map.mean()


__all__ = ["ssim_fused_compatible"]
