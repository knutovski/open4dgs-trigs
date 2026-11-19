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



# ---------------------------------------------------------------------------
# pytorch3d replacements: knn_points / sample_farthest_points
# ---------------------------------------------------------------------------
from collections import namedtuple
import os
from typing import Optional, Union

KNN = namedtuple("KNN", ["dists", "idx", "knn"])

# Max. number of pairwise distances held at once per chunk (default 2**28 ~ 1 GiB fp32).
_KNN_BUDGET = int(os.environ.get("TRIGS_KNN_BUDGET", str(1 << 28)))
_FORCE_TORCH = os.environ.get("TRIGS_COMPAT_FORCE_TORCH", "0") == "1"


def _pytorch3d_ops():
    """pytorch3d.ops if importable (keeps NVIDIA behaviour identical), else None."""
    if _FORCE_TORCH:
        return None
    try:
        from pytorch3d import ops  # type: ignore
        return ops
    except Exception:
        return None


def knn_points_torch(p1: Tensor, p2: Tensor, K: int = 1) -> KNN:
    """Brute-force KNN like ``pytorch3d.ops.knn_points`` (no lengths, return_nn=False).

    Args: p1 [B, P1, D] queries, p2 [B, P2, D] targets.
    Returns KNN(dists [B, P1, K] *squared* distances ascending, idx [B, P1, K], None).
    Queries are processed in chunks so that at most ``TRIGS_KNN_BUDGET`` distances
    exist at a time; memory stays O(chunk * P2) instead of O(P1 * P2).
    """
    if p1.dim() != 3 or p2.dim() != 3 or p1.shape[0] != p2.shape[0]:
        raise ValueError("expected p1 [B,P1,D] and p2 [B,P2,D]")
    B, P1, _ = p1.shape
    P2 = p2.shape[1]
    K = int(K)
    if K < 1 or K > P2:
        raise ValueError(f"K must be in [1, {P2}], got {K}")
    dists = torch.empty((B, P1, K), device=p1.device, dtype=p1.dtype)
    idx = torch.empty((B, P1, K), device=p1.device, dtype=torch.long)
    chunk = max(1, min(P1, _KNN_BUDGET // max(P2, 1)))
    for b in range(B):
        tgt = p2[b]
        tgt_sq = (tgt * tgt).sum(-1)
        for s in range(0, P1, chunk):
            q = p1[b, s:s + chunk]
            # Squared Euclidean distance via the expansion; clamp tiny negatives.
            d = (q * q).sum(-1, keepdim=True) + tgt_sq[None, :] - 2.0 * (q @ tgt.T)
            d.clamp_(min=0.0)
            v, i = torch.topk(d, K, dim=-1, largest=False, sorted=True)
            dists[b, s:s + chunk] = v
            idx[b, s:s + chunk] = i
    return KNN(dists, idx, None)


def knn_points(p1: Tensor, p2: Tensor, K: int = 1, **kwargs) -> KNN:
    """``pytorch3d.ops.knn_points`` when available on CUDA, else :func:`knn_points_torch`."""
    ops = _pytorch3d_ops()
    if ops is not None and p1.is_cuda:
        return ops.knn_points(p1, p2, K=K, **kwargs)
    if kwargs.get("lengths1") is not None or kwargs.get("lengths2") is not None or kwargs.get("return_nn"):
        raise NotImplementedError("compat knn_points: lengths/return_nn not supported")
    return knn_points_torch(p1, p2, K=K)


def sample_farthest_points_torch(
    points: Tensor,
    lengths: Optional[Tensor] = None,
    K: Union[int, Tensor] = 50,
    random_start_point: bool = False,
):
    """Iterative farthest point sampling like ``pytorch3d.ops.sample_farthest_points``.

    Args: points [B, P, D]; lengths [B] valid points per batch (default P);
    K int or [B] tensor. Returns (sampled [B, Kmax, D] zero-padded, idx [B, Kmax] -1-padded).
    Starts at index 0 unless ``random_start_point``. All batches advance in parallel.
    """
    B, P, D = points.shape
    dev = points.device
    if lengths is None:
        lengths = torch.full((B,), P, dtype=torch.long, device=dev)
    lengths = lengths.to(device=dev, dtype=torch.long)
    if isinstance(K, int):
        K = torch.full((B,), K, dtype=torch.long, device=dev)
    K = K.to(device=dev, dtype=torch.long)
    K = torch.minimum(K, lengths)
    k_max = int(K.max().item()) if B > 0 else 0
    idx = torch.full((B, k_max), -1, dtype=torch.long, device=dev)
    if k_max == 0:
        return points.new_zeros((B, 0, D)), idx
    valid = torch.arange(P, device=dev)[None, :] < lengths[:, None]
    # Invalid (padded) points can never be the farthest.
    min_d = torch.where(valid, torch.full((B, P), float("inf"), device=dev, dtype=points.dtype),
                        torch.full((B, P), -1.0, device=dev, dtype=points.dtype))
    if random_start_point:
        cur = (torch.rand(B, device=dev) * lengths.clamp(min=1)).long()
    else:
        cur = torch.zeros(B, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    active_k = torch.arange(k_max, device=dev)[None, :] < K[:, None]  # [B, Kmax]
    for i in range(k_max):
        idx[:, i] = cur
        sel = points[ar, cur]                              # [B, D]
        d = ((points - sel[:, None, :]) ** 2).sum(-1)      # [B, P]
        min_d = torch.minimum(min_d, torch.where(valid, d, min_d))
        cur = torch.argmax(min_d, dim=1)
    idx = torch.where(active_k, idx, torch.full_like(idx, -1))
    gather = idx.clamp(min=0)
    sampled = points[ar[:, None], gather] * active_k[..., None].to(points.dtype)
    return sampled, idx


def sample_farthest_points(points: Tensor, lengths: Optional[Tensor] = None,
                           K: Union[int, Tensor] = 50, random_start_point: bool = False):
    """``pytorch3d.ops.sample_farthest_points`` when available on CUDA, else the torch version."""
    ops = _pytorch3d_ops()
    if ops is not None and points.is_cuda:
        return ops.sample_farthest_points(points, lengths=lengths, K=K,
                                          random_start_point=random_start_point)
    return sample_farthest_points_torch(points, lengths=lengths, K=K,
                                        random_start_point=random_start_point)


__all__ = ["ssim_fused_compatible", "KNN", "knn_points", "knn_points_torch",
           "sample_farthest_points", "sample_farthest_points_torch"]
