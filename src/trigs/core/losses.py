"""Loss functions for TriGS training."""

import warnings

# Suppress torchmetrics deprecation warning before importing
warnings.filterwarnings("ignore", message=".*spectral_angle_mapper.*", category=FutureWarning)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch import Tensor  # noqa: E402

# Try to import fused_ssim, fall back to torchmetrics or custom implementation
_SSIM_BACKEND = None


def _get_ssim_backend():
    """Get the best available SSIM backend."""
    global _SSIM_BACKEND
    if _SSIM_BACKEND is not None:
        return _SSIM_BACKEND

    # Try fused_ssim first (fastest)
    try:
        from fused_ssim import fused_ssim  # noqa: F401
        _SSIM_BACKEND = "fused"
        return _SSIM_BACKEND
    except ImportError:
        pass

    # Try torchmetrics (good quality, widely available)
    try:
        from torchmetrics.image import structural_similarity_index_measure  # noqa: F401
        _SSIM_BACKEND = "torchmetrics"
        return _SSIM_BACKEND
    except ImportError:
        pass

    # Fallback to older torchmetrics.functional path
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            from torchmetrics.functional import structural_similarity_index_measure  # noqa: F401
        _SSIM_BACKEND = "torchmetrics"
        return _SSIM_BACKEND
    except ImportError:
        pass

    # Fall back to custom implementation
    _SSIM_BACKEND = "custom"
    return _SSIM_BACKEND


def _gaussian_kernel(size: int, sigma: float, device: torch.device) -> Tensor:
    """Create 1D Gaussian kernel."""
    coords = torch.arange(size, device=device, dtype=torch.float32)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g


def _ssim_custom(pred: Tensor, target: Tensor, window_size: int = 11) -> Tensor:
    """Custom SSIM implementation.

    Args:
        pred: (B, C, H, W) predicted images
        target: (B, C, H, W) target images
        window_size: Size of Gaussian window

    Returns:
        Scalar SSIM value (higher is better, max 1.0)
    """
    C = pred.shape[1]
    device = pred.device

    # Create 2D Gaussian kernel
    sigma = 1.5
    kernel_1d = _gaussian_kernel(window_size, sigma, device)
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.expand(C, 1, window_size, window_size)

    # Constants for stability
    K1, K2 = 0.01, 0.03
    L = 1.0  # dynamic range
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    # Compute means
    mu1 = F.conv2d(pred, kernel_2d, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, kernel_2d, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = F.conv2d(pred ** 2, kernel_2d, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target ** 2, kernel_2d, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel_2d, padding=window_size // 2, groups=C) - mu1_mu2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


def reconstruction_loss(
    pred: Tensor,
    target: Tensor,
    ssim_lambda: float = 0.2,
) -> Tensor:
    """Combined reconstruction loss (L1 + SSIM).

    This matches the loss computation from gsplat's simple_trainer.py.
    Automatically uses the best available SSIM backend.

    Args:
        pred: Predicted images [B, H, W, 3] or [H, W, 3]
        target: Target images [B, H, W, 3] or [H, W, 3]
        ssim_lambda: Weight for SSIM loss (L1 weight = 1 - ssim_lambda)

    Returns:
        Scalar loss tensor
    """
    # Ensure batch dimension
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    # L1 loss
    l1 = F.l1_loss(pred, target)

    # Get SSIM backend
    backend = _get_ssim_backend()

    # Convert to BCHW format for SSIM
    pred_bchw = pred.permute(0, 3, 1, 2)
    target_bchw = target.permute(0, 3, 1, 2)

    if backend == "fused":
        from fused_ssim import fused_ssim
        ssim_val = fused_ssim(pred_bchw, target_bchw, padding="valid")
    else:
        # macOS port: without fused_ssim (CUDA-only) use a pure-PyTorch SSIM that
        # reproduces fused_ssim(padding="valid"), instead of torchmetrics (reflect
        # padding, different value) or _ssim_custom (same-padding mean).
        from trigs.compat.ops import ssim_fused_compatible
        ssim_val = ssim_fused_compatible(pred_bchw, target_bchw, padding="valid")

    ssim_loss = 1.0 - ssim_val

    # Combined loss
    return l1 * (1.0 - ssim_lambda) + ssim_loss * ssim_lambda


def compute_psnr(pred: Tensor, target: Tensor) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Args:
        pred: Predicted image [H, W, 3] or [B, H, W, 3], values in [0, 1]
        target: Target image [H, W, 3] or [B, H, W, 3], values in [0, 1]

    Returns:
        PSNR value in dB (higher is better)
    """
    with torch.no_grad():
        mse = F.mse_loss(pred, target)
        if mse == 0:
            return float('inf')
        # Max value is 1.0 for normalized images
        psnr = 10 * torch.log10(1.0 / mse)
        return psnr.item()


def compute_ssim_metric(pred: Tensor, target: Tensor) -> float:
    """Compute SSIM metric between two images.

    Uses the best available SSIM backend (fused_ssim > torchmetrics > custom).

    Args:
        pred: Predicted image [H, W, 3] or [B, H, W, 3], values in [0, 1]
        target: Target image [H, W, 3] or [B, H, W, 3], values in [0, 1]

    Returns:
        SSIM value (higher is better, max 1.0)
    """
    with torch.no_grad():
        # Ensure batch dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        # Get SSIM backend
        backend = _get_ssim_backend()

        # Convert to BCHW format for SSIM
        pred_bchw = pred.permute(0, 3, 1, 2)
        target_bchw = target.permute(0, 3, 1, 2)

        if backend == "fused":
            from fused_ssim import fused_ssim
            ssim_val = fused_ssim(pred_bchw, target_bchw, padding="valid")
        elif backend == "torchmetrics":
            try:
                from torchmetrics.image import structural_similarity_index_measure  # type: ignore[attr-defined]
            except ImportError:
                from torchmetrics.functional import structural_similarity_index_measure  # type: ignore[attr-defined]
            ssim_val = structural_similarity_index_measure(
                pred_bchw, target_bchw, data_range=1.0
            )
        else:  # custom
            ssim_val = _ssim_custom(pred_bchw, target_bchw)

        return ssim_val.item()


def bezier_motion_loss(c0: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
    """TRiGS Eq. 11 — Bézier acceleration regularizer.

    For a quadratic Bézier ``ζ_res(τ) = (1−τ)² c0 + 2(1−τ)τ c1 + τ² c2`` the
    second derivative w.r.t. τ is the constant ``2·(c0 − 2 c1 + c2)``. The
    paper penalizes the un-scaled constant ``b = c0 − 2 c1 + c2`` to discourage
    abrupt curvature in both the rotation and translation components.

    Args:
        c0, c1, c2: [N, D] Bézier control points (D=6 for se(3)).

    Returns:
        Scalar L2 loss ``mean_i ‖b_i‖²``.
    """
    b = c0 - 2.0 * c1 + c2
    return (b ** 2).sum(dim=-1).mean()


def scale_size_loss(
    log_scales: Tensor,
    n_total_rows: int,
    rows: Tensor | slice = slice(None),
) -> Tensor:
    """Official TRiGS / gsplat-MCMC spatial-size penalty.

    ``sum_i sum_axis exp(s_i) / (3 · n_total_rows)``: the mean actual axis
    length over the whole model. ``rows`` restricts which rows are penalized
    and ``n_total_rows`` fixes the normalizer, so a resident subset (chunked
    training) carries the same per-row strength as the full model.
    """
    selected = log_scales[rows]
    if int(selected.numel()) == 0:
        return log_scales.new_zeros(())
    return selected.exp().sum() / float(3 * max(int(n_total_rows), 1))


def rigid_coherence_loss(
    zeta_base: Tensor,
    sh_dc: Tensor,
    nn_indices: Tensor,
    lambda_c: float,
    detach_color_weight: bool = True,
) -> Tensor:
    """TRiGS Eq. 12 — appearance-weighted k-NN base-motion coherence.

    ``K_ij  = exp(−λ_c · ‖c_i − c_j‖²)``
    ``L = mean_i mean_{j∈N(i)} K_ij · (‖ν_base_i − ν_base_j‖² + ‖ω_base_i − ω_base_j‖²)``

    The appearance weight suppresses coupling across object boundaries.
    ``zeta_base`` uses gsflow's (ρ, ω) convention — ``ρ`` corresponds to the
    paper's ``ν_base`` (translation coefficient), ``ω`` to ``ω_base``.

    Args:
        zeta_base: [N, 6] per-primitive base Lie-algebra coefficients.
        sh_dc: [N, 3] SH-DC (canonical) color for each primitive.
        nn_indices: [N, K] k-nearest-neighbor indices (anchor or mean space).
        lambda_c: Color-affinity temperature.
        detach_color_weight: Stop the loss from reaching ``sh_dc``. The
            official implementation keeps K_ij differentiable.

    Returns:
        Scalar loss.
    """
    nn_zeta = zeta_base[nn_indices]                          # [N, K, 6]
    nn_color = sh_dc[nn_indices]                             # [N, K, 3]

    diff_zeta = zeta_base.unsqueeze(1) - nn_zeta             # [N, K, 6]
    diff_color = sh_dc.unsqueeze(1) - nn_color               # [N, K, 3]

    color_sq = (diff_color ** 2).sum(dim=-1)                 # [N, K]
    K_ij = torch.exp(-lambda_c * color_sq)                   # [N, K]
    if detach_color_weight:
        K_ij = K_ij.detach()

    motion_sq = (diff_zeta ** 2).sum(dim=-1)                 # [N, K]
    denom = K_ij.sum().clamp_min(1e-6)
    # return (K_ij * motion_sq).sum() / denom
    return (K_ij * motion_sq).mean()
