"""Canonical-set initialization from SelfCap per-frame point clouds.

Mirrors :meth:`trigs.core.trainer.TrigsTrainer._build_canonical_set` but
consumes SelfCap's ``x/y/z/r/g/b`` PLYs instead of Gaussian PLYs.

Pipeline:

1. Pick a keyframe every ``keyframe_stride`` frames in
   ``[absolute_frame_origin, absolute_frame_origin + n_frames)``.
2. For each keyframe load its PLY — prefer ``dense_pcds/FFFFFF.ply``
   when present, otherwise fall back to ``pcds/FFFFFF.ply``.
3. Per-keyframe scale init from kNN distance; identity quats; logit-opacity
   from ``init_opacity``; SH-DC colors from RGB.
4. Centered finite-difference velocity init: match each keyframe's points
   to ±1-frame neighbor clouds via k-NN mean (resolving from dense first,
   then sparse), average f/b for a centered estimate (one-sided at sequence
   boundaries). Dense keyframes are initialized with zero velocity.
5. Radius-ball anchor init; local weighted rigid-twist fit for (ν, ω)
   in each keyframe neighborhood.
6. Optional per-keyframe farthest-point sampling down to ``target_count``
   using temporal-stratified allocation plus chunked batched FPS in XYZ.
7. Pack into the ParameterDict schema the trainer consumes.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor

from trigs.io.selfcap import _load_xyzrgb_ply
from .from_colmap import _SH_C0, _knn_distance, _radius_ball_centroid
from ..math.twist_init import fit_local_twist, local_knn_indices

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-frame PLY resolution


def _resolve_ply(
    dense_dir: Optional[Path],
    sparse_dir: Optional[Path],
    abs_frame: int,
    frame_digits: int = 6,
) -> Optional[Path]:
    """Prefer dense, fall back to sparse. Return None if neither exists."""
    name = f"{abs_frame:0{frame_digits}d}.ply"
    if dense_dir is not None:
        p = dense_dir / name
        if p.exists():
            return p
    if sparse_dir is not None:
        p = sparse_dir / name
        if p.exists():
            return p
    return None


def _load_frame(path: Path, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Load a SelfCap PLY as (positions, colors-in-[0,1]) torch tensors."""
    xyz_np, rgb_np = _load_xyzrgb_ply(path)
    xyz = torch.from_numpy(xyz_np).float().to(device)
    rgb = torch.from_numpy(rgb_np).float().to(device) / 255.0
    return xyz, rgb


# ---------------------------------------------------------------------------
# kNN — local free function so we don't need a TrigsTrainer instance.


@torch.no_grad()
def _knn_mean_positions(src: Tensor, tgt: Tensor, k: int) -> Tensor:
    """For each row of ``src``, mean 3-D position of its k nearest neighbors in tgt."""
    k = max(1, min(k, tgt.shape[0]))
    try:
        from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)
        res = knn_points(src.unsqueeze(0), tgt.unsqueeze(0), K=k)
        nn_idx = res.idx.squeeze(0)
    except ImportError:
        d = torch.cdist(src, tgt)
        nn_idx = d.topk(k, largest=False).indices
    return tgt[nn_idx].mean(dim=1)


# ---------------------------------------------------------------------------
# Main builder


def build_canonical_from_pcd_sequence(
    data_dir: str | Path,
    *,
    absolute_frame_origin: int,
    n_frames: int,
    fps: float = 60.0,
    keyframe_stride: int = 1,
    target_count: Optional[int] = None,
    fps_time_weight: float = 1.0,
    nn_velocity_init: bool = True,
    nn_velocity_k: int = 3,
    omega_init: bool = True,
    local_twist_k: int = 16,
    local_twist_reg: float = 1e-4,
    anchor_radius_frac: float = 0.1,
    init_opacity: float = 0.1,
    scale_knn_k: int = 4,
    scale_knn_factor: float = 0.3,
    init_sigma_t: float = 0.0,
    dense_subdir: str = "dense_pcds",
    sparse_subdir: str = "pcds",
    device: str = "auto",
    fps_chunk_size: int = 16,
    frame_digits: int = 6,
    verbose: bool = False,
) -> Tuple[nn.ParameterDict, int, float]:
    """Build a canonical ParameterDict from SelfCap's per-frame pcds.

    Args:
        data_dir: Preprocessed SelfCap directory containing ``pcds/`` and
            optionally ``dense_pcds/``.
        absolute_frame_origin: Absolute footage index of preprocessed frame
            0 (i.e. ``preprocess_meta.json["absolute_frame_origin"]``).
        n_frames: Number of frames in the preprocessing window.
        fps: Sequence frame rate.
        keyframe_stride: Stride in frames between keyframes.
        target_count: If set, downsample keyframe primitives to this count.
        fps_time_weight: Kept for API compatibility with trainer-side init.
        fps_chunk_size: Number of keyframes per batched FPS call. Lower values
            reduce peak memory at the cost of more kernel launches.
        nn_velocity_init / nn_velocity_k: centered-FD ν seed via kNN mean.
        omega_init: Seed (ν_base, ω_base) via local weighted rigid-twist fit.
        local_twist_k: Same-keyframe neighborhood size for local twist fitting.
        local_twist_reg: Tikhonov regularization for local twist fitting.
        anchor_radius_frac: Radius-ball fraction of per-keyframe extent.
        init_opacity / scale_knn_*: Same semantics as trainer defaults.
        init_sigma_t: Base sparse-keyframe σ_t in seconds. 0 → 1.0 second.
            Dense keyframes use 5x this value.
        dense_subdir / sparse_subdir: Where to look for PLYs.
        device / frame_digits / verbose: plumbing.

    Returns:
        ``(params, sh_degree=0, scene_extent)`` — same contract as
        :func:`trigs.init.from_colmap.build_canonical_from_npz`.
    """
    data_dir = Path(data_dir)
    dense_dir = (data_dir / dense_subdir) if (data_dir / dense_subdir).is_dir() else None
    sparse_dir = (data_dir / sparse_subdir) if (data_dir / sparse_subdir).is_dir() else None
    if dense_dir is None and sparse_dir is None:
        raise FileNotFoundError(
            f"Neither {dense_subdir}/ nor {sparse_subdir}/ found under {data_dir}"
        )

    dev = resolve_device(device)
    stride = max(keyframe_stride, 1)
    fps_safe = max(fps, 1e-6)
    dt_frame = 1.0 / fps_safe

    # ---------- collect keyframe frame-offsets + PLY paths ----------------
    kf_offsets: List[int] = []
    kf_paths: List[Path] = []
    kf_kinds: List[str] = []
    for f_off in range(0, n_frames, stride):
        abs_f = absolute_frame_origin + f_off
        p = _resolve_ply(dense_dir, sparse_dir, abs_f, frame_digits)
        if p is None:
            continue
        kf_offsets.append(f_off)
        kf_paths.append(p)
        kf_kinds.append("dense" if (dense_dir is not None and p.parent == dense_dir) else "sparse")

    if not kf_paths:
        raise FileNotFoundError(
            f"No PLYs found in [{absolute_frame_origin}, "
            f"{absolute_frame_origin + n_frames}) @ stride {stride}"
        )

    if verbose:
        n_dense = sum(k == "dense" for k in kf_kinds)
        log.info(
            f"[from_selfcap] {len(kf_paths)} keyframes "
            f"({n_dense} dense, {len(kf_paths) - n_dense} sparse), stride={stride}"
        )

    # ---------- load each keyframe, build Gaussians ------------------------
    all_means: List[Tensor] = []
    all_colors: List[Tensor] = []
    all_times: List[Tensor] = []
    all_log_scales: List[Tensor] = []
    all_kinds: List[str] = []
    all_frame_offsets: List[int] = []

    for f_off, path, kind in zip(kf_offsets, kf_paths, kf_kinds):
        xyz, rgb = _load_frame(path, dev)
        N_k = xyz.shape[0]
        if N_k == 0:
            if verbose:
                log.warning(f"  empty PLY skipped: {path}")
            continue

        # Scale init from per-keyframe kNN distance.
        knn_d = _knn_distance(xyz, k=scale_knn_k)
        
        all_means.append(xyz)
        all_colors.append(rgb)
        all_log_scales.append(knn_d) # Store raw knn_d for now
        all_times.append(torch.full((N_k, 1), f_off * dt_frame, device=dev))
        all_kinds.append(kind)
        all_frame_offsets.append(f_off)

    if not all_means:
        raise RuntimeError("All keyframes empty; nothing to initialize")

    # Replace the raw knn_d with the actual log scale tensors per-point
    for i in range(len(all_log_scales)):
        log_scale = torch.log(torch.clamp(all_log_scales[i] * scale_knn_factor, min=1e-6))
        all_log_scales[i] = log_scale.unsqueeze(-1).expand(-1, 3)

    # ---------- centered FD ν init via ±1-frame kNN ------------------------
    all_nu: List[Tensor] = []
    if nn_velocity_init:
        for ki, f_off in enumerate(all_frame_offsets):
            src = all_means[ki]
            if all_kinds[ki] == "dense":
                # Dense keyframes are treated as static priors at init.
                all_nu.append(torch.zeros_like(src))
                continue
            v_fwd: Optional[Tensor] = None
            v_bwd: Optional[Tensor] = None

            # Resolve ±1-frame neighbors via _resolve_ply (prefer dense, then sparse).
            if f_off + 1 < n_frames:
                p_next = _resolve_ply(
                    dense_dir, sparse_dir,
                    absolute_frame_origin + f_off + 1, frame_digits,
                )
                if p_next is not None:
                    nxt, _ = _load_frame(p_next, dev)
                    if nxt.shape[0] > 0:
                        v_fwd = (_knn_mean_positions(src, nxt, nn_velocity_k) - src) / dt_frame
                    del nxt

            if f_off - 1 >= 0:
                p_prev = _resolve_ply(
                    dense_dir, sparse_dir,
                    absolute_frame_origin + f_off - 1, frame_digits,
                )
                if p_prev is not None:
                    prv, _ = _load_frame(p_prev, dev)
                    if prv.shape[0] > 0:
                        v_bwd = (src - _knn_mean_positions(src, prv, nn_velocity_k)) / dt_frame
                    del prv

            if v_fwd is not None and v_bwd is not None:
                all_nu.append((v_fwd + v_bwd) * 0.5)
            elif v_fwd is not None:
                all_nu.append(v_fwd)
            elif v_bwd is not None:
                all_nu.append(v_bwd)
            else:
                all_nu.append(torch.zeros_like(src))

        if verbose:
            nu_mag = torch.cat(all_nu).norm(dim=-1).mean().item()
            n_dense_zero = sum(1 for kind in all_kinds if kind == "dense")
            log.info(
                f"  FD ν init (k={nn_velocity_k}): mean ‖ν‖ = {nu_mag:.4f} world/s "
                f"(dense zero-ν keyframes: {n_dense_zero})"
            )
    else:
        all_nu = [torch.zeros_like(m) for m in all_means]

    # ---------- radius-ball anchor init ------------------------------------
    all_anchors: List[Tensor] = []
    for means_ki in all_means:
        if not omega_init or means_ki.shape[0] <= 1:
            all_anchors.append(means_ki.clone())
            continue
        ext = (
            means_ki.max(dim=0).values - means_ki.min(dim=0).values
        ).max().clamp(min=1e-6)
        radius = anchor_radius_frac * float(ext.item())
        all_anchors.append(_radius_ball_centroid(means_ki, radius))

    # ---------- ω / ν init via local weighted rigid-twist fit --------------
    all_omega: List[Tensor] = []
    if omega_init:
        for ki in range(len(all_means)):
            means_ki = all_means[ki]
            v_targets_ki = all_nu[ki]
            anchors_ki = all_anchors[ki]
            if means_ki.shape[0] <= 1:
                all_omega.append(torch.zeros_like(means_ki))
                continue
            nn_idx = local_knn_indices(means_ki, local_twist_k)
            nu_ki, om, anchors_ki = fit_local_twist(
                means_ki,
                v_targets_ki,
                anchors_ki,
                nn_idx,
                reg=local_twist_reg,
            )
            all_omega.append(om)
            all_nu[ki] = nu_ki
            all_anchors[ki] = anchors_ki
        if verbose:
            mean_omega = torch.cat(all_omega).norm(dim=-1).mean().item()
            mean_nu = torch.cat(all_nu).norm(dim=-1).mean().item()
            log.info(
                f"  Local twist init (K={local_twist_k}, reg={local_twist_reg:g}): "
                f"mean ‖ω‖ = {mean_omega:.4f} rad/s, "
                f"mean ‖ν_base‖ = {mean_nu:.4f} world/s"
            )
    else:
        all_omega = [torch.zeros_like(m) for m in all_means]

    # ---------- temporal-stratified per-keyframe FPS -----------------------
    total_primitives = sum(m.shape[0] for m in all_means)
    if target_count is not None and total_primitives > target_count:
        counts = torch.tensor([m.shape[0] for m in all_means], dtype=torch.float32)
        # Linear weighting by per-keyframe contribution count.
        weights = counts

        allocations = (weights / weights.sum() * target_count).long()
        allocations[-1] += target_count - allocations.sum()

        keep_counts = [
            min(max(int(allocations[ki].item()), 0), int(all_means[ki].shape[0]))
            for ki in range(len(all_means))
        ]
        chunk_size = max(int(fps_chunk_size), 1)

        try:
            from trigs.compat.ops import sample_farthest_points  # macOS port (uses pytorch3d on CUDA if installed)

            for chunk_start in range(0, len(all_means), chunk_size):
                chunk_ids = list(range(chunk_start, min(chunk_start + chunk_size, len(all_means))))
                # Frames with 0 keep count are handled via empty indexing below.
                active_ids = [
                    ki for ki in chunk_ids
                    if 0 < keep_counts[ki] < int(all_means[ki].shape[0])
                ]
                if not active_ids:
                    continue

                p_max = max(int(all_means[ki].shape[0]) for ki in active_ids)
                B = len(active_ids)
                pts = torch.zeros((B, p_max, 3), device=dev, dtype=all_means[active_ids[0]].dtype)
                lengths = torch.zeros((B,), device=dev, dtype=torch.int64)
                k_tensor = torch.zeros((B,), device=dev, dtype=torch.int64)

                for bi, ki in enumerate(active_ids):
                    cur = all_means[ki]
                    n_i = int(cur.shape[0])
                    pts[bi, :n_i] = cur
                    lengths[bi] = n_i
                    k_tensor[bi] = keep_counts[ki]

                _, idx = sample_farthest_points(
                    pts,
                    lengths=lengths,
                    K=k_tensor,
                    random_start_point=False,
                )

                for bi, ki in enumerate(active_ids):
                    keep_idx = idx[bi, : keep_counts[ki]]
                    all_means[ki] = all_means[ki][keep_idx]
                    all_colors[ki] = all_colors[ki][keep_idx]
                    all_log_scales[ki] = all_log_scales[ki][keep_idx]
                    all_times[ki] = all_times[ki][keep_idx]
                    all_nu[ki] = all_nu[ki][keep_idx]
                    all_anchors[ki] = all_anchors[ki][keep_idx]
                    all_omega[ki] = all_omega[ki][keep_idx]

            # Apply explicit empty selections for frames allocated 0.
            for ki in range(len(all_means)):
                if keep_counts[ki] == 0 and all_means[ki].shape[0] > 0:
                    empty_idx = torch.empty((0,), device=dev, dtype=torch.long)
                    all_means[ki] = all_means[ki][empty_idx]
                    all_colors[ki] = all_colors[ki][empty_idx]
                    all_log_scales[ki] = all_log_scales[ki][empty_idx]
                    all_times[ki] = all_times[ki][empty_idx]
                    all_nu[ki] = all_nu[ki][empty_idx]
                    all_anchors[ki] = all_anchors[ki][empty_idx]
                    all_omega[ki] = all_omega[ki][empty_idx]

            if verbose:
                log.info(
                    f"  Stratified per-frame FPS: {total_primitives:,} → {target_count:,} "
                    f"(linear count weighting, chunk={chunk_size})"
                )
        except ImportError:
            for ki in range(len(all_means)):
                N_k = all_means[ki].shape[0]
                keep_count = keep_counts[ki]
                if keep_count < N_k:
                    keep_idx = torch.randperm(N_k, device=dev)[:keep_count]
                    all_means[ki] = all_means[ki][keep_idx]
                    all_colors[ki] = all_colors[ki][keep_idx]
                    all_log_scales[ki] = all_log_scales[ki][keep_idx]
                    all_times[ki] = all_times[ki][keep_idx]
                    all_nu[ki] = all_nu[ki][keep_idx]
                    all_anchors[ki] = all_anchors[ki][keep_idx]
                    all_omega[ki] = all_omega[ki][keep_idx]
            if verbose:
                log.info(
                    f"  Stratified random fallback (PyTorch3D unavailable): "
                    f"{total_primitives:,} → {target_count:,}"
                )

    # ---------- concatenate the super-set ---------------------------------
    means = torch.cat(all_means)
    colors = torch.cat(all_colors)
    scales = torch.cat(all_log_scales)
    times = torch.cat(all_times)
    nu_super = torch.cat(all_nu)
    anchors_super = torch.cat(all_anchors)
    omega_super = torch.cat(all_omega)

    N = means.shape[0]

    # ---------- pack -------------------------------------------------------
    quats = torch.zeros(N, 4, device=dev)
    quats[:, 0] = 1.0

    op = max(min(init_opacity, 1 - 1e-6), 1e-6)
    opacities = torch.full((N,), math.log(op / (1.0 - op)), device=dev)

    sh0 = (colors - 0.5) / _SH_C0
    shN = torch.zeros((N, 15, 3), device=dev)

    sparse_sigma_t_init = init_sigma_t if init_sigma_t > 0 else 1.0
    dense_sigma_t_init = 5.0 * sparse_sigma_t_init
    log_durations = torch.cat(
        [
            torch.full(
                (all_means[ki].shape[0], 1),
                math.log(
                    max(
                        dense_sigma_t_init if all_kinds[ki] == "dense" else sparse_sigma_t_init,
                        1e-6,
                    )
                ),
                device=dev,
            )
            for ki in range(len(all_means))
        ],
        dim=0,
    )

    def _zeros3() -> Tensor:
        return torch.zeros(N, 3, device=dev)

    params = nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "times": nn.Parameter(times, requires_grad=True),
        "log_durations": nn.Parameter(log_durations, requires_grad=True),
        "anchors": nn.Parameter(anchors_super.detach().clone(), requires_grad=True),
        "zeta_base_t": nn.Parameter(nu_super.detach().clone(), requires_grad=True),
        "zeta_base_r": nn.Parameter(omega_super.detach().clone(), requires_grad=True),
        "zeta_c0_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c0_r": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c1_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c1_r": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c2_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c2_r": nn.Parameter(_zeros3(), requires_grad=True),
    })

    scene_extent = float((
        means.max(dim=0).values - means.min(dim=0).values
    ).max().clamp(min=1e-6))
    return params, 0, scene_extent
