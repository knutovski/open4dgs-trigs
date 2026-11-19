"""TRiGS Trainer — coupled SE(3) motion with Bézier residuals.

Reconstructs the 4D Gaussian motion of a chunk from a keyframe subset (every
``keyframe_stride``-th dense frame). Each primitive carries:

* canonical geometry (mean, quat, scale, opacity, SH) taken from its keyframe,
* a temporal anchor ``µ_t`` = keyframe time and a temporal scale ``s_t``,
* a learnable local **anchor** ``a`` (gauge-fixed per TRiGS Eq. S8–S10),
* a Lie-algebra **base** coefficient ``ζ^base`` and three quadratic Bézier
  control points ``ζ^(0), ζ^(1), ζ^(2)`` (TRiGS Eq. 6–7).

At query time ``t`` we form
    ``ζ(t) = ζ^base + Bézier(τ(t))``,
gauge-fix the anchor / translation pair, build ``u = ζ·Δt`` and apply the
closed-form SE(3) exponential (``math/se3.py::exp_se3_quat``) to obtain a
relative rotation and translation. The deformed primitive is then

    ``µ(t) = R · (µ − a_⊥) + a_⊥ + p_∥``
    ``Σ(t) = R Σ Rᵀ`` (applied by rotating the canonical quaternion),
    ``α(t) = σ(α) · γ_i(t)``.

All equations cross-referenced in comments follow ``TriGS/sections/3_Method.tex``
and the supplement ``TriGS/sections/Suppl/3_MathMaticalFormulation.tex``.

Notation note: we store Lie-algebra vectors as ``xi = (ρ, ω)`` — translation
first, rotation second. The paper writes ``ζ = (ω, ν)``. Here ``ρ`` corresponds
to the paper's ``ν`` (translation coefficient); ``ω`` is the same.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch
from trigs.compat.device import resolve_device, empty_cache  # macOS port
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor
from tqdm import trange

from .losses import (
    bezier_motion_loss,
    compute_psnr,
    reconstruction_loss,
    rigid_coherence_loss,
)
from .optimizer_state import build_param_owners, remap_param_with_sliced_state
from ..math.quaternion import quaternion_multiply, quaternion_to_rotmat
from ..math.se3 import bezier_quadratic, exp_se3_quat, gauge_fix_omega, tau_from_window
from ..math.twist_init import fit_local_twist, local_knn_indices

if TYPE_CHECKING:
    from .camera import CameraConfig
    from .transforms import TransformConfig


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrigsTrainingConfig:
    """Configuration for TRiGS training."""

    # --- Keyframe subsampling ---
    keyframe_stride: int = 3
    """Take every N-th dense frame as a canonical keyframe. `1` keeps all."""

    # --- Temporal unit ---
    fps: float = 30.0
    """Sequence frame rate. Time is in SECONDS: ``t = frame_idx / fps``.
    All per-time quantities (ν, ω, σ_t, Bézier controls, loss magnitudes)
    are in physical units and are therefore sequence-length-invariant, so
    paper hyperparameters transfer across sequences of different lengths
    and FPS without retuning."""

    # --- 4D FPS reduction to a fixed canonical budget ---
    target_count: int | None = None
    """If set, reduce concatenated keyframes to this many primitives via 4D
    farthest-point sampling. ``None`` keeps all (may blow memory)."""

    fps_time_weight: float = 1.0
    """``λ_t`` for 4D FPS distance: ``‖Δp‖² + λ_t² · Δt²`` with ``t ∈ [0, 1]``.
    Scaled by the scene extent so time has comparable weight to space."""

    delayed_prune_target_count: int | None = None
    """Optional target budget applied *during* training.

    Intended for pre-built canonical sets (e.g. SelfCap mode): keep all
    initialized primitives at first, then periodically prune by opacity until
    the count reaches this budget. Relocation is disabled during this phase.
    """

    delayed_prune_start_iter: int = 0
    """First step where periodic delayed pruning may run."""

    delayed_prune_every: int = 100
    """Run delayed pruning every N steps while above the target budget."""

    # --- What to train ---
    freeze_geometry: bool = False
    """Train canonical means/quats/scales/SH along with motion. This matches
    TriGS, which is a full 3DGS + motion extension (the paper trains every
    primitive parameter). Set True only for ablations."""

    # --- Temporal init ---
    init_sigma_t: float = 0.0
    """σ_t at init, in SECONDS. ``0`` auto-computes as
    ``keyframe_stride / fps`` (one keyframe gap wide), giving an effective
    window ``4 σ_t`` = 4 keyframe gaps — enough cross-keyframe gradient
    without committing to a window too wide for the quadratic Bézier."""

    # --- Rough-motion init ---
    nn_velocity_init: bool = True
    """Initialize ν_base via inter-keyframe k-NN matching. Each primitive
    matches to its ``nn_velocity_k`` nearest neighbors in the *next* keyframe
    bucket; velocity = (mean neighbor displacement) / keyframe gap. Gives
    the model rough linear motion before any optimization — helps on long
    sequences where zero-init leaves primitives disconnected from their
    trajectories."""

    nn_velocity_k: int = 3
    """k for the k-NN velocity init. K=1 is fragile against single
    bad matches; averaging over ~5 neighbors cuts noise from topology
    changes and sparse regions without much bias."""

    nn_anchor_init: bool = True
    """Initialize each primitive's anchor ``a_i`` to the centroid of all
    neighbors within a radius ball (``anchor_radius_frac × extent``) in the
    **same keyframe's dense set**. With ``anchor = means[i]``, rotation is
    a spin-in-place on the covariance and cannot produce articulated motion.
    The radius-ball centroid displaces the anchor by the local neighborhood
    scale, so growing ``ω`` produces translation of ``µ_i`` around a nearby
    pivot — critical for arm swings / limb articulation."""

    anchor_radius_frac: float = 0.1
    """Fraction of per-keyframe scene extent used as the radius for anchor
    initialization when ``nn_anchor_init`` is True."""

    omega_init: bool = False
    """Initialize base motion with a local weighted rigid-twist fit.

    For each primitive i in a keyframe neighborhood P(i), solve
    ``min_{ν_i,ω_i} Σ_j w_ij ||v_j − ν_i − ω_i×(x_j−a_i)||²`` and then
    gauge-fix ν with :func:`gauge_fix_omega`. This is less noisy than the
    pointwise cross-product heuristic and better matches the TRiGS local
    rigid base-motion assumption.
    """

    local_twist_k: int = 16
    """Neighborhood size K for local rigid-twist fitting at init."""

    local_twist_reg: float = 1e-4
    """Tikhonov regularization added to each local normal equation."""

    # --- Training ---
    n_steps: int = 30_000

    unfreeze_temporal_step: int = 0
    """Keep ``opacities``, ``times``, and ``log_durations`` frozen for this
    many steps. Forces the model to learn motion before it can fit to
    individual frames by adjusting temporal visibility or opacity. Set 0
    to disable (train everything from step 0)."""

    # --- Learning rates ---
    # Convention (matches 3DGS): unit-scale values stored here; LRs on
    # scene-sized parameters (means, anchor, ν components of ζ) are
    # multiplied by the canonical scene extent at optimizer build time.
    #
    # Translation components of ζ carry units of (world / s),
    # so they behave like FTGS velocities and get scene-scaled too.
    # Rotation components of ζ carry units of (rad / s) and
    # do not depend on scene size.
    lr_opacity: float = 5e-2
    lr_times: float = 1.6e-4
    lr_durations: float = 5e-3

    lr_anchor: float = 1.6e-4                 # scene-scaled (same as means)
    lr_means: float = 1.6e-4                  # scene-scaled
    lr_zeta_base_trans: float = 1.6e-4          # scene-scaled (ν_base)
    lr_zeta_base_rot: float = 1e-3            # unscaled (ω_base, 3DGS rotation LR)
    lr_zeta_bezier_trans: float = 1.6e-4        # scene-scaled (ν^(k))
    lr_zeta_bezier_rot: float = 1e-3          # unscaled (ω^(k))

    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20.0

    # --- Exponential LR decay (3DGS convention: positional LRs schedule) ---
    # `lr(step) = lr_init · (lr_final / lr_init)^(step / n_steps)`.
    # Only the four scene-scaled positional groups decay; rotation / non-
    # positional LRs stay constant as in 3DGS.
    lr_anchor_final: float = 1.6e-6            # = 1.6e-4 · 0.01
    lr_means_final: float = 1.6e-6
    lr_zeta_base_trans_final: float = 1.6e-6     # = 5e-3 · 0.01
    lr_zeta_bezier_trans_final: float = 1e-5

    # --- Loss weights (TRiGS Sec. 4.1 — exact paper values) ---
    lambda_render: float = 1.0
    ssim_lambda: float = 0.2
    lambda_motion: float = 1e-4           # Eq. 11 — Bézier acceleration
    lambda_rigid: float = 1.0             # Eq. 12 — k-NN base coherence
    lambda_reg: float = 1e-2              # FTGS-style L_reg weight
    lambda_sigma_floor: float = 0.0
    """Weight on the one-sided σ_t floor regularizer. Penalty is
    quadratic below ``sigma_floor_target``, zero above. Necessary for
    segmented-dynamic scenes where no static-background primitive pool
    naturally anchors the wide-σ_t population — without it, photometric
    pressure drives every σ_t to collapse."""

    sigma_floor_target: float = 0.0
    """Target σ_t floor, in SECONDS. ``0`` auto-computes as
    ``keyframe_stride / fps`` (matches the default σ_t init). Set
    explicitly to override."""

    # --- Rigid k-NN graph (TRiGS Sec. 4.1: K = 3, λ_c = 50) ---
    knn_k: int = 3
    knn_recompute_every: int = 100
    rigid_color_lambda: float = 50.0       # λ_c (SH-DC color affinity)

    rigid_knn_time_weight: float = 0.0
    """Scale factor for time in the k-NN distance metric used for L_rigid.
    The graph is built over ``[x, y, z, w·t]`` where
    ``w = rigid_knn_time_weight · scene_extent``. With default 0.0, k-NN is
    spatial-only on canonical means (paper-faithful Eq. 12 neighborhood). A
    value of 1.0 makes a time delta of 1.0 (full sequence) cost the same
    Euclidean distance as a
    spatial delta of ``scene_extent`` — cross-time neighbors get pruned
    from each primitive's rigid-coherence group. Lower (e.g. 0.2)
    for slow-motion scenes where cross-time primitives can legitimately
    share motion; higher (5+) for fast articulated motion."""

    # --- Gauge-fix numerical safety (TRiGS Eq. S9/S10) ---
    gauge_eps: float = 1e-8
    gauge_denom_clamp: float = 1e-10

    # --- Motion-guided relocation (TRiGS §3.4 + Suppl §4) ---
    relocate: bool = True
    relocate_every: int = 100
    relocate_start_iter: int = 500
    relocate_tau_alpha: float = 0.005      # Sec. 4.1: τ_α = 0.005
    relocate_scale_factor: float = 0.66    # Suppl: 66 % of original scale

    # --- Numerical safety ---
    clamp_mu_t: bool = True
    """Post-step clamp of central times µ_t to the physical range
    ``[0, (n_frames - 1) / fps]`` in seconds.
    Drifted primitives contribute ~0 visibility at every real frame; the
    clamp reclaims that capacity without changing gradient semantics (L_reg,
    relocation cues, and photometric loss all see the clamped value on the
    next forward)."""

    opacity_logit_upper_clamp: float | None = 6.0
    """Optional post-step upper clamp on opacity logits.

    Keeps sigmoid(opacity) out of the saturated high-opacity regime so the
    regularizer retains gradient signal. ``None`` disables this clamp.
    """

    opacity_logit_upper_clamp_until_step: int = 25_000
    """Last iteration (inclusive) where opacity-logit upper clamping is active.

    Use a negative value to keep the clamp active for all training steps.
    """

    # --- Misc ---
    verbose: bool = True

    # --- Logging ---
    tb_log_every: int = 50
    """Push scalar losses to tensorboard every N steps (0 disables)."""

    tb_stats_every: int = 500
    """Push motion/temporal distribution stats every N steps (0 disables)."""


@dataclass
class TrigsTrainingResult:
    """Return payload of :meth:`TrigsTrainer.train`."""

    params: nn.ParameterDict
    temporal_params: Dict[str, Tensor]
    motion_params: Dict[str, Tensor]
    metrics: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Service
# =============================================================================

class TrigsTrainer:
    """TRiGS coupled-SE(3) + Bézier-residual 4D trainer.

    Canonical primitive count is fixed: there is no densification and no ADMM
    pruning. Wasted capacity is recycled by motion-guided relocation.
    """

    def __init__(self, device: str = "auto"):
        self.device = resolve_device(device)

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    def train(
        self,
        ply_files: List[Path],
        cameras: "CameraConfig",
        config: TrigsTrainingConfig,
        transform_config: "TransformConfig",
        gt_images_per_frame: Optional[Dict[int, List[Tensor]]] = None,
        gt_alphas_per_frame: Optional[Dict[int, List[Tensor]]] = None,
        writer: Optional[Any] = None,
    ) -> TrigsTrainingResult:
        """Run TRiGS training over the given PLY sequence.

        Args:
            ply_files: Dense per-frame PLYs in temporal order.
            cameras: Multi-view camera configuration.
            config: TRiGS training config.
            transform_config: Scene transform applied to loaded GSTensors.
            gt_images_per_frame / gt_alphas_per_frame: Optional pre-rendered GT.
                When missing, GT is rendered from the input PLYs (uint8 on GPU).
        """
        n_frames = len(ply_files)
        if n_frames < 1:
            raise ValueError("Need at least 1 frame to train")

        # -- Canonical init from keyframes -----------------------------------
        params, sh_degree, scene_extent = self._build_canonical_set(
            ply_files, config, transform_config, n_frames,
        )
        n_primitives = params["means"].shape[0]

        if config.verbose:
            print(
                f"TrigsTrainer: {n_primitives:,} canonical primitives, "
                f"{n_frames} frames, stride={config.keyframe_stride}, "
                f"scene_extent={scene_extent:.3f}"
            )

        # NN ν_base init is now performed inside _build_canonical_set on
        # the *dense* keyframes (before FPS). The standalone
        # `_nn_velocity_init` method is retained for tests / manual re-seeding
        # on an already-built parameter dict.

        # -- GT cache --------------------------------------------------------
        if gt_images_per_frame is None or gt_alphas_per_frame is None:
            gt_images_per_frame, gt_alphas_per_frame = self._render_gt(
                ply_files, cameras, transform_config, sh_degree,
                verbose=config.verbose,
            )

        # -- Optimizers + k-NN graph -----------------------------------------
        optimizers = self._build_optimizers(params, config, scene_extent)
        # Paper-faithful Eq. 12 neighborhood: k-NN in canonical 3D means only.
        nn_indices = self._compute_knn(
            params["means"].detach(),
            config.knn_k,
            times=None,
            time_weight=0.0,
        )

        # Resolve σ_t floor target in SECONDS (auto = stride / fps).
        if config.sigma_floor_target > 0:
            sigma_floor_target = config.sigma_floor_target
        else:
            sigma_floor_target = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)
        if config.verbose:
            print(
                f"  σ_t floor target = {sigma_floor_target:.4f} s "
                f"(λ_sigma_floor = {config.lambda_sigma_floor})"
            )

        # -- Train -----------------------------------------------------------
        metrics = self._train_loop(
            params, optimizers, nn_indices,
            cameras, n_frames, sh_degree,
            gt_images_per_frame, gt_alphas_per_frame,
            config,
            scene_extent=scene_extent,
            sigma_floor_target=sigma_floor_target,
            writer=writer,
        )

        return self._package_result(params, metrics)

    # -------------------------------------------------------------------------
    # Entry point — pre-built canonical params (COLMAP / ROMA flow)
    # -------------------------------------------------------------------------

    def train_from_params(
        self,
        params: nn.ParameterDict,
        sh_degree: int,
        scene_extent: float,
        cameras: "CameraConfig",
        n_frames: int,
        gt_images_per_frame: Dict[int, List[Tensor]],
        gt_alphas_per_frame: Dict[int, List[Tensor]],
        config: TrigsTrainingConfig,
        gt_times_per_frame: Optional[Dict[int, List[float]]] = None,
        writer: Optional[Any] = None,
    ) -> TrigsTrainingResult:
        """Run TRiGS training starting from a pre-built canonical parameter dict.

        Used by the COLMAP + ROMA init flow where the canonical set is
        constructed from :func:`trigs.init.from_colmap.build_canonical_from_npz`
        rather than from dense PLYs. The caller is responsible for supplying
        GT images (typically loaded from :class:`trigs.io.FreeTimeDataset`).
        """
        n_primitives = params["means"].shape[0]
        if config.verbose:
            print(
                f"TrigsTrainer: {n_primitives:,} canonical primitives (pre-built), "
                f"{n_frames} frames, scene_extent={scene_extent:.3f}"
            )

        optimizers = self._build_optimizers(params, config, scene_extent)
        nn_indices = self._compute_knn(
            params["means"].detach(),
            config.knn_k,
            times=None,
            time_weight=0.0,
        )

        if config.sigma_floor_target > 0:
            sigma_floor_target = config.sigma_floor_target
        else:
            sigma_floor_target = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)

        metrics = self._train_loop(
            params, optimizers, nn_indices,
            cameras, n_frames, sh_degree,
            gt_images_per_frame, gt_alphas_per_frame,
            config,
            scene_extent=scene_extent,
            gt_times_per_frame=gt_times_per_frame,
            sigma_floor_target=sigma_floor_target,
            writer=writer,
        )
        return self._package_result(params, metrics)

    def _package_result(
        self,
        params: nn.ParameterDict,
        metrics: Dict[str, Any],
    ) -> TrigsTrainingResult:
        """Assemble the training result payload (temporal + motion tensors)."""
        temporal_params = {
            "times": params["times"].detach().cpu(),
            "log_durations": params["log_durations"].detach().cpu(),
        }
        motion_params = {
            "anchors": params["anchors"].detach().cpu(),
            "zeta_base_t": params["zeta_base_t"].detach().cpu(),
            "zeta_base_r": params["zeta_base_r"].detach().cpu(),
            "zeta_c0_t": params["zeta_c0_t"].detach().cpu(),
            "zeta_c0_r": params["zeta_c0_r"].detach().cpu(),
            "zeta_c1_t": params["zeta_c1_t"].detach().cpu(),
            "zeta_c1_r": params["zeta_c1_r"].detach().cpu(),
            "zeta_c2_t": params["zeta_c2_t"].detach().cpu(),
            "zeta_c2_r": params["zeta_c2_r"].detach().cpu(),
        }
        return TrigsTrainingResult(
            params=params,
            temporal_params=temporal_params,
            motion_params=motion_params,
            metrics=metrics,
        )

    # -------------------------------------------------------------------------
    # Canonical set construction
    # -------------------------------------------------------------------------

    def _build_canonical_set(
        self,
        ply_files: List[Path],
        config: TrigsTrainingConfig,
        transform_config: "TransformConfig",
        n_frames: int,
    ) -> tuple[nn.ParameterDict, int, float]:
        """Load keyframes and concatenate their Gaussians into the canonical set.

        Every ``config.keyframe_stride``-th frame contributes its entire set of
        primitives. Each primitive inherits that keyframe's time as its ``µ_t``.
        """
        from ..io.ply import load_ply

        keyframe_indices = list(range(0, n_frames, max(config.keyframe_stride, 1)))

        all_means, all_quats, all_scales = [], [], []
        all_opacities, all_sh0, all_shN = [], [], []
        all_times = []
        per_kf_t: List[float] = []          # per-keyframe normalized time

        has_shN = False
        for kf_idx in keyframe_indices:
            gs = load_ply(ply_files[kf_idx], device=str(self.device))
            gs = transform_config.apply(gs)
            has_shN = has_shN or (gs.shN is not None)

            n = len(gs)
            # Time in SECONDS (physical units, FPS-independent). Paper
            # hyperparameters (λ_rigid, λ_motion, σ_t, ...) stay valid
            # across sequences of different length and frame rate.
            t_sec = float(kf_idx) / max(config.fps, 1e-6)

            all_means.append(gs.means)
            all_quats.append(gs.quats)
            all_scales.append(gs.scales)
            all_opacities.append(gs.opacities)
            all_sh0.append(gs.sh0)
            if gs.shN is not None:
                all_shN.append(gs.shN)
            all_times.append(torch.full((n, 1), t_sec, device=self.device))
            per_kf_t.append(t_sec)
            del gs

        # ---------------------------------------------------------------
        # Centered finite-difference ν init (±1 dense frames, before FPS)
        # ---------------------------------------------------------------
        # For each keyframe at frame f_k, match to dense frames f_k−1 and
        # f_k+1 via k-NN, then average forward/backward velocities for a
        # centered estimate. Boundary keyframes use one-sided FD.
        all_nu: List[Tensor] = []
        if config.nn_velocity_init and len(all_means) >= 1:
            k_nn = max(1, config.nn_velocity_k)
            dt = 1.0 / max(config.fps, 1e-6)  # seconds between adjacent frames
            for ki in range(len(all_means)):
                src = all_means[ki]
                if src.shape[0] == 0:
                    all_nu.append(torch.zeros_like(src))
                    continue
                f_k = keyframe_indices[ki]
                v_fwd: Tensor | None = None
                v_bwd: Tensor | None = None
                if f_k + 1 < n_frames:
                    gs_next = load_ply(
                        ply_files[f_k + 1], device=str(self.device),
                    )
                    gs_next = transform_config.apply(gs_next)
                    nn_next = self._knn_mean_positions(src, gs_next.means, k_nn)
                    v_fwd = (nn_next - src) / dt
                    del gs_next
                if f_k - 1 >= 0:
                    gs_prev = load_ply(
                        ply_files[f_k - 1], device=str(self.device),
                    )
                    gs_prev = transform_config.apply(gs_prev)
                    nn_prev = self._knn_mean_positions(src, gs_prev.means, k_nn)
                    v_bwd = (src - nn_prev) / dt
                    del gs_prev
                if v_fwd is not None and v_bwd is not None:
                    all_nu.append((v_fwd + v_bwd) / 2.0)
                elif v_fwd is not None:
                    all_nu.append(v_fwd)
                elif v_bwd is not None:
                    all_nu.append(v_bwd)
                else:
                    all_nu.append(torch.zeros_like(src))
            if config.verbose:
                mean_mag = torch.cat(all_nu).norm(dim=-1).mean().item()
                print(
                    f"  Centered FD ν init (K={k_nn}) on "
                    f"{sum(m.shape[0] for m in all_means):,} primitives: "
                    f"mean ‖ν‖ = {mean_mag:.4f} world/s"
                )
        else:
            all_nu = [torch.zeros_like(m) for m in all_means]

        # ---------------------------------------------------------------
        # Radius-ball anchor init (per-keyframe, before FPS).
        # ---------------------------------------------------------------
        # For each primitive i in keyframe k, anchor_i = centroid of all
        # neighbors within ``anchor_radius_frac × extent`` of the same
        # keyframe. With dense Gaussians, k-NN (k=3) neighbors are
        # essentially co-located so the anchor ≈ µ — useless for rotation.
        # A 10 % radius ball captures the local region scale, displacing
        # the anchor enough that rotation produces real displacement of µ.
        all_anchors: List[Tensor] = []
        if config.nn_anchor_init:
            for means_ki in all_means:
                if means_ki.shape[0] <= 1:
                    all_anchors.append(means_ki.clone())
                    continue
                ext = (
                    means_ki.max(dim=0).values - means_ki.min(dim=0).values
                ).max().clamp(min=1e-6)
                radius = config.anchor_radius_frac * ext.item()
                all_anchors.append(self._radius_ball_centroid(means_ki, radius))
            if config.verbose:
                disp_mag = (torch.cat(all_anchors) - torch.cat(all_means)).norm(dim=-1)
                print(
                    f"  Radius-ball anchor init "
                    f"(r={config.anchor_radius_frac:.0%} extent): "
                    f"mean |a − µ| = {disp_mag.mean().item():.4f} world units"
                )
        else:
            all_anchors = [m.clone() for m in all_means]

        # ---------------------------------------------------------------
        # ω / ν init from local weighted rigid-twist fit (before FPS).
        # ---------------------------------------------------------------
        all_omega: List[Tensor] = []
        if config.omega_init:
            for ki in range(len(all_means)):
                means_ki = all_means[ki]
                v_targets_ki = all_nu[ki]
                anchors_ki = all_anchors[ki]
                if means_ki.shape[0] <= 1:
                    all_omega.append(torch.zeros_like(means_ki))
                    continue
                nn_idx = local_knn_indices(means_ki, config.local_twist_k)
                nu_ki, omega_ki, all_anchors[ki] = fit_local_twist(
                    means_ki,
                    v_targets_ki,
                    anchors_ki,
                    nn_idx,
                    reg=config.local_twist_reg,
                    gauge_eps=config.gauge_eps,
                    gauge_denom_clamp=config.gauge_denom_clamp,
                )
                all_omega.append(omega_ki)
                all_nu[ki] = nu_ki
            if config.verbose:
                mean_omega = torch.cat(all_omega).norm(dim=-1).mean().item()
                mean_nu_base = torch.cat(all_nu).norm(dim=-1).mean().item()
                print(
                    f"  Local twist init (K={config.local_twist_k}, "
                    f"reg={config.local_twist_reg:g}): "
                    f"mean ‖ω‖ = {mean_omega:.4f} rad/s, "
                    f"mean ‖ν_base‖ = {mean_nu_base:.4f} world/s"
                )
        else:
            all_omega = [torch.zeros_like(m) for m in all_means]

        # Concatenate the full super-set.
        means = torch.cat(all_means)
        quats = torch.cat(all_quats)
        scales = torch.cat(all_scales)
        opacities = torch.cat(all_opacities)
        sh0 = torch.cat(all_sh0)
        times = torch.cat(all_times)
        nu_super = torch.cat(all_nu)
        anchors_super = torch.cat(all_anchors)
        omega_super = torch.cat(all_omega)
        shN_cat = torch.cat(all_shN) if (has_shN and len(all_shN) == len(all_means)) else None

        # 4D FPS reduction (TriGS-style fixed budget). Distance is
        #   d² = ‖Δ means‖² + (λ_t · extent · Δ t_norm)²
        # where t_norm ∈ [0, 1] is time normalized for FPS only; outside
        # this block ``times`` stays in seconds. At λ_t = 1, time and
        # space contribute on the same scale.
        if config.target_count is not None and means.shape[0] > config.target_count:
            scene_extent = (means.max(dim=0).values - means.min(dim=0).values).max()
            scene_extent = scene_extent.clamp(min=1e-6)
            t_total_sec = max(n_frames - 1, 1) / max(config.fps, 1e-6)
            t_norm_fps = times / t_total_sec
            pts4d = torch.cat(
                [means, config.fps_time_weight * scene_extent * t_norm_fps],
                dim=-1,
            ).unsqueeze(0)  # [1, N, 4]
            from trigs.compat.ops import sample_farthest_points  # macOS port (uses pytorch3d on CUDA if installed)
            _, idx = sample_farthest_points(
                pts4d, K=config.target_count, random_start_point=False,
            )
            keep = idx.squeeze(0)
            if config.verbose:
                print(
                    f"  4D FPS: {means.shape[0]:,} → {config.target_count:,} "
                    f"(λ_t={config.fps_time_weight}, extent={scene_extent.item():.3f})"
                )
            means = means[keep]
            quats = quats[keep]
            scales = scales[keep]
            opacities = opacities[keep]
            sh0 = sh0[keep]
            times = times[keep]
            nu_super = nu_super[keep]
            anchors_super = anchors_super[keep]
            omega_super = omega_super[keep]
            if shN_cat is not None:
                shN_cat = shN_cat[keep]

        N = means.shape[0]

        # σ_t init in SECONDS. 0 ⇒ auto: keyframe_stride / fps
        # (= one keyframe gap wide, window 4σ_t = 4 keyframe gaps).
        if config.init_sigma_t > 0:
            sigma_t_init = config.init_sigma_t
        else:
            sigma_t_init = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)
        log_durations = torch.full(
            (N, 1), math.log(max(sigma_t_init, 1e-6)), device=self.device,
        )

        # Motion parameters — ν_base seeded from centered FD (residual after
        # ω extraction), ω_base from cross-product init, anchors from
        # radius-ball centroids; Bézier controls zero at init. Translation
        # and rotation halves stored separately for independent LR scaling.
        anchors = anchors_super.detach().clone()
        def _zeros3():
            return torch.zeros(N, 3, device=self.device)

        geometry_trainable = not config.freeze_geometry
        params = nn.ParameterDict({
            # Canonical geometry (frozen by default).
            "means": nn.Parameter(means, requires_grad=geometry_trainable),
            "quats": nn.Parameter(quats, requires_grad=geometry_trainable),
            "scales": nn.Parameter(scales, requires_grad=geometry_trainable),
            "sh0": nn.Parameter(sh0, requires_grad=geometry_trainable),
            # Opacity always trainable.
            "opacities": nn.Parameter(opacities, requires_grad=True),
            # Temporal.
            "times": nn.Parameter(times, requires_grad=True),
            "log_durations": nn.Parameter(log_durations, requires_grad=True),
            # Motion (TRiGS §3). Split ν (translation) / ω (rotation) halves.
            "anchors": nn.Parameter(anchors, requires_grad=True),
            "zeta_base_t": nn.Parameter(nu_super.detach().clone(), requires_grad=True),
            "zeta_base_r": nn.Parameter(omega_super.detach().clone(), requires_grad=True),
            "zeta_c0_t":   nn.Parameter(_zeros3(), requires_grad=True),
            "zeta_c0_r":   nn.Parameter(_zeros3(), requires_grad=True),
            "zeta_c1_t":   nn.Parameter(_zeros3(), requires_grad=True),
            "zeta_c1_r":   nn.Parameter(_zeros3(), requires_grad=True),
            "zeta_c2_t":   nn.Parameter(_zeros3(), requires_grad=True),
            "zeta_c2_r":   nn.Parameter(_zeros3(), requires_grad=True),
        })
        if shN_cat is not None:
            params["shN"] = nn.Parameter(shN_cat, requires_grad=geometry_trainable)

        sh_degree = 0
        if has_shN and "shN" in params:
            # sh_degree is inferred from shN shape (N, (deg+1)²-1, 3).
            n_bands = params["shN"].shape[1] + 1
            sh_degree = int(math.isqrt(n_bands) - 1)

        # Scene extent on the canonical set — used to scene-scale positional LRs.
        scene_extent = float((
            means.max(dim=0).values - means.min(dim=0).values
        ).max().clamp(min=1e-6))

        return params, sh_degree, scene_extent

    # -------------------------------------------------------------------------
    # Forward deformation — TRiGS Eq. 4–9
    # -------------------------------------------------------------------------

    def _deform(
        self,
        params: nn.ParameterDict,
        t: float,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (means_t, quats_t, temporal_opacity) at query time ``t``.

        Implements TRiGS Eqs. 4–9 end-to-end.
        """
        device = params["means"].device

        mu_t = params["times"]                                      # [N, 1]
        log_sigma_t = params["log_durations"]                       # [N, 1]
        t_scalar = torch.tensor(t, device=device, dtype=mu_t.dtype)

        # Eq. 5 — clamped normalized local time.
        tau = tau_from_window(t_scalar, mu_t, log_sigma_t)           # [N, 1]

        # Eq. 6/7 — ζ(t) = ζ^base + ζ_res(τ), with translation (ν) and
        # rotation (ω) halves stored as separate [N, 3] tensors.
        nu_res = bezier_quadratic(
            tau, params["zeta_c0_t"], params["zeta_c1_t"], params["zeta_c2_t"],
        )
        omega_res = bezier_quadratic(
            tau, params["zeta_c0_r"], params["zeta_c1_r"], params["zeta_c2_r"],
        )
        nu = params["zeta_base_t"] + nu_res
        omega = params["zeta_base_r"] + omega_res

        # Eq. S9/S10 — gauge fix.
        a_perp, nu_par = gauge_fix_omega(
            omega, nu, params["anchors"],
            eps=config.gauge_eps, denom_clamp=config.gauge_denom_clamp,
        )

        # Eq. 3 — log-parameter u = ζ·Δt.
        dt = t_scalar - mu_t                                         # [N, 1]
        u = torch.cat([nu_par * dt, omega * dt], dim=-1)             # [N, 6]

        # Eq. S1/S2/S3 via closed-form quaternion exp.
        q_rel, p_par = exp_se3_quat(u)                               # [N, 4], [N, 3]

        # Eq. 9 — deformed mean.
        R = quaternion_to_rotmat(q_rel)                              # [N, 3, 3]
        centered = (params["means"] - a_perp).unsqueeze(-1)           # [N, 3, 1]
        means_t = (R @ centered).squeeze(-1) + a_perp + p_par         # [N, 3]

        # Covariance rotation: Σ(t) = R Σ Rᵀ is absorbed by rotating the
        # canonical quaternion q → q_rel ⊗ q.
        quats_t = quaternion_multiply(q_rel, F.normalize(params["quats"], dim=-1))

        # Eq. 1 — temporal opacity γ(t) = exp(−(t − µ_t)² / (2 σ_t²)).
        # No artificial floor on σ_t: the paper lets it collapse freely and
        # relies on motion-guided relocation to recycle collapsed primitives.
        sigma_t = torch.exp(log_sigma_t).clamp(min=1e-6)
        gamma = torch.exp(-0.5 * (dt / sigma_t) ** 2)                # [N, 1]

        return means_t, quats_t, gamma

    # -------------------------------------------------------------------------
    # Rasterization pass
    # -------------------------------------------------------------------------

    def _render(
        self,
        params: nn.ParameterDict,
        cameras: "CameraConfig",
        cam_idx: int,
        t: float,
        sh_degree: int,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor, Dict]:
        means_t, quats_t, gamma = self._deform(params, t, config)

        base_opa = torch.sigmoid(params["opacities"])
        if base_opa.dim() == 1:
            base_opa = base_opa.unsqueeze(-1)
        effective_opa = torch.clamp(base_opa * gamma, min=1e-10).squeeze(-1)

        scales = torch.exp(params["scales"])
        sh0 = params["sh0"]
        shN = params.get("shN")
        if sh_degree > 0 and shN is not None:
            colors = torch.cat([sh0.unsqueeze(1), shN], dim=1)
        else:
            colors = sh0.unsqueeze(1)
            sh_degree = 0

        render_colors, render_alphas, info = rasterization(
            means=means_t,
            quats=quats_t,
            scales=scales,
            opacities=effective_opa,
            colors=colors,
            viewmats=cameras.viewmats[cam_idx:cam_idx + 1],
            Ks=cameras.Ks[cam_idx:cam_idx + 1],
            width=cameras.width_for(cam_idx),
            height=cameras.height_for(cam_idx),
            sh_degree=sh_degree,
            packed=False,
        )
        # Retain grad on the 2D projected means so the motion-guided
        # relocation's ∇_2D µ_i cue (TRiGS Suppl §4) can read it after
        # backward. It is a non-leaf tensor; without this, .grad is None.
        if info.get("means2d") is not None and info["means2d"].requires_grad:
            info["means2d"].retain_grad()
        return render_colors, render_alphas, gamma, info

    # -------------------------------------------------------------------------
    # GT caching (uint8 on GPU)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _render_gt(
        self,
        ply_files: List[Path],
        cameras: "CameraConfig",
        transform_config: "TransformConfig",
        sh_degree: int,
        verbose: bool,
    ) -> tuple[Dict[int, List[Tensor]], Dict[int, List[Tensor]]]:
        from ..io.ply import load_ply
        from .renderer import Renderer

        renderer = Renderer(device=str(self.device))
        gt_images: Dict[int, List[Tensor]] = {}
        gt_alphas: Dict[int, List[Tensor]] = {}
        if verbose:
            print("  Rendering GT from dense PLYs (uint8 cache on GPU)...")
        for fi, ply_file in enumerate(ply_files):
            gs = load_ply(ply_file, device=str(self.device))
            gs = transform_config.apply(gs)
            result = renderer.render(
                gs, cameras, return_alphas=True, show_progress=False,
            )
            gt_images[fi] = [
                img.clamp(0, 1).mul(255).to(torch.uint8).cpu()
                for img in result.images
            ]
            gt_alphas[fi] = [
                a.clamp(0, 1).mul(255).to(torch.uint8).cpu()
                for a in result.alphas
            ]
            del gs, result
        empty_cache()
        return gt_images, gt_alphas

    # -------------------------------------------------------------------------
    # Optimizers
    # -------------------------------------------------------------------------

    def _build_optimizers(
        self,
        params: nn.ParameterDict,
        config: TrigsTrainingConfig,
        scene_extent: float,
    ) -> Dict[str, torch.optim.Optimizer]:
        """Per-parameter Adam optimizers.

        Scene-scaled LRs (multiplied by ``scene_extent``):
          - ``means``, ``anchors`` — world-space positions;
          - ``zeta_{base,c0,c1,c2}_t`` — translation coefficients ν (world/time).

        Unscaled LRs (standard 3DGS convention):
          - ``opacities``, ``scales``, ``quats``, ``sh0``, ``shN``;
          - ``zeta_{base,c0,c1,c2}_r`` — rotation coefficients ω (rad/time);
          - ``times``, ``log_durations`` — dimensionless.
        """
        def adam(ps, lr):
            return torch.optim.Adam(ps, lr=lr, eps=1e-15)

        optimizers: Dict[str, torch.optim.Optimizer] = {
            "opacities": adam([params["opacities"]], config.lr_opacity),
            "times": adam([params["times"]], config.lr_times),
            "log_durations": adam([params["log_durations"]], config.lr_durations),
            # Positional — scene-scaled.
            "anchors": adam([params["anchors"]], config.lr_anchor * scene_extent),
            # ζ translation halves — scene-scaled (like FTGS velocities).
            "zeta_base_t": adam([params["zeta_base_t"]], config.lr_zeta_base_trans * scene_extent),
            "zeta_bezier_t": adam(
                [params["zeta_c0_t"], params["zeta_c1_t"], params["zeta_c2_t"]],
                config.lr_zeta_bezier_trans * scene_extent,
            ),
            # ζ rotation halves — unscaled (rad/time, independent of scene size).
            "zeta_base_r": adam([params["zeta_base_r"]], config.lr_zeta_base_rot),
            "zeta_bezier_r": adam(
                [params["zeta_c0_r"], params["zeta_c1_r"], params["zeta_c2_r"]],
                config.lr_zeta_bezier_rot,
            ),
        }
        if not config.freeze_geometry:
            optimizers["means"] = adam([params["means"]], config.lr_means * scene_extent)
            optimizers["quats"] = adam([params["quats"]], config.lr_quats)
            optimizers["scales"] = adam([params["scales"]], config.lr_scales)
            optimizers["sh0"] = adam([params["sh0"]], config.lr_sh0)
            if "shN" in params:
                optimizers["shN"] = adam([params["shN"]], config.lr_shN)
        return optimizers

    # -------------------------------------------------------------------------
    # NN velocity init — rough linear motion from inter-keyframe matching
    # -------------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _knn_mean_positions(src: Tensor, tgt: Tensor, k: int) -> Tensor:
        """For each row of ``src``, return the mean 3-D position of its ``k``
        nearest neighbors in ``tgt``.

        Args:
            src: [Ns, 3] query points.
            tgt: [Nt, 3] target points.
            k: Number of neighbors to average (clamped to ``min(k, Nt)``).

        Returns:
            [Ns, 3] per-source centroid of matched neighbors.
        """
        k = max(1, min(k, tgt.shape[0]))
        try:
            from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)
            res = knn_points(src.unsqueeze(0), tgt.unsqueeze(0), K=k)
            nn_idx = res.idx.squeeze(0)                  # [Ns, k]
        except ImportError:
            d = torch.cdist(src, tgt)
            nn_idx = d.topk(k, largest=False).indices
        nn_pos = tgt[nn_idx]                             # [Ns, k, 3]
        return nn_pos.mean(dim=1)                        # [Ns, 3]

    @staticmethod
    @torch.no_grad()
    def _radius_ball_centroid(
        means: Tensor,
        radius: float,
        chunk_size: int = 2048,
    ) -> Tensor:
        """Centroid of neighbors within ``radius`` for each point (self excluded).

        Falls back to ``means[i]`` for isolated points with no neighbors in
        the ball. Computed in chunks to bound GPU memory.

        Args:
            means: [N, 3] point positions.
            radius: Ball radius in world units.
            chunk_size: Number of query points per chunk.

        Returns:
            [N, 3] per-point centroid of radius-ball neighbors.
        """
        N = means.shape[0]
        centroids = torch.empty_like(means)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = means[start:end]                              # [C, 3]
            d = torch.cdist(chunk, means)                         # [C, N]
            mask = (d > 1e-8) & (d < radius)                     # [C, N]
            mask_f = mask.float()
            counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1) # [C, 1]
            centroids[start:end] = (mask_f @ means) / counts      # [C, 3]
            no_nb = ~mask.any(dim=1)
            if no_nb.any():
                centroids[start:end][no_nb] = chunk[no_nb]
        return centroids

    @torch.no_grad()
    def _nn_velocity_init(
        self,
        params: nn.ParameterDict,
        verbose: bool = False,
        k: int = 5,
    ) -> int:
        """Seed ``ν_base = ζ_base_t`` from k-NN-averaged inter-keyframe
        displacements.

        After ``_build_canonical_set`` (and any FPS reduction), every
        primitive retains its *source keyframe's* normalized time in
        ``params["times"]``. We group primitives by unique time value, and
        for each bucket ``k_bucket`` with time ``t_k``:

            1. Find the "next" bucket (``k+1``); for the last keyframe,
               match to ``k-1`` with a sign-flipped Δt.
            2. Run k-NN from bucket-k primitives to bucket-(k+1) primitives
               in 3D (canonical means), keeping up to ``k`` neighbors per
               source (clamped to the target bucket's size).
            3. Set ``ν_base[i] = mean_k (µ_NNᵏ − µ_i) / Δt_keyframe``.

        Rationale: ``µ(t_k + Δt) = µ + ν · Δt`` for identity rotation, so
        each primitive is pushed toward the centroid of its ~k matches —
        more robust than 1-NN against topology changes / sparse regions.

        Args:
            params: Canonical parameter dict.
            k: Number of neighbors per source. Default 5; K=1 is fragile.

        Returns:
            Number of primitives seeded with a non-zero ν.
        """
        times = params["times"].data.squeeze(-1)                 # [N]
        means = params["means"].data                             # [N, 3]
        nu = params["zeta_base_t"].data                          # [N, 3]

        unique_t = torch.unique(times).sort().values             # [K_buckets]
        if unique_t.numel() < 2:
            return 0

        # Map each primitive to its source-keyframe bucket index.
        bucket_idx = torch.searchsorted(unique_t, times)         # [N]

        n_seeded = 0
        K = unique_t.numel()
        for b in range(K):
            mask_k = bucket_idx == b
            if not mask_k.any():
                continue
            # Target bucket: next by default, last-keyframe matches backward.
            tgt_b = b + 1 if b < K - 1 else b - 1
            mask_tgt = bucket_idx == tgt_b
            if not mask_tgt.any():
                continue

            dt = (unique_t[tgt_b] - unique_t[b]).item()          # signed
            if abs(dt) < 1e-12:
                continue

            src_pos = means[mask_k]
            tgt_pos = means[mask_tgt]
            nn_mean_pos = self._knn_mean_positions(src_pos, tgt_pos, k)
            displacement = nn_mean_pos - src_pos                 # [Ns, 3]
            nu[mask_k] = displacement / dt
            n_seeded += int(mask_k.sum().item())

        if verbose:
            nnz = (nu.norm(dim=-1) > 0).float().mean().item()
            print(f"  NN ν_base init: {nnz * 100:.1f}% of primitives have non-zero ν")

        return n_seeded

    # -------------------------------------------------------------------------
    # k-NN graph for L_rigid
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _compute_knn(
        self,
        means: Tensor,
        k: int,
        times: Optional[Tensor] = None,
        time_weight: float = 0.0,
    ) -> Tensor:
        """Build k-nearest-neighbor indices, optionally time-aware.

        When ``times`` and ``time_weight > 0`` are both provided, the k-NN
        graph is built over 4-D points ``[x, y, z, time_weight · t]``
        instead of raw 3-D means. The caller is responsible for setting
        ``time_weight`` so that ``1.0`` of normalized time has a
        meaningful magnitude relative to ``means`` — typically
        ``time_weight = k · scene_extent`` for some k∈[0.1, 10] controlling
        strictness of cross-time exclusion.

        Self excluded. Returns ``[N, k]`` long tensor.
        """
        # Build the query/target points (3-D or 4-D).
        if times is not None and time_weight > 0:
            t_col = times if times.dim() == 2 else times.unsqueeze(-1)
            pts = torch.cat([means, time_weight * t_col], dim=-1)
        else:
            pts = means

        try:
            from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)
            pts_batched = pts.unsqueeze(0)
            result = knn_points(pts_batched, pts_batched, K=k + 1)
            idx = result.idx.squeeze(0)[:, 1:]  # drop self
            return idx.contiguous()
        except ImportError:
            # CPU fallback — O(N²), fine for small canonical sets.
            d = torch.cdist(pts, pts)
            d.fill_diagonal_(float("inf"))
            return d.topk(k, largest=False).indices

    # -------------------------------------------------------------------------
    # Delayed budget pruning (periodic opacity-based shrink)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _select_drop_mask_by_inactive(
        self,
        params: nn.ParameterDict,
        target_count: int,
        tau_alpha: float,
    ) -> Tensor:
        """Select drop mask from inactive primitives.

        Inactive is defined by opacity thresholding, with a more aggressive
        delayed-prune gate (``10 * tau_alpha``) to reach budget faster.
        """
        N = int(params["means"].shape[0])
        if target_count >= N:
            return torch.zeros(N, dtype=torch.bool, device=params["means"].device)
        if target_count <= 0:
            raise ValueError("target_count must be > 0 when delayed pruning is enabled")
        alpha = torch.sigmoid(params["opacities"].data).reshape(-1)
        inactive_mask = alpha < (10 * tau_alpha)
        inactive_idx = torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)

        n_remove = min(max(0, N - target_count), int(inactive_idx.numel()))
        if n_remove <= 0:
            return torch.zeros(N, dtype=torch.bool, device=params["means"].device)

        # Remove the most inactive first (lowest alpha).
        if n_remove == int(inactive_idx.numel()):
            return inactive_mask
        inactive_alpha = alpha[inactive_idx]
        drop_local = torch.topk(inactive_alpha, k=n_remove, largest=False).indices
        drop_idx = inactive_idx[drop_local]
        drop_mask = torch.zeros(N, dtype=torch.bool, device=params["means"].device)
        drop_mask[drop_idx] = True
        return drop_mask

    @torch.no_grad()
    def _apply_delayed_prune(
        self,
        params: nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        drop_mask: Tensor,
    ) -> Optional[Tensor]:
        """Prune primitive-wise params and slice optimizer rows in place.

        Returns:
            Kept indices after pruning, or ``None`` when this is a no-op.

        We intentionally replace leaf Parameters (instead of in-place shape
        mutation on ``.data``), because changing a leaf shape in-place can leave
        autograd with stale shape expectations on the next backward pass.
        """
        n_old = int(params["means"].shape[0])
        keep_idx = torch.where(~drop_mask)[0]
        if int(keep_idx.numel()) == n_old:
            # Fast no-op path: nothing to prune, skip optimizer surgery.
            return None

        # Snapshot ownership once (optimizer/group/index), then remap each
        # primitive-wise parameter without rescanning optimizer groups.
        owners = build_param_owners(optimizers)
        primitive_items = [
            (name, p)
            for name, p in list(params.items())
            if p.data.ndim >= 1 and int(p.data.shape[0]) == n_old
        ]

        for name, old_param in primitive_items:
            new_data = old_param.data[keep_idx].contiguous()
            new_param = nn.Parameter(new_data, requires_grad=old_param.requires_grad)
            params[name] = new_param
            remap_param_with_sliced_state(
                owners=owners,
                old_param=old_param,
                new_param=new_param,
                keep_idx=keep_idx,
                n_old=n_old,
            )
        return keep_idx

    # -------------------------------------------------------------------------
    # Tensorboard stats helper
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _log_stats_to_tb(
        self,
        writer: Any,
        step: int,
        params: nn.ParameterDict,
        psnr: float,
        t_max: float = 1.0,
    ) -> None:
        """Push motion / temporal / opacity distribution stats to tensorboard.

        Chosen specifically to expose the two known TriGS pathologies cheaply:
        Bézier collinearity (b_t / b_r near zero with large c_i) and σ_t
        runaway (p99 >> target).
        """
        writer.add_scalar("psnr", psnr, step)
        writer.add_scalar("n_primitives", params["means"].shape[0], step)

        # --- Motion magnitudes -------------------------------------------
        nu = params["zeta_base_t"].data
        om = params["zeta_base_r"].data
        writer.add_scalar("motion/nu_base_mean", nu.norm(dim=-1).mean().item(), step)
        writer.add_scalar("motion/omega_base_mean", om.norm(dim=-1).mean().item(), step)

        # --- Bézier control-point magnitudes & acceleration --------------
        c0_t, c1_t, c2_t = params["zeta_c0_t"].data, params["zeta_c1_t"].data, params["zeta_c2_t"].data
        c0_r, c1_r, c2_r = params["zeta_c0_r"].data, params["zeta_c1_r"].data, params["zeta_c2_r"].data
        writer.add_scalar(
            "motion/ctrl_t_mean",
            (c0_t.norm(dim=-1).mean() + c1_t.norm(dim=-1).mean() + c2_t.norm(dim=-1).mean()).item() / 3,
            step,
        )
        writer.add_scalar(
            "motion/ctrl_r_mean",
            (c0_r.norm(dim=-1).mean() + c1_r.norm(dim=-1).mean() + c2_r.norm(dim=-1).mean()).item() / 3,
            step,
        )
        b_t = (c0_t - 2 * c1_t + c2_t).norm(dim=-1)
        b_r = (c0_r - 2 * c1_r + c2_r).norm(dim=-1)
        writer.add_scalar("motion/bezier_accel_t_mean", b_t.mean().item(), step)
        writer.add_scalar("motion/bezier_accel_r_mean", b_r.mean().item(), step)
        writer.add_scalar(
            "motion/accel_over_ctrl_t",
            (b_t.mean() / (c0_t.norm(dim=-1).mean() + 1e-12)).item(),
            step,
        )  # diagnostic: low ratio ⇒ Bézier collapsed to collinear (linear motion)

        # --- Anchor displacement ------------------------------------------
        anchor_disp = (params["anchors"].data - params["means"].data).norm(dim=-1)
        writer.add_scalar("anchor/disp_mean", anchor_disp.mean().item(), step)

        # --- Temporal window ---------------------------------------------
        mu_t = params["times"].data.squeeze(-1)
        sigma_t = torch.exp(params["log_durations"].data.squeeze(-1))
        writer.add_scalar("sigma_t/mean", sigma_t.mean().item(), step)
        writer.add_scalar("sigma_t/median", sigma_t.median().item(), step)
        writer.add_scalar("sigma_t/p99", torch.quantile(sigma_t.float(), 0.99).item(), step)
        writer.add_scalar("sigma_t/max", sigma_t.max().item(), step)
        writer.add_scalar(
            "mu_t/oob_frac",
            ((mu_t < 0) | (mu_t > t_max)).float().mean().item(),
            step,
        )

        # --- Opacity ------------------------------------------------------
        alpha = torch.sigmoid(params["opacities"].data).squeeze(-1)
        writer.add_scalar("opacity/median", alpha.median().item(), step)
        writer.add_scalar(
            "opacity/saturated_frac", (alpha > 0.99).float().mean().item(), step,
        )
        writer.add_scalar(
            "opacity/dead_frac", (alpha < 0.01).float().mean().item(), step,
        )

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------

    def _train_loop(
        self,
        params: nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        nn_indices: Tensor,
        cameras: "CameraConfig",
        n_frames: int,
        sh_degree: int,
        gt_images: Dict[int, List[Tensor]],
        gt_alphas: Dict[int, List[Tensor]],
        config: TrigsTrainingConfig,
        scene_extent: float,
        gt_times_per_frame: Optional[Dict[int, List[float]]] = None,
        sigma_floor_target: float = 0.0,
        writer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from ..strategy.motion_guided_relocation_ply_upstream import (
            MotionGuidedRelocationConfig,
            MotionGuidedRelocationStrategy,
        )

        n_views = len(cameras)
        if n_views <= 0:
            raise ValueError("Expected at least one camera/view for training")

        def _build_time_first_epoch_pairs() -> List[tuple[int, int]]:
            """Create one epoch of (time, view) pairs with time-first ordering.

            Policy:
            1) Shuffle times once per epoch.
            2) For each time, shuffle its view order once per epoch.
            3) Emit pairs in rounds over shuffled times:
               round 0 picks each time's first shuffled view, round 1 the second,
               etc. This keeps full epoch coverage while making time the primary
               progression axis.
            """
            frame_order = list(range(n_frames))
            random.shuffle(frame_order)

            shuffled_views_by_frame: Dict[int, List[int]] = {}
            for fi in frame_order:
                views = list(range(n_views))
                random.shuffle(views)
                shuffled_views_by_frame[fi] = views

            epoch_pairs: List[tuple[int, int]] = []
            for view_round in range(n_views):
                for fi in frame_order:
                    epoch_pairs.append((fi, shuffled_views_by_frame[fi][view_round]))
            return epoch_pairs

        epoch_iter = iter([])

        metrics: Dict[str, Any] = {
            "n_initial": params["means"].shape[0],
            "snapshots": [],
            "prune_events": [],
        }
        iterator = (
            trange(config.n_steps, desc="TRiGS")
            if config.verbose else range(config.n_steps)
        )

        # --- LR schedule: exponential decay for positional groups only ---
        decay_ratios: Dict[str, float] = {
            "anchors": config.lr_anchor_final / max(config.lr_anchor, 1e-30),
            "zeta_base_t": config.lr_zeta_base_trans_final / max(config.lr_zeta_base_trans, 1e-30),
            "zeta_bezier_t": config.lr_zeta_bezier_trans_final / max(config.lr_zeta_bezier_trans, 1e-30),
        }
        if not config.freeze_geometry:
            decay_ratios["means"] = config.lr_means_final / max(config.lr_means, 1e-30)
        # Capture the (scene-scaled) initial LRs as each optimizer was built.
        initial_lrs: Dict[str, float] = {
            name: opt.param_groups[0]["lr"]
            for name, opt in optimizers.items()
            if name in decay_ratios
        }

        # --- Temporal freeze: force motion learning before temporal fitting ---
        temporal_names = ("opacities", "times", "log_durations")
        temporal_frozen = config.unfreeze_temporal_step > 0
        if temporal_frozen:
            for name in temporal_names:
                params[name].requires_grad_(False)
            if config.verbose:
                print(
                    f"  Temporal params frozen until step {config.unfreeze_temporal_step}"
                )

        # Relocation strategy (gsplat Strategy subclass). Owns the
        # relocation cadence + the grad accumulators populated by the trainer.
        relocation_strategy = MotionGuidedRelocationStrategy(
            MotionGuidedRelocationConfig(
                relocate=config.relocate,
                relocate_every=config.relocate_every,
                relocate_start_iter=config.relocate_start_iter,
                relocate_tau_alpha=config.relocate_tau_alpha,
                relocate_scale_factor=config.relocate_scale_factor,
            )
        )
        reloc_state = relocation_strategy.initialize_state(
            n_primitives=params["means"].shape[0], device=params["means"].device,
        )
        rigid_knn_dirty = False

        for step in iterator:
            if "shN" in params:
                if step == 1000:
                    sh_degree = 1
                elif step == 2000:
                    sh_degree = 2
                elif step == 3000:
                    sh_degree = 3

            # Unfreeze temporal params once motion has had time to develop.
            if temporal_frozen and step == config.unfreeze_temporal_step:
                for name in temporal_names:
                    params[name].requires_grad_(True)
                # Reset optimizer state so Adam starts fresh (no stale moments).
                for name in temporal_names:
                    if name in optimizers:
                        for pg in optimizers[name].param_groups:
                            for p in pg["params"]:
                                if p in optimizers[name].state:
                                    optimizers[name].state[p].clear()
                temporal_frozen = False
                if config.verbose:
                    print(f"  Step {step}: unfreezing temporal params")

            # Apply exponential LR decay for scheduled groups.
            progress = step / max(config.n_steps - 1, 1)
            for name, ratio in decay_ratios.items():
                if name not in optimizers:
                    continue
                lr = initial_lrs[name] * (ratio ** progress)
                for pg in optimizers[name].param_groups:
                    pg["lr"] = lr
            try:
                frame_idx, cam_idx = next(epoch_iter)
            except StopIteration:
                epoch_iter = iter(_build_time_first_epoch_pairs())
                frame_idx, cam_idx = next(epoch_iter)

            if gt_times_per_frame is not None and gt_times_per_frame[frame_idx][cam_idx] is not None:
                t = gt_times_per_frame[frame_idx][cam_idx]
            else:
                t = float(frame_idx) / max(config.fps, 1e-6)

            render_colors, render_alphas, gamma, info = self._render(
                params, cameras, cam_idx, t, sh_degree, config,
            )
            gt_rgb = gt_images[frame_idx][cam_idx].to(self.device).float().div(255.0)
            if tuple(render_colors[0].shape) != tuple(gt_rgb.shape):
                raise RuntimeError(
                    "Render/GT shape mismatch before loss. "
                    f"frame={frame_idx}, cam={cam_idx}, "
                    f"render={tuple(render_colors[0].shape)}, "
                    f"gt={tuple(gt_rgb.shape)}"
                )
            delayed_target = config.delayed_prune_target_count
            delayed_pruning_active = (
                delayed_target is not None
                and delayed_target > 0
                and params["means"].shape[0] > delayed_target
            )

            rgb_loss = reconstruction_loss(
                render_colors[0], gt_rgb, ssim_lambda=config.ssim_lambda,
            )

            # L_motion (Eq. 11) — acceleration on the full (ν, ω) 6-vector;
            # reassemble from split halves.
            c0_full = torch.cat([params["zeta_c0_t"], params["zeta_c0_r"]], dim=-1)
            c1_full = torch.cat([params["zeta_c1_t"], params["zeta_c1_r"]], dim=-1)
            c2_full = torch.cat([params["zeta_c2_t"], params["zeta_c2_r"]], dim=-1)
            motion_loss = bezier_motion_loss(c0_full, c1_full, c2_full)

            if delayed_pruning_active:
                # Budget-reduction phase: skip rigid loss / kNN updates entirely.
                rigid_loss = params["means"].new_zeros(())
            else:
                # L_rigid (Eq. 12) — recompute kNN periodically.
                if (
                    rigid_knn_dirty
                    or (
                        step > 0
                        and config.knn_recompute_every > 0
                        and step % config.knn_recompute_every == 0
                    )
                ):
                    nn_indices = self._compute_knn(
                        params["means"].detach(),
                        config.knn_k,
                        times=None,
                        time_weight=0.0,
                    )
                    rigid_knn_dirty = False
                zeta_base_full = torch.cat(
                    [params["zeta_base_t"], params["zeta_base_r"]], dim=-1,
                )
                rigid_loss = rigid_coherence_loss(
                    zeta_base_full,
                    params["sh0"].detach(),
                    nn_indices,
                    lambda_c=config.rigid_color_lambda,
                )

            # L_reg — FTGS Eq. 2, inherited verbatim by TriGS Eq. 10:
            #     L_reg = (1/N) Σ_i σ(α_i) · sg[γ_i(t)]
            # Pushes low-visibility primitives toward low opacity so they
            # get recycled by relocation. The stop-gradient on γ prevents
            # the regularizer from minimizing temporal opacity.
            base_opa = torch.sigmoid(params["opacities"])
            if base_opa.dim() == 1:
                base_opa = base_opa.unsqueeze(-1)
            reg_loss = (base_opa * gamma.detach()).mean()

            # L_sigma_floor — one-sided soft floor on σ_t.
            #     L = mean(max(target − σ_t, 0)²)
            # Quadratic pull upward when σ_t < target; zero above (preserves
            # always-on primitives for static content). Not in the TriGS
            # paper — added because segmented-dynamic scenes lack the
            # static-background population that naturally anchors σ_t wide.
            sigma_t = torch.exp(params["log_durations"])
            sigma_floor_loss = torch.clamp(
                sigma_floor_target - sigma_t, min=0.0,
            ).pow(2).mean()

            total_loss = (
                config.lambda_render * rgb_loss
                + config.lambda_motion * motion_loss
                + config.lambda_rigid * rigid_loss
                + config.lambda_reg * reg_loss
                + config.lambda_sigma_floor * sigma_floor_loss
            )

            # Tensorboard loss scalars (pre-backward so .item() is cheap).
            if (
                writer is not None
                and config.tb_log_every > 0
                and step % config.tb_log_every == 0
            ):
                writer.add_scalar("loss/total", total_loss.item(), step)
                writer.add_scalar("loss/rgb", rgb_loss.item(), step)
                writer.add_scalar("loss/motion", motion_loss.item(), step)
                writer.add_scalar("loss/rigid", rigid_loss.item(), step)
                writer.add_scalar("loss/reg", reg_loss.item(), step)
                writer.add_scalar("loss/sigma_floor", sigma_floor_loss.item(), step)
                # Current LRs for scheduled groups.
                for name, opt in optimizers.items():
                    if name in decay_ratios:
                        writer.add_scalar(
                            f"lr/{name}", opt.param_groups[0]["lr"], step,
                        )

            total_loss.backward()

            # --- Accumulate relocation cues (TRiGS Suppl §4) -----------------
            # These reads must happen BEFORE zero_grad. Guard with None-checks
            # because a zero-loss step (rare but possible) leaves .grad unset.
            relocation_strategy.accumulate_grads(reloc_state, params, info)

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            # Post-step clamp: µ_t ∈ [0, (n_frames − 1) / fps] (seconds).
            # Physical domain of time is the sequence range; values outside
            # waste primitive capacity (γ≈0 at every real frame). Clamp
            # happens after Adam has applied the gradient so the optimizer
            # momentum stays intact.
            if config.clamp_mu_t:
                t_max = float(max(n_frames - 1, 0)) / max(config.fps, 1e-6)
                params["times"].data.clamp_(0.0, t_max)

            # Optional post-step clamp: keep opacity logits in a trainable
            # high-end range for the early/mid phase.
            upper_logit = config.opacity_logit_upper_clamp
            if upper_logit is not None:
                until_step = config.opacity_logit_upper_clamp_until_step
                clamp_active = until_step < 0 or step <= until_step
                if clamp_active:
                    params["opacities"].data.clamp_(max=upper_logit)

            # Optional periodic delayed prune:
            # while above the target budget, prune by opacity every N steps.
            # Relocation stays disabled during this phase and starts only once
            # the target count is reached.
            if (
                delayed_pruning_active
                and step >= config.delayed_prune_start_iter
                and config.delayed_prune_every > 0
                and step % config.delayed_prune_every == 0
            ):
                drop_mask = self._select_drop_mask_by_inactive(
                    params=params,
                    target_count=delayed_target,
                    tau_alpha=config.relocate_tau_alpha,
                )
                n_before = int(params["means"].shape[0])
                keep_idx = self._apply_delayed_prune(
                    params=params,
                    optimizers=optimizers,
                    drop_mask=drop_mask,
                )
                if keep_idx is not None:
                    reloc_state["grad_2d_accum"] = reloc_state["grad_2d_accum"][keep_idx].contiguous()
                    reloc_state["grad_t_accum"] = reloc_state["grad_t_accum"][keep_idx].contiguous()
                    n_after = int(params["means"].shape[0])
                    rigid_knn_dirty = True
                    metrics["prune_events"].append(
                        {"step": step, "before": n_before, "after": n_after}
                    )
                    if config.verbose:
                        print(
                            f"  Delayed prune @ step {step}: {n_before:,} → {n_after:,} "
                            f"(inactive α<{config.relocate_tau_alpha:g})"
                        )

            delayed_pruning_active = (
                delayed_target is not None
                and delayed_target > 0
                and params["means"].shape[0] > delayed_target
            )
            if not delayed_pruning_active:
                relocation_strategy.step_post_backward(
                    params=params,
                    optimizers=optimizers,
                    state=reloc_state,
                    step=step,
                )

            if config.verbose and step % 500 == 0:
                psnr = compute_psnr(render_colors[0].detach(), gt_rgb)
                snap = {
                    "step": step,
                    "psnr": psnr,
                    "loss_total": total_loss.item(),
                    "loss_rgb": rgb_loss.item(),
                    "loss_motion": motion_loss.item(),
                    "loss_rigid": rigid_loss.item(),
                    "loss_reg": reg_loss.item(),
                    "loss_sigma_floor": sigma_floor_loss.item(),
                }
                metrics["snapshots"].append(snap)
                if hasattr(iterator, "set_postfix"):
                    iterator.set_postfix(psnr=f"{psnr:.1f}", loss=f"{total_loss.item():.4f}")

            # Tensorboard motion / temporal / opacity distribution stats.
            if (
                writer is not None
                and config.tb_stats_every > 0
                and step % config.tb_stats_every == 0
            ):
                with torch.no_grad():
                    psnr_val = compute_psnr(render_colors[0].detach(), gt_rgb)
                    t_max = float(max(n_frames - 1, 0)) / max(config.fps, 1e-6)
                    self._log_stats_to_tb(
                        writer, step, params, psnr_val, t_max=t_max,
                    )

        metrics["final_count"] = params["means"].shape[0]
        return metrics
