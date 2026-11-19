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
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TYPE_CHECKING

import torch
from trigs.compat.device import resolve_device, empty_cache, device_synchronize, manual_seed_all, accelerator_rng_states, f64  # macOS port
import torch.nn as nn
import torch.nn.functional as F
from gsplat import rasterization
from torch import Tensor
from tqdm import tqdm, trange

from .losses import (
    bezier_motion_loss,
    compute_psnr,
    reconstruction_loss,
    rigid_coherence_loss,
    scale_size_loss,
)
from .optimizer_state import (
    build_param_owners,
    export_param_state,
    load_param_state,
    remap_param_with_sliced_state,
)
from .viewer_snapshot import should_snapshot, write_viewer_snapshot
from ..math.quaternion import quaternion_multiply, quaternion_to_rotmat
from ..math.se3 import bezier_quadratic, bezier_rewindow, exp_se3_quat, gauge_fix_omega, tau_from_window
from ..math.twist_init import fit_local_twist, local_knn_indices

import os

import logging

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .camera import CameraConfig
    from .transforms import TransformConfig
    from ..init.from_plys import PlyMotionInitConfig


# =============================================================================
# Config
# =============================================================================

@dataclass
class TrigsTrainingConfig:
    """Configuration for TRiGS training."""

    # --- Keyframe subsampling ---
    # The COLMAP CLI fills keyframe_stride and fps from the NPZ when they are not given.
    keyframe_stride: int = 3
    fps: float = 30.0
    target_count: int | None = None
    # Dynamic-row budget per keyframe in the training window (COLMAP NPZ path). The CLI sets
    # target_count = this x the number of keyframes kept, unless target_count is given. 0 keeps
    # every row. 25_862 is the tested example capture density (181,035 rows over 7 keyframes).
    target_count_per_keyframe: int = 25_862
    fps_time_weight: float = 1.0
    delayed_prune_target_count: int | None = None
    delayed_prune_start_iter: int = 0
    delayed_prune_every: int = 100
    freeze_geometry: bool = False
    # Starting opacity of every NPZ row (COLMAP NPZ path). 0.1 is the 3DGS
    # default; official TRiGS's own init is unreleased.
    init_opacity: float = 0.1
    # Re-size dynamic rows' starting scales after the target-count cap, on the
    # rows actually trained (same 0.3 x mean 4-NN rule, per keyframe). The NPZ
    # build sizes splats against the full prep cloud, so after keeping 1 in N
    # rows they start several times smaller than their new spacing.
    init_rescale_after_budget: bool = False
    # Starting splat size of dynamic NPZ rows: this x the mean distance to the
    # row's 4 nearest rows in its keyframe (COLMAP NPZ path, also used by
    # init_rescale_after_budget). 0.3 leaves dots about 0.43 of the gap to their
    # neighbour; example capture's prep cloud is clumped, so only ~69% of the
    # keyframe surface lies within 2 sigma of a splat. ~0.9 matches the 3DGS
    # rule (RMS of 3 nearest, ~1.3 x the gap; 99.8% covered).
    init_scale_knn_factor: float = 0.3
    init_sigma_t: float = 0.0
    # Initial temporal width of dynamic NPZ rows (COLMAP NPZ path):
    #   "uniform": every row gets init_sigma_t, else keyframe_stride / fps
    #              (the TriGSVanilla rule; official TRiGS leaves init to an
    #              unreleased pipeline).
    #   "npz":     the prep's speed-scaled per-row durations, clamped to
    #              [init_sigma_t_floor_frac, 1] x keyframe_stride / fps. Fast
    #              rows then stop lingering at extrapolated positions across
    #              neighbouring keyframes (example capture: a fast hand starts
    #              as a spray of ~6 overlapping keyframe copies under
    #              "uniform"), while the floor keeps adjacent keyframes
    #              cross-fading instead of leaving gaps between them.
    #   "speed":   per row, the time over which its starting curve moves
    #              init_sigma_t_speed_spacings local point spacings
    #              (|v| s + 1/2 |a| s^2 = K d, neighbourhood-median |v| and,
    #              under init_accel_mode "fit", the smoothed |a|), clamped to
    #              [init_sigma_t_floor_frac x init_sigma_t, init_sigma_t_max].
    #              Still parts start long-lived (no hand-off churn on
    #              faces/walls), fast parts at the floor, and motion noise moves
    #              a row at most K spacings within +/-1 sigma.
    init_sigma_t_mode: str = "npz"
    init_sigma_t_floor_frac: float = 0.4
    # K = 3: on example capture's r05 init the still faces then start at p50 7.9
    # frames (K = 1: 3.3), above the 6.0 r05 trained them to; fast rows stay
    # at the floor.
    init_sigma_t_speed_spacings: float = 3.0
    init_sigma_t_max: float = 0.5
    # Starting curve for dynamic NPZ rows (COLMAP NPZ path):
    #   "off": straight screw line from the keyframe velocity (Bezier = 0).
    #   "fit": Bezier controls fitted so each row starts on
    #          x + v dt + 1/2 a dt^2 over its +/-2 sigma window, with a = the
    #          prep's measured acceleration (--centred-velocity), median-
    #          smoothed over the twist-fit neighbours and scaled by
    #          init_accel_scale. example capture (every-frame burst, real
    #          points): worst-10% reversal error at +/-5 frames -34% vs the
    #          straight line at scale 0.5; raw (unsmoothed, x1) is worse than
    #          no acceleration.
    init_accel_mode: str = "fit"
    init_accel_scale: float = 1.0
    init_accel_fit_iters: int = 300
    # Fit each dynamic row's starting curve through the NPZ's chained positions
    # (prep/init/chain-targets-npz.py) over the widest window its chain reaches
    # on both sides, and widen its starting lifetime to match (window = +-2 sigma).
    # Rows reaching fewer than init_chain_min_span_frames keep the local fit.
    # example capture: trained motion held only ~+-9 frames, while the prep
    # cloud chains the moving girl much further.
    init_chain_fit: bool = False
    init_chain_min_span_frames: int = 6
    nn_velocity_init: bool = True
    nn_velocity_k: int = 3
    nn_anchor_init: bool = True
    anchor_radius_frac: float = 0.1
    omega_init: bool = False
    local_twist_k: int = 16
    local_twist_reg: float = 1e-4
    n_steps: int = 30_000
    unfreeze_temporal_step: int = 0
    # Chunked training rebases unfreeze_temporal_step globally, so only the
    # chunk calls covering the first `unfreeze_temporal_step` GLOBAL steps get
    # a motion-only phase — later chunks train temporal params from their first
    # step. Set this to give EVERY chunk call its own motion-only phase of
    # min(unfreeze_temporal_step, steps_this_call // 2) local steps.
    unfreeze_temporal_per_chunk: bool = False
    lr_opacity: float = 5e-2
    lr_times: float = 1.6e-4
    lr_durations: float = 5e-3
    lr_anchor: float = 1.6e-4
    lr_means: float = 1.6e-4
    lr_zeta_base_trans: float = 1.6e-4
    lr_zeta_base_rot: float = 1e-3
    lr_zeta_bezier_trans: float = 1.6e-4
    lr_zeta_bezier_rot: float = 1e-3
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_sh0: float = 2.5e-3
    lr_shN: float = 2.5e-3 / 20.0
    lr_decay_steps: int = 0
    lr_anchor_final: float = 1.6e-6
    lr_means_final: float = 1.6e-6
    lr_zeta_base_trans_final: float = 1.6e-6
    lr_zeta_bezier_trans_final: float = 1e-5
    # Official TRiGS also decays the angular base twist and angular Bézier
    # controls. Its factors are unreleased; these mirror the translational
    # ratios above (100x base, 16x Bézier). Set equal to the initial rate to
    # disable.
    lr_zeta_base_rot_final: float = 1e-5
    lr_zeta_bezier_rot_final: float = 6.25e-5
    # Cameras rendered per optimizer step (official TRiGS batch_size). Each
    # camera is rendered at its own capture time; B=1 is single-image
    # training.
    batch_cameras: int = 20
    # How each step picks its frame and cameras.
    #   "epoch":    every (frame, camera) pair once per epoch, frames shuffled,
    #               ``batch_cameras`` distinct cameras per step (ours).
    #   "official": official TRiGS FastInfiniteLoader -- a uniformly random
    #               frame each step and ``batch_cameras`` cameras drawn with
    #               replacement (random.choices).
    camera_sampling: str = "official"
    # Weighted frame choice (OPT-IN, unchunked, camera_sampling "official"). A short-lived row only
    # learns on the steps that pick a frame inside its window, so its update count is about
    # steps / frames x lifetime. This fraction of the steps picks the frame in proportion to the
    # short-lived rows active there (counted once, at the start of training); the rest stay
    # uniform, so every frame keeps at least (1 - frac) of its uniform share. 0 = uniform.
    frame_sampling_fast_frac: float = 0.0
    # Extra visits for fast frames, ADDED to normal training (unchunked, camera_sampling "official";
    # skipped with a warning otherwise; 0 = off; excludes frame_sampling_fast_frac). n_steps stays the normal-training budget
    # (n_steps / frames visits per frame); each frame with short-lived rows is topped up to this many
    # visits (in full once it has frame_sampling_fast_min_rows of them, proportionally below), and
    # the extra steps are added to n_steps and mixed into the run. Slow frames keep exactly their
    # normal-training visits and no frame exceeds the target. 387 = the tested example capture level.
    frame_sampling_fast_visits: float = 387.0
    frame_sampling_fast_min_rows: int = 200
    # "Short-lived" for the weights: starting sigma_t at or below this many frames. 1.5 selects
    # speed-aware sub-keyframe rows (1 frame) and not the stride-5 lifetime floor (2 frames).
    frame_sampling_fast_sigma_frames: float = 1.5
    # Official batch rule for the Gaussian optimizers: lr * sqrt(B),
    # eps / sqrt(B), betas ** B. A no-op at B=1.
    batch_optimizer_scaling: bool = True
    # Apply the sqrt(B) factor to the learning rates too (official TRiGS). Off by default: the
    # lr_* values are then the effective per-step rates at any batch size, while eps and betas
    # keep the batch rule. Full sqrt(B) rates crashed example capture at B=20; every tuned run since
    # (example capture train_H) used lr_* / sqrt(B) with the rule on, which is exactly this setting.
    batch_lr_scaling: bool = False
    # How the COLMAP CLI scales the positional/translational LRs.
    #   "official": max distance of the training cameras (holdouts excluded)
    #               from their mean -- trigs._math.scene_extent.
    #   "parser":   parser.scene_scale, 2x that radius over every camera
    #               (the FreeTimeGS convention these LRs were run with before).
    # The default max_scale_exp cap and rigid_knn_time_weight also scale
    # with this extent, so "official" halves them relative to "parser".
    scene_extent_mode: str = "official"
    lambda_render: float = 1.0
    # Clamp the rendered image to [0, 1] before the photometric loss, as the
    # official trainer does (after camera appearance and compositing).
    clamp_render_for_loss: bool = True
    ssim_lambda: float = 0.2
    lambda_motion: float = 1e-4
    # L2 on Bézier residual control magnitudes (c0, c1, c2). The paper's
    # L_motion penalizes only the curvature c0-2c1+c2, so a linear-in-τ
    # residual of any magnitude is free; the γ-damped window-edge phases then
    # acquire rogue velocities over long schedules (measured: at any playback
    # time ~8% of visible splats deviate >0.5 m/s from the local velocity
    # field, worst at phase >0.8). Photometrically supervised residuals easily
    # outpull 1e-3; unsupervised ones decay toward base-dominated motion.
    lambda_motion_residual: float = 0.0
    lambda_rigid: float = 1.0
    lambda_reg: float = 1e-2
    # Official TRiGS spatial-size penalty: lambda * mean(exp(scales)) over
    # every row and axis (gsplat MCMC scale_reg). Normalized by the FULL
    # model row count, so a chunk's resident subset gets the same per-row
    # strength as the unchunked model. A soft preference, not a cap; the
    # max_scale_exp / max_scale_anisotropy clamps remain the hard limits.
    lambda_scale_reg: float = 1e-2
    # The official model has no static rows; include ours by default so
    # background splats get the same size pressure.
    scale_reg_include_static: bool = True
    lambda_sigma_floor: float = 0.0
    sigma_floor_target: float = 0.0
    knn_k: int = 3
    # Official TRiGS rebuilds the rigidity graph every 500 steps (and after
    # every relocation, which we also do).
    knn_recompute_every: int = 500
    # Space the rigidity KNN is searched in. "anchors" (official TRiGS)
    # couples primitives that share a rotation pivot; "means" couples
    # primitives whose canonical centers are close.
    rigid_knn_space: str = "anchors"
    rigid_color_lambda: float = 50.0
    # Official TRiGS keeps the appearance affinity K_ij differentiable, so
    # L_rigid also reaches sh0. True restores the motion-only gradient.
    rigid_color_weight_detach: bool = False
    rigid_knn_time_weight: float = 0.0
    gauge_eps: float = 1e-8
    gauge_denom_clamp: float = 1e-10
    # Evaluate the Bézier coordinate τ with μ_t and σ_t detached (official
    # TRiGS). False lets image error retime the motion curve through the
    # temporal params.
    bezier_window_detach: bool = True
    # Keep each row's Bézier residual on the same timing when an optimiser
    # step (and its clamps) changes sigma_t: the window is +-2 sigma, so a new
    # sigma otherwise stretches the curve and moves the row inside its old
    # window. Official TRiGS stretches. example capture (results-23, 10k):
    # lengthening the girl's sigma x1.5 cost +12.8% L1 with stretching vs
    # +5.6% with timing kept.
    bezier_keep_timing: bool = False
    relocate: bool = True
    relocate_every: int = 200
    relocate_start_iter: int = 1000
    relocate_tau_alpha: float = 0.005
    relocate_scale_factor: float = 0.66
    # Clones' Bézier residual controls collapse to the donor residual evaluated
    # at the clone's jittered μ_t (constant, phase-correct) instead of a
    # verbatim curve copy replayed at the wrong phase. See
    # strategy/motion_guided_relocation.py.
    relocate_recenter_residual: bool = False
    # Divide the accumulated |dL/d mu_t| donor cue by the steps each row was
    # active (ours: rows clipped by the sequence edges are otherwise starved
    # of clones). False uses the official total, which favours long-lived
    # rows over short-lived fast ones.
    relocate_time_cue_per_visit: bool = False
    # Speed x lifetime donor cue: count only the part of the +/-2 sigma
    # support inside the clip (ours; measured edge-frame starvation without
    # it). False is official TRiGS: |velocity| x sigma over the full support.
    relocate_speed_cue_clip_edges: bool = False
    clamp_mu_t: bool = True
    opacity_logit_upper_clamp: float | None = None
    opacity_logit_upper_clamp_until_step: int = 25_000
    clamp_log_durations: bool = True
    min_sigma_t: float = 0.005
    max_sigma_t: float = -1.0
    clamp_log_scales: bool = True
    min_scale_exp: float = 1e-6
    max_scale_exp: float = 0.0
    # Cap only a Gaussian's longest spatial axis relative to its median axis.
    # This preserves surface-like (two-wide, one-thin) splats while preventing
    # the one-axis needles that render as long foreground/background streaks.
    # A value <= 0 disables the cap.
    max_scale_anisotropy: float = 0.0
    verbose: bool = True
    # Diagnostic console output (per-relocation counts, init/budget statistics, loader checks).
    # Off by default; the progress bar, run summary, evals, checkpoints and warnings always print.
    diagnostics: bool = False
    tb_log_every: int = 50
    tb_stats_every: int = 500
    # Opacity-drain diagnostics (logging only; training is unchanged). Every
    # step accumulates, per row, the photo loss's opacity gradient, the
    # regularisers' share, and the optimizer's actual opacity change; every
    # ``opacity_diag_every`` steps their per-lifetime-band summary is logged to
    # TensorBoard under ``opacity_diag/`` and printed. Set it equal to
    # ``relocate_every`` so no window spans a relocation (the log runs before
    # that step's relocation). 0 disables.
    opacity_diag_every: int = 0
    checkpoint_dir: str | None = None
    checkpoint_progress_interval: float = 0.25
    checkpoint_include_optimizers: bool = False
    # Write no safety checkpoints after this completed step (0 = no limit), so
    # a fine interval for a short diagnostic window cannot fill the disk.
    checkpoint_until_step: int = 0
    # Relocation lineage (logging only; training is unchanged): per-row
    # ``relocation_generation`` (+1 each time relocation refills the slot) and
    # ``relocation_donor_count`` (+1 per clone made from the row). The chunked
    # path always tracks the generation; this adds both to the unchunked path
    # so full checkpoints record exactly which slots relocation touched.
    record_relocation_lineage: bool = False
    # Target-count split (single/merged NPZ init): within each keyframe's
    # quota q, deterministically keep the round(frac*q) fastest dynamic rows
    # (step-0 one-frame deformed displacement) and draw the rest of q
    # uniformly at random. 0.0 = the original purely random draw, unchanged.
    budget_protect_speed_top_frac: float = 0.25
    # Live-viewer snapshots: params-only files for ``viewer.py --watch``,
    # written every `viewer_snapshot_every` GLOBAL steps (0 = off). Chunked
    # runs write the whole CPU master between chunk calls, so a snapshot is
    # always a complete, consistent model. The CLIs default the directory to
    # <output>/viewer_snapshots.
    viewer_snapshot_every: int = 1000
    viewer_snapshot_dir: str | None = None
    viewer_snapshot_keep_last: int = 3
    # Also keep every snapshot whose step is a multiple of this (0 = off).
    viewer_snapshot_keep_every: int = 0
    # "float16" stores shN (most of the bytes) as half.
    viewer_snapshot_dtype: str = "float32"
    masked_rgb_loss: bool = False
    # Composite the render over a constant background and supervise the full
    # frame, instead of substituting GT outside the mask (which zeroes the
    # gradient there). Restores photometric pressure on out-of-mask mass and
    # gives depth a background anchor again. Requires decontaminated straight
    # foreground RGB (for COLMAP, use --foregrounds-root); the GT is matted
    # onto masked_rgb_bg_color, so the background never has to be represented.
    # OPT-IN: changes the meaning of loss/rgb, so existing lambda_alpha_*
    # tunings do not transfer unchanged.
    masked_rgb_composite: bool = False
    masked_rgb_bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # Randomize the composite plate during training. This is useful only when
    # gt_rgb contains decontaminated straight foreground RGB; applying it to
    # captured RGB with an existing background bakes that background into the
    # boundary. Evaluation remains deterministic on masked_rgb_bg_color.
    masked_rgb_random_bg: bool = False
    # Both alpha terms are normalized by the full frame (see the masked-loss
    # block in _train_loop), so these weights are directly comparable to
    # lambda_render: the alpha-vs-RGB per-pixel gradient ratio is 3*lambda for
    # the outside term and 1.5*lambda for the inside BCE, independent of how
    # tightly the subject is framed. Keep the outside term subordinate to
    # reconstruction -- it exists to stop background mass, not to drive the fit.
    lambda_alpha_outside_mask: float = 0.1
    lambda_alpha_mask: float = 0.5
    alpha_loss_eps: float = 1e-4
    psnr_eval_every: int = 1000
    psnr_eval_num_samples: int = 48
    psnr_eval_seed: int = 12345
    # Seed for the training loop's own randomness (Python ``random``: frame and
    # camera sampling; torch CPU/CUDA: relocation donor draws and time jitter),
    # set at the start of _train_loop. -1 (default) leaves the RNGs unseeded,
    # as before. Same seed + same config gives the same sample stream; GPU
    # floating-point nondeterminism (atomic adds in the rasterizer backward)
    # can still make the model states drift apart. Unchunked training only
    # (train_chunked rejects it: each chunk call would reseed).
    train_seed: int = -1
    psnr_mask_threshold: float = 0.5
    psnr_clamp_rgb: bool = True
    # Camera positions (0-based, indexing the training camera list) withheld
    # from the training pair stream and evaluated separately. Without this the
    # psnr/eval_* series is a training-set fit metric -- _make_psnr_eval_pairs
    # enumerates range(n_views), so every eval pair is also a training pair and
    # the number carries no information about novel views. Held-out cameras
    # still get GT loaded and rendered; they are only excluded from the loss.
    holdout_views: tuple[int, ...] = ()
    static_convert: bool = False
    static_convert_start_iter: int = 1000
    static_convert_every: int = 500
    static_convert_speed_thresh: float = 0.025
    static_convert_rot_thresh: float = 0.025
    static_convert_accel_thresh: float = 0.025
    static_convert_alpha_min: float = 0.02
    # Conversion makes a row active at EVERY t (see
    # _enforce_relocation_class_invariants) and is irreversible — nothing ever
    # sets is_static back to 0. So "motionless" alone is not sufficient: a row
    # must ALREADY span the supervised window, or a short-lived row gets
    # promoted to permanently visible, which renders as a floater across the
    # whole clip. A row qualifies when μₜ ± (this many) σₜ covers [0, t_max].
    # Set <= 0 to restore the old motion-only behaviour.
    static_convert_coverage_sigmas: float = 2.0
    # Gate rotation on the world-space displacement it induces (‖ω × r‖, with
    # r = mean − anchor) rather than on the raw angular rate. The rotation and
    # translation gates then share physical units, so they can share a
    # threshold. False restores the old raw-radian comparison.
    static_convert_rot_uses_lever: bool = True
    relocate_dynamic_only: bool = True
    # Official TRiGS relocation: gsplat's MCMC math redistributes donor
    # opacity/scale across donor + clones by copy count. False restores the
    # paper-supplement recipe (clone at scale x relocate_scale_factor, donor
    # untouched). Measured: at thousands of events per run the split drains
    # opacity (v4) -- keep the event count near official (~245 per 30k steps)
    # with relocate_every / relocate_stop_frac / batch_cameras.
    relocate_mcmc_split: bool = True
    # Official TRiGS donor window perturbation: each selected donor's sigma_t
    # is divided by this, and its mu_t moves by U(+/- frac x sigma_t); clones
    # copy the result. 1.0 / 0.0 disable.
    relocate_source_duration_divisor: float = 1.0
    relocate_source_time_jitter_frac: float = 0.25
    # Keep a donor's trajectory when its window moves: its canonical mean and
    # quaternion follow it to the new mu_t and its Bézier residual collapses
    # to the value there. False applies the bare official mu_t shift, which
    # replays the whole trajectory at the wrong time.
    relocate_reanchor_source: bool = False
    # Relocate only while the (global) LR-schedule progress is below this
    # fraction. The official default is unreleased; 25000/30000 mirrors the
    # gsplat MCMCStrategy defaults its relocation code is adapted from.
    # >= 1 relocates to the end.
    relocate_stop_frac: float = 25_000 / 30_000
    """When hybrid static/dynamic rows are present, relocation only recycles
    non-static rows and only uses non-static donor rows."""
    static_background_mode: bool = True
    static_background_sigma_t: float = 0.0
    static_exclude_from_temporal_reg: bool = True
    static_exclude_from_motion_losses: bool = True
    # Keep trusted static NPZ centers fixed while still allowing their color,
    # opacity, orientation, and scale to fit the images.
    freeze_static_means: bool = False
    # --- Continuous Offset Learning ---
    learn_camera_time_offsets: bool = False
    lr_camera_time_offsets: float = 1e-4
    lambda_camera_time_offset_reg: float = 0.01
    # --- Camera time-offset refinement (well-conditioned) ---
    camera_time_offset_warmup_steps: int = 8000
    camera_time_offset_clamp_s: float = 0.0083  # ~0.5 frame @ 60fps
    # Path to an observability.json (prep/sync/measure-offset-observability.py).
    # Resolved into a per-training-position scale in [0, 1] by the CLI and
    # stashed at params["camera_time_offset_scale"] before training starts.
    # A camera whose photometric temporal gradient cannot beat its ladder
    # prior gets scale -> 0 and stays pinned near the sync.json/ladder value
    # instead of railing to the clamp under Adam's per-parameter step norm.
    camera_time_offset_observability_json: Optional[str] = None
    # --- Rolling-shutter aware rendering (OPT-IN, 0 disables) ---------------
    # A pixel's capture time depends on its COLUMN, not just its camera:
    #     t(u) = t_cam + s_cam * rho * (u / W - 0.5)
    # `camera_time_offsets` is one scalar per camera and structurally CANNOT
    # represent this: the gradient varies inside a single image and its sign
    # flips between camera groups. Left uncorrected it is an irreducible error
    # on moving content, scaling with subject speed -- fine detail on movers
    # (guitar frets) washes out while static geometry stays sharp.
    #
    # When enabled, each primitive is re-deformed at the capture time of the
    # column it projects to (screen-space approximation, cf. arXiv 2403.13327),
    # costing one extra deform per render and no extra rasterization.
    #
    # Requires per-camera signs on CameraConfig.rs_signs; without them this is
    # a no-op. Measure rho with prep/sync/audit-rolling-shutter.py.
    rs_readout_ms: float = 0.0
    """Full-span rolling-shutter readout in ms. 0 = off (global shutter)."""
    # Timing convention (trigs.capture_timing):  t = t_nominal + c_k + sign_k * readout * (s/S - 0.5), where c_k
    # (sync / timing file + learned offset) is the camera's MIDPOINT constant and s is the coordinate along the
    # readout axis. Defaults reproduce the original behaviour exactly.
    rs_axis: str = "column"
    """'column' (delivered columns; extracts rotated +-90, per-camera sign) or 'row' (unrotated landscape)."""
    rs_sensor_coords: bool = False
    """Map delivered (undistorted, cropped, resized) pixels to raw sensor lines via CameraConfig.rs_sensor_*."""
    rs_fixed_point_iters: int = 1
    """Extra passes that re-project the rolling-shutter-deformed means and recompute their capture times."""
    capture_timing_json: str = ""
    """Provenance only (set by the CLI): path and sha256 of the --timing-json file used, '' = sync.json.
    Eval rebuilds the run's capture times from it."""

    def rs_enabled(self) -> bool:
        return float(self.rs_readout_ms) > 0.0

    # --- Per-camera appearance correction (decouples exposure/WB from timing) ---
    learn_camera_appearance: bool = False
    lr_camera_appearance: float = 1e-3
    lambda_camera_appearance_reg: float = 1e-2

    # --- Temporal support / LongVolCap-inspired active rendering ---
    duration_init_mode: str = "class"
    # "class"        = SelfCap-style dense/sparse init
    # "npz"          = use NPZ durations directly
    # "npz_clamped"  = use NPZ durations with sane floor/ceil
    # "hybrid_npz"   = recommended: NPZ durations, but dense rows get a wider floor

    duration_min_s: float = 0.0
    duration_max_s: float = 0.0
    dense_duration_multiplier: float = 5.0
    sparse_duration_multiplier: float = 1.0

    temporal_active_cull: bool = True
    # Semantic support threshold used by diagnostics and legacy configs.
    temporal_active_threshold: float = 0.01
    # Optional lower hard-cull threshold.  <= 0 reuses
    # temporal_active_threshold.  Setting this to 0.01 or 0.005 makes rows
    # enter the renderer at 1%/0.5% temporal opacity instead of popping in at
    # 5%, at the cost of a wider resident set.
    temporal_active_cull_threshold: float = 0.0
    temporal_active_extra_margin_s: float = 0.0
    temporal_active_min_fraction: float = 0.0

    # Renormalize temporal opacity so gamma reaches exactly zero at the
    # semantic support edge: gamma' = clamp((gamma - tau) / (1 - tau), 0)
    # with tau = temporal_active_threshold.  Without this a row enters the
    # renderer as a step at tau (5%) opacity, already spatially extrapolated
    # by its full motion model; cohorts sharing a support edge (relocation
    # clones, NPZ block edges) pop in together as visible clouds.  With it,
    # hard culling at tau is mathematically exact (culled rows would have
    # contributed exactly zero) and every entry is a continuous fade from
    # zero.  Applied identically in training and rendering.
    temporal_gamma_smooth_cull: bool = True

    temporal_hierarchy_enabled: bool = False
    temporal_hierarchy_levels: int = 9
    temporal_hierarchy_root_s: float = 10.0
    temporal_hierarchy_rebuild_every: int = 250

    # Direct-vs-hierarchy validation during development. Set to 0 later.
    temporal_hierarchy_validate_every: int = 50

    # Adam beta1 for mu_t and log sigma_t.
    # Official TRiGS uses 0.9 for every param. 0 was once required because
    # the hierarchy only re-filed rows that had a gradient, so culled rows
    # moving on momentum went stale; rows are now re-filed by value change.
    temporal_optimizer_beta1: float = 0.9

    temporal_hierarchy_use_direct_debug: bool = False

    # Zero disables outlier-drift-triggered rebuilds. Periodic and structural
    # rebuilds still happen.
    temporal_hierarchy_max_drift_finest_segments: float = 0.0

    # Dynamic rows whose sequence-clipped influence interval covers at least
    # this fraction of the sequence are promoted to the global bucket instead
    # of distorting the tree. <= 0 or >= 1 disables promotion.
    temporal_hierarchy_global_coverage_fraction: float = 0.5

    # Per-row drift slack as a fraction of the finest segment. Rows drifting
    # further join a bounded overflow candidate set; the query window stays
    # [t - slack, t + slack] instead of growing with worst-case drift.
    temporal_hierarchy_drift_slack_finest: float = 0.5

    # Rebuild when the overflow set exceeds this fraction of dynamic rows.
    temporal_hierarchy_overflow_rebuild_fraction: float = 0.05

    # Second half-shifted grid per level (LongVolCap-style offsets) so short
    # intervals straddling a boundary are not promoted to coarser levels.
    temporal_hierarchy_offset_grids: bool = True

    # Queries slightly outside [0, t_max] (learned camera time offsets)
    # remain exact within this pad.
    temporal_hierarchy_bounds_pad_s: float = 0.05

    # ------------------------------------------------------------------
    # Temporal chunked training (LongVolCap-style residency). > 0 enables:
    # the full model lives in CPU RAM; each chunk materializes only the
    # rows whose influence intervals intersect the chunk window (plus
    # static rows) on the GPU, so parameters, gradients, and optimizer
    # state scale with the chunk, not the sequence.
    # ------------------------------------------------------------------
    temporal_chunk_len_s: float = 0.0
    temporal_chunk_overlap_s: float = 0.5
    # Passes over all chunks. Later rounds refine against neighbors'
    # updated boundaries; total n_steps is divided across rounds x chunks.
    temporal_chunk_rounds: int = 2
    # Interleaved chunk visits (> 0 enables; temporal_chunk_rounds is then
    # derived as n_steps / (visit_steps * n_windows)). Instead of a few long
    # calls per window, cycle through the windows many times in visits of
    # about this many steps, in a shuffled order each cycle. Each cycle is a
    # thin slice of the LR schedule, so a row spanning several windows is
    # updated by all of them in alternation and late visits run at the final
    # LR. With long calls, the last window touching a long-lived row trained
    # it for thousands of steps at mid-schedule LR on its own frames only,
    # which on example capture left such rows visible outside the silhouette
    # in every earlier frame (-5.8 dB full-frame eval). Temporal support is
    # left unbounded, as in official TRiGS (a random frame per step).
    # In this mode Adam moments are carried with the rows between visits (in
    # host RAM, about 2x the optimized parameters), relocation follows the
    # whole-run schedule (relocate_start_iter / relocate_every /
    # relocate_stop_frac) in every cycle, static rows train in every visit,
    # and the outside-mass penalty is not applied (rows' out-of-window frames
    # are supervised by other windows' visits).
    temporal_chunk_visit_steps: int = 0
    # Keep lambda_temporal_chunk_outside_mass on in interleaved visits (off by default there).
    # A visit renders only its window's frames, so within a visit nothing else stops an owned
    # row's temporal support (and the motion it carries) from growing into other windows' time.
    temporal_chunk_visit_outside_mass: bool = False

    # Freeze rows not owned by the current chunk (ownership = the window
    # whose interior contains mu_t, with overlap midpoints as boundaries).
    # Non-owned rows stay resident for rendering context but receive no
    # gradient, preventing each chunk from damaging its neighbors' fits.
    temporal_chunk_freeze_nonowned: bool = True

    # "first_round": static rows train in every chunk of the first round,
    # then freeze (background is time-invariant; per-chunk re-fitting in
    # later rounds thrashes it, but a single chunk call under-trains it).
    # "first": static trains only in the very first chunk call.
    # "all": static rows train in every chunk (original behavior).
    temporal_chunk_train_static: str = "first_round"

    # Relocation in the final round leaves freshly teleported, half-trained
    # rows in the master at every chunk end. Off by default.
    temporal_chunk_relocate_final_round: bool = False

    # Supervise the complete temporal support of rows owned by a chunk.
    #   < 0: automatically extend each chunk to the current cull-support of
    #        its owned dynamic rows.  WARNING: this is max-based, so a single
    #        wide-sigma owner row expands the window to (nearly) the whole
    #        sequence, restoring full-model residency, diluting the step
    #        budget over every frame, and letting owner rows grow duration to
    #        explain frames whose true owners are frozen (results-th-04
    #        sigma_t bifurcation).  Do not use without a percentile rewrite
    #        and a hard pad cap.
    #   = 0: render only the nominal chunk window (default).  Boundary-
    #        crossing supports are handled by interval-based trainability
    #        (every rendered (row, frame) pair is optimizable in the chunk
    #        owning that frame) plus the outside-mass loss, which becomes
    #        active exactly in this mode.
    #   > 0: add a fixed number of seconds to each side of the nominal window
    temporal_chunk_supervision_pad_s: float = 0.0
    # Optional per-side cap for automatic/fixed supervision padding. 0 means
    # no cap. If a cap clips support, the outside-mass loss below discourages
    # temporal support from growing into the unsupervised region.
    temporal_chunk_supervision_pad_max_s: float = 0.0
    # Soft penalty on Gaussian temporal mass beyond the frames supervised by
    # the owning chunk. The loss is zero while the configured cull-support is
    # contained and only activates when mu_t/sigma_t drift outside it.
    lambda_temporal_chunk_outside_mass: float = 1e-3
    # Bound the per-step cost of the analytic erf-based mass penalty. 0 uses
    # every owned row; positive values sample with replacement when needed.
    temporal_chunk_outside_mass_sample_size: int = 65_536
    # Print the largest per-frame active-entry cohorts before training and
    # after every chunk round, including dense/sparse/relocated composition
    # and exact deformed displacement statistics for the entering rows.
    temporal_chunk_entry_diagnostics: bool = False
    temporal_chunk_entry_diagnostics_topk: int = 12
    temporal_chunk_entry_diagnostics_max_rows: int = 50_000

    # Skip relocation during the last N steps of a run so relocated rows
    # always get time to integrate before the model is frozen/scattered.
    # 0 disables. Chunked training sets this per chunk when unset.
    relocate_settle_steps: int = 0

    # Continue one global LR decay schedule across chunk calls instead of
    # restarting the schedule every chunk. 0 = per-call schedule.
    lr_schedule_offset_steps: int = 0
    lr_schedule_total_steps: int = 0
    # Local-to-global step scaling for the shared schedule:
    #     global_step = lr_schedule_offset_steps + step * lr_schedule_step_scale
    # Chunked training uses this to give every window in a round the SAME LR
    # band. Keying the offset off the running global step instead made the
    # band depend on a window's position in the time-ordered chunk loop: with
    # 9 windows x 2 rounds the final (authoritative) write for t=0 ran at
    # ~10x the LR of the write for the last window, so reconstruction quality
    # ramped monotonically along the sequence. Rounds still anneal against
    # each other because each round maps onto its own slice of the schedule.
    lr_schedule_step_scale: float = 1.0
    # Steps already trained before this call. The SH degree warms up on the
    # cumulative step (official TRiGS: min(it // 1000, sh_degree) on the
    # global iteration); keyed on the local step, every chunk call dropped
    # back to degree 0 for 1000 steps and re-fit view dependence into sh0.
    sh_warmup_step_offset: int = 0

    # De-cohort temporal supports at init (opt-in, both off by default).
    # NPZ/track pipelines that process fixed keyframe blocks leave cohorts of
    # rows whose supports END at the same absolute time; the shared fade
    # edge renders as splats "releasing" from moving subjects. Jitter
    # scatters mu_t by uniform +/- this many seconds (dynamic rows only);
    # dilate multiplies initial sigma_t so adjacent blocks cross-fade.
    temporal_init_jitter_s: float = 0.0
    temporal_init_sigma_dilate: float = 1.0

    # Opt-in: jitter relocated clones' mu_t by uniform
    # +/- (this factor x donor sigma_t). Relocation otherwise copies donor
    # mu_t exactly, replicating cohort structure every pass instead of
    # diffusing it. 0 disables (TRiGS-paper behavior).
    relocate_time_jitter_sigma: float = 0.0

@dataclass
class TrigsTrainingResult:
    """Return payload of :meth:`TrigsTrainer.train`."""

    params: nn.ParameterDict
    temporal_params: Dict[str, Tensor]
    motion_params: Dict[str, Tensor]
    metrics: Dict[str, Any] = field(default_factory=dict)
    # Adam moments per parameter name (CPU), when requested by the caller.
    optimizer_state: Dict[str, Dict[str, Tensor]] = field(default_factory=dict)


def _as_float_rgb01(x: Tensor, clamp: bool = True) -> Tensor:
    """Convert RGB tensor to float [0,1], preserving HWC shape.

    Only uint8 input is rescaled. Float input is already in [0, 1] units:
    guessing from its magnitude divided a whole render by 255 as soon as one
    pixel passed 2.0 (possible once the loss clamps the render, since pixels
    above 1 get no gradient), which pinned psnr/eval_* at the GT's energy.
    """
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
    if clamp:
        x = x.clamp(0.0, 1.0)
    return x


def _bounded_diagnostic_sample(
    values: Tensor,
    max_samples: int = 1_000_000,
) -> Tensor:
    """Return a deterministic flat sample for expensive diagnostic statistics.

    CUDA ``torch.quantile`` rejects very large inputs, and sorting tens of
    millions of values for logging is wasteful even where it is supported.
    Training computations must use the full tensors; this helper is only for
    human-facing summaries and TensorBoard statistics.
    """
    flat = values.reshape(-1)
    limit = max(int(max_samples), 1)
    n = int(flat.numel())
    if n <= limit:
        return flat
    if limit == 1:
        return flat[:1]
    positions = torch.arange(limit, device=flat.device, dtype=torch.int64)
    indices = torch.div(
        positions * (n - 1),
        limit - 1,
        rounding_mode="floor",
    )
    return flat.index_select(0, indices)


def _lr_schedule_progress(config: TrigsTrainingConfig, step: int) -> float:
    """Fraction of the LR decay schedule consumed at ``step``, in [0, 1].

    With ``lr_schedule_total_steps > 0`` the call is one slice of a schedule
    shared with other calls (chunked training):

        global_step = lr_schedule_offset_steps + step * lr_schedule_step_scale

    so several calls can be given the same band by sharing an offset and
    scaling their local steps onto it. Otherwise the schedule is local to the
    call and spans ``lr_decay_steps`` (or ``n_steps``).
    """
    total = int(getattr(config, "lr_schedule_total_steps", 0))
    if total > 0:
        scale = float(getattr(config, "lr_schedule_step_scale", 1.0))
        if not (scale > 0.0):
            scale = 1.0
        global_step = float(step) * scale + float(
            getattr(config, "lr_schedule_offset_steps", 0)
        )
        return min(global_step / max(total - 1, 1), 1.0)

    decay_steps = (
        config.lr_decay_steps if config.lr_decay_steps > 0 else config.n_steps
    )
    return min(step / max(decay_steps - 1, 1), 1.0)


OPACITY_DIAG_BANDS: tuple[tuple[str, float, float], ...] = (
    ("sigma_0_3", 0.0, 3.0),
    ("sigma_3_6", 3.0, 6.0),
    ("sigma_6_12", 6.0, 12.0),
    ("sigma_12_48", 12.0, 48.0),
    ("sigma_48_200", 48.0, 200.0),
    ("sigma_200_inf", 200.0, float("inf")),
)


@torch.no_grad()
def opacity_diag_summary(
    *,
    sigma_frames: Tensor,
    is_static: Tensor,
    opacity: Tensor,
    photo: Tensor,
    reg: Tensor,
    delta: Tensor,
    photo_neg: Tensor,
    photo_nz: Tensor,
) -> Dict[str, Dict[str, float]]:
    """Per-lifetime-band summary of one window's opacity forces.

    ``photo`` / ``reg`` are per-row sums of the opacity-logit gradient over the
    window (positive pushes opacity down), ``delta`` the summed logit change the
    optimizer applied, ``photo_neg`` / ``photo_nz`` the number of steps whose
    photo gradient was negative / nonzero. Only rows the photo loss touched in
    the window count; still-class rows form their own band. Per band:
    ``photo_wants_more`` (photo sum < 0), ``net_drains`` (photo + reg sum > 0),
    ``reg_over_photo_p50``, ``steps_say_less`` (most touched steps had photo
    > 0), ``dlogit_p50`` and the ``rose`` / ``fell`` shares (|change| > 1e-3).
    """
    seen = photo_nz > 0
    bands = [
        (name, (~is_static) & (sigma_frames >= lo) & (sigma_frames < hi))
        for name, lo, hi in OPACITY_DIAG_BANDS
    ]
    bands.append(("still_class", is_static))
    out: Dict[str, Dict[str, float]] = {}
    for name, mask in bands:
        m = mask & seen
        n = int(m.sum())
        if n == 0:
            continue
        p, r, d = photo[m], reg[m], delta[m]
        out[name] = {
            "count": float(n),
            "opacity_p50": float(opacity[m].median()),
            "photo_wants_more": float((p < 0).float().mean()),
            "net_drains": float(((p + r) > 0).float().mean()),
            "reg_over_photo_p50": float((r.abs() / p.abs().clamp_min(1e-30)).median()),
            "steps_say_less": float((photo_neg[m] < 0.5 * photo_nz[m]).float().mean()),
            "dlogit_p50": float(d.median()),
            "rose": float((d > 1e-3).float().mean()),
            "fell": float((d < -1e-3).float().mean()),
        }
    return out


def _log_opacity_diag(summary: Dict[str, Dict[str, float]], writer: Any, step: int) -> None:
    if writer is not None:
        for band, stats in summary.items():
            for key, value in stats.items():
                writer.add_scalar(f"opacity_diag/{band}/{key}", value, step)
    line = " | ".join(
        f"{band} n={int(s['count'])} op={s['opacity_p50']:.2f} "
        f"more={s['photo_wants_more']:.0%} drains={s['net_drains']:.0%} "
        f"less_steps={s['steps_say_less']:.0%} d={s['dlogit_p50']:+.4f} "
        f"rose={s['rose']:.0%} fell={s['fell']:.0%}"
        for band, s in summary.items()
    )
    print(f"[opacity_diag] step {step}: {line}", flush=True)


def _lr_decay_ratios(config: TrigsTrainingConfig) -> Dict[str, float]:
    """final/initial LR per exponentially decayed optimizer group."""
    ratios: Dict[str, float] = {
        "anchors": config.lr_anchor_final / max(config.lr_anchor, 1e-30),
        "zeta_base_t": config.lr_zeta_base_trans_final / max(config.lr_zeta_base_trans, 1e-30),
        "zeta_bezier_t": config.lr_zeta_bezier_trans_final / max(config.lr_zeta_bezier_trans, 1e-30),
        "zeta_base_r": config.lr_zeta_base_rot_final / max(config.lr_zeta_base_rot, 1e-30),
        "zeta_bezier_r": config.lr_zeta_bezier_rot_final / max(config.lr_zeta_bezier_rot, 1e-30),
    }
    if not config.freeze_geometry:
        ratios["means"] = config.lr_means_final / max(config.lr_means, 1e-30)
    return ratios


def training_camera_extent(
    viewmats: Tensor, exclude: Sequence[int] = (),
) -> float:
    """Official TRiGS scene extent: the largest distance of a training
    camera center from the mean center. ``viewmats`` are world-to-camera;
    ``exclude`` lists held-out camera positions.
    """
    drop = {int(i) for i in exclude}
    keep = [i for i in range(int(viewmats.shape[0])) if i not in drop]
    if not keep:
        raise ValueError("training_camera_extent: every camera is excluded")
    centers = torch.linalg.inv(f64(viewmats[keep]))[:, :3, 3]
    return float((centers - centers.mean(dim=0)).norm(dim=-1).max().item())


def _build_camera_batch_epoch(
    frames: Sequence[int],
    train_views: Sequence[int],
    batch: int,
) -> List[tuple[int, List[int]]]:
    """One epoch of (frame, camera group) steps covering every pair.

    Frames are shuffled once; each frame's cameras are shuffled and split
    into groups of ``batch``, and round r visits every frame with its r-th
    group. A short last group is topped up with other cameras of the same
    frame, so every step renders ``batch`` distinct views. With batch == 1
    this is the single-pair schedule, with the same RNG consumption.
    """
    frame_order = [int(fi) for fi in frames]
    random.shuffle(frame_order)
    batch = min(max(int(batch), 1), len(train_views))
    groups_by_frame: Dict[int, List[List[int]]] = {}
    for fi in frame_order:
        views = list(train_views)
        random.shuffle(views)
        groups = [views[i:i + batch] for i in range(0, len(views), batch)]
        short = batch - len(groups[-1])
        if short > 0:
            extra = [v for v in views if v not in groups[-1]]
            random.shuffle(extra)
            groups[-1] = groups[-1] + extra[:short]
        groups_by_frame[fi] = groups
    n_rounds = math.ceil(len(train_views) / batch)
    return [
        (fi, groups_by_frame[fi][r])
        for r in range(n_rounds)
        for fi in frame_order
    ]


def _draw_official_camera_batch(
    frames: Sequence[int],
    train_views: Sequence[int],
    batch: int,
    *,
    frame_weights: Optional[Sequence[float]] = None,
    weighted_frac: float = 0.0,
) -> tuple[int, List[int]]:
    """Official TRiGS step: a uniformly random frame, and ``batch`` cameras of
    it drawn with replacement (FastInfiniteLoader + random.choices).

    With ``frame_weights`` (aligned with ``frames``), ``weighted_frac`` of the steps
    draw the frame from the weights instead. Without them the random stream is
    exactly the uniform one."""
    frame_list = [int(fi) for fi in frames]
    if frame_weights is not None and weighted_frac > 0.0 and random.random() < weighted_frac:
        frame = random.choices(frame_list, weights=frame_weights, k=1)[0]
    else:
        frame = frame_list[random.randint(0, len(frame_list) - 1)]
    return frame, random.choices(list(train_views), k=max(int(batch), 1))


def fast_frame_topup(counts: Sequence[float], n_steps: int, target_visits: float,
                     full_rows: int, multiplicity: Optional[Sequence[float]] = None,
                     ) -> tuple[List[float], int]:
    """Per-frame expected visits and the extra steps for frame_sampling_fast_visits: every frame
    keeps its normal visits; a frame with short-lived rows is topped up toward ``target_visits``,
    fully at ``full_rows`` rows or more. Normal visits are n_steps / frames, or, with
    ``multiplicity`` (how many chunk windows train each frame), n_steps shared in proportion to it.
    Returns (visits per frame, extra steps)."""
    n = max(len(counts), 1)
    mult = [1.0] * len(counts) if multiplicity is None else [max(float(m), 0.0) for m in multiplicity]
    total = sum(mult) or float(n)
    base = [float(n_steps) * m / total for m in mult]
    extra = [max(float(target_visits) - b, 0.0) * min(1.0, float(c) / max(float(full_rows), 1.0))
             for b, c in zip(base, counts)]
    return [b + e for b, e in zip(base, extra)], int(round(sum(extra)))


@torch.no_grad()
def alive_rows_per_frame(params: Mapping[str, Tensor], n_frames: int, fps: float, tau_alpha: float) -> Tensor:
    """Dynamic rows with opacity >= tau_alpha, binned by the frame nearest their mu_t (CPU, int64)."""
    mu = params["times"].detach().reshape(-1)
    keep = torch.sigmoid(params["opacities"].detach().reshape(-1)) >= float(tau_alpha)
    if "is_static" in params:
        keep &= params["is_static"].detach().reshape(-1) <= 0.5
    idx = torch.round(f64(mu[keep]) * float(fps)).long().clamp(0, max(int(n_frames) - 1, 0))
    return torch.bincount(idx.cpu(), minlength=int(n_frames))


@torch.no_grad()
def short_lived_rows_per_frame(
    params: Mapping[str, Tensor],
    frames: Sequence[int],
    fps: float,
    max_sigma_frames: float,
) -> List[float]:
    """Per frame, the dynamic rows with sigma_t <= ``max_sigma_frames`` frames whose
    +/-2 sigma window (their Bezier window) covers it."""
    fps = max(float(fps), 1e-6)
    mu = params["times"].detach().reshape(-1).cpu().double() * fps
    sigma = params["log_durations"].detach().reshape(-1).cpu().double().exp() * fps
    keep = sigma <= float(max_sigma_frames)
    if "is_static" in params:
        keep &= params["is_static"].detach().reshape(-1).cpu() <= 0.5
    mu, sigma = mu[keep], sigma[keep]
    if int(mu.numel()) == 0:
        return [0.0 for _ in frames]
    last = max(int(fi) for fi in frames) + 2
    # Rows sit on whole frames with whole-frame sigma, so window edges land on frames: without the
    # tolerance, float32 rounding of times/log_durations would drop an edge frame at random.
    eps = 1e-3
    lo = torch.ceil(mu - 2.0 * sigma - eps).long().clamp(0, last)
    hi = torch.floor(mu + 2.0 * sigma + eps).long().clamp(-1, last - 1) + 1
    edges = torch.bincount(lo, minlength=last + 1).double() - torch.bincount(hi.clamp(min=0), minlength=last + 1).double()
    active = torch.cumsum(edges, dim=0)
    return [float(active[int(fi)].item()) for fi in frames]


def _ticking_range(n_steps: int, bar: Any):
    """range(n_steps) that advances an outer progress bar by one per completed step."""
    for i in range(int(n_steps)):
        yield i
        bar.update(1)


def _capacity_summary(row: Mapping[str, float]) -> str:
    return (f"  Capacity: fast-frame share of alive splats {row['fast_share_start']:.1%} -> "
            f"{row['fast_share']:.1%}; other frames keep {row['slow_ratio_median']:.0%} of their "
            f"starting splats (median; lowest frame {row['slow_ratio_min']:.0%}). Includes normal "
            f"opacity pruning: compare with a run at --config.frame-sampling-fast-visits 0")


def _relocation_past_stop(config: TrigsTrainingConfig, step: int) -> bool:
    """True once the global schedule passes relocate_stop_frac."""
    stop = float(getattr(config, "relocate_stop_frac", 1.0))
    return stop < 1.0 and _lr_schedule_progress(config, step) >= stop


def _sh_degree_for_step(config: TrigsTrainingConfig, step: int, max_degree: int) -> int:
    """Active SH degree: one band per 1000 cumulative steps, capped."""
    offset = max(int(getattr(config, "sh_warmup_step_offset", 0)), 0)
    return min((int(step) + offset) // 1000, int(max_degree))


def _motion_residual_magnitude_loss(
    c0: Tensor,
    c1: Tensor,
    c2: Tensor,
    rows: Tensor | slice = slice(None),
) -> Tensor:
    """Mean squared residual magnitude, with an empty-selection safe zero.

    All-static models deliberately select zero dynamic rows. PyTorch's mean
    over that empty tensor is NaN, which previously poisoned the total loss
    whenever ``lambda_motion_residual`` was non-zero.
    """
    c0_selected = c0[rows]
    if int(c0_selected.numel()) == 0:
        return c0.new_zeros(())
    return (
        (c0_selected ** 2).sum(dim=-1)
        + (c1[rows] ** 2).sum(dim=-1)
        + (c2[rows] ** 2).sum(dim=-1)
    ).mean()


def _as_mask01(mask: Optional[Tensor], shape_hw: tuple[int, int], *, threshold: float = 0.5) -> Optional[Tensor]:
    """Return mask as float [H,W,1] in {0,1}; None if unavailable."""
    if mask is None:
        return None

    m = mask.float()
    if m.max().detach() > 2.0:
        m = m / 255.0

    if m.ndim == 2:
        m = m[..., None]
    elif m.ndim == 3 and m.shape[-1] != 1:
        m = m[..., :1]

    H, W = shape_hw
    if int(m.shape[0]) != H or int(m.shape[1]) != W:
        m = torch.nn.functional.interpolate(
            m.permute(2, 0, 1).unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0].permute(1, 2, 0)

    return (m >= float(threshold)).float()


@torch.no_grad()
def compute_psnr_metrics(
    pred_rgb: Tensor,
    gt_rgb: Tensor,
    mask: Optional[Tensor] = None,
    *,
    mask_threshold: float = 0.5,
    clamp: bool = True,
    eps: float = 1e-12,
) -> Dict[str, float]:
    pred = _as_float_rgb01(pred_rgb, clamp=clamp)
    gt = _as_float_rgb01(gt_rgb, clamp=clamp)

    if tuple(pred.shape) != tuple(gt.shape):
        raise RuntimeError(f"PSNR shape mismatch: pred={tuple(pred.shape)} gt={tuple(gt.shape)}")

    diff2 = f64(pred - gt).pow(2)
    mse_full = diff2.mean().clamp_min(eps)
    out: Dict[str, float] = {
        "mse_full": float(mse_full.item()),
        "psnr_full": float((-10.0 * torch.log10(mse_full)).item()),
    }

    m = _as_mask01(mask, (pred.shape[0], pred.shape[1]), threshold=mask_threshold)
    if m is not None:
        m = m.to(device=diff2.device, dtype=torch.float64)
        denom = (m.sum() * pred.shape[-1]).clamp_min(1.0)
        mse_masked = (diff2 * m).sum() / denom
        mse_masked = mse_masked.clamp_min(eps)
        out.update({
            "mse_masked": float(mse_masked.item()),
            "psnr_masked": float((-10.0 * torch.log10(mse_masked)).item()),
            "mask_pixels": float(m.sum().item()),
        })

    return out


def _make_psnr_eval_pairs(
    n_frames: int,
    n_views: int,
    num_samples: int,
    seed: int,
    views: Sequence[int] | None = None,
) -> List[tuple[int, int]]:
    """Build a fixed (frame, view) evaluation grid.

    Args:
        views: Restrict evaluation to these view positions. ``None`` uses all
            ``n_views``. Pass the held-out cameras to get a metric that is not
            a training-set fit; pass the trained cameras to reproduce the
            historical psnr/eval_* series.
    """
    view_pool = list(range(int(n_views))) if views is None else [int(v) for v in views]
    if not view_pool:
        return []
    n_view_pool = len(view_pool)
    total = int(n_frames) * n_view_pool
    if num_samples <= 0:
        return []
    if num_samples >= total:
        return [(f, v) for f in range(n_frames) for v in view_pool]

    n_eval_frames = min(n_frames, max(1, int(round(math.sqrt(num_samples)))))
    n_eval_views = min(n_view_pool, max(1, int(math.ceil(num_samples / n_eval_frames))))

    if n_eval_frames == 1:
        frames = [0]
    else:
        frames = [
            int(round(i * (n_frames - 1) / max(n_eval_frames - 1, 1)))
            for i in range(n_eval_frames)
        ]

    if n_eval_views == 1:
        views_sel = [view_pool[0]]
    else:
        views_sel = [
            view_pool[int(round(i * (n_view_pool - 1) / max(n_eval_views - 1, 1)))]
            for i in range(n_eval_views)
        ]

    pairs = sorted(set((f, v) for f in frames for v in views_sel))
    rng = random.Random(int(seed))
    if len(pairs) > num_samples:
        pairs = sorted(rng.sample(pairs, num_samples))
    return pairs


# =============================================================================
# Step 2: Relocation pool masks and flag copier
# =============================================================================

def _relocation_pool_masks(params: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      dynamic_pool [N] bool: rows allowed to participate in dynamic relocation
      static_pool  [N] bool: rows allowed to participate in static relocation

    No cross-pool relocation:
      - dynamic rows do not donate to static rows
      - static rows do not donate to dynamic rows
      - dynamic rows outside dynamic_allowed are disallowed
    """
    device = params["means"].device
    n = params["means"].shape[0]

    is_static = params.get("is_static", None)
    if is_static is None:
        is_static_b = torch.zeros((n,), device=device, dtype=torch.bool)
    else:
        is_static_b = is_static.detach().reshape(-1) > 0.5

    dynamic_allowed = params.get("dynamic_allowed", None)
    if dynamic_allowed is None:
        dynamic_allowed_b = torch.ones((n,), device=device, dtype=torch.bool)
    else:
        dynamic_allowed_b = dynamic_allowed.detach().reshape(-1) > 0.5

    dynamic_pool = (~is_static_b) & dynamic_allowed_b
    static_pool = is_static_b & (~dynamic_allowed_b)

    return dynamic_pool, static_pool


@torch.no_grad()
def _copy_relocation_flags_same_pool(
    params: dict,
    dst_idx: torch.Tensor,
    src_idx: torch.Tensor,
) -> None:
    """
    Preserve/copy non-trainable row class fields during relocation.

    This is safe when dst/src are already chosen from the same relocation pool.
    """
    for key in ("is_static", "dynamic_protect", "dynamic_allowed"):
        if key in params:
            params[key].data[dst_idx] = params[key].data[src_idx]


# =============================================================================
# Step 4: Hard post-relocation class invariant enforcer
# =============================================================================

@torch.no_grad()
def _enforce_relocation_class_invariants(params: dict) -> int:
    """Returns the number of rows demoted dynamic -> static.

    A nonzero return means row activity semantics changed (static rows are
    active at every t); callers holding a temporal hierarchy must invalidate
    it, or demoted rows go missing from queries outside their stale segment.
    """
    if "is_static" not in params:
        return 0

    is_static = params["is_static"].detach().reshape(-1) > 0.5

    if "dynamic_allowed" in params:
        dynamic_allowed = params["dynamic_allowed"].detach().reshape(-1) > 0.5
    else:
        dynamic_allowed = ~is_static

    # Static rows are not dynamic-allowed/protected.
    if "dynamic_allowed" in params:
        params["dynamic_allowed"].data[is_static] = 0.0

    if "dynamic_protect" in params:
        params["dynamic_protect"].data[is_static] = 0.0

    # Any non-static row outside dynamic_allowed is illegal. Demote to static.
    illegal_dynamic = (~is_static) & (~dynamic_allowed)
    n_demoted = int(illegal_dynamic.sum().item())
    if illegal_dynamic.any():
        params["is_static"].data[illegal_dynamic] = 1.0
        if "dynamic_protect" in params:
            params["dynamic_protect"].data[illegal_dynamic] = 0.0

    # Re-read after potential demotion above.
    is_static = params["is_static"].detach().reshape(-1) > 0.5

    # Static rows must not carry motion.
    for key in (
        "zeta_base_t",
        "zeta_base_r",
        "zeta_c0_t",
        "zeta_c1_t",
        "zeta_c2_t",
        "zeta_c0_r",
        "zeta_c1_r",
        "zeta_c2_r",
    ):
        if key in params:
            params[key].data[is_static] = 0.0

    if "anchors" in params:
        params["anchors"].data[is_static] = params["means"].data[is_static]

    return n_demoted


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
        self.temporal_hierarchy = None
        self.temporal_hierarchy_last_validation_step = -1

    def _slice_rowwise_params(
        self,
        params: nn.ParameterDict | dict,
        idx: Tensor,
    ) -> dict:
        n = int(params["means"].shape[0])
        out = {}
        for k, v in params.items():
            if torch.is_tensor(v) and v.ndim >= 1 and int(v.shape[0]) == n:
                out[k] = v[idx]
            else:
                out[k] = v
        return out

    def _invalidate_temporal_hierarchy(self) -> None:
        if self.temporal_hierarchy is not None:
            self.temporal_hierarchy.invalidate()

        self.temporal_hierarchy_last_validation_step = -1

    @staticmethod
    def _effective_temporal_cull_threshold(
        config: TrigsTrainingConfig,
    ) -> float:
        """Hard residency/render cutoff, optionally below the semantic 5% edge."""
        base = float(getattr(config, "temporal_active_threshold", 0.05))
        override = float(
            getattr(config, "temporal_active_cull_threshold", 0.0)
        )
        value = override if override > 0.0 else base
        # A lower cutoff can only make residency more conservative.  If the
        # override is accidentally higher, retain the legacy/base threshold.
        return min(max(value, 1e-8), min(max(base, 1e-8), 1.0 - 1e-8))

    @staticmethod
    @torch.no_grad()
    def _temporal_snapshot(params: nn.ParameterDict) -> Dict[str, Tensor]:
        """Copy of the temporal params the hierarchy files rows by."""
        return {
            name: params[name].detach().clone()
            for name in ("times", "log_durations")
            if name in params
        }

    @staticmethod
    @torch.no_grad()
    def _temporal_rows_changed(
        params: nn.ParameterDict,
        before: Dict[str, Tensor],
    ) -> Tensor:
        """Dynamic rows whose mu_t or sigma_t differ from ``before``.

        Value-based rather than gradient-based: with Adam momentum (beta1 >
        0) a culled row with zero gradient still moves, and a gradient-only
        test would leave it filed under a stale hierarchy segment. NaN
        compares unequal, so such rows are conservatively included.
        """
        n = int(params["means"].shape[0])
        changed = torch.zeros(n, device=params["means"].device, dtype=torch.bool)
        for name, old in before.items():
            now = params[name].detach()
            if now.shape != old.shape:
                # Row count changed (pruning); the hierarchy is rebuilt.
                return torch.arange(n, device=changed.device)
            changed |= (now != old).reshape(n, -1).any(dim=1)
        if "is_static" in params:
            changed &= ~(params["is_static"].detach().reshape(-1) > 0.5)
        return torch.nonzero(changed, as_tuple=False).reshape(-1)

    def _active_idx_for_time(
        self,
        params: nn.ParameterDict,
        t: float | Tensor,
        config: TrigsTrainingConfig,
        *,
        step: int | None = None,
    ) -> Tensor:
        from .temporal_hierarchy import (
            TemporalGaussianHierarchy,
            direct_temporal_active_idx,
        )

        n = int(params["means"].shape[0])
        device = params["means"].device
        step_i = 0 if step is None else int(step)

        if not config.temporal_active_cull:
            return torch.arange(
                n,
                device=device,
                dtype=torch.long,
            )

        if (
            not config.temporal_hierarchy_enabled
            or getattr(config, "temporal_hierarchy_use_direct_debug", False)
        ):
            return direct_temporal_active_idx(
                params,
                t,
                opacity_threshold=self._effective_temporal_cull_threshold(config),
                extra_margin_s=config.temporal_active_extra_margin_s,
                include_static=True,
            )

        if self.temporal_hierarchy is None:
            t_max = getattr(self, "_sequence_t_max", None)
            self.temporal_hierarchy = TemporalGaussianHierarchy(
                levels=config.temporal_hierarchy_levels,
                root_segment_s=config.temporal_hierarchy_root_s,
                opacity_threshold=self._effective_temporal_cull_threshold(config),
                extra_margin_s=config.temporal_active_extra_margin_s,
                include_static=True,
                t_min=0.0 if t_max is not None else None,
                t_max=t_max,
                bounds_pad_s=getattr(
                    config, "temporal_hierarchy_bounds_pad_s", 0.05,
                ),
                global_coverage_fraction=getattr(
                    config,
                    "temporal_hierarchy_global_coverage_fraction",
                    0.5,
                ),
                drift_slack_finest=getattr(
                    config, "temporal_hierarchy_drift_slack_finest", 0.5,
                ),
                offset_grids=getattr(
                    config, "temporal_hierarchy_offset_grids", True,
                ),
            )

        hierarchy = self.temporal_hierarchy

        if hierarchy.needs_rebuild(
            n_rows=n,
            step=step_i,
            rebuild_every=config.temporal_hierarchy_rebuild_every,
            overflow_fraction=getattr(
                config,
                "temporal_hierarchy_overflow_rebuild_fraction",
                0.05,
            ),
            max_drift_finest_segments=getattr(
                config,
                "temporal_hierarchy_max_drift_finest_segments",
                0.0,
            ),
        ):
            hierarchy.rebuild(
                params,
                step=step_i,
            )

            stats = hierarchy.distribution_stats(params)

            log.debug(
                "temporal hierarchy rebuild @ step %d: "
                "total=%d global=%d static_global=%d dynamic_global=%d "
                "root=%.4fs finest=%.6fs slack=%.6fs levels=%s",
                step_i,
                stats["n_rows"],
                stats["global"],
                stats.get("global_static", 0),
                stats.get("global_dynamic", 0),
                stats["effective_root_s"],
                stats["finest_segment_s"],
                stats.get("drift_slack_s", 0.0),
                stats["levels"],
            )

        active_idx = hierarchy.query(params, t)

        # Defensive fallback only. With persistent static rows, this should
        # ordinarily never execute.
        if int(active_idx.numel()) == 0:
            active_idx = direct_temporal_active_idx(
                params,
                t,
                opacity_threshold=self._effective_temporal_cull_threshold(config),
                extra_margin_s=config.temporal_active_extra_margin_s,
                include_static=True,
            )

        validate_every = int(
            config.temporal_hierarchy_validate_every
        )
        if (
            validate_every > 0
            and step_i % validate_every == 0
            and self.temporal_hierarchy_last_validation_step != step_i
        ):
            mismatch = self._validate_temporal_hierarchy(
                params=params,
                t=t,
                config=config,
                step=step_i,
                hierarchy_idx=active_idx,
            )
            if mismatch is not None:
                # A stale index (missed drift/staleness notification) is
                # cured by a rebuild. Rebuild, requery, and revalidate; only
                # a mismatch that survives a fresh rebuild is a genuine
                # hierarchy bug worth aborting a long run for.
                log.warning(
                    "hierarchy mismatch at step %d (%s); "
                    "forcing rebuild and revalidating",
                    step_i,
                    mismatch,
                )
                hierarchy.rebuild(params, step=step_i)
                active_idx = hierarchy.query(params, t)
                mismatch = self._validate_temporal_hierarchy(
                    params=params,
                    t=t,
                    config=config,
                    step=step_i,
                    hierarchy_idx=active_idx,
                )
                if mismatch is not None:
                    raise RuntimeError(
                        "Dynamic temporal hierarchy mismatch persists after "
                        f"a fresh rebuild at step={step_i}: {mismatch}"
                    )
            self.temporal_hierarchy_last_validation_step = step_i

        return active_idx
    
    @torch.no_grad()
    def _validate_temporal_hierarchy(
        self,
        *,
        params: nn.ParameterDict,
        t: float | Tensor,
        config: TrigsTrainingConfig,
        step: int,
        hierarchy_idx: Tensor | None = None,
    ) -> Optional[str]:
        """Compare hierarchy vs direct query; returns a mismatch description
        or None. Raising is the caller's decision: a mismatch caused by a
        missed staleness notification is cured by a rebuild, so only a
        mismatch that survives a rebuild should abort the run."""
        from .temporal_hierarchy import direct_temporal_active_idx

        if self.temporal_hierarchy is None:
            return

        direct_idx = direct_temporal_active_idx(
            params,
            t,
            opacity_threshold=self._effective_temporal_cull_threshold(config),
            extra_margin_s=config.temporal_active_extra_margin_s,
            include_static=True,
        )

        if hierarchy_idx is None:
            hierarchy_idx = self.temporal_hierarchy.query(
                params,
                t,
            )

        n = int(params["means"].shape[0])
        device = params["means"].device

        direct_mask = torch.zeros(
            n,
            device=device,
            dtype=torch.bool,
        )
        hierarchy_mask = torch.zeros_like(direct_mask)

        direct_mask[direct_idx] = True
        hierarchy_mask[hierarchy_idx] = True

        missing = int(
            (direct_mask & ~hierarchy_mask).sum().item()
        )
        extra = int(
            (hierarchy_mask & ~direct_mask).sum().item()
        )

        log.debug(
            "hierarchy validation @ step %d: "
            "direct=%d hierarchy=%d missing=%d extra=%d "
            "candidates=%d dirty=%d",
            step,
            int(direct_idx.numel()),
            int(hierarchy_idx.numel()),
            missing,
            extra,
            self.temporal_hierarchy.last_candidate_count,
            self.temporal_hierarchy.dirty_count,
        )

        if missing != 0 or extra != 0:
            return f"missing={missing}, extra={extra}"
        return None
        
    @torch.no_grad()
    def _update_temporal_hierarchy_drift(
        self,
        *,
        params: nn.ParameterDict,
        changed_idx: Tensor,
        config: TrigsTrainingConfig,
        step: int,
    ) -> None:
        hierarchy = self.temporal_hierarchy

        if (
            hierarchy is None
            or not hierarchy.valid
            or not config.temporal_hierarchy_enabled
            or step < int(config.unfreeze_temporal_step)
            or int(changed_idx.numel()) == 0
        ):
            return

        hierarchy.update_changed_rows(
            params,
            changed_idx,
        )

    @staticmethod
    def _camera_time_offset_scale(
        params: Mapping[str, Tensor], n_cameras: int,
    ) -> Tensor:
        """Per-camera multiplier in [0, 1] on the learned time offset.

        Set once by the CLI at params["camera_time_offset_scale"] from
        observability.json (see prep/sync/measure-offset-observability.py).
        A camera with no measurable photometric temporal gradient gets a
        near-zero scale, so its effective offset (scale * raw_param) stays
        near the ladder/sync.json prior no matter how far Adam's per-
        parameter step norm carries the underlying raw parameter. Missing
        or absent -> all-ones, i.e. today's unscaled behavior.

        staticmethod (not self.device): called from _post_step_safety_clamp,
        which is itself a staticmethod with no trainer instance in scope.
        Device is taken from an existing params tensor so it always matches
        what the caller will multiply this against.
        """
        reference = params.get("camera_time_offsets")
        device = reference.device if reference is not None else "cpu"
        scale = params.get("camera_time_offset_scale")
        if scale is None:
            return torch.ones(n_cameras, device=device)
        return scale.to(device=device, dtype=torch.float32)

    def _rigid_time_weight_value(
        self,
        config: TrigsTrainingConfig,
        scene_extent: float,
        n_frames: int,
    ) -> float:
        if config.rigid_knn_time_weight <= 0:
            return 0.0
        total_sec = max(n_frames - 1, 1) / max(config.fps, 1e-6)
        return float(config.rigid_knn_time_weight) * float(scene_extent) / max(total_sec, 1e-6)

    @staticmethod
    def _add_relocation_lineage(params: nn.ParameterDict, *, donor_count: bool) -> None:
        """Add the non-trainable per-row relocation lineage counters (zeros)."""
        names = ["relocation_generation"] + (["relocation_donor_count"] if donor_count else [])
        for name in names:
            if name not in params:
                params[name] = nn.Parameter(
                    torch.zeros(
                        int(params["means"].shape[0]), 1,
                        device=params["means"].device,
                        dtype=torch.float32,
                    ),
                    requires_grad=False,
                )

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
        n_frames = len(ply_files)
        if n_frames < 1:
            raise ValueError("Need at least 1 frame to train")

        # Sequence time bounds, used by the temporal hierarchy to clip
        # influence intervals to the queryable range.
        self._sequence_t_max = (
            float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
        )

        params, sh_degree, scene_extent = self._build_canonical_set(
            ply_files, config, transform_config, n_frames,
        )
        n_primitives = params["means"].shape[0]
        self._pin_static_background_rows(params, config, n_frames)

        if getattr(config, "learn_camera_time_offsets", False):
            params["camera_time_offsets"] = nn.Parameter(
                torch.zeros(len(cameras), device=self.device), requires_grad=True
            )
        if getattr(config, "learn_camera_appearance", False):
            params["cam_gain"] = nn.Parameter(torch.ones(len(cameras), 3, device=self.device))
            params["cam_bias"] = nn.Parameter(torch.zeros(len(cameras), 3, device=self.device))
        if getattr(config, "record_relocation_lineage", False):
            self._add_relocation_lineage(params, donor_count=True)

        if config.verbose:
            print(
                f"TrigsTrainer: {n_primitives:,} canonical primitives, "
                f"{n_frames} frames, stride={config.keyframe_stride}, "
                f"scene_extent={scene_extent:.3f}"
            )

        if gt_images_per_frame is None or gt_alphas_per_frame is None:
            gt_images_per_frame, gt_alphas_per_frame = self._render_gt(
                ply_files, cameras, transform_config, sh_degree,
                verbose=config.verbose,
            )

        optimizers = self._build_optimizers(params, config, scene_extent)
        rigid_time_weight = self._rigid_time_weight_value(config, scene_extent, n_frames)
        nn_indices = self._compute_knn(
            self._rigid_knn_points(params, config).detach(),
            config.knn_k,
            times=params["times"].detach(),
            time_weight=rigid_time_weight,
        )

        if config.sigma_floor_target > 0:
            sigma_floor_target = config.sigma_floor_target
        else:
            sigma_floor_target = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)
        if config.verbose:
            print(
                f"  σ_t floor target = {sigma_floor_target:.4f} s "
                f"(λ_sigma_floor = {config.lambda_sigma_floor})"
            )

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
        frame_indices: Optional[List[int]] = None,
        initial_optimizer_state: Optional[Dict[str, Dict[str, Tensor]]] = None,
        initial_optimizer_step: int = 0,
        return_optimizer_state: bool = False,
        frame_weights: Optional[Sequence[float]] = None,
        progress_bar: Optional[Any] = None,
    ) -> TrigsTrainingResult:
        """``frame_weights`` (aligned with ``frame_indices``) sets the frame choice of every
        step directly; chunked training uses it to carry the fast-frame top-up into chunk calls."""
        n_primitives = params["means"].shape[0]

        # Sequence time bounds for temporal-hierarchy interval clipping.
        # This entry point is used by the COLMAP pipeline; train() sets the
        # same attribute for the PLY pipeline.
        self._sequence_t_max = (
            float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
        )

        if getattr(config, "learn_camera_time_offsets", False):
            if "camera_time_offsets" not in params:
                params["camera_time_offsets"] = nn.Parameter(
                    torch.zeros(len(cameras), device=self.device), requires_grad=True
                )
        if getattr(config, "learn_camera_appearance", False):
            if "cam_gain" not in params:
                params["cam_gain"] = nn.Parameter(torch.ones(len(cameras), 3, device=self.device))
            if "cam_bias" not in params:
                params["cam_bias"] = nn.Parameter(torch.zeros(len(cameras), 3, device=self.device))
        if getattr(config, "record_relocation_lineage", False):
            self._add_relocation_lineage(params, donor_count=True)
        if config.verbose:
            print(
                f"TrigsTrainer: {n_primitives:,} canonical primitives (pre-built), "
                f"{n_frames} frames, scene_extent={scene_extent:.3f}"
            )

        optimizers = self._build_optimizers(params, config, scene_extent)
        if initial_optimizer_state:
            load_param_state(
                optimizers, params, initial_optimizer_state, initial_optimizer_step,
            )
        self._pin_static_background_rows(params, config, n_frames)
        rigid_time_weight = self._rigid_time_weight_value(config, scene_extent, n_frames)
        nn_indices = self._compute_knn(
            self._rigid_knn_points(params, config).detach(),
            config.knn_k,
            times=params["times"].detach(),
            time_weight=rigid_time_weight,
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
            frame_indices=frame_indices,
            frame_weights_override=frame_weights,
            progress_bar=progress_bar,
        )
        result = self._package_result(params, metrics)
        if return_optimizer_state:
            result.optimizer_state = export_param_state(optimizers, params)
        return result

    def train_chunked(
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
        result_device: str | torch.device | None = None,
    ) -> TrigsTrainingResult:
        """Temporally chunked training with seam-safe ownership.

        Dynamic rows are assigned to exactly one owner chunk for an entire
        round.  The owner chunk renders enough neighboring frames to supervise
        the current temporal support of those rows, while adjacent chunks keep
        them only as frozen rendering context.  Only owner rows are scattered
        back into the CPU master, so overlap no longer behaves as
        last-writer-wins parameter stitching.
        """
        import dataclasses

        from .temporal_chunks import (
            ChunkResidency,
            OffsetTBWriter,
            allocate_integer_steps,
            chunk_row_idx,
            chunk_windows,
            temporal_radius_scale,
            window_leak_stats,
        )

        if float(getattr(config, "temporal_chunk_len_s", 0.0)) <= 0.0:
            raise ValueError(
                "train_chunked requires config.temporal_chunk_len_s > 0"
            )
        delayed_target = getattr(config, "delayed_prune_target_count", None)
        if delayed_target is not None and int(delayed_target) > 0:
            raise ValueError(
                "Chunked training preserves row identity within a chunk; "
                "disable delayed pruning (delayed_prune_target_count)."
            )
        if int(getattr(config, "train_seed", -1)) >= 0:
            # _train_loop seeds at its start, and every chunk call runs its
            # own _train_loop, so each chunk would replay the same stream.
            raise ValueError(
                "train_seed is supported only for unchunked training; "
                "leave it at -1 with temporal_chunk_len_s > 0."
            )
        if float(getattr(config, "frame_sampling_fast_frac", 0.0)) > 0.0:
            raise ValueError(
                "frame_sampling_fast_frac is supported only for unchunked training; "
                "leave it at 0 with temporal_chunk_len_s > 0."
            )

        t_max = float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
        self._sequence_t_max = t_max
        fps = max(float(config.fps), 1e-6)
        cull_threshold = self._effective_temporal_cull_threshold(config)
        radius_scale = temporal_radius_scale(cull_threshold)

        windows = chunk_windows(
            t_max,
            float(config.temporal_chunk_len_s),
            float(getattr(config, "temporal_chunk_overlap_s", 0.0)),
        )
        # Fast-frame top-up over the whole sequence, as in unchunked training: every frame keeps
        # its normal visits and frames with short-lived rows are topped up toward
        # frame_sampling_fast_visits. Frames in a window overlap are trained by two windows, so
        # normal visits follow how many windows' supervised spans contain each frame. The extra
        # steps join the run's budget (before rounds/cycles are derived from it); chunk calls get
        # their share by frame weight and draw their frames with the same weights.
        fast_weights: Optional[List[float]] = None
        fast_mask_global: Optional[Tensor] = None
        fast_visits = float(getattr(config, "frame_sampling_fast_visits", 0.0))
        if fast_visits > 0.0 and str(getattr(config, "camera_sampling", "epoch")).lower() != "official":
            log.warning("frame_sampling_fast_visits needs camera_sampling='official'; fast frames get no top-up")
            fast_visits = 0.0
        if fast_visits > 0.0 and n_frames > 0:
            pad = max(float(getattr(config, "temporal_chunk_supervision_pad_s", 0.0)), 0.0)
            pad_cap = float(getattr(config, "temporal_chunk_supervision_pad_max_s", 0.0))
            if pad_cap > 0.0:
                pad = min(pad, pad_cap)
            if float(getattr(config, "temporal_chunk_supervision_pad_s", 0.0)) < 0.0:
                log.warning("automatic supervision padding changes chunk spans every round; the fast-frame "
                            "top-up assumes the nominal windows")
            mult = [0.0] * n_frames
            for t_lo_w, t_hi_w in windows:
                lo, hi = max(0.0, t_lo_w - pad), min(t_max, t_hi_w + pad)
                for fi in range(n_frames):
                    if lo - 1e-9 <= fi / fps <= hi + 1e-9:
                        mult[fi] += 1.0
            counts = short_lived_rows_per_frame(
                params, range(n_frames), config.fps, config.frame_sampling_fast_sigma_frames,
            )
            base_steps = int(config.n_steps)
            visits, extra_steps = fast_frame_topup(
                counts, base_steps, fast_visits, int(config.frame_sampling_fast_min_rows), multiplicity=mult,
            )
            mult_total = sum(mult) or 1.0
            normal = [base_steps * m / mult_total for m in mult]
            if extra_steps > 0:
                fast_weights = [v / b if b > 0.0 else 1.0 for v, b in zip(visits, normal)]
                fast_mask_global = torch.tensor([w > 1.0 + 1e-9 for w in fast_weights], dtype=torch.bool)
                config = dataclasses.replace(config, n_steps=base_steps + extra_steps)
                if config.verbose:
                    print(f"  Fast frames: {int(fast_mask_global.sum())}/{n_frames} topped up toward "
                          f"{fast_visits:g} visits (normal {sorted(normal)[len(normal) // 2]:.0f}); +{extra_steps:,} steps -> "
                          f"n_steps {base_steps:,} -> {config.n_steps:,}")
            elif config.verbose:
                print(f"  Fast frames: none need a top-up to {fast_visits:g} visits; n_steps unchanged")
        # Chunk calls never run their own top-up: their frame weights come from here.
        config = dataclasses.replace(config, frame_sampling_fast_visits=0.0)

        rounds = max(int(getattr(config, "temporal_chunk_rounds", 1)), 1)
        n_windows = len(windows)
        visit_steps = int(getattr(config, "temporal_chunk_visit_steps", 0))
        interleaved = visit_steps > 0
        if interleaved:
            rounds = max(
                1, int(round(int(config.n_steps) / float(visit_steps * n_windows)))
            )
        visit_order_rng = random.Random(int(getattr(config, "psnr_eval_seed", 12345)))

        # Midpoints of the actual pairwise overlaps partition frame time.
        ownership_bounds: List[float] = []
        for (a_lo, a_hi), (b_lo, b_hi) in zip(windows, windows[1:]):
            overlap_lo = max(a_lo, b_lo)
            overlap_hi = min(a_hi, b_hi)
            if overlap_hi < overlap_lo:
                # chunk_windows should not create gaps, but use the midpoint
                # between windows as a defensive ownership split.
                ownership_bounds.append(0.5 * (a_hi + b_lo))
            else:
                ownership_bounds.append(0.5 * (overlap_lo + overlap_hi))
        ownership_intervals: List[tuple[float, float]] = []
        for w_i in range(n_windows):
            own_lo = ownership_bounds[w_i - 1] if w_i > 0 else 0.0
            own_hi = ownership_bounds[w_i] if w_i < n_windows - 1 else t_max + 0.5 / fps
            ownership_intervals.append((float(own_lo), float(own_hi)))

        # Non-row parameters must exist before ChunkResidency is created;
        # otherwise train_from_params() would recreate and then discard them
        # at every chunk boundary.
        if getattr(config, "learn_camera_time_offsets", False):
            if "camera_time_offsets" not in params:
                params["camera_time_offsets"] = nn.Parameter(
                    torch.zeros(len(cameras), device=self.device),
                    requires_grad=True,
                )
        if getattr(config, "learn_camera_appearance", False):
            if "cam_gain" not in params:
                params["cam_gain"] = nn.Parameter(
                    torch.ones(len(cameras), 3, device=self.device),
                    requires_grad=True,
                )
            if "cam_bias" not in params:
                params["cam_bias"] = nn.Parameter(
                    torch.zeros(len(cameras), 3, device=self.device),
                    requires_grad=True,
                )

        # Track whether entry cohorts were created by relocation.  This is
        # non-trainable row metadata and is preserved by chunk residency.
        self._add_relocation_lineage(
            params,
            donor_count=bool(getattr(config, "record_relocation_lineage", False)),
        )

        residency = ChunkResidency.from_params(params)
        for name in list(params.keys()):
            del params[name]
        # Capacity watch (top-up runs only), on the CPU master after every round.
        capacity_start: Optional[Tensor] = None
        if fast_mask_global is not None and bool(fast_mask_global.any()) and bool((~fast_mask_global).any()):
            capacity_start = alive_rows_per_frame(
                residency.master, n_frames, config.fps, config.relocate_tau_alpha,
            ).double()
        empty_cache()

        freeze_nonowned = bool(
            getattr(config, "temporal_chunk_freeze_nonowned", True)
        )
        train_static_mode = str(
            getattr(config, "temporal_chunk_train_static", "first_round")
        ).lower()
        if interleaved:
            train_static_mode = "all"
        relocate_final = bool(
            getattr(config, "temporal_chunk_relocate_final_round", False)
        )
        if not freeze_nonowned:
            log.warning(
                "temporal_chunk_freeze_nonowned=False restores overlapping "
                "multi-writer updates and defeats the seam-safe scatter path"
            )
        if relocate_final:
            log.warning(
                "temporal_chunk_relocate_final_round=True is unsafe for seam "
                "diagnosis: final chunks may end with newly relocated rows"
            )
        if (
            bool(config.relocate)
            and float(getattr(config, "relocate_time_jitter_sigma", 0.0)) <= 0.0
            and bool(getattr(config, "temporal_chunk_entry_diagnostics", False))
        ):
            # Exact mu_t copies are the default (official TRiGS); this only matters to the
            # opt-in entry diagnostics, so it is not a warning.
            log.info(
                "relocation copies mu_t exactly (relocate_time_jitter_sigma=0); "
                "entry diagnostics will report whether cloned support cohorts "
                "remain a visible amplifier"
            )

        static_master = residency.master.get("is_static")
        if static_master is None:
            static_master_mask = torch.zeros(residency.n_rows, dtype=torch.bool)
        else:
            static_master_mask = static_master.detach().reshape(-1) > 0.5
        dynamic_master_mask = ~static_master_mask

        with torch.no_grad():
            sigma_dyn = (
                residency.master["log_durations"].detach().exp().reshape(-1)[
                    dynamic_master_mask
                ].float()
            )
            if int(sigma_dyn.numel()) > 0:
                radius_dyn = sigma_dyn * radius_scale + float(
                    config.temporal_active_extra_margin_s
                )
                radius_quantile_sample = _bounded_diagnostic_sample(radius_dyn)
                q = torch.tensor([0.5, 0.9, 0.99], dtype=torch.float32)
                rq = torch.quantile(radius_quantile_sample, q).tolist()
                overlap = float(getattr(config, "temporal_chunk_overlap_s", 0.0))
                log.debug(
                    "chunk dynamic cull-radius @ threshold %.4g: "
                    "p50=%.4fs p90=%.4fs p99=%.4fs max=%.4fs; overlap=%.4fs",
                    cull_threshold, rq[0], rq[1], rq[2],
                    float(radius_dyn.max().item()), overlap,
                )

        if config.verbose:
            print(
                f"Chunked training: {n_windows} windows x {rounds} rounds "
                f"(cull threshold={cull_threshold:g}"
                + (f", interleaved visits of ~{visit_steps} steps)" if interleaved else ")")
            )
            print(
                "  windows="
                f"{[(round(a, 3), round(b, 3)) for a, b in windows]}"
            )
            print(
                "  ownership="
                f"{[(round(a, 3), round(min(b, t_max), 3)) for a, b in ownership_intervals]}"
            )

        metrics: Dict[str, Any] = {
            "n_initial": residency.n_rows,
            "chunks": [],
            "snapshots": [],
            "prune_events": [],
            "windows": [[float(a), float(b)] for a, b in windows],
            "ownership_bounds": [float(x) for x in ownership_bounds],
            "entry_diagnostics": [],
        }

        if bool(getattr(config, "temporal_chunk_entry_diagnostics", True)):
            diag = self._temporal_entry_diagnostics(
                residency.master,
                config=config,
                n_frames=n_frames,
                windows=windows,
                ownership_bounds=ownership_bounds,
                label="initial",
            )
            metrics["entry_diagnostics"].append(diag)

        round_budgets = allocate_integer_steps(
            int(config.n_steps), [1.0] * rounds, min_each=1,
        )
        global_step_offset = 0
        last_eval_step = 0
        viewer_dir = self._viewer_snapshot_dir(config)
        last_viewer_step = 0
        if viewer_dir is not None and config.verbose:
            print(f"  Viewer snapshots: {viewer_dir} every {config.viewer_snapshot_every} steps")
        # Interleaved visits carry Adam moments with the rows (CPU, like the
        # parameters). Adam scales each step by recently seen gradient sizes;
        # a fresh optimizer per visit would rescale a long-lived row's weak
        # gradient from one window to full-size steps for the whole visit,
        # while un-chunked training keeps the scale set by its strong
        # gradients from other windows.
        opt_master: Dict[str, Dict[str, Tensor]] = {}
        call_index = 0
        # Global step at which each round begins. Every window in a round is
        # mapped onto that round's slice of the LR schedule so a window's
        # position in the time-ordered chunk loop cannot decide how much LR
        # its authoritative write receives.
        round_base_offsets = [
            sum(round_budgets[:i]) for i in range(rounds)
        ]

        # One progress bar for the whole run (global step, total, ETA); its label names the
        # chunk in training and each chunk's start line is written above it, so both persist.
        pbar = tqdm(total=int(config.n_steps), desc="TRiGS", dynamic_ncols=True) if config.verbose else None
        say = pbar.write if pbar is not None else print

        for round_i in range(rounds):
            # Freeze ownership for the whole round.  Every dynamic row has one
            # and only one writer even if mu_t crosses a boundary mid-round;
            # ownership is reconsidered at the next round.
            round_mu = residency.master["times"].detach().reshape(-1).clone()
            if ownership_bounds:
                bounds_t = torch.tensor(
                    ownership_bounds, dtype=round_mu.dtype, device=round_mu.device,
                )
                owner_id = torch.bucketize(round_mu, bounds_t, right=True)
            else:
                owner_id = torch.zeros_like(round_mu, dtype=torch.long)

            plans: List[Dict[str, Any]] = []
            pad_cfg = float(
                getattr(config, "temporal_chunk_supervision_pad_s", -1.0)
            )
            pad_cap = float(
                getattr(config, "temporal_chunk_supervision_pad_max_s", 0.0)
            )
            master_sigma = (
                residency.master["log_durations"].detach().exp().reshape(-1)
                .clamp_min(1e-6)
            )
            master_radius = (
                master_sigma * radius_scale
                + float(config.temporal_active_extra_margin_s)
            )

            for w_i, (t_lo, t_hi) in enumerate(windows):
                owner_dynamic = (
                    (owner_id == w_i) & dynamic_master_mask
                )
                sup_lo, sup_hi = float(t_lo), float(t_hi)

                if pad_cfg < 0.0 and bool(owner_dynamic.any()):
                    owner_mu = round_mu[owner_dynamic]
                    owner_r = master_radius[owner_dynamic]
                    sup_lo = min(sup_lo, float((owner_mu - owner_r).min().item()))
                    sup_hi = max(sup_hi, float((owner_mu + owner_r).max().item()))
                elif pad_cfg > 0.0:
                    sup_lo -= pad_cfg
                    sup_hi += pad_cfg

                if pad_cap > 0.0:
                    sup_lo = max(sup_lo, float(t_lo) - pad_cap)
                    sup_hi = min(sup_hi, float(t_hi) + pad_cap)

                sup_lo = max(0.0, sup_lo)
                sup_hi = min(t_max, sup_hi)
                chunk_frames = [
                    fi for fi in range(n_frames)
                    if sup_lo - 1e-9 <= fi / fps <= sup_hi + 1e-9
                ]
                plans.append({
                    "window_index": w_i,
                    "window": (float(t_lo), float(t_hi)),
                    "supervision": (float(sup_lo), float(sup_hi)),
                    "owner_dynamic": owner_dynamic,
                    "frames": chunk_frames,
                })

            call_steps = allocate_integer_steps(
                round_budgets[round_i],
                [
                    max(len(plan["frames"]), 1) if fast_weights is None
                    else max(sum(fast_weights[fi] for fi in plan["frames"]), 1e-9)
                    for plan in plans
                ],
                min_each=1,
            )

            visits = list(zip(plans, call_steps))
            if interleaved:
                # Every window in a round shares one LR band, so order only
                # decides who writes a shared row last; shuffle it per cycle.
                visit_order_rng.shuffle(visits)
            for plan, steps_this_call in visits:
                w_i = int(plan["window_index"])
                t_lo, t_hi = plan["window"]
                sup_lo, sup_hi = plan["supervision"]
                owner_dynamic_master = plan["owner_dynamic"]
                chunk_frames = plan["frames"]

                idx = chunk_row_idx(
                    residency.master["times"],
                    residency.master["log_durations"],
                    sup_lo,
                    sup_hi,
                    opacity_threshold=cull_threshold,
                    extra_margin_s=config.temporal_active_extra_margin_s,
                    is_static=residency.master.get("is_static"),
                    include_static=True,
                )
                if int(idx.numel()) == 0 or not chunk_frames or steps_this_call <= 0:
                    global_step_offset += int(steps_this_call)
                    call_index += 1
                    if pbar is not None:
                        pbar.update(int(max(steps_this_call, 0)))
                    continue

                sub_params = residency.materialize(idx, self.device)
                owner_local = owner_dynamic_master[idx].to(
                    device=self.device, dtype=torch.bool,
                )
                reloc_owned = owner_local.clone()

                # TRAINABILITY (interval-based, restores pre-th-04 semantics):
                # a row is trainable in every chunk whose owned frame span its
                # influence interval intersects.  With supervision_pad = 0
                # this guarantees every rendered (row, frame) contribution is
                # optimizable by the chunk that owns that frame — mu-only
                # ownership (th-04) left support tails crossing an ownership
                # boundary permanently unsupervised, the original "particle
                # shedding" failure.  Boundary-crossing rows are therefore
                # written by more than one chunk per round (later write wins
                # in absorb); the outside-mass loss bounds how much support
                # the final writer can park beyond its own supervised window.
                # RELOCATION ownership stays mu-partitioned (reloc_owned):
                # recycling must remain unique per row within a round.
                own_lo, own_hi = ownership_intervals[w_i]
                mu_sub = sub_params["times"].detach().reshape(-1)
                sigma_sub = (
                    sub_params["log_durations"].detach().exp().reshape(-1)
                    .clamp_min(1e-6)
                )
                radius_sub = sigma_sub * radius_scale + float(
                    config.temporal_active_extra_margin_s
                )
                trainable = (
                    (mu_sub + radius_sub >= float(own_lo))
                    & (mu_sub - radius_sub <= float(own_hi))
                )

                if "is_static" in sub_params:
                    static_sub = (
                        sub_params["is_static"].detach().reshape(-1) > 0.5
                    )
                    if train_static_mode == "all":
                        train_static = static_sub
                    elif train_static_mode == "first_round":
                        train_static = static_sub if round_i == 0 else torch.zeros_like(static_sub)
                    else:  # "first"
                        train_static = static_sub if call_index == 0 else torch.zeros_like(static_sub)
                    # Static rows have sigma ~ t_max, so the interval test
                    # always selects them; strip them first, then re-add only
                    # when the static schedule says so.
                    trainable = (trainable & (~static_sub)) | train_static
                    reloc_owned = reloc_owned & (~static_sub)

                # chunk_owner_mask is always installed for diagnostics/loss.
                sub_params["chunk_owner_mask"] = nn.Parameter(
                    owner_local.to(torch.float32).reshape(-1, 1),
                    requires_grad=False,
                )
                sub_params["chunk_reloc_mask"] = nn.Parameter(
                    reloc_owned.to(torch.float32).reshape(-1, 1),
                    requires_grad=False,
                )
                if freeze_nonowned:
                    sub_params["chunk_train_mask"] = nn.Parameter(
                        trainable.to(torch.float32).reshape(-1, 1),
                        requires_grad=False,
                    )

                for name, value in {
                    "chunk_supervision_lo": sup_lo,
                    "chunk_supervision_hi": sup_hi,
                    "chunk_sequence_t_max": t_max,
                    "chunk_cull_threshold": cull_threshold,
                    "chunk_total_rows": residency.n_rows,
                }.items():
                    sub_params[name] = nn.Parameter(
                        torch.tensor(float(value), device=self.device),
                        requires_grad=False,
                    )

                self.temporal_hierarchy = None
                self._invalidate_temporal_hierarchy()

                is_final_round = round_i == rounds - 1
                settle = int(getattr(config, "relocate_settle_steps", 0))
                if settle <= 0:
                    settle = max(4 * int(config.relocate_every), 1000)
                visit_overrides: Dict[str, Any] = {}
                if interleaved:
                    # Relocation runs on the whole-run schedule: its start is
                    # a global step (first pass after one relocate_every of
                    # cue accumulation inside the visit), its stop is
                    # relocate_stop_frac of the global schedule, and relocated
                    # rows settle in later visits rather than an end-of-call
                    # window (which would disable relocation in short visits).
                    every = max(int(config.relocate_every), 1)
                    settle = 0
                    visit_overrides = {
                        "relocate": bool(config.relocate),
                        "relocate_start_iter": max(
                            every,
                            int(config.relocate_start_iter) - int(global_step_offset),
                        ),
                        "lambda_temporal_chunk_outside_mass": (
                            float(config.lambda_temporal_chunk_outside_mass)
                            if bool(getattr(config, "temporal_chunk_visit_outside_mass", False))
                            else 0.0
                        ),
                    }

                if bool(getattr(config, "unfreeze_temporal_per_chunk", False)):
                    # Every chunk call gets its own motion-only phase, capped
                    # at half the call's steps so temporal params always train.
                    unfreeze_this_call = min(
                        int(config.unfreeze_temporal_step),
                        max(int(steps_this_call) // 2, 0),
                    )
                else:
                    unfreeze_this_call = max(
                        int(config.unfreeze_temporal_step)
                        - int(global_step_offset),
                        0,
                    )

                chunk_config = dataclasses.replace(
                    config,
                    n_steps=int(steps_this_call),
                    unfreeze_temporal_step=unfreeze_this_call,
                    camera_time_offset_warmup_steps=max(
                        int(config.camera_time_offset_warmup_steps)
                        - int(global_step_offset),
                        0,
                    ),
                    lr_schedule_offset_steps=int(round_base_offsets[round_i]),
                    lr_schedule_total_steps=int(config.n_steps),
                    lr_schedule_step_scale=(
                        float(round_budgets[round_i]) / float(steps_this_call)
                    ),
                    sh_warmup_step_offset=int(global_step_offset),
                    relocate_settle_steps=settle,
                    relocate=(config.relocate and (
                        relocate_final or not is_final_round
                    )),
                    psnr_eval_every=0,
                    checkpoint_dir=None,
                    # A chunk call only holds part of the model; snapshots
                    # are written from the master between calls instead.
                    viewer_snapshot_every=0,
                )
                if visit_overrides:
                    chunk_config = dataclasses.replace(chunk_config, **visit_overrides)
                if call_index > 0:
                    # Every call would repeat the same setup lines; the run's bar shows progress.
                    chunk_config = dataclasses.replace(chunk_config, verbose=False)

                chunk_writer = writer
                if writer is not None:
                    chunk_writer = OffsetTBWriter(writer, global_step_offset)

                if pbar is not None:
                    pbar.set_description(
                        f"TRiGS {'cycle' if interleaved else 'round'} {round_i + 1}/{rounds} "
                        f"window {w_i + 1}/{n_windows} [{t_lo:.2f},{t_hi:.2f}]s "
                        f"rows {int(idx.numel()):,}"
                    )
                if config.verbose:
                    say(
                        f"  Chunk {'cycle' if interleaved else 'round'} {round_i + 1}/{rounds} "
                        f"window {w_i + 1}/{n_windows} "
                        f"base=[{t_lo:.3f},{t_hi:.3f}]s "
                        f"supervised=[{sup_lo:.3f},{sup_hi:.3f}]s: "
                        f"owner_dyn={int(owner_local.sum().item()):,}, "
                        f"resident={int(idx.numel()):,}/{residency.n_rows:,}, "
                        f"frames={len(chunk_frames)}, steps={steps_this_call} "
                        f"(global {global_step_offset:,} -> {global_step_offset + int(steps_this_call):,} "
                        f"of {int(config.n_steps):,})"
                    )

                visit_state: Dict[str, Dict[str, Tensor]] = {}
                if interleaved and opt_master:
                    idx_cpu = idx.detach().reshape(-1).to("cpu", dtype=torch.long)
                    keep = (
                        trainable.detach().reshape(-1).to("cpu")
                        if freeze_nonowned else torch.ones(int(idx_cpu.numel()), dtype=torch.bool)
                    )
                    for name, moments in opt_master.items():
                        if residency.row_aligned.get(name, False):
                            # Frozen context rows start from zero moments so
                            # they cannot coast on momentum during the visit.
                            visit_state[name] = {
                                key: value.index_select(0, idx_cpu)
                                * keep.reshape(-1, *([1] * (value.dim() - 1))).to(value.dtype)
                                for key, value in moments.items()
                            }
                        else:
                            visit_state[name] = moments

                result = self.train_from_params(
                    params=sub_params,
                    sh_degree=sh_degree,
                    scene_extent=scene_extent,
                    cameras=cameras,
                    n_frames=n_frames,
                    gt_images_per_frame=gt_images_per_frame,
                    gt_alphas_per_frame=gt_alphas_per_frame,
                    config=chunk_config,
                    gt_times_per_frame=gt_times_per_frame,
                    writer=chunk_writer,
                    frame_indices=chunk_frames,
                    initial_optimizer_state=visit_state or None,
                    initial_optimizer_step=int(global_step_offset),
                    return_optimizer_state=interleaved,
                    progress_bar=pbar,
                    frame_weights=(
                        None if fast_weights is None
                        or all(abs(fast_weights[fi] - 1.0) < 1e-9 for fi in chunk_frames)
                        else [fast_weights[fi] for fi in chunk_frames]
                    ),
                )

                if freeze_nonowned and "chunk_train_mask" in result.params:
                    write_mask = (
                        result.params["chunk_train_mask"].detach().reshape(-1) > 0.5
                    )
                else:
                    write_mask = torch.ones(
                        int(idx.numel()), device=self.device, dtype=torch.bool,
                    )
                residency.absorb(
                    result.params, idx, row_write_mask=write_mask,
                )
                if interleaved:
                    wm = write_mask.detach().reshape(-1).to("cpu")
                    rows_written = idx.detach().reshape(-1).to("cpu", dtype=torch.long)[wm]
                    for name, moments in result.optimizer_state.items():
                        if residency.row_aligned.get(name, False):
                            store = opt_master.setdefault(name, {
                                key: torch.zeros(
                                    (residency.n_rows, *value.shape[1:]), dtype=value.dtype,
                                )
                                for key, value in moments.items()
                            })
                            for key, value in moments.items():
                                store[key][rows_written] = value[wm]
                        else:
                            opt_master[name] = moments
                metrics["chunks"].append({
                    "round": int(round_i),
                    "window": [float(t_lo), float(t_hi)],
                    "supervision_window": [float(sup_lo), float(sup_hi)],
                    "rows": int(idx.numel()),
                    "owner_dynamic_rows": int(owner_local.sum().item()),
                    "written_rows": int(write_mask.sum().item()),
                    "frames": len(chunk_frames),
                    "steps": int(steps_this_call),
                    "final_count": result.metrics.get("final_count"),
                })

                del sub_params, result
                self.temporal_hierarchy = None
                empty_cache()
                global_step_offset += int(steps_this_call)
                call_index += 1

                # The master now holds this call's result: a complete model.
                if viewer_dir is not None and should_snapshot(
                    global_step_offset, config.viewer_snapshot_every, last_viewer_step,
                ):
                    if self._write_viewer_snapshot(
                        out_dir=viewer_dir, params=residency.master,
                        step=global_step_offset, config=config, n_frames=n_frames,
                        sh_degree=sh_degree,
                        sh_degree_active=_sh_degree_for_step(config, global_step_offset, sh_degree),
                        writer=writer,
                        extra=self._chunked_snapshot_extra(
                            metrics, residency.n_rows, round_i, rounds, w_i, n_windows,
                            (t_lo, t_hi), (sup_lo, sup_hi),
                        ),
                    ):
                        last_viewer_step = global_step_offset

            # Chunk calls run with psnr_eval_every=0 (a chunk only holds part
            # of the sequence), so evaluate the assembled model at the first
            # round end past each psnr_eval_every global steps, and at the
            # end, on the same fixed pairs an unchunked run uses.
            eval_every = int(config.psnr_eval_every)
            if eval_every > 0 and (
                round_i == rounds - 1
                or global_step_offset // eval_every > last_eval_step // eval_every
            ):
                last_eval_step = global_step_offset
                self.temporal_hierarchy = None
                self._invalidate_temporal_hierarchy()
                round_eval = self._chunked_psnr_eval(
                    residency=residency,
                    ownership_intervals=ownership_intervals,
                    cameras=cameras,
                    n_frames=n_frames,
                    sh_degree=_sh_degree_for_step(config, global_step_offset, sh_degree),
                    config=config,
                    gt_images_per_frame=gt_images_per_frame,
                    gt_alphas_per_frame=gt_alphas_per_frame,
                    gt_times_per_frame=gt_times_per_frame,
                    cull_threshold=cull_threshold,
                    step=global_step_offset,
                )
                for name, values in round_eval.items():
                    if writer is not None:
                        for k, v in values.items():
                            writer.add_scalar(k, v, global_step_offset)
                    metrics.setdefault(f"psnr_{name}", []).append({
                        "step": int(global_step_offset),
                        "round": int(round_i),
                        **{k.replace("/", "_"): float(v) for k, v in values.items()},
                    })
                if config.verbose and round_eval:
                    summary = ", ".join(
                        f"{k}={v:.2f}"
                        for values in round_eval.values()
                        for k, v in values.items()
                        if k.startswith("psnr/") and "num_samples" not in k
                    )
                    say(f"  Chunk round {round_i + 1}/{rounds} eval @ step {global_step_offset}: {summary}")

            if capacity_start is not None and fast_mask_global is not None:
                self._log_capacity(residency.master, config, n_frames, fast_mask_global, capacity_start,
                                   writer, global_step_offset, metrics)

            # Window leak: live rows whose temporal support reaches past their owner window
            # (frames that window's visits never render). Compare with window_leak_stats on an
            # unchunked checkpoint (trigs.cli.chunk_leak) to see what chunking adds.
            leak = window_leak_stats(
                residency.master, windows, t_max, fps,
                cull_threshold=cull_threshold, alive_opacity=float(config.relocate_tau_alpha),
                extra_margin_s=float(config.temporal_active_extra_margin_s),
            )
            metrics.setdefault("window_leak", []).append({"step": int(global_step_offset), **leak})
            if writer is not None:
                for k in ("leak_frac", "mass_outside", "overshoot_p50_frames", "overshoot_p99_frames"):
                    writer.add_scalar(f"chunk/{k}", leak[k], global_step_offset)
            if config.verbose:
                say(f"  Window leak @ step {global_step_offset:,}: {leak['leak_frac']:.1%} of "
                    f"{int(leak['rows']):,} live rows reach past their window (overshoot p50 "
                    f"{leak['overshoot_p50_frames']:.0f} / p99 {leak['overshoot_p99_frames']:.0f} frames); "
                    f"{leak['mass_outside']:.1%} of their temporal mass is outside it")

            if bool(getattr(config, "temporal_chunk_entry_diagnostics", True)) and (
                not interleaved or round_i == rounds - 1
            ):
                diag = self._temporal_entry_diagnostics(
                    residency.master,
                    config=config,
                    n_frames=n_frames,
                    windows=windows,
                    ownership_bounds=ownership_bounds,
                    label=f"round_{round_i + 1}",
                )
                metrics["entry_diagnostics"].append(diag)

        if viewer_dir is not None and should_snapshot(
            global_step_offset, config.viewer_snapshot_every, last_viewer_step, final=True,
        ):
            self._write_viewer_snapshot(
                out_dir=viewer_dir, params=residency.master,
                step=global_step_offset, config=config, n_frames=n_frames,
                sh_degree=sh_degree,
                sh_degree_active=_sh_degree_for_step(config, global_step_offset, sh_degree),
                writer=writer,
                extra=self._chunked_snapshot_extra(
                    metrics, residency.n_rows, rounds - 1, rounds, None, n_windows,
                    None, None,
                ),
            )

        if pbar is not None:
            pbar.close()
        if config.verbose and metrics.get("capacity"):
            print(_capacity_summary(metrics["capacity"][-1]))

        full_idx = residency.full_index()
        # Preserve the historical/default contract for existing callers:
        # return on the trainer device unless a caller explicitly opts into
        # CPU assembly (the long PLY pipeline does this to avoid a final OOM).
        final_device = self.device if result_device is None else result_device
        full_params = residency.materialize(full_idx, final_device)
        metrics["final_count"] = int(full_params["means"].shape[0])
        self.temporal_hierarchy = None
        self._invalidate_temporal_hierarchy()
        return self._package_result(full_params, metrics)

    def _package_result(
        self,
        params: nn.ParameterDict,
        metrics: Dict[str, Any],
    ) -> TrigsTrainingResult:
        temporal_params = {
            "times": params["times"].detach().cpu(),
            "log_durations": params["log_durations"].detach().cpu(),
        }
        if "camera_time_offsets" in params:
            temporal_params["camera_time_offsets"] = params["camera_time_offsets"].detach().cpu()
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
        ply_init: "PlyMotionInitConfig | None" = None,
    ) -> tuple[nn.ParameterDict, int, float]:
        from ..init.from_plys import (
            allocate_keyframe_quotas,
            coverage_keep_mask,
            motion_adaptive_sigma,
            select_keyframe_gaussians,
        )
        from ..io.ply import load_ply

        keyframe_indices = list(range(0, n_frames, max(config.keyframe_stride, 1)))
        keyframe_quotas: List[int] | None = None
        if config.target_count is not None and int(config.target_count) > 0:
            keyframe_quotas = allocate_keyframe_quotas(
                int(config.target_count),
                len(keyframe_indices),
            )

        all_means, all_quats, all_scales = [], [], []
        all_opacities, all_sh0, all_shN = [], [], []
        all_times = []
        per_kf_t: List[float] = []

        has_shN = False
        original_row_count = 0
        for ki, kf_idx in enumerate(keyframe_indices):
            gs = load_ply(ply_files[kf_idx], device=str(self.device))
            gs = transform_config.apply(gs)
            has_shN = has_shN or (gs.shN is not None)

            original_n = len(gs)
            original_row_count += original_n
            keep: Tensor | slice = slice(None)
            if keyframe_quotas is not None and original_n > keyframe_quotas[ki]:
                keep = select_keyframe_gaussians(
                    gs.means,
                    keyframe_quotas[ki],
                    seed=12_345 + int(kf_idx),
                )

            n = (
                original_n
                if isinstance(keep, slice)
                else int(keep.numel())
            )
            t_sec = float(kf_idx) / max(config.fps, 1e-6)

            all_means.append(gs.means[keep])
            all_quats.append(gs.quats[keep])
            all_scales.append(gs.scales[keep])
            all_opacities.append(gs.opacities[keep])
            all_sh0.append(gs.sh0[keep])
            if gs.shN is not None:
                all_shN.append(gs.shN[keep])
            all_times.append(torch.full((n, 1), t_sec, device=self.device))
            per_kf_t.append(t_sec)
            del gs

        if keyframe_quotas is not None and config.verbose:
            retained = sum(int(m.shape[0]) for m in all_means)
            print(
                f"  Early temporal-stratified PLY selection: "
                f"{original_row_count:,} → {retained:,} primitives "
                f"across {len(keyframe_indices)} keyframes"
            )

        all_nu: List[Tensor] = []
        # Centered second difference. The constant-twist model is exact for
        # uniform motion, so speed does not bound a row's usable window --
        # acceleration does.
        all_accel: List[Tensor] = []
        if config.nn_velocity_init and len(all_means) >= 1:
            k_nn = max(1, config.nn_velocity_k)
            dt = 1.0 / max(config.fps, 1e-6)
            for ki in range(len(all_means)):
                src = all_means[ki]
                if src.shape[0] == 0:
                    all_nu.append(torch.zeros_like(src))
                    continue
                f_k = keyframe_indices[ki]
                v_fwd: Tensor | None = None
                v_bwd: Tensor | None = None
                if f_k + 1 < n_frames:
                    gs_next = load_ply(ply_files[f_k + 1], device=str(self.device))
                    gs_next = transform_config.apply(gs_next)
                    nn_next = self._knn_mean_positions(src, gs_next.means, k_nn)
                    v_fwd = (nn_next - src) / dt
                    del gs_next
                if f_k - 1 >= 0:
                    gs_prev = load_ply(ply_files[f_k - 1], device=str(self.device))
                    gs_prev = transform_config.apply(gs_prev)
                    nn_prev = self._knn_mean_positions(src, gs_prev.means, k_nn)
                    v_bwd = (src - nn_prev) / dt
                    del gs_prev
                if v_fwd is not None and v_bwd is not None:
                    all_nu.append((v_fwd + v_bwd) / 2.0)
                    all_accel.append((v_fwd - v_bwd) / dt)
                elif v_fwd is not None:
                    all_nu.append(v_fwd)
                    all_accel.append(torch.zeros_like(src))
                elif v_bwd is not None:
                    all_nu.append(v_bwd)
                    all_accel.append(torch.zeros_like(src))
                else:
                    all_nu.append(torch.zeros_like(src))
                    all_accel.append(torch.zeros_like(src))
            if config.verbose:
                mean_mag = torch.cat(all_nu).norm(dim=-1).mean().item()
                print(
                    f"  Centered FD ν init (K={k_nn}) on "
                    f"{sum(m.shape[0] for m in all_means):,} primitives: "
                    f"mean ‖ν‖ = {mean_mag:.4f} world/s"
                )
        else:
            all_nu = [torch.zeros_like(m) for m in all_means]
            all_accel = [torch.zeros_like(m) for m in all_means]

        # Motion-adaptive sigma + coverage dedup (PLY path only).  Runs on the
        # full finite-difference velocity, before the twist fit replaces
        # all_nu with the translation residual.  Thinning here also shrinks
        # the anchor/twist work below.
        per_row_sigma: Tensor | None = None
        if ply_init is not None and bool(getattr(ply_init, "enabled", False)):
            cohort_dt = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)
            sigma_floor = float(ply_init.sigma_floor_s) or 0.5 * cohort_dt
            sigma_ceiling = float(ply_init.sigma_ceiling_s) or 8.0 * cohort_dt
            sigma_keep: List[Tensor] = []
            n_before = sum(int(m.shape[0]) for m in all_means)
            for ki in range(len(all_means)):
                if all_means[ki].shape[0] == 0:
                    sigma_keep.append(all_means[ki].new_zeros((0,)))
                    continue
                # A raw centered second difference divides NN matching noise
                # by dt^2 (~900x at 30 fps), which would hand every row a
                # spurious acceleration and collapse sigma everywhere.
                # Averaging over the local neighborhood recovers the coherent
                # component: real acceleration is shared by neighboring rows
                # on the same limb, matching noise is not.
                accel_ki = all_accel[ki]
                if accel_ki.shape[0] > config.local_twist_k:
                    nn_idx = local_knn_indices(
                        all_means[ki], config.local_twist_k,
                    )
                    accel_ki = accel_ki[nn_idx].mean(dim=1)
                sigma_ki = motion_adaptive_sigma(
                    accel_ki.norm(dim=-1),
                    all_scales[ki].exp().mean(dim=-1),
                    sigma_floor=sigma_floor,
                    sigma_ceiling=sigma_ceiling,
                    travel_radii=ply_init.travel_radii,
                )
                keep = coverage_keep_mask(
                    sigma_ki,
                    cohort_dt=cohort_dt,
                    redundancy=ply_init.redundancy,
                    support_sigmas=ply_init.support_sigmas,
                    seed=int(ply_init.seed) + int(keyframe_indices[ki]),
                )
                all_means[ki] = all_means[ki][keep]
                all_quats[ki] = all_quats[ki][keep]
                all_scales[ki] = all_scales[ki][keep]
                all_opacities[ki] = all_opacities[ki][keep]
                all_sh0[ki] = all_sh0[ki][keep]
                all_times[ki] = all_times[ki][keep]
                all_nu[ki] = all_nu[ki][keep]
                if all_shN:
                    all_shN[ki] = all_shN[ki][keep]
                sigma_keep.append(sigma_ki[keep])
            per_row_sigma = torch.cat(sigma_keep).reshape(-1, 1)
            if config.verbose:
                n_after = int(per_row_sigma.shape[0])
                sig_dbg = _bounded_diagnostic_sample(per_row_sigma.reshape(-1))
                q = torch.tensor([0.5, 0.9], device=sig_dbg.device)
                sq = torch.quantile(sig_dbg, q)
                print(
                    f"  Motion-adaptive PLY selection: {n_before:,} → "
                    f"{n_after:,} primitives ({n_after / max(n_before, 1):.1%}), "
                    f"σ_t p50={sq[0].item():.4f}s p90={sq[1].item():.4f}s "
                    f"(floor={sigma_floor:.4f}s ceiling={sigma_ceiling:.4f}s)"
                )

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
                    means_ki, v_targets_ki, anchors_ki, nn_idx,
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

        N = means.shape[0]

        if per_row_sigma is not None:
            # Derived per row above; a single init_sigma_t cannot serve both a
            # stationary torso and a swinging foot.
            log_durations = per_row_sigma.clamp_min(1e-6).log()
        else:
            if config.init_sigma_t > 0:
                sigma_t_init = config.init_sigma_t
            else:
                sigma_t_init = max(config.keyframe_stride, 1) / max(config.fps, 1e-6)
            log_durations = torch.full(
                (N, 1), math.log(max(sigma_t_init, 1e-6)), device=self.device,
            )

        anchors = anchors_super.detach().clone()
        def _zeros3():
            return torch.zeros(N, 3, device=self.device)

        geometry_trainable = not config.freeze_geometry
        params = nn.ParameterDict({
            "means": nn.Parameter(means, requires_grad=geometry_trainable),
            "quats": nn.Parameter(quats, requires_grad=geometry_trainable),
            "scales": nn.Parameter(scales, requires_grad=geometry_trainable),
            "sh0": nn.Parameter(sh0, requires_grad=geometry_trainable),
            "opacities": nn.Parameter(opacities, requires_grad=True),
            "times": nn.Parameter(times, requires_grad=True),
            "log_durations": nn.Parameter(log_durations, requires_grad=True),
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
            n_bands = params["shN"].shape[1] + 1
            sh_degree = int(math.isqrt(n_bands) - 1)

        scene_extent = float((
            means.max(dim=0).values - means.min(dim=0).values
        ).max().clamp(min=1e-6))

        return params, sh_degree, scene_extent

    @torch.no_grad()
    def _convert_temporally_invariant_to_static(
        self,
        params: nn.ParameterDict,
        config: TrigsTrainingConfig,
        n_frames: int,
    ) -> int:
        if "is_static" not in params:
            return 0

        is_static = params["is_static"].data > 0.5
        protected = (
            params["dynamic_protect"].data > 0.5
            if "dynamic_protect" in params
            else torch.zeros_like(is_static, dtype=torch.bool)
        )
        dynamic = (~is_static) & (~protected)

        alpha = torch.sigmoid(params["opacities"].data).reshape(-1, 1)
        speed = params["zeta_base_t"].data.norm(dim=-1, keepdim=True)
        accel_t = (
            params["zeta_c0_t"].data - 2.0 * params["zeta_c1_t"].data + params["zeta_c2_t"].data
        ).norm(dim=-1, keepdim=True)

        rot_vec = params["zeta_base_r"].data
        accel_r_vec = (
            params["zeta_c0_r"].data - 2.0 * params["zeta_c1_r"].data + params["zeta_c2_r"].data
        )
        if bool(getattr(config, "static_convert_rot_uses_lever", True)) and "anchors" in params:
            # ζ_r is an angular rate (rad/s); the translation gates are world
            # units/s. Comparing them to the same threshold is a unit mismatch.
            # What decides whether freezing a row is safe is how far it actually
            # moves, and rotation about the anchor moves it by ω × r, where
            # r = mean − anchor. That lever averages 0.3 world units here, so
            # the raw-radian gate rejects rows whose real displacement is
            # negligible — and once a background row is rejected it keeps its
            # spurious rotation, which is what smears it across the clip.
            lever = params["means"].data - params["anchors"].data
            rot = torch.linalg.cross(rot_vec, lever, dim=-1).norm(dim=-1, keepdim=True)
            accel_r = torch.linalg.cross(accel_r_vec, lever, dim=-1).norm(dim=-1, keepdim=True)
        else:
            rot = rot_vec.norm(dim=-1, keepdim=True)
            accel_r = accel_r_vec.norm(dim=-1, keepdim=True)

        convert = (
            dynamic
            & (alpha >= float(config.static_convert_alpha_min))
            & (speed <= float(config.static_convert_speed_thresh))
            & (rot <= float(config.static_convert_rot_thresh))
            & (accel_t <= float(config.static_convert_accel_thresh))
            & (accel_r <= float(config.static_convert_accel_thresh))
        )

        # Temporal-coverage gate. Static rows are active at every t, so a row
        # that is merely motionless must not be promoted unless its visibility
        # window already covers the supervised clip. Without this a row with a
        # small σₜ — present for a fraction of the sequence — becomes
        # permanently visible, and because conversion is one-way it can never
        # be demoted again. Time units are seconds throughout.
        k = float(getattr(config, "static_convert_coverage_sigmas", 2.0))
        if k > 0.0:
            t_max = max(int(n_frames) - 1, 1) / max(float(config.fps), 1e-6)
            mu_t = params["times"].data.reshape(-1, 1)
            sigma_t = torch.exp(params["log_durations"].data).reshape(-1, 1)
            covers = (mu_t - k * sigma_t <= 0.0) & (mu_t + k * sigma_t >= t_max)
            convert = convert & covers

        convert = convert.reshape(-1)

        n = int(convert.sum().item())
        if n <= 0:
            return 0

        params["is_static"].data[convert, 0] = 1.0

        for name in (
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        ):
            if name in params:
                params[name].data[convert] = 0.0

        # Enforce class invariants after any static conversion.
        _enforce_relocation_class_invariants(params)

        return n

    # -------------------------------------------------------------------------
    # Forward deformation — TRiGS Eq. 4–9
    # -------------------------------------------------------------------------

    def _deform(
        self,
        params: nn.ParameterDict,
        t: float | Tensor,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        device = params["means"].device

        mu_t = params["times"]
        log_sigma_t = params["log_durations"]
        if isinstance(t, Tensor):
            t_scalar = t.to(dtype=mu_t.dtype)
        else:
            t_scalar = torch.tensor(t, device=device, dtype=mu_t.dtype)

        # Official TRiGS evaluates the Bézier window with μ_t / σ_t detached:
        # otherwise the temporal params can "retime" the motion curve (shift
        # or stretch the window to read a more convenient part of it) instead
        # of the curve being corrected. μ_t still learns through Δt and γ,
        # σ_t through γ.
        if getattr(config, "bezier_window_detach", True):
            tau = tau_from_window(t_scalar, mu_t.detach(), log_sigma_t.detach())
        else:
            tau = tau_from_window(t_scalar, mu_t, log_sigma_t)

        nu_res = bezier_quadratic(tau, params["zeta_c0_t"], params["zeta_c1_t"], params["zeta_c2_t"])
        omega_res = bezier_quadratic(tau, params["zeta_c0_r"], params["zeta_c1_r"], params["zeta_c2_r"])
        nu = params["zeta_base_t"] + nu_res
        omega = params["zeta_base_r"] + omega_res

        static_mask = None
        if "is_static" in params:
            static_mask = params["is_static"] > 0.5
            static3 = static_mask.expand_as(nu)
            nu = torch.where(static3, torch.zeros_like(nu), nu)
            omega = torch.where(static3, torch.zeros_like(omega), omega)

        a_perp, nu_par = gauge_fix_omega(
            omega, nu, params["anchors"],
            eps=config.gauge_eps, denom_clamp=config.gauge_denom_clamp,
        )

        dt = t_scalar - mu_t
        u = torch.cat([nu_par * dt, omega * dt], dim=-1)
        u = u.clamp(-1e4, 1e4)

        q_rel, p_par = exp_se3_quat(u)

        R = quaternion_to_rotmat(q_rel)
        centered = (params["means"] - a_perp).unsqueeze(-1)
        means_t = (R @ centered).squeeze(-1) + a_perp + p_par

        quats_t = quaternion_multiply(q_rel, F.normalize(params["quats"], dim=-1))

        sigma_t = torch.exp(log_sigma_t).clamp(min=1e-6)
        gamma = torch.exp(-0.5 * (dt / sigma_t) ** 2)

        # Smooth-cull renormalization: contribution is exactly zero at the
        # semantic support edge, so hard active culling at tau is lossless
        # and no row (or cohort of rows) can pop in at tau opacity while
        # spatially extrapolated. Static rows are overridden to 1 below.
        if (
            getattr(config, "temporal_gamma_smooth_cull", False)
            and getattr(config, "temporal_active_cull", False)
        ):
            tau = min(
                max(float(getattr(config, "temporal_active_threshold", 0.05)), 0.0),
                0.999,
            )
            if tau > 0.0:
                gamma = ((gamma - tau) / (1.0 - tau)).clamp(min=0.0)

        if static_mask is not None:
            means_t = torch.where(static_mask.expand_as(means_t), params["means"], means_t)
            quats_t = torch.where(
                static_mask.expand_as(quats_t),
                F.normalize(params["quats"], dim=-1),
                quats_t,
            )
            gamma = torch.where(static_mask, torch.ones_like(gamma), gamma)

        means_t = torch.nan_to_num(means_t, nan=0.0, posinf=0.0, neginf=0.0)
        quats_t = torch.nan_to_num(quats_t, nan=0.0, posinf=0.0, neginf=0.0)
        qt_norm_sq = (quats_t * quats_t).sum(dim=-1, keepdim=True)
        if (qt_norm_sq < 1e-16).any():
            identity = torch.zeros_like(quats_t)
            identity[..., 0] = 1.0
            quats_t = torch.where(qt_norm_sq < 1e-16, identity, quats_t)

        return means_t, quats_t, gamma

    # -------------------------------------------------------------------------
    # Rolling shutter
    # -------------------------------------------------------------------------

    def _rolling_shutter_times(
        self,
        means_t: Tensor,
        cameras: "CameraConfig",
        cam_idx: int,
        t: float | Tensor,
        config: TrigsTrainingConfig,
    ) -> Tensor | None:
        """Per-primitive capture time from the readout line it projects to.

        t is the camera's midpoint capture time (nominal + constant + learned offset); the result is
        t + sign * readout * (s / S - 0.5) with s along config.rs_axis (trigs.capture_timing convention).
        Returns an (N, 1) tensor of times, or None when rolling-shutter correction is off / unavailable for
        this camera.

        The projection is detached on purpose. Letting gradients run
        means -> column -> capture time -> means couples a primitive's position
        to its own exposure time; the effect is second order (the shift is
        bounded by rho/2) and the coupling is a needless source of instability.
        """
        if not config.rs_enabled():
            return None
        rs_signs = getattr(cameras, "rs_signs", None)
        if rs_signs is None:
            return None
        sign = float(rs_signs[cam_idx].item())
        if sign == 0.0:
            return None
        axis = str(getattr(config, "rs_axis", "column")).lower()
        if axis not in ("column", "row"):
            raise ValueError(f"rs_axis must be 'column' or 'row', got {axis!r}")
        use_sensor = bool(getattr(config, "rs_sensor_coords", False))
        if use_sensor and getattr(cameras, "rs_sensor_K", None) is None:
            raise ValueError("rs_sensor_coords=True needs CameraConfig.rs_sensor_K/_dist/_hw")

        readout_s = float(config.rs_readout_ms) / 1000.0

        with torch.no_grad():
            viewmat = cameras.viewmats[cam_idx]
            K = cameras.Ks[cam_idx]
            cam_xyz = means_t @ viewmat[:3, :3].T + viewmat[:3, 3]
            z = cam_xyz[:, 2]
            # Behind-camera and near-plane primitives get no shift rather than
            # a garbage column from dividing by ~0.
            safe = z > 1e-6
            z_safe = torch.where(safe, z, torch.ones_like(z))
            u = K[0, 0] * (cam_xyz[:, 0] / z_safe) + K[0, 2]
            v = K[1, 1] * (cam_xyz[:, 1] / z_safe) + K[1, 2]
            if use_sensor:
                from trigs import capture_timing as ct
                Kr = cameras.rs_sensor_K[cam_idx]
                u, v = ct.delivered_to_raw(u, v, K, Kr, cameras.rs_sensor_dist[cam_idx].tolist())
                raw_h, raw_w = (float(x) for x in cameras.rs_sensor_hw[cam_idx].tolist())
                coord, extent = (v, raw_h) if axis == "row" else (u, raw_w)
            elif axis == "row":
                coord, extent = v, float(cameras.height_for(cam_idx))
            else:
                coord, extent = u, float(cameras.width_for(cam_idx))
            # In-frame coordinates give frac in [-0.5, 0.5], so the readout spans
            # exactly rho across the image. Primitives projecting outside the
            # frame are pinned to the nearest edge's capture time: the sensor
            # has no rows out there, and an unbounded extrapolation would let a
            # far-off-screen primitive claim a time well outside the exposure.
            frac = torch.where(safe, coord / max(extent, 1.0) - 0.5, torch.zeros_like(coord))
            frac = frac.clamp(-0.5, 0.5)
            delta = (sign * readout_s) * frac

        if isinstance(t, Tensor):
            t_base = t.to(device=means_t.device, dtype=means_t.dtype).reshape(())
        else:
            t_base = torch.tensor(t, device=means_t.device, dtype=means_t.dtype)
        return (t_base + delta.to(means_t.dtype)).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Rasterization pass
    # -------------------------------------------------------------------------

    def _render(
        self,
        params: nn.ParameterDict,
        cameras: "CameraConfig",
        cam_idx: int,
        t: float | Tensor,
        sh_degree: int,
        config: TrigsTrainingConfig,
        step: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Dict]:
        active_idx = self._active_idx_for_time(params, t, config, step=step)
        if active_idx.numel() == 0:
            # gsplat 1.5.3 divides means.numel() by N * 3 in its native
            # projection launcher before checking N == 0, causing SIGFPE.
            # Fail with the temporal context rather than silently skipping
            # supervision (which could hide a timestamp/NPZ mismatch).
            query_time = float(t.detach().item()) if isinstance(t, Tensor) else float(t)
            times = params["times"].detach().reshape(-1)
            time_range = (
                f"[{float(times.min()):.6f}, {float(times.max()):.6f}] s"
                if times.numel() else "empty"
            )
            raise RuntimeError(
                "No temporally active Gaussians before rasterization: "
                f"camera={cam_idx}, step={step}, t={query_time:.9f} s, "
                f"total_primitives={params['means'].shape[0]}, "
                f"primitive_center_time_range={time_range}, "
                f"cull_threshold={self._effective_temporal_cull_threshold(config):.6g}. "
                "Check that frame_timestamps.json/sync.json and the NPZ use "
                "the same time origin, frame range, and FPS; also check "
                "temporal support and culling. Refusing an empty gsplat "
                "call because it can abort the process with SIGFPE."
            )
        render_params = self._slice_rowwise_params(params, active_idx)

        means_t, quats_t, gamma = self._deform(render_params, t, config)

        # Rolling shutter: the first deform placed every primitive at the
        # camera's nominal time. Re-deform each one at the capture time of the
        # column it actually lands in. The active set is deliberately NOT
        # recomputed -- the shift is bounded by rho/2 (a few ms) against
        # temporal supports of +/-2 sigma (hundreds of ms), and smooth-cull
        # makes gamma exactly zero at the support edge, so no row that matters
        # can be missed.
        t_rs = self._rolling_shutter_times(means_t, cameras, cam_idx, t, config)
        if t_rs is not None:
            means_t, quats_t, gamma = self._deform(render_params, t_rs, config)
            # Optional fixed-point refinement: the line was chosen from the nominal-time position; re-pick it
            # from the rolling-shutter position (one extra deform per pass).
            for _ in range(int(getattr(config, "rs_fixed_point_iters", 0))):
                t_rs = self._rolling_shutter_times(means_t, cameras, cam_idx, t, config)
                means_t, quats_t, gamma = self._deform(render_params, t_rs, config)

        base_opa = torch.sigmoid(render_params["opacities"])
        if base_opa.dim() == 1:
            base_opa = base_opa.unsqueeze(-1)
        effective_opa = torch.clamp(base_opa * gamma, min=1e-10).squeeze(-1)

        scales = torch.exp(render_params["scales"])
        sh0 = render_params["sh0"]
        shN = render_params.get("shN")
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

        if info.get("means2d") is not None and info["means2d"].requires_grad:
            info["means2d"].retain_grad()

        info["active_idx"] = active_idx
        info["n_active"] = int(active_idx.numel())
        info["n_total"] = int(params["means"].shape[0])

        gamma_full = params["means"].new_zeros((int(params["means"].shape[0]), 1))
        gamma_full[active_idx] = gamma.detach()
        info["gamma_active"] = gamma

        return render_colors, render_alphas, gamma_full, info

    # -------------------------------------------------------------------------
    # GT caching
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
            print("  Rendering GT from dense PLYs (uint8 cache in CPU RAM)...")
        for fi, ply_file in enumerate(ply_files):
            gs = load_ply(ply_file, device=str(self.device))
            gs = transform_config.apply(gs)
            result = renderer.render(gs, cameras, return_alphas=True, show_progress=False)
            gt_images[fi] = [
                img.clamp(0, 1).mul(255).to(torch.uint8).cpu() for img in result.images
            ]
            gt_alphas[fi] = [
                a.clamp(0, 1).mul(255).to(torch.uint8).cpu() for a in result.alphas
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
        temporal_betas = (
            float(config.temporal_optimizer_beta1),
            0.999,
        )
        batch = max(int(getattr(config, "batch_cameras", 1)), 1)
        if not getattr(config, "batch_optimizer_scaling", True):
            batch = 1
        lr_batch = batch if getattr(config, "batch_lr_scaling", False) else 1

        def adam(
            ps,
            lr,
            *,
            betas: tuple[float, float] = (0.9, 0.999),
        ):
            # Official TRiGS batch rule; identity at batch == 1.
            return torch.optim.Adam(
                ps,
                lr=lr * lr_batch ** 0.5,
                betas=(betas[0] ** batch, betas[1] ** batch),
                eps=1e-15 / batch ** 0.5,
            )

        optimizers: Dict[str, torch.optim.Optimizer] = {
            "opacities": adam([params["opacities"]], config.lr_opacity),
            "times": adam([params["times"]], config.lr_times, betas=temporal_betas),
            "log_durations": adam([params["log_durations"]], config.lr_durations, betas=temporal_betas),
            "anchors": adam([params["anchors"]], config.lr_anchor * scene_extent),
            "zeta_base_t": adam([params["zeta_base_t"]], config.lr_zeta_base_trans * scene_extent),
            "zeta_bezier_t": adam(
                [params["zeta_c0_t"], params["zeta_c1_t"], params["zeta_c2_t"]],
                config.lr_zeta_bezier_trans * scene_extent,
            ),
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

        if "camera_time_offsets" in params:
            optimizers["camera_time_offsets"] = torch.optim.Adam(
                [params["camera_time_offsets"]], lr=config.lr_camera_time_offsets
            )
        if "cam_gain" in params:
            optimizers["camera_appearance"] = torch.optim.Adam(
                [params["cam_gain"], params["cam_bias"]], lr=config.lr_camera_appearance
            )
        return optimizers

    # -------------------------------------------------------------------------
    # NN velocity init
    # -------------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def _knn_mean_positions(src: Tensor, tgt: Tensor, k: int) -> Tensor:
        k = max(1, min(k, tgt.shape[0]))
        try:
            from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)
            res = knn_points(src.unsqueeze(0), tgt.unsqueeze(0), K=k)
            nn_idx = res.idx.squeeze(0)
        except ImportError:
            d = torch.cdist(src, tgt)
            nn_idx = d.topk(k, largest=False).indices
        nn_pos = tgt[nn_idx]
        return nn_pos.mean(dim=1)

    @staticmethod
    @torch.no_grad()
    def _radius_ball_centroid(means: Tensor, radius: float, chunk_size: int = 2048) -> Tensor:
        N = means.shape[0]
        centroids = torch.empty_like(means)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = means[start:end]
            d = torch.cdist(chunk, means)
            mask = (d > 1e-8) & (d < radius)
            mask_f = mask.float()
            counts = mask_f.sum(dim=1, keepdim=True).clamp(min=1)
            centroids[start:end] = (mask_f @ means) / counts
            no_nb = ~mask.any(dim=1)
            if no_nb.any():
                centroids[start:end][no_nb] = chunk[no_nb]
        return centroids

    @torch.no_grad()
    def _nn_velocity_init(self, params: nn.ParameterDict, verbose: bool = False, k: int = 5) -> int:
        times = params["times"].data.squeeze(-1)
        means = params["means"].data
        nu = params["zeta_base_t"].data

        unique_t = torch.unique(times).sort().values
        if unique_t.numel() < 2:
            return 0

        bucket_idx = torch.searchsorted(unique_t, times)
        n_seeded = 0
        K = unique_t.numel()
        for b in range(K):
            mask_k = bucket_idx == b
            if not mask_k.any():
                continue
            tgt_b = b + 1 if b < K - 1 else b - 1
            mask_tgt = bucket_idx == tgt_b
            if not mask_tgt.any():
                continue
            dt = (unique_t[tgt_b] - unique_t[b]).item()
            if abs(dt) < 1e-12:
                continue
            src_pos = means[mask_k]
            tgt_pos = means[mask_tgt]
            nn_mean_pos = self._knn_mean_positions(src_pos, tgt_pos, k)
            displacement = nn_mean_pos - src_pos
            nu[mask_k] = displacement / dt
            n_seeded += int(mask_k.sum().item())

        if verbose:
            nnz = (nu.norm(dim=-1) > 0).float().mean().item()
            print(f"  NN ν_base init: {nnz * 100:.1f}% of primitives have non-zero ν")
        return n_seeded

    # -------------------------------------------------------------------------
    # k-NN graph for L_rigid
    # -------------------------------------------------------------------------

    @staticmethod
    def _rigid_knn_points(
        params: nn.ParameterDict, config: TrigsTrainingConfig,
    ) -> Tensor:
        space = str(getattr(config, "rigid_knn_space", "anchors")).lower()
        if space == "anchors":
            return params["anchors"]
        if space == "means":
            return params["means"]
        raise ValueError(
            f"rigid_knn_space must be 'anchors' or 'means', got {space!r}"
        )

    @torch.no_grad()
    def _compute_knn(
        self, means: Tensor, k: int,
        times: Optional[Tensor] = None, time_weight: float = 0.0,
    ) -> Tensor:
        if times is not None and time_weight > 0:
            t_col = times if times.dim() == 2 else times.unsqueeze(-1)
            pts = torch.cat([means, time_weight * t_col], dim=-1)
        else:
            pts = means

        try:
            from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)
            pts_batched = pts.unsqueeze(0)
            result = knn_points(pts_batched, pts_batched, K=k + 1)
            idx = result.idx.squeeze(0)[:, 1:]
            return idx.contiguous()
        except ImportError:
            d = torch.cdist(pts, pts)
            d.fill_diagonal_(float("inf"))
            return d.topk(k, largest=False).indices

    # -------------------------------------------------------------------------
    # Delayed budget pruning
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _select_drop_mask_by_inactive(
        self, params: nn.ParameterDict, target_count: int, tau_alpha: float,
    ) -> Tensor:
        N = int(params["means"].shape[0])
        if target_count >= N:
            return torch.zeros(N, dtype=torch.bool, device=params["means"].device)
        if target_count <= 0:
            raise ValueError("target_count must be > 0 when delayed pruning is enabled")
        alpha = torch.sigmoid(params["opacities"].data).reshape(-1)
        inactive_mask = alpha < (10 * tau_alpha)

        if "is_static" in params:
            static_mask = params["is_static"].data.reshape(-1) > 0.5
            inactive_mask &= ~static_mask

        inactive_idx = torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)
        n_remove = min(max(0, N - target_count), int(inactive_idx.numel()))
        if n_remove <= 0:
            return torch.zeros(N, dtype=torch.bool, device=params["means"].device)

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
        self, params: nn.ParameterDict, optimizers: Dict[str, torch.optim.Optimizer],
        drop_mask: Tensor,
    ) -> Optional[Tensor]:
        n_old = int(params["means"].shape[0])
        keep_idx = torch.where(~drop_mask)[0]
        if int(keep_idx.numel()) == n_old:
            return None

        owners = build_param_owners(optimizers)
        primitive_items = [
            (name, p) for name, p in list(params.items())
            if p.data.ndim >= 1 and int(p.data.shape[0]) == n_old
        ]

        for name, old_param in primitive_items:
            new_data = old_param.data[keep_idx].contiguous()
            new_param = nn.Parameter(new_data, requires_grad=old_param.requires_grad)
            params[name] = new_param
            remap_param_with_sliced_state(
                owners=owners, old_param=old_param, new_param=new_param,
                keep_idx=keep_idx, n_old=n_old,
            )
        return keep_idx

    # -------------------------------------------------------------------------
    # Step 3: Pool-aware relocation
    # -------------------------------------------------------------------------

    def _photometric_losses(
        self,
        params: nn.ParameterDict,
        render_colors: Tensor,
        render_alphas: Tensor,
        gt_rgb: Tensor,
        gt_alphas_per_frame: Optional[Dict[int, List[Tensor]]],
        frame_idx: int,
        cam_idx: int,
        config: TrigsTrainingConfig,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """RGB, alpha-outside and alpha-mask losses for one rendered camera."""
        pred_color = render_colors[0]
        clamp_for_loss = bool(getattr(config, "clamp_render_for_loss", True))
        if "cam_gain" in params:
            pred_color = pred_color * params["cam_gain"][cam_idx] + params["cam_bias"][cam_idx]

        if config.masked_rgb_loss:
            mask_obj = gt_alphas_per_frame[frame_idx][cam_idx]
            if not torch.is_tensor(mask_obj):
                raise RuntimeError(
                    f"masked_rgb_loss=True but gt_alphas_per_frame did not return a tensor. "
                    f"Got {type(mask_obj)}."
                )
            gt_mask = mask_obj.to(self.device).float()
            if gt_mask.max() > 1.0:
                gt_mask = gt_mask / 255.0
            if gt_mask.ndim == 2:
                gt_mask = gt_mask[..., None]
            elif gt_mask.ndim == 3 and gt_mask.shape[-1] != 1:
                gt_mask = gt_mask[..., :1]
            if tuple(gt_mask.shape[:2]) != tuple(render_colors[0].shape[:2]):
                gt_mask = F.interpolate(
                    gt_mask.permute(2, 0, 1).unsqueeze(0),
                    size=render_colors[0].shape[:2], mode="bilinear", align_corners=False,
                )[0].permute(1, 2, 0)
            gt_mask = gt_mask.clamp(0.0, 1.0)

            alpha = render_alphas[0]
            if alpha.ndim == 2:
                alpha = alpha[..., None]
            elif alpha.ndim == 3 and alpha.shape[-1] != 1:
                alpha = alpha[..., :1]
            alpha = alpha.float().clamp(0.0, 1.0)
            outside = (1.0 - gt_mask).clamp(0.0, 1.0)

            if config.masked_rgb_composite:
                # Composite the (premultiplied) render over a constant
                # background and supervise the FULL frame against the GT
                # matted onto that same background.
                #
                # The alternative below replaces the prediction with GT
                # outside the mask, so those pixels contribute exactly zero
                # gradient: nothing photometric constrains out-of-mask mass,
                # and lambda_alpha_outside_mask is the only thing holding it
                # back. That also removes the background as a depth anchor —
                # a primitive at the wrong depth that still projects inside
                # the silhouette from every training camera pays no penalty,
                # which is what novel views expose as floaters.
                #
                # Compositing restores a real RGB gradient outside the
                # silhouette (a floater darkens a white plate) without ever
                # asking the model to represent the background.
                if config.masked_rgb_random_bg:
                    bg = torch.rand(
                        (1, 1, pred_color.shape[-1]),
                        device=pred_color.device,
                        dtype=pred_color.dtype,
                    )
                else:
                    bg = torch.tensor(
                        config.masked_rgb_bg_color,
                        device=pred_color.device, dtype=pred_color.dtype,
                    ).reshape(1, 1, -1)
                if bg.shape[-1] != pred_color.shape[-1]:
                    raise ValueError(
                        f"masked_rgb_bg_color has {bg.shape[-1]} channels but "
                        f"the render has {pred_color.shape[-1]}."
                    )
                pred_for_loss = pred_color + (1.0 - alpha) * bg
                if clamp_for_loss:
                    pred_for_loss = pred_for_loss.clamp(0.0, 1.0)
                gt_for_loss = gt_rgb * gt_mask + bg * outside
            else:
                if clamp_for_loss:
                    pred_color = pred_color.clamp(0.0, 1.0)
                pred_for_loss = pred_color * gt_mask + gt_rgb * (1.0 - gt_mask)
                gt_for_loss = gt_rgb
            rgb_loss = reconstruction_loss(
                pred_for_loss, gt_for_loss, ssim_lambda=config.ssim_lambda,
            )
            # Normalize by the full frame, matching F.l1_loss's mean over
            # H*W*3 in the RGB term above. Region-relative denominators
            # (outside.sum() / gt_mask.sum()) make the alpha-to-RGB
            # gradient ratio scale as 1/(1-f) in the foreground fraction f,
            # so a tightly framed camera suppresses opacity orders of
            # magnitude harder than a wide one. The canonical set is shared
            # across the rig, so that per-view imbalance lands on the same
            # parameters and the optimum degenerates to a translucent
            # cloud. Frame-relative normalization keeps lambda meaning one
            # thing for every camera.
            n_px = float(gt_mask.shape[0] * gt_mask.shape[1])
            alpha_outside_loss = (alpha * outside).sum() / n_px
            alpha_safe = alpha.clamp(config.alpha_loss_eps, 1.0 - config.alpha_loss_eps)
            bce = F.binary_cross_entropy(alpha_safe, gt_mask, reduction="none")
            inside_bce = (bce * gt_mask).sum() / n_px
            outside_bce = (bce * outside).sum() / n_px
            alpha_mask_loss = 0.5 * (inside_bce + outside_bce)
        else:
            if clamp_for_loss:
                pred_color = pred_color.clamp(0.0, 1.0)
            rgb_loss = reconstruction_loss(pred_color, gt_rgb, ssim_lambda=config.ssim_lambda)
            alpha_outside_loss = torch.zeros((), device=self.device)
            alpha_mask_loss = torch.zeros((), device=self.device)

        return rgb_loss, alpha_outside_loss, alpha_mask_loss

    def _temporal_chunk_outside_mass_loss(
        self,
        params: nn.ParameterDict,
        config: TrigsTrainingConfig,
    ) -> Tensor:
        """Penalize visible temporal mass beyond the owner's supervised span.

        The baseline tail mass outside the hard cull radius is subtracted, so
        the loss is exactly zero while the row's cull-support is contained in
        the supervised interval.  Sequence-edge tails are ignored because no
        captured frame exists beyond the clip.
        """
        if float(getattr(config, "lambda_temporal_chunk_outside_mass", 0.0)) <= 0.0:
            return params["means"].new_zeros(())
        if (
            "chunk_owner_mask" not in params
            or "chunk_supervision_lo" not in params
            or "chunk_supervision_hi" not in params
        ):
            return params["means"].new_zeros(())

        owner = params["chunk_owner_mask"].detach().reshape(-1) > 0.5
        if "is_static" in params:
            owner &= params["is_static"].detach().reshape(-1) <= 0.5
        owner_idx = torch.nonzero(owner, as_tuple=False).reshape(-1)
        if int(owner_idx.numel()) == 0:
            return params["means"].new_zeros(())
        sample_size = int(
            getattr(config, "temporal_chunk_outside_mass_sample_size", 65_536)
        )
        if sample_size > 0 and int(owner_idx.numel()) > sample_size:
            draw = torch.randint(
                0, int(owner_idx.numel()), (sample_size,), device=owner_idx.device,
            )
            owner_idx = owner_idx[draw]

        mu = params["times"].reshape(-1)[owner_idx]
        sigma = params["log_durations"].reshape(-1)[owner_idx].exp().clamp_min(1e-6)
        lo = params["chunk_supervision_lo"].reshape(())
        hi = params["chunk_supervision_hi"].reshape(())
        t_max = params.get("chunk_sequence_t_max", hi).reshape(())

        sqrt2 = math.sqrt(2.0)
        z_lo = (lo - mu) / (sqrt2 * sigma)
        z_hi = (hi - mu) / (sqrt2 * sigma)
        cdf_lo = 0.5 * (1.0 + torch.erf(z_lo))
        cdf_hi = 0.5 * (1.0 + torch.erf(z_hi))

        left_mass = cdf_lo.clamp(0.0, 1.0)
        right_mass = (1.0 - cdf_hi).clamp(0.0, 1.0)
        # Do not penalize support outside the captured sequence itself.
        left_mass = torch.where(
            lo <= 1e-8, torch.zeros_like(left_mass), left_mass,
        )
        right_mass = torch.where(
            hi >= t_max - 1e-8, torch.zeros_like(right_mass), right_mass,
        )

        threshold = self._effective_temporal_cull_threshold(config)
        k = math.sqrt(-2.0 * math.log(max(min(threshold, 1.0 - 1e-8), 1e-8)))
        one_sided_baseline = 0.5 * math.erfc(k / sqrt2)
        excess_mass = (
            torch.relu(left_mass - one_sided_baseline)
            + torch.relu(right_mass - one_sided_baseline)
        )

        alpha = torch.sigmoid(params["opacities"].reshape(-1)[owner_idx]).detach()
        return (alpha * excess_mass).mean()

    @torch.no_grad()
    def _zero_frozen_chunk_grads(self, params: nn.ParameterDict) -> None:
        """
        Chunked-training ownership freezing. When train_chunked() installs a
        ``chunk_train_mask`` (1 = owned by this chunk, 0 = resident for
        rendering context only), zero the gradients of non-owned rows for
        every row-aligned parameter, so no chunk can modify rows another
        chunk is authoritative for.
        """
        if "chunk_train_mask" not in params:
            return
        mask = params["chunk_train_mask"].detach().reshape(-1)
        frozen = mask <= 0.5
        if not bool(frozen.any()):
            return
        n_rows = int(mask.shape[0])
        for name in params.keys():
            p = params[name]
            if p.grad is None:
                continue
            if p.dim() >= 1 and int(p.shape[0]) == n_rows:
                p.grad[frozen] = 0

    @torch.no_grad()
    def _apply_pool_aware_relocation(
        self,
        params: nn.ParameterDict,
        dead_idx: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> int:
        """Apply pool-aware relocation: dynamic → dynamic only, static → static only.

        Args:
            params: Canonical parameter dict.
            dead_idx: [M] long tensor of dead (low-opacity) primitive indices.
            candidate_mask: [N] bool tensor of donor candidates.

        Returns:
            Number of primitives relocated.
        """
        dynamic_pool, static_pool = _relocation_pool_masks(params)

        dead_idx = dead_idx.reshape(-1).long()
        dead_dynamic = dead_idx[dynamic_pool[dead_idx]]
        dead_static = dead_idx[static_pool[dead_idx]]

        candidate_mask = candidate_mask.reshape(-1).bool()
        dynamic_src_candidates = torch.where(candidate_mask & dynamic_pool)[0]
        static_src_candidates = torch.where(candidate_mask & static_pool)[0]

        n_relocated = 0

        # --- Dynamic pool ---
        if len(dead_dynamic) > 0 and len(dynamic_src_candidates) > 0:
            src_dynamic = dynamic_src_candidates[
                torch.randint(0, len(dynamic_src_candidates), (len(dead_dynamic),),
                              device=dead_dynamic.device)
            ]

            params["means"].data[dead_dynamic] = params["means"].data[src_dynamic]
            params["quats"].data[dead_dynamic] = params["quats"].data[src_dynamic]
            params["scales"].data[dead_dynamic] = params["scales"].data[src_dynamic]
            if "features_dc" in params:
                params["features_dc"].data[dead_dynamic] = params["features_dc"].data[src_dynamic]
            if "features_rest" in params:
                params["features_rest"].data[dead_dynamic] = params["features_rest"].data[src_dynamic]
            params["opacities"].data[dead_dynamic] = params["opacities"].data[src_dynamic]

            # Motion fields are dynamic-pool only.
            for key in (
                "times", "log_durations", "anchors",
                "zeta_base_t", "zeta_base_r",
                "zeta_c0_t", "zeta_c1_t", "zeta_c2_t",
                "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
            ):
                if key in params:
                    params[key].data[dead_dynamic] = params[key].data[src_dynamic]

            _copy_relocation_flags_same_pool(params, dead_dynamic, src_dynamic)
            n_relocated += len(dead_dynamic)

        # --- Static pool ---
        if len(dead_static) > 0 and len(static_src_candidates) > 0:
            src_static = static_src_candidates[
                torch.randint(0, len(static_src_candidates), (len(dead_static),),
                              device=dead_static.device)
            ]

            params["means"].data[dead_static] = params["means"].data[src_static]
            params["quats"].data[dead_static] = params["quats"].data[src_static]
            params["scales"].data[dead_static] = params["scales"].data[src_static]
            if "features_dc" in params:
                params["features_dc"].data[dead_static] = params["features_dc"].data[src_static]
            if "features_rest" in params:
                params["features_rest"].data[dead_static] = params["features_rest"].data[src_static]
            params["opacities"].data[dead_static] = params["opacities"].data[src_static]

            # Static rows must stay motionless/persistent — zero motion fields.
            for key in (
                "zeta_base_t", "zeta_base_r",
                "zeta_c0_t", "zeta_c1_t", "zeta_c2_t",
                "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
            ):
                if key in params:
                    params[key].data[dead_static].zero_()

            if "anchors" in params:
                params["anchors"].data[dead_static] = params["means"].data[dead_static]

            if "times" in params:
                # Static rows are persistent; midpoint fine because gamma=1 for static in render.
                params["times"].data[dead_static].fill_(0.5)

            if "log_durations" in params:
                params["log_durations"].data[dead_static].fill_(0.0)

            # Explicitly keep class flags static/background.
            if "is_static" in params:
                params["is_static"].data[dead_static].fill_(1.0)
            if "dynamic_protect" in params:
                params["dynamic_protect"].data[dead_static].zero_()
            if "dynamic_allowed" in params:
                params["dynamic_allowed"].data[dead_static].zero_()

            n_relocated += len(dead_static)

        # Step 4: apply hard post-relocation invariant clamp.
        _enforce_relocation_class_invariants(params)

        return n_relocated

    # -------------------------------------------------------------------------
    # Tensorboard stats helper
    # -------------------------------------------------------------------------

    @classmethod
    def _eval_render_time(
        cls, params, frame_idx: int, cam_idx: int, gt_times_per_frame, config,
    ) -> float:
        """Capture time an eval pair is rendered at (sync time + learned offset)."""
        if gt_times_per_frame is not None and gt_times_per_frame[frame_idx][cam_idx] is not None:
            t_base = gt_times_per_frame[frame_idx][cam_idx]
        else:
            t_base = float(frame_idx) / max(config.fps, 1e-6)
        if "camera_time_offsets" in params:
            scale = cls._camera_time_offset_scale(
                params, params["camera_time_offsets"].shape[0]
            )
            return float(t_base) + float(
                (scale[cam_idx] * params["camera_time_offsets"][cam_idx]).item()
            )
        return float(t_base)

    @torch.no_grad()
    def _evaluate_psnr_subset(
        self, *, params, cameras, eval_pairs, sh_degree, config,
        gt_images_per_frame, gt_alphas_per_frame, gt_times_per_frame,
        step: int,
    ) -> Dict[str, float]:
        if not eval_pairs:
            return {}
        return self._psnr_metrics_from_accum(self._psnr_subset_accum(
            params=params, cameras=cameras, eval_pairs=eval_pairs,
            sh_degree=sh_degree, config=config,
            gt_images_per_frame=gt_images_per_frame,
            gt_alphas_per_frame=gt_alphas_per_frame,
            gt_times_per_frame=gt_times_per_frame, step=step,
        ))

    @torch.no_grad()
    def _psnr_subset_accum(
        self, *, params, cameras, eval_pairs, sh_degree, config,
        gt_images_per_frame, gt_alphas_per_frame, gt_times_per_frame,
        step: int,
    ) -> Dict[str, Any]:
        """Per-pair squared-error sums, so partial evals (one per temporal
        chunk) combine into exactly the metrics of a single full eval."""
        sse_full = 0.0; denom_full = 0.0
        sse_masked = 0.0; denom_masked = 0.0
        image_psnrs_full: List[float] = []
        image_psnrs_masked: List[float] = []

        for frame_idx, cam_idx in eval_pairs:
            t = self._eval_render_time(
                params, frame_idx, cam_idx, gt_times_per_frame, config,
            )

            render_colors, render_alphas, gamma, info = self._render(
                params,
                cameras,
                cam_idx,
                t,
                sh_degree,
                config,
                step=step,
            )
            pred = _as_float_rgb01(render_colors[0].detach(), clamp=config.psnr_clamp_rgb)
            gt = _as_float_rgb01(gt_images_per_frame[frame_idx][cam_idx].to(self.device),
                                  clamp=config.psnr_clamp_rgb)

            if gt_alphas_per_frame is not None:
                try:
                    mask_raw = gt_alphas_per_frame[frame_idx][cam_idx].to(self.device)
                except Exception:
                    mask_raw = None
            else:
                mask_raw = None

            mask = _as_mask01(mask_raw, (pred.shape[0], pred.shape[1]),
                               threshold=config.psnr_mask_threshold)

            if (
                config.masked_rgb_loss
                and config.masked_rgb_composite
                and mask is not None
            ):
                # Match the training objective. Without this, psnr/eval_full
                # compares a premultiplied-over-black render against a GT whose
                # background is not black, and stays pinned near a constant
                # regardless of reconstruction quality.
                bg_eval = torch.tensor(
                    config.masked_rgb_bg_color,
                    device=pred.device, dtype=pred.dtype,
                ).reshape(1, 1, -1)
                a = render_alphas[0].detach()
                if a.ndim == 2:
                    a = a[..., None]
                elif a.ndim == 3 and a.shape[-1] != 1:
                    a = a[..., :1]
                a = a.to(pred.dtype).clamp(0.0, 1.0)
                m = mask.to(device=pred.device, dtype=pred.dtype)
                pred = pred + (1.0 - a) * bg_eval
                gt = gt * m + bg_eval * (1.0 - m)
                if config.psnr_clamp_rgb:
                    pred = pred.clamp(0.0, 1.0)
                    gt = gt.clamp(0.0, 1.0)

            diff2 = f64(pred - gt).pow(2)
            sse_i = float(diff2.sum().item())
            denom_i = float(diff2.numel())
            sse_full += sse_i; denom_full += denom_i
            mse_i = max(sse_i / max(denom_i, 1.0), 1e-12)
            image_psnrs_full.append(float(-10.0 * math.log10(mse_i)))

            if mask is not None:
                mask = mask.to(device=diff2.device, dtype=torch.float64)
                denom_m_i = float((mask.sum() * pred.shape[-1]).item())
                if denom_m_i > 0:
                    sse_m_i = float((diff2 * mask).sum().item())
                    sse_masked += sse_m_i; denom_masked += denom_m_i
                    mse_m_i = max(sse_m_i / max(denom_m_i, 1.0), 1e-12)
                    image_psnrs_masked.append(float(-10.0 * math.log10(mse_m_i)))

        return {
            "sse_full": sse_full, "denom_full": denom_full,
            "sse_masked": sse_masked, "denom_masked": denom_masked,
            "image_psnrs_full": image_psnrs_full,
            "image_psnrs_masked": image_psnrs_masked,
            "n_pairs": len(eval_pairs),
        }

    @staticmethod
    def _merge_psnr_accums(accums: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "sse_full": 0.0, "denom_full": 0.0,
            "sse_masked": 0.0, "denom_masked": 0.0,
            "image_psnrs_full": [], "image_psnrs_masked": [], "n_pairs": 0,
        }
        for acc in accums:
            for key in ("sse_full", "denom_full", "sse_masked", "denom_masked", "n_pairs"):
                merged[key] += acc[key]
            merged["image_psnrs_full"] += list(acc["image_psnrs_full"])
            merged["image_psnrs_masked"] += list(acc["image_psnrs_masked"])
        return merged

    @staticmethod
    def _psnr_metrics_from_accum(acc: Dict[str, Any]) -> Dict[str, float]:
        sse_full, denom_full = acc["sse_full"], acc["denom_full"]
        sse_masked, denom_masked = acc["sse_masked"], acc["denom_masked"]
        image_psnrs_full = acc["image_psnrs_full"]
        image_psnrs_masked = acc["image_psnrs_masked"]
        out: Dict[str, float] = {}
        if denom_full > 0:
            mse_full = max(sse_full / denom_full, 1e-12)
            out["psnr/eval_full"] = float(-10.0 * math.log10(mse_full))
            out["mse/eval_full"] = float(mse_full)
            out["psnr/eval_full_mean_image"] = float(sum(image_psnrs_full) / max(len(image_psnrs_full), 1))
        if denom_masked > 0:
            mse_masked_val = max(sse_masked / denom_masked, 1e-12)
            out["psnr/eval_masked"] = float(-10.0 * math.log10(mse_masked_val))
            out["mse/eval_masked"] = float(mse_masked_val)
            out["psnr/eval_masked_mean_image"] = float(
                sum(image_psnrs_masked) / max(len(image_psnrs_masked), 1))
            out["mask/eval_pixels_mean"] = float(
                denom_masked / (3.0 * max(len(image_psnrs_masked), 1)))
        out["psnr/eval_num_samples"] = float(acc["n_pairs"])
        return out

    @torch.no_grad()
    def _chunked_psnr_eval(
        self,
        *,
        residency,
        ownership_intervals: Sequence[tuple[float, float]],
        cameras,
        n_frames: int,
        sh_degree: int,
        config: TrigsTrainingConfig,
        gt_images_per_frame,
        gt_alphas_per_frame,
        gt_times_per_frame,
        cull_threshold: float,
        step: int,
    ) -> Dict[str, Dict[str, float]]:
        """psnr/eval_* (and psnr/holdout_*) for a chunked model.

        Uses the same fixed (frame, view) pairs as an unchunked run. Each pair
        is rendered from the rows active at its capture time, materialized
        from the master one ownership window at a time, so GPU memory stays
        at chunk scale and the result equals a full-model eval under the same
        temporal culling.
        """
        import dataclasses

        from .temporal_chunks import chunk_row_idx

        n_views = len(cameras)
        holdout = sorted({int(v) for v in getattr(config, "holdout_views", ())})
        pair_sets = {
            "eval": _make_psnr_eval_pairs(
                n_frames=n_frames, n_views=n_views,
                num_samples=config.psnr_eval_num_samples, seed=config.psnr_eval_seed,
                views=[v for v in range(n_views) if v not in set(holdout)],
            ),
        }
        if holdout:
            pair_sets["holdout"] = _make_psnr_eval_pairs(
                n_frames=n_frames, n_views=n_views,
                num_samples=config.psnr_eval_num_samples, seed=config.psnr_eval_seed,
                views=holdout,
            )

        # Direct culling: exact, and needs no hierarchy over the eval rows.
        eval_config = dataclasses.replace(config, temporal_hierarchy_enabled=False)
        fps = max(float(config.fps), 1e-6)
        master = residency.master
        accums: Dict[str, List[Dict[str, Any]]] = {name: [] for name in pair_sets}
        for own_lo, own_hi in ownership_intervals:
            owned = {
                name: [(f, c) for f, c in pairs if own_lo <= f / fps < own_hi]
                for name, pairs in pair_sets.items()
            }
            all_owned = [p for pairs in owned.values() for p in pairs]
            if not all_owned:
                continue
            times = [
                self._eval_render_time(master, f, c, gt_times_per_frame, config)
                for f, c in all_owned
            ]
            idx = chunk_row_idx(
                master["times"], master["log_durations"], min(times), max(times),
                opacity_threshold=cull_threshold,
                extra_margin_s=config.temporal_active_extra_margin_s,
                is_static=master.get("is_static"), include_static=True,
            )
            if int(idx.numel()) == 0:
                continue
            sub_params = residency.materialize(idx, self.device)
            for name, pairs in owned.items():
                if pairs:
                    accums[name].append(self._psnr_subset_accum(
                        params=sub_params, cameras=cameras, eval_pairs=pairs,
                        sh_degree=sh_degree, config=eval_config,
                        gt_images_per_frame=gt_images_per_frame,
                        gt_alphas_per_frame=gt_alphas_per_frame,
                        gt_times_per_frame=gt_times_per_frame, step=step,
                    ))
            del sub_params
            empty_cache()

        results: Dict[str, Dict[str, float]] = {}
        for name, parts in accums.items():
            if parts:
                metrics = self._psnr_metrics_from_accum(self._merge_psnr_accums(parts))
                if name == "holdout":
                    metrics = {k.replace("eval_", "holdout_"): v for k, v in metrics.items()}
                results[name] = metrics
        return results

    @torch.no_grad()
    def _temporal_entry_diagnostics(
        self,
        params: dict,
        *,
        config: TrigsTrainingConfig,
        n_frames: int,
        windows: Sequence[tuple[float, float]],
        ownership_bounds: Sequence[float],
        label: str,
    ) -> Dict[str, Any]:
        """Summarize cohorts that cross the hard temporal cull edge.

        Entry frames are computed analytically from ``mu_t - k*sigma_t``.
        For the largest cohorts we then run the real deformation path on a
        bounded sample and report exact displacement from canonical means.
        """
        n = int(params["means"].shape[0])
        fps = max(float(config.fps), 1e-6)
        threshold = self._effective_temporal_cull_threshold(config)
        k = math.sqrt(-2.0 * math.log(threshold))

        mu = params["times"].detach().reshape(-1).float().cpu()
        sigma = (
            params["log_durations"].detach().exp().reshape(-1)
            .float().cpu().clamp_min(1e-6)
        )
        radius = sigma * k + float(config.temporal_active_extra_margin_s)
        lo = mu - radius
        hi = mu + radius

        if "is_static" in params:
            dynamic = params["is_static"].detach().reshape(-1).cpu() <= 0.5
        else:
            dynamic = torch.ones(n, dtype=torch.bool)

        entry_frame = torch.ceil(lo * fps - 1e-7).to(torch.long)
        entry_time = entry_frame.float() / fps
        valid = (
            dynamic
            & (entry_frame >= 1)
            & (entry_frame < int(n_frames))
            & (entry_time <= hi + 1e-7)
        )
        valid_idx = torch.nonzero(valid, as_tuple=False).reshape(-1)

        counts = torch.bincount(
            entry_frame[valid], minlength=max(int(n_frames), 1),
        ).to(torch.long)
        dt_entry = entry_time[valid] - mu[valid]
        gamma_entry = torch.exp(-0.5 * (dt_entry / sigma[valid]).pow(2))
        # Match the renderer: with smooth-cull renormalization the visible
        # opacity at the support edge is ~0, and this metric should report
        # the actual pop amplitude the viewer sees.
        if (
            getattr(config, "temporal_gamma_smooth_cull", False)
            and getattr(config, "temporal_active_cull", False)
        ):
            tau_entry = min(
                max(float(getattr(config, "temporal_active_threshold", 0.05)), 0.0),
                0.999,
            )
            if tau_entry > 0.0:
                gamma_entry = (
                    (gamma_entry - tau_entry) / (1.0 - tau_entry)
                ).clamp(min=0.0)
        if "opacities" in params:
            alpha = torch.sigmoid(
                params["opacities"].detach().reshape(-1).float().cpu()
            )
        else:
            alpha = torch.ones(n, dtype=torch.float32)
        entry_mass = torch.bincount(
            entry_frame[valid],
            weights=alpha[valid] * gamma_entry,
            minlength=max(int(n_frames), 1),
        )

        if ownership_bounds:
            bounds = torch.tensor(ownership_bounds, dtype=mu.dtype)
            owner_id = torch.bucketize(mu, bounds, right=True)
        else:
            owner_id = torch.zeros(n, dtype=torch.long)

        topk = min(
            max(int(getattr(config, "temporal_chunk_entry_diagnostics_topk", 12)), 0),
            int((counts > 0).sum().item()),
        )
        if topk > 0:
            # Count is the primary signal; mass breaks ties toward visible
            # cohorts.  The tiny multiplier cannot reorder different counts.
            score = counts.float() + entry_mass / entry_mass.max().clamp_min(1e-12) * 1e-3
            top_frames = torch.topk(score, k=topk).indices.tolist()
        else:
            top_frames = []

        kind = params.get("hybrid_kind")
        kind_cpu = kind.detach().reshape(-1).cpu() if kind is not None else None
        generation = params.get("relocation_generation")
        generation_cpu = (
            generation.detach().reshape(-1).cpu()
            if generation is not None
            else torch.zeros(n)
        )
        max_rows = max(
            int(getattr(config, "temporal_chunk_entry_diagnostics_max_rows", 50_000)),
            1,
        )

        rows: List[Dict[str, Any]] = []
        for frame in sorted(top_frames):
            mask = valid & (entry_frame == int(frame))
            idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if int(idx.numel()) == 0:
                continue

            owner_hist = torch.bincount(
                owner_id[idx], minlength=max(len(windows), 1),
            )
            owner_window = int(owner_hist.argmax().item())
            dense_count = int((kind_cpu[idx] == 1).sum().item()) if kind_cpu is not None else 0
            sparse_count = int((kind_cpu[idx] == 0).sum().item()) if kind_cpu is not None else 0
            relocated_count = int((generation_cpu[idx] > 0).sum().item())

            if int(idx.numel()) > max_rows:
                sample_pos = torch.linspace(
                    0, int(idx.numel()) - 1, steps=max_rows,
                ).round().to(torch.long)
                sample_idx = idx[sample_pos]
            else:
                sample_idx = idx

            sub = self._slice_rowwise_params(params, sample_idx)
            means_t, _, _ = self._deform(
                sub, float(frame) / fps, config,
            )
            disp = (
                means_t.detach().cpu() - sub["means"].detach().cpu()
            ).norm(dim=-1).float()
            scale_max = (
                sub["scales"].detach().cpu().exp().amax(dim=-1).float()
            )
            disp_p50 = float(torch.quantile(disp, 0.50).item())
            disp_p95 = float(torch.quantile(disp, 0.95).item())
            disp_max = float(disp.max().item())
            scale_p99 = float(torch.quantile(scale_max, 0.99).item())

            rows.append({
                "frame": int(frame),
                "time_s": float(frame / fps),
                "count": int(idx.numel()),
                "opacity_gamma_mass": float(entry_mass[frame].item()),
                "dense": dense_count,
                "sparse": sparse_count,
                "relocated": relocated_count,
                "dominant_owner_window": owner_window,
                "displacement_p50": disp_p50,
                "displacement_p95": disp_p95,
                "displacement_max": disp_max,
                "scale_p99": scale_p99,
                "sampled_rows": int(sample_idx.numel()),
            })

        summary: Dict[str, Any] = {
            "label": str(label),
            "cull_threshold": float(threshold),
            "total_internal_entries": int(valid_idx.numel()),
            "top_frames": rows,
        }
        log.debug(
            "temporal entry diagnostics [%s]: threshold=%.4g internal_entries=%d",
            label, threshold, int(valid_idx.numel()),
        )
        for row in rows:
            log.debug(
                "entry[%s] f=%d t=%.3fs n=%d mass=%.3f "
                "dense=%d sparse=%d relocated=%d owner=%d "
                "disp[p50/p95/max]=%.4f/%.4f/%.4f scale_p99=%.4f",
                label, row["frame"], row["time_s"], row["count"],
                row["opacity_gamma_mass"], row["dense"], row["sparse"],
                row["relocated"], row["dominant_owner_window"],
                row["displacement_p50"], row["displacement_p95"],
                row["displacement_max"], row["scale_p99"],
            )
        return summary

    @torch.no_grad()
    def _log_stats_to_tb(self, writer, step, params, psnr, t_max=1.0):
        writer.add_scalar("psnr", psnr, step)
        writer.add_scalar("n_primitives", params["means"].shape[0], step)

        nu = params["zeta_base_t"].data
        om = params["zeta_base_r"].data
        writer.add_scalar("motion/nu_base_mean", nu.norm(dim=-1).mean().item(), step)
        writer.add_scalar("motion/omega_base_mean", om.norm(dim=-1).mean().item(), step)

        c0_t, c1_t, c2_t = params["zeta_c0_t"].data, params["zeta_c1_t"].data, params["zeta_c2_t"].data
        c0_r, c1_r, c2_r = params["zeta_c0_r"].data, params["zeta_c1_r"].data, params["zeta_c2_r"].data
        writer.add_scalar("motion/ctrl_t_mean",
            (c0_t.norm(dim=-1).mean() + c1_t.norm(dim=-1).mean() + c2_t.norm(dim=-1).mean()).item() / 3, step)
        writer.add_scalar("motion/ctrl_r_mean",
            (c0_r.norm(dim=-1).mean() + c1_r.norm(dim=-1).mean() + c2_r.norm(dim=-1).mean()).item() / 3, step)
        b_t = (c0_t - 2 * c1_t + c2_t).norm(dim=-1)
        b_r = (c0_r - 2 * c1_r + c2_r).norm(dim=-1)
        writer.add_scalar("motion/bezier_accel_t_mean", b_t.mean().item(), step)
        writer.add_scalar("motion/bezier_accel_r_mean", b_r.mean().item(), step)
        writer.add_scalar("motion/accel_over_ctrl_t",
            (b_t.mean() / (c0_t.norm(dim=-1).mean() + 1e-12)).item(), step)

        anchor_disp = (params["anchors"].data - params["means"].data).norm(dim=-1)
        writer.add_scalar("anchor/disp_mean", anchor_disp.mean().item(), step)

        mu_t = params["times"].data.squeeze(-1)
        sigma_t = torch.exp(params["log_durations"].data.squeeze(-1))
        sigma_t_sample = _bounded_diagnostic_sample(sigma_t)
        writer.add_scalar("sigma_t/mean", sigma_t.mean().item(), step)
        writer.add_scalar("sigma_t/median", sigma_t_sample.median().item(), step)
        writer.add_scalar(
            "sigma_t/p99",
            torch.quantile(sigma_t_sample.float(), 0.99).item(),
            step,
        )
        writer.add_scalar("sigma_t/max", sigma_t.max().item(), step)

        # Dynamic-only duration stats. Whole-model mean/p99 are dominated by
        # static rows pinned to the full sequence and are not informative
        # about learned dynamic supports.
        if "is_static" in params:
            dyn_mask = params["is_static"].data.reshape(-1) <= 0.5
            if bool(dyn_mask.any()):
                sig_dyn = sigma_t.reshape(-1)[dyn_mask].float()
                sig_dyn_sample = _bounded_diagnostic_sample(sig_dyn)
                writer.add_scalar("sigma_t_dyn/mean", sig_dyn.mean().item(), step)
                writer.add_scalar(
                    "sigma_t_dyn/median",
                    sig_dyn_sample.median().item(),
                    step,
                )
                writer.add_scalar(
                    "sigma_t_dyn/p99",
                    torch.quantile(sig_dyn_sample, 0.99).item(),
                    step,
                )
                writer.add_scalar("sigma_t_dyn/max", sig_dyn.max().item(), step)
                if t_max and t_max > 0:
                    # sqrt(-2 ln 0.05): influence radius per sigma at the
                    # default 0.05 active threshold.
                    scale = 2.447746
                    coverage = (2.0 * scale * sig_dyn / float(t_max)).clamp(max=1.0)
                    writer.add_scalar(
                        "sigma_t_dyn/persistent_frac",
                        (coverage >= 0.5).float().mean().item(),
                        step,
                    )
        writer.add_scalar("mu_t/oob_frac",
            ((mu_t < 0) | (mu_t > t_max)).float().mean().item(), step)

        alpha = torch.sigmoid(params["opacities"].data).squeeze(-1)
        writer.add_scalar("opacity/median", alpha.median().item(), step)
        writer.add_scalar("opacity/saturated_frac", (alpha > 0.99).float().mean().item(), step)
        writer.add_scalar("opacity/dead_frac", (alpha < 0.01).float().mean().item(), step)

        # Spatial size: the longest axis per row and its ratio to the median
        # axis (the needle measure max_scale_anisotropy caps).
        log_scales = params["scales"].data
        longest = _bounded_diagnostic_sample(log_scales.max(dim=-1).values.exp().float())
        anisotropy = _bounded_diagnostic_sample(
            (log_scales.max(dim=-1).values - log_scales.median(dim=-1).values).exp().float()
        )
        writer.add_scalar("scale/max_axis_p50", longest.median().item(), step)
        writer.add_scalar("scale/max_axis_p99", torch.quantile(longest, 0.99).item(), step)
        writer.add_scalar("scale/max_axis_max", log_scales.max().exp().item(), step)
        writer.add_scalar("scale/anisotropy_p99", torch.quantile(anisotropy, 0.99).item(), step)

        with torch.no_grad():
            sig = params["log_durations"].exp().reshape(-1)
            kind = params.get("hybrid_kind", None)
            if kind is not None:
                k = kind.reshape(-1)
                for name, mask in {
                    "sparse": k == 0,
                    "dense": k == 1,
                    "static": k < 0,
                }.items():
                    if mask.any():
                        vals = sig[mask]
                        if step % 1000 == 0:
                            log.debug(
                                "duration[%s] min/med/max = %.4fs / %.4fs / %.4fs, n=%d",
                                name,
                                vals.min().item(),
                                vals.median().item(),
                                vals.max().item(),
                                int(mask.sum().item()),
                            )

    # -------------------------------------------------------------------------
    # Debug / numerical safety helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _cuda_probe(label: str) -> None:
        if os.environ.get("TRIGS_CUDA_PROBE", "0") != "1":
            return
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise RuntimeError(f"[TRIGS_CUDA_PROBE] CUDA failed after: {label}") from e

    @staticmethod
    def _static_rows(params: nn.ParameterDict) -> Optional[Tensor]:
        if "is_static" not in params:
            return None
        return (params["is_static"].data.reshape(-1) > 0.5)

    @staticmethod
    def _total_model_rows(params: nn.ParameterDict) -> int:
        """Row count of the whole model, not just a chunk's resident subset."""
        total = params.get("chunk_total_rows")
        if total is not None:
            return max(int(total.reshape(()).item()), 1)
        return max(int(params["means"].shape[0]), 1)

    @staticmethod
    def _dynamic_rows(params: nn.ParameterDict) -> Tensor:
        N = int(params["means"].shape[0])
        if "is_static" not in params:
            return torch.ones(N, dtype=torch.bool, device=params["means"].device)
        return ~(params["is_static"].data.reshape(-1) > 0.5)

    @staticmethod
    @torch.no_grad()
    def _pin_static_background_rows(
        params: nn.ParameterDict, config: TrigsTrainingConfig, n_frames: int,
    ) -> None:
        if not getattr(config, "static_background_mode", True):
            return
        if "is_static" not in params:
            return

        static = params["is_static"].data.reshape(-1) > 0.5
        if not bool(static.any()):
            return

        t_max = float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
        t_mid = 0.5 * t_max

        if float(getattr(config, "static_background_sigma_t", 0.0)) > 0.0:
            static_sigma = float(config.static_background_sigma_t)
        else:
            static_sigma = max(t_max, float(config.min_sigma_t), 1e-6)

        if "times" in params:
            params["times"].data[static] = t_mid
        if "log_durations" in params:
            params["log_durations"].data[static] = math.log(max(static_sigma, 1e-6))

        for name in (
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        ):
            if name in params:
                params[name].data[static] = 0.0

    @staticmethod
    def _zero_static_temporal_motion_grads(
        params: nn.ParameterDict, config: TrigsTrainingConfig,
    ) -> None:
        if not getattr(config, "static_background_mode", True):
            return
        if "is_static" not in params:
            return

        static = params["is_static"].data.reshape(-1) > 0.5
        if not bool(static.any()):
            return

        for name in (
            "times", "log_durations",
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        ):
            if name in params and params[name].grad is not None:
                params[name].grad[static] = 0.0

    @staticmethod
    def _zero_static_mean_grads(
        params: nn.ParameterDict, config: TrigsTrainingConfig,
    ) -> None:
        if not getattr(config, "freeze_static_means", False):
            return
        if "is_static" not in params or "means" not in params:
            return
        static = params["is_static"].data.reshape(-1) > 0.5
        if bool(static.any()) and params["means"].grad is not None:
            params["means"].grad[static] = 0.0

    @staticmethod
    @torch.no_grad()
    def _keep_bezier_timing(params: nn.ParameterDict, log_sigma_before: Tensor) -> int:
        """Re-express the Bézier controls of rows whose sigma_t changed so their
        residual keeps its timing (``bezier_keep_timing``). Returns rows changed."""
        ratio = (params["log_durations"].data - log_sigma_before).exp()
        rows = torch.nonzero((ratio - 1.0).abs().reshape(-1) > 1e-7).reshape(-1)
        if int(rows.numel()) == 0:
            return 0
        r = ratio[rows]
        for kind in ("t", "r"):
            names = [f"zeta_c{i}_{kind}" for i in range(3)]
            if not all(n in params for n in names):
                continue
            new = bezier_rewindow(*(params[n].data[rows] for n in names), r)
            for n, value in zip(names, new):
                params[n].data[rows] = value
        return int(rows.numel())

    @staticmethod
    @torch.no_grad()
    def _post_step_safety_clamp(
        params: nn.ParameterDict, config: TrigsTrainingConfig,
        n_frames: int, scene_extent: float,
    ) -> tuple[float, float, float, float]:
        t_max = float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)

        _deform_params = (
            "means", "anchors",
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        )
        for p_name in _deform_params:
            if p_name in params:
                params[p_name].data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)

        if "quats" in params:
            q = params["quats"].data
            q.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            norms = q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            degenerate = (norms < 1e-8).squeeze(-1)
            if degenerate.any():
                q[degenerate] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype)
            params["quats"].data.copy_(q / norms)

        if config.clamp_mu_t and "times" in params:
            params["times"].data.nan_to_num_(nan=0.0, posinf=t_max, neginf=0.0)
            params["times"].data.clamp_(0.0, t_max)

        sigma_min = float(config.min_sigma_t)
        if config.max_sigma_t > 0:
            sigma_max = float(config.max_sigma_t)
        elif config.max_sigma_t < 0:
            # "Unclamped" mode, bounded at the sequence length. Beyond
            # sigma = t_max, gamma is already near-flat across the clip, so
            # further growth has no representational benefit — but under
            # chunked supervision it is free (out-of-window times are never
            # rendered) and buys unsupervised persistence into other
            # windows. Sequence length is the correct ceiling; genuinely
            # persistent rows are drained by static conversion.
            t_max_clamp = (
                float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
            )
            sigma_max = max(t_max_clamp, 0.20)
        else:
            sigma_max = max(6.0 * float(config.keyframe_stride) / max(float(config.fps), 1e-6), 0.20)

        if config.clamp_log_durations and "log_durations" in params:
            lo = math.log(max(sigma_min, 1e-6))
            if math.isinf(sigma_max):
                # Generous NaN/overflow guard well above any usable duration.
                hi = math.log(1e6)
                params["log_durations"].data.nan_to_num_(nan=lo, posinf=hi, neginf=lo)
                params["log_durations"].data.clamp_(min=lo, max=hi)
            else:
                hi = math.log(max(sigma_max, sigma_min + 1e-6))
                params["log_durations"].data.nan_to_num_(nan=lo, posinf=hi, neginf=lo)
                params["log_durations"].data.clamp_(lo, hi)

        scale_min = float(config.min_scale_exp)
        if config.max_scale_exp > 0:
            scale_max = float(config.max_scale_exp)
        else:
            scale_max = max(0.25 * float(scene_extent), 0.25)

        if config.clamp_log_scales and "scales" in params:
            lo = math.log(max(scale_min, 1e-12))
            hi = math.log(max(scale_max, scale_min * 2.0))
            params["scales"].data.nan_to_num_(nan=lo, posinf=hi, neginf=lo)
            params["scales"].data.clamp_(lo, hi)
            max_anisotropy = float(
                getattr(config, "max_scale_anisotropy", 0.0)
            )
            if max_anisotropy > 1.0:
                log_scales = params["scales"].data
                median_log = log_scales.median(dim=-1, keepdim=True).values
                longest_log = median_log + math.log(max_anisotropy)
                log_scales.copy_(torch.minimum(log_scales, longest_log))

        if "camera_time_offsets" in params:
            clamp_s = float(getattr(config, "camera_time_offset_clamp_s", 0.0083))
            params["camera_time_offsets"].data.clamp_(-clamp_s, clamp_s)
            # Gauge-fix the EFFECTIVE (scale * raw) offset to zero mean, not
            # the raw parameter. With non-uniform scale a rigid shift of the
            # raw values does not zero the mean of what actually reaches the
            # renderer; solve for the shift that does:
            #   mean_c(scale_c * (raw_c - shift)) = 0
            #   shift = mean_c(scale_c * raw_c) / mean_c(scale_c)
            scale = TrigsTrainer._camera_time_offset_scale(
                params, params["camera_time_offsets"].shape[0]
            )
            denom = scale.mean().clamp_min(1e-8)
            shift = (scale * params["camera_time_offsets"].data).mean() / denom
            params["camera_time_offsets"].data.sub_(shift)
        if "cam_gain" in params:
            params["cam_gain"].data.clamp_(0.5, 2.0)
            params["cam_bias"].data.clamp_(-0.25, 0.25)

        upper_logit = config.opacity_logit_upper_clamp
        if upper_logit is not None and "opacities" in params:
            params["opacities"].data.nan_to_num_(nan=-20.0, posinf=float(upper_logit), neginf=-20.0)

        TrigsTrainer._pin_static_background_rows(params=params, config=config, n_frames=n_frames)

        return sigma_min, sigma_max, scale_min, scale_max

    # -------------------------------------------------------------------------
    # Safety checkpoint helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _checkpoint_steps_for_training(n_steps: int, progress_interval: float) -> set[int]:
        if n_steps <= 0 or progress_interval <= 0:
            return set()
        steps: set[int] = set()
        p = float(progress_interval)
        cur = p
        while cur <= 1.0 + 1e-8:
            completed = max(1, min(int(n_steps), int(round(float(n_steps) * cur))))
            steps.add(completed)
            cur += p
        return steps

    @staticmethod
    def _cpu_detached_tree(obj: Any) -> Any:
        if torch.is_tensor(obj):
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: TrigsTrainer._cpu_detached_tree(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [TrigsTrainer._cpu_detached_tree(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(TrigsTrainer._cpu_detached_tree(v) for v in obj)
        return obj

    @torch.no_grad()
    def _save_training_snapshot(
        self, *, checkpoint_dir, step_completed, params, optimizers,
        metrics, config, n_frames, sh_degree, scene_extent, reloc_state=None,
    ) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        pct = int(round(100.0 * float(step_completed) / max(float(config.n_steps), 1.0)))
        out_path = checkpoint_dir / f"trigs_step_{step_completed:06d}_pct{pct:03d}.pt"
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        params_cpu = {k: v.detach().cpu() for k, v in params.items()}
        payload: Dict[str, Any] = {
            "format": "trigs_training_snapshot",
            "step": int(step_completed),
            "step_completed": int(step_completed),
            "n_steps": int(config.n_steps),
            "progress": float(step_completed) / max(float(config.n_steps), 1.0),
            "n_frames": int(n_frames), "fps": float(config.fps),
            "sh_degree": int(sh_degree), "scene_extent": float(scene_extent),
            "params": params_cpu, "metrics": dict(metrics), "config": dict(vars(config)),
        }
        payload["temporal_params"] = {"times": params_cpu["times"], "log_durations": params_cpu["log_durations"]}
        if "camera_time_offsets" in params_cpu:
            payload["temporal_params"]["camera_time_offsets"] = params_cpu["camera_time_offsets"]
        payload["motion_params"] = {
            "anchors": params_cpu["anchors"],
            "zeta_base_t": params_cpu["zeta_base_t"], "zeta_base_r": params_cpu["zeta_base_r"],
            "zeta_c0_t": params_cpu["zeta_c0_t"], "zeta_c0_r": params_cpu["zeta_c0_r"],
            "zeta_c1_t": params_cpu["zeta_c1_t"], "zeta_c1_r": params_cpu["zeta_c1_r"],
            "zeta_c2_t": params_cpu["zeta_c2_t"], "zeta_c2_r": params_cpu["zeta_c2_r"],
        }
        if reloc_state is not None:
            payload["reloc_state"] = self._cpu_detached_tree(reloc_state)
        if config.checkpoint_include_optimizers:
            payload["optimizer_states"] = {
                name: self._cpu_detached_tree(opt.state_dict()) for name, opt in optimizers.items()
            }
        payload["rng_state"] = {
            "python": random.getstate(), "torch": torch.get_rng_state(),
            **accelerator_rng_states(),
        }
        torch.save(payload, tmp_path)
        tmp_path.replace(out_path)
        (checkpoint_dir / "latest_checkpoint.txt").write_text(str(out_path.name) + "\n")
        if config.verbose:
            print(f"\n[TRIGS_CHECKPOINT] saved {out_path}", flush=True)
        return out_path

    @staticmethod
    def _log_capacity(params, config, n_frames: int, fast_mask: Tensor, start: Tensor,
                      writer: Optional[Any], step: int, metrics: Dict[str, Any]) -> None:
        """Where relocation has moved splat capacity: share of alive dynamic rows centred on fast
        frames, and per other frame its alive rows relative to the start of training."""
        now = alive_rows_per_frame(params, n_frames, config.fps, config.relocate_tau_alpha).double()
        slow = ~fast_mask
        share_now = float(now[fast_mask].sum() / now.sum().clamp(min=1.0))
        share_start = float(start[fast_mask].sum() / start.sum().clamp(min=1.0))
        ratio = (now[slow] / start[slow].clamp(min=1.0))[start[slow] > 0]
        row = {"step": int(step), "fast_share": share_now, "fast_share_start": share_start,
               "slow_ratio_min": float(ratio.min()) if ratio.numel() else 1.0,
               "slow_ratio_p10": float(torch.quantile(ratio, 0.10)) if ratio.numel() else 1.0,
               "slow_ratio_median": float(ratio.median()) if ratio.numel() else 1.0}
        metrics.setdefault("capacity", []).append(row)
        if writer is not None:
            writer.add_scalar("capacity/fast_frames_row_share", share_now, step)
            writer.add_scalar("capacity/fast_frames_row_share_start", share_start, step)
            for key in ("slow_ratio_min", "slow_ratio_p10", "slow_ratio_median"):
                writer.add_scalar(f"capacity/{key}", row[key], step)
        log.debug("capacity @ step %d: fast-frame row share %.3f (start %.3f); other frames' rows vs start "
                  "min %.2f p10 %.2f median %.2f", step, share_now, share_start, row["slow_ratio_min"],
                  row["slow_ratio_p10"], row["slow_ratio_median"])

    @staticmethod
    def _viewer_snapshot_dir(config: "TrigsTrainingConfig") -> Optional[Path]:
        if int(getattr(config, "viewer_snapshot_every", 0)) <= 0:
            return None
        if not config.viewer_snapshot_dir:
            # Library use without an output directory: no snapshots. The CLIs default the
            # directory to <output>/viewer_snapshots (viewer_snapshot.apply_default_snapshot_dir).
            return None
        return Path(config.viewer_snapshot_dir)

    @staticmethod
    def _chunked_snapshot_extra(
        metrics: Dict[str, Any], n_rows: int, round_i: int, rounds: int,
        w_i: Optional[int], n_windows: int,
        trained_window: Optional[tuple], supervision_window: Optional[tuple],
    ) -> Dict[str, Any]:
        """Progress info shown by the viewer next to a chunked snapshot.
        ``w_i``/windows are None for the final snapshot (no single window)."""
        extra: Dict[str, Any] = {
            "mode": "chunked", "n_rows": int(n_rows),
            "round": int(round_i) + 1, "n_rounds": int(rounds),
            "n_windows": int(n_windows),
        }
        if w_i is not None:
            extra["window_idx"] = int(w_i) + 1
        if trained_window is not None:
            extra["trained_window"] = [float(trained_window[0]), float(trained_window[1])]
        if supervision_window is not None:
            extra["supervision_window"] = [
                float(supervision_window[0]), float(supervision_window[1]),
            ]
        evals = metrics.get("psnr_eval")
        if evals:
            extra["last_psnr"] = {
                k: float(v) for k, v in evals[-1].items()
                if isinstance(v, (int, float)) and k.startswith("psnr")
            }
        return extra

    @torch.no_grad()
    def _write_viewer_snapshot(
        self, *, out_dir: Path, params, step: int, config: "TrigsTrainingConfig",
        n_frames: int, sh_degree: int, sh_degree_active: int,
        extra: Dict[str, Any], writer=None,
    ) -> bool:
        """Write one live-viewer snapshot; never let a disk error kill training."""
        try:
            path, nbytes, secs = write_viewer_snapshot(
                out_dir, params=params, step=step, n_steps=int(config.n_steps),
                fps=float(config.fps), n_frames=int(n_frames),
                sh_degree=int(sh_degree), sh_degree_active=int(sh_degree_active),
                config=config, extra=extra,
                keep_last=int(config.viewer_snapshot_keep_last),
                keep_every=int(config.viewer_snapshot_keep_every),
                dtype=str(config.viewer_snapshot_dtype),
            )
        except OSError as exc:
            log.warning("viewer snapshot at step %d failed: %s", step, exc)
            return False
        if writer is not None:
            writer.add_scalar("viewer_snapshot/save_s", secs, step)
            writer.add_scalar("viewer_snapshot/bytes", nbytes, step)
        if config.verbose and config.diagnostics:
            print(
                f"\n[VIEWER_SNAPSHOT] step={step} {path.name} "
                f"{nbytes / 2**20:.1f} MiB in {secs:.2f}s",
                flush=True,
            )
        return True

    @torch.no_grad()
    def _log_fixed_active_stats(self, writer, step, params, config, t_max: float):
        if writer is None:
            return

        times = [0.0, 0.25 * t_max, 0.50 * t_max, 0.75 * t_max, t_max]
        vals = []

        for i, t in enumerate(times):
            idx = self._active_idx_for_time(params, float(t), config, step=step)
            ratio = float(idx.numel()) / max(int(params["means"].shape[0]), 1)
            vals.append(ratio)
            writer.add_scalar(f"temporal_fixed/active_ratio_{i}", ratio, step)
            writer.add_scalar(f"temporal_fixed/n_active_{i}", int(idx.numel()), step)

        writer.add_scalar("temporal_fixed/active_ratio_mean", sum(vals) / len(vals), step)

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
        frame_indices: Optional[List[int]] = None,
        frame_weights_override: Optional[Sequence[float]] = None,
        progress_bar: Optional[Any] = None,
    ) -> Dict[str, Any]:
        from ..strategy.motion_guided_relocation import (
            MotionGuidedRelocationConfig,
            MotionGuidedRelocationStrategy,
        )

        gt_images_per_frame = gt_images
        gt_alphas_per_frame = gt_alphas

        train_seed = int(getattr(config, "train_seed", -1))
        if train_seed >= 0:
            random.seed(train_seed)
            torch.manual_seed(train_seed)
            manual_seed_all(train_seed)
            if config.verbose:
                print(f"  Training RNGs seeded: train_seed={train_seed}")

        n_views = len(cameras)
        if n_views <= 0:
            raise ValueError("Expected at least one camera/view for training")

        holdout_views = sorted({int(v) for v in getattr(config, "holdout_views", ())})
        bad_holdout = [v for v in holdout_views if not (0 <= v < n_views)]
        if bad_holdout:
            raise ValueError(
                f"holdout_views {bad_holdout} out of range for n_views={n_views}"
            )
        train_views = [v for v in range(n_views) if v not in set(holdout_views)]
        if not train_views:
            raise ValueError(
                f"holdout_views {holdout_views} would withhold every camera; "
                f"leave at least one of the {n_views} views for training."
            )

        psnr_eval_pairs = _make_psnr_eval_pairs(
            n_frames=n_frames, n_views=n_views,
            num_samples=config.psnr_eval_num_samples, seed=config.psnr_eval_seed,
            views=train_views,
        )
        psnr_holdout_pairs = _make_psnr_eval_pairs(
            n_frames=n_frames, n_views=n_views,
            num_samples=config.psnr_eval_num_samples, seed=config.psnr_eval_seed,
            views=holdout_views,
        ) if holdout_views else []

        if config.verbose and psnr_eval_pairs and config.psnr_eval_every > 0:
            print(f"  PSNR eval: {len(psnr_eval_pairs)} fixed frame/view pairs "
                  f"every {config.psnr_eval_every} steps")
        if config.verbose and holdout_views:
            print(f"  Holdout views (excluded from training): {holdout_views} "
                  f"-> {len(psnr_holdout_pairs)} eval pairs, logged as psnr/holdout_*")

        batch_cameras = min(
            max(int(getattr(config, "batch_cameras", 1)), 1), len(train_views),
        )
        if config.verbose and batch_cameras > 1:
            print(f"  Camera batch: {batch_cameras} cameras per optimizer step")

        frame_weights: Optional[List[float]] = None
        fast_frame_mask: Optional[Tensor] = None
        fast_frac = float(getattr(config, "frame_sampling_fast_frac", 0.0))
        fast_visits = float(getattr(config, "frame_sampling_fast_visits", 0.0))
        if not 0.0 <= fast_frac <= 1.0:
            raise ValueError(f"frame_sampling_fast_frac must be in [0, 1], got {fast_frac}")
        if fast_visits > 0.0 and fast_frac > 0.0:
            raise ValueError("use frame_sampling_fast_visits or frame_sampling_fast_frac, not both "
                             "(set --config.frame-sampling-fast-visits 0 to use the fraction)")
        if frame_weights_override is not None:
            n_sampled = len(frame_indices) if frame_indices is not None else n_frames
            if len(frame_weights_override) != n_sampled:
                raise ValueError(f"frame_weights has {len(frame_weights_override)} entries for {n_sampled} frames")
            if fast_visits > 0.0 or fast_frac > 0.0:
                raise ValueError("frame_weights replaces frame_sampling_fast_visits/_frac; set both to 0")
            if str(getattr(config, "camera_sampling", "epoch")).lower() != "official":
                raise ValueError("frame_weights requires camera_sampling='official'")
            frame_weights, fast_frac = [float(w) for w in frame_weights_override], 1.0
        if fast_visits > 0.0 and str(getattr(config, "camera_sampling", "epoch")).lower() != "official":
            log.warning("frame_sampling_fast_visits needs camera_sampling='official'; fast frames get no top-up")
            fast_visits = 0.0
        if fast_visits > 0.0:
            sampled_frames = [int(fi) for fi in (frame_indices if frame_indices is not None else range(n_frames))]
            counts = short_lived_rows_per_frame(
                params, sampled_frames, config.fps, config.frame_sampling_fast_sigma_frames,
            )
            base_steps = int(config.n_steps)
            visits, extra_steps = fast_frame_topup(
                counts, base_steps, fast_visits, int(config.frame_sampling_fast_min_rows),
            )
            if extra_steps > 0:
                config.n_steps = base_steps + extra_steps   # schedules, relocation stop and checkpoints follow it
                frame_weights, fast_frac = visits, 1.0
                fast_frame_mask = torch.zeros(n_frames, dtype=torch.bool)
                for fi, v in zip(sampled_frames, visits):
                    if v > base_steps / len(sampled_frames) + 1e-9 and 0 <= fi < n_frames:
                        fast_frame_mask[fi] = True
                if config.verbose:
                    n_fast = int(fast_frame_mask.sum())
                    print(f"  Fast frames: {n_fast}/{len(sampled_frames)} topped up toward {fast_visits:g} visits "
                          f"(normal {base_steps / len(sampled_frames):.0f}); +{extra_steps:,} steps -> "
                          f"n_steps {base_steps:,} -> {config.n_steps:,}")
                log.debug("fast-frame expected visits: %s", {f: round(v) for f, v in zip(sampled_frames, visits)})
            elif config.verbose:
                print(f"  Fast frames: none need a top-up to {fast_visits:g} visits; n_steps unchanged")
        elif fast_frac > 0.0 and frame_weights_override is None:
            if str(getattr(config, "camera_sampling", "epoch")).lower() != "official":
                raise ValueError("frame_sampling_fast_frac > 0 requires camera_sampling='official'")
            sampled_frames = [int(fi) for fi in (frame_indices if frame_indices is not None else range(n_frames))]
            counts = short_lived_rows_per_frame(
                params, sampled_frames, config.fps, config.frame_sampling_fast_sigma_frames,
            )
            total = float(sum(counts))
            if total <= 0.0:
                log.warning("frame_sampling_fast_frac: no rows with sigma_t <= %g frames; frame choice stays uniform",
                            config.frame_sampling_fast_sigma_frames)
            else:
                frame_weights = counts
                fast_frame_mask = torch.zeros(n_frames, dtype=torch.bool)
                for fi, c in zip(sampled_frames, counts):
                    if c > 0 and 0 <= fi < n_frames:
                        fast_frame_mask[fi] = True
                n = len(sampled_frames)
                visits = [config.n_steps * ((1.0 - fast_frac) / n + fast_frac * c / total) for c in counts]
                if config.verbose:
                    print(f"  Frame sampling: {fast_frac:.0%} of steps weighted by short-lived rows; "
                          f"{sum(c > 0 for c in counts)}/{n} frames have them; expected visits per frame "
                          f"{min(visits):.0f}..{max(visits):.0f} (uniform {config.n_steps / n:.0f})")
                log.debug("frame sampling expected visits: %s",
                          {f: round(v) for f, v in zip(sampled_frames, visits)})

        # Capacity watch (weighted frame choice only): alive dynamic rows by the frame of their mu_t,
        # so relocation moving splats toward the fast frames shows up in TensorBoard (capacity/*).
        capacity_start = None
        if fast_frame_mask is not None and bool(fast_frame_mask.any()) and bool((~fast_frame_mask).any()):
            capacity_start = alive_rows_per_frame(params, n_frames, config.fps, config.relocate_tau_alpha).double()

        epoch_iter = iter([])
        metrics: Dict[str, Any] = {
            "n_initial": params["means"].shape[0],
            "snapshots": [], "prune_events": [],
        }
        if progress_bar is not None:
            # Chunked training: one bar for the whole run, owned by train_chunked.
            iterator = _ticking_range(config.n_steps, progress_bar)
            bar = progress_bar
        else:
            iterator = trange(config.n_steps, desc="TRiGS") if config.verbose else range(config.n_steps)
            bar = iterator

        viewer_dir = self._viewer_snapshot_dir(config)
        last_viewer_step = 0
        if viewer_dir is not None and config.verbose:
            print(f"  Viewer snapshots: {viewer_dir} every {config.viewer_snapshot_every} steps")

        checkpoint_dir: Optional[Path] = None
        checkpoint_steps: set[int] = set()
        if config.checkpoint_dir:
            checkpoint_dir = Path(config.checkpoint_dir)
            checkpoint_steps = self._checkpoint_steps_for_training(
                n_steps=config.n_steps, progress_interval=config.checkpoint_progress_interval,
            )
            until = int(getattr(config, "checkpoint_until_step", 0))
            if until > 0:
                checkpoint_steps = {s for s in checkpoint_steps if s <= until}
            if config.verbose and checkpoint_steps:
                print(f"  Safety checkpoints enabled: {checkpoint_dir} at completed steps "
                      f"{sorted(checkpoint_steps)}")

        decay_ratios = _lr_decay_ratios(config)
        initial_lrs: Dict[str, float] = {
            name: opt.param_groups[0]["lr"]
            for name, opt in optimizers.items()
            if name in decay_ratios
        }

        temporal_names = ["opacities", "times", "log_durations"]
        if "camera_time_offsets" in params:
            temporal_names.append("camera_time_offsets")
        temporal_frozen = config.unfreeze_temporal_step > 0
        if temporal_frozen:
            for name in temporal_names:
                params[name].requires_grad_(False)
            if config.verbose:
                print(f"  Temporal params frozen until step {config.unfreeze_temporal_step}")

        relocation_strategy = MotionGuidedRelocationStrategy(
            MotionGuidedRelocationConfig(
                relocate=config.relocate,
                relocate_every=config.relocate_every,
                relocate_start_iter=config.relocate_start_iter,
                relocate_tau_alpha=config.relocate_tau_alpha,
                relocate_scale_factor=config.relocate_scale_factor,
                relocate_dynamic_only=config.relocate_dynamic_only,
                relocate_mcmc_split=getattr(
                    config, "relocate_mcmc_split", True,
                ),
                relocate_source_duration_divisor=getattr(
                    config, "relocate_source_duration_divisor", 1.5,
                ),
                relocate_source_time_jitter_frac=getattr(
                    config, "relocate_source_time_jitter_frac", 0.25,
                ),
                relocate_reanchor_source=getattr(
                    config, "relocate_reanchor_source", True,
                ),
                relocate_time_max=(
                    float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
                ),
                relocate_time_jitter_sigma=getattr(
                    config, "relocate_time_jitter_sigma", 0.0,
                ),
                relocate_recenter_residual=getattr(
                    config, "relocate_recenter_residual", True,
                ),
                relocate_time_cue_per_visit=getattr(
                    config, "relocate_time_cue_per_visit", True,
                ),
                relocate_speed_cue_clip_edges=getattr(
                    config, "relocate_speed_cue_clip_edges", True,
                ),
            )
        )
        reloc_state = relocation_strategy.initialize_state(
            n_primitives=params["means"].shape[0], device=params["means"].device,
        )

        def deform_rows(rows: Dict[str, Tensor], t: Tensor) -> tuple[Tensor, Tensor]:
            means_t, quats_t, _ = self._deform(rows, t, config)
            return means_t, quats_t
        rigid_knn_dirty = False

        def _refresh_rigid_knn_for_current_rows() -> tuple[Optional[Tensor], Tensor]:
            dyn_rows = self._dynamic_rows(params)
            if (
                getattr(config, "static_exclude_from_motion_losses", True)
                and "is_static" in params
            ):
                n_dyn = int(dyn_rows.sum().item())
                if n_dyn <= max(int(config.knn_k), 1):
                    return None, dyn_rows
                rigid_time_weight_cur = self._rigid_time_weight_value(config, scene_extent, n_frames)
                nn_dyn = self._compute_knn(
                    self._rigid_knn_points(params, config).detach()[dyn_rows],
                    config.knn_k,
                    times=params["times"].detach()[dyn_rows], time_weight=rigid_time_weight_cur,
                )
                return nn_dyn, dyn_rows
            return nn_indices, dyn_rows

        nn_indices, rigid_loss_rows = _refresh_rigid_knn_for_current_rows()

        if config.diagnostics and nn_indices is not None:
            with torch.no_grad():
                rigid_time_weight_dbg = self._rigid_time_weight_value(
                    config, scene_extent, n_frames
                )

                t_dbg = params["times"].detach().reshape(-1)
                if torch.is_tensor(rigid_loss_rows) and rigid_loss_rows.dtype == torch.bool:
                    t_dbg = t_dbg[rigid_loss_rows]

                # This is logging only. Constructing every neighbor delta can
                # allocate tens of millions of values before quantile() even
                # runs, so sample rows first.
                debug_row_count = min(int(t_dbg.numel()), 250_000)
                if int(t_dbg.numel()) > debug_row_count:
                    row_idx = torch.div(
                        torch.arange(
                            debug_row_count,
                            device=t_dbg.device,
                            dtype=torch.int64,
                        ) * (int(t_dbg.numel()) - 1),
                        debug_row_count - 1,
                        rounding_mode="floor",
                    )
                    nn_dbg = nn_indices[row_idx]
                    dt_dbg = (
                        t_dbg[row_idx, None] - t_dbg[nn_dbg]
                    ).abs().reshape(-1).float()
                else:
                    dt_dbg = (
                        t_dbg[:, None] - t_dbg[nn_indices]
                    ).abs().reshape(-1).float()

                print(
                    f"  rigid kNN time_weight={rigid_time_weight_dbg:.6g} "
                    f"user_weight={config.rigid_knn_time_weight:.3g} "
                    f"dt[p50/p90/p99/max]="
                    f"{torch.quantile(dt_dbg, 0.50).item():.4f}/"
                    f"{torch.quantile(dt_dbg, 0.90).item():.4f}/"
                    f"{torch.quantile(dt_dbg, 0.99).item():.4f}/"
                    f"{dt_dbg.max().item():.4f}s "
                    f"frac>0.5s={(dt_dbg > 0.5).float().mean().item():.3f} "
                    f"frac>1s={(dt_dbg > 1.0).float().mean().item():.3f}",
                    flush=True,
                )

        if "shN" in params:
            n_bands = params["shN"].shape[1] + 1
            max_sh_degree_runtime = int(math.isqrt(n_bands) - 1)
        else:
            max_sh_degree_runtime = 0

        if config.verbose:
            print(f"  SH warmup: 0 → {max_sh_degree_runtime} over {max_sh_degree_runtime * 1000} steps")

        t_max = float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)

        opdiag_every = int(getattr(config, "opacity_diag_every", 0))
        opdiag: Optional[Dict[str, Tensor]] = None
        for step in iterator:
            sh_degree = _sh_degree_for_step(config, step, max_sh_degree_runtime)

            if temporal_frozen and step == config.unfreeze_temporal_step:
                for name in temporal_names:
                    params[name].requires_grad_(True)
                for name in temporal_names:
                    if name in optimizers:
                        for pg in optimizers[name].param_groups:
                            for p in pg["params"]:
                                if p in optimizers[name].state:
                                    optimizers[name].state[p].clear()
                temporal_frozen = False
                if config.verbose:
                    print(f"  Step {step}: unfreezing temporal params")

            progress = _lr_schedule_progress(config, step)
            for name, ratio in decay_ratios.items():
                if name not in optimizers:
                    continue
                lr = initial_lrs[name] * (ratio ** progress)
                for pg in optimizers[name].param_groups:
                    pg["lr"] = lr

            sampling = str(getattr(config, "camera_sampling", "epoch")).lower()
            if sampling not in ("epoch", "official"):
                raise ValueError(f"camera_sampling must be 'epoch' or 'official', got {sampling!r}")
            if sampling == "official":
                frame_idx, cam_group = _draw_official_camera_batch(
                    frame_indices if frame_indices is not None else range(n_frames),
                    train_views,
                    batch_cameras,
                    frame_weights=frame_weights,
                    weighted_frac=fast_frac,
                )
            else:
                try:
                    frame_idx, cam_group = next(epoch_iter)
                except StopIteration:
                    epoch_iter = iter(_build_camera_batch_epoch(
                        frame_indices if frame_indices is not None else range(n_frames),
                        train_views,
                        batch_cameras,
                    ))
                    frame_idx, cam_group = next(epoch_iter)

            offset_active = (
                "camera_time_offsets" in params
                and step >= config.camera_time_offset_warmup_steps
            )

            step_completed = int(step) + 1
            if checkpoint_dir is not None and step_completed in checkpoint_steps:
                snap_path = self._save_training_snapshot(
                    checkpoint_dir=checkpoint_dir, step_completed=step_completed,
                    params=params, optimizers=optimizers, metrics=metrics,
                    config=config, n_frames=n_frames, sh_degree=sh_degree,
                    scene_extent=scene_extent, reloc_state=reloc_state,
                )
                metrics["snapshots"].append({"step": step_completed, "path": str(snap_path)})

            delayed_target = config.delayed_prune_target_count
            delayed_pruning_active = (
                delayed_target is not None and delayed_target > 0
                and params["means"].shape[0] > delayed_target
            )

            # Each camera in the group renders at its own capture time (sync
            # + learned offset + rolling shutter) and backpropagates its
            # 1/B share of the photometric loss immediately, so peak memory
            # stays at one camera's graph. Regularizers are added once below.
            n_group = len(cam_group)
            rgb_loss_sum = 0.0
            alpha_outside_sum = 0.0
            alpha_mask_sum = 0.0
            gamma = None
            active_parts: List[Tensor] = []
            for group_i, cam_idx in enumerate(cam_group):
                if gt_times_per_frame is not None and gt_times_per_frame[frame_idx][cam_idx] is not None:
                    t_base = gt_times_per_frame[frame_idx][cam_idx]
                else:
                    t_base = float(frame_idx) / max(config.fps, 1e-6)

                if offset_active:
                    scale = self._camera_time_offset_scale(
                        params, params["camera_time_offsets"].shape[0]
                    )
                    t = torch.tensor(t_base, device=self.device, dtype=torch.float32) \
                        + scale[cam_idx] * params["camera_time_offsets"][cam_idx]
                else:
                    t = t_base

                self._cuda_probe(f"step={step} before render frame={frame_idx} cam={cam_idx}")

                render_colors, render_alphas, cam_gamma, cam_info = self._render(
                    params, cameras, cam_idx, t, sh_degree, config, step=step,
                )
                device_synchronize()

                self._cuda_probe(f"step={step} after render frame={frame_idx} cam={cam_idx}")

                cam_gt_rgb = gt_images[frame_idx][cam_idx].to(self.device).float().div(255.0)

                self._cuda_probe(f"step={step} after gt_rgb copy frame={frame_idx} cam={cam_idx}")

                if tuple(render_colors[0].shape) != tuple(cam_gt_rgb.shape):
                    raise RuntimeError(
                        f"Render/GT shape mismatch before loss. frame={frame_idx}, cam={cam_idx}, "
                        f"render={tuple(render_colors[0].shape)}, gt={tuple(cam_gt_rgb.shape)}"
                    )

                cam_rgb, cam_alpha_out, cam_alpha_mask = self._photometric_losses(
                    params, render_colors, render_alphas, cam_gt_rgb,
                    gt_alphas_per_frame, frame_idx, cam_idx, config,
                )
                cam_loss = (
                    config.lambda_render * cam_rgb
                    + config.lambda_alpha_outside_mask * cam_alpha_out
                    + config.lambda_alpha_mask * cam_alpha_mask
                )
                if cam_loss.requires_grad:
                    (cam_loss / n_group).backward()

                relocation_strategy.accumulate_2d(
                    reloc_state, cam_info, batch_size=n_group,
                )
                rgb_loss_sum += float(cam_rgb.detach())
                alpha_outside_sum += float(cam_alpha_out.detach())
                alpha_mask_sum += float(cam_alpha_mask.detach())
                gamma = cam_gamma if gamma is None else gamma + cam_gamma
                active_parts.append(cam_info["active_idx"])

                if group_i == 0:
                    # The first camera stands in for the step in image logs.
                    cam_idx_log = cam_idx
                    render_log = render_colors[0].detach()
                    gt_rgb = cam_gt_rgb
                    info = {
                        "n_active": cam_info["n_active"],
                        "n_total": cam_info["n_total"],
                    }
                del render_colors, render_alphas, cam_info, cam_loss

            gamma = gamma / float(n_group)
            rgb_loss = torch.tensor(rgb_loss_sum / n_group)
            alpha_outside_loss = torch.tensor(alpha_outside_sum / n_group)
            alpha_mask_loss = torch.tensor(alpha_mask_sum / n_group)
            photometric_total = (
                config.lambda_render * rgb_loss
                + config.lambda_alpha_outside_mask * alpha_outside_loss
                + config.lambda_alpha_mask * alpha_mask_loss
            )
            step_active_idx = (
                active_parts[0] if n_group == 1
                else torch.unique(torch.cat(active_parts))
            )
            cam_idx = cam_idx_log

            c0_full = torch.cat([params["zeta_c0_t"], params["zeta_c0_r"]], dim=-1)
            c1_full = torch.cat([params["zeta_c1_t"], params["zeta_c1_r"]], dim=-1)
            c2_full = torch.cat([params["zeta_c2_t"], params["zeta_c2_r"]], dim=-1)

            if getattr(config, "static_exclude_from_motion_losses", True) and "is_static" in params:
                dyn_rows_for_motion = self._dynamic_rows(params)
                if bool(dyn_rows_for_motion.any()):
                    motion_loss = bezier_motion_loss(
                        c0_full[dyn_rows_for_motion], c1_full[dyn_rows_for_motion],
                        c2_full[dyn_rows_for_motion],
                    )
                else:
                    motion_loss = params["means"].new_zeros(())
            else:
                motion_loss = bezier_motion_loss(c0_full, c1_full, c2_full)

            lambda_res = float(getattr(config, "lambda_motion_residual", 0.0))
            if lambda_res > 0.0:
                if getattr(config, "static_exclude_from_motion_losses", True) and "is_static" in params:
                    rows_res = self._dynamic_rows(params)
                else:
                    rows_res = slice(None)
                motion_residual_loss = _motion_residual_magnitude_loss(
                    c0_full,
                    c1_full,
                    c2_full,
                    rows_res,
                )
            else:
                motion_residual_loss = params["means"].new_zeros(())

            if delayed_pruning_active:
                rigid_loss = params["means"].new_zeros(())
            else:
                if (
                    rigid_knn_dirty
                    or (step > 0 and config.knn_recompute_every > 0
                        and step % config.knn_recompute_every == 0)
                ):
                    nn_indices, rigid_loss_rows = _refresh_rigid_knn_for_current_rows()
                    rigid_knn_dirty = False

                zeta_base_full = torch.cat([params["zeta_base_t"], params["zeta_base_r"]], dim=-1)
                rigid_color_detach = bool(
                    getattr(config, "rigid_color_weight_detach", False)
                )

                if nn_indices is None:
                    rigid_loss = params["means"].new_zeros(())
                elif getattr(config, "static_exclude_from_motion_losses", True) and "is_static" in params:
                    rigid_loss = rigid_coherence_loss(
                        zeta_base_full[rigid_loss_rows], params["sh0"][rigid_loss_rows],
                        nn_indices, lambda_c=config.rigid_color_lambda,
                        detach_color_weight=rigid_color_detach,
                    )
                else:
                    rigid_loss = rigid_coherence_loss(
                        zeta_base_full, params["sh0"], nn_indices,
                        lambda_c=config.rigid_color_lambda,
                        detach_color_weight=rigid_color_detach,
                    )

            base_opa = torch.sigmoid(params["opacities"])
            if base_opa.dim() == 1:
                base_opa = base_opa.unsqueeze(-1)
            if getattr(config, "static_exclude_from_temporal_reg", True) and "is_static" in params:
                dyn_rows_for_reg = self._dynamic_rows(params)
                if bool(dyn_rows_for_reg.any()):
                    reg_loss = (base_opa[dyn_rows_for_reg] * gamma[dyn_rows_for_reg].detach()).mean()
                else:
                    reg_loss = params["means"].new_zeros(())
            else:
                reg_loss = (base_opa * gamma.detach()).mean()

            sigma_t = torch.exp(params["log_durations"])
            if getattr(config, "static_exclude_from_temporal_reg", True) and "is_static" in params:
                dyn_rows_for_sigma = self._dynamic_rows(params)
                if bool(dyn_rows_for_sigma.any()):
                    sigma_floor_loss = torch.clamp(
                        sigma_floor_target - sigma_t[dyn_rows_for_sigma], min=0.0,
                    ).pow(2).mean()
                else:
                    sigma_floor_loss = params["means"].new_zeros(())
            else:
                sigma_floor_loss = torch.clamp(sigma_floor_target - sigma_t, min=0.0).pow(2).mean()

            lambda_scale = float(getattr(config, "lambda_scale_reg", 0.0))
            if lambda_scale > 0.0:
                if getattr(config, "scale_reg_include_static", True):
                    scale_rows = slice(None)
                else:
                    scale_rows = self._dynamic_rows(params)
                scale_reg_loss = scale_size_loss(
                    params["scales"],
                    self._total_model_rows(params),
                    scale_rows,
                )
            else:
                scale_reg_loss = params["means"].new_zeros(())

            if "camera_time_offsets" in params and offset_active:
                # Pull low-observability cameras (small scale) toward the
                # prior harder than high-observability ones: Adam's per-
                # parameter step norm does not on its own shrink to zero for
                # a persistently-tiny-but-nonzero gradient (the eps floor),
                # which is what let a signal-free camera random-walk to the
                # clamp. weight = 1/scale, clipped and mean-normalized so the
                # all-ones (no observability.json) case reproduces the old
                # unweighted regularizer exactly.
                scale = self._camera_time_offset_scale(
                    params, params["camera_time_offsets"].shape[0]
                )
                reg_weight = (1.0 / scale.clamp_min(0.02)).clamp_max(50.0)
                reg_weight = reg_weight / reg_weight.mean().clamp_min(1e-8)
                time_offset_reg_loss = (
                    reg_weight * params["camera_time_offsets"].pow(2)
                ).mean()
            else:
                time_offset_reg_loss = params["means"].new_zeros(())

            if "cam_gain" in params:
                appearance_reg_loss = (
                    (params["cam_gain"] - 1.0).pow(2).mean() + params["cam_bias"].pow(2).mean()
                )
            else:
                appearance_reg_loss = params["means"].new_zeros(())

            chunk_outside_mass_loss = self._temporal_chunk_outside_mass_loss(
                params, config,
            )

            reg_total = (
                config.lambda_motion * motion_loss
                + lambda_res * motion_residual_loss
                + config.lambda_rigid * rigid_loss
                + config.lambda_reg * reg_loss
                + lambda_scale * scale_reg_loss
                + config.lambda_sigma_floor * sigma_floor_loss
                + getattr(config, "lambda_camera_time_offset_reg", 0.01) * time_offset_reg_loss
                + getattr(config, "lambda_camera_appearance_reg", 1e-2) * appearance_reg_loss
                + getattr(config, "lambda_temporal_chunk_outside_mass", 0.0)
                * chunk_outside_mass_loss
            )
            total_loss = reg_total.detach() + photometric_total.to(reg_total.device)

            if writer is not None and config.tb_log_every > 0 and step % config.tb_log_every == 0:
                writer.add_scalar("loss/total", total_loss.item(), step)
                writer.add_scalar("loss/rgb", rgb_loss.item(), step)
                writer.add_scalar("loss/motion", motion_loss.item(), step)
                writer.add_scalar("loss/motion_residual", motion_residual_loss.item(), step)
                writer.add_scalar("loss/rigid", rigid_loss.item(), step)
                writer.add_scalar("loss/reg", reg_loss.item(), step)
                writer.add_scalar("loss/scale_reg", scale_reg_loss.item(), step)
                writer.add_scalar("loss/sigma_floor", sigma_floor_loss.item(), step)
                writer.add_scalar("loss/alpha_outside_mask", alpha_outside_loss.item(), step)
                writer.add_scalar("loss/alpha_mask", alpha_mask_loss.item(), step)
                writer.add_scalar(
                    "loss/temporal_chunk_outside_mass",
                    chunk_outside_mass_loss.item(),
                    step,
                )
                for name, opt in optimizers.items():
                    if name in decay_ratios:
                        writer.add_scalar(f"lr/{name}", opt.param_groups[0]["lr"], step)
                if "camera_time_offsets" in params:
                    writer.add_scalar("loss/time_offset_reg", time_offset_reg_loss.item(), step)
                    off_ms = params["camera_time_offsets"].detach() * 1000.0
                    writer.add_scalar("motion/time_offset_max_ms",
                                      off_ms.abs().max().item(), step)
                    # Residuals sit on top of the sync.json baseline already in
                    # gt_times; large/std-drifting residuals mean sync is stale
                    # or the offsets are absorbing motion-model error.
                    writer.add_scalar("motion/time_offset_std_ms",
                                      off_ms.std().item(), step)
                    writer.add_histogram("motion/time_offset_ms_hist", off_ms, step)

            # Opacity diagnostics: read the photo loss's opacity gradient before
            # the regularisers add theirs.
            opdiag_photo = None
            if opdiag_every > 0:
                g_op = params["opacities"].grad
                opdiag_photo = (
                    g_op.detach().reshape(-1).clone() if g_op is not None
                    else torch.zeros_like(params["opacities"].data.reshape(-1))
                )

            # Separate from the per-camera backward above, so it can be a
            # constant (e.g. an all-static model with no size penalty).
            if reg_total.requires_grad:
                reg_total.backward()
            self._cuda_probe(f"step={step} after backward")

            self._zero_static_temporal_motion_grads(
                params,
                config,
            )
            self._zero_static_mean_grads(params, config)
            self._zero_frozen_chunk_grads(params)

            # Snapshot mu_t / sigma_t so the hierarchy can re-file every row
            # the step moved, including momentum-only moves of culled rows.
            temporal_before = (
                self._temporal_snapshot(params)
                if config.temporal_hierarchy_enabled else None
            )

            relocation_strategy.accumulate_time(reloc_state, params, step_active_idx)
            self._cuda_probe(
                f"step={step} after relocation accumulate_grads"
            )

            log_sigma_before = (
                params["log_durations"].data.clone()
                if config.bezier_keep_timing and "log_durations" in params else None
            )

            opdiag_total = opdiag_logit_before = None
            if opdiag_photo is not None:
                g_op = params["opacities"].grad
                opdiag_total = (
                    g_op.detach().reshape(-1).clone() if g_op is not None
                    else torch.zeros_like(opdiag_photo)
                )
                opdiag_logit_before = params["opacities"].data.reshape(-1).clone()

            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)

            self._cuda_probe(
                f"step={step} after optimizer step"
            )

            sigma_min, sigma_max, scale_min, scale_max = (
                self._post_step_safety_clamp(
                    params=params,
                    config=config,
                    n_frames=n_frames,
                    scene_extent=scene_extent,
                )
            )
            if log_sigma_before is not None:
                self._keep_bezier_timing(params, log_sigma_before)

            if temporal_before is not None:
                self._update_temporal_hierarchy_drift(
                    params=params,
                    changed_idx=self._temporal_rows_changed(params, temporal_before),
                    config=config,
                    step=step,
                )

            self._cuda_probe(
                f"step={step} after post-step safety clamp"
            )

            upper_logit = config.opacity_logit_upper_clamp
            if upper_logit is not None:
                until_step = config.opacity_logit_upper_clamp_until_step
                if until_step < 0 or step <= until_step:
                    params["opacities"].data.clamp_(max=upper_logit)

            if opdiag_photo is not None:
                with torch.no_grad():
                    n_op = int(opdiag_photo.shape[0])
                    if opdiag is None or int(opdiag["photo"].shape[0]) != n_op:
                        opdiag = {
                            key: torch.zeros(n_op, device=opdiag_photo.device)
                            for key in ("photo", "reg", "delta", "photo_neg", "photo_nz")
                        }
                    opdiag["photo"] += opdiag_photo
                    opdiag["reg"] += opdiag_total - opdiag_photo
                    opdiag["delta"] += params["opacities"].data.reshape(-1) - opdiag_logit_before
                    opdiag["photo_neg"] += (opdiag_photo < 0).float()
                    opdiag["photo_nz"] += (opdiag_photo != 0).float()
                    if step % opdiag_every == 0:
                        _log_opacity_diag(
                            opacity_diag_summary(
                                sigma_frames=params["log_durations"].detach().exp().reshape(-1)
                                * float(config.fps),
                                is_static=(
                                    params["is_static"].detach().reshape(-1) > 0.5
                                    if "is_static" in params
                                    else torch.zeros(n_op, dtype=torch.bool, device=opdiag_photo.device)
                                ),
                                opacity=torch.sigmoid(params["opacities"].detach()).reshape(-1),
                                **opdiag,
                            ),
                            writer,
                            step,
                        )
                        opdiag = None

            if (
                delayed_pruning_active
                and step >= config.delayed_prune_start_iter
                and config.delayed_prune_every > 0
                and step % config.delayed_prune_every == 0
            ):
                drop_mask = self._select_drop_mask_by_inactive(
                    params=params, target_count=delayed_target,
                    tau_alpha=config.relocate_tau_alpha,
                )
                n_before = int(params["means"].shape[0])
                keep_idx = self._apply_delayed_prune(params=params, optimizers=optimizers,
                                                      drop_mask=drop_mask)
                if keep_idx is not None:
                    reloc_state["grad_2d_accum"] = reloc_state["grad_2d_accum"][keep_idx].contiguous()
                    reloc_state["grad_t_accum"] = reloc_state["grad_t_accum"][keep_idx].contiguous()
                    for key in ("visit_accum", "visible_accum"):
                        if key in reloc_state:
                            reloc_state[key] = reloc_state[key][keep_idx].contiguous()
                    n_after = int(params["means"].shape[0])
                    rigid_knn_dirty = True
                    self._invalidate_temporal_hierarchy()

                    metrics["prune_events"].append({"step": int(step), "before": n_before, "after": n_after})
                    if config.verbose:
                        print(f"  Delayed prune @ step {step}: {n_before:,} → {n_after:,} "
                              f"(inactive α<{config.relocate_tau_alpha:g})")

            delayed_pruning_active = (
                delayed_target is not None and delayed_target > 0
                and int(params["means"].shape[0]) > int(delayed_target)
            )

            n_reloc = 0
            settle = int(getattr(config, "relocate_settle_steps", 0))
            in_settle_window = settle > 0 and step >= int(config.n_steps) - settle
            past_reloc_stop = _relocation_past_stop(config, step)
            if not delayed_pruning_active and not in_settle_window:
                self._cuda_probe(f"step={step} before relocation")

                if not past_reloc_stop:
                    n_reloc = relocation_strategy.step_post_backward(
                        params=params, optimizers=optimizers, state=reloc_state,
                        step=step, deform_fn=deform_rows,
                    )

                if n_reloc > 0:
                    self._invalidate_temporal_hierarchy()
                    log.debug("relocation @ step %d: relocated %d primitives", step, n_reloc)
                    if capacity_start is not None:
                        self._log_capacity(params, config, n_frames, fast_frame_mask, capacity_start,
                                           writer, step, metrics)
                    if config.diagnostics:
                        with torch.no_grad():
                            alpha = torch.sigmoid(params["opacities"]).reshape(-1)
                            tau = float(config.relocate_tau_alpha)

                            dead = alpha < tau
                            dead_01 = alpha < 0.01
                            dead_02 = alpha < 0.02
                            dead_05 = alpha < 0.05

                            if "is_static" in params:
                                dyn = params["is_static"].reshape(-1) <= 0.5
                            else:
                                dyn = torch.ones_like(dead, dtype=torch.bool)

                            log.debug(
                                "reloc diagnostics @ step %d: "
                                "tau=%.4g dead<tau=%d dead_dyn<tau=%d "
                                "dead<0.01=%d dead<0.02=%d dead<0.05=%d "
                                "alpha min/med/p01=%.5f/%.5f/%.5f",
                                step,
                                tau,
                                int(dead.sum().item()),
                                int((dead & dyn).sum().item()),
                                int(dead_01.sum().item()),
                                int(dead_02.sum().item()),
                                int(dead_05.sum().item()),
                                float(alpha.min().item()),
                                float(alpha.median().item()),
                                float(torch.quantile(
                                    _bounded_diagnostic_sample(alpha.float()),
                                    0.01,
                                ).item()),
                            )
                elif (
                    not past_reloc_stop
                    and step >= config.relocate_start_iter
                    and config.relocate_every > 0
                    and step % config.relocate_every == 0
                ):
                    log.debug("relocation @ step %d: triggered, relocated 0 primitives", step)

                self._cuda_probe(f"step={step} after relocation n_reloc={n_reloc}")

                if n_reloc:
                    # Step 4: enforce class invariants after every relocation pass.
                    _enforce_relocation_class_invariants(params)
                    rigid_knn_dirty = True

                    device_synchronize()

                    if (
                        os.environ.get("TRIGS_RELOC_LOG", "0") == "1"
                        and (step % 500 == 0 or step >= 5900)
                    ):
                        with torch.no_grad():
                            alpha = torch.sigmoid(params["opacities"].detach()).reshape(-1)
                            sigma_t_val = torch.exp(params["log_durations"].detach()).reshape(-1)
                            scale_exp = torch.exp(params["scales"].detach()).reshape(-1)
                            nu = params["zeta_base_t"].detach().norm(dim=-1)
                            print(
                                f"\n[TRIGS_RELOC] step={step} n={n_reloc} "
                                f"alpha[min/med/max]={alpha.min().item():.3e}/"
                                f"{alpha.median().item():.3e}/{alpha.max().item():.3e} "
                                f"sigma[min/med/max]={sigma_t_val.min().item():.3e}/"
                                f"{sigma_t_val.median().item():.3e}/{sigma_t_val.max().item():.3e} "
                                f"scale_exp[min/med/max]={scale_exp.min().item():.3e}/"
                                f"{scale_exp.median().item():.3e}/{scale_exp.max().item():.3e} "
                                f"nu[max]={nu.max().item():.3e} "
                                f"sigma_clamp=[{sigma_min:.3e},{sigma_max:.3e}] "
                                f"scale_clamp=[{scale_min:.3e},{scale_max:.3e}]", flush=True,
                            )

                    if os.environ.get("TRIGS_RELOC_DEBUG", "0") == "1":
                        for _name, _tensor in params.items():
                            if torch.is_tensor(_tensor) and not torch.isfinite(_tensor.data).all():
                                bad = (~torch.isfinite(_tensor.data)).sum().item()
                                raise RuntimeError(
                                    f"[TRIGS_RELOC_DEBUG] non-finite param after relocation: "
                                    f"{_name}, bad={bad}, step={step}"
                                )

                if (
                    config.static_convert
                    and step >= config.static_convert_start_iter
                    # opacities / μₜ / log σₜ are frozen for the motion-only
                    # phase, so before this step the alpha gate passes for every
                    # row (all still at their init value) and σₜ carries no
                    # information. Converting here would freeze rows static on
                    # untrained temporal params, irreversibly.
                    and step >= int(config.unfreeze_temporal_step)
                    and config.static_convert_every > 0
                    and step % config.static_convert_every == 0
                ):
                    n_static_new = self._convert_temporally_invariant_to_static(
                        params, config, n_frames,
                    )
                    if n_static_new:
                        self._invalidate_temporal_hierarchy()
                        if config.verbose:
                            total_static = int((params["is_static"].data > 0.5).sum().item())
                            rigid_knn_dirty = True
                            print(
                                f"  Static convert @ step {step}: +{n_static_new:,} "
                                f"(total static={total_static:,})"
                            )

            # Re-pin static rows and enforce class invariants as a final hard clamp
            # after relocation/static-convert.
            self._pin_static_background_rows(params, config, n_frames)
            if _enforce_relocation_class_invariants(params) > 0:
                # Demoted rows are now active at every t but still filed under
                # their old dynamic segment; the hierarchy must be rebuilt.
                self._invalidate_temporal_hierarchy()

            # End of the step body: this step's update, clamps, relocation and
            # static re-pin have all run, so step + 1 updates are in `params`.
            if viewer_dir is not None and should_snapshot(
                step + 1, config.viewer_snapshot_every, last_viewer_step,
                final=(step + 1 == int(config.n_steps)),
            ):
                if self._write_viewer_snapshot(
                    out_dir=viewer_dir, params=params, step=step + 1, config=config,
                    n_frames=n_frames, sh_degree=max_sh_degree_runtime,
                    sh_degree_active=sh_degree, writer=writer,
                    extra={"mode": "unchunked", "n_rows": int(params["means"].shape[0])},
                ):
                    last_viewer_step = step + 1

            current_mask = None
            if gt_alphas_per_frame is not None:
                try:
                    current_mask = gt_alphas_per_frame[frame_idx][cam_idx].to(self.device)
                except Exception:
                    current_mask = None

            if config.verbose and step % 500 == 0:
                psnr_metrics = compute_psnr_metrics(
                    render_log, gt_rgb, current_mask,
                    mask_threshold=config.psnr_mask_threshold, clamp=config.psnr_clamp_rgb,
                )
                psnr = psnr_metrics["psnr_full"]
                psnr_masked = psnr_metrics.get("psnr_masked", float("nan"))
                snap = {
                    "step": step, "psnr_full_current": psnr, "psnr_masked_current": psnr_masked,
                    "loss_total": total_loss.item(), "loss_rgb": rgb_loss.item(),
                    "loss_motion": motion_loss.item(), "loss_rigid": rigid_loss.item(),
                    "loss_reg": reg_loss.item(), "loss_sigma_floor": sigma_floor_loss.item(),
                    "loss_chunk_outside_mass": chunk_outside_mass_loss.item(),
                }
                metrics["snapshots"].append(snap)
                if hasattr(bar, "set_postfix"):
                    if math.isfinite(psnr_masked):
                        bar.set_postfix(psnr=f"{psnr:.1f}", psnr_m=f"{psnr_masked:.1f}",
                                        loss=f"{total_loss.item():.4f}")
                    else:
                        bar.set_postfix(psnr=f"{psnr:.1f}", loss=f"{total_loss.item():.4f}")

            if writer is not None and config.tb_stats_every > 0 and step % config.tb_stats_every == 0:
                with torch.no_grad():
                    current_mask = None
                    if gt_alphas_per_frame is not None:
                        try:
                            current_mask = gt_alphas_per_frame[frame_idx][cam_idx].to(self.device)
                        except Exception:
                            current_mask = None

                    cur_psnr = compute_psnr_metrics(
                        render_log, gt_rgb, current_mask,
                        mask_threshold=config.psnr_mask_threshold, clamp=config.psnr_clamp_rgb,
                    )
                    psnr_val = cur_psnr["psnr_full"]
                    writer.add_scalar("psnr/current_full", cur_psnr["psnr_full"], step)
                    # With clamp_render_for_loss, pixels above 1 get no
                    # gradient; watch how far the raw render overshoots.
                    writer.add_scalar("render/max", float(render_log.max()), step)
                    writer.add_scalar(
                        "render/overbright_frac",
                        float((render_log > 1.0).float().mean()),
                        step,
                    )
                    writer.add_scalar("mse/current_full", cur_psnr["mse_full"], step)
                    if "psnr_masked" in cur_psnr:
                        writer.add_scalar("psnr/current_masked", cur_psnr["psnr_masked"], step)
                        writer.add_scalar("mse/current_masked", cur_psnr["mse_masked"], step)
                        writer.add_scalar("mask/current_pixels", cur_psnr["mask_pixels"], step)

                    if (
                        config.psnr_eval_every > 0 and psnr_eval_pairs
                        and step % config.psnr_eval_every == 0
                    ):
                        eval_metrics = self._evaluate_psnr_subset(
                            params=params,
                            cameras=cameras,
                            eval_pairs=psnr_eval_pairs,
                            sh_degree=sh_degree,
                            config=config,
                            gt_images_per_frame=gt_images_per_frame,
                            gt_alphas_per_frame=gt_alphas_per_frame,
                            gt_times_per_frame=gt_times_per_frame,
                            step=step,
                        )
                        for k, v in eval_metrics.items():
                            writer.add_scalar(k, v, step)
                        metrics.setdefault("psnr_eval", []).append({
                            "step": int(step),
                            **{k.replace("/", "_"): float(v) for k, v in eval_metrics.items()},
                        })

                        if psnr_holdout_pairs:
                            holdout_metrics = self._evaluate_psnr_subset(
                                params=params,
                                cameras=cameras,
                                eval_pairs=psnr_holdout_pairs,
                                sh_degree=sh_degree,
                                config=config,
                                gt_images_per_frame=gt_images_per_frame,
                                gt_alphas_per_frame=gt_alphas_per_frame,
                                gt_times_per_frame=gt_times_per_frame,
                                step=step,
                            )
                            # Same keys as the train-view pass; re-prefix so the
                            # two series are directly comparable in TB.
                            holdout_metrics = {
                                k.replace("eval_", "holdout_"): v
                                for k, v in holdout_metrics.items()
                            }
                            for k, v in holdout_metrics.items():
                                writer.add_scalar(k, v, step)
                            metrics.setdefault("psnr_holdout", []).append({
                                "step": int(step),
                                **{
                                    k.replace("/", "_"): float(v)
                                    for k, v in holdout_metrics.items()
                                },
                            })

                    if "n_active" in info:
                        writer.add_scalar("temporal/n_active", info["n_active"], step)
                        writer.add_scalar(
                            "temporal/active_ratio",
                            info["n_active"] / max(info["n_total"], 1),
                            step,
                        )
                    if (
                        config.temporal_hierarchy_enabled
                        and self.temporal_hierarchy is not None
                        and self.temporal_hierarchy.valid
                    ):
                        hierarchy = self.temporal_hierarchy
                        n_total = max(int(params["means"].shape[0]), 1)

                        writer.add_scalar(
                            "temporal_hierarchy/candidate_ratio",
                            hierarchy.last_candidate_count / n_total,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/active_ratio",
                            hierarchy.last_active_count / n_total,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/endpoint_drift_s",
                            hierarchy.max_endpoint_drift_s,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/segments_visited",
                            hierarchy.last_segments_visited,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/effective_root_s",
                            hierarchy.effective_root_segment_s,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/overflow_count",
                            hierarchy.overflow_count,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/overflow_fraction",
                            hierarchy.dirty_fraction,
                            step,
                        )
                        writer.add_scalar(
                            "temporal_hierarchy/drift_slack_s",
                            hierarchy.drift_slack_s,
                            step,
                        )
                    self._log_stats_to_tb(writer, step, params, psnr_val, t_max=t_max)
                    self._log_fixed_active_stats(writer, step, params, config, t_max=t_max)
                

        metrics["final_count"] = params["means"].shape[0]
        if config.verbose and metrics.get("capacity"):
            print(_capacity_summary(metrics["capacity"][-1]))
        return metrics
