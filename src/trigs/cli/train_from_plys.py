"""CLI entry point for the TRiGS trainer (PLY-sequence mode).

Takes a directory of dense per-frame 3DGS PLYs, picks every ``keyframe_stride``-th
PLY as the canonical set, and optimizes per-primitive coupled SE(3) motion
with Bézier residuals against rendered GT from all frames.

Usage:
    uv run trigs-train-plys \
        --input-dir path/to/dense_frames \
        --output-dir path/to/output \
        --config.keyframe-stride 3 \
        --config.n-steps 30000
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import torch
from trigs.compat.device import resolve_device, empty_cache  # macOS port
import tyro
from gsply import GSTensor
from tqdm import tqdm

from trigs.config.camera import CameraCLIConfig
from trigs.core.transforms import TransformConfig
from trigs.core.renderer import Renderer
from trigs.core.viewer_snapshot import apply_default_snapshot_dir
from trigs.core.trainer import (
    TrigsTrainer,
    TrigsTrainingConfig,
)
from trigs.init.from_plys import (
    PlyMotionInitConfig,
    count_keyframe_gaussians,
    discover_ply_files,
)
from trigs.io.ply import load_ply
from trigs.io.ply_gt import PlyGtStore

log = logging.getLogger(__name__)


@dataclass
class TrigsPipelineArgs:
    """Arguments for the TRiGS PLY-sequence trainer."""

    input_dir: Path
    """Directory containing dense PLY files (one per frame, sorted)."""

    output_dir: Path
    """Directory to write checkpoint + debug renders."""

    config: TrigsTrainingConfig = field(default_factory=TrigsTrainingConfig)
    """TRiGS training configuration (weights, LRs, relocation, ...)."""

    camera: CameraCLIConfig = field(default_factory=CameraCLIConfig)
    """Camera orbit configuration for rendering GT + debug views."""

    transform: TransformConfig = field(default_factory=TransformConfig)
    """Scene coordinate transform applied to every loaded PLY."""

    ply_init: PlyMotionInitConfig = field(default_factory=PlyMotionInitConfig)
    """Motion-adaptive canonical selection. Disabled by default."""

    gt_cache_frames: int = 0
    """Render GT on demand, retaining this many frames. ``0`` renders up front.

    The eager cache is ``n_frames * n_cameras * H * W * 4`` bytes for the whole
    sequence, which caps resolution by sequence length. Chunked training only
    reads the frames in the current window, so retaining just those frees that
    budget for cameras and resolution instead. Set this at or above the frame
    span one chunk supervises; below that, every step evicts the frame the next
    step needs. Pair with ``--config.temporal-chunk-len-s`` for locality.
    """

    device: str = "auto"
    """Target device."""

    render_debug: bool = True
    """Render deformed frames for every keyframe after training (diagnostic)."""

    save_plys: bool = True
    """Decode and save every trained frame after training."""

    compact_profile: bool = False
    """Enable PLY-only dense-sequence initialization.

    The target is derived from the actual input keyframe population, rather
    than copied from a paper benchmark. Temporal support remains local and
    source PLY appearance parameters are preserved.
    """

    vanilla_profile: bool = False
    """Restore TriGSVanilla's exposed PLY training defaults.

    This disables local temporal/scale clamps, active temporal culling,
    smooth-cull gamma renormalization, and relocation residual recentering. It
    also restores Adam beta1=0.9 for learned time/duration parameters. The
    PLY-specific ``unfreeze_temporal_step``, ``masked_rgb_loss``, and
    sigma-floor controls are deliberately preserved so segmented PLY inputs
    can opt into the upstream-documented safeguards without an upper σ_t cap.
    Their base defaults remain zero/disabled for an exact exposed-default
    control. The profile is confined to this PLY CLI and is mutually exclusive
    with ``compact_profile``. It retains this fork's PLY loader, cache,
    diagnostics, and hardened core implementation; run the upstream checkout
    itself for a code-identical control.
    """

    target_retention: float = 0.40
    """Fraction of source keyframe rows retained by the compact profile.

    With σ initialized to one keyframe interval, neighboring temporal Gaussian
    weights have about 2.5 cohorts' worth of effective mass. Retaining 40%
    therefore starts near one source frame's effective spatial density. For
    the 20.8M-row sequence this produces about 8.33M canonical rows. An
    explicit ``config.target_count`` overrides this value.
    """

    initial_opacity: float = 0.0
    """PLY-only canonical opacity initialization.

    Values in ``(0, 1)`` replace inherited per-frame PLY opacity logits before
    training. ``0`` preserves source logits and is the compact-profile default.
    This option exists for controlled PLY experiments only.
    """


def _apply_vanilla_profile(args: TrigsPipelineArgs) -> None:
    """Restore the exposed upstream TriGSVanilla PLY defaults.

    Audited against OpsiClear-4DGS/TriGSVanilla main at
    ``bb51ea6482513bb3075a11e959a2beec46f1702b``. Only PLY CLI arguments are
    mutated; COLMAP and SelfCap entry points never call this function.
    """
    if not args.vanilla_profile:
        return
    if args.compact_profile:
        raise ValueError("--vanilla-profile and --compact-profile are mutually exclusive")

    config = args.config

    # Upstream's exposed default trains jointly from step zero, but its README
    # also documents the motion-only phase specifically to prevent temporal
    # visibility from shrinking before motion is learned. Preserve the parsed
    # PLY setting so an explicit warm-up is not silently discarded.
    config.omega_init = False
    config.freeze_geometry = False

    # Upstream renders the raw temporal Gaussian for every row. There is no
    # hard active-set cutoff or renormalization at the cutoff boundary.
    config.temporal_active_cull = False
    config.temporal_gamma_smooth_cull = False
    config.temporal_hierarchy_enabled = False
    config.temporal_chunk_len_s = 0.0

    # Upstream has no post-step duration or spatial-scale clamp. Its forward
    # pass retains only the 1e-6 numerical sigma floor inside exp/gamma.
    config.clamp_log_durations = False
    config.clamp_log_scales = False
    config.max_sigma_t = 0.0
    config.max_scale_exp = 0.0

    # Preserve explicit mask and one-sided sigma-floor controls. The audited
    # upstream defaults leave them disabled, while upstream's own config notes
    # that the floor is needed for segmented-dynamic scenes without a static
    # background population. Neither control imposes a maximum sigma_t.
    config.lambda_motion_residual = 0.0
    config.temporal_optimizer_beta1 = 0.9

    # Upstream relocation copies every row tensor verbatim except for the
    # documented 0.66 scale multiplier.
    config.relocate_dynamic_only = False
    config.relocate_mcmc_split = False
    config.relocate_time_jitter_sigma = 0.0
    config.relocate_recenter_residual = False
    config.relocate_settle_steps = 0

    log.info(
        "PLY TriGSVanilla mechanics profile (bb51ea6): motion-only=%d steps, "
        "raw/unbounded temporal Gaussian, all-row rendering, Adam temporal "
        "beta1=0.9, verbatim relocation, masked_loss=%s, sigma_floor_lambda=%g "
        "target=%g",
        config.unfreeze_temporal_step,
        config.masked_rgb_loss,
        config.lambda_sigma_floor,
        config.sigma_floor_target,
    )


def _apply_compact_profile(
    args: TrigsPipelineArgs,
    *,
    keyframe_row_count: int | None = None,
) -> None:
    """Apply input-derived dense-sequence defaults to the PLY CLI only."""
    if not args.compact_profile:
        return

    config = args.config
    if config.target_count is None:
        if keyframe_row_count is None or keyframe_row_count <= 0:
            raise ValueError(
                "keyframe_row_count is required to derive the PLY compact target",
            )
        if not 0.0 < args.target_retention <= 1.0:
            raise ValueError("target_retention must lie in (0, 1]")
        config.target_count = max(
            math.ceil(int(keyframe_row_count) * float(args.target_retention)),
            1,
        )
    if config.init_sigma_t <= 0:
        config.init_sigma_t = (
            max(int(config.keyframe_stride), 1)
            / max(float(config.fps), 1e-6)
        )
    if config.max_sigma_t in (0.0, TrigsTrainingConfig().max_sigma_t):  # not set by the user
        # TensorBoard showed a healthy ~0.057s median but a pathological p99
        # growing to ~0.506s and pinning against the old 0.6s ceiling. Keep the
        # tail local without returning to the overly short 0.08s experiment.
        config.max_sigma_t = max(2.0 * config.init_sigma_t, 0.2)
    if config.unfreeze_temporal_step <= 0:
        config.unfreeze_temporal_step = min(5_000, max(config.n_steps // 2, 1))
    config.omega_init = True

    # PLY GT includes an exact alpha mask. Without the masked objective the
    # mostly-white image dominates: the failed run reached 36.8 dB full-image
    # PSNR but only 22.2 dB on the subject, while both alpha losses stayed 0.
    config.masked_rgb_loss = True

    # Saturated opacity grew from 1.67% at 25k to 6.84% at 60k after the
    # default clamp expired. Keep a sub-saturation ceiling for the full PLY
    # run. Respect deliberate non-default overrides.
    if config.opacity_logit_upper_clamp in (6.0, TrigsTrainingConfig().opacity_logit_upper_clamp):  # not set
        config.opacity_logit_upper_clamp = 4.0
    if config.opacity_logit_upper_clamp_until_step == 25_000:
        config.opacity_logit_upper_clamp_until_step = config.n_steps

    target_source = (
        f"{100.0 * float(args.target_retention):.1f}%"
        if keyframe_row_count is not None
        else "explicit"
    )
    log.info(
        "PLY dense profile: source_keyframe_rows=%s, target=%s (%s), "
        "sigma_init=%.3fs, sigma_max=%.3fs, motion-only=%d steps, "
        "opacity=%s, opacity_logit_max=%s through step %d, "
        "masked_loss=%s, cameras=%d",
        f"{keyframe_row_count:,}" if keyframe_row_count is not None else "explicit",
        f"{config.target_count:,}" if config.target_count is not None else "none",
        target_source,
        config.init_sigma_t,
        config.max_sigma_t,
        config.unfreeze_temporal_step,
        "source" if args.initial_opacity <= 0 else f"{args.initial_opacity:.3f}",
        (
            "none"
            if config.opacity_logit_upper_clamp is None
            else f"{config.opacity_logit_upper_clamp:.3f}"
        ),
        config.opacity_logit_upper_clamp_until_step,
        config.masked_rgb_loss,
        args.camera.n_cameras,
    )


@torch.no_grad()
def _initialize_ply_opacity(
    params: torch.nn.ParameterDict,
    initial_opacity: float,
) -> None:
    """Reset PLY canonical opacity without touching other input modes."""
    if initial_opacity <= 0:
        return
    if not 0.0 < initial_opacity < 1.0:
        raise ValueError("initial_opacity must be 0 (preserve) or lie in (0, 1)")

    old_alpha = torch.sigmoid(params["opacities"].detach().float()).reshape(-1)
    # This is logging only. Bound the sample so diagnostics remain safe even
    # when someone requests an opacity reset without a compact target count.
    if int(old_alpha.numel()) > 250_000:
        sample_idx = torch.linspace(
            0,
            int(old_alpha.numel()) - 1,
            steps=250_000,
            device=old_alpha.device,
        ).long()
        alpha_stats = old_alpha[sample_idx]
    else:
        alpha_stats = old_alpha
    logit = math.log(initial_opacity / (1.0 - initial_opacity))
    params["opacities"].data.fill_(logit)
    log.info(
        "PLY opacity init: source alpha median=%.4f p95=%.4f -> %.4f",
        alpha_stats.median().item(),
        torch.quantile(alpha_stats, 0.95).item(),
        initial_opacity,
    )


@torch.no_grad()
def _configure_ply_scale_guard(
    params: torch.nn.ParameterDict,
    config: TrigsTrainingConfig,
) -> None:
    """Bound PLY scale growth using the retained source PLY distribution."""
    if config.max_scale_exp > 0:
        return

    source_scales = params["scales"].detach().float().exp().amax(dim=-1)
    if int(source_scales.numel()) > 250_000:
        sample_idx = torch.linspace(
            0,
            int(source_scales.numel()) - 1,
            steps=250_000,
            device=source_scales.device,
        ).long()
        scale_stats = source_scales[sample_idx]
    else:
        scale_stats = source_scales

    median = scale_stats.median()
    p999 = torch.quantile(scale_stats, 0.999)
    source_max = source_scales.max()
    robust_limit = torch.maximum(p999 * 2.0, median * 8.0)
    scale_limit = torch.minimum(robust_limit, source_max * 1.05)
    scale_limit = torch.maximum(scale_limit, p999)
    config.max_scale_exp = max(float(scale_limit.item()), 1e-6)
    log.info(
        "PLY scale guard: median=%.6f p99.9=%.6f source_max=%.6f -> max=%.6f",
        median.item(),
        p999.item(),
        source_max.item(),
        config.max_scale_exp,
    )


def _warn_on_gt_cache_thrash(args: TrigsPipelineArgs, n_frames: int) -> None:
    """Flag LRU sizes that will evict the frame the next step needs.

    The training epoch is time-first: every frame is visited once per view
    round. Without chunking the working set is therefore the whole sequence,
    and any smaller LRU misses on nearly every step -- a PLY load plus a full
    rig render per iteration. Chunked training narrows the working set to one
    window, which is what makes the LRU viable.
    """
    if args.gt_cache_frames >= n_frames:
        return

    chunk_len_s = float(getattr(args.config, "temporal_chunk_len_s", 0.0))
    if chunk_len_s <= 0.0:
        log.warning(
            "gt_cache_frames=%d < %d frames with chunked training disabled. "
            "The time-first epoch touches every frame per view round, so the "
            "cache will miss on nearly every step and re-render the rig each "
            "time. Set --config.temporal-chunk-len-s (e.g. 4.0), or raise "
            "gt_cache_frames to %d.",
            args.gt_cache_frames, n_frames, n_frames,
        )
        return

    pad_s = float(getattr(args.config, "temporal_chunk_supervision_pad_s", 0.0))
    span_frames = math.ceil((chunk_len_s + 2.0 * pad_s) * float(args.config.fps)) + 1
    if args.gt_cache_frames < span_frames:
        log.warning(
            "gt_cache_frames=%d is below the %d frames one chunk supervises "
            "(%.2fs chunk + 2x%.2fs pad @ %.0f fps); expect heavy re-rendering. "
            "Raise it to at least %d.",
            args.gt_cache_frames, span_frames, chunk_len_s, pad_s,
            float(args.config.fps), span_frames,
        )


def _save_checkpoint(
    result,
    output_dir: Path,
    *,
    fps: float,
    n_frames: int,
) -> None:
    """Persist parameters + temporal/motion tensors for later decoding."""
    ckpt_path = output_dir / "trigs_ckpt.pt"
    sh_degree = 0
    if "shN" in result.params:
        sh_degree = int(round((result.params["shN"].shape[1] + 1) ** 0.5)) - 1
    payload = {
        "params": {k: v.detach().cpu() for k, v in result.params.items()},
        "temporal_params": result.temporal_params,
        "motion_params": result.motion_params,
        "metrics": result.metrics,
        # Top-level playback metadata is consumed directly by viewer.py and
        # export-trigs.py. Without it, the viewer silently falls back to 300
        # frames at 30 FPS even for a longer PLY sequence.
        "fps": float(fps),
        "n_frames": int(n_frames),
        "sh_degree": int(sh_degree),
    }
    torch.save(payload, ckpt_path)
    log.info(f"Saved checkpoint -> {ckpt_path}")


def _main_extended(args: TrigsPipelineArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(TrigsPipelineArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.config.diagnostics else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snap_dir = apply_default_snapshot_dir(args.config, output_path)
    if snap_dir is not None:
        log.info(
            "Viewer snapshots every %d steps -> %s (view: viewer.py --watch %s)",
            args.config.viewer_snapshot_every, snap_dir, snap_dir,
        )

    ply_files = discover_ply_files(input_path)
    log.info(f"Found {len(ply_files)} PLY files in {input_path}")
    _apply_vanilla_profile(args)
    keyframe_row_count = None
    if args.compact_profile and args.config.target_count is None:
        keyframe_row_count = count_keyframe_gaussians(
            ply_files,
            args.config.keyframe_stride,
        )
    _apply_compact_profile(args, keyframe_row_count=keyframe_row_count)

    service = TrigsTrainer(device=args.device)
    runtime_device = str(service.device)
    if runtime_device != args.device:
        log.warning(
            "Requested device %s is unavailable; using %s",
            args.device,
            runtime_device,
        )

    # -- Cameras from frame 0 ----------------------------------------------
    gs0 = load_ply(ply_files[0], device=runtime_device)
    if not args.transform.is_identity:
        gs0 = args.transform.apply(gs0)
    cameras = args.camera.build(gs0)
    log.info(f"Built {cameras.viewmats.shape[0]} cameras")
    del gs0
    empty_cache()

    # -- Tensorboard writer -------------------------------------------------
    tb_writer = None
    if args.config.tb_log_every > 0 or args.config.tb_stats_every > 0:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_path / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))
        log.info(f"Tensorboard log dir: {tb_dir}")

    # -- PLY-only canonical build + training --------------------------
    gt_store = None
    try:
        params, sh_degree, scene_extent = service._build_canonical_set(
            ply_files,
            args.config,
            args.transform,
            len(ply_files),
            ply_init=args.ply_init,
        )
        if args.compact_profile:
            _configure_ply_scale_guard(params, args.config)
        _initialize_ply_opacity(params, args.initial_opacity)
        gt_store = None
        if args.gt_cache_frames > 0:
            gt_store = PlyGtStore(
                ply_files,
                cameras,
                args.transform,
                device=runtime_device,
                capacity_frames=args.gt_cache_frames,
                verbose=args.config.verbose,
            )
            _warn_on_gt_cache_thrash(args, len(ply_files))
            log.info(
                "Lazy PLY GT: %d-frame LRU over %d cameras -> %.1f GB resident "
                "(eager cache for %d frames would be %.1f GB)",
                args.gt_cache_frames,
                cameras.viewmats.shape[0],
                gt_store.resident_bytes() / 1e9,
                len(ply_files),
                gt_store.resident_bytes() / 1e9
                * len(ply_files) / max(args.gt_cache_frames, 1),
            )
            gt_images, gt_alphas = gt_store.images, gt_store.alphas
        else:
            gt_images, gt_alphas = service._render_gt(
                ply_files,
                cameras,
                args.transform,
                sh_degree,
                verbose=args.config.verbose,
            )

        if args.config.temporal_chunk_len_s > 0:
            log.info(
                "Preparing PLY sequence for chunked training "
                "(chunk=%.3fs, overlap=%.3fs, rounds=%d)",
                args.config.temporal_chunk_len_s,
                args.config.temporal_chunk_overlap_s,
                args.config.temporal_chunk_rounds,
            )
            result = service.train_chunked(
                params=params,
                sh_degree=sh_degree,
                scene_extent=scene_extent,
                cameras=cameras,
                n_frames=len(ply_files),
                gt_images_per_frame=gt_images,
                gt_alphas_per_frame=gt_alphas,
                config=args.config,
                writer=tb_writer,
                result_device="cpu",
            )
        else:
            result = service.train_from_params(
                params=params,
                sh_degree=sh_degree,
                scene_extent=scene_extent,
                cameras=cameras,
                n_frames=len(ply_files),
                gt_images_per_frame=gt_images,
                gt_alphas_per_frame=gt_alphas,
                config=args.config,
                writer=tb_writer,
            )
    finally:
        if tb_writer is not None:
            tb_writer.close()
    if gt_store is not None:
        stats = gt_store.stats
        # A low hit rate means the LRU is undersized for the chunk span and
        # the run paid a PLY load + rig render on most steps.
        log.info(
            "Lazy PLY GT: %.1f%% hit rate (%s hits / %s renders), %d/%d frames resident",
            100.0 * stats["hit_rate"],
            f"{stats['hits']:,}",
            f"{stats['misses']:,}",
            stats["resident_frames"],
            stats["capacity_frames"],
        )
    _save_checkpoint(
        result,
        output_path,
        fps=args.config.fps,
        n_frames=len(ply_files),
    )

    # -- Decode every frame to PLY, optionally render debug views ---------
    if not args.save_plys:
        log.info("Skipping per-frame PLY decoding (--no-save-plys)")
        return

    ply_out_dir = output_path / "frames_ply"
    ply_out_dir.mkdir(parents=True, exist_ok=True)

    render_dir = None
    n_views = cameras.viewmats.shape[0]
    if args.render_debug:
        render_dir = output_path / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        for v in range(n_views):
            (render_dir / f"cam_{v:04d}").mkdir(parents=True, exist_ok=True)

    n_frames = len(ply_files)
    renderer = Renderer(device=runtime_device) if args.render_debug else None

    for fi in tqdm(range(n_frames), desc="Decode + save"):
        t = float(fi) / max(args.config.fps, 1e-6)
        with torch.no_grad():
            means_t, quats_t, gamma = service._deform(result.params, t, args.config)
            base_opa = torch.sigmoid(result.params["opacities"])
            if base_opa.dim() == 1:
                base_opa = base_opa.unsqueeze(-1)
            eff_opa = (base_opa * gamma).squeeze(-1)
            active_mask = eff_opa > 1e-4
            if not bool(active_mask.any()):
                # Keep one primitive to avoid writing an empty frame file.
                active_mask[torch.argmax(eff_opa)] = True
            # gsply convention: opacities stored in logit space, the
            # renderer applies sigmoid. Convert the activated effective opacity
            # back to logit so the round-trip recovers σ(logit) = eff_opa.
            eff_opa_clamped = eff_opa[active_mask].clamp(1e-6, 1.0 - 1e-6)
            eff_opa_logit = torch.logit(eff_opa_clamped)
        gs_t = GSTensor.from_arrays(
            means=means_t.detach()[active_mask],
            scales=result.params["scales"].detach()[active_mask],
            quats=quats_t.detach()[active_mask],
            opacities=eff_opa_logit.detach(),
            sh0=result.params["sh0"].detach()[active_mask],
            shN=result.params["shN"].detach()[active_mask] if "shN" in result.params else None,
            format="ply",
            device=runtime_device,
        )
        gs_t.save(str(ply_out_dir / f"frame_{fi:06d}.ply"), compressed=False)

        if renderer is not None and render_dir is not None:
            frame_result = renderer.render(gs_t, cameras, show_progress=False)
            for v in range(n_views):
                img = frame_result.images[v]
                img_np = (torch.clamp(img, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                imageio.imwrite(
                    str(render_dir / f"cam_{v:04d}" / f"render_{fi:06d}.png"),
                    img_np,
                )
            del frame_result
        del gs_t

    log.info(f"Wrote {n_frames} deformed-frame PLYs to {ply_out_dir}")
    if render_dir is not None:
        log.info(f"Wrote debug renders for {n_frames} frames to {render_dir}")


def main() -> None:
    """Compatibility redirect to the pinned upstream PLY control path.

    Existing virtual environments may still have a console script pointing
    at ``trigs.cli.train_from_plys:main`` even after ``pyproject.toml`` changes.
    Keep that stale entry point safe by delegating here as well.  COLMAP and
    SelfCap import neither CLI module and remain unaffected.
    """
    from trigs.cli.train_from_plys_upstream import main as upstream_main

    upstream_main()


if __name__ == "__main__":
    main()
