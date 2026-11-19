"""CLI entry point to export deformed PLYs from a TRiGS checkpoint.

Usage:
    uv run trigs-export-plys \
        --ckpt-path path/to/trigs_ckpt.pt \
        --output-dir path/to/output_plys

``--n-frames`` / ``--fps`` are optional: both are read from the checkpoint when
it carries playback metadata (final ``trigs_ckpt.pt`` from ``train_from_plys``,
mid-training safety snapshots, or SelfCap eval metadata).

The decode path deliberately goes through ``TrigsTrainer._deform`` with the
config the checkpoint was TRAINED with, not a fresh default config. Several
deform-relevant knobs (``temporal_gamma_smooth_cull``, ``temporal_active_*``,
``gauge_*``) change the geometry/opacity a checkpoint decodes to, so exporting
under defaults silently produces frames that do not match what the trainer
rendered.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from trigs.compat.device import resolve_device  # macOS port
import tyro
from gsply import GSTensor
from torch import Tensor
from tqdm import tqdm

from trigs.core.temporal_hierarchy import direct_temporal_active_idx
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig

log = logging.getLogger(__name__)


# Canonical per-row params required by _deform. Anything else in the dict
# (camera_time_offsets, camera_appearance, relocation pool flags, ...) is
# either not row-wise or not needed to decode a frame.
_REQUIRED_PARAMS = (
    "means", "quats", "scales", "opacities", "sh0",
    "times", "log_durations", "anchors",
    "zeta_base_t", "zeta_base_r",
    "zeta_c0_t", "zeta_c1_t", "zeta_c2_t",
    "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
)

# Config fields that change what _deform returns. Logged at export time so a
# mismatch between the training run and the export is visible, not silent.
_DEFORM_RELEVANT_FIELDS = (
    "fps",
    "gauge_eps",
    "gauge_denom_clamp",
    "temporal_active_cull",
    "temporal_active_threshold",
    "temporal_active_cull_threshold",
    "temporal_active_extra_margin_s",
    "temporal_gamma_smooth_cull",
)


@dataclass
class ExportArgs:
    """Arguments for TRiGS PLY exporter."""

    ckpt_path: Path
    """Path to the trigs_ckpt.pt file (final checkpoint or training snapshot)."""

    output_dir: Path
    """Directory to write output PLY files."""

    n_frames: int | None = None
    """Number of frames to export. None = read from the checkpoint."""

    fps: float | None = None
    """Frames per second for time mapping. None = read from the checkpoint."""

    start_frame: int = 0
    """First frame index to decode; time is (start_frame + i) / fps seconds."""

    device: str = "auto"
    """Target device for deformation. Falls back to CPU when CUDA is absent."""

    active_opacity_thresh: float = 1e-4
    """Export only frame-active Gaussians with effective opacity above this."""

    apply_temporal_cull: bool = True
    """Also apply the trainer's temporal residency cull, so exported frames
    contain exactly the rows the trainer's renderer would have drawn. Ignored
    when the checkpoint was trained with temporal_active_cull=False."""

    ignore_ckpt_config: bool = False
    """Decode with default config instead of the checkpoint's training config."""

    write_meta: bool = True
    """Write export_meta.json (fps, frame range, per-frame primitive counts)."""


def _resolve_device(requested: str) -> str:
    # macOS port: cuda -> mps -> cpu fallback via the central helper.
    return str(resolve_device(requested))


def _load_checkpoint(ckpt_path: Path, device: str) -> dict[str, Any]:
    # Training snapshots carry a config dict and pickled RNG state, so
    # weights_only=True (the torch>=2.6 default) cannot load them.
    try:
        return torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:  # torch too old to know the kwarg
        return torch.load(ckpt_path, map_location=device)


def _resolve_params(ckpt: dict[str, Any], device: str) -> dict[str, Tensor]:
    raw = ckpt.get("params")
    if raw is None:
        raw = ckpt.get("splats")  # legacy/upstream layout
    if raw is None:
        raise SystemExit(
            f"Checkpoint has no 'params' (or 'splats') entry; top-level keys: "
            f"{sorted(ckpt.keys())}"
        )

    # A plain dict, not nn.ParameterDict: some rows (is_static, relocation pool
    # flags) may be stored non-float, and nn.Parameter rejects those. _deform
    # only ever indexes and tests membership.
    params: dict[str, Tensor] = {}
    for k, v in raw.items():
        params[k] = v.detach().to(device) if torch.is_tensor(v) else v

    missing = [k for k in _REQUIRED_PARAMS if k not in params]
    if missing:
        raise SystemExit(
            f"Checkpoint is missing required parameters {missing}. "
            f"Present keys: {sorted(params.keys())}"
        )
    return params


def _config_from_ckpt(
    ckpt: dict[str, Any],
    *,
    fps_override: float | None,
    ignore_ckpt_config: bool,
) -> TrigsTrainingConfig:
    """Rebuild the training config, falling back to defaults when absent."""
    raw = None if ignore_ckpt_config else ckpt.get("config")

    if isinstance(raw, TrigsTrainingConfig):
        config = dataclasses.replace(raw)
    elif isinstance(raw, dict):
        known = {f.name for f in dataclasses.fields(TrigsTrainingConfig)}
        kwargs = {k: v for k, v in raw.items() if k in known}
        dropped = sorted(set(raw) - known)
        try:
            config = TrigsTrainingConfig(**kwargs)
        except TypeError as exc:
            log.warning("Could not rebuild config from checkpoint (%s); using defaults", exc)
            config = TrigsTrainingConfig()
        else:
            log.info("Decoding with the checkpoint's training config")
            if dropped:
                log.info("  ignored %d unknown config key(s): %s", len(dropped), dropped)
    else:
        config = TrigsTrainingConfig()
        if not ignore_ckpt_config:
            log.warning(
                "Checkpoint carries no training config (final checkpoints from "
                "train_from_colmap/selfcap do not save one). Decoding with "
                "defaults -- if the run overrode any of %s, pass a checkpoint "
                "from --config.checkpoint-dir instead.",
                ", ".join(_DEFORM_RELEVANT_FIELDS),
            )

    if fps_override is not None:
        config.fps = float(fps_override)

    defaults = TrigsTrainingConfig()
    diffs = [
        f"{name}={getattr(config, name)!r} (default {getattr(defaults, name)!r})"
        for name in _DEFORM_RELEVANT_FIELDS
        if getattr(config, name, None) != getattr(defaults, name, None)
    ]
    if diffs:
        log.info("Deform-relevant config differs from defaults: %s", "; ".join(diffs))
    return config


def _resolve_playback(
    ckpt: dict[str, Any],
    config: TrigsTrainingConfig,
    args: ExportArgs,
) -> tuple[int, float]:
    meta = ckpt.get("selfcap_eval_meta") or {}

    fps = args.fps
    if fps is None:
        for candidate in (meta.get("fps"), ckpt.get("fps"), config.fps):
            if candidate:
                fps = float(candidate)
                break
    if not fps or fps <= 0:
        raise SystemExit("Could not determine fps; pass --fps explicitly.")

    n_frames = args.n_frames
    if n_frames is None:
        for candidate in (meta.get("n_frames"), ckpt.get("n_frames")):
            if candidate:
                n_frames = int(candidate)
                break
    if not n_frames or n_frames <= 0:
        raise SystemExit(
            "Checkpoint carries no frame count; pass --n-frames explicitly. "
            f"Top-level keys: {sorted(ckpt.keys())}"
        )
    return int(n_frames), float(fps)


def _log_checkpoint_summary(
    ckpt: dict[str, Any], params: dict[str, Tensor], n_frames: int, fps: float, start_frame: int
) -> None:
    n = int(params["means"].shape[0])
    fmt = ckpt.get("format", "trigs_ckpt")
    step = ckpt.get("step_completed", ckpt.get("step"))
    log.info(
        "Checkpoint: format=%s%s, %d primitives", fmt,
        f", step={step}" if step is not None else "", n,
    )
    if "is_static" in params:
        n_static = int((params["is_static"].reshape(-1) > 0.5).sum())
        log.info("  hybrid set: %d static / %d dynamic rows", n_static, n - n_static)
    if "shN" not in params:
        log.info("  no shN: exporting SH degree 0")

    mu = params["times"].reshape(-1)
    t_lo = (start_frame) / fps
    t_hi = (start_frame + n_frames - 1) / fps
    log.info(
        "  temporal centers span [%.3f, %.3f]s; exporting [%.3f, %.3f]s",
        float(mu.min()), float(mu.max()), t_lo, t_hi,
    )
    if float(mu.max()) > t_hi + 1e-6 or float(mu.min()) < t_lo - 1e-6:
        log.warning(
            "  export window does not cover all temporal centers -- some "
            "primitives are never emitted. Check --start-frame/--n-frames/--fps."
        )


def main(args: ExportArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(ExportArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ckpt_path = Path(args.ckpt_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device)

    log.info(f"Loading checkpoint from {ckpt_path}")
    ckpt = _load_checkpoint(ckpt_path, device)
    params = _resolve_params(ckpt, device)
    config = _config_from_ckpt(
        ckpt, fps_override=args.fps, ignore_ckpt_config=args.ignore_ckpt_config
    )
    n_frames, fps = _resolve_playback(ckpt, config, args)
    config.fps = fps
    _log_checkpoint_summary(ckpt, params, n_frames, fps, args.start_frame)

    trainer = TrigsTrainer(device=device)

    cull = bool(args.apply_temporal_cull and config.temporal_active_cull)
    cull_thresh = TrigsTrainer._effective_temporal_cull_threshold(config)
    if args.apply_temporal_cull and not config.temporal_active_cull:
        log.info("Checkpoint trained with temporal_active_cull=False; residency cull skipped")
    elif cull:
        log.info("Applying trainer residency cull at gamma > %.4g", cull_thresh)

    log.info(f"Exporting {n_frames} frames at {fps} FPS -> {output_dir}")

    counts: list[int] = []
    for i in tqdm(range(n_frames), desc="Decode + save PLYs"):
        fi = args.start_frame + i
        t = float(fi) / fps
        with torch.no_grad():
            means_t, quats_t, gamma = trainer._deform(params, t, config)

            base_opa = torch.sigmoid(params["opacities"])
            if base_opa.dim() == 1:
                base_opa = base_opa.unsqueeze(-1)
            eff_opa = (base_opa * gamma).squeeze(-1)

            active_mask = eff_opa > args.active_opacity_thresh
            if cull:
                # Match the trainer's renderer exactly: it only ever rasterizes
                # rows inside the temporal residency window (static rows always
                # included), so rows outside it were never supervised at this t.
                idx = direct_temporal_active_idx(
                    params, t,
                    opacity_threshold=cull_thresh,
                    extra_margin_s=config.temporal_active_extra_margin_s,
                    include_static=True,
                )
                resident = torch.zeros_like(active_mask)
                resident[idx] = True
                active_mask &= resident

            if not bool(active_mask.any()):
                # Keep one primitive to avoid writing an empty frame file.
                active_mask[torch.argmax(eff_opa)] = True

            # gsply convention: opacities stored in logit space
            eff_opa_clamped = eff_opa[active_mask].clamp(1e-6, 1.0 - 1e-6)
            eff_opa_logit = torch.logit(eff_opa_clamped)

        counts.append(int(active_mask.sum()))

        gs_t = GSTensor.from_arrays(
            means=means_t[active_mask],
            scales=params["scales"][active_mask],
            quats=quats_t[active_mask],
            opacities=eff_opa_logit,
            sh0=params["sh0"][active_mask],
            shN=params["shN"][active_mask] if "shN" in params else None,
            format="ply",
            device=device,
        )

        gs_t.save(str(output_dir / f"frame_{fi:06d}.ply"), compressed=False)
        del gs_t

    if args.write_meta:
        meta_path = output_dir / "export_meta.json"
        meta_path.write_text(json.dumps({
            "source_checkpoint": str(ckpt_path),
            "fps": fps,
            "start_frame": int(args.start_frame),
            "n_frames": int(n_frames),
            "n_primitives_total": int(params["means"].shape[0]),
            "active_opacity_thresh": float(args.active_opacity_thresh),
            "temporal_cull_applied": cull,
            "temporal_cull_threshold": float(cull_thresh) if cull else None,
            "config_source": "checkpoint" if (
                not args.ignore_ckpt_config and ckpt.get("config") is not None
            ) else "defaults",
            "per_frame_counts": counts,
        }, indent=2) + "\n")
        log.info(f"Wrote {meta_path}")

    if counts:
        log.info(
            "Primitives per frame: min=%d mean=%.0f max=%d",
            min(counts), sum(counts) / len(counts), max(counts),
        )
    log.info(f"Saved {n_frames} PLYs to {output_dir}")


if __name__ == "__main__":
    main()
