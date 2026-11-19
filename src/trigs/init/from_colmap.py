"""COLMAP + ROMA initialization path for TriGS.

Mode B global/macro NPZ filtering version.

This file is intended for testing a macro/global ROMA NPZ while training
local chunks.

Mode B policy:
    - keep ALL static rows from the macro NPZ in every training chunk
    - keep ONLY dynamic rows whose source_frames are inside the local chunk
      and aligned to the local keyframe stride
    - skip expensive radius-ball / twist initialization for static rows so
      the global static group does not cause OOM

The trainer entrypoint should pass:
    source_frame_start=args.start_frame
    source_frame_end=args.end_frame
    keep_global_static_when_filtering=True
    filter_to_local_keyframe_stride=True

For compatibility, this function also accepts the older
``keep_static_when_filtering`` keyword and maps it to
``keep_global_static_when_filtering``.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor

from ..math.twist_init import (
    fit_bezier_to_acceleration,
    fit_bezier_to_targets,
    fit_local_twist,
    local_knn_indices,
    screw_deform_means,
    smooth_accelerations,
)

log = logging.getLogger(__name__)


# SH degree-0 DC coefficient.
# RGB stored as SH-DC is divided by this constant so the rasterizer recovers
# the original color: color = sh0 * Y_0^0.
_SH_C0 = 0.28209479177387814


def _knn_distance(points: Tensor, k: int = 4) -> Tensor:
    """Mean distance to k nearest neighbors, excluding self."""
    n_points = points.shape[0]
    if n_points <= 1:
        return torch.zeros(n_points, device=points.device)

    k_eff = max(1, min(k, n_points - 1))

    try:
        from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)

        res = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=k_eff + 1)
        dists = res.dists.squeeze(0).sqrt()[:, 1:]
        return dists.mean(dim=1)
    except ImportError:
        d = torch.cdist(points, points)
        d.fill_diagonal_(float("inf"))
        vals, _ = d.topk(k_eff, largest=False)
        return vals.mean(dim=1)


def _radius_ball_centroid(
    means: Tensor,
    radius: float,
    chunk_size: int = 2048,
) -> Tensor:
    """Centroid of neighbors within radius for each point, excluding self.

    This is used only for dynamic rows in Mode B. Static rows intentionally
    skip this path.
    """
    n_points = means.shape[0]
    centroids = torch.empty_like(means)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
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


def _scalar_int(npz: np.lib.npyio.NpzFile, key: str, default: int | None = None) -> int | None:
    """Read a scalar int from an NPZ if present."""
    if key not in npz:
        return default

    try:
        return int(np.asarray(npz[key]).reshape(()))
    except Exception:
        return default


def _infer_source_grid_start(
    npz: np.lib.npyio.NpzFile,
    source_frames_np: np.ndarray | None,
) -> int:
    """Infer source-frame grid start.

    Important: do not prefer logical frame_start, because some ROMA NPZs store
    logical TriGS frame_start=0 while source images start at 1.
    """
    value = _scalar_int(npz, "source_frame_start", None)
    if value is not None:
        return value

    if source_frames_np is not None and source_frames_np.size > 0:
        return int(source_frames_np.min())

    value = _scalar_int(npz, "frame_start", None)
    if value is not None:
        return value

    return 0


def _require_source_frames(npz: np.lib.npyio.NpzFile, n_rows: int) -> np.ndarray:
    """Return source_frames as [N] int64, or fail loudly."""
    if "source_frames" not in npz:
        raise ValueError(
            "Mode B global filtering requires NPZ key 'source_frames'. "
            "The macro ROMA NPZ must save source_frames so dynamic rows can "
            "be filtered per training chunk."
        )

    source_frames = np.asarray(npz["source_frames"]).reshape(-1).astype(np.int64)
    if len(source_frames) != n_rows:
        raise RuntimeError(
            f"NPZ row mismatch: positions={n_rows}, source_frames={len(source_frames)}"
        )

    return source_frames


def _npz_durations_seconds(npz, durations: np.ndarray, fps: float) -> np.ndarray:
    """Convert the prep's per-row durations to seconds using its time_units."""
    units = str(npz["time_units"]) if "time_units" in npz else "seconds"
    d = np.asarray(durations, dtype=np.float64).reshape(-1)
    if "normalized" in units:
        span = float(int(npz["frame_end"]) - int(npz["frame_start"]))
        return d * span / max(float(fps), 1e-6)
    if "frame" in units:
        return d / max(float(fps), 1e-6)
    if "second" in units:
        return d
    raise ValueError(f"Unknown NPZ time_units {units!r} for durations")


def build_canonical_from_npz(
    npz_path: str | Path,
    *,
    fps: float = 30.0,
    init_opacity: float = 0.1,
    scale_knn_k: int = 4,
    scale_knn_factor: float = 0.3,
    anchor_radius_frac: float = 0.1,
    local_twist_k: int = 16,
    local_twist_reg: float = 1e-4,
    init_sigma_t: float = 0.0,
    init_sigma_t_mode: str = "uniform",
    init_sigma_t_floor_frac: float = 0.4,
    init_sigma_t_speed_spacings: float = 3.0,
    init_sigma_t_max: float = 0.5,
    init_accel_mode: str = "off",
    init_accel_scale: float = 0.5,
    init_accel_fit_iters: int = 300,
    init_chain_fit: bool = False,
    init_chain_min_span_frames: int = 6,
    keyframe_stride: int = 1,
    max_sh_degree: int = 3,
    device: str = "auto",
    source_frame_start: int | None = None,
    source_frame_end: int | None = None,
    keep_global_static_when_filtering: bool = True,
    filter_to_local_keyframe_stride: bool = True,
    # Backwards compatibility with the previous experimental patch.
    keep_static_when_filtering: bool | None = None,
    compute_device: str | None = None,
) -> Tuple[nn.ParameterDict, int, float]:
    """Build a TriGS canonical ParameterDict from a ROMA NPZ.

    If source_frame_start/source_frame_end are provided, the NPZ is treated as
    a macro/global NPZ and filtered with Mode B:

        keep = all static rows + local dynamic rows

    Static rows are retained in every chunk but do not run expensive dynamic
    anchor/twist initialization.

    ``device`` holds the returned parameters; ``compute_device`` (default:
    ``device``) runs the per-keyframe KNN/anchor/twist fits, one keyframe at a
    time. Chunked training builds on the CPU and computes on the GPU, so the
    whole model never has to fit in GPU memory.
    """
    if keep_static_when_filtering is not None:
        keep_global_static_when_filtering = bool(keep_static_when_filtering)

    npz = np.load(str(npz_path), allow_pickle=True)
    dev = resolve_device(device)
    cdev = torch.device(
        resolve_device(compute_device) if compute_device is not None
        else dev
    )

    positions_full = np.asarray(npz["positions"])
    n_before = int(positions_full.shape[0])

    for name in ("positions", "velocities", "colors", "times"):
        if name not in npz:
            raise KeyError(f"NPZ missing required key: {name}")
        if len(npz[name]) != n_before:
            raise RuntimeError(
                f"NPZ row mismatch: positions={n_before}, {name}={len(npz[name])}"
            )

    for name in (
        "durations",
        "has_velocity",
        "is_static",
        "dynamic_protect",
        "dynamic_allowed",
        "stage_keep",
        "source_frames",
        "accelerations",
        "has_acceleration",
        "local_stride",
    ):
        if name in npz and len(npz[name]) != n_before:
            raise RuntimeError(
                f"NPZ row mismatch: positions={n_before}, {name}={len(npz[name])}"
            )

    if "is_static" in npz:
        is_static_full = np.asarray(npz["is_static"]).astype(np.float32).reshape(-1, 1)
    else:
        is_static_full = np.zeros((n_before, 1), dtype=np.float32)

    if "dynamic_protect" in npz:
        dynamic_protect_full = (
            np.asarray(npz["dynamic_protect"]).astype(np.float32).reshape(-1, 1)
        )
    else:
        dynamic_protect_full = np.zeros((n_before, 1), dtype=np.float32)

    # New ROI metadata.
    #
    # dynamic_allowed:
    #   True  = row is allowed to be part of the actor/dynamic relocation pool.
    #   False = row is background/static-only; dynamic rows here are illegal.
    #
    # Backward-compatible fallback for old NPZs:
    #   old dynamic rows are allowed;
    #   old static rows are background/static-only.
    if "dynamic_allowed" in npz:
        dynamic_allowed_full = (
            np.asarray(npz["dynamic_allowed"]).astype(np.float32).reshape(-1, 1)
        )
    else:
        dynamic_allowed_full = (is_static_full <= 0.5).astype(np.float32)

    # stage_keep is mostly diagnostic/trainer metadata. Rows outside stage_keep
    # should normally already have been dropped by the ROMA NPZ script.
    if "stage_keep" in npz:
        stage_keep_full = (
            np.asarray(npz["stage_keep"]).astype(np.float32).reshape(-1, 1)
        )
    else:
        stage_keep_full = np.ones((n_before, 1), dtype=np.float32)

    # Protected actor rows always count as dynamic-allowed.
    protected_outside_allowed = (
        (dynamic_protect_full.reshape(-1) > 0.5)
        & (dynamic_allowed_full.reshape(-1) <= 0.5)
    )
    if int(protected_outside_allowed.sum()) > 0:
        log.warning(
            "[from_colmap ROI] "
            f"{int(protected_outside_allowed.sum()):,} protected rows were outside "
            "dynamic_allowed; forcing them dynamic_allowed=True.",
        )
        dynamic_allowed_full[protected_outside_allowed] = 1.0

    # Hard invariant for the trainer:
    #   no non-static row may live outside dynamic_allowed.
    #
    # This fixes the old one-capsule behavior:
    #   outside capsule + moving shadow velocity -> unprotected dynamic.
    #
    # With this invariant:
    #   outside dynamic_allowed -> static/background-only.
    illegal_dynamic_full = (
        (is_static_full.reshape(-1) <= 0.5)
        & (dynamic_allowed_full.reshape(-1) <= 0.5)
    )
    if int(illegal_dynamic_full.sum()) > 0:
        log.warning(
            "[from_colmap ROI] "
            f"{int(illegal_dynamic_full.sum()):,} rows were dynamic but outside "
            "dynamic_allowed; forcing them static.",
        )
        is_static_full[illegal_dynamic_full] = 1.0
        dynamic_protect_full[illegal_dynamic_full] = 0.0

    durations_full = np.asarray(npz["durations"]) if "durations" in npz else None

    # Per-row keyframe spacing (frames) from prep/init/subkeyframe-merge-npz.py:
    # fast rows emitted on a finer grid than keyframe_stride. Absent = every row
    # uses keyframe_stride (the original behaviour).
    local_stride_full = (
        np.asarray(npz["local_stride"]).reshape(-1).astype(np.int64) if "local_stride" in npz else None
    )

    # Measured accelerations (prep --centred-velocity): X(t+1) - 2X(t) + X(t-1).
    accel_mode = str(init_accel_mode).lower()
    if accel_mode not in ("off", "fit"):
        raise ValueError(f"init_accel_mode must be 'off' or 'fit', got {init_accel_mode!r}")
    if accel_mode == "fit" and "accelerations" not in npz:
        raise ValueError(
            f"init_accel_mode='fit' but {npz_path} has no 'accelerations'; "
            "re-run the prep with --centred-velocity."
        )
    accel_full = np.asarray(npz["accelerations"]) if accel_mode == "fit" else None
    has_accel_full = (
        np.asarray(npz["has_acceleration"]).astype(bool).reshape(-1)
        if accel_mode == "fit" else None
    )

    # Long-range chained positions (prep chain-targets-npz.py): each dynamic
    # row's starting curve is fitted through them over the widest window both
    # sides of its chain reach, and its lifetime widened to match.
    if init_chain_fit and ("chain_pos" not in npz or "chain_offsets_frames" not in npz):
        raise ValueError(
            f"init_chain_fit needs 'chain_pos' and 'chain_offsets_frames' in {npz_path}; "
            "add them with prep/init/chain-targets-npz.py."
        )
    chain_full = np.asarray(npz["chain_pos"]) if init_chain_fit else None
    chain_offsets = (
        np.asarray(npz["chain_offsets_frames"]).reshape(-1).astype(np.int64) if init_chain_fit else None
    )

    source_frames_full: np.ndarray | None = None

    if source_frame_start is not None or source_frame_end is not None:
        if source_frame_start is None or source_frame_end is None:
            raise ValueError(
                "source_frame_start and source_frame_end must be provided together."
            )

        source_frames_full = _require_source_frames(npz, n_before)

        # A filename is not a coverage guarantee: an f60 NPZ can have been
        # generated from only 30 frames. Reject extrapolation of dynamic
        # seeds before expensive initialization or an empty renderer call.
        # Use declared source bounds, not the last sparse keyframe (which
        # normally precedes the end by up to keyframe_step - 1 frames).
        declared_start = _scalar_int(npz, "source_frame_start", None)
        declared_end = _scalar_int(npz, "source_frame_end", None)
        if (
            np.any(is_static_full <= 0.5)
            and declared_start is not None
            and declared_end is not None
            and (
                int(source_frame_start) < declared_start
                or int(source_frame_end) > declared_end
            )
        ):
            raise ValueError(
                f"Requested source frames {source_frame_start}..{source_frame_end} "
                f"exceed NPZ source coverage {declared_start}..{declared_end}: "
                f"{npz_path}. Regenerate the NPZ for the requested range or "
                "restrict --start-frame/--end-frame to its declared coverage."
            )

        source_grid_start = _infer_source_grid_start(npz, source_frames_full)
        npz_keyframe_step = _scalar_int(npz, "keyframe_step", keyframe_stride)
        npz_keyframe_step = max(int(npz_keyframe_step or keyframe_stride), 1)

        if (int(source_frame_start) - int(source_grid_start)) % npz_keyframe_step != 0:
            raise ValueError(
                f"Requested source_frame_start={source_frame_start} is incompatible "
                f"with NPZ source grid start={source_grid_start}, "
                f"keyframe_step={npz_keyframe_step}. Use chunk starts aligned "
                f"to the macro ROMA grid."
            )

        is_static_bool_full = is_static_full.reshape(-1) > 0.5

        range_keep = (
            (source_frames_full >= int(source_frame_start))
            & (source_frames_full <= int(source_frame_end))
        )

        if filter_to_local_keyframe_stride:
            stride = max(int(keyframe_stride), 1)
            stride_keep = ((source_frames_full - int(source_frame_start)) % stride) == 0
            if local_stride_full is not None:
                # Speed-aware sub-keyframes (prep-roma-trigs-memory.py --subkeyframe-min-speed):
                # a row with local_stride L < stride lives on its own L-frame grid.
                ls = np.clip(local_stride_full, 1, stride)
                stride_keep |= (ls < stride) & (((source_frames_full - int(source_frame_start)) % ls) == 0)
        else:
            stride_keep = np.ones_like(range_keep, dtype=bool)

        dynamic_keep = range_keep & stride_keep & (~is_static_bool_full)

        if keep_global_static_when_filtering:
            keep_mask = dynamic_keep | is_static_bool_full
        else:
            keep_mask = dynamic_keep

        dynamic_total = int((~is_static_bool_full).sum())
        dynamic_kept = int(dynamic_keep.sum())
        if dynamic_total > 0 and dynamic_kept <= 0:
            raise ValueError(
                f"No dynamic rows remain after filtering NPZ to "
                f"{source_frame_start}-{source_frame_end}. This usually means "
                f"the training range is not aligned to the macro ROMA frame-step grid."
            )

        if int(keep_mask.sum()) <= 0:
            raise ValueError(
                f"No rows remain after filtering NPZ to "
                f"{source_frame_start}-{source_frame_end}."
            )

        if dynamic_kept > 0:
            dynamic_source_summary = (
                f"dynamic_source_min={int(source_frames_full[dynamic_keep].min())} "
                f"dynamic_source_max={int(source_frames_full[dynamic_keep].max())}"
            )
        else:
            dynamic_source_summary = "dynamic_source_min=n/a dynamic_source_max=n/a"

        log.debug(
            "[from_colmap] "
            f"requested={source_frame_start}-{source_frame_end} "
            f"total_before={n_before:,} "
            f"kept={int(keep_mask.sum()):,} "
            f"kept_static={int((keep_mask & is_static_bool_full).sum()):,} "
            f"kept_dynamic={int((keep_mask & ~is_static_bool_full).sum()):,} "
            f"{dynamic_source_summary} "
            f"source_grid_start={source_grid_start} "
            f"npz_keyframe_step={npz_keyframe_step} "
            f"keep_global_static={keep_global_static_when_filtering}",
        )
    else:
        keep_mask = np.ones((n_before,), dtype=bool)
        if "source_frames" in npz:
            source_frames_full = np.asarray(npz["source_frames"]).reshape(-1).astype(np.int64)

        log.debug(
            "[from_colmap ModeB filter] no source-frame filter requested; "
            f"using all {n_before:,} rows",
        )

    positions = torch.from_numpy(np.asarray(npz["positions"])[keep_mask]).float().to(dev)
    velocities = torch.from_numpy(np.asarray(npz["velocities"])[keep_mask]).float().to(dev)
    colors = torch.from_numpy(np.asarray(npz["colors"])[keep_mask]).float().to(dev)
    times_norm = torch.from_numpy(np.asarray(npz["times"])[keep_mask]).float().to(dev)

    is_static_np = is_static_full[keep_mask]
    dynamic_protect_np = dynamic_protect_full[keep_mask]
    dynamic_allowed_np = dynamic_allowed_full[keep_mask]
    stage_keep_np = stage_keep_full[keep_mask]

    if durations_full is not None:
        durations_np = durations_full[keep_mask]
    else:
        durations_np = None

    if source_frames_full is not None:
        source_frames_np = source_frames_full[keep_mask]
    else:
        source_frames_np = None

    stride_meta = max(int(keyframe_stride), 1)
    local_stride_np = (
        np.clip(local_stride_full[keep_mask], 1, stride_meta) if local_stride_full is not None else None
    )

    n_rows = int(positions.shape[0])

    is_static = torch.from_numpy(is_static_np).to(dev)
    dynamic_protect = torch.from_numpy(dynamic_protect_np).to(dev)
    dynamic_allowed = torch.from_numpy(dynamic_allowed_np).to(dev)
    stage_keep = torch.from_numpy(stage_keep_np).to(dev)

    static_bool = is_static > 0.5
    static_flat = static_bool.reshape(-1)
    dynamic_flat = ~static_flat

    # ------------------------------------------------------------------
    # Time setup.
    # ------------------------------------------------------------------
    if source_frame_start is not None and source_frames_np is not None:
        local_offsets = (
            torch.from_numpy(source_frames_np.reshape(-1, 1).astype(np.float32)).to(dev)
            - float(source_frame_start)
        )
        times = local_offsets / max(float(fps), 1e-6)

        n_frames_local = int(source_frame_end) - int(source_frame_start) + 1
        total_sec = max(n_frames_local - 1, 1) / max(float(fps), 1e-6)
    else:
        n_frames = (
            int(npz.get("frame_end", np.asarray(-1)))
            - int(npz.get("frame_start", np.asarray(0)))
            + 1
        )
        if n_frames > 1:
            total_sec = (n_frames - 1) / max(float(fps), 1e-6)
        else:
            total_sec = 1.0 / max(float(fps), 1e-6)

        times = times_norm * total_sec

    # Static rows are global context. Put them at the middle of the local
    # chunk and widen duration below.
    if bool(static_flat.any()):
        times = torch.where(
            static_bool,
            torch.full_like(times, 0.5 * float(total_sec)),
            times,
        )

    # ------------------------------------------------------------------
    # Velocities: m/frame -> m/second.
    # ------------------------------------------------------------------
    velocity_units = str(npz.get("velocity_units", "world_per_video_frame"))
    keyframe_step_meta = int(npz.get("keyframe_step", keyframe_stride))

    if "world_per_video_frame" in velocity_units:
        nu = velocities * fps
        per_second = float(fps)
    elif "world_per_keyframe_interval" in velocity_units:
        nu = velocities * (fps / max(keyframe_step_meta, 1))
        per_second = float(fps) / max(keyframe_step_meta, 1)
    else:
        raise ValueError(f"Unknown velocity_units: {velocity_units}")

    # The twist fit below replaces nu; keep the per-row measured velocity
    # (world/s) as the target the starting curve must reproduce.
    v_measured = nu.clone()
    if accel_full is not None:
        accel = (
            torch.from_numpy(accel_full[keep_mask]).float().to(dev) * per_second * per_second
        )
        has_accel = torch.from_numpy(has_accel_full[keep_mask]).to(dev)
        accel_smooth = torch.zeros_like(accel)

    log.debug(f"[from_colmap] pre-twist mean|nu|={ (velocities*fps).norm(dim=-1).mean().item():.4f} world/s "
      f"(raw per-frame {velocities.norm(dim=-1).mean().item():.5f})")

    # ------------------------------------------------------------------
    # Dynamic-only anchors/scales/twist init.
    # Static rows skip this expensive path.
    # ------------------------------------------------------------------
    anchors = positions.clone()
    omega = torch.zeros_like(positions)
    scales_out = torch.zeros(n_rows, 3, device=dev)

    # Group dynamic rows by keyframe time once (ascending row order within
    # each group, as the old per-time mask produced) instead of scanning all
    # rows for every keyframe, which is O(rows x keyframes) on long sequences.
    dyn_rows = dynamic_flat.nonzero(as_tuple=False).squeeze(-1)
    if int(dyn_rows.numel()) > 0:
        _, inverse = torch.unique(
            times[dyn_rows].squeeze(-1), sorted=True, return_inverse=True,
        )
        order = torch.argsort(inverse, stable=True)
        groups = order.split(torch.bincount(inverse).tolist())
    else:
        groups = ()

    # "speed" lifetimes: per dynamic row, the time over which its starting
    # curve travels init_sigma_t_speed_spacings local point spacings.
    speed_mode = str(init_sigma_t_mode).lower() == "speed"
    sigma_speed = torch.full((n_rows,), float("nan"), device=dev) if speed_mode else None

    for group in groups:
        idx = dyn_rows[group]

        if int(idx.numel()) <= 0:
            continue

        kf_pos = positions[idx].to(cdev)
        kf_vel = nu[idx].to(cdev)

        knn_d = _knn_distance(kf_pos, k=scale_knn_k)
        log_scale = torch.log(torch.clamp(knn_d * scale_knn_factor, min=1e-6))
        scales_out[idx] = log_scale.unsqueeze(-1).expand(-1, 3).to(dev)

        if kf_pos.shape[0] > 1:
            ext = (
                kf_pos.max(dim=0).values - kf_pos.min(dim=0).values
            ).max().clamp(min=1e-6)

            radius = anchor_radius_frac * ext.item()
            kf_anchors = _radius_ball_centroid(kf_pos, radius)
            anchors[idx] = kf_anchors.to(dev)

            nn_idx = local_knn_indices(kf_pos, local_twist_k)
            nu_k, om_k, anchors_k = fit_local_twist(
                kf_pos,
                kf_vel,
                kf_anchors,
                nn_idx,
                reg=local_twist_reg,
            )
            omega[idx] = om_k.to(dev)
            nu[idx] = nu_k.to(dev)
            anchors[idx] = anchors_k.to(dev)

            if accel_full is not None:
                accel_smooth[idx] = (
                    smooth_accelerations(
                        accel[idx].to(cdev), has_accel[idx].to(cdev), nn_idx,
                    ) * float(init_accel_scale)
                ).to(dev)

            if speed_mode:
                # Lifetime over which the starting curve moves the row
                # init_sigma_t_speed_spacings local spacings: solve
                # |v| s + 1/2 |a| s^2 = K d (the accel term grows as s^2, so a
                # velocity-only bound lets long lives drift). Neighbourhood
                # medians, so one noisy track neither makes a still point
                # short-lived nor a moving point long-lived.
                def _nb_median(x: Tensor) -> Tensor:
                    return torch.cat([x.unsqueeze(1), x[nn_idx]], dim=1).median(dim=1).values

                spd = _nb_median(v_measured[idx].to(cdev).norm(dim=-1))
                if accel_full is not None:
                    acc = _nb_median(accel_smooth[idx].to(cdev).norm(dim=-1))
                else:
                    acc = torch.zeros_like(spd)
                kd = float(init_sigma_t_speed_spacings) * knn_d
                sigma_speed[idx] = (
                    2.0 * kd / (spd + torch.sqrt(spd * spd + 2.0 * acc * kd)).clamp_min(1e-9)
                ).to(dev)

    # Static rows: cheap deterministic init.
    if bool(static_flat.any()):
        static_idx = static_flat.nonzero(as_tuple=False).squeeze(-1)

        anchors[static_idx] = positions[static_idx]
        nu[static_idx] = 0.0
        omega[static_idx] = 0.0

        if bool(dynamic_flat.any()) and scales_out[dynamic_flat].numel() > 0:
            fallback_scale = scales_out[dynamic_flat].median(dim=0).values
        else:
            fallback_scale = torch.full((3,), math.log(0.01), device=dev)

        scales_out[static_idx] = fallback_scale.reshape(1, 3)

    # Enforce static motion zero.
    nu = torch.where(static_bool.expand_as(nu), torch.zeros_like(nu), nu)
    omega = torch.where(static_bool.expand_as(omega), torch.zeros_like(omega), omega)

    quats = torch.zeros(n_rows, 4, device=dev)
    quats[:, 0] = 1.0

    # Opacity: logit(init_opacity).
    op = max(min(float(init_opacity), 1.0 - 1e-6), 1e-6)
    opacities = torch.full((n_rows,), math.log(op / (1.0 - op)), device=dev)

    # SH-DC colors.
    sh0 = (colors - 0.5) / _SH_C0
    if sh0.dim() > 2:
        sh0 = sh0.unsqueeze(1).squeeze(1)

    # Higher-order SH bands zero-init.
    n_shn_bands = (int(max_sh_degree) + 1) ** 2 - 1
    shN = torch.zeros(n_rows, n_shn_bands, 3, device=dev)

    # Durations.
    # Community-port-faithful probe: ignore NPZ speed-adaptive durations.
    # The original community trainer initializes σ_t uniformly:
    #   init_sigma_t if provided, else keyframe_stride / fps.
    if init_sigma_t > 0:
        sigma_t_init = float(init_sigma_t)
    else:
        sigma_t_init = max(int(keyframe_stride), 1) / max(float(fps), 1e-6)

    # Rows on a finer local grid (local_stride L < keyframe_stride) get lifetimes
    # scaled by L / keyframe_stride, so a sub-keyframe row covers its own L-frame
    # gap the way a keyframe row covers keyframe_stride. 1.0 for every row when
    # the NPZ has no local_stride.
    if local_stride_np is not None:
        sigma_scale = torch.from_numpy(
            (local_stride_np.astype(np.float32) / float(stride_meta)).reshape(-1, 1)
        ).to(dev)
    else:
        sigma_scale = torch.ones((n_rows, 1), device=dev)
    sigma_init_row = sigma_t_init * sigma_scale
    sigma_t = sigma_init_row.clone()

    mode = str(init_sigma_t_mode).lower()
    if mode == "npz":
        if durations_np is None:
            raise ValueError(f"init_sigma_t_mode='npz' but {npz_path} has no 'durations'")
        npz_sigma = _npz_durations_seconds(npz, durations_np, fps)
        floor = float(init_sigma_t_floor_frac) * sigma_t_init
        sigma_t = torch.minimum(
            torch.maximum(
                torch.from_numpy(npz_sigma.reshape(-1, 1)).float().to(dev),
                float(init_sigma_t_floor_frac) * sigma_init_row,
            ),
            sigma_init_row,
        )
        dyn_sig = sigma_t[dynamic_flat].reshape(-1).cpu()
        if int(dyn_sig.numel()) > 0:
            q = torch.quantile(dyn_sig[:1_000_000].float(), torch.tensor([0.1, 0.5, 0.9]))
            log.debug(
                f"[from_colmap] NPZ speed-scaled durations clamped to "
                f"[{floor:.4f}, {sigma_t_init:.4f}]s: dynamic sigma_t p10/50/90 "
                f"{q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}s; at floor "
                f"{float((dyn_sig <= (float(init_sigma_t_floor_frac) * sigma_init_row)[dynamic_flat].reshape(-1).cpu() + 1e-7).float().mean()):.1%}",
            )
    elif mode == "uniform":
        log.debug(
            f"[from_colmap] IGNORING NPZ durations; dynamic sigma_t_init={sigma_t_init:.6f}s",
        )
    elif mode == "speed":
        # sigma from |v| s + 1/2 |a| s^2 = K * local spacing: a row's starting
        # curve carries it at most K spacings over +/-1 sigma, so slow and still
        # rows live long and motion noise cannot turn a long life into large
        # drift. Fast rows keep
        # the npz floor (adjacent keyframes still cross-fade); still rows stop at
        # init_sigma_t_max so a wall is not dozens of overlapping copies.
        floor = float(init_sigma_t_floor_frac) * sigma_t_init
        cap = max(float(init_sigma_t_max), floor)
        assert sigma_speed is not None
        fill = torch.isnan(sigma_speed)
        sig = torch.where(fill, sigma_init_row.reshape(-1), sigma_speed)
        sigma_t = torch.minimum(
            torch.maximum(sig.reshape(-1, 1), float(init_sigma_t_floor_frac) * sigma_init_row),
            torch.full_like(sigma_init_row, cap),
        )
        dyn_sig = sigma_t[dynamic_flat].reshape(-1).cpu()
        if int(dyn_sig.numel()) > 0:
            q = torch.quantile(dyn_sig[:1_000_000].float(), torch.tensor([0.1, 0.5, 0.9]))
            log.debug(
                f"[from_colmap] speed-aware lifetimes ({float(init_sigma_t_speed_spacings):g} "
                f"spacings / speed) clamped to [{floor:.4f}, {cap:.4f}]s: dynamic sigma_t "
                f"p10/50/90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}s; at floor "
                f"{float((dyn_sig <= floor + 1e-7).float().mean()):.1%}, at cap "
                f"{float((dyn_sig >= cap - 1e-7).float().mean()):.1%}",
            )
    else:
        raise ValueError(
            f"init_sigma_t_mode must be 'uniform', 'npz' or 'speed', got {init_sigma_t_mode!r}"
        )

    if local_stride_np is not None:
        sub = torch.from_numpy(local_stride_np < stride_meta).to(dev) & dynamic_flat
        n_sub = int(sub.sum().item())
        if n_sub > 0:
            sf_sub = source_frames_np[local_stride_np < stride_meta] if source_frames_np is not None else None
            off_grid = (
                int((((sf_sub - int(source_frame_start)) % stride_meta) != 0).sum())
                if sf_sub is not None and source_frame_start is not None else -1
            )
            qs = torch.quantile(sigma_t[sub].reshape(-1)[:1_000_000].float().cpu() * float(fps),
                                torch.tensor([0.1, 0.5, 0.9]))
            log.info(
                f"Speed-aware sub-keyframes: {n_sub:,} dynamic rows with local_stride < "
                f"{stride_meta} ({off_grid:,} between keyframes); their sigma_t p10/50/90 "
                f"{qs[0]:.2f}/{qs[1]:.2f}/{qs[2]:.2f} frames",
            )

    chain_rows = None
    if chain_full is not None:
        chain = torch.from_numpy(chain_full[keep_mask]).float()
        chain_valid = ~torch.isnan(chain[..., 0])
        offs = torch.from_numpy(chain_offsets)
        step = int(offs[offs > 0].min()) if bool((offs > 0).any()) else 1
        # Chains stop at their first failed link, so the valid entries on each
        # side are contiguous from the keyframe outward.
        fwd = torch.where(chain_valid & (offs > 0), offs, torch.zeros_like(offs)).amax(dim=1)
        bwd = torch.where(chain_valid & (offs < 0), -offs, torch.zeros_like(offs)).amax(dim=1)
        # Centre each row's window on its chain, at a keyframe offset so the new
        # centre is a measured position (the moving subject at keyframe 193 chains 21
        # frames back but ~3 forward), keeping the centre inside the clip.
        mu_frames = (times.squeeze(-1).cpu() * fps).round().long()
        last_frame = int(round(float(total_sec) * fps))
        centre = (((fwd - bwd).float() / (2 * step)).round().long() * step)
        centre = torch.minimum(torch.maximum(centre, -mu_frames), last_frame - mu_frames)
        reach = torch.minimum(centre + bwd, fwd - centre)
        chain_rows = (dynamic_flat.cpu() & (reach >= int(init_chain_min_span_frames))).nonzero().squeeze(-1)
        # Window +-2 sigma spans the chain around its centre.
        sigma_chain = (reach[chain_rows].float() / 2.0 / max(float(fps), 1e-6)).to(dev).unsqueeze(-1)
        sigma_t[chain_rows.to(dev)] = torch.maximum(sigma_t[chain_rows.to(dev)], sigma_chain)
        h = reach[chain_rows].float()
        log.debug(
            f"[from_colmap] init chain fit: {int(chain_rows.numel()):,}/{int(dynamic_flat.sum().item()):,} "
            f"dynamic rows chain >= {int(init_chain_min_span_frames)} frames each side of a centre "
            f"({int((centre[chain_rows] != 0).sum()):,} re-centred); reach p50/p90 "
            f"{float(h.median()) if h.numel() else 0:.0f}/{float(torch.quantile(h, 0.9)) if h.numel() else 0:.0f} frames",
        )

    # Static rows visible over the whole local chunk.
    sigma_t = torch.where(
        static_bool,
        torch.full_like(sigma_t, max(float(total_sec), 1e-6)),
        sigma_t,
    ).clamp(min=1e-6)

    log_durations = torch.log(sigma_t)

    def _zeros3() -> Tensor:
        return torch.zeros(n_rows, 3, device=dev)

    ctrl = {name: _zeros3() for name in (
        "zeta_c0_t", "zeta_c1_t", "zeta_c2_t", "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
    )}
    if accel_full is not None:
        # Start each dynamic row on x + v dt + 1/2 a dt^2 over its window
        # instead of the straight screw line. Keyframes are fitted together in
        # batches of up to accel_fit_batch_rows rows (a per-keyframe loop is
        # launch-bound: 300 iterations x every keyframe); each row weighs
        # 1/len(its keyframe), so the result equals the per-keyframe fit.
        accel_fit_batch_rows = 262_144
        batches, cur, cur_n = [], [], 0
        for group in groups:
            idx = dyn_rows[group]
            if int(idx.numel()) == 0:
                continue
            if cur and cur_n + int(idx.numel()) > accel_fit_batch_rows:
                batches.append(cur)
                cur, cur_n = [], 0
            cur.append(idx)
            cur_n += int(idx.numel())
        if cur:
            batches.append(cur)
        for parts in batches:
            idx = torch.cat(parts)
            row_weight = torch.cat([
                torch.full((int(p.numel()),), 1.0 / int(p.numel()), device=cdev) for p in parts
            ])
            fitted = fit_bezier_to_acceleration(
                positions[idx].to(cdev), anchors[idx].to(cdev), times[idx].to(cdev),
                log_durations[idx].to(cdev), nu[idx].to(cdev), omega[idx].to(cdev),
                v_measured[idx].to(cdev), accel_smooth[idx].to(cdev),
                iters=int(init_accel_fit_iters), row_weight=row_weight,
            )
            for name, value in fitted.items():
                ctrl[name][idx] = value.to(dev)
        n_acc = int(has_accel[dynamic_flat].sum().item()) if bool(dynamic_flat.any()) else 0
        a_mag = accel_smooth[dynamic_flat].norm(dim=-1)[:1_000_000].float().cpu()
        a_q = torch.quantile(a_mag, torch.tensor([0.5, 0.9])) if a_mag.numel() else torch.zeros(2)
        log.debug(
            f"[from_colmap] init accel fit: {n_acc:,}/{int(dynamic_flat.sum().item()):,} dynamic rows "
            f"measured; smoothed x{float(init_accel_scale):g} |a| p50/p90 "
            f"{float(a_q[0]):.3f}/{float(a_q[1]):.3f} world/s^2; "
            f"{int(init_accel_fit_iters)} fit iterations per keyframe",
        )

    if chain_rows is not None and int(chain_rows.numel()) > 0:
        rows_dev = chain_rows.to(dev)
        # Same keyframe batching and 1/len(keyframe) weights as the accel fit.
        _, kf_inverse, kf_counts = torch.unique(
            times[rows_dev].squeeze(-1), return_inverse=True, return_counts=True,
        )
        kf_weight = (1.0 / kf_counts.float())[kf_inverse]
        # Re-centred rows move to their chained position at the new centre time;
        # the anchor moves with them so the local screw geometry is kept, and
        # their keyframe position becomes one more target.
        c_rows = centre[chain_rows]
        moved = c_rows != 0
        keyframe_pos = positions[rows_dev].cpu().clone()
        if bool(moved.any()):
            m = chain_rows[moved]
            col = torch.searchsorted(offs, c_rows[moved])
            delta = (chain[m, col] - keyframe_pos[moved]).to(dev)
            positions[m.to(dev)] += delta
            anchors[m.to(dev)] += delta
            times[m.to(dev)] += (c_rows[moved].float() / max(float(fps), 1e-6)).to(dev).unsqueeze(-1)
        all_offs = torch.cat([offs, offs.new_zeros(1)])
        targets_all = torch.cat([chain[chain_rows], keyframe_pos.unsqueeze(1)], dim=1)
        rel = all_offs.unsqueeze(0) - c_rows.unsqueeze(1)                       # frames from the new centre
        valid_all = torch.cat([chain_valid[chain_rows], moved.unsqueeze(1)], dim=1)
        valid_all &= rel.abs() <= reach[chain_rows].unsqueeze(1)
        dt_all = rel.float() / max(float(fps), 1e-6)
        err_before, err_after = [], []
        for start in range(0, int(chain_rows.numel()), 262_144):
            sl = slice(start, start + 262_144)
            idx = rows_dev[sl]
            args = (
                positions[idx].to(cdev), anchors[idx].to(cdev), times[idx].to(cdev),
                log_durations[idx].to(cdev), nu[idx].to(cdev), omega[idx].to(cdev),
                dt_all[sl].to(cdev), targets_all[sl].to(cdev), valid_all[sl].to(cdev),
            )
            # The accel fit was made around the keyframe; re-centred rows start from a straight screw.
            keep_warm = (~moved[sl]).to(cdev).unsqueeze(-1).float()
            warm = {name: ctrl[name][idx].to(cdev) * keep_warm for name in ctrl}
            fitted = fit_bezier_to_targets(
                *args, init_ctrl=warm, iters=int(init_accel_fit_iters),
                row_weight=kf_weight[sl].to(cdev),
            )
            for name, ctrls in (("before", warm), ("after", fitted)):
                with torch.no_grad():
                    e = torch.stack([
                        (screw_deform_means(*args[:6], ctrls, args[2] + args[6][:, s:s + 1]) - args[7][:, s]).norm(dim=-1)
                        for s in range(int(args[6].shape[1]))
                    ], dim=1)
                    (err_before if name == "before" else err_after).append(e[args[8]].cpu())
            for name, value in fitted.items():
                ctrl[name][idx] = value.to(dev)
        eb, ea = torch.cat(err_before), torch.cat(err_after)
        log.debug(
            f"[from_colmap] init chain fit: distance to chained positions p50/p90 "
            f"{float(eb.median()):.4f}/{float(torch.quantile(eb[:1_000_000], 0.9)):.4f} -> "
            f"{float(ea.median()):.4f}/{float(torch.quantile(ea[:1_000_000], 0.9)):.4f} world units",
        )

    params = nn.ParameterDict({
        "means": nn.Parameter(positions.clone(), requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "scales": nn.Parameter(scales_out, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "times": nn.Parameter(times, requires_grad=True),
        "log_durations": nn.Parameter(log_durations, requires_grad=True),
        "anchors": nn.Parameter(anchors, requires_grad=True),
        "zeta_base_t": nn.Parameter(nu.detach().clone(), requires_grad=True),
        "zeta_base_r": nn.Parameter(omega.detach().clone(), requires_grad=True),
        "zeta_c0_t": nn.Parameter(ctrl["zeta_c0_t"], requires_grad=True),
        "zeta_c0_r": nn.Parameter(ctrl["zeta_c0_r"], requires_grad=True),
        "zeta_c1_t": nn.Parameter(ctrl["zeta_c1_t"], requires_grad=True),
        "zeta_c1_r": nn.Parameter(ctrl["zeta_c1_r"], requires_grad=True),
        "zeta_c2_t": nn.Parameter(ctrl["zeta_c2_t"], requires_grad=True),
        "zeta_c2_r": nn.Parameter(ctrl["zeta_c2_r"], requires_grad=True),
        "is_static": nn.Parameter(is_static, requires_grad=False),
        "dynamic_protect": nn.Parameter(dynamic_protect, requires_grad=False),
        "dynamic_allowed": nn.Parameter(dynamic_allowed, requires_grad=False),
        "stage_keep": nn.Parameter(stage_keep, requires_grad=False),
    })

    scene_extent = float((
        positions.max(dim=0).values - positions.min(dim=0).values
    ).max().clamp(min=1e-6))

    dynamic_allowed_flat = dynamic_allowed.reshape(-1) > 0.5
    stage_keep_flat = stage_keep.reshape(-1) > 0.5

    log.debug(
        "[from_colmap] "
        f"N={n_rows:,} "
        f"static={int(static_flat.sum().item()):,} "
        f"dynamic={int(dynamic_flat.sum().item()):,} "
        f"protected={int((dynamic_protect.reshape(-1) > 0.5).sum().item()):,} "
        f"dynamic_allowed={int(dynamic_allowed_flat.sum().item()):,} "
        f"dynamic_outside_allowed={int((dynamic_flat & ~dynamic_allowed_flat).sum().item()):,} "
        f"stage_keep={int(stage_keep_flat.sum().item()):,} "
        f"scene_extent={scene_extent:.3f}",
    )

    return params, int(max_sh_degree), scene_extent


@torch.no_grad()
def release_static_rows_to_dynamic(params: nn.ParameterDict) -> int:
    """Turn static rows into ordinary dynamic rows, keeping their static init.

    The builder starts static rows motionless, centred mid-clip and visible for
    the whole clip. As dynamic rows they keep that start but can move, be
    recycled by relocation and act as relocation donors, so background
    capacity can grow at the background's own depth (the official model has
    no static class). dynamic_allowed is raised too, or the trainer would
    demote them straight back to static. Returns the number of rows released.
    """
    if "is_static" not in params:
        return 0
    static = params["is_static"].data.reshape(-1) > 0.5
    n_released = int(static.sum().item())
    if n_released == 0:
        return 0
    params["is_static"].data[static] = 0.0
    if "dynamic_allowed" in params:
        params["dynamic_allowed"].data[static] = 1.0
    return n_released
