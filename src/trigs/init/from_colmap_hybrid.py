"""SelfCap-style hybrid initializer for COLMAP/RoMa NPZ pairs.

This treats two RoMa prep NPZs as the COLMAP equivalent of SelfCap's
``dense_pcds`` and ``pcds`` folders:

  * ``anchor_npz``   == dense_pcds  (preferred when present at a source frame)
  * ``coverage_npz`` == pcds        (fallback when no anchor frame exists)

The important part is ordering: dense/sparse resolution happens before scale
init, anchor init, local twist init, optional FPS, and packing. This mirrors
``from_selfcap.build_canonical_from_pcd_sequence`` more closely than building a
merged NPZ generically and retroactively applying a ``kind`` tag afterwards.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor

from .from_colmap import _SH_C0, _knn_distance, _radius_ball_centroid
from ..math.twist_init import fit_local_twist, local_knn_indices

log = logging.getLogger(__name__)


_REQUIRED = ("positions", "colors", "velocities", "source_frames")
_OPTIONAL_FLAGS = ("is_static", "dynamic_protect", "dynamic_allowed", "stage_keep")

def _durations_to_seconds(
    src: _LoadedNPZ,
    idx: np.ndarray,
    fps: float,
) -> np.ndarray | None:
    if src.durations is None:
        return None

    d = src.durations[idx].astype(np.float32).reshape(-1, 1)

    if "normalized_0_1" in src.time_units:
        # Matches prep's denom = frame_end - frame_start.
        source_span_frames = max(int(src.source_frame_end) - int(src.source_frame_start), 1)
        source_total_sec = float(source_span_frames) / max(float(fps), 1e-6)
        d = d * source_total_sec
    elif "second" in src.time_units or "seconds" in src.time_units:
        pass
    else:
        # Conservative fallback: old NPZs most likely used normalized time.
        source_span_frames = max(int(src.source_frame_end) - int(src.source_frame_start), 1)
        source_total_sec = float(source_span_frames) / max(float(fps), 1e-6)
        d = d * source_total_sec

    return d.astype(np.float32)

class _LoadedNPZ:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as f:
            self.data = {k: f[k] for k in f.files}

        for k in _REQUIRED:
            if k not in self.data:
                raise KeyError(f"{self.path}: missing required key {k!r}")

        self.n = int(len(self.data["positions"]))
        for k, v in self.data.items():
            # Scalars like fps/frame_start are 0-d; only validate row arrays.
            if np.ndim(v) > 0 and len(v) == self.n:
                continue
            if k in _REQUIRED or k in _OPTIONAL_FLAGS or k in ("has_velocity", "durations"):
                raise RuntimeError(
                    f"{self.path}: row field {k!r} has len={len(v)}, expected {self.n}"
                )

        self.positions = np.asarray(self.data["positions"], dtype=np.float32)
        self.colors = np.asarray(self.data["colors"], dtype=np.float32)
        self.velocities = np.asarray(self.data["velocities"], dtype=np.float32)
        self.source_frames = np.asarray(self.data["source_frames"]).reshape(-1).astype(np.int64)

        self.is_static = self._flag("is_static", default=False)
        self.dynamic_protect = self._flag("dynamic_protect", default=False)
        self.dynamic_allowed = self._flag("dynamic_allowed", default=None)
        self.stage_keep = self._flag("stage_keep", default=True)

        # Backward-compatible dynamic_allowed fallback from from_colmap.py.
        if self.dynamic_allowed is None:
            self.dynamic_allowed = ~self.is_static

        self.velocity_units = str(np.asarray(self.data.get("velocity_units", "world_per_video_frame")).reshape(()))
        self.keyframe_step = int(np.asarray(self.data.get("keyframe_step", 1)).reshape(()))

        self.durations = (
            np.asarray(self.data["durations"], dtype=np.float32).reshape(-1, 1)
            if "durations" in self.data
            else None
        )

        self.has_velocity = (
            np.asarray(self.data["has_velocity"]).reshape(-1).astype(bool)
            if "has_velocity" in self.data
            else np.ones((self.n,), dtype=bool)
        )

        self.time_units = str(
            np.asarray(self.data.get("time_units", "normalized_0_1_over_source_frame_range")).reshape(())
        )

        self.source_frame_start = int(
            np.asarray(self.data.get("source_frame_start", self.source_frames.min())).reshape(())
        )
        self.source_frame_end = int(
            np.asarray(self.data.get("source_frame_end", self.source_frames.max())).reshape(())
        )

    def _flag(self, key: str, default: Optional[bool]) -> Optional[np.ndarray]:
        if key not in self.data:
            if default is None:
                return None
            return np.full((self.n,), bool(default), dtype=bool)
        return np.asarray(self.data[key]).reshape(-1).astype(bool)

    def dynamic_indices_by_frame(self) -> Dict[int, np.ndarray]:
        dyn = ~self.is_static
        out: Dict[int, np.ndarray] = {}
        for f in np.unique(self.source_frames[dyn]):
            idx = np.nonzero(dyn & (self.source_frames == int(f)))[0]
            if len(idx) > 0:
                out[int(f)] = idx
        return out

    def static_indices(self) -> np.ndarray:
        return np.nonzero(self.is_static)[0]


def _velocity_to_world_per_second(src: _LoadedNPZ, idx: np.ndarray, fps: float) -> np.ndarray:
    v = src.velocities[idx]
    units = src.velocity_units
    if "world_per_video_frame" in units:
        return v * float(fps)
    if "world_per_keyframe_interval" in units:
        return v * (float(fps) / max(int(src.keyframe_step), 1))
    raise ValueError(f"{src.path}: unknown velocity_units={units!r}")


def _as_tensor(x: np.ndarray, dev: torch.device, dtype=torch.float32) -> Tensor:
    return torch.from_numpy(np.asarray(x)).to(device=dev, dtype=dtype)


@torch.no_grad()
def _append_rows(
    *,
    src: _LoadedNPZ,
    idx_np: np.ndarray,
    frame_offset: float,
    kind: str,
    fps: float,
    dev: torch.device,
    all_means: List[Tensor],
    all_colors: List[Tensor],
    all_times: List[Tensor],
    all_log_scales: List[Tensor],
    all_nu: List[Tensor],
    all_kinds: List[str],
    all_flag_is_static: List[Tensor],
    all_flag_dynamic_protect: List[Tensor],
    all_flag_dynamic_allowed: List[Tensor],
    all_flag_stage_keep: List[Tensor],
    all_source_frames: List[Tensor],
    scale_knn_k: int,
    scale_knn_factor: float,
    all_durations: List[Tensor],
) -> None:
    xyz = _as_tensor(src.positions[idx_np], dev)
    rgb = _as_tensor(src.colors[idx_np], dev).clamp(0.0, 1.0)
    n = int(xyz.shape[0])
    if n == 0:
        return

    knn_d = _knn_distance(xyz, k=scale_knn_k)
    log_scale = torch.log(torch.clamp(knn_d * float(scale_knn_factor), min=1e-6))

    all_means.append(xyz)
    all_colors.append(rgb)
    all_log_scales.append(log_scale.unsqueeze(-1).expand(-1, 3))
    all_times.append(torch.full((n, 1), float(frame_offset) / max(float(fps), 1e-6), device=dev))
    all_kinds.append(kind)
    all_source_frames.append(_as_tensor(src.source_frames[idx_np].astype(np.int64), dev, dtype=torch.long).reshape(-1, 1))

    if kind in {"dense", "sparse"}:
        all_nu.append(
            _as_tensor(_velocity_to_world_per_second(src, idx_np, fps), dev)
        )
    else:
        raise ValueError(f"unknown kind={kind!r}")

    all_flag_is_static.append(_as_tensor(src.is_static[idx_np].astype(np.float32).reshape(-1, 1), dev))
    all_flag_dynamic_protect.append(_as_tensor(src.dynamic_protect[idx_np].astype(np.float32).reshape(-1, 1), dev))
    all_flag_dynamic_allowed.append(_as_tensor(src.dynamic_allowed[idx_np].astype(np.float32).reshape(-1, 1), dev))
    all_flag_stage_keep.append(_as_tensor(src.stage_keep[idx_np].astype(np.float32).reshape(-1, 1), dev))

    dur_s = _durations_to_seconds(src, idx_np, fps)
    if dur_s is None:
        all_durations.append(torch.full((n, 1), float("nan"), device=dev))
    else:
        all_durations.append(_as_tensor(dur_s, dev).reshape(-1, 1))


@torch.no_grad()
def _apply_selfcap_fps(
    *,
    dev: torch.device,
    target_count: int,
    fps_chunk_size: int,
    lists: Tuple[List[Tensor], ...],
) -> None:
    """In-place count-weighted per-keyframe FPS, matching from_selfcap.py."""
    all_means = lists[0]
    total = sum(int(m.shape[0]) for m in all_means)
    if target_count <= 0 or total <= target_count:
        return

    counts = torch.tensor([m.shape[0] for m in all_means], dtype=torch.float32)
    allocations = (counts / counts.sum() * int(target_count)).long()
    allocations[-1] += int(target_count) - int(allocations.sum().item())
    keep_counts = [
        min(max(int(allocations[i].item()), 0), int(all_means[i].shape[0]))
        for i in range(len(all_means))
    ]

    try:
        from pytorch3d.ops import sample_farthest_points

        chunk_size = max(int(fps_chunk_size), 1)
        for chunk_start in range(0, len(all_means), chunk_size):
            chunk_ids = list(range(chunk_start, min(chunk_start + chunk_size, len(all_means))))
            active_ids = [i for i in chunk_ids if 0 < keep_counts[i] < int(all_means[i].shape[0])]
            if not active_ids:
                continue
            p_max = max(int(all_means[i].shape[0]) for i in active_ids)
            pts = torch.zeros((len(active_ids), p_max, 3), device=dev, dtype=all_means[active_ids[0]].dtype)
            lengths = torch.zeros((len(active_ids),), device=dev, dtype=torch.long)
            k_tensor = torch.zeros((len(active_ids),), device=dev, dtype=torch.long)
            for bi, i in enumerate(active_ids):
                n_i = int(all_means[i].shape[0])
                pts[bi, :n_i] = all_means[i]
                lengths[bi] = n_i
                k_tensor[bi] = keep_counts[i]
            _, idx = sample_farthest_points(pts, lengths=lengths, K=k_tensor, random_start_point=False)
            for bi, i in enumerate(active_ids):
                keep_idx = idx[bi, : keep_counts[i]]
                for arr_list in lists:
                    arr_list[i] = arr_list[i][keep_idx]
    except ImportError:
        for i in range(len(all_means)):
            n_i = int(all_means[i].shape[0])
            keep = keep_counts[i]
            if keep < n_i:
                keep_idx = torch.randperm(n_i, device=dev)[:keep]
                for arr_list in lists:
                    arr_list[i] = arr_list[i][keep_idx]

    # Explicit empty selections for 0-allocation frames.
    for i in range(len(all_means)):
        if keep_counts[i] == 0 and int(all_means[i].shape[0]) > 0:
            empty = torch.empty((0,), device=dev, dtype=torch.long)
            for arr_list in lists:
                arr_list[i] = arr_list[i][empty]

    log.info(
        "[from_colmap_selfcap_hybrid] SelfCap FPS: %s -> %s dynamic primitives",
        f"{total:,}",
        f"{target_count:,}",
    )


def build_canonical_from_npz_pair_as_selfcap(
    *,
    anchor_npz_path: str | Path,
    coverage_npz_path: str | Path,
    fps: float = 60.0,
    keyframe_stride: int = 1,
    source_frame_start: int,
    source_frame_end: int,
    static_from: str = "anchor",
    target_count: Optional[int] = None,
    fps_chunk_size: int = 16,
    init_opacity: float = 0.1,
    scale_knn_k: int = 4,
    scale_knn_factor: float = 0.3,
    anchor_radius_frac: float = 0.1,
    local_twist_k: int = 16,
    local_twist_reg: float = 1e-4,
    init_sigma_t: float = 0.0,
    max_sh_degree: int = 3,
    omega_init: bool = True,
    duration_init_mode: str = "class",
    duration_min_s: float = 0.0,
    duration_max_s: float = 0.0,
    dense_duration_multiplier: float = 5.0,
    sparse_duration_multiplier: float = 1.0,
    device: str = "auto",
    verbose: bool = True,
) -> Tuple[nn.ParameterDict, int, float]:
    """Build COLMAP params by treating two NPZs like SelfCap dense/sparse PCDs.

    Dynamic rows are resolved per source frame with SelfCap's dense-preferred
    fallback policy: if the anchor NPZ has dynamic rows at a training keyframe,
    those rows are used and coverage rows for that same frame are ignored;
    otherwise coverage rows are used.

    Static rows are COLMAP-specific background context, not present in SelfCap.
    They are appended once from ``static_from`` and initialized with the same
    static/background policy as ``from_colmap.py``.
    """
    if static_from not in {"anchor", "coverage", "both", "none"}:
        raise ValueError("static_from must be one of: anchor, coverage, both, none")

    dev = resolve_device(device)
    anchor = _LoadedNPZ(anchor_npz_path)
    coverage = _LoadedNPZ(coverage_npz_path)

    stride = max(int(keyframe_stride), 1)
    fps_safe = max(float(fps), 1e-6)
    n_frames_local = int(source_frame_end) - int(source_frame_start) + 1
    if n_frames_local <= 0:
        raise ValueError("source_frame_end must be >= source_frame_start")
    total_sec = max(n_frames_local - 1, 1) / fps_safe

    anchor_dyn = anchor.dynamic_indices_by_frame()
    coverage_dyn = coverage.dynamic_indices_by_frame()

    all_means: List[Tensor] = []
    all_colors: List[Tensor] = []
    all_times: List[Tensor] = []
    all_log_scales: List[Tensor] = []
    all_nu: List[Tensor] = []
    all_kinds: List[str] = []
    all_source_frames: List[Tensor] = []
    all_is_static: List[Tensor] = []
    all_dynamic_protect: List[Tensor] = []
    all_dynamic_allowed: List[Tensor] = []
    all_stage_keep: List[Tensor] = []
    all_durations: List[Tensor] = []

    selected_dense = 0
    selected_sparse = 0
    missing_frames: List[int] = []

    # SelfCap-style keyframe resolver: dense/anchor preferred, sparse/coverage fallback.
    for src_f in range(int(source_frame_start), int(source_frame_end) + 1, stride):
        f_off = int(src_f) - int(source_frame_start)
        if src_f in anchor_dyn:
            idx = anchor_dyn[src_f]
            _append_rows(
                src=anchor,
                idx_np=idx,
                frame_offset=f_off,
                kind="dense",
                fps=fps_safe,
                dev=dev,
                all_means=all_means,
                all_colors=all_colors,
                all_times=all_times,
                all_log_scales=all_log_scales,
                all_nu=all_nu,
                all_kinds=all_kinds,
                all_flag_is_static=all_is_static,
                all_flag_dynamic_protect=all_dynamic_protect,
                all_flag_dynamic_allowed=all_dynamic_allowed,
                all_flag_stage_keep=all_stage_keep,
                all_source_frames=all_source_frames,
                scale_knn_k=scale_knn_k,
                scale_knn_factor=scale_knn_factor,
                all_durations=all_durations,
            )
            selected_dense += 1
        elif src_f in coverage_dyn:
            idx = coverage_dyn[src_f]
            _append_rows(
                src=coverage,
                idx_np=idx,
                frame_offset=f_off,
                kind="sparse",
                fps=fps_safe,
                dev=dev,
                all_means=all_means,
                all_colors=all_colors,
                all_times=all_times,
                all_log_scales=all_log_scales,
                all_nu=all_nu,
                all_kinds=all_kinds,
                all_flag_is_static=all_is_static,
                all_flag_dynamic_protect=all_dynamic_protect,
                all_flag_dynamic_allowed=all_dynamic_allowed,
                all_flag_stage_keep=all_stage_keep,
                all_source_frames=all_source_frames,
                scale_knn_k=scale_knn_k,
                scale_knn_factor=scale_knn_factor,
                all_durations=all_durations,
            )
            selected_sparse += 1
        else:
            missing_frames.append(src_f)

    if not all_means:
        raise RuntimeError(
            "No dynamic rows selected from anchor/coverage NPZ pair. "
            "Check source frame range and keyframe stride."
        )

    if verbose:
        log.info(
            "[from_colmap_selfcap_hybrid] dynamic keyframes: %d dense(anchor), %d sparse(coverage), %d missing; stride=%d",
            selected_dense,
            selected_sparse,
            len(missing_frames),
            stride,
        )
        if missing_frames[:5]:
            log.warning("[from_colmap_selfcap_hybrid] first missing dynamic keyframes: %s", missing_frames[:5])

    # Radius-ball anchors and local twist, after dense/sparse resolution just like SelfCap.
    all_anchors: List[Tensor] = []
    for means_i in all_means:
        if not omega_init or int(means_i.shape[0]) <= 1:
            all_anchors.append(means_i.clone())
            continue
        ext = (means_i.max(dim=0).values - means_i.min(dim=0).values).max().clamp(min=1e-6)
        radius = float(anchor_radius_frac) * float(ext.item())
        all_anchors.append(_radius_ball_centroid(means_i, radius))

    all_omega: List[Tensor] = []
    if omega_init:
        for i in range(len(all_means)):
            means_i = all_means[i]
            if int(means_i.shape[0]) <= 1:
                all_omega.append(torch.zeros_like(means_i))
                continue
            nn_idx = local_knn_indices(means_i, local_twist_k)
            nu_i, om_i, all_anchors[i] = fit_local_twist(
                means_i,
                all_nu[i],
                all_anchors[i],
                nn_idx,
                reg=local_twist_reg,
            )
            all_nu[i] = nu_i
            all_omega.append(om_i)
    else:
        all_omega = [torch.zeros_like(m) for m in all_means]

    if target_count is not None and int(target_count) > 0:
        _apply_selfcap_fps(
            dev=dev,
            target_count=int(target_count),
            fps_chunk_size=fps_chunk_size,
            lists=(
                all_means,
                all_colors,
                all_times,
                all_log_scales,
                all_nu,
                all_anchors,
                all_omega,
                all_source_frames,
                all_is_static,
                all_dynamic_protect,
                all_dynamic_allowed,
                all_stage_keep,
                all_durations,
            ),
        )

    # Dynamic concat.
    means_dyn = torch.cat(all_means, dim=0)
    colors_dyn = torch.cat(all_colors, dim=0)
    scales_dyn = torch.cat(all_log_scales, dim=0)
    times_dyn = torch.cat(all_times, dim=0)
    nu_dyn = torch.cat(all_nu, dim=0)
    anchors_dyn = torch.cat(all_anchors, dim=0)
    omega_dyn = torch.cat(all_omega, dim=0)
    is_static_dyn = torch.cat(all_is_static, dim=0)
    dynamic_protect_dyn = torch.cat(all_dynamic_protect, dim=0)
    dynamic_allowed_dyn = torch.cat(all_dynamic_allowed, dim=0)
    stage_keep_dyn = torch.cat(all_stage_keep, dim=0)
    source_frames_dyn = torch.cat(all_source_frames, dim=0)

    # Temporal support init.
    # Keep old behavior as "class"; add NPZ-aware options for hierarchy/active culling.
    base_sparse_s = (
        float(init_sigma_t)
        if float(init_sigma_t) > 0
        else max(float(keyframe_stride) / max(float(fps_safe), 1e-6), 1e-6)
    )

    # This preserves the SelfCap intuition that dense/anchor rows are longer-lived priors.
    base_dense_s = float(dense_duration_multiplier) * base_sparse_s

    dur_npz = torch.cat(all_durations, dim=0).reshape(-1, 1)
    kind_dyn = torch.cat(
        [
            torch.full(
                (all_means[i].shape[0], 1),
                1.0 if all_kinds[i] == "dense" else 0.0,
                device=dev,
            )
            for i in range(len(all_means))
        ],
        dim=0,
    )

    class_sigma = torch.where(
        kind_dyn > 0.5,
        torch.full_like(kind_dyn, base_dense_s),
        torch.full_like(kind_dyn, base_sparse_s),
    )

    mode = str(duration_init_mode)

    if mode == "class":
        sigma_dyn = class_sigma

    elif mode == "npz":
        sigma_dyn = dur_npz

    elif mode == "npz_clamped":
        sigma_dyn = dur_npz
        floor = float(duration_min_s) if float(duration_min_s) > 0 else base_sparse_s
        ceil = float(duration_max_s) if float(duration_max_s) > 0 else max(float(total_sec), floor)
        sigma_dyn = sigma_dyn.clamp(min=max(floor, 1e-6), max=max(ceil, floor))

    elif mode == "hybrid_npz":
        # Recommended initial mode.
        # Uses RoMa speed-adaptive durations, but prevents dense anchor rows from becoming
        # too temporally narrow and disappearing between keyframes.
        sigma_dyn = dur_npz
        sparse_floor = float(sparse_duration_multiplier) * base_sparse_s
        dense_floor = float(dense_duration_multiplier) * base_sparse_s
        floor = torch.where(
            kind_dyn > 0.5,
            torch.full_like(kind_dyn, dense_floor),
            torch.full_like(kind_dyn, sparse_floor),
        )
        sigma_dyn = torch.maximum(sigma_dyn, floor)

        if float(duration_max_s) > 0:
            sigma_dyn = sigma_dyn.clamp(max=float(duration_max_s))
        else:
            sigma_dyn = sigma_dyn.clamp(max=max(float(total_sec), float(base_sparse_s)))

    else:
        raise ValueError(
            f"Unknown duration_init_mode={duration_init_mode!r}; "
            "expected class, npz, npz_clamped, or hybrid_npz"
        )

    sigma_dyn = torch.nan_to_num(sigma_dyn, nan=base_sparse_s, posinf=base_sparse_s, neginf=base_sparse_s)
    sigma_dyn = sigma_dyn.clamp(min=1e-6)
    log_durations_dyn = torch.log(sigma_dyn)
    kind_dyn = torch.cat(
        [
            torch.full((all_means[i].shape[0], 1), 1.0 if all_kinds[i] == "dense" else 0.0, device=dev)
            for i in range(len(all_means))
        ],
        dim=0,
    )

    # Append COLMAP static background once. This is not in SelfCap, but preserves your Mode-B static path.
    static_sources: List[_LoadedNPZ] = []
    if static_from in {"anchor", "both"}:
        static_sources.append(anchor)
    if static_from in {"coverage", "both"}:
        static_sources.append(coverage)

    static_chunks = []
    for src in static_sources:
        idx = src.static_indices()
        if len(idx) == 0:
            continue
        static_chunks.append((src, idx))

    if static_chunks:
        static_pos = torch.cat([_as_tensor(src.positions[idx], dev) for src, idx in static_chunks], dim=0)
        static_colors = torch.cat([_as_tensor(src.colors[idx], dev).clamp(0.0, 1.0) for src, idx in static_chunks], dim=0)
        n_static = int(static_pos.shape[0])
        static_times = torch.full((n_static, 1), 0.5 * float(total_sec), device=dev)
        static_nu = torch.zeros_like(static_pos)
        static_omega = torch.zeros_like(static_pos)
        static_anchors = static_pos.clone()
        if int(scales_dyn.numel()) > 0:
            fallback_scale = scales_dyn.median(dim=0).values.reshape(1, 3)
        else:
            fallback_scale = torch.full((1, 3), math.log(0.01), device=dev)
        static_scales = fallback_scale.expand(n_static, 3).clone()
        static_log_durations = torch.full((n_static, 1), math.log(max(float(total_sec), 1e-6)), device=dev)
        static_is_static = torch.ones((n_static, 1), device=dev)
        static_dynamic_protect = torch.zeros((n_static, 1), device=dev)
        static_dynamic_allowed = torch.zeros((n_static, 1), device=dev)
        static_stage_keep = torch.ones((n_static, 1), device=dev)
        static_kind = torch.full((n_static, 1), -1.0, device=dev)
        static_source_frames = torch.cat([
            _as_tensor(src.source_frames[idx].astype(np.int64), dev, dtype=torch.long).reshape(-1, 1)
            for src, idx in static_chunks
        ], dim=0)

        means = torch.cat([means_dyn, static_pos], dim=0)
        colors = torch.cat([colors_dyn, static_colors], dim=0)
        scales = torch.cat([scales_dyn, static_scales], dim=0)
        times = torch.cat([times_dyn, static_times], dim=0)
        nu = torch.cat([nu_dyn, static_nu], dim=0)
        omega = torch.cat([omega_dyn, static_omega], dim=0)
        anchors_out = torch.cat([anchors_dyn, static_anchors], dim=0)
        log_durations = torch.cat([log_durations_dyn, static_log_durations], dim=0)
        is_static = torch.cat([is_static_dyn, static_is_static], dim=0)
        dynamic_protect = torch.cat([dynamic_protect_dyn, static_dynamic_protect], dim=0)
        dynamic_allowed = torch.cat([dynamic_allowed_dyn, static_dynamic_allowed], dim=0)
        stage_keep = torch.cat([stage_keep_dyn, static_stage_keep], dim=0)
        kind = torch.cat([kind_dyn, static_kind], dim=0)
        source_frames = torch.cat([source_frames_dyn, static_source_frames], dim=0)
    else:
        means = means_dyn
        colors = colors_dyn
        scales = scales_dyn
        times = times_dyn
        nu = nu_dyn
        omega = omega_dyn
        anchors_out = anchors_dyn
        log_durations = log_durations_dyn
        is_static = is_static_dyn
        dynamic_protect = dynamic_protect_dyn
        dynamic_allowed = dynamic_allowed_dyn
        stage_keep = stage_keep_dyn
        kind = kind_dyn
        source_frames = source_frames_dyn

    n_rows = int(means.shape[0])
    quats = torch.zeros(n_rows, 4, device=dev)
    quats[:, 0] = 1.0
    op = max(min(float(init_opacity), 1.0 - 1e-6), 1e-6)
    opacities = torch.full((n_rows,), math.log(op / (1.0 - op)), device=dev)
    sh0 = (colors - 0.5) / _SH_C0
    n_shn_bands = (int(max_sh_degree) + 1) ** 2 - 1
    shN = torch.zeros(n_rows, n_shn_bands, 3, device=dev)

    def _zeros3() -> Tensor:
        return torch.zeros(n_rows, 3, device=dev)

    params = nn.ParameterDict({
        "means": nn.Parameter(means, requires_grad=True),
        "quats": nn.Parameter(quats, requires_grad=True),
        "scales": nn.Parameter(scales, requires_grad=True),
        "sh0": nn.Parameter(sh0, requires_grad=True),
        "shN": nn.Parameter(shN, requires_grad=True),
        "opacities": nn.Parameter(opacities, requires_grad=True),
        "times": nn.Parameter(times, requires_grad=True),
        "log_durations": nn.Parameter(log_durations, requires_grad=True),
        "anchors": nn.Parameter(anchors_out.detach().clone(), requires_grad=True),
        "zeta_base_t": nn.Parameter(nu.detach().clone(), requires_grad=True),
        "zeta_base_r": nn.Parameter(omega.detach().clone(), requires_grad=True),
        "zeta_c0_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c0_r": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c1_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c1_r": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c2_t": nn.Parameter(_zeros3(), requires_grad=True),
        "zeta_c2_r": nn.Parameter(_zeros3(), requires_grad=True),
        "is_static": nn.Parameter(is_static, requires_grad=False),
        "dynamic_protect": nn.Parameter(dynamic_protect, requires_grad=False),
        "dynamic_allowed": nn.Parameter(dynamic_allowed, requires_grad=False),
        "stage_keep": nn.Parameter(stage_keep, requires_grad=False),
        # Debug-only metadata: 1=dense/anchor, 0=sparse/coverage, -1=static.
        "hybrid_kind": nn.Parameter(kind, requires_grad=False),
        "source_frames": nn.Parameter(source_frames.float(), requires_grad=False),
    })

    scene_extent = float((means.max(dim=0).values - means.min(dim=0).values).max().clamp(min=1e-6))

    if verbose:
        dense_rows = int((kind.reshape(-1) == 1).sum().item())
        sparse_rows = int((kind.reshape(-1) == 0).sum().item())
        static_rows = int((is_static.reshape(-1) > 0.5).sum().item())

        sig = log_durations.detach().exp().reshape(-1)
        kind_flat = kind.reshape(-1)

        def _med_for(mask: torch.Tensor) -> float:
            if bool(mask.any()):
                return float(sig[mask].median().item())
            return float("nan")

        log.info(
            "[from_colmap_selfcap_hybrid] "
            "N=%s dense=%s sparse=%s static=%s "
            "sigma_dense_med=%.4fs sigma_sparse_med=%.4fs sigma_static_med=%.4fs "
            "scene_extent=%.3f",
            f"{n_rows:,}",
            f"{dense_rows:,}",
            f"{sparse_rows:,}",
            f"{static_rows:,}",
            _med_for(kind_flat == 1),
            _med_for(kind_flat == 0),
            _med_for(kind_flat < 0),
            scene_extent,
        )

    return params, int(max_sh_degree), scene_extent