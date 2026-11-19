"""CLI entry point for the TRiGS trainer (multi-view COLMAP + ROMA mode).

Takes:
  - a COLMAP sparse reconstruction (``<data_dir>/sparse/0/`` + ``<data_dir>/images/cam##/``),
  - a directory of per-frame ROMA-triangulated point NPY files, and
  - a frame range + keyframe stride.

Pipeline:
  1. Run :func:`trigs.init.roma_combine.main` to produce a combined NPZ
     (if the NPZ doesn't already exist).
  2. Build canonical TriGS params from the NPZ
     (:func:`trigs.init.from_colmap.build_canonical_from_npz`).
  3. Parse COLMAP cameras + images via :class:`trigs.io.FreeTimeParser`.
  4. Collect GT images (per-frame × per-camera) into the format the trainer
     expects.
  5. Run :meth:`TrigsTrainer.train_from_params` and checkpoint the result.

Usage:
    uv run trigs-train-colmap \
        --data-dir path/to/colmap_data \
        --roma-input-dir path/to/per_frame_npys \
        --output-dir path/to/output \
        --start-frame 0 --end-frame 60 --keyframe-step 5
"""

from __future__ import annotations

import logging
import math
import subprocess
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch
from trigs.compat.device import resolve_device  # macOS port
import tyro

from trigs.core.trainer import (
    TrigsTrainer,
    TrigsTrainingConfig,
    training_camera_extent,
)
from trigs.core.viewer_snapshot import apply_default_snapshot_dir
from trigs.init.from_colmap import (
    _knn_distance,
    build_canonical_from_npz,
    release_static_rows_to_dynamic,
)
from trigs.init.from_colmap_hybrid import build_canonical_from_npz_pair_as_selfcap
from trigs.io.dataset import FreeTimeParser, FreeTimeDataset

import cv2
import numpy as np
from PIL import Image

import json

log = logging.getLogger(__name__)

def _apply_anchor_init_from_npz_kind(
    params,
    npz_path: Path,
    sparse_sigma_t: float,
    anchor_sigma_multiplier: float = 5.0,
    zero_anchor_velocity: bool = True,
    min_sigma_t: float = 0.005,
    nn_max_distance: float = 1e-4,
    device: str = "auto",
) -> None:
    """SelfCap-style per-class init for the COLMAP-NPZ canonical set.

    Reads the `kind` per-point field from the merged NPZ and, for primitives
    flagged as anchor (kind=1), overrides:

      * `log_durations` to log(anchor_sigma_multiplier * sparse_sigma_t).
        Mirrors `from_selfcap.py`'s 5x dense_sigma_t_init factor.
      * `zeta_base_t` to zero. Mirrors SelfCap's "dense keyframes are
        treated as static priors at init" — anchors don't carry ROMA
        velocity; sparse primitives carry the motion.

    Coverage rows (kind=0) keep log_durations = log(sparse_sigma_t) (matching
    `--config.init_sigma_t` if it's >0, else 1.0s to match SelfCap default)
    and keep their NPZ-provided velocities untouched.

    Mapping canonical primitives back to source NPZ rows uses a KD-tree
    nearest-neighbor lookup on positions. `build_canonical_from_npz`
    preserves positions exactly through any internal subsampling, so the
    match is effectively 1:1; we still warn if any canonical primitive lands
    farther than `nn_max_distance` from its nearest source row, which would
    indicate position-mutating logic inside the builder that we'd want to
    know about.

    No-op if the NPZ has no `kind` field (e.g. single-pass prep), so it's
    safe to call unconditionally.
    """
    device = resolve_device(device)  # macOS port: "auto" -> cuda/mps/cpu
    import numpy as np
    from scipy.spatial import cKDTree

    with np.load(npz_path, allow_pickle=False) as f:
        if "kind" not in f.files:
            log.debug(
                "NPZ has no 'kind' field; skipping per-class anchor init "
                "(this is normal for single-pass preps)"
            )
            return
        src_positions = f["positions"].astype(np.float32)
        src_kind = f["kind"].astype(np.int8)

    canonical_means = params["means"].detach().cpu().numpy().astype(np.float32)
    n_canon = canonical_means.shape[0]

    # Nearest-neighbor from canonical means -> source NPZ rows.
    tree = cKDTree(src_positions)
    dists, idx = tree.query(canonical_means, k=1)
    canon_kind = src_kind[idx]  # (n_canon,)

    max_d = float(dists.max()) if n_canon > 0 else 0.0
    n_far = int((dists > nn_max_distance).sum())
    if n_far > 0:
        log.warning(
            f"Anchor kind lookup: {n_far}/{n_canon} canonical primitives "
            f"matched > {nn_max_distance:g} to nearest source NPZ row "
            f"(max d = {max_d:.6f}). Builder may be transforming positions; "
            f"kind tags for those rows are best-effort."
        )
    else:
        log.info(
            f"Anchor kind lookup: all {n_canon:,} canonical primitives "
            f"matched within {nn_max_distance:g} to source (max d = {max_d:.2e})"
        )

    n_anchor = int((canon_kind == 1).sum())
    n_coverage = int((canon_kind == 0).sum())
    log.info(
        f"Per-class init: {n_anchor:,} anchor primitives (kind=1), "
        f"{n_coverage:,} coverage primitives (kind=0)"
    )
    if n_anchor == 0:
        log.warning(
            "No anchor primitives found after kind lookup. Either the merger "
            "wasn't run, or all anchor rows were subsampled away. Skipping."
        )
        return

    # Match SelfCap's sparse_sigma_t_init default (1.0s) when config gives 0.
    sparse_sigma_t = max(sparse_sigma_t, 1e-6) if sparse_sigma_t > 0 else 1.0
    dense_sigma_t = anchor_sigma_multiplier * sparse_sigma_t

    sigma_t_per_point = np.where(
        canon_kind == 1, dense_sigma_t, sparse_sigma_t,
    ).astype(np.float32)
    sigma_t_per_point = np.maximum(sigma_t_per_point, min_sigma_t)
    log_dur_np = np.log(sigma_t_per_point)

    # Existing log_durations shape may be (N,) or (N, 1) depending on builder
    # version; preserve the existing shape so downstream code doesn't break.
    existing = params["log_durations"]
    if existing.dim() == 2 and existing.shape[-1] == 1:
        log_dur_t = torch.from_numpy(log_dur_np).to(device).unsqueeze(-1)
    else:
        log_dur_t = torch.from_numpy(log_dur_np).to(device)

    with torch.no_grad():
        params["log_durations"].data.copy_(log_dur_t)
        if zero_anchor_velocity:
            anchor_mask = torch.from_numpy(canon_kind == 1).to(device)
            params["zeta_base_t"].data[anchor_mask] = 0.0
            params["zeta_base_r"].data[anchor_mask] = 0.0

    log.info(
        f"Applied per-class init: sparse sigma_t = {sparse_sigma_t:.3f}s, "
        f"anchor sigma_t = {dense_sigma_t:.3f}s "
        f"(multiplier x{anchor_sigma_multiplier:g}); "
        f"{'zeroed' if zero_anchor_velocity else 'kept'} anchor velocities"
    )

@dataclass
class TrigsColmapArgs:
    """Arguments for the TRiGS COLMAP + ROMA trainer."""

    data_dir: Path
    """Root of the COLMAP dataset (contains ``sparse/`` and ``images/``)."""

    output_dir: Path
    """Directory to write NPZ, checkpoint, and debug renders."""

    roma_input_dir: Path | None = None
    """Directory containing per-frame ``points3d_frameXXXXXX.npy`` and
    ``colors_frameXXXXXX.npy`` files. Only needed when no ``--npz-path`` is given
    and ``output_dir/keyframes.npz`` does not exist yet."""

    start_frame: int | None = None
    """First frame index (inclusive). None = the NPZ's first source frame."""

    end_frame: int | None = None
    """Last frame index (inclusive). None = the NPZ's last source frame.

    This matches ``trigs.init.roma_combine`` semantics; the COLMAP image loader
    uses an exclusive end internally, so we pass ``end_frame + 1`` there.
    """

    keyframe_step: int | None = None
    """Keyframe stride. None = the NPZ's ``keyframe_step`` (5 if it has none). Also sets
    ``config.keyframe_stride``. A value different from the NPZ's stops the run: it would
    silently drop keyframes."""

    rs_rotation_json: Path | None = None
    """JSON mapping each camera to its extract rotation ("l"/"r"), e.g.
    ``{"Camera 01": "l", ...}`` -- the same file ``prep-roma-trigs-memory.py
    --rotation-json`` and ``prep/sync/audit-rolling-shutter.py`` take.

    Only consulted when ``--config.rs-readout-ms > 0``. Without it, rolling
    shutter correction stays off even if a readout is configured, because the
    per-camera readout DIRECTION is what makes the correction meaningful."""

    rs_flip_sign: bool = False
    """Invert the l/r -> readout-direction convention."""

    timing_json: Path | None = None
    """Candidate capture-timing file (convention_id 'midpoint_v1', see trigs.capture_timing): per-camera MIDPOINT
    constants (replacing <data>/sync.json, which is then not read), per-camera rolling-shutter flags, the readout
    and its axis. Row axis: every camera with rs=true gets sign +1, rs=false gets 0, raw-sensor coordinates are
    used, and --rs-rotation-json is not needed. If --config.rs-readout-ms is 0 it is taken from the file;
    otherwise it must match. None = original behaviour."""

    npz_path: Path | None = None
    """Pre-built single/merged keyframes NPZ. If ``None`` and no hybrid pair is
    provided, one is generated at ``output_dir/keyframes.npz`` via
    :mod:`trigs.init.roma_combine`."""

    hybrid_anchor_npz: Path | None = None
    """Optional dense/anchor RoMa-prep NPZ. When provided with
    ``hybrid_coverage_npz``, the COLMAP initializer treats these two NPZs like
    SelfCap's dense_pcds/ and pcds/ rather than using a merged NPZ."""

    hybrid_coverage_npz: Path | None = None
    """Optional sparse/coverage RoMa-prep NPZ used as SelfCap-style fallback
    for frames that do not have anchor rows."""

    hybrid_static_from: str = "anchor"
    """Which NPZ supplies global COLMAP static/background rows in hybrid-pair mode.
    One of: anchor, coverage, both, none. Default matches your merger's
    --keep-static-from anchor behavior."""

    data_factor: int = 1
    """COLMAP image downsampling factor (passed to :class:`FreeTimeParser`)."""

    reference_colmap: Path | None = None
    """Optional COLMAP model used for camera poses and intrinsics.

    When omitted, the loader keeps its normal ``sparse/0`` auto-detection.
    This is required when ROMA was generated from a refined calibration so
    training does not silently fall back to the older camera model.
    """

    test_every: int = 8
    """COLMAP test split stride (passed to :class:`FreeTimeParser`)."""

    config: TrigsTrainingConfig = field(default_factory=TrigsTrainingConfig)
    """TRiGS training configuration."""

    device: str = "auto"
    """Target device."""

    masks_root: Path | None = None
    """Optional foreground mask root. Expected layout mirrors images:
    masks/<camera_folder>/<frame>.png."""

    mask_pattern: str = "{frame:06d}.png"
    """Filename pattern for masks inside each camera folder."""

    mask_frame_offset: int = 0
    """Additional offset from the source image frame number to the mask frame.

    Masks normally use the same filenames as images, so the default is zero.
    For example, ``start_frame=1`` maps dataset offsets ``0..119`` to masks
    ``000001.png..000120.png``. Use ``1`` only when masks are deliberately
    numbered one frame later than the source images.
    """

    mask_cache_size: int = 300
    """LRU cache size for lazy mask tensors."""

    error_on_missing_mask: bool = True
    """If True, raise when a mask is missing. If False, use mask_missing_value."""

    mask_missing_value: float = 0.0
    """Fallback mask value when error_on_missing_mask=False. 0=background, 1=foreground."""

    foregrounds_root: Path | None = None
    """Optional decontaminated straight-foreground RGB root. Expected layout:
    foregrounds/<camera_folder>/<frame>.png. When set, masked composite training
    uses these images instead of the captured RGB; --masks-root,
    --config.masked-rgb-loss, and --config.masked-rgb-composite are required."""

    foreground_pattern: str = "{frame:06d}.png"
    """Filename pattern for foreground RGB inside each camera folder."""

    foreground_cache_size: int = 300
    """LRU cache size for lazy foreground RGB tensors."""

    static_as_dynamic: bool = False
    """Train the NPZ's static rows as dynamic rows (same whole-clip, motionless
    start). Lets relocation add and recycle background capacity at the
    background's depth instead of borrowing dynamic rows near the subject."""


def _run_roma_combine(
    input_dir: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    keyframe_step: int,
) -> None:
    """Invoke :mod:`trigs.init.roma_combine` as a subprocess."""
    from trigs.init import roma_combine

    cmd = [
        sys.executable, "-u", roma_combine.__file__,
        "--input-dir", str(input_dir),
        "--output-path", str(output_path),
        "--frame-start", str(start_frame),
        "--frame-end", str(end_frame),
        "--keyframe-step", str(keyframe_step),
    ]
    log.info(f"Running roma_combine: {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------------
# Lazy GT loader — replaces the original _load_gt_from_dataset which loaded
# all frames into RAM upfront (causing OOM on long sequences).
#
# The trainer accesses GT as:  gt_images[frame_idx][cam_idx]
# where frame_idx = 0..n_frames-1  (relative offset, same as epoch-pair loop)
# and   cam_idx   = 0..n_cameras-1 (position in the training-split list).
#
# _LazyGTImages satisfies that interface but loads each image from disk only
# when it is first requested, keeping an LRU cache of recently-used images.
# ---------------------------------------------------------------------------

class _FrameProxy:
    def __init__(self, parent, frame_offset: int):
        self._parent = parent
        self._frame_offset = int(frame_offset)

    def __getitem__(self, cam_pos: int) -> torch.Tensor:
        return self._parent._get(self._frame_offset, int(cam_pos))

class _DatasetGTImages:
    """
    Lazy gt_images[frame_offset][cam_pos] backed by FreeTimeDataset.

    This guarantees GT pixels match the dataset sample used to derive K/HW.
    The trainer currently does `.float().div(255.0)`, so this returns uint8.
    """
    def __init__(self, dataset, n_cameras: int, orig_to_pos: dict[int, int], cache_size: int = 300):
        self._dataset = dataset
        self._n_cameras = int(n_cameras)
        self._cache = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._item_idx: dict[tuple[int, int], int] = {}

        # dataset.pairs is (original_parser_cam_idx, frame_offset)
        for i, (orig_cam_idx, frame_offset) in enumerate(dataset.pairs):
            cp = orig_to_pos.get(int(orig_cam_idx))
            if cp is not None:
                self._item_idx[(int(frame_offset), int(cp))] = int(i)

        expected = len(dataset)
        found = len(self._item_idx)
        if found == 0:
            raise RuntimeError("DatasetGTImages found no dataset pairs.")

    def __getitem__(self, frame_offset: int):
        return _FrameProxy(self, int(frame_offset))

    @staticmethod
    def _to_uint8_hwc(img: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)

        img = img.detach().cpu()

        # HWC expected. If your dataset ever returns CHW, convert defensively.
        if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
            img = img.permute(1, 2, 0).contiguous()

        if img.dtype == torch.uint8:
            return img.contiguous()

        x = img.float()
        if x.numel() and float(x.max()) <= 2.0:
            x = x * 255.0
        return x.round().clamp(0, 255).to(torch.uint8).contiguous()

    def _get(self, frame_offset: int, cam_pos: int) -> torch.Tensor:
        key = (int(frame_offset), int(cam_pos))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        item_idx = self._item_idx.get(key)
        if item_idx is None:
            raise RuntimeError(
                f"No dataset item for frame_offset={frame_offset}, cam_pos={cam_pos}"
            )

        sample = self._dataset[item_idx]
        if sample is None:
            try:
                pair = self._dataset.pairs[item_idx]
            except Exception:
                pair = None
            raise RuntimeError(
                f"FreeTimeDataset returned None for frame_offset={frame_offset}, "
                f"cam_pos={cam_pos}, item_idx={item_idx}, pair={pair}. "
                "This usually means the dataset could not resolve/load the image path "
                "for this sample. Check FreeTimeDataset._get_image_path / frame indexing."
            )

        img = self._to_uint8_hwc(sample["image"])

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = img
        return img

RawWarp = tuple[torch.Tensor, tuple[int, int, int, int]]


def _raw_warps_by_pos(parser, training_cam_ids: Sequence[int]) -> list[RawWarp | None]:
    """The dataset's undistortion (sampling grid + ROI crop) per training camera.

    Masks and foreground RGB are generated from the raw frames, so they need
    the same warp the dataset applies to captured RGB before they line up with
    it and with the rendered camera. Resizing alone leaves them misregistered
    by the lens distortion and ROI zoom (landscape-test-02: 3 px median at the
    image centre, 12.8 px on the worst camera). None means the dataset loads
    that camera without a warp.

    Reuses the parser's own sampling grid (already on the GPU on a training
    box) so the warp is the dataset's and costs ~nothing per load; the CPU
    fallback builds the identical grid from the maps.
    """
    grids = getattr(parser, "grid_dict", None) or {}
    warps: list[RawWarp | None] = []
    for orig in training_cam_ids:
        camera_id = parser.camera_ids[int(orig)]
        if camera_id in parser.mapx_dict and len(parser.params_dict.get(camera_id, [])) > 0:
            grid = grids.get(camera_id)
            if grid is None:
                mapx = parser.mapx_dict[camera_id]
                mapy = parser.mapy_dict[camera_id]
                H, W = mapx.shape
                # Same normalization as FreeTimeParser.grid_dict.
                grid = torch.stack((
                    2.0 * torch.from_numpy(mapx).float() / max(W - 1, 1) - 1.0,
                    2.0 * torch.from_numpy(mapy).float() / max(H - 1, 1) - 1.0,
                ), dim=-1)
            x, y, w, h = (int(v) for v in parser.roi_undist_dict[camera_id])
            warps.append((grid, (x, y, w, h)))
        else:
            warps.append(None)
    return warps


def _warp_raw_like_dataset(arr: np.ndarray, warp: RawWarp | None) -> np.ndarray:
    """Apply a camera's undistortion + ROI crop to a raw-frame uint8 array.

    Mirrors FreeTimeDataset's RGB path (bilinear grid_sample, align_corners,
    zero padding, truncating .byte(), ROI crop), after resizing to the
    resolution the maps were built for. Accepts [H, W] or [H, W, C].
    """
    if warp is None:
        return arr
    grid, (x, y, w, h) = warp
    if arr.shape[:2] != tuple(grid.shape[:2]):
        arr = cv2.resize(arr, (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_LINEAR)
    # np.array copies: PIL-backed arrays are read-only, which torch warns about.
    img = torch.from_numpy(np.array(arr, dtype=np.float32)).to(grid.device)
    img = img[None] if img.ndim == 2 else img.permute(2, 0, 1)
    out = torch.nn.functional.grid_sample(
        img.unsqueeze(0), grid.unsqueeze(0),
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )[0].byte()
    out = out[0] if arr.ndim == 2 else out.permute(1, 2, 0)
    return np.ascontiguousarray(out[y:y + h, x:x + w].cpu().numpy())


class _LazyGTImages:
    def __init__(
        self,
        images_root: Path,
        cam_folders_by_pos: list[str],
        *,
        source_start_frame: int,
        target_hw_by_pos: list[tuple[int, int]],
        image_pattern: str = "{frame:06d}.jpg",
        cache_size: int = 300,
        raw_warps_by_pos: Sequence[RawWarp | None] | None = None,
    ) -> None:
        self._images_root = Path(images_root)
        self._cam_folders_by_pos = list(cam_folders_by_pos)
        self._source_start_frame = int(source_start_frame)
        self._target_hw_by_pos = [(int(h), int(w)) for h, w in target_hw_by_pos]
        self._image_pattern = str(image_pattern)
        self._cache = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._raw_warps_by_pos = (
            list(raw_warps_by_pos) if raw_warps_by_pos is not None
            else [None] * len(self._cam_folders_by_pos)
        )

    def __getitem__(self, frame_offset: int):
        return _FrameProxy(self, int(frame_offset))

    def _path(self, frame_offset: int, cam_pos: int) -> Path:
        folder = self._cam_folders_by_pos[int(cam_pos)]
        source_frame = self._source_start_frame + int(frame_offset)
        return self._images_root / folder / self._image_pattern.format(frame=source_frame)

    def validate_boundaries(self, n_frames: int) -> None:
        """Fail before GPU training if first/last foreground images are absent."""
        if int(n_frames) <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")
        for frame_offset in sorted({0, int(n_frames) - 1}):
            for cam_pos in range(len(self._cam_folders_by_pos)):
                path = self._path(frame_offset, cam_pos)
                if not path.exists():
                    raise FileNotFoundError(
                        "Foreground boundary validation failed before training: "
                        f"frame_offset={frame_offset}, cam_pos={cam_pos}: {path}"
                    )

    @staticmethod
    def _fit_to_hw(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        h, w = int(arr.shape[0]), int(arr.shape[1])

        if h == target_h and w == target_w:
            return arr.copy()

        # Match _LazyGTMasks exactly so foreground RGB and alpha remain
        # registered even for the common 1920x1080 -> 1919x1079 adjustment.
        img = Image.fromarray(arr)
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.uint8).copy()

    def _get(self, frame_offset: int, cam_pos: int) -> torch.Tensor:
        key = (int(frame_offset), int(cam_pos))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        p = self._path(frame_offset, cam_pos)
        if not p.exists():
            raise RuntimeError(
                f"GT image file missing for frame_offset={frame_offset}, "
                f"cam_pos={cam_pos}: {p}"
            )

        arr = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
        arr = _warp_raw_like_dataset(arr, self._raw_warps_by_pos[int(cam_pos)])
        target_h, target_w = self._target_hw_by_pos[int(cam_pos)]
        arr = self._fit_to_hw(arr, target_h, target_w)

        if arr.shape[:2] != (target_h, target_w):
            raise RuntimeError(
                f"GT image shape still wrong after fit: path={p}, "
                f"got={arr.shape[:2]}, expected={(target_h, target_w)}"
            )

        tensor = torch.from_numpy(arr).to(torch.uint8).cpu()

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = tensor
        return tensor

class _DummyGTAlphas:
    """Placeholder for gt_alphas_per_frame.

    The original trainer accepts this parameter but never reads it inside
    _train_loop, so a dummy that accepts any subscript is sufficient.
    """
    def __getitem__(self, _):
        return self

def _load_rs_signs(
    rotation_json: Path, folders: List[str], flip: bool = False
) -> List[float]:
    """Resolve per-camera rolling-shutter readout direction (+1 'l' / -1 'r').

    Key matching mirrors prep/sync/audit-rolling-shutter.py so the same file
    drives prep, the audit, and training. Missing cameras are a hard error: a
    silent 0 would leave part of the rig uncorrected while the rest is
    corrected, which is a worse state than not correcting at all.
    """
    raw = json.loads(Path(rotation_json).read_text(encoding="utf-8"))
    table = {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}

    signs: List[float] = []
    missing: List[str] = []
    for folder in folders:
        keys = [str(folder), str(folder).lower()]
        if str(folder).isdigit():
            n = int(folder)
            numeric = [f"{n:04d}", f"{n:02d}", str(n)]
            keys += numeric
            for prefix in ("camera ", "cam ", "camera", "cam"):
                keys += [f"{prefix}{v}" for v in numeric]
        val = next((table[k] for k in keys if k in table), None)
        if val is None:
            missing.append(str(folder))
            signs.append(0.0)
        elif val.startswith("l"):
            signs.append(-1.0 if flip else +1.0)
        elif val.startswith("r"):
            signs.append(+1.0 if flip else -1.0)
        else:
            raise ValueError(f"camera {folder}: expected 'l'/'r', got {val!r}")

    if missing:
        raise KeyError(
            f"--rs-rotation-json is missing {len(missing)} of {len(folders)} "
            f"cameras: {missing[:10]}"
        )
    return signs


def _infer_camera_folders(parser: FreeTimeParser, data_dir: Path) -> List[str]:
    """
    Return camera folders in parser/original camera-index order.

    For this dataset the parser camera order is the sorted camera-folder order:
        0 -> "01"
        1 -> "02"
        ...
        42 -> "43"

    Do not infer this from COLMAP image_names, because COLMAP image iteration
    order can start with arbitrary folders like 37/000001.jpg.
    """
    images_dir = Path(data_dir) / "images"
    folders = sorted(
        [p.name for p in images_dir.iterdir() if p.is_dir()],
        key=lambda s: int(s) if str(s).isdigit() else str(s),
    )

    n_parser = int(getattr(parser, "camtoworlds").shape[0])
    if len(folders) != n_parser:
        raise RuntimeError(
            f"Camera folder count mismatch: images has {len(folders)} folders, "
            f"parser has {n_parser} cameras. First folders={folders[:8]}"
        )

    return folders

def _resolve_camera_key(
    data: Mapping[str, object], *, cam_pos: int, orig_cam_id: int, folder: Optional[str],
) -> Optional[str]:
    """Match a training camera to a key in sync.json / frame_timestamps.json
    / observability.json. All three files share this key space (camera
    folder, e.g. "0000"), so every per-camera sync file uses this one
    resolver instead of three near-duplicate lookups."""
    candidates = []

    if folder is not None:
        folder_s = str(folder)
        candidates.append(folder_s)
        # Directly map to padded folder name, no subtraction
        if folder_s.isdigit():
            candidates.append(f"{int(folder_s):04d}")

    # Parser/training position convention.
    candidates.append(f"{int(cam_pos):04d}")
    candidates.append(str(int(cam_pos)))

    # Original parser/COLMAP camera id convention LAST.
    candidates.append(f"{int(orig_cam_id):04d}")
    candidates.append(str(int(orig_cam_id)))

    seen = set()
    for k in candidates:
        if k in seen:
            continue
        seen.add(k)
        if k in data:
            return k
    return None


def _lookup_sync_offset(sync_data, cam_pos, orig_cam_id, folder):
    key = _resolve_camera_key(sync_data, cam_pos=cam_pos, orig_cam_id=orig_cam_id, folder=folder)
    if key is None:
        return 0.0, None
    return float(sync_data[key]), key


def _rs_sensor_mapping(parser, training_cam_ids, device) -> dict:
    """Raw-sensor mapping per training camera (original K and size at the 1/factor scale, OPENCV distortion),
    for row-axis rolling shutter (trigs.capture_timing.delivered_to_raw)."""
    Ks, dists, hws = [], [], []
    for orig in training_cam_ids:
        cid = parser.camera_ids[int(orig)]
        K = getattr(parser, "Ks_raw_dict", {}).get(cid, parser.Ks_dict[cid])
        w, h = getattr(parser, "imsize_raw_dict", {}).get(cid, parser.imsize_dict[cid])
        prm = np.asarray(parser.params_dict.get(cid, []), dtype=np.float64).reshape(-1)
        if getattr(parser, "camtype", "perspective") != "perspective" and len(prm):
            raise SystemExit("row-axis rolling shutter supports perspective (OPENCV) cameras only")
        d = np.zeros(5); d[:min(len(prm), 5)] = prm[:5]
        Ks.append(torch.as_tensor(np.asarray(K, dtype=np.float64), dtype=torch.float32))
        dists.append(torch.as_tensor(d, dtype=torch.float32)); hws.append(torch.tensor([float(h), float(w)]))
    return dict(rs_sensor_K=torch.stack(Ks).to(device), rs_sensor_dist=torch.stack(dists).to(device),
                rs_sensor_hw=torch.stack(hws).to(device))


def _build_gt_times_from_sync(
    *,
    sync_file: Path,
    parser: FreeTimeParser,
    data_dir: Path,
    training_cam_ids: Sequence[int],
    n_frames: int,
    fps: float,
    sync_data: Dict[str, float] | None = None,
) -> Dict[int, List[float]] | None:
    """Build gt_times[frame_offset][train_camera_position] in seconds. ``sync_data`` (a {key: seconds} dict)
    replaces reading ``sync_file`` when given (timing-candidate files)."""
    if sync_data is None:
        if not sync_file.exists():
            return None
        with sync_file.open('r', encoding='utf-8') as f:
            raw = json.load(f)
        sync_data = {str(k): float(v) for k, v in raw.items()}
    else:
        sync_data = {str(k): float(v) for k, v in sync_data.items()}
    camera_folders = _infer_camera_folders(parser, data_dir)

    offsets_by_pos: list[float] = []
    keys_by_pos: list[str | None] = []
    for cam_pos, orig_cam_id in enumerate(training_cam_ids):
        folder = (
            camera_folders[int(orig_cam_id)]
            if int(orig_cam_id) < len(camera_folders)
            else None
        )
        off, key = _lookup_sync_offset(
            sync_data,
            cam_pos=int(cam_pos),
            orig_cam_id=int(orig_cam_id),
            folder=folder,
        )
        offsets_by_pos.append(float(off))
        keys_by_pos.append(key)

    missing = [
        (int(training_cam_ids[i]), i)
        for i, key in enumerate(keys_by_pos)
        if key is None
    ]
    if missing:
        log.warning(
            "sync.json missing offsets for %d training cameras; using 0 for first few %s",
            len(missing),
            missing[:8],
        )

    preview = [
        {
            "cam_pos": int(i),
            "orig_cam_id": int(training_cam_ids[i]),
            "sync_key": keys_by_pos[i],
            "offset_ms": round(1000.0 * offsets_by_pos[i], 3),
        }
        for i in range(min(8, len(training_cam_ids)))
    ]
    log.debug("sync.json mapping preview: %s", preview)

    gt_times: Dict[int, List[float]] = {}
    fps = max(float(fps), 1e-6)
    for frame_idx in range(int(n_frames)):
        base_t = float(frame_idx) / fps
        gt_times[frame_idx] = [base_t + off for off in offsets_by_pos]

    return gt_times


def _build_gt_times_from_ladder(
    *,
    frame_timestamps_file: Path,
    parser: FreeTimeParser,
    data_dir: Path,
    training_cam_ids: Sequence[int],
    n_frames: int,
) -> Dict[int, List[float]] | None:
    """Build gt_times[frame_offset][train_camera_position] in seconds from
    the per-frame camera ladder written by extract_4dv_sequence.py
    (frame_timestamps.json "cameras"."<folder>"."timestamp_ns").

    Unlike sync.json's single scalar per camera, this varies per exported
    frame: each camera's independent, possibly-extrapolated clock ladder
    (see prep/extract/extract_4dv_sequence.py CameraLadder). Preferred over
    sync.json whenever available; falls back to None so the caller can try
    sync.json next.
    """
    if not frame_timestamps_file.exists():
        return None

    with frame_timestamps_file.open('r', encoding='utf-8') as f:
        raw = json.load(f)

    cameras = raw.get("cameras")
    if not cameras:
        return None
    logical_origin_ns = int(raw["logical_origin_timestamp_ns"])
    camera_folders = _infer_camera_folders(parser, data_dir)

    keys_by_pos: list[str | None] = []
    for cam_pos, orig_cam_id in enumerate(training_cam_ids):
        folder = (
            camera_folders[int(orig_cam_id)]
            if int(orig_cam_id) < len(camera_folders)
            else None
        )
        key = _resolve_camera_key(
            cameras, cam_pos=int(cam_pos), orig_cam_id=int(orig_cam_id), folder=folder,
        )
        keys_by_pos.append(key)

    missing = [
        (int(training_cam_ids[i]), i)
        for i, key in enumerate(keys_by_pos)
        if key is None
    ]
    if missing:
        log.warning(
            "frame_timestamps.json missing ladder entries for %d training cameras; "
            "using 0 offset for first few %s",
            len(missing),
            missing[:8],
        )

    preview = [
        {"cam_pos": int(i), "orig_cam_id": int(training_cam_ids[i]), "ladder_key": keys_by_pos[i]}
        for i in range(min(8, len(training_cam_ids)))
    ]
    log.debug("frame_timestamps.json ladder mapping preview: %s", preview)

    per_cam_ts_ns: list[list[int] | None] = []
    for key in keys_by_pos:
        per_cam_ts_ns.append(cameras[key]["timestamp_ns"] if key is not None else None)

    gt_times: Dict[int, List[float]] = {}
    for frame_idx in range(int(n_frames)):
        row = []
        for ts in per_cam_ts_ns:
            if ts is None or frame_idx >= len(ts):
                row.append(0.0)
            else:
                row.append((int(ts[frame_idx]) - logical_origin_ns) / 1e9)
        gt_times[frame_idx] = row

    return gt_times


def _build_camera_time_offset_scale(
    *,
    observability_file: Path,
    parser: FreeTimeParser,
    data_dir: Path,
    training_cam_ids: Sequence[int],
) -> "torch.Tensor | None":
    """Per-training-position shrinkage weight in [0, 1] from observability.json
    (prep/sync/measure-offset-observability.py), keyed the same way as
    sync.json / frame_timestamps.json. Returns None if the file is absent, in
    which case the trainer treats every camera as fully observable (today's
    behavior)."""
    if not observability_file.exists():
        return None

    with observability_file.open('r', encoding='utf-8') as f:
        raw = json.load(f)
    cameras = raw.get("cameras")
    if not cameras:
        return None
    camera_folders = _infer_camera_folders(parser, data_dir)

    scale_by_pos: list[float] = []
    keys_by_pos: list[str | None] = []
    for cam_pos, orig_cam_id in enumerate(training_cam_ids):
        folder = (
            camera_folders[int(orig_cam_id)]
            if int(orig_cam_id) < len(camera_folders)
            else None
        )
        key = _resolve_camera_key(
            cameras, cam_pos=int(cam_pos), orig_cam_id=int(orig_cam_id), folder=folder,
        )
        keys_by_pos.append(key)
        if key is None:
            scale_by_pos.append(1.0)  # unmeasured -> behave as fully observable
        else:
            scale_by_pos.append(float(cameras[key]["shrinkage"]))

    missing = [i for i, key in enumerate(keys_by_pos) if key is None]
    if missing:
        log.warning(
            "observability.json missing entries for %d training cameras; "
            "treating them as fully observable (scale=1) %s",
            len(missing), missing[:8],
        )

    preview = [
        {"cam_pos": i, "shrinkage": round(scale_by_pos[i], 4)}
        for i in range(min(8, len(training_cam_ids)))
    ]
    log.info("observability.json shrinkage preview: %s", preview)

    return torch.tensor(scale_by_pos, dtype=torch.float32)


class _LazyGTMasks:
    """
    Drop-in for gt_alphas_per_frame.

    Access pattern:
        gt_alphas[frame_offset][cam_pos] -> uint8 tensor [H, W, 1]

    Mask layout:
        masks_root/<camera_folder>/<mask_pattern.format(frame=...)>
    """

    def __init__(
        self,
        dataset: FreeTimeDataset,
        n_cameras: int,
        orig_to_pos: Dict[int, int],
        masks_root: Path,
        cam_folders_by_pos: List[str],
        *,
        source_start_frame: int,
        mask_pattern: str = "{frame:06d}.png",
        mask_frame_offset: int = 0,
        cache_size: int = 300,
        error_on_missing: bool = True,
        missing_value: float = 0.0,
        raw_warps_by_pos: Sequence[RawWarp | None] | None = None,
    ) -> None:
        self._item_idx: Dict[tuple, int] = {}
        self._raw_warps_by_pos = (
            list(raw_warps_by_pos) if raw_warps_by_pos is not None
            else [None] * int(n_cameras)
        )

        # dataset.pairs is List[(cam_idx, frame_offset)] using original parser cam_idx.
        for i, (cam_idx, frame_offset) in enumerate(dataset.pairs):
            cp = orig_to_pos.get(cam_idx)
            if cp is not None:
                self._item_idx[(int(frame_offset), int(cp))] = i

        self._dataset = dataset
        self._n_cameras = int(n_cameras)
        self._masks_root = Path(masks_root)
        self._cam_folders_by_pos = list(cam_folders_by_pos)
        self._source_start_frame = int(source_start_frame)
        self._mask_pattern = str(mask_pattern)
        self._mask_frame_offset = int(mask_frame_offset)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        self._fallback_shapes: Dict[int, tuple[int, int]] = {}
        self._fallback_tensors: Dict[int, torch.Tensor] = {}
        self._error_on_missing = bool(error_on_missing)
        self._missing_value = float(missing_value)

        if len(self._cam_folders_by_pos) != self._n_cameras:
            raise ValueError(
                f"cam_folders_by_pos has {len(self._cam_folders_by_pos)} entries, "
                f"but n_cameras={self._n_cameras}."
            )

    def __getitem__(self, frame_offset: int) -> _FrameProxy:
        return _FrameProxy(self, int(frame_offset))

    def _mask_path(self, frame_offset: int, cam_pos: int) -> Path:
        frame_number = (
            self._source_start_frame
            + int(frame_offset)
            + self._mask_frame_offset
        )
        cam_folder = self._cam_folders_by_pos[int(cam_pos)]
        return self._masks_root / cam_folder / self._mask_pattern.format(frame=frame_number)

    def validate_boundaries(self, n_frames: int) -> None:
        """Validate first/last mask mappings before GPU training starts."""
        if int(n_frames) <= 0:
            raise ValueError(f"n_frames must be positive, got {n_frames}")

        frame_offsets = sorted({0, int(n_frames) - 1})
        missing: List[tuple[int, int, Path]] = []
        for frame_offset in frame_offsets:
            for cam_pos in range(self._n_cameras):
                path = self._mask_path(frame_offset, cam_pos)
                if not path.exists():
                    missing.append((frame_offset, cam_pos, path))

        if missing:
            frame_offset, cam_pos, path = missing[0]
            raise FileNotFoundError(
                "Mask boundary validation failed before training: "
                f"frame_offset={frame_offset}, cam_pos={cam_pos}: {path}. "
                f"Mapping is source_start_frame ({self._source_start_frame}) + "
                f"frame_offset + mask_frame_offset ({self._mask_frame_offset}). "
                "If mask filenames match image filenames, use "
                "--mask-frame-offset 0."
            )

    def _shape_from_dataset(self, frame_offset: int, cam_pos: int) -> tuple[int, int]:
        if cam_pos in self._fallback_shapes:
            return self._fallback_shapes[cam_pos]

        item_idx = self._item_idx.get((int(frame_offset), int(cam_pos)))
        if item_idx is None:
            # Try any sample for this camera to get a shape.
            for (fo, cp), idx in self._item_idx.items():
                if cp == cam_pos:
                    item_idx = idx
                    break

        if item_idx is None:
            raise RuntimeError(
                f"Cannot infer mask shape for cam_pos={cam_pos}; no dataset item found."
            )

        sample = self._dataset[item_idx]
        img = sample["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)

        H, W = int(img.shape[0]), int(img.shape[1])
        self._fallback_shapes[cam_pos] = (H, W)
        return H, W

    def _fallback_mask(self, frame_offset: int, cam_pos: int) -> torch.Tensor:
        if cam_pos in self._fallback_tensors:
            return self._fallback_tensors[cam_pos]

        H, W = self._shape_from_dataset(frame_offset, cam_pos)
        value = int(round(np.clip(self._missing_value, 0.0, 1.0) * 255.0))
        mask = torch.full((H, W, 1), value, dtype=torch.uint8)
        self._fallback_tensors[cam_pos] = mask
        return mask

    def _get(self, frame_offset: int, cam_pos: int) -> torch.Tensor:
        key = (int(frame_offset), int(cam_pos))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        H, W = self._shape_from_dataset(frame_offset, cam_pos)
        path = self._mask_path(frame_offset, cam_pos)

        if not path.exists():
            if self._error_on_missing:
                raise FileNotFoundError(
                    f"Missing mask for frame_offset={frame_offset}, cam_pos={cam_pos}: {path}"
                )
            return self._fallback_mask(frame_offset, cam_pos)

        mask_img = Image.open(path).convert("L")
        warp = self._raw_warps_by_pos[int(cam_pos)]
        if warp is not None:
            mask_img = Image.fromarray(_warp_raw_like_dataset(np.asarray(mask_img), warp))

        # Ensure exact render/GT resolution.
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), Image.Resampling.BILINEAR)

        mask_np = np.asarray(mask_img, dtype=np.uint8).copy()
        mask = torch.from_numpy(mask_np).unsqueeze(-1).contiguous().cpu()

        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = mask
        return mask

@torch.no_grad()
def _apply_split_budget_to_prebuilt_params(
    params,
    *,
    target_count: int,
    static_target_count: int = 75000,
    seed: int = 12345,
    protect_speed_top_frac: float = 0.0,
    fps: float = 60.0,
    config: TrigsTrainingConfig | None = None,
    compute_device: str | torch.device = "auto",
    keyframe_stride: int = 0,
):
    n_old = int(params["means"].shape[0])
    device = params["means"].device

    if "is_static" in params:
        static = params["is_static"].detach().reshape(-1) > 0.5
    else:
        static = torch.zeros(n_old, dtype=torch.bool, device=device)

    dyn = ~static

    static_idx = torch.nonzero(static, as_tuple=False).squeeze(-1)
    dyn_idx = torch.nonzero(dyn, as_tuple=False).squeeze(-1)

    # Speed-aware sub-keyframe rows (NPZ local_stride < keyframe stride) sit at
    # frames between keyframes. They are kept whole and stay outside the budget:
    # target_count, the per-keyframe quotas and the random draw then see exactly
    # the keyframe rows they would see without sub-keyframes.
    sub_keep = dyn_idx[:0]
    if int(keyframe_stride) > 1 and int(dyn_idx.numel()) > 0:
        frame_off = torch.round(params["times"].detach().reshape(-1)[dyn_idx].double() * float(fps)).long()
        off_grid = (frame_off % int(keyframe_stride)) != 0
        if bool(off_grid.any()):
            sub_keep = dyn_idx[off_grid]
            dyn_idx = dyn_idx[~off_grid]

    static_budget = min(int(static_target_count), int(static_idx.numel()))
    dyn_budget = min(int(target_count) - static_budget, int(dyn_idx.numel()))

    if dyn_budget <= 0:
        raise ValueError("Dynamic budget collapsed; lower static_target_count.")

    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Static: random/simple cap for now. Better: voxel merge upstream.
    if int(static_idx.numel()) > static_budget:
        perm = torch.randperm(int(static_idx.numel()), device=device, generator=gen)[:static_budget]
        static_keep = static_idx[perm]
    else:
        static_keep = static_idx

    # Dynamic: temporal-stratified cap.
    times = params["times"].detach().reshape(-1)
    dyn_times = times[dyn_idx]
    unique_t, inverse, counts = torch.unique(
        dyn_times, sorted=True, return_inverse=True, return_counts=True
    )

    desired = counts.float() / counts.sum().float() * float(dyn_budget)
    quotas = torch.floor(desired).long().clamp(min=1)
    quotas = torch.minimum(quotas, counts)

    remainder = int(dyn_budget - int(quotas.sum().item()))
    frac = desired - torch.floor(desired)

    if remainder > 0:
        order = torch.argsort(frac, descending=True)
        for j in order.tolist():
            if remainder <= 0:
                break
            cap = int((counts[j] - quotas[j]).item())
            if cap <= 0:
                continue
            add = min(cap, remainder)
            quotas[j] += add
            remainder -= add
    elif remainder < 0:
        order = torch.argsort(frac, descending=False)
        for j in order.tolist():
            if remainder >= 0:
                break
            removable = max(0, int(quotas[j].item()) - 1)
            take = min(removable, -remainder)
            quotas[j] -= take
            remainder += take

    protect = float(protect_speed_top_frac)
    dyn_speed = None
    if protect > 0.0:
        dyn_speed = _step0_frame_displacement(
            params, dyn_idx, fps=fps, config=config, compute_device=compute_device,
        ).to(device)
    n_protected = 0
    n_random = 0

    dyn_keep_parts = []
    for group_idx, q in enumerate(quotas.tolist()):
        local = torch.nonzero(inverse == group_idx, as_tuple=False).squeeze(-1)
        rows = dyn_idx[local]
        if q >= int(rows.numel()):
            dyn_keep_parts.append(rows)
        elif dyn_speed is None:
            perm = torch.randperm(int(rows.numel()), device=device, generator=gen)[:q]
            dyn_keep_parts.append(rows[perm])
        else:
            m = min(max(int(round(q * protect)), 0), q)
            order = torch.argsort(dyn_speed[local], descending=True, stable=True)
            fast, rest = order[:m], order[m:]
            perm = torch.randperm(int(rest.numel()), device=device, generator=gen)[: q - m]
            dyn_keep_parts.append(torch.cat([rows[fast], rows[rest[perm]]]))
            n_protected += m
            n_random += q - m

    dyn_keep = torch.cat(dyn_keep_parts, dim=0) if dyn_keep_parts else dyn_idx[:0]

    keep_idx = torch.cat([static_keep, dyn_keep, sub_keep], dim=0).sort().values.contiguous()

    for name, p in list(params.items()):
        if torch.is_tensor(p) and p.data.ndim >= 1 and int(p.data.shape[0]) == n_old:
            params[name] = torch.nn.Parameter(
                p.data[keep_idx].contiguous(),
                requires_grad=p.requires_grad,
            )

    means = params["means"].data
    scene_extent = float(
        (means.max(dim=0).values - means.min(dim=0).values).max().clamp(min=1e-6)
    )

    log.debug(
        "[budget split] "
        f"{n_old:,} -> {int(params['means'].shape[0]):,} "
        f"static {int(static_idx.numel()):,}->{int(static_keep.numel()):,} "
        f"dynamic {int(dyn_idx.numel()):,}->{int(dyn_keep.numel()):,}"
    )
    if dyn_speed is not None:
        log.debug(
            "[budget split] speed protection "
            f"top_frac={protect:g}: protected {n_protected:,} fastest dynamic rows, "
            f"random fill {n_random:,} (groups with q >= rows kept whole)"
        )
    if int(sub_keep.numel()) > 0:
        log.info(
            f"Sub-keyframe rows kept whole outside the budget: {int(sub_keep.numel()):,} "
            f"(total {int(params['means'].shape[0]):,})"
        )

    return params, scene_extent


@torch.no_grad()
def _step0_frame_displacement(
    params,
    rows: torch.Tensor,
    *,
    fps: float,
    config: TrigsTrainingConfig | None,
    compute_device: str | torch.device,
    chunk: int = 262144,
) -> torch.Tensor:
    """||deform(mu + 0.5/fps) - deform(mu - 0.5/fps)|| per row: the starting
    curve's world displacement over one video frame centred on mu_t."""
    cfg = config if config is not None else TrigsTrainingConfig()
    dev = resolve_device(compute_device)
    n = int(params["means"].shape[0])
    names = [k for k, v in params.items() if torch.is_tensor(v) and v.ndim >= 1 and int(v.shape[0]) == n]
    out = []
    h = 0.5 / float(fps)
    for s in range(0, int(rows.numel()), chunk):
        r = rows[s:s + chunk].to(params["means"].device)
        sub = {k: params[k].detach()[r].float().to(dev) for k in names}
        mu = sub["times"]
        ma, _, _ = TrigsTrainer._deform(None, sub, mu - h, cfg)
        mb, _, _ = TrigsTrainer._deform(None, sub, mu + h, cfg)
        out.append((mb - ma).norm(dim=-1).reshape(-1).cpu())
    return torch.cat(out) if out else torch.zeros(0)


@torch.no_grad()
def _rescale_dynamic_init_after_budget(params, *, k: int = 4, factor: float = 0.3) -> int:
    """Re-size dynamic rows' starting scales against the rows that survived the
    target-count cap: per keyframe (rows sharing a starting mu_t), log(factor *
    mean k-NN distance) over the kept means, the rule build_canonical_from_npz
    applies to the full prep cloud. Static rows keep their scales. Returns the
    number of rows rescaled."""
    n = int(params["means"].shape[0])
    if "is_static" in params:
        static = params["is_static"].detach().reshape(-1) > 0.5
    else:
        static = torch.zeros(n, dtype=torch.bool, device=params["means"].device)
    dyn_idx = torch.nonzero(~static, as_tuple=False).squeeze(-1)
    if int(dyn_idx.numel()) == 0:
        return 0
    _, inverse = torch.unique(params["times"].detach().reshape(-1)[dyn_idx], return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    scales = params["scales"].data
    for group in order.split(torch.bincount(inverse).tolist()):
        idx = dyn_idx[group]
        d = _knn_distance(params["means"].data[idx], k=k)
        log_scale = torch.log(torch.clamp(d * factor, min=1e-6))
        scales[idx] = log_scale.unsqueeze(-1).expand(-1, scales.shape[1]).to(scales)
    return int(dyn_idx.numel())


@torch.no_grad()
def _apply_target_count_to_prebuilt_params(
    params,
    target_count: int | None,
    *,
    seed: int = 12345,
) -> tuple[object, float]:
    """Fast temporal-stratified reduction for COLMAP/NPZ prebuilt params.

    COLMAP mode enters the trainer through train_from_params(), so the trainer's
    _build_canonical_set() target_count path is bypassed. This applies the
    target count immediately after build_canonical_from_npz().
    """
    if target_count is None or target_count <= 0:
        means = params["means"].data
        scene_extent = float(
            (means.max(dim=0).values - means.min(dim=0).values).max().clamp(min=1e-6)
        )
        return params, scene_extent

    n_old = int(params["means"].shape[0])
    if target_count >= n_old:
        means = params["means"].data
        scene_extent = float(
            (means.max(dim=0).values - means.min(dim=0).values).max().clamp(min=1e-6)
        )
        return params, scene_extent

    device = params["means"].device
    times = params["times"].detach().reshape(-1)

    unique_t, inverse, counts = torch.unique(
        times, sorted=True, return_inverse=True, return_counts=True
    )

    n_groups = int(unique_t.numel())
    if target_count < n_groups:
        raise ValueError(
            f"target_count={target_count} is smaller than number of time groups={n_groups}"
        )

    desired = counts.float() / counts.sum().float() * float(target_count)
    quotas = torch.floor(desired).long()
    quotas = torch.clamp(quotas, min=1)
    quotas = torch.minimum(quotas, counts)

    remainder = int(target_count - int(quotas.sum().item()))
    frac = desired - torch.floor(desired)

    if remainder > 0:
        order = torch.argsort(frac, descending=True)
        for j in order.tolist():
            if remainder <= 0:
                break
            cap = int((counts[j] - quotas[j]).item())
            if cap <= 0:
                continue
            add = min(cap, remainder)
            quotas[j] += add
            remainder -= add

    elif remainder < 0:
        order = torch.argsort(frac, descending=False)
        for j in order.tolist():
            if remainder >= 0:
                break
            removable = max(0, int(quotas[j].item()) - 1)
            if removable <= 0:
                continue
            take = min(removable, -remainder)
            quotas[j] -= take
            remainder += take

    if int(quotas.sum().item()) != target_count:
        diff = int(target_count - int(quotas.sum().item()))
        for j in range(n_groups):
            if diff == 0:
                break
            cap = int((counts[j] - quotas[j]).item())
            if diff > 0 and cap > 0:
                add = min(cap, diff)
                quotas[j] += add
                diff -= add
            elif diff < 0 and int(quotas[j].item()) > 1:
                take = min(int(quotas[j].item()) - 1, -diff)
                quotas[j] -= take
                diff += take

    if int(quotas.sum().item()) != target_count:
        raise RuntimeError(
            f"Could not allocate exact target_count: got {int(quotas.sum().item())}, "
            f"wanted {target_count}"
        )

    try:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    except Exception:
        gen = None

    keep_parts = []
    for group_idx, q in enumerate(quotas.tolist()):
        idx = torch.nonzero(inverse == group_idx, as_tuple=False).squeeze(-1)
        if q >= int(idx.numel()):
            keep_parts.append(idx)
        else:
            if gen is not None:
                perm = torch.randperm(int(idx.numel()), device=device, generator=gen)[:q]
            else:
                perm = torch.randperm(int(idx.numel()), device=device)[:q]
            keep_parts.append(idx[perm])

    keep_idx = torch.cat(keep_parts, dim=0).sort().values.contiguous()

    for name, p in list(params.items()):
        if (
            torch.is_tensor(p)
            and p.data.ndim >= 1
            and int(p.data.shape[0]) == n_old
        ):
            params[name] = torch.nn.Parameter(
                p.data[keep_idx].contiguous(),
                requires_grad=p.requires_grad,
            )

    means = params["means"].data
    scene_extent = float(
        (means.max(dim=0).values - means.min(dim=0).values).max().clamp(min=1e-6)
    )

    log.info(
        "COLMAP target-count reduction: "
        f"{n_old:,} → {int(params['means'].shape[0]):,} primitives "
        f"across {n_groups} time groups"
    )

    return params, scene_extent


@dataclass
class ColmapScene:
    """What :func:`build_colmap_scene` returns."""
    parser: FreeTimeParser
    dataset: FreeTimeDataset
    training_cam_ids: list[int]
    cam_folders_by_pos: list[str]
    cameras: object
    n_frames: int
    gt_images: object
    gt_alphas: object
    gt_times: object


def build_colmap_scene(args: TrigsColmapArgs) -> ColmapScene:
    """Cameras, GT images/masks and per-camera capture times exactly as
    training sees them. Shared by training and ``trigs-eval-colmap`` so the
    evaluator cannot drift from what the trainer was supervised with."""
    # -- Step 3: parse COLMAP, build cameras + dataset ----------------------
    parser = FreeTimeParser(
        data_dir=str(args.data_dir),
        factor=args.data_factor,
        test_every=args.test_every,
        start_frame=args.start_frame,
        # FreeTimeParser expects an exclusive end frame.
        end_frame=args.end_frame + 1,
        reference_colmap=(
            str(args.reference_colmap) if args.reference_colmap is not None else None
        ),
    )
    dataset = FreeTimeDataset(parser, split="train", test_set = [])
    n_frames = parser.total_frames
    log.info(
        f"COLMAP: {parser.camtoworlds.shape[0]} cameras, "
        f"dataset len={len(dataset)}, n_frames={n_frames}"
    )

    # Build a CameraConfig compatible with the trainer. Pull only the
    # training-split cameras; index them in the same order the dataset uses.
    from trigs.core.camera import CameraConfig
    training_cam_ids = dataset.camera_indices                   # original parser indices
    c2w_all = torch.from_numpy(parser.camtoworlds).float()       # [N_all, 4, 4]
    c2w = c2w_all[training_cam_ids].to(args.device)              # [n_cams, 4, 4]
    viewmats = torch.linalg.inv(c2w)                             # world→camera

    # First-sample-per-camera lookup of K + (H, W).
    first_K_by_pos: Dict[int, torch.Tensor] = {}
    first_HW_by_pos: Dict[int, tuple] = {}
    orig_to_pos = {orig: pos for pos, orig in enumerate(training_cam_ids)}
    for i in range(len(dataset)):
        s = dataset[i]
        if s is None:
            continue
        pos = orig_to_pos.get(int(s["camera_idx"]))
        if pos is None or pos in first_K_by_pos:
            continue
        first_K_by_pos[pos] = torch.as_tensor(s["K"]).float()
        first_HW_by_pos[pos] = (int(s["image"].shape[0]), int(s["image"].shape[1]))
        if len(first_K_by_pos) == len(training_cam_ids):
            break
    if not first_K_by_pos:
        # Every sample came back None, which only happens when the frame-name
        # template the parser detected does not match what is on disk.
        example = dataset._get_image_path(
            parser.campaths[training_cam_ids[0]], args.start_frame
        )
        raise RuntimeError(
            "No K intrinsics could be read from the dataset: every "
            f"(camera, frame) sample resolved to a missing file. Expected e.g. "
            f"{example} (detected {parser.frame_digits}-digit names, "
            f"'{parser.image_format}'). Check the actual file names under "
            "images/<cam>/ and remove any stray non-frame files "
            "(macOS '._*' sidecars are the usual culprit)."
        )
    Ks = torch.stack(
        [first_K_by_pos[p] for p in range(len(training_cam_ids))], dim=0,
    ).to(args.device)
    hw_by_pos = [first_HW_by_pos[p] for p in range(len(training_cam_ids))]
    widths = [int(hw_by_pos[p][1]) for p in range(len(training_cam_ids))]
    heights = [int(hw_by_pos[p][0]) for p in range(len(training_cam_ids))]
    rs_signs = None
    rs_sensor = {}
    timing = None
    if getattr(args, "timing_json", None) is not None:
        from trigs import capture_timing as ct
        timing = ct.load_timing_candidate(args.timing_json)
        import hashlib
        args.config.capture_timing_json = (
            f"{Path(args.timing_json).resolve()}#sha256={hashlib.sha256(Path(args.timing_json).read_bytes()).hexdigest()}")
        if float(args.config.rs_readout_ms) == 0.0:
            args.config.rs_readout_ms = float(timing.readout_ms)
        elif abs(float(args.config.rs_readout_ms) - float(timing.readout_ms)) > 1e-9:
            raise SystemExit(f"--config.rs-readout-ms {args.config.rs_readout_ms} disagrees with "
                             f"{args.timing_json} readout_ms {timing.readout_ms}")
        if abs(float(timing.fps) - float(args.config.fps)) > 1e-6:
            raise SystemExit(f"{args.timing_json} fps {timing.fps} != --config.fps {args.config.fps}")
        args.config.rs_axis = timing.axis
        folders_for_cams = [
            _infer_camera_folders(parser, args.data_dir)[int(orig)] for orig in training_cam_ids
        ]
        entries = [timing.lookup(f) for f in folders_for_cams]
        missing = [f for f, e in zip(folders_for_cams, entries) if e is None]
        if missing:
            raise SystemExit(f"{args.timing_json} has no entry for training cameras {missing}")
        if timing.axis == "row":
            rs_signs = [1.0 if e.rs else 0.0 for e in entries]
            args.config.rs_sensor_coords = True
            rs_sensor = _rs_sensor_mapping(parser, training_cam_ids, args.device)
        log.info("Timing file %s: readout %.2f ms along %ss; rs on %d / off %d cameras; statuses %s",
                 args.timing_json, timing.readout_ms, timing.axis,
                 sum(1 for e in entries if e.rs), sum(1 for e in entries if not e.rs),
                 {st: sum(1 for e in entries if e.status == st) for st in sorted({e.status for e in entries})})
    if args.config.rs_enabled() and rs_signs is None:
        if args.rs_rotation_json is None:
            raise SystemExit(
                "--config.rs-readout-ms > 0 requires --rs-rotation-json. "
                "The readout DIRECTION is per-camera; without it the "
                "correction would be applied with the wrong sign on part of "
                "the rig, which is worse than leaving it off."
            )
        folders_for_cams = [
            _infer_camera_folders(parser, args.data_dir)[int(orig)]
            for orig in training_cam_ids
        ]
        rs_signs = _load_rs_signs(
            Path(args.rs_rotation_json), folders_for_cams, flip=args.rs_flip_sign
        )
        if timing is not None:  # column-axis timing file: its per-camera rs flags can only switch cameras off
            rs_signs = [sg if timing.lookup(f).rs else 0.0 for sg, f in zip(rs_signs, folders_for_cams)]
        n_l = int(sum(1 for v in rs_signs if v > 0))
        log.info(
            "Rolling shutter ON: readout %.2f ms full-span, %d 'l' / %d 'r' cameras",
            args.config.rs_readout_ms, n_l, len(rs_signs) - n_l,
        )

    cameras = CameraConfig.from_matrices(
        viewmats=viewmats,
        Ks=Ks,
        widths=widths,
        heights=heights,
        rs_signs=rs_signs,
        **rs_sensor,
    )

    # -- Step 4: build lazy GT image accessor (no pre-loading into RAM) -----
    # The trainer accesses gt_images[frame_idx][cam_idx] where frame_idx is
    # 0..n_frames-1 (relative offset from start_frame, same as the epoch-pair
    # loop) and cam_idx is 0..n_cameras-1 (training-split position).
    # _LazyGTImages satisfies this interface but loads each image from disk
    # only when requested, keeping a small LRU cache.
    n_cameras = len(training_cam_ids)
    all_camera_folders = _infer_camera_folders(parser, args.data_dir)
    cam_folders_by_pos = [all_camera_folders[int(orig)] for orig in training_cam_ids]
    raw_warps_by_pos = _raw_warps_by_pos(parser, training_cam_ids)
    if args.masks_root is not None or args.foregrounds_root is not None:
        log.info(
            "Masks/foregrounds get the dataset's undistortion + ROI crop on %d/%d cameras",
            sum(w is not None for w in raw_warps_by_pos), len(raw_warps_by_pos),
        )

    if args.foregrounds_root is not None:
        gt_images = _LazyGTImages(
            images_root=args.foregrounds_root,
            cam_folders_by_pos=cam_folders_by_pos,
            source_start_frame=args.start_frame,
            target_hw_by_pos=hw_by_pos,
            image_pattern=args.foreground_pattern,
            cache_size=args.foreground_cache_size,
            raw_warps_by_pos=raw_warps_by_pos,
        )
        gt_images.validate_boundaries(n_frames)
        log.info(
            "Using decontaminated foreground RGB from %s with pattern %s",
            args.foregrounds_root,
            args.foreground_pattern,
        )
    else:
        gt_images = _DatasetGTImages(
            dataset=dataset,
            n_cameras=n_cameras,
            orig_to_pos=orig_to_pos,
            cache_size=300,
        )

    # Verify the dataset-backed loader matches the actual FreeTimeDataset
    # sample. Foreground targets intentionally differ from captured RGB.
    item_by_pair = {}
    for i, (orig_cam, frame_offset) in enumerate(dataset.pairs):
        cp = orig_to_pos.get(int(orig_cam))
        if cp is not None:
            item_by_pair[(int(frame_offset), int(cp))] = i

    def to_u8_hwc(x):
        x = torch.as_tensor(x)
        if x.dtype != torch.uint8:
            xf = x.float()
            if xf.max() <= 2.0:
                xf = xf * 255.0
            x = xf.clamp(0, 255).to(torch.uint8)
        return x.cpu()

    if args.foregrounds_root is None:
        for fo in (0, min(5, n_frames - 1), n_frames - 1):
            for cp in (0, min(1, n_cameras - 1), n_cameras - 1):
                item_idx = gt_images._item_idx[(fo, cp)]
                ds = _DatasetGTImages._to_uint8_hwc(dataset[item_idx]["image"])
                lazy = gt_images[fo][cp]
                diff = (ds.float() - lazy.float()).abs()
                log.debug(
                    "GT_DATASET_BACKED_COMPARE fo %d cp %d mean_abs %g max_abs %g",
                    fo, cp, float(diff.mean()), float(diff.max()),
                )
                assert float(diff.mean()) < 0.01
                assert float(diff.max()) < 1.0

    for fo in (0, n_frames - 1):
        for cp in range(n_cameras):
            img = gt_images[fo][cp]
            expected = hw_by_pos[cp]
            got = tuple(int(x) for x in img.shape[:2])
            if got != tuple(expected):
                raise RuntimeError(
                    f"Direct GT image loader shape mismatch at frame_offset={fo}, "
                    f"cam_pos={cp}: got={got}, expected={expected}"
                )

    log.debug("Direct GT image loader validated first/last frame shapes for all train cameras.")

    if args.masks_root is not None:
        all_camera_folders = _infer_camera_folders(parser, args.data_dir)

        max_cam_id = max(int(x) for x in training_cam_ids)
        if max_cam_id >= len(all_camera_folders):
            raise RuntimeError(
                f"Training cam id {max_cam_id} is out of range for inferred "
                f"camera folders length {len(all_camera_folders)}. "
                f"Folders: {all_camera_folders[:5]}..."
            )

        cam_folders_by_pos = [
            all_camera_folders[int(orig_cam_id)]
            for orig_cam_id in training_cam_ids
        ]

        log.info(
            "Using foreground masks from %s; first camera folders: %s",
            args.masks_root,
            cam_folders_by_pos[:5],
        )

        gt_alphas = _LazyGTMasks(
            dataset=dataset,
            n_cameras=n_cameras,
            orig_to_pos=orig_to_pos,
            masks_root=args.masks_root,
            cam_folders_by_pos=cam_folders_by_pos,
            source_start_frame=args.start_frame,
            mask_pattern=args.mask_pattern,
            mask_frame_offset=args.mask_frame_offset,
            cache_size=args.mask_cache_size,
            error_on_missing=args.error_on_missing_mask,
            missing_value=args.mask_missing_value,
            raw_warps_by_pos=raw_warps_by_pos,
        )
        if args.error_on_missing_mask:
            gt_alphas.validate_boundaries(n_frames)
        log.info(
            "Mask frame mapping: source frames %d..%d -> mask frames %d..%d",
            args.start_frame,
            args.start_frame + n_frames - 1,
            args.start_frame + args.mask_frame_offset,
            args.start_frame + n_frames - 1 + args.mask_frame_offset,
        )
    else:
        gt_alphas = _DummyGTAlphas()

    frame_timestamps_file = args.data_dir / "frame_timestamps.json"
    sync_file = args.data_dir / "sync.json"
    gt_times = _build_gt_times_from_ladder(
        frame_timestamps_file=frame_timestamps_file,
        parser=parser,
        data_dir=args.data_dir,
        training_cam_ids=training_cam_ids,
        n_frames=n_frames,
    )
    if gt_times is not None and timing is not None:
        raise SystemExit(f"{frame_timestamps_file} and --timing-json both define capture times; use one")
    if gt_times is not None:
        log.info(f"✅ Loaded per-frame camera-ladder timestamps from {frame_timestamps_file}")
    elif timing is not None:
        gt_times = _build_gt_times_from_sync(
            sync_file=Path(args.timing_json),
            parser=parser,
            data_dir=args.data_dir,
            training_cam_ids=training_cam_ids,
            n_frames=n_frames,
            fps=args.config.fps,
            sync_data=timing.as_sync_offsets(),
        )
        log.info(f"✅ Loaded MIDPOINT camera constants from {args.timing_json} (sync.json not read)")
    else:
        gt_times = _build_gt_times_from_sync(
            sync_file=sync_file,
            parser=parser,
            data_dir=args.data_dir,
            training_cam_ids=training_cam_ids,
            n_frames=n_frames,
            fps=args.config.fps,
        )
        if gt_times is not None:
            log.info(f"✅ Loaded sub-frame temporal offsets from {sync_file}")
        else:
            log.warning(
                "No frame_timestamps.json or sync.json found. Falling back to "
                "discrete frame buckets."
            )
    return ColmapScene(
        parser=parser, dataset=dataset, training_cam_ids=list(training_cam_ids),
        cam_folders_by_pos=cam_folders_by_pos, cameras=cameras, n_frames=n_frames,
        gt_images=gt_images, gt_alphas=gt_alphas, gt_times=gt_times,
    )


_CONFIG_FPS_DEFAULT = TrigsTrainingConfig().fps


def resolve_npz_defaults(args: "TrigsColmapArgs") -> None:
    """Fill the frame range, keyframe stride and fps from the NPZ when they are not given.

    The NPZ's ``keyframe_step`` and ``fps`` are what its rows were built with: a different
    keyframe step silently drops keyframes and a different fps rescales every time, so an
    explicit value that disagrees stops the run instead."""
    npz_path = args.npz_path if args.npz_path is not None else Path(args.output_dir) / "keyframes.npz"
    meta: dict = {}
    if args.hybrid_anchor_npz is None and Path(npz_path).exists():
        import numpy as np
        with np.load(npz_path, allow_pickle=False) as z:
            for key in ("source_frame_start", "source_frame_end", "keyframe_step", "fps"):
                if key in z.files:
                    meta[key] = z[key].item()
    if args.start_frame is None:
        if "source_frame_start" not in meta:
            raise SystemExit("--start-frame is required (no NPZ with source_frame_start to read it from)")
        args.start_frame = int(meta["source_frame_start"])
    if args.end_frame is None:
        if "source_frame_end" not in meta:
            raise SystemExit("--end-frame is required (no NPZ with source_frame_end to read it from)")
        args.end_frame = int(meta["source_frame_end"])
    npz_step = int(meta["keyframe_step"]) if "keyframe_step" in meta else None
    if args.keyframe_step is None:
        args.keyframe_step = npz_step if npz_step is not None else 5
    elif npz_step is not None and int(args.keyframe_step) != npz_step:
        raise SystemExit(f"--keyframe-step {args.keyframe_step} != the NPZ's keyframe_step {npz_step}: "
                         "this would silently drop keyframes. Omit --keyframe-step to use the NPZ's.")
    args.config.keyframe_stride = int(args.keyframe_step)
    if "fps" in meta:
        npz_fps = float(meta["fps"])
        if abs(float(args.config.fps) - npz_fps) > 1e-6:
            if abs(float(args.config.fps) - _CONFIG_FPS_DEFAULT) > 1e-9:
                raise SystemExit(f"--config.fps {args.config.fps} != the NPZ's fps {npz_fps}")
            args.config.fps = npz_fps


def write_run_args(args: "TrigsColmapArgs", output_path: Path) -> None:
    """Record the resolved run inputs (<output>/run_args.json) so trigs-eval-colmap and other tools
    need not repeat them. The training config itself is saved in every snapshot and checkpoint."""
    import json
    keys = ("data_dir", "npz_path", "timing_json", "start_frame", "end_frame", "keyframe_step",
            "data_factor", "test_every", "reference_colmap")
    payload = {k: (str(v) if isinstance(v, Path) else v) for k in keys if (v := getattr(args, k, None)) is not None}
    (output_path / "run_args.json").write_text(json.dumps(payload, indent=1) + "\n")


def resolve_target_count(args: "TrigsColmapArgs", params: dict) -> None:
    """target_count = target_count_per_keyframe x keyframes in the training window, unless
    --config.target-count is given. Rows between keyframes (speed-aware sub-keyframes) are
    kept whole on top of it by _apply_split_budget_to_prebuilt_params."""
    if args.config.target_count is not None or int(args.config.target_count_per_keyframe) <= 0:
        return
    times = params["times"].detach().reshape(-1)
    dyn = ~(params["is_static"].detach().reshape(-1) > 0.5) if "is_static" in params else torch.ones_like(times, dtype=torch.bool)
    frame_off = torch.round(times[dyn].double() * float(args.config.fps)).long()
    stride = max(int(args.keyframe_step), 1)
    n_keyframes = int(torch.unique(frame_off[(frame_off % stride) == 0]).numel())
    if n_keyframes == 0:
        return
    args.config.target_count = int(args.config.target_count_per_keyframe) * n_keyframes
    log.info("Splat budget: %d per keyframe x %d keyframes = %d dynamic rows (plus sub-keyframe rows)",
             int(args.config.target_count_per_keyframe), n_keyframes, int(args.config.target_count))


def main(args: TrigsColmapArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(TrigsColmapArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.config.diagnostics else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    for noisy in ("matplotlib", "PIL", "urllib3", "h5py"):
        # Third-party DEBUG chatter stays off even when --config.diagnostics turns ours on.
        logging.getLogger(noisy).setLevel(logging.WARNING)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolve_npz_defaults(args)
    write_run_args(args, output_path)

    snap_dir = apply_default_snapshot_dir(args.config, output_path)
    if snap_dir is not None:
        log.info(
            "Viewer snapshots every %d steps -> %s (view: viewer.py --watch %s)",
            args.config.viewer_snapshot_every, snap_dir, snap_dir,
        )
    if args.end_frame < args.start_frame:
        raise ValueError(
            f"end_frame ({args.end_frame}) must be >= start_frame ({args.start_frame})"
        )
    if args.foregrounds_root is not None:
        missing_flags = []
        if args.masks_root is None:
            missing_flags.append("--masks-root")
        if not args.config.masked_rgb_loss:
            missing_flags.append("--config.masked-rgb-loss")
        if not args.config.masked_rgb_composite:
            missing_flags.append("--config.masked-rgb-composite")
        if missing_flags:
            raise ValueError(
                "--foregrounds-root contains straight foreground RGB and requires "
                + ", ".join(missing_flags)
            )

    hybrid_pair_mode = (args.hybrid_anchor_npz is not None) or (args.hybrid_coverage_npz is not None)
    if hybrid_pair_mode and (args.hybrid_anchor_npz is None or args.hybrid_coverage_npz is None):
        raise ValueError(
            "Hybrid SelfCap-style mode requires both --hybrid-anchor-npz and "
            "--hybrid-coverage-npz."
        )

    if hybrid_pair_mode:
        # -- Step 1/2: SelfCap-style dense/sparse resolution from two NPZs ----
        # No merged NPZ is used here. The builder treats anchor as dense_pcds
        # and coverage as pcds, resolving dense-preferred fallback before scale,
        # velocity, anchor, twist, FPS, and pack.
        log.info(
            "Using SelfCap-style hybrid NPZ pair: anchor=%s coverage=%s",
            args.hybrid_anchor_npz,
            args.hybrid_coverage_npz,
        )
        params, sh_degree, scene_extent = build_canonical_from_npz_pair_as_selfcap(
            anchor_npz_path=args.hybrid_anchor_npz,
            coverage_npz_path=args.hybrid_coverage_npz,
            fps=args.config.fps,
            keyframe_stride=args.keyframe_step,
            source_frame_start=args.start_frame,
            source_frame_end=args.end_frame,
            static_from=args.hybrid_static_from,
            target_count=args.config.target_count,
            init_opacity=args.config.init_opacity,
            scale_knn_k=4,
            scale_knn_factor=args.config.init_scale_knn_factor,
            anchor_radius_frac=args.config.anchor_radius_frac,
            local_twist_k=args.config.local_twist_k,
            local_twist_reg=args.config.local_twist_reg,
            init_sigma_t=args.config.init_sigma_t,
            max_sh_degree=3,
            omega_init=True,
            duration_init_mode=args.config.duration_init_mode,
            duration_min_s=args.config.duration_min_s,
            duration_max_s=args.config.duration_max_s,
            dense_duration_multiplier=args.config.dense_duration_multiplier,
            sparse_duration_multiplier=args.config.sparse_duration_multiplier,
            device=args.device,
            verbose=True,
        )
        used_selfcap_hybrid_pair = True
    else:
        # -- Step 1: produce the ROMA-combined NPZ --------------------------
        npz_path = args.npz_path if args.npz_path is not None else output_path / "keyframes.npz"
        if not npz_path.exists():
            if args.roma_input_dir is None:
                raise SystemExit(f"{npz_path} does not exist: pass --npz-path or --roma-input-dir")
            _run_roma_combine(
                input_dir=args.roma_input_dir,
                output_path=npz_path,
                start_frame=args.start_frame,
                end_frame=args.end_frame,
                keyframe_step=args.keyframe_step,
            )
        else:
            log.info(f"Using existing NPZ: {npz_path}")

        # -- Step 2: build canonical params from NPZ ------------------------
        # Chunked runs keep the whole model in host RAM (the trainer only
        # puts one chunk on the GPU), so build it there too; the per-keyframe
        # fits still run on the GPU.
        chunked = float(getattr(args.config, "temporal_chunk_len_s", 0.0)) > 0.0
        params_device = "cpu" if chunked else args.device
        params, sh_degree, scene_extent = build_canonical_from_npz(
            npz_path=npz_path,
            fps=args.config.fps,
            keyframe_stride=args.keyframe_step,
            init_opacity=args.config.init_opacity,
            scale_knn_factor=args.config.init_scale_knn_factor,
            anchor_radius_frac=args.config.anchor_radius_frac,
            init_sigma_t=args.config.init_sigma_t,
            init_sigma_t_mode=args.config.init_sigma_t_mode,
            init_sigma_t_floor_frac=args.config.init_sigma_t_floor_frac,
            init_sigma_t_speed_spacings=args.config.init_sigma_t_speed_spacings,
            init_sigma_t_max=args.config.init_sigma_t_max,
            init_accel_mode=args.config.init_accel_mode,
            init_accel_scale=args.config.init_accel_scale,
            init_accel_fit_iters=args.config.init_accel_fit_iters,
            init_chain_fit=args.config.init_chain_fit,
            init_chain_min_span_frames=args.config.init_chain_min_span_frames,
            local_twist_k=args.config.local_twist_k,
            local_twist_reg=args.config.local_twist_reg,
            source_frame_start=args.start_frame,
            source_frame_end=args.end_frame,
            keep_static_when_filtering=True,
            filter_to_local_keyframe_stride=True,
            device=params_device,
            compute_device=args.device,
        )
        log.info(
            f"Canonical set: {params['means'].shape[0]:,} primitives, "
            f"scene_extent={scene_extent:.3f} (held on {params['means'].device})"
        )
        # Backward-compatible merged-NPZ mode: post-build kind patch.
        _apply_anchor_init_from_npz_kind(
            params=params,
            npz_path=npz_path,
            sparse_sigma_t=args.config.init_sigma_t,
            device=params_device,
        )
        used_selfcap_hybrid_pair = False

    sig = params["log_durations"].detach().exp().cpu().numpy()
    import numpy as np
    log.debug(f"POST-INIT σ_t: min={sig.min():.4f} max={sig.max():.4f} "
              f"mean={sig.mean():.4f} median={np.median(sig):.4f}")
    log.debug(f"  fraction > 3s: {(sig > 3.0).mean():.4f}")
    log.debug(f"  fraction > 4.5s: {(sig > 4.5).mean():.4f}")
    if "is_static" in params:
        static_np = (
            params["is_static"].detach().reshape(-1).cpu().numpy() > 0.5
        )
        dyn_sig = sig.reshape(-1)[~static_np]
        if dyn_sig.size:
            semantic_thr = float(args.config.temporal_active_threshold)
            cull_override = float(
                getattr(args.config, "temporal_active_cull_threshold", 0.0)
            )
            cull_thr = cull_override if cull_override > 0 else semantic_thr
            cull_thr = min(max(cull_thr, 1e-8), min(max(semantic_thr, 1e-8), 1.0 - 1e-8))
            radius_scale = math.sqrt(-2.0 * math.log(cull_thr))
            radius = dyn_sig * radius_scale + float(
                args.config.temporal_active_extra_margin_s
            )
            log.debug(
                "  dynamic σ_t p50/p90/p99/max: "
                f"{np.quantile(dyn_sig, 0.50):.4f}/"
                f"{np.quantile(dyn_sig, 0.90):.4f}/"
                f"{np.quantile(dyn_sig, 0.99):.4f}/"
                f"{dyn_sig.max():.4f}s"
            )
            log.debug(
                f"  dynamic cull radius @ γ={cull_thr:g} "
                "p50/p90/p99/max: "
                f"{np.quantile(radius, 0.50):.4f}/"
                f"{np.quantile(radius, 0.90):.4f}/"
                f"{np.quantile(radius, 0.99):.4f}/"
                f"{radius.max():.4f}s; "
                f"chunk overlap={args.config.temporal_chunk_overlap_s:.4f}s"
            )
    log.debug(f"mean ‖ν_base‖ = {params['zeta_base_t'].norm(dim=-1).mean().item():.4f} world/s "
              "(roughly typical displacement per frame x fps)")

    # In SelfCap-hybrid-pair mode target_count has already been applied inside
    # the builder with count-weighted per-keyframe FPS. Preserve the old split
    # budget only for single/merged NPZ mode.
    if not used_selfcap_hybrid_pair:
        resolve_target_count(args, params)
    if (not used_selfcap_hybrid_pair) and args.config.target_count is not None and args.config.target_count > 0:
        params, scene_extent = _apply_split_budget_to_prebuilt_params(
            params,
            target_count=args.config.target_count,
            static_target_count=100000,
            protect_speed_top_frac=float(getattr(args.config, "budget_protect_speed_top_frac", 0.0)),
            fps=float(args.config.fps),
            config=args.config,
            compute_device=args.device,
            keyframe_stride=int(args.keyframe_step),
        )

    log.info(
        f"Canonical set after target-count: {params['means'].shape[0]:,} primitives, "
        f"scene_extent={scene_extent:.3f}"
    )

    if args.config.init_rescale_after_budget:
        before = params["scales"].detach().exp().amax(dim=-1).median().item()
        n_rescaled = _rescale_dynamic_init_after_budget(
            params, factor=args.config.init_scale_knn_factor,
        )
        after = params["scales"].detach().exp().amax(dim=-1).median().item()
        log.info(
            f"init-rescale-after-budget: {n_rescaled:,} dynamic rows re-sized on the kept set; "
            f"median scale {before:.5f} -> {after:.5f}"
        )

    if args.static_as_dynamic:
        n_released = release_static_rows_to_dynamic(params)
        log.info(f"--static-as-dynamic: {n_released:,} static rows now train as dynamic")

    # -- Steps 3-4: cameras, GT images/masks, capture times -----------------
    scene = build_colmap_scene(args)
    parser, training_cam_ids = scene.parser, scene.training_cam_ids
    cameras, n_frames = scene.cameras, scene.n_frames
    gt_images, gt_alphas, gt_times = scene.gt_images, scene.gt_alphas, scene.gt_times

    # -- Step 5: train ------------------------------------------------------
    tb_writer = None
    if args.config.tb_log_every > 0 or args.config.tb_stats_every > 0:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_path / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

    observability_json = getattr(args.config, "camera_time_offset_observability_json", None)
    camera_time_offset_scale = None
    if observability_json and getattr(args.config, "learn_camera_time_offsets", False):
        camera_time_offset_scale = _build_camera_time_offset_scale(
            observability_file=Path(observability_json),
            parser=parser,
            data_dir=args.data_dir,
            training_cam_ids=training_cam_ids,
        )
        if camera_time_offset_scale is not None:
            log.info(f"✅ Loaded camera time-offset observability scale from {observability_json}")
            params["camera_time_offset_scale"] = torch.nn.Parameter(
                camera_time_offset_scale.to(params["means"].device), requires_grad=False,
            )
        else:
            log.warning(f"--config.camera-time-offset-observability-json set but not found: {observability_json}")

    canonical_scene_extent = float(scene_extent)
    parser_scene_scale = float(getattr(parser, "scene_scale", canonical_scene_extent))
    camera_extent = training_camera_extent(
        cameras.viewmats, exclude=args.config.holdout_views,
    )
    extent_mode = str(args.config.scene_extent_mode).lower()
    if extent_mode == "official":
        train_scene_extent = camera_extent
    elif extent_mode == "parser":
        train_scene_extent = parser_scene_scale
    else:
        raise SystemExit(
            f"--config.scene-extent-mode must be 'official' or 'parser', got {extent_mode!r}"
        )

    log.info(
        "Scene extent for optimizer scaling: "
        f"canonical={canonical_scene_extent:.6f}, "
        f"parser.scene_scale={parser_scene_scale:.6f}, "
        f"training-camera radius={camera_extent:.6f}, "
        f"using={train_scene_extent:.6f} ({extent_mode})"
    )

    trainer = TrigsTrainer(device=args.device)

    # Opt-in de-cohorting of initial temporal supports (see
    # TrigsTrainingConfig.temporal_init_jitter_s). Applied post-build so it
    # covers every init path uniformly; static rows untouched.
    _jit = float(getattr(args.config, "temporal_init_jitter_s", 0.0))
    _dil = float(getattr(args.config, "temporal_init_sigma_dilate", 1.0))
    if _jit > 0.0 or _dil != 1.0:
        with torch.no_grad():
            dyn = torch.ones(
                params["times"].shape[0],
                dtype=torch.bool,
                device=params["times"].device,
            )
            if "is_static" in params:
                dyn = params["is_static"].reshape(-1) <= 0.5
            if _jit > 0.0:
                noise = (
                    torch.rand_like(params["times"].data) - 0.5
                ) * 2.0 * _jit
                params["times"].data[dyn] += noise[dyn]
                t_max_init = float(max(n_frames - 1, 0)) / max(
                    float(args.config.fps), 1e-6
                )
                params["times"].data.clamp_(0.0, t_max_init)
            if _dil != 1.0:
                params["log_durations"].data[dyn] += math.log(max(_dil, 1e-3))
        log.info(
            "Init de-cohort: jitter=+/-%.3fs sigma_dilate=x%.2f on %d dynamic rows",
            _jit, _dil, int(dyn.sum()),
        )

    try:
        if float(getattr(args.config, "temporal_chunk_len_s", 0.0)) > 0.0:
            result = trainer.train_chunked(
                params=params,
                sh_degree=sh_degree,
                scene_extent=train_scene_extent,
                cameras=cameras,
                n_frames=n_frames,
                gt_images_per_frame=gt_images,
                gt_alphas_per_frame=gt_alphas,
                config=args.config,
                gt_times_per_frame=gt_times,
                writer=tb_writer,
                # The whole model only has to fit in host RAM; assembling it
                # on the GPU for the save would OOM at the end of a long run.
                result_device="cpu",
            )
        else:
            result = trainer.train_from_params(
                params=params,
                sh_degree=sh_degree,
                scene_extent=train_scene_extent,
                cameras=cameras,
                n_frames=n_frames,
                gt_images_per_frame=gt_images,
                gt_alphas_per_frame=gt_alphas,
                config=args.config,
                gt_times_per_frame=gt_times,
                writer=tb_writer,
            )
    finally:
        if tb_writer is not None:
            tb_writer.close()

    ckpt_path = output_path / "trigs_ckpt.pt"
    torch.save(
        {
            "params": {k: v.detach().cpu() for k, v in result.params.items()},
            "temporal_params": result.temporal_params,
            "motion_params": result.motion_params,
            "metrics": result.metrics,
        },
        ckpt_path,
    )
    log.info(f"Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main(tyro.cli(TrigsColmapArgs))
