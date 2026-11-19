"""CLI entry point for the TriGS trainer (SelfCap / 4DV mode).

Takes a SelfCap scene preprocessed by ``scripts/preprocess_selfcap.py``
(ffmpeg-extracted jpgs + EasyVolcap calibration + per-frame pcds) and
trains a TriGS model against it.

Pipeline:
  1. Parse EasyVolcap extri.yml / intri.yml via :class:`SelfCapParser`.
  2. Build the canonical set from the first-frame pcd via
     :func:`build_canonical_from_pcd_sequence`.
  3. Collect GT images from :class:`SelfCapDataset` (inherits undistortion
     + patch crop from FreeTimeDataset).
  4. Run :meth:`TrigsTrainer.train_from_params` and checkpoint.

Usage:
    uv run trigs-train-selfcap \
        --data-dir /scratch/corgi_prepped \
        --output-dir runs/corgi \
        --config.n-steps 30000
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from trigs.compat.device import resolve_device  # macOS port
import tyro
from gsply import GSTensor
from tqdm import tqdm

from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.core.viewer_snapshot import apply_default_snapshot_dir
from trigs.init.from_selfcap import build_canonical_from_pcd_sequence
from trigs.io.selfcap import SelfCapParser, SelfCapDataset

log = logging.getLogger(__name__)


@dataclass
class TrigsSelfCapArgs:
    """Arguments for the TriGS SelfCap trainer."""

    data_dir: Path
    """Preprocessed SelfCap directory (contains ``preprocess_meta.json``,
    ``optimized/``, ``images/``, ``pcds/``)."""

    output_dir: Path
    """Directory to write checkpoint + optional debug renders."""

    data_factor: int = 1
    """Additional image downsample factor on top of the preprocessing factor."""

    test_camera_idx: int = 0
    """Single held-out camera index used as the test view."""

    dense_subdir: str = "dense_pcds"
    """Preferred per-keyframe PLY source (RealityCapture-dense when present)."""

    sparse_subdir: str = "pcds"
    """Fallback per-keyframe PLY source + neighbor source when dense files
    are unavailable."""

    init_target_count: int | None = None
    """Optional 4D-FPS target count for the canonical set (None = keep all)."""

    defer_init_target_to_prune: bool = False
    """If True, skip init-time downsampling and prune during training.

    Uses ``init_target_count`` as the delayed target budget and enables the
    trainer's periodic opacity-based prune stage. Relocation is disabled
    until the target count is reached.
    """

    defer_init_target_start_iter: int = 0
    """First step where delayed pruning may run in defer mode."""

    defer_init_target_every: int = 100
    """Run delayed pruning every N steps while above target in defer mode."""

    no_sync: bool = False
    """Disable sync.json per-camera timing correction."""

    preload_gt: bool = False
    """If True, preload every (frame, camera) GT image into RAM before training
    (faster per-step, large memory footprint). If False (default), GT jpgs are
    decoded lazily on each training step — slower but uses O(1) RAM."""

    config: TrigsTrainingConfig = field(default_factory=TrigsTrainingConfig)
    """TriGS training configuration."""

    device: str = "auto"

    save_plys: bool = True
    """If True, decode and save per-frame PLYs after training."""


class _LazyGTRow:
    """One frame's worth of GT, indexed by training-split camera position.

    Resolves a (frame, cam_pos) pair to the underlying dataset sample on
    access. Zero-fills missing cells so the trainer's all_pairs iteration
    is safe even when some (frame, cam) samples are missing on disk.
    """

    __slots__ = ("_lazy", "_fi", "_kind")

    def __init__(self, lazy: "_LazyGT", fi: int, kind: str):
        self._lazy = lazy
        self._fi = fi
        self._kind = kind  # "image" or "alpha"

    def __getitem__(self, ci: int) -> torch.Tensor:
        idx = self._lazy._pair_index.get((self._fi, ci))
        if idx is None:
            return self._lazy._zero(self._kind, ci)
        sample = self._lazy._dataset[idx]
        if sample is None:
            return self._lazy._zero(self._kind, ci)
        img = sample["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img_u8 = img.clamp(0, 255).to(torch.uint8)
        if self._kind == "image":
            return img_u8
        # alpha: full-255 of matching H, W.
        H, W, _ = img_u8.shape
        return torch.full((H, W, 1), 255, dtype=torch.uint8)


class _LazyGT:
    """Dict-of-lists shim over SelfCapDataset — lazy per-step GT decode."""

    def __init__(self, dataset: SelfCapDataset, n_cameras: int, kind: str):
        self._dataset = dataset
        self._n_cameras = n_cameras
        self._kind = kind
        # (frame_offset, cam_pos) -> flat dataset index.
        orig_to_pos = {orig: pos for pos, orig in enumerate(dataset.camera_indices)}
        self._pair_index: Dict[tuple[int, int], int] = {}
        for i, (cam_orig, fo) in enumerate(dataset.pairs):
            pos = orig_to_pos.get(cam_orig)
            if pos is not None:
                self._pair_index[(fo, pos)] = i

        # Learn per-camera (H, W) for zero placeholders.
        self._HW_by_cam: Dict[int, tuple[int, int]] = {}
        for (_fi, ci), idx in self._pair_index.items():
            if ci in self._HW_by_cam:
                continue
            s = dataset[idx]
            if s is None:
                continue
            self._HW_by_cam[ci] = (int(s["image"].shape[0]), int(s["image"].shape[1]))
            if len(self._HW_by_cam) == n_cameras:
                break
        if len(self._HW_by_cam) != n_cameras:
            missing = sorted(set(range(n_cameras)) - set(self._HW_by_cam.keys()))
            raise RuntimeError(
                "SelfCapDataset yielded no valid samples for some cameras: "
                f"{missing}"
            )

    def _zero(self, kind: str, ci: int) -> torch.Tensor:
        H, W = self._HW_by_cam[ci]
        if kind == "image":
            return torch.zeros(H, W, 3, dtype=torch.uint8)
        return torch.zeros(H, W, 1, dtype=torch.uint8)

    def __getitem__(self, fi: int) -> _LazyGTRow:
        return _LazyGTRow(self, fi, self._kind)


def _lazy_gt_from_dataset(
    dataset: SelfCapDataset, n_cameras: int,
) -> tuple[_LazyGT, _LazyGT]:
    """Return (images, alphas) as lazy-decode dict-of-lists shims."""
    return _LazyGT(dataset, n_cameras, "image"), _LazyGT(dataset, n_cameras, "alpha")


def _build_gt_times(
    dataset: SelfCapDataset,
    n_frames: int,
    n_cameras: int,
) -> Dict[int, List[float | None]]:
    """Build per-(frame, camera) sync-corrected timestamps from parser metadata.

    Pure metadata — no JPG decode — so it's safe to call eagerly regardless
    of ``preload_gt``. Mirrors the computation in
    :meth:`SelfCapDataset.__getitem__` (``frame_offset / fps − sync[cam]``).
    """
    parser = dataset.parser
    fps = float(getattr(parser, "fps", 60.0))
    fps_safe = max(fps, 1e-6)
    sync_offsets = getattr(parser, "sync_offsets", {}) or {}
    orig_to_pos = {orig: pos for pos, orig in enumerate(dataset.camera_indices)}

    gt_times: Dict[int, List[float | None]] = {
        fi: [None] * n_cameras for fi in range(n_frames)
    }
    for cam_orig, fo in dataset.pairs:
        pos = orig_to_pos.get(int(cam_orig))
        if pos is None or not (0 <= fo < n_frames):
            continue
        cam_name = parser.camera_names[int(cam_orig)]
        sync_off = float(sync_offsets.get(cam_name, 0.0))
        gt_times[fo][pos] = float(fo) / fps_safe - sync_off
    return gt_times


def _load_gt_from_dataset(
    dataset: SelfCapDataset,
    n_frames: int,
    n_cameras: int,
) -> tuple[Dict[int, List[torch.Tensor]], Dict[int, List[torch.Tensor]]]:
    """Assemble per-(frame, camera) GT image + alpha dicts.

    Mirrors the helper in ``train_from_colmap.py``: missing cells get a
    zero placeholder so the trainer's ``all_pairs`` iteration is safe.
    """
    orig_to_pos = {orig: pos for pos, orig in enumerate(dataset.camera_indices)}

    gt_images: Dict[int, List[torch.Tensor | None]] = {
        fi: [None] * n_cameras for fi in range(n_frames)
    }
    gt_alphas: Dict[int, List[torch.Tensor | None]] = {
        fi: [None] * n_cameras for fi in range(n_frames)
    }
    hw_by_cam: Dict[int, tuple[int, int]] = {}

    for i in tqdm(range(len(dataset)), desc="Collect GT"):
        sample = dataset[i]
        if sample is None:
            continue
        fi = int(sample["frame_offset"])
        cam_pos = orig_to_pos.get(int(sample["camera_idx"]))
        if cam_pos is None or not (0 <= fi < n_frames):
            continue
        img = sample["image"]
        if not torch.is_tensor(img):
            img = torch.as_tensor(img)
        img_u8 = img.clamp(0, 255).to(torch.uint8)
        gt_images[fi][cam_pos] = img_u8
        H, W, _ = img_u8.shape
        hw_by_cam[cam_pos] = (H, W)
        gt_alphas[fi][cam_pos] = torch.full((H, W, 1), 255, dtype=torch.uint8)

    if len(hw_by_cam) != n_cameras:
        missing = sorted(set(range(n_cameras)) - set(hw_by_cam.keys()))
        raise RuntimeError(
            "SelfCapDataset yielded no valid samples for some cameras: "
            f"{missing}"
        )
    zero_img_by_cam = {
        ci: torch.zeros(hw_by_cam[ci][0], hw_by_cam[ci][1], 3, dtype=torch.uint8)
        for ci in range(n_cameras)
    }
    zero_alpha_by_cam = {
        ci: torch.zeros(hw_by_cam[ci][0], hw_by_cam[ci][1], 1, dtype=torch.uint8)
        for ci in range(n_cameras)
    }
    for fi in range(n_frames):
        for ci in range(n_cameras):
            if gt_images[fi][ci] is None:
                gt_images[fi][ci] = zero_img_by_cam[ci]
                gt_alphas[fi][ci] = zero_alpha_by_cam[ci]
    return gt_images, gt_alphas  # type: ignore[return-value]


def _build_test_set(n_cameras: int, test_camera_idx: int) -> List[int]:
    idx = int(test_camera_idx)
    if not (0 <= idx < n_cameras):
        raise ValueError(
            f"--test-camera-idx={idx} out of range for {n_cameras} cameras"
        )
    return [idx]


def main(args: TrigsSelfCapArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(TrigsSelfCapArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(
        level=logging.DEBUG if args.config.diagnostics else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    snap_dir = apply_default_snapshot_dir(args.config, output_path)
    if snap_dir is not None:
        log.info(
            "Viewer snapshots every %d steps -> %s (view: viewer.py --watch %s)",
            args.config.viewer_snapshot_every, snap_dir, snap_dir,
        )

    # -- Parse calibration + build dataset ----------------------------------
    parser = SelfCapParser(
        data_dir=args.data_dir,
        factor=args.data_factor,
        apply_sync=not args.no_sync,
        pcd_source=args.sparse_subdir,
    )
    # Pin trainer fps to the dataset's fps so σ_t autoinit is consistent.
    args.config.fps = parser.fps

    test_set = _build_test_set(len(parser.camera_names), args.test_camera_idx)
    if len(test_set) >= len(parser.camera_names):
        raise ValueError(
            "Holding out one test camera leaves no training cameras. "
            "Need at least 2 cameras in the dataset."
        )

    dataset = SelfCapDataset(parser, split="train", test_set=test_set)
    n_frames = parser.total_frames
    log.info(
        f"SelfCap: {parser.camtoworlds.shape[0]} cameras, "
        f"n_frames={n_frames}, train cameras={len(dataset.camera_indices)}, "
        f"held-out test camera={test_set[0]}, dataset len={len(dataset)}"
    )

    # -- Build canonical set from per-frame pcds ----------------------------
    init_target_count = args.init_target_count
    if args.defer_init_target_to_prune:
        if args.init_target_count is None:
            raise ValueError(
                "--defer-init-target-to-prune requires --init-target-count"
            )
        init_target_count = None
        args.config.delayed_prune_target_count = args.init_target_count
        args.config.delayed_prune_start_iter = args.defer_init_target_start_iter
        args.config.delayed_prune_every = args.defer_init_target_every
        log.info(
            "Defer target budget to training: "
            f"periodic opacity-based prune to {args.init_target_count:,} "
            f"(start={args.defer_init_target_start_iter}, "
            f"every={args.defer_init_target_every})"
        )

    params, sh_degree, scene_extent = build_canonical_from_pcd_sequence(
        data_dir=args.data_dir,
        absolute_frame_origin=parser.absolute_frame_origin,
        n_frames=n_frames,
        fps=parser.fps,
        keyframe_stride=args.config.keyframe_stride,
        target_count=init_target_count,
        fps_time_weight=args.config.fps_time_weight,
        nn_velocity_init=args.config.nn_velocity_init,
        nn_velocity_k=args.config.nn_velocity_k,
        omega_init=args.config.omega_init,
        local_twist_k=args.config.local_twist_k,
        local_twist_reg=args.config.local_twist_reg,
        anchor_radius_frac=args.config.anchor_radius_frac,
        init_sigma_t=args.config.init_sigma_t,
        dense_subdir=args.dense_subdir,
        sparse_subdir=args.sparse_subdir,
        device=args.device,
        verbose=args.config.verbose,
    )
    log.info(
        f"Canonical set: {params['means'].shape[0]:,} primitives, "
        f"scene_extent={scene_extent:.3f}"
    )

    # -- Build CameraConfig from training-split cameras ---------------------
    from trigs.core.camera import CameraConfig
    training_cam_ids = dataset.camera_indices
    c2w_all = torch.from_numpy(parser.camtoworlds).float()
    c2w = c2w_all[training_cam_ids].to(args.device)
    viewmats = torch.linalg.inv(c2w)

    first_K_by_pos: Dict[int, torch.Tensor] = {}
    first_HW_by_pos: Dict[int, tuple[int, int]] = {}
    orig_to_pos = {orig: pos for pos, orig in enumerate(training_cam_ids)}
    # dataset.pairs is camera-major; skip decoding entirely for cameras whose
    # K we've already collected (naive `for i in range(len(dataset))` would do
    # ~n_frames wasted decodes per camera before reaching the next one).
    for i, (cam_orig, _fo) in enumerate(dataset.pairs):
        pos = orig_to_pos.get(int(cam_orig))
        if pos is None or pos in first_K_by_pos:
            continue
        s = dataset[i]
        if s is None:
            continue
        first_K_by_pos[pos] = torch.as_tensor(s["K"]).float()
        first_HW_by_pos[pos] = (int(s["image"].shape[0]), int(s["image"].shape[1]))
        if len(first_K_by_pos) == len(training_cam_ids):
            break
    if not first_K_by_pos:
        raise RuntimeError("No intrinsics loadable from SelfCapDataset")
    if len(first_K_by_pos) != len(training_cam_ids):
        missing = sorted(set(range(len(training_cam_ids))) - set(first_K_by_pos.keys()))
        raise RuntimeError(
            "Failed to collect intrinsics/size for all training cameras. "
            f"Missing camera positions: {missing}"
        )

    hw_by_pos = [first_HW_by_pos[p] for p in range(len(training_cam_ids))]

    Ks = torch.stack(
        [first_K_by_pos[p] for p in range(len(training_cam_ids))], dim=0,
    ).to(args.device)
    widths = [int(hw_by_pos[p][1]) for p in range(len(training_cam_ids))]
    heights = [int(hw_by_pos[p][0]) for p in range(len(training_cam_ids))]
    cameras = CameraConfig.from_matrices(
        viewmats=viewmats,
        Ks=Ks,
        widths=widths,
        heights=heights,
    )

    # -- GT: eager (preload) vs lazy (per-step decode) ---------------------
    if args.preload_gt:
        log.info("Preloading GT images into RAM (may take a while) …")
        gt_images, gt_alphas = _load_gt_from_dataset(
            dataset, n_frames, n_cameras=len(training_cam_ids),
        )
    else:
        log.info("Using lazy GT loader (jpgs decoded on each training step)")
        gt_images, gt_alphas = _lazy_gt_from_dataset(
            dataset, n_cameras=len(training_cam_ids),
        )
    # Timestamps are pure metadata (frame_offset/fps − sync[cam]) — preload
    # eagerly so the training loop never touches the JPG decode path just
    # to read a time.
    gt_times = _build_gt_times(
        dataset, n_frames, n_cameras=len(training_cam_ids),
    )

    # -- Train --------------------------------------------------------------
    tb_writer = None
    if args.config.tb_log_every > 0 or args.config.tb_stats_every > 0:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir = output_path / "tb"
        tb_dir.mkdir(parents=True, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=str(tb_dir))

    trainer = TrigsTrainer(device=args.device)
    try:
        result = trainer.train_from_params(
            params=params,
            sh_degree=sh_degree,
            scene_extent=parser.scene_scale,
            cameras=cameras,
            n_frames=n_frames,
            gt_images_per_frame=gt_images,
            gt_alphas_per_frame=gt_alphas,
            gt_times_per_frame=gt_times,
            config=args.config,
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
            "selfcap_eval_meta": {
                "test_camera_idx": int(test_set[0]),
                "test_set": test_set,
                "data_factor": int(args.data_factor),
                "no_sync": bool(args.no_sync),
                "sparse_subdir": args.sparse_subdir,
                "fps": float(parser.fps),
                "n_frames": int(n_frames),
                "absolute_frame_origin": int(parser.absolute_frame_origin),
            },
        },
        ckpt_path,
    )
    log.info(f"Saved checkpoint -> {ckpt_path}")

    # -- Decode every frame to PLY ----------------------------------------
    if args.save_plys:
        ply_out_dir = output_path / "frames_ply"
        ply_out_dir.mkdir(parents=True, exist_ok=True)

        for fi in tqdm(range(n_frames), desc="Decode + save PLYs"):
            t = float(fi) / max(args.config.fps, 1e-6)
            with torch.no_grad():
                means_t, quats_t, gamma = trainer._deform(result.params, t, args.config)
                base_opa = torch.sigmoid(result.params["opacities"])
                if base_opa.dim() == 1:
                    base_opa = base_opa.unsqueeze(-1)
                eff_opa = (base_opa * gamma).squeeze(-1)
                active_mask = eff_opa > 1e-4
                if not bool(active_mask.any()):
                    # Keep one primitive to avoid writing an empty frame file.
                    active_mask[torch.argmax(eff_opa)] = True
                
                # gsply convention: opacities stored in logit space
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
                device=args.device,
            )
            
            gs_t.save(str(ply_out_dir / f"frame_{fi:06d}.ply"), compressed=False)
            del gs_t
            
        log.info(f"Wrote {n_frames} deformed-frame PLYs to {ply_out_dir}")

if __name__ == "__main__":
    main(tyro.cli(TrigsSelfCapArgs))
