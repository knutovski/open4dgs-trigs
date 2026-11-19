"""CLI evaluator for TRiGS SelfCap checkpoints.

Evaluates a checkpoint on held-out test views and reports average PSNR, SSIM,
and LPIPS. Also renders a constant-view video across all times.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
import tyro
from tqdm import tqdm

from trigs.core.camera import CameraConfig
from trigs.core.losses import compute_psnr, compute_ssim_metric
from trigs.core.trainer import TrigsTrainer, TrigsTrainingConfig
from trigs.io.selfcap import SelfCapDataset, SelfCapParser

log = logging.getLogger(__name__)


@dataclass
class EvalArgs:
    """Arguments for TRiGS SelfCap evaluator."""

    ckpt_path: Path
    """Path to ``trigs_ckpt.pt``."""

    data_dir: Path
    """Preprocessed SelfCap directory."""

    output_dir: Path
    """Directory to write evaluation outputs."""

    device: str = "auto"
    """Torch device for rendering and metric computation."""

    data_factor: int | None = None
    """Override parser downsample factor. None = use checkpoint metadata or 1."""

    no_sync: bool | None = None
    """Override sync usage. None = use checkpoint metadata or False."""

    sparse_subdir: str | None = None
    """Override parser PLY source. None = use checkpoint metadata or ``pcds``."""

    test_camera_idx: int | None = None
    """Held-out test camera index. None = use checkpoint metadata."""

    max_eval_samples: int | None = None
    """Optional cap on number of test samples for quick checks."""

    video_name: str = "constant_view.mp4"
    """Output video filename."""

    lpips_net: Literal["alex", "vgg", "squeeze"] = "alex"
    """Backbone for LPIPS metric."""


class _LPIPSMetric:
    """Thin wrapper around torchmetrics LPIPS with [0, 1] HWC inputs."""

    def __init__(self, device: torch.device, net_type: str = "alex"):
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS requires torchmetrics.image.lpip. Install torchmetrics extras "
                "or update torchmetrics to a version that includes LPIPS."
            ) from exc

        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net_type,
            normalize=True,  # accepts [0, 1], internally maps to [-1, 1]
        ).to(device)
        self.metric.eval()

    @torch.no_grad()
    def __call__(self, pred_hwc: torch.Tensor, target_hwc: torch.Tensor) -> float:
        pred = pred_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        target = target_hwc.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
        return float(self.metric(pred, target).item())


def _resolve_test_camera_idx(
    n_cameras: int,
    test_camera_idx: int | None,
    ckpt_meta: Dict[str, Any],
) -> int:
    if test_camera_idx is not None:
        idx = int(test_camera_idx)
    elif "test_camera_idx" in ckpt_meta:
        idx = int(ckpt_meta["test_camera_idx"])
    else:
        saved = ckpt_meta.get("test_set")
        if isinstance(saved, list) and len(saved) > 0:
            idx = int(saved[0])
        else:
            raise ValueError(
                "Could not infer held-out test camera from checkpoint metadata. "
                "Pass --test-camera-idx explicitly for this checkpoint."
            )

    if not (0 <= idx < n_cameras):
        raise ValueError(
            f"Resolved test camera idx {idx} out of range for {n_cameras} cameras."
        )
    return idx


def _infer_sh_degree(params: nn.ParameterDict) -> int:
    if "shN" not in params or params["shN"] is None:
        return 0
    k = int(params["shN"].shape[1])
    return int(round((k + 1) ** 0.5)) - 1


def _load_checkpoint(ckpt_path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    if "params" not in ckpt:
        raise KeyError(f"Checkpoint missing 'params': {ckpt_path}")
    return ckpt


def _load_checkpoint_params(ckpt: Dict[str, Any], device: torch.device) -> nn.ParameterDict:
    params = nn.ParameterDict()
    for k, v in ckpt["params"].items():
        params[k] = nn.Parameter(v.to(device), requires_grad=False)
    return params


def _build_split_cameras(
    parser: Any,
    split_camera_indices: List[int],
    device: torch.device,
) -> CameraConfig:
    c2w_all = torch.from_numpy(parser.camtoworlds).float()
    c2w = c2w_all[split_camera_indices].to(device)
    viewmats = torch.linalg.inv(c2w)

    Ks = []
    widths: List[int] = []
    heights: List[int] = []
    for cam_orig in split_camera_indices:
        camera_id = parser.camera_ids[cam_orig]
        K = torch.from_numpy(parser.Ks_dict[camera_id]).float()
        w, h = parser.imsize_dict[camera_id]
        Ks.append(K)
        widths.append(int(w))
        heights.append(int(h))

    if len(widths) == 0 or len(heights) == 0:
        raise RuntimeError("No cameras found in selected split.")
    Ks_t = torch.stack(Ks, dim=0).to(device)
    return CameraConfig.from_matrices(
        viewmats=viewmats,
        Ks=Ks_t,
        widths=widths,
        heights=heights,
    )


def _sample_time_seconds(sample: Dict[str, Any], fps: float) -> float:
    if "time_sec" in sample:
        return float(sample["time_sec"])
    return float(sample["frame_offset"]) / max(float(fps), 1e-6)


def _open_ffmpeg_video_pipe(
    output_path: Path,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen[bytes]:
    """Open an ffmpeg subprocess that accepts raw RGB24 frames on stdin."""
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{max(float(fps), 1.0):.6f}",
        "-i",
        "-",
        "-an",
        # libx264 + yuv420p requires even output dims; pad odd-sized renders.
        "-vf",
        "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg binary not found. Install system ffmpeg to export evaluation videos."
        ) from exc


def main(args: EvalArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(EvalArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = resolve_device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = _load_checkpoint(Path(args.ckpt_path), device)
    ckpt_meta = ckpt.get("selfcap_eval_meta", {})
    if not isinstance(ckpt_meta, dict):
        ckpt_meta = {}

    data_factor = int(
        args.data_factor if args.data_factor is not None else ckpt_meta.get("data_factor", 1)
    )
    no_sync = bool(args.no_sync) if args.no_sync is not None else bool(ckpt_meta.get("no_sync", False))
    sparse_subdir = (
        args.sparse_subdir
        if args.sparse_subdir is not None
        else str(ckpt_meta.get("sparse_subdir", "pcds"))
    )

    # -- Dataset parser + held-out split -------------------------------------
    parser = SelfCapParser(
        data_dir=args.data_dir,
        factor=data_factor,
        apply_sync=not no_sync,
        pcd_source=sparse_subdir,
    )
    fps = float(parser.fps)
    n_frames = int(parser.total_frames)
    test_camera_idx = _resolve_test_camera_idx(
        len(parser.camera_names), args.test_camera_idx, ckpt_meta,
    )
    test_set = [test_camera_idx]
    dataset = SelfCapDataset(parser, split="test", test_set=test_set)

    if len(dataset.camera_indices) == 0:
        raise RuntimeError(
            "Resolved test split is empty. Adjust --test-every or checkpoint metadata."
        )
    log.info(
        f"Eval split: {len(dataset.camera_indices)} cameras × {n_frames} frames "
        f"= {len(dataset)} samples (test camera={test_camera_idx})"
    )

    # -- Model + rendering setup --------------------------------------------
    params = _load_checkpoint_params(ckpt, device)
    sh_degree = _infer_sh_degree(params)
    trainer = TrigsTrainer(device=str(device))
    eval_cfg = TrigsTrainingConfig(fps=fps, verbose=False)
    cameras = _build_split_cameras(parser, dataset.camera_indices, device)
    cam_orig_to_pos = {orig: pos for pos, orig in enumerate(dataset.camera_indices)}
    lpips_metric = _LPIPSMetric(device=device, net_type=args.lpips_net)

    # -- Per-frame test evaluation ------------------------------------------
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluate test samples"):
            sample = dataset[i]
            if sample is None:
                continue

            cam_orig = int(sample["camera_idx"])
            cam_pos = cam_orig_to_pos.get(cam_orig)
            if cam_pos is None:
                continue

            t_sec = _sample_time_seconds(sample, fps=fps)
            render_colors, _, _, _ = trainer._render(
                params=params,
                cameras=cameras,
                cam_idx=cam_pos,
                t=t_sec,
                sh_degree=sh_degree,
                config=eval_cfg,
            )
            pred_rgb = render_colors[0].clamp(0.0, 1.0)

            gt = sample["image"]
            if not torch.is_tensor(gt):
                gt = torch.as_tensor(gt)
            gt_rgb = gt.to(device).float().div(255.0).clamp(0.0, 1.0)

            if pred_rgb.shape != gt_rgb.shape:
                raise RuntimeError(
                    f"Pred/GT shape mismatch for sample {i}: "
                    f"{tuple(pred_rgb.shape)} vs {tuple(gt_rgb.shape)}"
                )

            psnr_vals.append(compute_psnr(pred_rgb, gt_rgb))
            ssim_vals.append(compute_ssim_metric(pred_rgb, gt_rgb))
            lpips_vals.append(lpips_metric(pred_rgb, gt_rgb))

            if args.max_eval_samples is not None and len(psnr_vals) >= args.max_eval_samples:
                break

    if len(psnr_vals) == 0:
        raise RuntimeError("No valid evaluation samples found in the test split.")

    metrics = {
        "num_samples": len(psnr_vals),
        "avg_psnr": float(np.mean(psnr_vals)),
        "avg_ssim": float(np.mean(ssim_vals)),
        "avg_lpips": float(np.mean(lpips_vals)),
        "std_psnr": float(np.std(psnr_vals)),
        "std_ssim": float(np.std(ssim_vals)),
        "std_lpips": float(np.std(lpips_vals)),
        "fps": fps,
        "test_camera_idx": int(test_camera_idx),
        "data_factor": data_factor,
        "no_sync": no_sync,
        "sparse_subdir": sparse_subdir,
    }

    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info(
        "Averages: "
        f"PSNR={metrics['avg_psnr']:.3f} dB, "
        f"SSIM={metrics['avg_ssim']:.5f}, "
        f"LPIPS={metrics['avg_lpips']:.5f}"
    )
    log.info(f"Wrote metrics -> {metrics_path}")

    # -- Constant-view temporal video (held-out test camera) ----------------
    cam_pos = 0
    cam_orig = dataset.camera_indices[cam_pos]
    cam_name = parser.camera_names[cam_orig]
    sync_off = 0.0
    if not no_sync:
        sync_offsets = getattr(parser, "sync_offsets", {}) or {}
        sync_off = float(sync_offsets.get(cam_name, 0.0))

    video_cams = CameraConfig.from_matrices(
        viewmats=cameras.viewmats[cam_pos : cam_pos + 1],
        Ks=cameras.Ks[cam_pos : cam_pos + 1],
        widths=cameras.widths[cam_pos : cam_pos + 1],
        heights=cameras.heights[cam_pos : cam_pos + 1],
    )
    video_w = cameras.width_for(cam_pos)
    video_h = cameras.height_for(cam_pos)
    video_path = out_dir / args.video_name
    ffmpeg_proc = _open_ffmpeg_video_pipe(
        output_path=video_path,
        width=video_w,
        height=video_h,
        fps=fps,
    )
    if (video_w % 2) != 0 or (video_h % 2) != 0:
        log.info(
            "Video resolution is odd (%dx%d); ffmpeg will pad to even dims for H.264.",
            video_w,
            video_h,
        )
    try:
        if ffmpeg_proc.stdin is None:
            raise RuntimeError("ffmpeg stdin pipe was not created.")
        with torch.no_grad():
            for fi in tqdm(range(n_frames), desc="Render constant-view video"):
                t_sec = float(fi) / max(fps, 1e-6) - sync_off
                render_colors, _, _, _ = trainer._render(
                    params=params,
                    cameras=video_cams,
                    cam_idx=0,
                    t=t_sec,
                    sh_degree=sh_degree,
                    config=eval_cfg,
                )
                frame = (
                    render_colors[0]
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .to(torch.uint8)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                ffmpeg_proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        # ffmpeg terminated early (e.g. invalid codec args). We'll surface the
        # full encoder stderr after wait() below.
        pass
    finally:
        stderr_msg = ""
        if ffmpeg_proc.stdin is not None:
            ffmpeg_proc.stdin.close()
        if ffmpeg_proc.stderr is not None:
            stderr_msg = ffmpeg_proc.stderr.read().decode("utf-8", errors="ignore")
        ret = ffmpeg_proc.wait()
        if ret != 0:
            raise RuntimeError(
                f"ffmpeg failed while writing video ({video_path}) with exit code {ret}. "
                f"{stderr_msg}"
            )
    log.info(f"Wrote constant-view video -> {video_path}")


if __name__ == "__main__":
    main()
