"""Evaluate COLMAP-mode TRiGS runs through the trainer's own render path.

Loads each run's viewer snapshot (or final checkpoint) with the config that
run was trained with, rebuilds cameras, GT images and capture times with
:func:`trigs.cli.train_from_colmap.build_colmap_scene`, and:

1. Re-computes ``psnr/eval_full`` on the trainer's fixed eval pairs and prints
   it next to the value TensorBoard logged at the same step. The two must agree
   before any other number from this tool is trusted.
2. Optionally renders fixed crops (camera folder + box + source frames) for
   every run next to the photo, scores them with PSNR / SSIM / LPIPS and the
   rendered opacity, and writes side-by-side PNGs to judge by eye.

Usage (from the dataset folder, same frame range as training)::

    python -m trigs.cli.eval_colmap --data-dir . --start-frame 1 --end-frame 838 \
        --runs results-31-opdiag results-32-reg0001 --step 10000 \
        --out-dir eval-r31-r32 \
        --crop-cams 32 43 --crop-boxes 0.25,0.05,0.75,0.95 0.25,0.05,0.75,0.95 \
        --crop-frames 186 192 198
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from trigs.compat.device import resolve_device, empty_cache  # macOS port
import tyro
from PIL import Image
from torch import nn

from trigs.cli.train_from_colmap import (
    TrigsColmapArgs,
    _raw_warps_by_pos,
    _warp_raw_like_dataset,
    build_colmap_scene,
)
from trigs.core.losses import compute_psnr, compute_ssim_metric
from trigs.core.trainer import (
    TrigsTrainer,
    TrigsTrainingConfig,
    _as_float_rgb01,
    _make_psnr_eval_pairs,
)
from trigs.core.viewer_snapshot import CAMERA_KEYS, list_snapshots, load_snapshot

log = logging.getLogger(__name__)


@dataclass
class EvalColmapArgs:
    """Arguments for the COLMAP-mode evaluator."""

    data_dir: Path
    """Dataset root (``sparse/`` + ``images/``), as given to training."""

    runs: tuple[Path, ...]
    """Training output directories to evaluate (each has ``viewer_snapshots/``)."""

    step: int
    """Snapshot step to load from every run. ``-1`` loads ``trigs_ckpt.pt``."""

    out_dir: Path
    """Where ``eval.json`` and the crop PNGs go."""

    start_frame: int | None = None
    """Same as training. None = the first run's run_args.json (written by trigs-train-colmap)."""

    end_frame: int | None = None
    """Same as training (inclusive). None = the first run's run_args.json."""

    data_factor: int = 1
    """Same as training."""

    test_every: int = 8
    """Same as training."""

    reference_colmap: Path | None = None
    """Same as training."""

    rs_rotation_json: Path | None = None
    """Same as training (only needed when the run used rolling shutter)."""

    rs_flip_sign: bool = False
    """Same as training."""

    device: str = "auto"
    """Torch device."""

    crop_cams: tuple[str, ...] = ()
    """Camera folder names to crop, e.g. ``32 43``."""

    crop_boxes: tuple[str, ...] = ()
    """One ``x0,y0,x1,y1`` box per crop camera, as fractions of the image."""

    crop_frames: tuple[int, ...] = ()
    """Source frame numbers (image file numbers) to crop."""

    lpips_net: str = "alex"
    """LPIPS backbone."""

    shutter: str = "captured"
    """'captured': render every evaluation camera as it captured its photo (the run's own timing file and
    rolling-shutter model; the only mode comparable to photos). 'instant': global-shutter frame at each camera's
    midpoint capture time (readout term off)."""

    blur_frames: tuple[int, ...] = ()
    """Source frame numbers at which to measure blur, over every training
    camera. Empty skips the blur measurement. Use the same frames for runs
    trained on different frame ranges so they are measured on the same views."""


def _config_from_snapshot(saved: dict[str, Any]) -> TrigsTrainingConfig:
    """The run's own training config; fields it predates keep their defaults."""
    names = {f.name for f in dataclasses.fields(TrigsTrainingConfig)}
    return dataclasses.replace(
        TrigsTrainingConfig(), **{k: v for k, v in saved.items() if k in names}
    )


def _load_run(run: Path, step: int) -> tuple[dict[str, torch.Tensor], dict[str, Any], str]:
    """``(params, payload, camera_param_source)`` for one run at ``step``."""
    snaps = list_snapshots(run / "viewer_snapshots")
    if not snaps:
        raise FileNotFoundError(f"{run}: no viewer snapshots (needed for the run's config)")
    ckpt_path = run / "trigs_ckpt.pt"
    if step < 0:
        payload = load_snapshot(snaps[-1][1])
        ckpt = load_snapshot(ckpt_path)
        params = dict(ckpt["params"])
        payload = {**payload, "step": int(payload["n_steps"])}
        return params, payload, "final checkpoint"

    path = run / "viewer_snapshots" / f"snap_s{int(step):07d}.pt"
    payload = load_snapshot(path)
    params = dict(payload["params"])
    camera_params = dict(payload.get("camera_params") or {})
    source = "snapshot"
    if not camera_params and ckpt_path.exists():
        # Older snapshots did not keep per-camera values; the final
        # checkpoint's are the closest available (offsets settle early).
        ckpt = load_snapshot(ckpt_path)
        camera_params = {k: ckpt["params"][k] for k in CAMERA_KEYS if k in ckpt["params"]}
        source = "final checkpoint (snapshot predates camera_params)"
    elif not camera_params:
        source = "none saved (rendered with zero offsets)"
    params.update(camera_params)
    return params, payload, source


def _tb_scalar(run: Path, tag: str, step: int) -> tuple[int, float] | None:
    """The logged ``tag`` at the eval step nearest ``step`` (``None`` if absent)."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return None
    best: tuple[int, float] | None = None
    for ev in sorted((run / "tb").glob("events.out.tfevents.*")):
        acc = EventAccumulator(str(ev), size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        for s in acc.Scalars(tag):
            if best is None or abs(s.step - step) < abs(best[0] - step):
                best = (int(s.step), float(s.value))
    return best


class _Lpips:
    def __init__(self, device: torch.device, net: str):
        try:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        except ImportError:
            log.warning("torchmetrics LPIPS unavailable; LPIPS reported as NaN")
            self.metric = None
            return
        self.metric = LearnedPerceptualImagePatchSimilarity(
            net_type=net, normalize=True,
        ).to(device).eval()

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, gt: torch.Tensor) -> float:
        if self.metric is None:
            return float("nan")
        to = lambda x: x.permute(2, 0, 1).unsqueeze(0).contiguous()
        return float(self.metric(to(pred), to(gt)).item())


def _box_px(spec: str, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(v) for v in spec.split(","))
    return round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)


def _to_u8(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()


# -- Blur measurement ----------------------------------------------------------
# For each view, the photo is blurred by a Gaussian of each sigma in
# _BLUR_SIGMAS and compared with the render; the sigma whose blurred photo
# matches the render best is the render's blur in pixels. A shifted render
# also prefers a blurred photo, so every sigma is also tried against the photo
# moved by up to 2 px (one shift per view and region): if allowing the shift
# removes most of the blur, misregistration rather than blur is the cause.
_BLUR_SIGMAS = tuple(0.25 * i for i in range(25))  # 0 .. 6 px
_SHIFT_MAX = 2
_SHIFTS = tuple(
    (dy, dx)
    for dy in range(-_SHIFT_MAX, _SHIFT_MAX + 1)
    for dx in range(-_SHIFT_MAX, _SHIFT_MAX + 1)
)
_NO_SHIFT = _SHIFTS.index((0, 0))
_REGIONS = ("static", "moving")


def _gauss(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a [C, H, W] image (reflect padding)."""
    if sigma <= 0:
        return img
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    im = img.unsqueeze(1)
    im = torch.nn.functional.conv2d(
        torch.nn.functional.pad(im, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1),
    )
    im = torch.nn.functional.conv2d(
        torch.nn.functional.pad(im, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1),
    )
    return im.squeeze(1)


def _blur_regions(photo: torch.Tensor, before: torch.Tensor, after: torch.Tensor) -> dict:
    """Textured static / moving pixel masks ([H, W] bool) of a [3, H, W] photo.

    Moving: the photos 2 frames before and after differ (dilated 9 px).
    Static: more than 31 px from any moving pixel. Only textured pixels (top
    quarter of the photo's gradient within each region) count, because blur
    cannot be measured on flat areas.
    """
    pool = torch.nn.functional.max_pool2d
    diff = (after - before).abs().mean(0, keepdim=True) > 0.06
    moving = pool(diff[None].float(), 9, 1, 4)[0, 0] > 0
    near = pool(diff[None].float(), 63, 1, 31)[0, 0] > 0
    gray = _gauss(photo.mean(0, keepdim=True), 1.0)[0]
    gy, gx = torch.gradient(gray)
    mag = torch.sqrt(gx ** 2 + gy ** 2)
    out = {}
    for name, region in (("static", ~near), ("moving", moving)):
        if int(region.sum()) < 500:
            out[name] = torch.zeros_like(region)
            continue
        thr = torch.quantile(mag[region][:: max(1, int(region.sum()) // 1_000_000)], 0.75)
        out[name] = region & (mag > thr)
    return out


def _blur_errors(pred: torch.Tensor, photo: torch.Tensor, masks: dict) -> dict:
    """Per region: squared-error sums [n_sigmas, n_shifts] and pixel count."""
    m = _SHIFT_MAX
    h, w = pred.shape[1:]
    pc = pred[:, m:h - m, m:w - m]
    names = list(masks)
    mf = torch.stack([masks[n][m:h - m, m:w - m] for n in names]).to(pred.dtype)
    err = torch.zeros(len(names), len(_BLUR_SIGMAS), len(_SHIFTS), device=pred.device,
                      dtype=torch.float64)
    for si, s in enumerate(_BLUR_SIGMAS):
        pb = _gauss(photo, s)
        for ki, (dy, dx) in enumerate(_SHIFTS):
            d2 = (pc - pb[:, m + dy:h - m + dy, m + dx:w - m + dx]).pow(2).sum(0)
            err[:, si, ki] = (d2[None] * mf).sum(dim=(1, 2)).double()
    err = err.cpu()
    counts = (mf.sum(dim=(1, 2)) * 3).round().long().tolist()
    return {n: [err[i], int(counts[i])] for i, n in enumerate(names)}


def _sigma_at_min(curve: np.ndarray) -> float:
    """Arg-min of an error-vs-sigma curve, refined by a parabola through the
    lowest point and its neighbours."""
    i = int(np.argmin(curve))
    if 0 < i < len(curve) - 1:
        a, b, c = curve[i - 1], curve[i], curve[i + 1]
        den = a - 2 * b + c
        if den > 0:
            return float(_BLUR_SIGMAS[i] + 0.25 * 0.5 * (a - c) / den)
    return float(_BLUR_SIGMAS[i])


def _blur_summary(per_view: list[dict]) -> dict:
    """Pool per-view errors into the blur sigma per region, without and with
    the per-view best shift, plus how much of the error the blur explains."""
    out = {}
    for name in _REGIONS:
        views = [v[name] for v in per_view if v[name][1] > 0]
        if not views:
            continue
        n = sum(c for _, c in views)
        fixed = sum(e[:, _NO_SHIFT] for e, _ in views).numpy() / n
        shifted = sum(e.min(dim=1).values for e, _ in views).numpy() / n
        sig_shift = _sigma_at_min(shifted)
        si = min(range(len(_BLUR_SIGMAS)), key=lambda i: abs(_BLUR_SIGMAS[i] - sig_shift))
        shifts = [_SHIFTS[int(e[si].argmin())] for e, _ in views]
        out[name] = {
            "sigma_px": _sigma_at_min(fixed),
            "sigma_px_after_shift": sig_shift,
            "mean_shift_px": float(np.mean([np.hypot(dy, dx) for dy, dx in shifts])),
            "error_explained_by_blur": float(1.0 - fixed.min() / max(fixed[0], 1e-12)),
            "views": len(views),
        }
    return out


def _fmt_blur(summary: dict) -> str:
    parts = []
    for name in _REGIONS:
        if name in summary:
            r = summary[name]
            parts.append(
                f"{name} σ {r['sigma_px']:.2f} px (σ {r['sigma_px_after_shift']:.2f} with "
                f"≤{_SHIFT_MAX} px shift, mean shift {r['mean_shift_px']:.2f} px; blur explains "
                f"{r['error_explained_by_blur']:.0%} of the error, {r['views']} views)"
            )
    return " | ".join(parts)


def main(args: EvalColmapArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(EvalColmapArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(args.crop_boxes) != len(args.crop_cams):
        raise SystemExit("--crop-boxes needs exactly one box per --crop-cams entry")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_frame is None or args.end_frame is None:
        run_args_path = Path(args.runs[0]) / "run_args.json"
        if not run_args_path.exists():
            raise SystemExit(f"--start-frame/--end-frame are required: {run_args_path} does not exist "
                             "(runs trained before run_args.json was written)")
        saved = json.loads(run_args_path.read_text())
        if args.start_frame is None:
            args.start_frame = int(saved["start_frame"])
        if args.end_frame is None:
            args.end_frame = int(saved["end_frame"])
        log.info("Frame range %d..%d from %s", args.start_frame, args.end_frame, run_args_path)
    loaded = [(Path(r), *_load_run(Path(r), args.step)) for r in args.runs]
    configs = [_config_from_snapshot(p.get("config", {})) for _, _, p, _ in loaded]
    if args.shutter not in ("captured", "instant"):
        raise SystemExit("--shutter must be 'captured' or 'instant'")
    for (run, _, _, _), cfg in zip(loaded, configs):
        if cfg.fps != configs[0].fps:
            raise SystemExit(f"{run}: fps differs from {loaded[0][0]}")
    # Runs may differ in capture timing (sync.json vs a --timing-json file) and rolling shutter: each run's
    # cameras and capture times are rebuilt from its own saved config below; photos/masks are shared.

    def timing_path(cfg) -> Path | None:
        src = str(getattr(cfg, "capture_timing_json", "") or "")
        if not src:
            return None
        path, _, digest = src.partition("#sha256=")
        import hashlib
        if not Path(path).exists():
            raise SystemExit(f"timing file {path} recorded in the run's config is missing")
        if digest and hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"timing file {path} changed since training (sha256 mismatch)")
        return Path(path)

    scene_args = TrigsColmapArgs(
        data_dir=args.data_dir, output_dir=out_dir,
        start_frame=args.start_frame, end_frame=args.end_frame,
        data_factor=args.data_factor, test_every=args.test_every,
        reference_colmap=args.reference_colmap, rs_rotation_json=args.rs_rotation_json,
        rs_flip_sign=args.rs_flip_sign, config=configs[0], device=args.device,
    )
    scene_args.timing_json = timing_path(configs[0])
    scene = build_colmap_scene(scene_args)
    n_frames, cameras, n_views = scene.n_frames, scene.cameras, len(scene.cameras)
    timing_scenes = {(str(scene_args.timing_json), configs[0].rs_enabled()): scene}

    def run_timing(cfg):
        """(cameras, gt_times) exactly as this run was supervised."""
        key = (str(timing_path(cfg)), cfg.rs_enabled())
        if key not in timing_scenes:
            sa = dataclasses.replace(scene_args, config=cfg, timing_json=timing_path(cfg))
            timing_scenes[key] = build_colmap_scene(sa)
        sc = timing_scenes[key]
        return sc.cameras, sc.gt_times
    device = resolve_device(args.device)
    lpips = _Lpips(device, args.lpips_net) if args.crop_cams else None

    crops: list[tuple[str, int, int, tuple[int, int, int, int]]] = []
    for folder, box in zip(args.crop_cams, args.crop_boxes):
        if folder not in scene.cam_folders_by_pos:
            raise SystemExit(f"crop camera {folder!r} is not a training camera")
        cp = scene.cam_folders_by_pos.index(folder)
        px = _box_px(box, cameras.width_for(cp), cameras.height_for(cp))
        for frame in args.crop_frames:
            fo = int(frame) - int(args.start_frame)
            if not 0 <= fo < n_frames:
                raise SystemExit(f"crop frame {frame} is outside the trained range")
            crops.append((folder, int(frame), cp, px))

    def photo_chw(fo: int, cp: int) -> torch.Tensor:
        return _as_float_rgb01(scene.gt_images[fo][cp].to(device)).permute(2, 0, 1).contiguous()

    blur_views: list[tuple[int, int]] = []
    holdout0 = {int(v) for v in getattr(configs[0], "holdout_views", ())}
    for frame in args.blur_frames:
        fo = int(frame) - int(args.start_frame)
        if not 0 <= fo < n_frames:
            raise SystemExit(f"blur frame {frame} is outside the trained range")
        blur_views += [(fo, cp) for cp in range(n_views) if cp not in holdout0]
    raw_warps = _raw_warps_by_pos(scene.parser, scene.training_cam_ids) if blur_views else []

    def neighbour_chw(fo: int, cp: int) -> torch.Tensor:
        """Photo ``fo`` frames from the start, also outside the trained range
        (read from disk with the dataset's undistortion), for the moving-pixel
        mask and the noise floor of short or single-frame runs."""
        if 0 <= fo < n_frames:
            return photo_chw(fo, cp)
        orig = scene.training_cam_ids[cp]
        path = scene.dataset._get_image_path(scene.parser.campaths[orig], int(args.start_frame) + fo)
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"blur measurement needs {path} (2 frames either side)")
        arr = _warp_raw_like_dataset(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), raw_warps[cp])
        w, h = cameras.width_for(cp), cameras.height_for(cp)
        if arr.shape[:2] != (h, w):
            arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_AREA)
        return _as_float_rgb01(torch.from_numpy(arr).to(device)).permute(2, 0, 1).contiguous()

    def view_regions(fo: int, cp: int) -> tuple[torch.Tensor, dict]:
        photo = photo_chw(fo, cp)
        return photo, _blur_regions(photo, neighbour_chw(fo - 2, cp), neighbour_chw(fo + 2, cp))

    report: dict[str, Any] = {"step": args.step, "runs": {}}
    if blur_views:
        # Floor: the next photo of the same camera against this one, on static
        # texture. Sensor noise and compression alone make a blurred photo
        # match a different capture a little better; renders cannot beat this.
        floor_views = []
        with torch.no_grad():
            for fo, cp in blur_views:
                photo, masks = view_regions(fo, cp)
                floor_views.append(_blur_errors(neighbour_chw(fo + 1, cp), photo, {"static": masks["static"]}))
        for v in floor_views:
            v["moving"] = [torch.zeros(1), 0]
        floor = _blur_summary(floor_views)
        report["blur_floor_next_photo"] = floor
        print(f"Blur floor (next photo of the same camera vs this one): {_fmt_blur(floor)}", flush=True)

    tiles: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    for (run, raw, payload, cam_src), config in zip(loaded, configs):
        step = int(payload["step"])
        params = nn.ParameterDict({
            k: nn.Parameter(
                (v.detach().float() if v.is_floating_point() else v.detach()).to(device),
                requires_grad=False,
            )
            for k, v in raw.items()
        })
        sh_degree = int(payload.get("sh_degree_active", payload.get("sh_degree", 3)))
        trainer = TrigsTrainer(device=str(device))
        trainer._sequence_t_max = float(max(n_frames - 1, 0)) / max(float(config.fps), 1e-6)
        eval_config = config
        if float(getattr(config, "temporal_chunk_len_s", 0.0)) > 0.0:
            # What the trainer's chunked eval uses on the assembled model.
            eval_config = dataclasses.replace(config, temporal_hierarchy_enabled=False)
        if args.shutter == "instant":
            eval_config = dataclasses.replace(eval_config, rs_readout_ms=0.0)
        run_cameras, run_gt_times = run_timing(config)

        holdout = {int(v) for v in getattr(config, "holdout_views", ())}
        pairs = _make_psnr_eval_pairs(
            n_frames=n_frames, n_views=n_views,
            num_samples=config.psnr_eval_num_samples, seed=config.psnr_eval_seed,
            views=[v for v in range(n_views) if v not in holdout],
        )
        ev = trainer._evaluate_psnr_subset(
            params=params, cameras=run_cameras, eval_pairs=pairs, sh_degree=sh_degree,
            config=eval_config, gt_images_per_frame=scene.gt_images,
            gt_alphas_per_frame=scene.gt_alphas, gt_times_per_frame=run_gt_times,
            step=step,
        )
        tb = _tb_scalar(run, "psnr/eval_full", step)
        entry: dict[str, Any] = {
            "snapshot_step": step, "camera_params": cam_src, "shutter": args.shutter,
            "capture_timing": str(getattr(config, "capture_timing_json", "") or "sync.json"),
            "rs_readout_ms": float(eval_config.rs_readout_ms), "rs_axis": str(getattr(config, "rs_axis", "column")),
            "eval_full": ev.get("psnr/eval_full"), "eval_pairs": len(pairs),
            "tb_eval_full": None if tb is None else {"step": tb[0], "value": tb[1]},
        }
        line = (f"{run.name} @ {step}: eval_full {ev.get('psnr/eval_full', float('nan')):.3f} dB "
                f"on {len(pairs)} pairs")
        if tb is not None:
            line += f" | TensorBoard @ {tb[0]}: {tb[1]:.3f} dB (diff {ev['psnr/eval_full'] - tb[1]:+.3f})"
        print(line + f" | per-camera values: {cam_src}", flush=True)

        crop_rows = []
        with torch.no_grad():
            for folder, frame, cp, (x0, y0, x1, y1) in crops:
                fo = frame - int(args.start_frame)
                t = trainer._eval_render_time(params, fo, cp, run_gt_times, config)
                colors, alphas, _, _ = trainer._render(
                    params, run_cameras, cp, t, sh_degree, eval_config, step=step,
                )
                pred = colors[0].clamp(0, 1)
                alpha = alphas[0].reshape(pred.shape[0], pred.shape[1])
                gt = _as_float_rgb01(scene.gt_images[fo][cp].to(device))
                pc, gc, ac = pred[y0:y1, x0:x1], gt[y0:y1, x0:x1], alpha[y0:y1, x0:x1]
                crop_rows.append({
                    "cam": folder, "frame": frame,
                    "psnr": compute_psnr(pc, gc), "ssim": compute_ssim_metric(pc, gc),
                    "lpips": lpips(pc, gc) if lpips is not None else float("nan"),
                    "alpha_below_0.9": float((ac < 0.9).float().mean()),
                    "alpha_mean": float(ac.mean()),
                })
                key = (folder, frame)
                if key not in tiles:
                    tiles[key] = [("photo", _to_u8(gc))]
                tiles[key].append((run.name, _to_u8(pc)))
        if crop_rows:
            mean = {k: float(np.nanmean([r[k] for r in crop_rows]))
                    for k in ("psnr", "ssim", "lpips", "alpha_below_0.9", "alpha_mean")}
            entry["crops"] = crop_rows
            entry["crops_mean"] = mean
            print(f"    crops ({len(crop_rows)}): PSNR {mean['psnr']:.2f} dB | SSIM {mean['ssim']:.4f} | "
                  f"LPIPS {mean['lpips']:.4f} (lower is better) | alpha<0.9 {mean['alpha_below_0.9']:.1%} | "
                  f"mean alpha {mean['alpha_mean']:.3f}", flush=True)
        if blur_views:
            per_view = []
            with torch.no_grad():
                for fo, cp in blur_views:
                    t = trainer._eval_render_time(params, fo, cp, run_gt_times, config)
                    colors, _, _, _ = trainer._render(
                        params, run_cameras, cp, t, sh_degree, eval_config, step=step,
                    )
                    pred = colors[0].clamp(0, 1).permute(2, 0, 1).contiguous()
                    photo, masks = view_regions(fo, cp)
                    per_view.append(_blur_errors(pred, photo, masks))
            entry["blur"] = _blur_summary(per_view)
            print(f"    blur: {_fmt_blur(entry['blur'])}", flush=True)
        report["runs"][str(run)] = entry
        del params, trainer
        empty_cache(device)

    for (folder, frame), row in tiles.items():
        strip = np.concatenate([img for _, img in row], axis=1)
        Image.fromarray(strip).save(out_dir / f"crop_cam{folder}_f{frame:06d}.png")
    if tiles:
        order = " | ".join(name for name, _ in next(iter(tiles.values())))
        print(f"Crop strips (left to right: {order}) -> {out_dir}", flush=True)
    (out_dir / "eval.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_dir / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
