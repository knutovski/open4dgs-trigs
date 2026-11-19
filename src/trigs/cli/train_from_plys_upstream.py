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
from trigs.core.trainer_ply_upstream import (
    TrigsTrainer,
    TrigsTrainingConfig,
)
from trigs.io.ply import load_ply

log = logging.getLogger(__name__)

UPSTREAM_COMMIT = "bb51ea6482513bb3075a11e959a2beec46f1702b"


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

    device: str = "auto"
    """Target device."""

    render_debug: bool = True
    """Render deformed frames for every keyframe after training (diagnostic)."""

    save_plys: bool = True
    """Decode and save every trained frame after training."""


def _save_checkpoint(
    result,
    output_dir: Path,
    *,
    fps: float,
    n_frames: int,
) -> None:
    """Persist parameters + temporal/motion tensors for later decoding."""
    ckpt_path = output_dir / "trigs_ckpt.pt"
    payload = {
        "params": {k: v.detach().cpu() for k, v in result.params.items()},
        "temporal_params": result.temporal_params,
        "motion_params": result.motion_params,
        "metrics": result.metrics,
        "fps": float(fps),
        "n_frames": int(n_frames),
    }
    torch.save(payload, ckpt_path)
    log.info(f"Saved checkpoint -> {ckpt_path}")


def main(args: TrigsPipelineArgs | None = None) -> None:
    if args is None:
        args = tyro.cli(TrigsPipelineArgs)
    args.device = str(resolve_device(args.device))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ply_files = sorted(input_path.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No PLY files found in {input_path}")
    log.info(f"Found {len(ply_files)} PLY files in {input_path}")
    log.info(
        "Pinned TriGSVanilla PLY trainer: "
        "OpsiClear-4DGS/TriGSVanilla@%s",
        UPSTREAM_COMMIT,
    )

    # -- Cameras from frame 0 ----------------------------------------------
    service = TrigsTrainer(device=args.device)
    runtime_device = str(service.device)
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
        tb_writer.add_text("control/upstream_commit", UPSTREAM_COMMIT, 0)
        tb_writer.add_scalar("control/upstream_ply_trainer", 1.0, 0)
        tb_writer.flush()
        log.info(f"Tensorboard log dir: {tb_dir}")

    # -- Run training -------------------------------------------------
    try:
        result = service.train(
            ply_files=ply_files,
            cameras=cameras,
            config=args.config,
            transform_config=args.transform,
            writer=tb_writer,
        )
    finally:
        if tb_writer is not None:
            tb_writer.close()
    _save_checkpoint(
        result,
        output_path,
        fps=args.config.fps,
        n_frames=len(ply_files),
    )

    if not args.save_plys:
        log.info("Skipping per-frame PLY decoding (--no-save-plys)")
        return

    # -- Decode every frame to PLY, optionally render debug views ---------
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


if __name__ == "__main__":
    main(tyro.cli(TrigsPipelineArgs))
