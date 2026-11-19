"""Renderer: Rendering service for Gaussian Splatting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from gsplat import rasterization

if TYPE_CHECKING:
    from gsply import GSTensor
    from .camera import CameraConfig


@dataclass
class RenderResult:
    """Result of rendering operation.

    Contains GPU tensors by default. Use to_numpy() for CPU numpy arrays
    or save() to write to disk.
    """

    images: List[Tensor]  # List of (H, W, 3) GPU tensors, values in [0, 1]
    alphas: Optional[List[Tensor]] = None  # List of (H, W, 1) GPU tensors
    depths: Optional[List[Tensor]] = None  # List of (H, W, 1) GPU tensors

    def __len__(self) -> int:
        """Return number of rendered images."""
        return len(self.images)

    def to_numpy(self) -> List[np.ndarray]:
        """Convert images to numpy arrays (H, W, 3) uint8.

        Returns:
            List of numpy arrays in uint8 format [0, 255]
        """
        result = []
        for img in self.images:
            np_img = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            result.append(np_img)
        return result

    def save(
        self,
        output_dir: Path | str,
        prefix: str = "render",
        format: str = "png",
    ) -> List[Path]:
        """Save images to disk as image files."""
        import imageio.v3 as imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        np_images = self.to_numpy()
        for i, img in enumerate(np_images):
            path = output_dir / f"{prefix}_{i:04d}.{format}"
            imageio.imwrite(path, img)
            paths.append(path)

        return paths


class Renderer:
    """Render Gaussian Splatting models to images.

    Provides methods to render GSTensor or ParameterDict to images,
    either returning GPU tensors or saving directly to disk.
    """

    def __init__(
        self,
        device: str = "auto",
        sh_degree: Optional[int] = None,
    ):
        """Initialize Renderer.

        Args:
            device: Target device for rendering
            sh_degree: Maximum SH degree to use (None = use model's degree)
        """
        self.device = resolve_device(device)
        self.sh_degree = sh_degree

    def _get_params_dict(
        self,
        params: "GSTensor | nn.ParameterDict",
    ) -> tuple[nn.ParameterDict, int]:
        """Convert input to ParameterDict and get sh_degree."""
        from gsply import GSTensor

        if isinstance(params, GSTensor):
            # Convert GSTensor to ParameterDict
            gs = params.to(self.device)
            params_dict = nn.ParameterDict(
                {
                    "means": nn.Parameter(gs.means, requires_grad=False),
                    "quats": nn.Parameter(gs.quats, requires_grad=False),
                    "scales": nn.Parameter(gs.scales, requires_grad=False),  # log-space
                    "opacities": nn.Parameter(
                        gs.opacities, requires_grad=False
                    ),  # logit-space
                    "sh0": nn.Parameter(gs.sh0, requires_grad=False),
                }
            )
            if gs.shN is not None:
                params_dict["shN"] = nn.Parameter(gs.shN, requires_grad=False)
                k = gs.shN.shape[1]
                sh_degree = int(round((k + 1) ** 0.5)) - 1
            else:
                sh_degree = 0
        else:
            params_dict = params
            if "shN" in params_dict and params_dict["shN"] is not None:
                k = params_dict["shN"].shape[1]
                sh_degree = int(round((k + 1) ** 0.5)) - 1
            else:
                sh_degree = 0

        # Use configured sh_degree if specified
        if self.sh_degree is not None:
            sh_degree = min(sh_degree, self.sh_degree)

        return params_dict, sh_degree

    def _render_single(
        self,
        params: nn.ParameterDict,
        viewmat: Tensor,
        K: Tensor,
        width: int,
        height: int,
        sh_degree: int,
        render_mode: str = "RGB",
    ) -> tuple[Tensor, Tensor, Tensor | None, Dict]:
        """Render a single view."""
        means = params["means"]
        quats = params["quats"]
        scales = torch.exp(params["scales"])
        opacities = torch.sigmoid(params["opacities"])
        sh0 = params["sh0"]
        shN = params.get("shN")

        # Colors: sh0 only (degree 0) or full SH
        if sh_degree > 0 and shN is not None:
            colors = torch.cat([sh0.unsqueeze(1), shN], dim=1)
        else:
            colors = sh0.unsqueeze(1)  # (N, 1, 3)
            sh_degree = 0

        render_outputs, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=width,
            height=height,
            sh_degree=sh_degree,
            packed=False,
            render_mode=render_mode,
        )

        if render_mode in ["RGB+ED", "RGB+D"]:
            render_colors = render_outputs[..., :3]
            render_depths = render_outputs[..., 3:4]
        else:
            render_colors = render_outputs
            render_depths = None

        return render_colors, render_alphas, render_depths, info

    def render(
        self,
        params: "GSTensor | nn.ParameterDict",
        cameras: "CameraConfig",
        *,
        return_alphas: bool = False,
        return_depths: bool = False,
        show_progress: bool = False,
    ) -> RenderResult:
        """Render images from Gaussian parameters.

        Args:
            params: GSTensor or ParameterDict with Gaussian data
            cameras: CameraConfig with view matrices and intrinsics
            return_alphas: Whether to include alpha channels in result
            return_depths: Whether to include depth channels in result
            show_progress: Show progress bar

        Returns:
            RenderResult with rendered images (and optionally alphas/depths)
        """
        params_dict, sh_degree = self._get_params_dict(params)

        images: list[Tensor] = []
        alphas: list[Tensor] | None = [] if return_alphas else None
        depths: list[Tensor] | None = [] if return_depths else None

        iterator = range(len(cameras))
        if show_progress:
            iterator = tqdm(iterator, desc="Rendering")

        render_mode = "RGB+D" if return_depths else "RGB"

        with torch.no_grad():
            for i in iterator:
                viewmat = cameras.viewmats[i : i + 1]
                K = cameras.Ks[i : i + 1]

                render_colors, render_alphas, render_depths_out, _ = (
                    self._render_single(
                        params_dict,
                        viewmat,
                        K,
                        cameras.width_for(i),
                        cameras.height_for(i),
                        sh_degree,
                        render_mode=render_mode,
                    )
                )

                images.append(render_colors[0].clone())
                if return_alphas and alphas is not None:
                    alphas.append(render_alphas[0].clone())
                if return_depths and depths is not None:
                    depths.append(render_depths_out[0].clone())

        return RenderResult(images=images, alphas=alphas, depths=depths)

    def render_to_disk(
        self,
        params: "GSTensor | nn.ParameterDict",
        cameras: "CameraConfig",
        output_dir: Path | str,
        *,
        prefix: str = "render",
        format: str = "png",
        show_progress: bool = True,
    ) -> List[Path]:
        """Render and save images directly to disk."""
        import imageio.v3 as imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        params_dict, sh_degree = self._get_params_dict(params)
        paths = []

        iterator = range(len(cameras))
        if show_progress:
            iterator = tqdm(iterator, desc="Rendering to disk")

        with torch.no_grad():
            for i in iterator:
                viewmat = cameras.viewmats[i : i + 1]
                K = cameras.Ks[i : i + 1]

                render_colors, _, _, _ = self._render_single(
                    params_dict,
                    viewmat,
                    K,
                    cameras.width_for(i),
                    cameras.height_for(i),
                    sh_degree,
                    render_mode="RGB",
                )

                img = render_colors[0]
                np_img = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                path = output_dir / f"{prefix}_{i:04d}.{format}"
                imageio.imwrite(path, np_img)
                paths.append(path)

        return paths
