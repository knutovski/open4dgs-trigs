"""
TRiGS Viewer for TriGSVanilla checkpoints
==========================================
Renders a trained TRiGS scene interactively via viser + nerfview.

Key differences from the FreeTimeGS (velocity-only) viewer:

  1. Time in SECONDS, not normalised [0, 1].
  2. SE(3) deformation: coupled rotation + translation via Lie algebra.
  3. Hierarchical quadratic Bézier residuals on ζ(t).
  4. Local anchor-centred deformation with gauge fixing.

Checkpoint layout expected (trigs_ckpt.pt → ckpt["splats"]):
  means          [N, 3]   canonical positions
  quats          [N, 4]   canonical quaternions (wxyz)
  scales         [N, 3]   log scales
  opacities      [N]      logit opacities
  sh0            [N, 1, 3]
  shN            [N, K, 3]
  times          [N, 1]   μₜ  (seconds)
  log_durations  [N, 1]   log σₜ (seconds)
  anchors        [N, 3]   per-primitive anchor aᵢ
  zeta_base_r    [N, 3]   base ω (rotation)
  zeta_base_t    [N, 3]   base ν (translation)
  zeta_c0_r/t    [N, 3]   Bézier control point c₀
  zeta_c1_r/t    [N, 3]   Bézier control point c₁
  zeta_c2_r/t    [N, 3]   Bézier control point c₂

Usage:
    python viewer.py --ckpt results/trigs_output/trigs_ckpt.pt --port 8080

Follow a training run (the trainer writes params-only snapshots with
--config.viewer-snapshot-every N; each new one is swapped in live):
    python viewer.py --watch results/trigs_output/viewer_snapshots --port 8080

Optional overrides:
    --t-start / --t-end    fix the rendered time range in seconds
                           (defaults to auto-inferred from checkpoint)
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from trigs.compat.device import resolve_device, empty_cache, is_oom_error  # macOS port
import torch.nn.functional as F
import viser
from gsplat.rendering import rasterization

import nerfview

try:  # installed package
    from trigs.core.viewer_snapshot import (
        list_snapshots, load_snapshot, read_latest_pointer,
    )
except ImportError:  # run as a bare script: viewer_snapshot.py sits next to us
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from viewer_snapshot import (  # type: ignore[no-redef]
        list_snapshots, load_snapshot, read_latest_pointer,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SE(3) maths  (all batched, no external Lie-algebra library required)
# ──────────────────────────────────────────────────────────────────────────────

def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix.  v: [N, 3] → [N, 3, 3]"""
    N = v.shape[0]
    z = torch.zeros(N, device=v.device, dtype=v.dtype)
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    return torch.stack([
        torch.stack([ z, -vz,  vy], dim=-1),
        torch.stack([ vz,  z, -vx], dim=-1),
        torch.stack([-vy,  vx,  z], dim=-1),
    ], dim=-2)  # [N, 3, 3]


def se3_exp_batch(
    omega: torch.Tensor,   # [N, 3]  rotation part of se(3), already scaled by Δt
    nu:    torch.Tensor,   # [N, 3]  translation part of se(3), already scaled by Δt
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form SE(3) exponential map (Rodrigues + left Jacobian).

    Returns
    -------
    R : [N, 3, 3]  rotation matrices
    p : [N, 3]     translation vectors
    """
    N = omega.shape[0]
    I3 = torch.eye(3, device=omega.device, dtype=omega.dtype).unsqueeze(0).expand(N, -1, -1)

    theta    = omega.norm(dim=-1)                      # [N]
    theta_sq = theta ** 2                              # [N]
    eps      = 1e-6
    small    = theta < eps                             # [N]  small-angle branch

    K  = _skew(omega)                                  # [N, 3, 3]
    K2 = torch.bmm(K, K)                               # [N, 3, 3]

    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)

    # Rodrigues coefficients  (broadcast shape → [N, 1, 1] for matrix multiply)
    A = torch.where(small, torch.ones_like(theta),        sin_t / theta.clamp(min=eps))
    B = torch.where(small, torch.full_like(theta, 0.5),   (1 - cos_t) / theta_sq.clamp(min=eps))
    C = torch.where(small, torch.full_like(theta, 1/6),   (theta - sin_t) / (theta_sq * theta).clamp(min=eps))

    A = A[:, None, None]
    B = B[:, None, None]
    C = C[:, None, None]

    # R = I + A·K + B·K²
    R = I3 + A * K + B * K2                          # [N, 3, 3]

    # Left Jacobian  J = I + B·K + C·K²
    J = I3 + B * K + C * K2                          # [N, 3, 3]

    # p = J @ nu
    p = torch.bmm(J, nu.unsqueeze(-1)).squeeze(-1)   # [N, 3]

    return R, p


def _rot_to_quat_wxyz(R: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable rotation-matrix → unit quaternion (wxyz).
    Uses Shepperd's 4-case method.  R: [N, 3, 3]
    """
    # Quantities for all four cases
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]   # [N]

    # Case 0: w largest
    s0 = torch.sqrt((1 + trace).clamp(min=0)) / 2
    q_w = torch.stack([
        s0,
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * s0.clamp(min=1e-10)),
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * s0.clamp(min=1e-10)),
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * s0.clamp(min=1e-10)),
    ], dim=-1)

    # Case 1: x largest
    s1 = torch.sqrt((1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]).clamp(min=0)) / 2
    q_x = torch.stack([
        (R[:, 2, 1] - R[:, 1, 2]) / (4 * s1.clamp(min=1e-10)),
        s1,
        (R[:, 0, 1] + R[:, 1, 0]) / (4 * s1.clamp(min=1e-10)),
        (R[:, 0, 2] + R[:, 2, 0]) / (4 * s1.clamp(min=1e-10)),
    ], dim=-1)

    # Case 2: y largest
    s2 = torch.sqrt((1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]).clamp(min=0)) / 2
    q_y = torch.stack([
        (R[:, 0, 2] - R[:, 2, 0]) / (4 * s2.clamp(min=1e-10)),
        (R[:, 0, 1] + R[:, 1, 0]) / (4 * s2.clamp(min=1e-10)),
        s2,
        (R[:, 1, 2] + R[:, 2, 1]) / (4 * s2.clamp(min=1e-10)),
    ], dim=-1)

    # Case 3: z largest
    s3 = torch.sqrt((1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]).clamp(min=0)) / 2
    q_z = torch.stack([
        (R[:, 1, 0] - R[:, 0, 1]) / (4 * s3.clamp(min=1e-10)),
        (R[:, 0, 2] + R[:, 2, 0]) / (4 * s3.clamp(min=1e-10)),
        (R[:, 1, 2] + R[:, 2, 1]) / (4 * s3.clamp(min=1e-10)),
        s3,
    ], dim=-1)

    # Select the numerically dominant case per sample
    magnitudes = torch.stack([
        1 + trace,
        1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
        1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2],
        1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2],
    ], dim=-1)  # [N, 4]
    best = magnitudes.argmax(dim=-1)  # [N]

    b = best[:, None]                 # [N, 1]  for broadcasting
    q = torch.where(b == 0, q_w,
        torch.where(b == 1, q_x,
        torch.where(b == 2, q_y, q_z)))  # [N, 4]

    return F.normalize(q, dim=-1)


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Hamilton product for wxyz quaternions.  [N, 4] × [N, 4] → [N, 4]"""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)


def _bezier_quadratic(
    tau: torch.Tensor,  # [N]  ∈ [0, 1]
    c0: torch.Tensor,   # [N, D]
    c1: torch.Tensor,   # [N, D]
    c2: torch.Tensor,   # [N, D]
) -> torch.Tensor:
    """Quadratic Bézier: Q(τ) = (1−τ)² c₀ + 2(1−τ)τ c₁ + τ² c₂.  [N, D]"""
    t = tau.unsqueeze(-1)          # [N, 1]
    b0 = (1 - t) ** 2
    b1 = 2 * (1 - t) * t
    b2 = t ** 2
    return b0 * c0 + b1 * c1 + b2 * c2


def _tau_from_window(
    t_sec:      float,
    mu_t:       torch.Tensor,   # [N, 1]  seconds
    sigma_t:    torch.Tensor,   # [N, 1]  seconds
) -> torch.Tensor:
    """
    Normalised Bézier local time τ ∈ [0, 1] clipped to the visibility window.

    τ = clamp( (t − (μₜ − 2σₜ)) / (4σₜ),  0, 1 )
    """
    w_start = mu_t - 2.0 * sigma_t           # [N, 1]
    w_width = (4.0 * sigma_t).clamp(min=1e-8)
    return ((t_sec - w_start) / w_width).clamp(0.0, 1.0).squeeze(-1)  # [N]

def _gauge_fix_omega_viewer(
    omega: torch.Tensor,
    nu: torch.Tensor,
    anchor: torch.Tensor,
    eps: float = 1e-8,
    denom_clamp: float = 1e-10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Match trainer/math/se3.py::gauge_fix_omega.

    For ||omega||^2 > eps:
        a_perp = a - (a·omega / ||omega||^2) omega
        nu_par = (nu·omega / ||omega||^2) omega

    For tiny omega:
        a_perp = a
        nu_par = nu

    This tiny-omega case matters: otherwise pure/mostly translational motion
    gets incorrectly suppressed in the viewer.
    """
    omega_sq = (omega * omega).sum(dim=-1, keepdim=True)
    mask_small = omega_sq < eps
    omega_sq_safe = omega_sq.clamp(min=denom_clamp)

    a_dot_omega = (anchor * omega).sum(dim=-1, keepdim=True)
    nu_dot_omega = (nu * omega).sum(dim=-1, keepdim=True)

    a_perp_proj = anchor - (a_dot_omega / omega_sq_safe) * omega
    nu_par_proj = (nu_dot_omega / omega_sq_safe) * omega

    a_perp = torch.where(mask_small, anchor, a_perp_proj)
    nu_par = torch.where(mask_small, nu, nu_par_proj)
    return a_perp, nu_par


def _robust01(
    x: torch.Tensor,
    valid: Optional[torch.Tensor] = None,
    *,
    log10: bool = False,
    qlo: float = 0.02,
    qhi: float = 0.98,
) -> torch.Tensor:
    """Robustly normalize scalar tensor to [0,1] for debug coloring."""
    y = x.float()
    if log10:
        y = torch.log10(y.clamp(min=1e-12))

    finite = torch.isfinite(y)
    if valid is not None:
        finite = finite & valid

    if not bool(finite.any()):
        return torch.zeros_like(y)

    vals = y[finite]
    lo = torch.quantile(vals, qlo)
    hi = torch.quantile(vals, qhi)
    den = (hi - lo).clamp(min=1e-8)
    return ((y - lo) / den).clamp(0.0, 1.0)


def _scalar_to_debug_rgb(x01: torch.Tensor) -> torch.Tensor:
    """
    Simple blue→green/yellow→red debug colormap.
    x01: [N] in [0,1]
    returns [N,3]
    """
    x = x01.clamp(0.0, 1.0)
    r = (1.5 * x - 0.25).clamp(0.0, 1.0)
    g = (1.0 - (2.0 * x - 1.0).abs()).clamp(0.0, 1.0)
    b = (1.25 - 1.5 * x).clamp(0.0, 1.0)
    return torch.stack([r, g, b], dim=-1)

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConfigTriGS:
    ckpt: str = ""
    npz_path: Optional[str] = None
    port: int = 8080
    device: str = "auto"

    # Live mode: follow a trainer's viewer-snapshot directory (see
    # viewer_snapshot.py) and swap each new snapshot in without restarting.
    watch_dir: Optional[str] = None
    poll_s: float = 2.0
    auto_reload: bool = True

    # Playback. fps / total_frames / sh_degree: None = take the checkpoint's
    # value (30 / 300 / 3 if it has none); an explicit value always wins,
    # including across snapshot reloads. The resolved values are written back
    # into these fields so the rest of the viewer reads plain numbers.
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    t_start: Optional[float] = None   # override auto-inferred start time (seconds)
    t_end:   Optional[float] = None   # override auto-inferred end time (seconds)

    # Visibility masking. The temporal threshold MUST match the training
    # temporal_active_threshold (exp(-2) = 0.135335 in production runs): below
    # it, splats are in their τ-clamped, never-supervised extrapolation band
    # and glide with constant twist velocity — pure display artifact.
    temporal_opacity_threshold: float = 0.135335
    base_opacity_threshold:     float = 0.005

    # Spatial filtering
    use_spatial_filter:        bool  = True
    spatial_filter_percentile: float = 95.0
    spatial_filter_padding:    float = 1.1

    # Rendering
    sh_degree:   Optional[int] = None
    near_plane:  float = 0.01
    far_plane:   float = 1e10
    antialiased: bool  = False
    packed:      bool  = True

    # Performance
    precompute_visibility:    bool = True
    visibility_cache_frames:  int  = 300

    def __post_init__(self):
        # Remember what the user explicitly asked for before the fields are
        # overwritten with resolved values (see SplatsTriGS._resolve_meta).
        self._explicit = {
            k: getattr(self, k)
            for k in ("fps", "total_frames", "sh_degree")
            if getattr(self, k) is not None
        }


# ──────────────────────────────────────────────────────────────────────────────
# Splat container
# ──────────────────────────────────────────────────────────────────────────────

class SplatsTriGS:
    """
    TRiGS canonical Gaussians + temporal deformation parameters.

    The deformation pipeline at each time t (in seconds):

        1.  Temporal opacity γ(t) = exp(−½ ((t − μₜ) / σₜ)²)
        2.  Bézier local time  τ  = clamp((t − (μₜ − 2σₜ)) / 4σₜ, 0, 1)
        3.  Bézier residuals   ωᵣₑₛ(τ), νᵣₑₛ(τ)   (quadratic, 3 control pts)
        4.  Total              ω = ζᵇᵃˢᵉ_r + ωᵣₑₛ
                               ν = ζᵇᵃˢᵉ_t + νᵣₑₛ
        5.  Gauge fix
                ω̂ = ω / ‖ω‖
                a⊥ = a − ⟨a, ω̂⟩ ω̂          (anchor ⊥ to rotation axis)
                ν∥ = ⟨ν, ω̂⟩ ω̂              (translation ∥ to rotation axis)
        6.  SE(3) exp  (with Δt = t − μₜ applied to the log-velocity)
                R, p = SE3_exp( ν∥·Δt,  ω·Δt )
        7.  New position   μ(t) = R (μ − a⊥) + a⊥ + p
            New quaternion  q(t) = qᵣₑₗ ⊗ qᶜᵃⁿᵒⁿ
    """

    @classmethod
    def from_path(
        cls, path: str, cfg: ConfigTriGS, *,
        source_npz: Optional[Dict[str, np.ndarray]] = None, verbose: bool = True,
    ) -> "SplatsTriGS":
        if verbose:
            print(f"\nLoading TRiGS checkpoint: {path}")
        return cls(
            load_snapshot(path), cfg, source_npz=source_npz, verbose=verbose,
        )

    def __init__(
        self,
        ckpt: Dict[str, Any],
        cfg: ConfigTriGS,
        *,
        source_npz: Optional[Dict[str, np.ndarray]] = None,
        strict_npz: bool = True,
        verbose: bool = True,
    ):
        """Build from a loaded checkpoint / viewer-snapshot dict.

        ``source_npz`` holds the arrays of the row-aligned NPZ (loaded once by
        the caller). With ``strict_npz=False`` a row-count mismatch turns the
        source-cohort diagnostics off (``npz_note`` says why) instead of
        raising -- snapshots of a running job may legitimately change rows.
        """
        self.cfg = cfg
        self.dev = str(resolve_device(cfg.device))
        self._log = print if verbose else (lambda *a, **k: None)
        self.step: Optional[int] = ckpt.get("step")
        self.npz_note = ""

        self._log(f"  Top-level keys: {list(ckpt.keys())}")

        # TriGSVanilla saves under "params" (not "splats")
        splats = ckpt["params"]
        self._log(f"  Param keys: {sorted(splats.keys())}")

        # fps / n_frames / SH degree: explicit CLI value > checkpoint > default.
        meta = ckpt.get("selfcap_eval_meta", {})
        explicit = cfg._explicit
        cfg.fps = float(explicit.get("fps", meta.get("fps", ckpt.get("fps", 30.0))))
        cfg.total_frames = int(
            explicit.get("total_frames", meta.get("n_frames", ckpt.get("n_frames", 300)))
        )
        # A live snapshot also records the degree training is rendering at
        # (it ramps one band per 1000 steps); show that, not the final degree.
        self.sh_degree_render = int(explicit.get(
            "sh_degree", ckpt.get("sh_degree_active", ckpt.get("sh_degree", 3)),
        ))
        cfg.sh_degree = self.sh_degree_render

        def _load(key: str) -> torch.Tensor:
            t = splats[key]
            if t.dtype == torch.float16:  # viewer snapshots may store shN as half
                t = t.float()
            return t.to(self.dev)

                # ── Standard 3DGS params ──────────────────────────────────
        self.means    = _load("means")       # [N, 3]
        self.N        = self.means.shape[0]  # <-- MOVE THIS HERE

        self.quats    = _load("quats")       # [N, 4]
        self.scales   = _load("scales")      # [N, 3] log-scales
        self.opacities= _load("opacities")   # [N]    logit-opacities
        self.sh0      = _load("sh0")         # [N, 1, 3]

        # shN may be absent in degree-0 / early checkpoints
        self.shN = (
            _load("shN")
            if "shN" in splats
            else torch.zeros(self.N, 0, 3, device=self.dev)
        )

        # ── Temporal params  (all in seconds) ────────────────────
        self.mu_t       = _load("times")          # [N, 1]
        self.log_sig_t  = _load("log_durations")  # [N, 1]

        # ── TRiGS motion params ───────────────────────────────────
        self.anchors    = _load("anchors")      # [N, 3]
        self.zeta_br    = _load("zeta_base_r")  # [N, 3]
        self.zeta_bt    = _load("zeta_base_t")  # [N, 3]

        # Hybrid static/dynamic flags.
        if "is_static" in splats:
            self.is_static = _load("is_static").reshape(-1) > 0.5
        else:
            self.is_static = torch.zeros(self.N, dtype=torch.bool, device=self.dev)

        if "dynamic_protect" in splats:
            self.dynamic_protect = _load("dynamic_protect").reshape(-1) > 0.5
        else:
            self.dynamic_protect = torch.zeros(self.N, dtype=torch.bool, device=self.dev)

        self._log(
            f"  Static rows: {int(self.is_static.sum().item()):,} / {self.N:,} "
            f"({100.0 * float(self.is_static.float().mean().item()):.1f}%)"
        )

        # Bézier control points
        self.c0r = _load("zeta_c0_r"); self.c0t = _load("zeta_c0_t")
        self.c1r = _load("zeta_c1_r"); self.c1t = _load("zeta_c1_t")
        self.c2r = _load("zeta_c2_r"); self.c2t = _load("zeta_c2_t")

        # ── Diagnostic scalar fields ───────────────────────────────
        self.sigma_t_sec = torch.exp(self.log_sig_t).reshape(-1).clamp(min=1e-8)

        self.accel_t = (self.c0t - 2.0 * self.c1t + self.c2t).norm(dim=-1)
        self.accel_r = (self.c0r - 2.0 * self.c1r + self.c2r).norm(dim=-1)

        self.base_speed_t = self.zeta_bt.norm(dim=-1)
        self.base_speed_r = self.zeta_br.norm(dim=-1)

        self.dynamic_mask = ~self.is_static

        # Optional row-aligned NPZ provenance for exact source-cohort
        # diagnostics. Learned temporal centres are not a reliable substitute:
        # a |t-mu| window can contain fragments from several source keyframes.
        self.source_frames: Optional[torch.Tensor] = None
        self.source_frame_values: Tuple[int, ...] = ()
        if source_npz is not None:
            try:
                self._attach_source_npz(source_npz, cfg.npz_path)
            except ValueError as exc:
                if strict_npz:
                    raise
                self.source_frames = None
                self.source_frame_values = ()
                self.npz_note = f"source-cohort diagnostics off: {exc}"
                self._log(f"  {self.npz_note}")
            else:
                self._log(
                    "  NPZ source provenance: "
                    f"{len(self.source_frame_values)} dynamic cohorts "
                    f"({self.source_frame_values[0]}..{self.source_frame_values[-1]})"
                )
            # Relocation recycles rows at new places/times, so cohorts that
            # matched the NPZ at init describe fewer and fewer of them.
            tcfg = ckpt.get("config") or {}
            if (
                self.source_frames is not None
                and tcfg.get("relocate")
                and self.step is not None
                and int(self.step) >= int(tcfg.get("relocate_start_iter", 0))
            ):
                self.npz_note = "source cohorts may be stale: relocation has run"

        with torch.no_grad():
            dyn = self.dynamic_mask
            # Quantiles are only printed, so skip them on quiet snapshot reloads.
            if verbose and bool(dyn.any()):
                sig = self.sigma_t_sec[dyn].float()
                at = self.accel_t[dyn].float()
                ar = self.accel_r[dyn].float()
                self._log(
                    "  Dynamic diagnostics:\n"
                    f"    sigma_t q50/q95/q99/max = "
                    f"{torch.quantile(sig, 0.50).item():.4g} / "
                    f"{torch.quantile(sig, 0.95).item():.4g} / "
                    f"{torch.quantile(sig, 0.99).item():.4g} / "
                    f"{sig.max().item():.4g} s\n"
                    f"    accel_t q95/q99/max = "
                    f"{torch.quantile(at, 0.95).item():.4g} / "
                    f"{torch.quantile(at, 0.99).item():.4g} / "
                    f"{at.max().item():.4g}\n"
                    f"    accel_r q95/q99/max = "
                    f"{torch.quantile(ar, 0.95).item():.4g} / "
                    f"{torch.quantile(ar, 0.99).item():.4g} / "
                    f"{ar.max().item():.4g}",
                    flush=True,
                )

        self.base_opa = torch.sigmoid(self.opacities)
        self._dynamic_opacity_floor = 0.0

        # ── Infer sequence time range ─────────────────────────────
        # Prefer physical sequence range over sigma-derived range.
        # Static rows intentionally have huge sigma, so using mu ± 2σ makes playback
        # much too long for hybrid static/dynamic checkpoints.
        default_start = 0.0
        default_end = max(float(cfg.total_frames - 1), 1.0) / max(float(cfg.fps), 1e-6)

        self.t_start = cfg.t_start if cfg.t_start is not None else default_start
        self.t_end = cfg.t_end if cfg.t_end is not None else default_end
        self.duration = self.t_end - self.t_start

        self._log(f"  {self.N:,} Gaussians  |  "
              f"t ∈ [{self.t_start:.3f}s, {self.t_end:.3f}s]  "
              f"(duration {self.duration:.3f}s)")

        # ── Scene bounds + spatial / base-opacity filter ──────────
        self._build_base_mask()

        # ── Precompute temporal visibility masks ──────────────────
        self.vis_cache: Optional[Dict[int, torch.Tensor]] = None
        self.vis_threshold = float(cfg.temporal_opacity_threshold)
        if cfg.precompute_visibility:
            self._precompute_visibility()

    def _attach_source_npz(
        self, source_npz: Dict[str, np.ndarray], npz_label: Optional[str],
    ) -> None:
        """Validate row-aligned NPZ provenance against this model and attach it.
        Raises ValueError when the rows do not line up."""
        if "source_frames" not in source_npz:
            raise ValueError(f"NPZ {npz_label!r} has no 'source_frames' array")

        source_frames_np = np.asarray(source_npz["source_frames"]).reshape(-1)
        if source_frames_np.shape[0] != self.N:
            raise ValueError(
                "Checkpoint/NPZ row mismatch: "
                f"checkpoint={self.N:,}, "
                f"NPZ source_frames={source_frames_np.shape[0]:,}"
            )

        ckpt_static = self.is_static.detach().cpu().numpy()
        if "is_static" in source_npz:
            npz_static = np.asarray(source_npz["is_static"]).reshape(-1).astype(bool)
            if not np.array_equal(npz_static, ckpt_static):
                raise ValueError(
                    "Checkpoint/NPZ is_static rows do not align; "
                    "source-cohort filtering would be unsafe"
                )
        else:
            npz_static = ckpt_static

        source_frames_np = source_frames_np.astype(np.int64, copy=False)
        dynamic_source_frames = np.unique(source_frames_np[~npz_static])
        if dynamic_source_frames.size == 0:
            raise ValueError(f"NPZ {npz_label!r} has no dynamic source cohorts")

        self.source_frames = torch.from_numpy(source_frames_np.copy()).to(self.dev)
        self.source_frame_values = tuple(int(v) for v in dynamic_source_frames.tolist())

    # ── Mask building ─────────────────────────────────────────────────────────

    def _build_base_mask(self):
        pts = self.means.detach().cpu().numpy()
        lo  = np.percentile(pts, 5,  axis=0)
        hi  = np.percentile(pts, 95, axis=0)
        center = torch.tensor((lo + hi) / 2, device=self.dev, dtype=torch.float32)
        dist = (self.means - center).norm(dim=1)

        if self.cfg.use_spatial_filter:
            r = np.percentile(dist.cpu().numpy(), self.cfg.spatial_filter_percentile)
            r *= self.cfg.spatial_filter_padding
            spatial = dist <= r
            # Static rows are background by definition — a subject-centered
            # percentile radius must never clip them. (With few statics, e.g.
            # a strict prep run, the 95th-percentile radius collapses to the
            # subject capsule and would silently hide every wall/ceiling row.)
            if hasattr(self, "is_static"):
                spatial = spatial | self.is_static.reshape(-1).to(spatial.device)
        else:
            spatial = torch.ones(self.N, dtype=torch.bool, device=self.dev)

        self.spatial_mask = spatial
        opa_ok = self.base_opa >= self.cfg.base_opacity_threshold
        self.base_mask = spatial & opa_ok

        n = self.base_mask.sum().item()
        self._log(f"  Base mask (spatial + opacity ≥ {self.cfg.base_opacity_threshold}): "
              f"{n:,} / {self.N:,} ({100*n/self.N:.1f}%)")

    def _precompute_visibility(self):
        self._log("  Precomputing visibility masks …")
        # Recorded so a reload can tell whether the threshold slider moved
        # while this cache was being built.
        self.vis_threshold = float(self.cfg.temporal_opacity_threshold)
        self.vis_cache = {}
        n = min(self.cfg.visibility_cache_frames, self.cfg.total_frames)
        for i in range(n):
            t = self._frame_to_secs(i)
            g = self._temporal_opa(t)
            # g is renormalized: 0 exactly at the raw-γ threshold (see
            # visibility_mask) — mask on g > 0, not a second threshold pass.
            self.vis_cache[i] = self.base_mask & (g > 0.0)
            if i % 60 == 0:
                nv = self.vis_cache[i].sum().item()
                self._log(f"    frame {i:4d}: {nv:,} visible")
        self._log(f"  Cached {len(self.vis_cache)} frames.")

    # ── Time helpers ──────────────────────────────────────────────────────────

    def _frame_to_secs(self, frame: int) -> float:
        frac = frame / max(1, self.cfg.total_frames - 1)
        return self.t_start + frac * self.duration

    def source_cohort_mask(
        self,
        selected_source_frame: int,
        *,
        include_next: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[int, ...]]:
        """Return static context plus one or two exact NPZ source cohorts."""
        if self.source_frames is None or not self.source_frame_values:
            raise RuntimeError(
                "Source-cohort diagnostics require viewer --npz-path"
            )

        values = self.source_frame_values
        try:
            selected_index = values.index(int(selected_source_frame))
        except ValueError as exc:
            raise ValueError(
                f"Source frame {selected_source_frame} is not an NPZ cohort"
            ) from exc

        selected_values = (values[selected_index],)
        if include_next:
            if selected_index + 1 < len(values):
                selected_values = (values[selected_index], values[selected_index + 1])

        dynamic_keep = torch.zeros(
            self.N, dtype=torch.bool, device=self.dev
        )
        for source_frame in selected_values:
            dynamic_keep |= self.source_frames == int(source_frame)

        return self.is_static | dynamic_keep, selected_values

    def canonical_cohort_visibility(
        self,
        cohort_mask: torch.Tensor,
        *,
        opacity_floor: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Visibility and base opacity for a canonical single-cohort view."""
        effective_base_opa = self.base_opa
        if opacity_floor > 0.0:
            effective_base_opa = torch.where(
                self.dynamic_mask,
                effective_base_opa.clamp_min(float(opacity_floor)),
                effective_base_opa,
            )

        visible_dynamic = (
            cohort_mask
            & self.dynamic_mask
            & self.spatial_mask
            & (effective_base_opa >= self.cfg.base_opacity_threshold)
        )
        return visible_dynamic, effective_base_opa

    def _temporal_opa(self, t_sec: float) -> torch.Tensor:
        sigma = torch.exp(self.log_sig_t).clamp(min=1e-6)   # [N, 1]
        dt    = t_sec - self.mu_t                           # [N, 1]
        g = torch.exp(-0.5 * (dt / sigma) ** 2).squeeze(-1)
        
        # NEW: Match smooth-cull renormalization from trainer
        tau = self.cfg.temporal_opacity_threshold
        if tau > 0.0:
            g = ((g - tau) / (1.0 - tau)).clamp(min=0.0)

        if hasattr(self, "is_static"):
            g = torch.where(self.is_static, torch.ones_like(g), g)
        return g

    def visibility_mask(self, frame: int, t_sec: float) -> torch.Tensor:
        opacity_floor = float(getattr(self, "_dynamic_opacity_floor", 0.0))
        if (
            opacity_floor <= 0.0
            and self.vis_cache is not None
            and frame in self.vis_cache
        ):
            return self.vis_cache[frame]
        g = self._temporal_opa(t_sec)
        base_mask = self.base_mask
        if opacity_floor > 0.0:
            effective_base_opa = torch.where(
                self.dynamic_mask,
                self.base_opa.clamp_min(opacity_floor),
                self.base_opa,
            )
            base_mask = (
                self.spatial_mask
                & (effective_base_opa >= self.cfg.base_opacity_threshold)
            )
        # g is already smooth-cull renormalized (0 exactly at the raw-γ
        # threshold), so the mask is g > 0; comparing the renormalized value
        # against the threshold again would double-apply it and shrink the
        # window to ~±2.16σ.
        return base_mask & (g > 0.0)

    def diagnostic_filter_mask(
        self,
        *,
        show_static: bool = True,
        show_dynamic: bool = True,
        max_dynamic_sigma: float = 0.0,
        min_dynamic_sigma: float = 0.0,
        max_dynamic_abs_dt: float = 0.0,
        current_time_sec: Optional[float] = None,
        min_accel_t: float = 0.0,
        min_accel_r: float = 0.0,
        min_motion: float = 0.0,
        motion_disp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extra viewer-side diagnostic mask.

        max_dynamic_sigma:
            Hide dynamic splats with sigma_t above this value.
            Static splats always pass this filter.

        min_dynamic_sigma:
            Show only dynamic splats with sigma_t above this value.
            Useful for isolating pathological long-lived dynamic splats.

        max_dynamic_abs_dt:
            Keep only dynamic splats whose learned temporal centre is within
            this many seconds of current_time_sec. Unlike the temporal opacity
            threshold, this is an absolute window and does not scale with
            each splat's learned sigma_t.

        min_accel_t / min_accel_r:
            Show only dynamic splats with residual acceleration above threshold.
            Useful for finding Bézier-wobble splats.
        """
        keep = torch.ones(self.N, dtype=torch.bool, device=self.dev)
        dyn = self.dynamic_mask

        if not show_static:
            keep &= ~self.is_static
        if not show_dynamic:
            keep &= self.is_static

        if max_dynamic_sigma > 0.0:
            keep &= (~dyn) | (self.sigma_t_sec <= float(max_dynamic_sigma))

        if min_dynamic_sigma > 0.0:
            keep &= (~dyn) | (self.sigma_t_sec >= float(min_dynamic_sigma))

        if max_dynamic_abs_dt > 0.0:
            if current_time_sec is None:
                raise ValueError(
                    "current_time_sec is required when max_dynamic_abs_dt is enabled"
                )
            abs_dt = (self.mu_t.reshape(-1) - float(current_time_sec)).abs()
            keep &= (~dyn) | (abs_dt <= float(max_dynamic_abs_dt))

        if min_accel_t > 0.0:
            keep &= (~dyn) | (self.accel_t >= float(min_accel_t))

        if min_accel_r > 0.0:
            keep &= (~dyn) | (self.accel_r >= float(min_accel_r))

        if min_motion > 0.0 and motion_disp is not None:
            keep &= (~dyn) | (motion_disp >= float(min_motion))

        return keep


    def diagnostic_colors(self, mode: str, gamma: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Return [N,3] RGB debug colors, or None for normal SH colors.
        """
        mode = str(mode).lower().strip()
        dyn = self.dynamic_mask

        if mode in ("normal", "none"):
            return None

        if mode == "static/dynamic":
            rgb = torch.zeros(self.N, 3, device=self.dev)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            rgb[dyn] = torch.tensor([0.05, 0.25, 1.00], device=self.dev)
            return rgb

        if mode == "sigma":
            x = _robust01(self.sigma_t_sec, valid=dyn, log10=True)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        if mode == "accel t":
            x = _robust01(self.accel_t, valid=dyn, log10=True)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        if mode == "accel r":
            x = _robust01(self.accel_r, valid=dyn, log10=True)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        if mode == "base speed":
            x = _robust01(self.base_speed_t, valid=dyn, log10=True)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        if mode == "gamma":
            x = gamma.float().clamp(0.0, 1.0)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        if mode == "base opacity":
            x = self.base_opa.reshape(-1).float().clamp(0.0, 1.0)
            return _scalar_to_debug_rgb(x)

        if mode == "motion disp":
            motion_disp = getattr(self, "_last_motion_disp", None)
            if motion_disp is None:
                return None

            x = _robust01(motion_disp, valid=dyn, log10=True)
            rgb = _scalar_to_debug_rgb(x)
            rgb[self.is_static] = torch.tensor([0.75, 0.75, 0.75], device=self.dev)
            return rgb

        return None    

    @torch.no_grad()
    def motion_displacement_over_window(self, t_sec: float, dt_sec: float) -> torch.Tensor:
        old_disable = getattr(self, "_disable_temporal", False)

        t0 = max(self.t_start, t_sec - dt_sec)
        t1 = min(self.t_end,   t_sec + dt_sec)

        self._disable_temporal = True
        m0, _, _, _ = self.deform(t0)
        m1, _, _, _ = self.deform(t1)
        self._disable_temporal = old_disable

        disp = (m1 - m0).norm(dim=-1)
        disp[self.is_static] = 0.0
        return disp

    # ── SE(3) + Bézier deformation ────────────────────────────────────────────

    @torch.no_grad()
    def deform(self, t_sec: float):
        """
        Full TRiGS deformation at time t_sec (seconds).

        Returns
        -------
        means_t   : [N, 3]  deformed positions
        quats_t   : [N, 4]  deformed quaternions (wxyz, unit)
        opacities : [N]     combined opacity  base_opa × γ
        gamma     : [N]     temporal opacity  γ (for masking)
        """
        # Inside SplatsTriGS.deform(self, t_sec: float)
        sigma_t = torch.exp(self.log_sig_t).clamp(min=1e-6)  # [N, 1]
        dt      = t_sec - self.mu_t                          # [N, 1]

        # ── 1.  Temporal opacity ──────────────────────────────────
        gamma = torch.exp(-0.5 * (dt / sigma_t) ** 2).squeeze(-1)  # [N]

        # Match smooth-cull renormalization from trainer
        tau = self.cfg.temporal_opacity_threshold
        if tau > 0.0:
            gamma = ((gamma - tau) / (1.0 - tau)).clamp(min=0.0)

        if hasattr(self, '_disable_temporal') and self._disable_temporal:
            gamma = torch.ones_like(gamma)
        if hasattr(self, "is_static"):
            gamma = torch.where(self.is_static, torch.ones_like(gamma), gamma)

        # ── 2.  Bézier local time τ ───────────────────────────────
        tau = _tau_from_window(t_sec, self.mu_t, sigma_t)           # [N]

        # ── 3.  Quadratic Bézier residuals ────────────────────────
        omega_res = _bezier_quadratic(tau, self.c0r, self.c1r, self.c2r)
        nu_res    = _bezier_quadratic(tau, self.c0t, self.c1t, self.c2t)

        if getattr(self, "_freeze_residual", False):
            omega_res = torch.zeros_like(omega_res)
            nu_res = torch.zeros_like(nu_res)

        zeta_br = self.zeta_br
        zeta_bt = self.zeta_bt
        freeze_base = getattr(self, "_freeze_base", False)
        if freeze_base or getattr(self, "_freeze_base_rotation", False):
            zeta_br = torch.zeros_like(zeta_br)
        if freeze_base or getattr(self, "_freeze_base_translation", False):
            zeta_bt = torch.zeros_like(zeta_bt)

        omega = zeta_br + omega_res
        nu    = zeta_bt + nu_res

        if hasattr(self, "is_static"):
            static3 = self.is_static[:, None]
            omega = torch.where(static3, torch.zeros_like(omega), omega)
            nu = torch.where(static3, torch.zeros_like(nu), nu)

        # ── 5. Gauge fix — match trainer exactly ───────────────────
        a_perp, nu_par = _gauge_fix_omega_viewer(
            omega,
            nu,
            self.anchors,
            eps=1e-8,
            denom_clamp=1e-10,
        )

        # ── 6.  SE(3) exp  (scaled by Δt) ────────────────────────
        dt_s          = dt.squeeze(-1)                    # [N]
        omega_scaled  = omega * dt_s.unsqueeze(-1)        # [N,3]  = ω·Δt
        nu_par_scaled = nu_par * dt_s.unsqueeze(-1)       # [N,3]  = ν∥·Δt

        R, p = se3_exp_batch(omega_scaled, nu_par_scaled)  # [N,3,3], [N,3]

        # ── 7.  Apply deformation ─────────────────────────────────
        #   μ(t) = R (μ − a⊥) + a⊥ + p
        rel   = (self.means - a_perp).unsqueeze(-1)       # [N,3,1]
        means_t = torch.bmm(R, rel).squeeze(-1) + a_perp + p  # [N,3]

        #   q(t) = q_rel ⊗ q_canonical
        q_rel   = _rot_to_quat_wxyz(R)                    # [N,4]
        q_can   = F.normalize(self.quats, dim=-1)
        quats_t = F.normalize(_quat_mul_wxyz(q_rel, q_can), dim=-1)  # [N,4]
        if hasattr(self, "is_static"):
            static3 = self.is_static[:, None]
            means_t = torch.where(static3, self.means, means_t)
            quats_t = torch.where(static3.expand_as(quats_t), F.normalize(self.quats, dim=-1), quats_t)

        # Combined opacity. A dynamic-only floor is available as a viewer
        # diagnostic for distinguishing learned opacity starvation from
        # missing geometry; it is off by default and never changes the
        # checkpoint.
        base_opa = self.base_opa
        opacity_floor = float(getattr(self, "_dynamic_opacity_floor", 0.0))
        if opacity_floor > 0.0:
            base_opa = torch.where(
                self.dynamic_mask,
                base_opa.clamp_min(opacity_floor),
                base_opa,
            )
        opacities = (base_opa * gamma).clamp(min=1e-6)                # [N]

        return means_t, quats_t, opacities, gamma

    def get_at_time(
        self,
        frame: int,
        t_sec: float,
        apply_mask: bool = True,
    ) -> Tuple[torch.Tensor, ...]:
        """
        All rasterization inputs at time t_sec.

        Returns
        -------
        means, quats, scales, opacities, colors, mask
        """
        means_t, quats_t, opacities, gamma = self.deform(t_sec)
        scales = torch.exp(self.scales)
        sh0 = self.sh0 if self.sh0.dim() == 3 else self.sh0.unsqueeze(1)
        colors = torch.cat([sh0, self.shN], dim=1)

        mask = (self.visibility_mask(frame, t_sec)
                if apply_mask
                else torch.ones(self.N, dtype=torch.bool, device=self.dev))

        return means_t, quats_t, scales, opacities, colors, mask, gamma


# ──────────────────────────────────────────────────────────────────────────────
# Viewer
# ──────────────────────────────────────────────────────────────────────────────

class ViewerTriGS:
    """Interactive viser + nerfview viewer for a TRiGS checkpoint."""

    def __init__(self, cfg: ConfigTriGS):
        self.cfg    = cfg
        self.dev    = str(resolve_device(cfg.device))

        # Guards `self.splats` against the snapshot watcher swapping it while a
        # render / stats refresh / visibility recompute is reading it. The
        # replacement is built outside the lock, so renders only ever wait
        # for the pointer swap.
        self._splats_lock = threading.RLock()

        # Row-aligned NPZ provenance: read once, reused for every reload.
        self._source_npz: Optional[Dict[str, np.ndarray]] = None
        if cfg.npz_path is not None:
            with np.load(cfg.npz_path, allow_pickle=False) as z:
                self._source_npz = {
                    k: np.asarray(z[k]) for k in ("source_frames", "is_static")
                    if k in z.files
                }

        # Live mode state (watch_dir): which snapshot is on screen, what the
        # trainer last published, and a request from the UI to load a file.
        self._watch_dir: Optional[Path] = (
            Path(cfg.watch_dir) if cfg.watch_dir else None
        )
        self._loaded_file: Optional[str] = None
        self._loaded_meta: Dict[str, Any] = {}
        self._latest_ptr: Optional[Dict[str, Any]] = None
        self._want_file: Optional[str] = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._status_note = ""

        if self._watch_dir is not None:
            path, ptr = self._wait_for_first_snapshot()
            self.splats = SplatsTriGS(
                load_snapshot(path), cfg, source_npz=self._source_npz,
                strict_npz=False,
            )
            self._loaded_file = path.name
            self._loaded_meta = self._meta_from_ptr(ptr)
            self._latest_ptr = ptr
        else:
            self.splats = SplatsTriGS.from_path(
                cfg.ckpt, cfg, source_npz=self._source_npz,
            )

        self.current_frame = 0
        self.auto_play     = False
        self.play_speed    = cfg.fps

        self.server = viser.ViserServer(port=cfg.port, verbose=False)
        self._build_ui()

        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._render,
            mode="rendering",
        )

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        sp = self.splats

        with self.server.gui.add_folder("Animation"):
            self.frame_slider = self.server.gui.add_slider(
                "Frame", min=0, max=self.cfg.total_frames - 1, step=1, initial_value=0,
            )
            self.time_label = self.server.gui.add_markdown(
                self._time_md(0)
            )
            self.autoplay_cb = self.server.gui.add_checkbox(
                "Auto Play", initial_value=False,
            )
            self.speed_slider = self.server.gui.add_slider(
                "Play Speed (FPS)", min=1.0, max=120.0, step=1.0,
                initial_value=self.cfg.fps,
            )

        with self.server.gui.add_folder("Visibility Masking"):
            self.disable_temporal_cb = self.server.gui.add_checkbox(
                "Disable Temporal Fade (debug)", initial_value=False,
            )
            self.thresh_slider = self.server.gui.add_slider(
                "Temporal Opacity Threshold",
                min=0.0, max=0.5, step=0.005,
                initial_value=self.cfg.temporal_opacity_threshold,
            )
            self.mask_cb = self.server.gui.add_checkbox(
                "Apply Visibility Mask", initial_value=True,
            )

        with self.server.gui.add_folder("Rendering"):
            self.background_color = self.server.gui.add_rgb(
                "Background Color", initial_value=(255, 255, 255),
            )

        with self.server.gui.add_folder("Diagnostics"):
            self.scale_slider = self.server.gui.add_slider(
                "Global Splat Scale", min=0.001, max=2.0, step=0.01, initial_value=1.0,
            )
            self.opacity_slider = self.server.gui.add_slider(
                "Global Opacity", min=0.01, max=1.0, step=0.01, initial_value=1.0,
            )
            self.dynamic_opacity_floor_slider = self.server.gui.add_slider(
                "Dynamic Opacity Floor (debug, 0=off)",
                min=0.0, max=1.0, step=0.01, initial_value=0.0,
            )
            if sp.source_frame_values:
                source_frame_options = tuple(
                    str(value) for value in sp.source_frame_values
                )
                middle_source_frame = source_frame_options[
                    len(source_frame_options) // 2
                ]
                self.source_cohort_mode_dd = self.server.gui.add_dropdown(
                    "Exact NPZ Cohort View",
                    options=(
                        "Off",
                        "Selected cohort: learned canonical",
                        "Selected + next: learned handoff (full motion)",
                    ),
                    initial_value="Off",
                )
                self.source_cohort_frame_dd = self.server.gui.add_dropdown(
                    "NPZ Source Frame",
                    options=source_frame_options,
                    initial_value=middle_source_frame,
                )
            self.show_static_cb = self.server.gui.add_checkbox(
                "Show Static", initial_value=True,
            )
            self.show_dynamic_cb = self.server.gui.add_checkbox(
                "Show Dynamic", initial_value=True,
            )

            self.max_dyn_sigma_slider = self.server.gui.add_slider(
                "Hide Dynamic σ > s (0=off)",
                min=0.0, max=30.0, step=0.1, initial_value=0.0,
            )
            self.min_dyn_sigma_slider = self.server.gui.add_slider(
                "Show Only Dynamic σ ≥ s (0=off)",
                min=0.0, max=30.0, step=0.1, initial_value=0.0,
            )
            self.max_dyn_abs_dt_slider = self.server.gui.add_slider(
                "Max Dynamic |t - μₜ| (s, 0=off)",
                min=0.0, max=0.5, step=0.005, initial_value=0.0,
            )

            self.motion_dt_slider = self.server.gui.add_slider(
                "Motion Debug Δframes",
                min=1, max=60, step=1, initial_value=6,
            )

            self.min_motion_slider = self.server.gui.add_slider(
                "Show Only Motion ≥ world units (0=off)",
                min=0.0, max=1.0, step=0.001, initial_value=0.0,
            )

            self.freeze_residual_cb = self.server.gui.add_checkbox(
                "Freeze Bézier Residuals (debug)",
                initial_value=False,
            )

            self.freeze_base_cb = self.server.gui.add_checkbox(
                "Freeze Base Motion (debug)",
                initial_value=False,
            )
            self.freeze_base_rotation_cb = self.server.gui.add_checkbox(
                "Freeze Base Rotation ω only (debug)",
                initial_value=False,
            )
            self.freeze_base_translation_cb = self.server.gui.add_checkbox(
                "Freeze Base Translation ν only (debug)",
                initial_value=False,
            )

            dyn = sp.dynamic_mask
            if bool(dyn.any()):
                acc_t_p99 = float(torch.quantile(sp.accel_t[dyn].float(), 0.99).item())
                acc_r_p99 = float(torch.quantile(sp.accel_r[dyn].float(), 0.99).item())
            else:
                acc_t_p99 = 1.0
                acc_r_p99 = 1.0

            acc_t_max = max(acc_t_p99 * 2.0, 1e-3)
            acc_r_max = max(acc_r_p99 * 2.0, 1e-3)

            self.min_accel_t_slider = self.server.gui.add_slider(
                "Show Only accel_t ≥ (0=off)",
                min=0.0, max=acc_t_max, step=acc_t_max / 500.0, initial_value=0.0,
            )
            self.min_accel_r_slider = self.server.gui.add_slider(
                "Show Only accel_r ≥ (0=off)",
                min=0.0, max=acc_r_max, step=acc_r_max / 500.0, initial_value=0.0,
            )

            self.color_mode_dd = self.server.gui.add_dropdown(
                "Color Mode",
                options=[
                    "Normal",
                    "Static/Dynamic",
                    "Sigma",
                    "Accel T",
                    "Accel R",
                    "Base Speed",
                    "Motion Disp",
                    "Gamma",
                    "Base Opacity",
                ],
                initial_value="Normal",
            )

            self.motion_dt_slider = self.server.gui.add_slider(
                "Motion Debug Δframes",
                min=1,
                max=60,
                step=1,
                initial_value=6,
            )

            self.min_motion_slider = self.server.gui.add_slider(
                "Show Only Motion ≥ world units (0=off)",
                min=0.0,
                max=1.0,
                step=0.001,
                initial_value=0.0,
            )

        with self.server.gui.add_folder("Stats"):
            self.stats_md = self.server.gui.add_markdown("Initialising…")

        if self._watch_dir is not None:
            with self.server.gui.add_folder("Training Snapshot"):
                self.snap_status_md = self.server.gui.add_markdown(
                    self._snapshot_status_md()
                )
                self.auto_reload_cb = self.server.gui.add_checkbox(
                    "Auto-reload newest", initial_value=self.cfg.auto_reload,
                )
                self.reload_btn = self.server.gui.add_button("Reload latest now")
                names = self._snapshot_names()
                self.snap_dd = self.server.gui.add_dropdown(
                    "Snapshot", options=tuple(names),
                    initial_value=self._loaded_file,
                )
                self.jump_btn = self.server.gui.add_button(
                    "Jump to last-trained window"
                )

        # ── Callbacks ─────────────────────────────────────────────
        @self.frame_slider.on_update
        def _(_):
            self.current_frame = int(self.frame_slider.value)
            self._refresh_stats()
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.autoplay_cb.on_update
        def _(_):
            self.auto_play = self.autoplay_cb.value

        @self.speed_slider.on_update
        def _(_):
            self.play_speed = self.speed_slider.value

        @self.thresh_slider.on_update
        def _(_):
            self.cfg.temporal_opacity_threshold = self.thresh_slider.value
            with self._splats_lock:
                if self.splats.vis_cache is not None:
                    self.splats._precompute_visibility()
                
        @self.scale_slider.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)
                
        @self.opacity_slider.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.background_color.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.dynamic_opacity_floor_slider.on_update
        def _(_):
            self.splats._dynamic_opacity_floor = float(
                self.dynamic_opacity_floor_slider.value
            )
            self._refresh_stats()
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        if hasattr(self, "source_cohort_mode_dd"):
            @self.source_cohort_mode_dd.on_update
            def _(_):
                self._refresh_stats()
                if hasattr(self, "viewer"):
                    self.viewer.rerender(None)

            @self.source_cohort_frame_dd.on_update
            def _(_):
                self._refresh_stats()
                if hasattr(self, "viewer"):
                    self.viewer.rerender(None)

        @self.max_dyn_abs_dt_slider.on_update
        def _(_):
            self._refresh_stats()
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.freeze_residual_cb.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.freeze_base_cb.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.freeze_base_rotation_cb.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        @self.freeze_base_translation_cb.on_update
        def _(_):
            if hasattr(self, "viewer"):
                self.viewer.rerender(None)

        if self._watch_dir is not None:
            @self.reload_btn.on_click
            def _(_):
                ptr = read_latest_pointer(self._watch_dir, allow_legacy=True)
                if ptr is not None:
                    self._want_file = str(ptr["file"])
                    self._wake.set()

            @self.snap_dd.on_update
            def _(_):
                name = str(self.snap_dd.value)
                if name == self._loaded_file:
                    return
                # Picking anything but the newest means "stay there": turn
                # auto-reload off so the next poll doesn't jump away.
                names = self._snapshot_names()
                if names and name != names[0]:
                    self.auto_reload_cb.value = False
                self._want_file = name
                self._wake.set()

            @self.jump_btn.on_click
            def _(_):
                window = self._loaded_meta.get("extra", {}).get("trained_window")
                if not window:
                    return
                t_mid = 0.5 * (float(window[0]) + float(window[1]))
                with self._splats_lock:
                    sp = self.splats
                    frac = (t_mid - sp.t_start) / max(sp.duration, 1e-9)
                    frame = int(round(frac * (self.cfg.total_frames - 1)))
                self.current_frame = max(0, min(frame, self.cfg.total_frames - 1))
                self.frame_slider.value = self.current_frame
                self._refresh_stats()
                self.viewer.rerender(None)

    def _time_md(self, frame: int) -> str:
        t = self.splats._frame_to_secs(frame)
        return (f"**t = {t:.3f} s** "
                f"(sequence: {self.splats.t_start:.2f}→{self.splats.t_end:.2f} s)")

    def _source_cohort_selection(
        self,
    ) -> Tuple[Optional[torch.Tensor], Tuple[int, ...], bool]:
        """Resolve the exact NPZ cohort diagnostic currently selected."""
        if not hasattr(self, "source_cohort_mode_dd"):
            return None, (), False

        mode = str(self.source_cohort_mode_dd.value)
        if mode == "Off" or self.splats.source_frames is None:
            return None, (), False

        canonical = mode == "Selected cohort: learned canonical"
        cohort_mask, selected_values = self.splats.source_cohort_mask(
            int(self.source_cohort_frame_dd.value),
            include_next=not canonical,
        )
        return cohort_mask, selected_values, canonical

    def _refresh_stats(self):
        with self._splats_lock:
            self._refresh_stats_locked()

    def _refresh_stats_locked(self):
        f = self.current_frame
        t = self.splats._frame_to_secs(f)

        cohort_mask, selected_cohorts, canonical_cohort = (
            self._source_cohort_selection()
        )
        learned_handoff = cohort_mask is not None and not canonical_cohort
        self.splats._dynamic_opacity_floor = (
            0.0
            if learned_handoff
            else float(self.dynamic_opacity_floor_slider.value)
        )

        base_vis = self.splats.visibility_mask(f, t)

        if cohort_mask is not None and canonical_cohort:
            selected_dynamic, _ = self.splats.canonical_cohort_visibility(
                cohort_mask,
                opacity_floor=float(self.dynamic_opacity_floor_slider.value),
            )
            if not bool(self.show_dynamic_cb.value):
                selected_dynamic = torch.zeros_like(selected_dynamic)
            static_visible = base_vis & self.splats.is_static
            if not bool(self.show_static_cb.value):
                static_visible = torch.zeros_like(static_visible)
            msk = static_visible | selected_dynamic
        elif cohort_mask is not None:
            msk = base_vis & cohort_mask
            if not bool(self.show_static_cb.value):
                msk &= ~self.splats.is_static
            if not bool(self.show_dynamic_cb.value):
                msk &= self.splats.is_static
        elif hasattr(self, "show_static_cb"):
            diag_mask = self.splats.diagnostic_filter_mask(
                show_static=bool(self.show_static_cb.value),
                show_dynamic=bool(self.show_dynamic_cb.value),
                max_dynamic_sigma=float(self.max_dyn_sigma_slider.value),
                min_dynamic_sigma=float(self.min_dyn_sigma_slider.value),
                max_dynamic_abs_dt=float(self.max_dyn_abs_dt_slider.value),
                current_time_sec=t,
                min_accel_t=float(self.min_accel_t_slider.value),
                min_accel_r=float(self.min_accel_r_slider.value),
            )
            msk = base_vis & diag_mask
        else:
            msk = base_vis

        nv = int(msk.sum().item())
        dyn = self.splats.dynamic_mask
        visible_dyn = msk & dyn

        long_sigma_3 = visible_dyn & (self.splats.sigma_t_sec > 3.0)
        long_sigma_5 = visible_dyn & (self.splats.sigma_t_sec > 5.0)

        self.time_label.content = self._time_md(f)
        cohort_md = ""
        if selected_cohorts:
            if canonical_cohort:
                cohort_kind = (
                    "learned canonical; no temporal motion/fade; "
                    f"opacity floor={float(self.dynamic_opacity_floor_slider.value):.2f}"
                )
            else:
                cohort_kind = "learned handoff; checkpoint motion/visibility forced"
            cohort_md = (
                f"  \n**Exact NPZ cohort:** "
                f"{', '.join(str(value) for value in selected_cohorts)} "
                f"({cohort_kind})"
            )

        self.stats_md.content = (
            f"**Frame:** {f} / {self.cfg.total_frames - 1}  \n"
            f"**Time:** {t:.4f} s  \n"
            f"**Visible Gaussians:** {nv:,} / {self.splats.N:,} "
            f"({100 * nv / self.splats.N:.1f}%)  \n"
            f"**Visible dynamic:** {int(visible_dyn.sum().item()):,}  \n"
            f"**Visible dynamic σ > 3s:** {int(long_sigma_3.sum().item()):,}  \n"
            f"**Visible dynamic σ > 5s:** {int(long_sigma_5.sum().item()):,}"
            f"{cohort_md}"
        )

    # ── Render callback ───────────────────────────────────────────────────────

    def _render(self, camera_state: "nerfview.CameraState", img_wh) -> np.ndarray:
        with self._splats_lock:
            return self._render_locked(camera_state, img_wh)

    @torch.no_grad()
    def _render_locked(
        self,
        camera_state: "nerfview.CameraState",
        img_wh,   # tuple (W, H) in old nerfview; RenderTabState in new nerfview
    ) -> np.ndarray:
        # Support both old (tuple) and new (RenderTabState) nerfview APIs
        if isinstance(img_wh, tuple):
            W, H = img_wh
        else:
            W, H = img_wh.render_width, img_wh.render_height
        c2w     = torch.from_numpy(camera_state.c2w).float().to(self.dev)
        K       = torch.from_numpy(camera_state.get_K((W, H))).float().to(self.dev)
        viewmat = c2w.inverse()

        t_sec = self.splats._frame_to_secs(self.current_frame)

        cohort_mask, _selected_cohorts, canonical_cohort = (
            self._source_cohort_selection()
        )
        learned_handoff = cohort_mask is not None and not canonical_cohort

        # The learned-handoff diagnostic always evaluates the actual checkpoint
        # motion and visibility, regardless of stale debug controls from another
        # experiment.
        self.splats._disable_temporal = (
            False if learned_handoff else self.disable_temporal_cb.value
        )
        self.splats._freeze_residual = (
            False if learned_handoff else self.freeze_residual_cb.value
        )
        self.splats._freeze_base = (
            False if learned_handoff else self.freeze_base_cb.value
        )
        self.splats._freeze_base_rotation = (
            False if learned_handoff else self.freeze_base_rotation_cb.value
        )
        self.splats._freeze_base_translation = (
            False if learned_handoff else self.freeze_base_translation_cb.value
        )
        self.splats._dynamic_opacity_floor = (
            0.0
            if learned_handoff
            else float(self.dynamic_opacity_floor_slider.value)
        )

        motion_dt_sec = float(self.motion_dt_slider.value) / max(float(self.cfg.fps), 1e-6)
        motion_disp = self.splats.motion_displacement_over_window(t_sec, motion_dt_sec)
        self.splats._last_motion_disp = motion_disp

        means, quats, scales, opacities, colors, mask, gamma = self.splats.get_at_time(
            frame=self.current_frame,
            t_sec=t_sec,
            apply_mask=True if learned_handoff else self.mask_cb.value,
        )
        visibility_mask = mask

        diag_mask = self.splats.diagnostic_filter_mask(
            show_static=bool(self.show_static_cb.value),
            show_dynamic=bool(self.show_dynamic_cb.value),
            max_dynamic_sigma=float(self.max_dyn_sigma_slider.value),
            min_dynamic_sigma=float(self.min_dyn_sigma_slider.value),
            max_dynamic_abs_dt=float(self.max_dyn_abs_dt_slider.value),
            current_time_sec=t_sec,
            min_accel_t=float(self.min_accel_t_slider.value),
            min_accel_r=float(self.min_accel_r_slider.value),
            min_motion=float(self.min_motion_slider.value),
            motion_disp=motion_disp,
        )
        mask = mask & diag_mask

        if cohort_mask is not None:
            if canonical_cohort:
                selected_dynamic, canonical_opacities = (
                    self.splats.canonical_cohort_visibility(
                        cohort_mask,
                        opacity_floor=float(
                            self.dynamic_opacity_floor_slider.value
                        ),
                    )
                )
                if not bool(self.show_dynamic_cb.value):
                    selected_dynamic = torch.zeros_like(selected_dynamic)

                # Inspect exactly what training stored for this source cohort,
                # independently of temporal/deformation parameters. Keep learned
                # canonical means, covariance, appearance, and base opacity. The
                # opacity floor is applied only when its visible UI slider is
                # explicitly non-zero.
                means = torch.where(
                    selected_dynamic[:, None], self.splats.means, means
                )
                canonical_quats = F.normalize(self.splats.quats, dim=-1)
                quats = torch.where(
                    selected_dynamic[:, None], canonical_quats, quats
                )
                opacities = torch.where(
                    selected_dynamic, canonical_opacities, opacities
                )
                gamma = torch.where(
                    selected_dynamic, torch.ones_like(gamma), gamma
                )
                mask = (
                    (mask & self.splats.is_static)
                    | selected_dynamic
                )
            else:
                # Preserve the checkpoint's learned deformation and temporal
                # weighting, but allow only the selected adjacent source pair.
                # Ignore unrelated sigma/|dt|/motion debug filters so this is
                # an exact source-provenance comparison rather than another
                # learned-time fragment.
                mask = visibility_mask & cohort_mask
                if not bool(self.show_static_cb.value):
                    mask &= ~self.splats.is_static
                if not bool(self.show_dynamic_cb.value):
                    mask &= self.splats.is_static

        # --- Apply Diagnostic Scale and Opacity Multipliers ---
        scales = scales * self.scale_slider.value
        opacities = opacities * self.opacity_slider.value
        # ------------------------------------------------------

        debug_rgb = self.splats.diagnostic_colors(
            str(self.color_mode_dd.value),
            gamma=gamma,
        )
        debug_sh_degree = None
        if debug_rgb is not None:
            colors = debug_rgb[:, None, :]
            debug_sh_degree = 0

        if self.mask_cb.value or cohort_mask is not None:
            if mask.sum() == 0:
                background = np.asarray(
                    self.background_color.value, dtype=np.float32
                ) / 255.0
                return np.broadcast_to(background, (H, W, 3)).copy()
            means    = means[mask]
            quats    = quats[mask]
            scales   = scales[mask]
            opacities= opacities[mask]
            colors   = colors[mask]

        rast_mode = "antialiased" if self.cfg.antialiased else "classic"
        # Clamp sh_degree to what's actually present in the colors tensor
        if debug_sh_degree is not None:
            sh_degree = debug_sh_degree
        else:
            n_coeffs = colors.shape[-2]
            max_sh = int(n_coeffs ** 0.5) - 1
            sh_degree = min(self.splats.sh_degree_render, max_sh)
        background = (torch.tensor(
            self.background_color.value,
            device=self.dev,
            dtype=torch.float32,
        ) / 255.0)[None]  # [C=1, 3]: gsplat 1.5.3 expects one background per camera
        render_colors, _, _ = rasterization(
            means      = means,
            quats      = quats,
            scales     = scales,
            opacities  = opacities,
            colors     = colors,
            viewmats   = viewmat[None],
            Ks         = K[None],
            width      = W,
            height     = H,
            sh_degree  = sh_degree,
            near_plane = self.cfg.near_plane,
            far_plane  = self.cfg.far_plane,
            # macOS port: mpsplat implements only packed=False on MPS.
            packed     = self.cfg.packed and not str(self.dev).startswith("mps"),
            rasterize_mode = rast_mode,
            backgrounds    = background,
        )

        return render_colors[0, ..., :3].clamp(0, 1).cpu().numpy()

    # ── Live snapshot following ───────────────────────────────────────────────

    def _wait_for_first_snapshot(self) -> Tuple[Path, Dict[str, Any]]:
        print(f"\nWaiting for the first snapshot in {self._watch_dir} …", flush=True)
        while True:
            ptr = read_latest_pointer(self._watch_dir, allow_legacy=True)
            if ptr is not None and (self._watch_dir / str(ptr["file"])).is_file():
                return self._watch_dir / str(ptr["file"]), ptr
            time.sleep(self.cfg.poll_s)

    @staticmethod
    def _meta_from_ptr(ptr: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "file": ptr["file"], "step": ptr.get("step"), "n_steps": ptr.get("n_steps"),
            "written_unix": ptr.get("written_unix"), "extra": ptr.get("extra") or {},
        }

    @staticmethod
    def _meta_from_payload(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        extra = payload.get("extra") or {}
        return {
            "file": name, "step": payload.get("step"), "n_steps": payload.get("n_steps"),
            "written_unix": extra.get("written_unix"), "extra": extra,
        }

    def _snapshot_names(self) -> List[str]:
        """Snapshot files in the watch dir, newest first (always includes the
        one on screen, even if the trainer has since pruned it)."""
        names = [
            p.name for _, p in reversed(
                list_snapshots(self._watch_dir, include_legacy=True)
            )
        ]
        if self._loaded_file and self._loaded_file not in names:
            names.append(self._loaded_file)
        return names or [self._loaded_file or "(none)"]

    def _snapshot_status_md(self) -> str:
        m = self._loaded_meta
        if not m:
            return "Waiting for first snapshot…"
        ex = m.get("extra") or {}
        step = int(m.get("step") or 0)
        n_steps = int(m.get("n_steps") or 0)
        head = f"**Loaded:** step {step:,}"
        if n_steps:
            head += f" / {n_steps:,} ({100.0 * step / n_steps:.0f}%)"
        lines = [head]
        if ex.get("round"):
            line = f"round {ex['round']}/{ex['n_rounds']}"
            if ex.get("window_idx"):
                line += f" · window {ex['window_idx']}/{ex['n_windows']}"
            lines.append(line)
        window = ex.get("trained_window")
        if window:
            lines.append(f"last-trained window: {window[0]:.2f}–{window[1]:.2f} s")
            if ex.get("round") == 1 and ex.get("window_idx", 0) < ex.get("n_windows", 0):
                lines.append("_round 1: later windows are still at their initial state_")
        if ex.get("n_rows"):
            lines.append(f"{int(ex['n_rows']):,} rows")
        psnr = {
            k: v for k, v in (ex.get("last_psnr") or {}).items()
            if "num_samples" not in k
        }
        if psnr:
            lines.append(
                "last eval: " + ", ".join(
                    f"{k.replace('psnr_', '')} {v:.1f}" for k, v in list(psnr.items())[:3]
                )
            )
        written = m.get("written_unix")
        if written:
            age = max(time.time() - float(written), 0.0)
            lines.append(
                f"snapshot age: {age:.0f} s" if age < 120 else f"snapshot age: {age / 60:.1f} min"
            )
        latest = self._latest_ptr
        if (
            latest and latest.get("file") != self._loaded_file
            and int(latest.get("step") or 0) > step
        ):
            lines.append(f"**newer available:** step {int(latest['step']):,}")
        sp = getattr(self, "splats", None)
        if sp is not None and sp.npz_note:
            lines.append(f"_{sp.npz_note}_")
        if self._status_note:
            lines.append(self._status_note)
        return "  \n".join(lines)

    def _update_snapshot_ui(self) -> None:
        self.snap_status_md.content = self._snapshot_status_md()
        try:
            names = tuple(self._snapshot_names())
            if tuple(self.snap_dd.options) != names:
                self.snap_dd.options = names
            if self.snap_dd.value != self._loaded_file:
                self.snap_dd.value = self._loaded_file
        except Exception:  # cosmetic: never let the dropdown stop the watcher
            pass

    def _watch_loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as exc:  # keep watching whatever happens
                self._status_note = f"reload failed: {type(exc).__name__}: {exc}"
                print(f"[watch] {self._status_note}", flush=True)
                try:
                    self._update_snapshot_ui()
                except Exception:
                    pass
            self._wake.wait(self.cfg.poll_s)
            self._wake.clear()

    def _poll_once(self) -> None:
        ptr = read_latest_pointer(self._watch_dir, allow_legacy=True)
        if ptr is not None:
            self._latest_ptr = ptr
        target: Optional[str] = None
        if self._want_file:
            target, self._want_file = self._want_file, None
        elif (
            ptr is not None and self.auto_reload_cb.value
            and ptr["file"] != self._loaded_file
        ):
            target = str(ptr["file"])
        if target and target != self._loaded_file:
            self._load_and_swap(target)
        self._update_snapshot_ui()

    def _load_and_swap(self, name: str) -> None:
        """Build the replacement splats off the render path, then swap."""
        path = self._watch_dir / name
        try:
            payload = load_snapshot(path)
        except FileNotFoundError:
            # Pruned by the trainer between listing and loading; next poll
            # re-reads the pointer.
            self._status_note = f"{name} was pruned before it could be loaded"
            return
        except (RuntimeError, EOFError, OSError) as exc:
            self._status_note = f"could not read {name}: {exc}"
            return

        t0 = time.time()

        def build() -> SplatsTriGS:
            return SplatsTriGS(
                payload, self.cfg, source_npz=self._source_npz,
                strict_npz=False, verbose=False,
            )

        try:
            new = build()
        except Exception as exc:  # CUDA OOM or MPS 'out of memory'
            if not is_oom_error(exc):
                raise
            # No room for two models: drop the old one (renders wait on the
            # lock meanwhile) and build into the freed memory.
            with self._splats_lock:
                old, self.splats = self.splats, None
                del old
                empty_cache()
                try:
                    self.splats = build()
                except Exception:
                    # Keep something on screen: put the previous snapshot back.
                    try:
                        self.splats = SplatsTriGS(
                            load_snapshot(self._watch_dir / str(self._loaded_file)),
                            self.cfg, source_npz=self._source_npz,
                            strict_npz=False, verbose=False,
                        )
                    except Exception:
                        pass
                    raise
            new = None
        if new is not None:
            with self._splats_lock:
                old, self.splats = self.splats, new
            del old
        empty_cache()

        self._loaded_file = name
        self._loaded_meta = self._meta_from_payload(name, payload)
        self._status_note = ""

        # The threshold slider may have moved while the cache was building.
        with self._splats_lock:
            sp = self.splats
            if (
                sp.vis_cache is not None
                and sp.vis_threshold != float(self.cfg.temporal_opacity_threshold)
            ):
                sp._precompute_visibility()

        self._refresh_stats()
        if hasattr(self, "viewer"):
            self.viewer.rerender(None)
        print(
            f"[watch] loaded {name} (step {self._loaded_meta.get('step')}) "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n🚀  TRiGS viewer running at  http://localhost:{self.cfg.port}")
        self._refresh_stats()
        if self._watch_dir is not None:
            print(f"    following snapshots in {self._watch_dir} "
                  f"(poll {self.cfg.poll_s:g}s)")
            threading.Thread(
                target=self._watch_loop, name="snapshot-watcher", daemon=True,
            ).start()

        last_tick = time.time()
        try:
            while True:
                if self.auto_play:
                    now = time.time()
                    if now - last_tick >= 1.0 / max(self.play_speed, 1):
                        self.current_frame = (self.current_frame + 1) % self.cfg.total_frames
                        self.frame_slider.value = self.current_frame
                        self._refresh_stats()
                        self.viewer.rerender(None)
                        last_tick = now
                time.sleep(0.01)
        except KeyboardInterrupt:
            self._stop.set()
            print("\nViewer stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Interactive viewer for TriGSVanilla (TRiGS) checkpoints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt",          type=str, default=None,
                   help="Path to trigs_ckpt.pt (or a training snapshot); "
                        "required unless --watch is given")
    p.add_argument("--watch",         type=str, default=None, metavar="DIR",
                   help="Follow a training run: a viewer_snapshots/ directory "
                        "(trainer flag --config.viewer-snapshot-every), or a "
                        "--config.checkpoint-dir. New snapshots are swapped in "
                        "live; waits for the first one if none exist yet")
    p.add_argument("--poll-s",        type=float, default=2.0,
                   help="--watch: seconds between checks for a new snapshot")
    p.add_argument("--no-auto-reload", action="store_true",
                   help="--watch: start pinned on the first snapshot and only "
                        "reload on request (UI button / dropdown)")
    p.add_argument("--npz-path",      type=str, default=None,
                   help="Row-aligned source NPZ for exact source-cohort diagnostics")
    p.add_argument("--port",          type=int,   default=8080)
    p.add_argument("--device",        type=str,   default="auto")
    p.add_argument("--fps",           type=float, default=None,
                   help="Original sequence FPS (used for default play speed). "
                        "Default: from the checkpoint, else 30")
    p.add_argument("--total-frames",  type=int,   default=None,
                   help="Number of viewer frames to map across the sequence. "
                        "Default: from the checkpoint, else 300")
    p.add_argument("--t-start",       type=float, default=None,
                   help="Override sequence start time in seconds")
    p.add_argument("--t-end",         type=float, default=None,
                   help="Override sequence end time in seconds")
    p.add_argument("--temporal-threshold", type=float, default=0.05,
                   help="Temporal opacity threshold for visibility masking")
    p.add_argument("--spatial-percentile", type=float, default=95.0,
                   help="Percentile radius for spatial filter sphere")
    p.add_argument("--no-spatial-filter",  action="store_true",
                   help="Disable spatial filtering")
    p.add_argument("--no-precompute",      action="store_true",
                   help="Disable precomputed visibility cache")
    p.add_argument("--sh-degree",    type=int,  default=None,
                   help="Render SH degree. Default: the checkpoint's (a live "
                        "snapshot's is the degree training currently renders "
                        "at), else 3")
    p.add_argument("--antialiased",  action="store_true")
    a = p.parse_args()
    if a.watch is None and a.ckpt is None:
        p.error("one of --ckpt or --watch is required")
    if a.watch is not None and a.ckpt is not None:
        print("note: --watch given, ignoring --ckpt")

    cfg = ConfigTriGS(
        ckpt                       = a.ckpt or "",
        watch_dir                  = a.watch,
        poll_s                     = a.poll_s,
        auto_reload                = not a.no_auto_reload,
        npz_path                   = a.npz_path,
        port                       = a.port,
        device                     = a.device,
        fps                        = a.fps,
        total_frames               = a.total_frames,
        t_start                    = a.t_start,
        t_end                      = a.t_end,
        temporal_opacity_threshold = a.temporal_threshold,
        spatial_filter_percentile  = a.spatial_percentile,
        use_spatial_filter         = not a.no_spatial_filter,
        precompute_visibility      = not a.no_precompute,
        sh_degree                  = a.sh_degree,
        antialiased                = a.antialiased,
    )

    ViewerTriGS(cfg).run()


if __name__ == "__main__":
    main()
