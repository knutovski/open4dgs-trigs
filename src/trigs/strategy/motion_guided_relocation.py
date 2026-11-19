"""Motion-guided relocation as a gsplat :class:`Strategy` subclass.

Implements TRiGS §3.4 + Supplement §4: inactive primitives (``σ(α) < τ_α``)
are replaced by clones of active primitives, multinomial-sampled by a
four-term score combining opacity, motion × visibility, and accumulated
2-D / temporal gradients.

This version keeps relocation behavior intact, but adds:
  - finite/sane relocation score handling,
  - destination-row parameter sanitation,
  - destination-only optimizer state reset,
  - CUDA synchronization barriers around all in-place tensor surgery
    (fixes cudaErrorIllegalAddress that surfaces ~4 h into training),
  - optional debug shape checks via TRIGS_RELOC_DEBUG=1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Dict

import torch
from trigs.compat.device import device_synchronize, resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor

from ..core.optimizer_state import build_param_owners, zero_param_state_rows
from gsplat.strategy.base import Strategy

try:
    # MCMC-correct relocation split (donor opacity/scale are reduced so the
    # rendered image is approximately preserved when a clone is stamped).
    # Available in gsplat >= 1.0; fall back to legacy copy behavior if not.
    from gsplat.relocation import compute_relocation as _compute_relocation
except Exception:  # pragma: no cover - depends on installed gsplat
    _compute_relocation = None

_BINOMS_N_MAX = 51
_binoms_cache: dict = {}


def _relocation_binoms(device) -> Tensor:
    key = str(device)
    if key not in _binoms_cache:
        binoms = torch.zeros((_BINOMS_N_MAX, _BINOMS_N_MAX))
        for n in range(_BINOMS_N_MAX):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        _binoms_cache[key] = binoms.to(device)
    return _binoms_cache[key]


def _compute_relocation_torch(
    opacities: Tensor, scales: Tensor, ratios: Tensor, binoms: Tensor,
) -> tuple[Tensor, Tensor]:
    """Pure-torch port of gsplat's relocation kernel (3DGS-MCMC Eq. 9).

    For a donor split into ``n = ratio`` copies:
      new_opacity = 1 - (1 - opacity) ** (1 / n)
      new_scale   = scale * opacity / sum_{i=1..n} sum_{k<i}
                    C(i-1, k) (-1)^k / sqrt(k+1) * new_opacity^(k+1)
    Evaluated in float64: the alternating binomial sum cancels badly in
    float32 for large n. Used for CPU tensors; CUDA uses gsplat's kernel.
    """
    n_max = int(binoms.shape[0])
    ratios = ratios.to(torch.long).clamp(min=1, max=n_max)
    b = binoms.to(device=opacities.device, dtype=torch.float64)
    k = torch.arange(n_max, device=opacities.device, dtype=torch.float64)
    sign_sqrt = torch.where(k % 2 == 0, 1.0, -1.0) / torch.sqrt(k + 1.0)
    # weights[n, k] = sum_{i=k+1..n} C(i-1, k) * (-1)^k / sqrt(k+1)
    weights = torch.cat(
        [torch.zeros(1, n_max, dtype=torch.float64, device=b.device),
         torch.cumsum(b, dim=0) * sign_sqrt],
        dim=0,
    )
    op = opacities.to(torch.float64)
    new_op = 1.0 - (1.0 - op) ** (1.0 / ratios.to(torch.float64))
    powers = new_op.unsqueeze(-1) ** (k + 1.0)
    denom = (weights[ratios] * powers).sum(dim=-1)
    coeff = op / denom
    new_scales = coeff.unsqueeze(-1) * scales.to(torch.float64)
    return new_op.to(opacities.dtype), new_scales.to(scales.dtype)


def _relocation_split(
    opacities: Tensor, scales: Tensor, ratios: Tensor,
) -> tuple[Tensor, Tensor]:
    binoms = _relocation_binoms(opacities.device)
    if opacities.is_cuda and _compute_relocation is not None:
        return _compute_relocation(
            opacities=opacities, scales=scales, ratios=ratios, binoms=binoms,
        )
    return _compute_relocation_torch(opacities, scales, ratios, binoms)


# ---------------------------------------------------------------------------
# One helper so every in-place surgery site uses the same guard.
# ---------------------------------------------------------------------------

def _cuda_sync_if_available() -> None:
    """Drain all pending CUDA work.

    Must be called:
      1. Before any in-place ``tensor.data[idx] = ...`` write that touches
         storage that may still be read by an in-flight backward/optimizer
         kernel.
      2. After those writes, before optimizer-state tensors that share the
         same storage are accessed by ``zero_param_state_rows``.

    Without these barriers the scattered writes race with still-running CUDA
    kernels, producing ``cudaErrorIllegalAddress`` that surfaces one iteration
    later at the next host-device sync.
    """
    device_synchronize()


@dataclass
class MotionGuidedRelocationConfig:
    """Hyperparameters for motion-guided relocation (TRiGS §3.4 + Suppl §4)."""

    relocate: bool = True
    """Master switch. If False, relocation never fires."""

    relocate_every: int = 100
    """Run relocation every N steps. 0 disables."""

    relocate_start_iter: int = 500
    """First step at which relocation may run."""

    relocate_tau_alpha: float = 0.005
    """Sec. 4.1: τ_α. Primitives with sigmoid(opacity) below this are inactive."""

    relocate_scale_factor: float = 0.66
    """Suppl. §4: cloned primitives get their scale multiplied by this factor."""

    relocate_dynamic_only: bool = True
    """When hybrid static/dynamic params are present, only non-static rows may be
    used as relocation destinations or donors. Static rows are persistent 3D
    background rows and should not be recycled."""

    relocate_mcmc_split: bool = True
    """Official TRiGS relocation: gsplat's MCMC relocation math redistributes
    donor opacity and scale between donor and clones by copy count, so total
    opacity mass is approximately preserved and relocation events do not pop
    visibly (gsplat's CUDA kernel, or an exact torch port on CPU). False
    restores the paper-supplement recipe (clone scale x
    relocate_scale_factor, donor untouched). Measured risk: at thousands of events
    per run the split drains opacity; keep the event count near the
    official ~245/30k steps (relocate_every, relocate_stop_frac, batching)."""

    relocate_source_duration_divisor: float = 1.5
    """Official TRiGS: each selected donor's sigma_t is divided by this once
    per event (however many clones it spawns), and clones copy the result.
    Long-lived fast movers are the likeliest donors, so this is a downward
    pressure on sigma_t. 1.0 disables."""

    relocate_source_time_jitter_frac: float = 0.25
    """Official TRiGS: each selected donor's mu_t moves by uniform
    +/- (this x its pre-shrink sigma_t), and clones copy the result. 0
    disables."""

    relocate_reanchor_source: bool = True
    """Keep a donor's trajectory when its temporal window changes. Means are
    canonical at mu_t, so a bare mu_t shift (the official code) replays the
    whole trajectory at the wrong time. With this on, the donor's mean and
    quaternion are moved to where it actually is at the new mu_t and its
    Bézier residual collapses to the value there (exact position/orientation
    at the new center, twist preserved to first order). Needs a deform_fn;
    without one the verbatim official shift is applied."""

    relocate_time_max: float = 0.0
    """Upper clamp for jittered donor mu_t (the sequence end). <= 0 clamps
    only at zero."""

    relocate_time_jitter_sigma: float = 0.0
    """OPT-IN, off by default. Jitter each clone's mu_t by uniform
    +/- (factor x donor sigma_t). Exact-copy relocation replicates temporal
    cohort structure (shared support edges) every pass; a sigma-scaled
    temporal proposal diffuses it while keeping clones inside the donor's
    own support."""

    relocate_time_cue_per_visit: bool = True
    relocate_speed_cue_clip_edges: bool = True
    """Count only the part of the +/-2 sigma support inside the clip in the
    speed x lifetime cue (ours). False is official TRiGS: |velocity| x sigma."""
    """Divide the accumulated |dL/d mu_t| donor cue by the number of steps each
    row was in the rendered active set. False uses the official TRiGS total,
    which weights rows by how long they live as well as by how hard their
    frames are."""

    relocate_recenter_residual: bool = True
    """Collapse each clone's Bézier residual controls to the donor residual
    evaluated at the clone's (possibly jittered) μ_t phase: constant residual,
    correct instantaneous ζ, zero curvature. Verbatim control-point copies
    replay the donor's curve at the wrong phase and seed rogue window-edge
    velocities that compound over relocation generations. Disable to restore
    the verbatim copy behavior."""


class MotionGuidedRelocationStrategy(Strategy):
    """gsplat :class:`Strategy` wrapping TRiGS motion-guided relocation."""

    def __init__(self, config: MotionGuidedRelocationConfig):
        self.config = config

    def check_sanity(
        self,
        params: Dict[str, nn.Parameter] | nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        required = (
            "means", "quats", "scales", "opacities", "sh0",
            "times", "log_durations",
            "anchors",
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        )
        missing = [k for k in required if k not in params]
        if missing:
            raise ValueError(
                f"MotionGuidedRelocationStrategy: missing required params {missing}"
            )

    def initialize_state(
        self,
        n_primitives: int,
        device: torch.device | str = "auto",
    ) -> Dict[str, Any]:
        device = resolve_device(device)  # macOS port: "auto" -> cuda/mps/cpu
        return {
            "grad_2d_accum": torch.zeros(n_primitives, device=device),
            "grad_t_accum": torch.zeros(n_primitives, device=device),
            # Steps in which each row was in the rendered active set since the
            # last relocation. Donor cues are normalized by this so rows whose
            # temporal support is clipped by the sequence edges (half the
            # active frames of an interior row) are not structurally starved
            # of clones.
            "visit_accum": torch.zeros(n_primitives, device=device),
            # Camera views in which each row actually projected (radii > 0).
            # The 2-D cue is averaged over these, as in official TRiGS, so
            # off-screen active rows do not dilute it.
            "visible_accum": torch.zeros(n_primitives, device=device),
        }

    def step_pre_backward(self, *args: Any, **kwargs: Any) -> None:
        return

    def step_post_backward(
        self,
        *,
        params: nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        **kwargs: Any,
    ) -> int:
        cfg = self.config
        if not (
            cfg.relocate
            and step >= cfg.relocate_start_iter
            and cfg.relocate_every > 0
            and step % cfg.relocate_every == 0
        ):
            return 0

        # FIX: sync here so that any optimizer.step() or backward kernels
        # launched by the caller have fully completed before we touch tensor
        # storage inside _motion_guided_relocation.  Even if the caller also
        # syncs before calling us, doing it here makes the strategy
        # self-contained and safe regardless of call site.
        _cuda_sync_if_available()

        n_reloc = _motion_guided_relocation(
            params, optimizers, cfg,
            state["grad_2d_accum"], state["grad_t_accum"],
            visit_accum=state.get("visit_accum"),
            visible_accum=state.get("visible_accum"),
            deform_fn=kwargs.get("deform_fn"),
        )

        # FIX: sync again after relocation so that the scattered writes to
        # tensor.data[dst_idx] have fully landed before the caller resumes
        # and potentially reads those same memory locations (e.g. on the
        # very next render call or the post-step clamp).
        _cuda_sync_if_available()

        state["grad_2d_accum"].zero_()
        state["grad_t_accum"].zero_()
        for key in ("visit_accum", "visible_accum"):
            if key in state:
                state[key].zero_()
        return n_reloc

    @torch.no_grad()
    def accumulate_2d(
        self,
        state: Dict[str, Any],
        info: Dict[str, Any] | None,
        batch_size: int = 1,
    ) -> None:
        """Accumulate one rendered camera's image-space gradient cue.

        Official TRiGS: the means2d gradient is scaled to NDC (W/2, H/2) and
        by the batch size (undoing the 1/B loss average), and only views in
        which the row projected (radii > 0) count as appearances.
        """
        if not isinstance(info, dict):
            return
        m2d = info.get("means2d")
        if m2d is None or m2d.grad is None:
            return
        g = m2d.grad.detach().clone()
        width = info.get("width")
        height = info.get("height")
        b = float(max(int(batch_size), 1))
        if width is not None and height is not None:
            g[..., 0] *= float(width) / 2.0 * b
            g[..., 1] *= float(height) / 2.0 * b
        grad_norm = g.norm(dim=-1)
        if grad_norm.dim() == 2:
            grad_norm = grad_norm.sum(dim=0)
        grad_norm = grad_norm.reshape(-1)

        radii = info.get("radii")
        if radii is not None:
            vis = radii.detach() > 0
            if vis.dim() >= 1 and vis.shape[-1] == 2 and vis.dim() == g.dim():
                vis = vis.all(dim=-1)
            if vis.dim() == 2:
                vis = vis.sum(dim=0)
            vis = vis.reshape(-1).to(grad_norm.dtype)
        else:
            vis = torch.ones_like(grad_norm)

        visible_accum = state.get("visible_accum")
        active_idx = info.get("active_idx")
        if grad_norm.shape == state["grad_2d_accum"].shape:
            state["grad_2d_accum"] += grad_norm
            if visible_accum is not None:
                visible_accum += vis
        elif active_idx is not None:
            active_idx = active_idx.detach().long().reshape(-1)
            if int(active_idx.numel()) == int(grad_norm.numel()):
                state["grad_2d_accum"].index_add_(0, active_idx, grad_norm)
                if visible_accum is not None:
                    visible_accum.index_add_(0, active_idx, vis)

    @torch.no_grad()
    def accumulate_time(
        self,
        state: Dict[str, Any],
        params: nn.ParameterDict,
        active_idx: Tensor | None = None,
    ) -> None:
        """Accumulate one optimizer step's temporal cue.

        ``visit_accum`` counts steps in which a row was in the rendered active
        set; the time cue is averaged over it so rows whose support is clipped
        by the sequence edges are not starved of clones. ``active_idx`` None
        means every row was active.
        """
        visit_accum = state.get("visit_accum")
        if visit_accum is not None:
            if active_idx is None:
                visit_accum += 1.0
            else:
                idx = active_idx.detach().long().reshape(-1)
                visit_accum.index_add_(
                    0, idx, torch.ones(idx.numel(), device=visit_accum.device),
                )

        if params["times"].grad is not None:
            t_grad = params["times"].grad.detach().abs()
            if t_grad.dim() > 1:
                t_grad = t_grad.squeeze(-1)
            if t_grad.shape == state["grad_t_accum"].shape:
                state["grad_t_accum"] += t_grad

    @torch.no_grad()
    def accumulate_grads(
        self,
        state: Dict[str, Any],
        params: nn.ParameterDict,
        info: Dict[str, Any] | None,
    ) -> None:
        """Single-camera step: both cues from one render."""
        self.accumulate_2d(state, info, batch_size=1)
        active_idx = info.get("active_idx") if isinstance(info, dict) else None
        self.accumulate_time(state, params, active_idx)


@torch.no_grad()
def _debug_check_param_shapes(
    params: nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    n_rows: int,
) -> None:
    """Optional relocation sanity check. Enabled with TRIGS_RELOC_DEBUG=1."""
    if os.environ.get("TRIGS_RELOC_DEBUG", "0") != "1":
        return

    for name, p in params.items():
        if p.data.ndim >= 1 and int(p.data.shape[0]) != n_rows:
            raise RuntimeError(
                f"[TRIGS_RELOC_DEBUG] param {name} shape mismatch: "
                f"{tuple(p.data.shape)} expected first dim {n_rows}"
            )

    for opt_name, opt in optimizers.items():
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, {})
                for state_name, state_value in st.items():
                    if not torch.is_tensor(state_value):
                        continue
                    if state_value.ndim == 0:
                        continue
                    if state_value.shape == p.shape:
                        continue
                    if state_value.numel() == 1:
                        continue
                    raise RuntimeError(
                        f"[TRIGS_RELOC_DEBUG] optimizer state shape mismatch: "
                        f"opt={opt_name} state={state_name} "
                        f"state_shape={tuple(state_value.shape)} "
                        f"param_shape={tuple(p.shape)}"
                    )


DeformFn = Callable[[Dict[str, Tensor], Tensor], tuple[Tensor, Tensor]]
"""(row-sliced params, per-row times [n, 1]) -> (means_t, quats_t)."""


def _bezier_at(tau: Tensor, c0: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
    one_m = 1.0 - tau
    return one_m * one_m * c0 + 2.0 * one_m * tau * c1 + tau * tau * c2


@torch.no_grad()
def _update_donor_windows(
    params: nn.ParameterDict,
    donors: Tensor,
    config: MotionGuidedRelocationConfig,
    deform_fn: DeformFn | None,
) -> None:
    """Shrink and jitter each unique donor's temporal window in place.

    With ``relocate_reanchor_source`` and a ``deform_fn``, the donor's
    trajectory is kept: its canonical mean/quaternion move to its deformed
    state at the new mu_t and its Bézier residual collapses to the constant
    value there under the old window.
    """
    divisor = float(getattr(config, "relocate_source_duration_divisor", 1.0))
    jitter = float(getattr(config, "relocate_source_time_jitter_frac", 0.0))
    shrink = divisor > 1.0
    if int(donors.numel()) == 0 or (not shrink and jitter <= 0.0):
        return

    times = params["times"].data
    log_durations = params["log_durations"].data
    old_mu = torch.nan_to_num(times[donors], nan=0.0, posinf=0.0, neginf=0.0)
    old_log_sigma = log_durations[donors]
    old_sigma = torch.exp(old_log_sigma).clamp_min(1e-9)

    new_mu = old_mu
    if jitter > 0.0:
        new_mu = old_mu + (torch.rand_like(old_mu) - 0.5) * 2.0 * jitter * old_sigma
        new_mu = new_mu.clamp_min(0.0)
        t_max = float(getattr(config, "relocate_time_max", 0.0))
        if t_max > 0.0:
            new_mu = new_mu.clamp_max(t_max)

    reanchor = bool(getattr(config, "relocate_reanchor_source", True))
    if reanchor and deform_fn is not None:
        donor_rows = {
            name: params[name].data[donors]
            for name in (
                "means", "quats", "anchors", "times", "log_durations",
                "zeta_base_t", "zeta_base_r",
                "zeta_c0_t", "zeta_c1_t", "zeta_c2_t",
                "zeta_c0_r", "zeta_c1_r", "zeta_c2_r",
                "is_static",
            )
            if name in params
        }
        means_new, quats_new = deform_fn(donor_rows, new_mu)
        tau_new = (
            (new_mu - (old_mu - 2.0 * old_sigma)) / (4.0 * old_sigma)
        ).clamp(0.0, 1.0)
        for suffix in ("_t", "_r"):
            residual = _bezier_at(
                tau_new,
                donor_rows[f"zeta_c0{suffix}"],
                donor_rows[f"zeta_c1{suffix}"],
                donor_rows[f"zeta_c2{suffix}"],
            )
            residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
            for k in ("0", "1", "2"):
                params[f"zeta_c{k}{suffix}"].data[donors] = residual
        params["means"].data[donors] = torch.nan_to_num(
            means_new, nan=0.0, posinf=0.0, neginf=0.0,
        )
        params["quats"].data[donors] = torch.nn.functional.normalize(
            torch.nan_to_num(quats_new, nan=0.0), dim=-1,
        )

    times[donors] = new_mu
    if shrink:
        log_durations[donors] = old_log_sigma - math.log(divisor)


@torch.no_grad()
def _motion_guided_relocation(
    params: nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    config: MotionGuidedRelocationConfig,
    grad_2d_accum: Tensor,
    grad_t_accum: Tensor,
    visit_accum: Tensor | None = None,
    visible_accum: Tensor | None = None,
    deform_fn: DeformFn | None = None,
) -> int:
    """Clone active primitives over inactive ones — TRiGS §3.4 + Suppl §4.

    CUDA safety contract
    --------------------
    The caller (``step_post_backward``) MUST have called
    ``torch.cuda.synchronize()`` before entering this function, so that all
    backward / optimizer-step kernels have retired and no in-flight kernel
    holds a reference to the tensor storage we are about to overwrite.

    Internally this function calls ``_cuda_sync_if_available()`` a second time
    immediately before ``zero_param_state_rows`` to ensure the scattered
    ``tensor.data[dst_idx] = src_vals`` writes have committed before Adam's
    moment tensors (which share that storage layout) are zeroed.
    """

    alpha = torch.sigmoid(params["opacities"].data).reshape(-1)

    n_rows = int(alpha.shape[0])
    eligible_mask = torch.ones(n_rows, dtype=torch.bool, device=alpha.device)

    if getattr(config, "relocate_dynamic_only", True) and "is_static" in params:
        static_mask = params["is_static"].data.reshape(-1) > 0.5
        eligible_mask &= ~static_mask

    # Chunked-training ownership (trainer.train_chunked). Recycling must be
    # unique per row, so prefer the mu-partition mask (chunk_reloc_mask);
    # the trainability mask (chunk_train_mask) can span several chunks for
    # boundary-crossing supports and is only a fallback.
    if "chunk_reloc_mask" in params:
        owned_mask = params["chunk_reloc_mask"].data.reshape(-1) > 0.5
        eligible_mask &= owned_mask
    elif "chunk_train_mask" in params:
        owned_mask = params["chunk_train_mask"].data.reshape(-1) > 0.5
        eligible_mask &= owned_mask

    # Destinations: only inactive eligible rows.
    inactive_mask = (alpha < config.relocate_tau_alpha) & eligible_mask

    # Donors: only active eligible rows.
    active_mask = (alpha >= config.relocate_tau_alpha) & eligible_mask

    n_inactive = int(inactive_mask.sum().item())
    n_active = int(active_mask.sum().item())
    if n_inactive == 0 or n_active == 0:
        return 0

    _debug_check_param_shapes(params, optimizers, n_rows)

    nu_base = params["zeta_base_t"].data
    log_sigma_t = params["log_durations"].data.squeeze(-1)

    # Sanitize score cues to keep multinomial well-defined.
    raw_alpha = torch.nan_to_num(alpha, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    sigma_t = torch.exp(log_sigma_t.clamp(min=-20.0, max=20.0))
    # Motion x visibility uses the +/-2 sigma support clipped to the captured
    # sequence: support past the first/last frame has no photometric evidence
    # and counting it biases donor sampling toward the sequence interior,
    # draining relocation capacity from the edges (blurry first seconds).
    mu_t = params["times"].data.reshape(-1)
    if getattr(config, "relocate_speed_cue_clip_edges", True):
        support_lo = (mu_t - 2.0 * sigma_t).clamp_min(0.0)
        support_hi = mu_t + 2.0 * sigma_t
        t_max_param = params.get("chunk_sequence_t_max")
        if t_max_param is not None:
            support_hi = support_hi.clamp_max(float(t_max_param.reshape(()).item()))
        effective_span = (support_hi - support_lo).clamp_min(0.0)
    else:
        effective_span = sigma_t
    raw_dnu = torch.nan_to_num(
        nu_base.norm(dim=-1) * effective_span, nan=0.0, posinf=0.0, neginf=0.0,
    ).clamp_min(0.0)

    raw_2d = torch.nan_to_num(grad_2d_accum, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    raw_t  = torch.nan_to_num(grad_t_accum,  nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if visible_accum is not None:
        # Official TRiGS: the 2-D cue is the mean gradient per view in which
        # the row actually projected.
        raw_2d = raw_2d / torch.nan_to_num(
            visible_accum, nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(1.0)
    elif visit_accum is not None:
        raw_2d = raw_2d / torch.nan_to_num(
            visit_accum, nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(1.0)
    if visit_accum is not None and getattr(config, "relocate_time_cue_per_visit", True):
        # Per-active-step temporal gradient instead of total: rows active in
        # fewer sampled frames (temporal support clipped by the sequence
        # edges) otherwise accumulate proportionally less and are never
        # cloned.
        raw_t = raw_t / torch.nan_to_num(
            visit_accum, nan=0.0, posinf=0.0, neginf=0.0,
        ).clamp_min(1.0)

    mask_f = active_mask.float()
    eligible_f = eligible_mask.float()

    def _norm_over_eligible(x: Tensor) -> Tensor:
        # Official TRiGS normalizes each cue over every row (dead included)
        # and then samples among the alive ones.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        s = (x * eligible_f).sum()
        if not torch.isfinite(s) or s <= 1e-12:
            return torch.zeros_like(x)
        return (x * eligible_f) / s

    score = (
        _norm_over_eligible(raw_alpha)
        + _norm_over_eligible(raw_dnu)
        + _norm_over_eligible(raw_2d)
        + _norm_over_eligible(raw_t)
    )
    score = torch.nan_to_num(score * mask_f, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

    score_sum = score.sum()
    if not torch.isfinite(score_sum) or score_sum <= 1e-12:
        return 0

    probs = score / score_sum
    if not torch.isfinite(probs).all():
        bad = int((~torch.isfinite(probs)).sum().item())
        raise RuntimeError(f"[TRIGS_RELOC] non-finite relocation probs: {bad} entries")

    src_idx = torch.multinomial(probs, n_inactive, replacement=True)
    dst_idx = torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)

    # Official TRiGS perturbs each selected donor's temporal window before
    # copying it: sigma_t / divisor, mu_t + U(+/- frac * sigma_t). Clones
    # below read the updated donor rows.
    _update_donor_windows(params, torch.unique(src_idx), config, deform_fn)

    # FreeTimeGS++ / 3DGS-MCMC relocation split: redistribute the donor's
    # opacity and scale across (donor + its clones) so the rendered image is
    # approximately preserved. Without this, each clone adds full donor
    # opacity on top of the untouched donor — relocation events then pop in
    # as visible floaters. Duplicate donors receive identical adjusted
    # values (ratios are per-donor), so scattered writes are race-free.
    mcmc_op_logit = None
    mcmc_log_scales = None
    if getattr(config, "relocate_mcmc_split", True):
        eps = torch.finfo(torch.float32).eps
        ratios = (
            torch.bincount(src_idx, minlength=n_rows)[src_idx] + 1
        ).clamp(max=_BINOMS_N_MAX - 1).to(torch.int32)
        new_op, new_scales3 = _relocation_split(
            alpha[src_idx],
            torch.exp(params["scales"].data[src_idx]),
            ratios,
        )
        new_op = torch.nan_to_num(
            new_op, nan=config.relocate_tau_alpha,
        ).clamp(min=float(config.relocate_tau_alpha), max=1.0 - eps)
        mcmc_op_logit = torch.logit(new_op)
        mcmc_log_scales = torch.log(
            torch.nan_to_num(new_scales3, nan=1e-6).clamp(min=1e-8)
        ).clamp(min=-12.0, max=6.0)

    owners = build_param_owners(optimizers)
    primitive_items = [
        (name, p)
        for name, p in params.items()
        if p.data.ndim >= 1 and int(p.data.shape[0]) == n_rows
    ]

    # Clone times (with optional jitter) are computed once, up front: both the
    # "times" write and the Bézier-residual re-centering below need them.
    donor_times = torch.nan_to_num(
        params["times"].data[src_idx], nan=0.0, posinf=0.0, neginf=0.0,
    )
    donor_sigma = torch.exp(
        params["log_durations"].data[src_idx]
    ).reshape(donor_times.shape).clamp_min(1e-9)
    clone_times = donor_times
    jitter = float(getattr(config, "relocate_time_jitter_sigma", 0.0))
    if jitter > 0:
        clone_times = donor_times + (
            torch.rand_like(donor_times) - 0.5
        ) * 2.0 * jitter * donor_sigma

    # Re-center clone residuals: verbatim control-point copies replay the
    # donor's residual curve at the clone's (jittered) phase — phase-wrong
    # motion until re-optimized, and the γ-damped window-edge coefficients
    # then drift (measured: rogue-velocity rows have relocation_generation
    # p50 7 vs 2 overall). Collapsing the controls to the donor residual
    # evaluated at the clone's μ_t keeps the instantaneous ζ correct, zeroes
    # the curvature, and lets the clone relearn its own curve.
    residual_star: Dict[str, Tensor] = {}
    if bool(getattr(config, "relocate_recenter_residual", True)):
        tau_star = (
            (clone_times - (donor_times - 2.0 * donor_sigma))
            / (4.0 * donor_sigma)
        ).clamp(0.0, 1.0)
        one_m = 1.0 - tau_star
        for suffix in ("_t", "_r"):
            c0 = torch.nan_to_num(params[f"zeta_c0{suffix}"].data[src_idx])
            c1 = torch.nan_to_num(params[f"zeta_c1{suffix}"].data[src_idx])
            c2 = torch.nan_to_num(params[f"zeta_c2{suffix}"].data[src_idx])
            residual_star[suffix] = (
                one_m * one_m * c0 + 2.0 * one_m * tau_star * c1 + tau_star * tau_star * c2
            )

    # --- Phase 1: scattered writes to tensor.data ----------------------------
    # The caller synced before entering; no in-flight kernel should be
    # reading these locations.  We build the full list of
    # (tensor, values, target_idx) triples first so all clone() calls happen
    # before any write, avoiding any read-after-write hazard between
    # parameters that share storage.
    writes: list[tuple[Tensor, Tensor, Tensor]] = []
    for name, tensor in primitive_items:
        src_vals = tensor.data[src_idx].clone()

        if name == "is_static":
            # Relocated rows are always dynamic rows.
            src_vals = torch.zeros_like(src_vals)

        elif name in ("chunk_train_mask", "chunk_reloc_mask", "chunk_owner_mask"):
            # Rows reborn inside this chunk's owner partition remain owned
            # (and, when applicable, trainable) by this chunk. Donors are
            # already owner-only, so this is defence-in-depth.
            src_vals = torch.ones_like(src_vals)

        elif name == "relocation_generation":
            # Diagnostic lineage: zero means original, >0 means the row was
            # produced by one or more relocation events.
            src_vals = torch.nan_to_num(
                src_vals, nan=0.0, posinf=0.0, neginf=0.0,
            ) + 1.0

        elif name == "relocation_donor_count":
            # Diagnostic: a clone starts its own count; donors are counted
            # after the writes below.
            src_vals = torch.zeros_like(src_vals)

        elif name == "dynamic_protect":
            # Inherit donor protection status. This is useful if a protected dynamic
            # subject row is cloned into a weak dynamic destination.
            src_vals = torch.nan_to_num(src_vals, nan=0.0, posinf=0.0, neginf=0.0)
            src_vals = (src_vals > 0.5).to(src_vals.dtype)

        elif name == "scales":
            if mcmc_log_scales is not None:
                # Donor and clones share the redistributed scale.
                src_vals = mcmc_log_scales.to(src_vals.dtype).reshape(
                    src_vals.shape
                )
                writes.append((tensor, src_vals.clone(), src_idx))
            else:
                src_vals = src_vals + math.log(config.relocate_scale_factor)
                src_vals = torch.nan_to_num(src_vals, nan=-12.0, posinf=6.0, neginf=-12.0)
                src_vals = src_vals.clamp(min=-12.0, max=6.0)

        elif name == "quats":
            src_vals = torch.nan_to_num(src_vals, nan=0.0, posinf=0.0, neginf=0.0)
            norms = src_vals.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            degenerate = (norms < 1e-8).squeeze(-1)
            if degenerate.any():
                src_vals[degenerate] = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=src_vals.device, dtype=src_vals.dtype
                )
            src_vals = src_vals / norms

        elif name == "opacities":
            if mcmc_op_logit is not None:
                # Donor opacity is reduced; clones get the same reduced
                # value, keeping total opacity mass roughly constant.
                src_vals = mcmc_op_logit.to(src_vals.dtype).reshape(
                    src_vals.shape
                )
                writes.append((tensor, src_vals.clone(), src_idx))
            else:
                src_vals = torch.nan_to_num(src_vals, nan=-20.0, posinf=6.0, neginf=-20.0)
                src_vals = src_vals.clamp(min=-20.0, max=6.0)

        elif name == "log_durations":
            src_vals = torch.nan_to_num(src_vals, nan=-5.0, posinf=5.0, neginf=-5.0)
            src_vals = src_vals.clamp(min=-8.0, max=4.0)

        elif name == "times":
            src_vals = clone_times.reshape(src_vals.shape)

        elif name in (
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        ) and residual_star:
            src_vals = residual_star[name[-2:]].reshape(src_vals.shape).clone()

        elif name in (
            "means", "anchors",
            "zeta_base_t", "zeta_base_r",
            "zeta_c0_t", "zeta_c0_r",
            "zeta_c1_t", "zeta_c1_r",
            "zeta_c2_t", "zeta_c2_r",
        ):
            # Defence-in-depth: _post_step_safety_clamp should already have
            # zeroed any NaN here, but clamp again at the copy site so a bad
            # source primitive can never propagate NaN into a destination row.
            src_vals = torch.nan_to_num(src_vals, nan=0.0, posinf=0.0, neginf=0.0)

        writes.append((tensor, src_vals, dst_idx))

    for tensor, vals, target_idx in writes:
        tensor.data[target_idx] = vals

    if "relocation_donor_count" in params:
        donor_count = params["relocation_donor_count"].data
        donor_count.index_add_(
            0, src_idx,
            torch.ones(
                (int(src_idx.numel()),) + tuple(donor_count.shape[1:]),
                device=donor_count.device, dtype=donor_count.dtype,
            ),
        )

    # --- Phase 2: sync before touching optimizer state ----------------------
    # The scattered writes above are async CUDA scatter kernels.  We must
    # drain them before zero_param_state_rows reads or modifies the
    # optimizer moment tensors that cover the same parameter rows —
    # otherwise Adam's exp_avg / exp_avg_sq for dst rows may be zeroed while
    # the scatter write is still in flight, causing illegal memory access.
    _cuda_sync_if_available()

    for name, tensor in primitive_items:
        zero_param_state_rows(owners, tensor, src_idx, n_rows)  
        zero_param_state_rows(owners, tensor, dst_idx, n_rows)

    _debug_check_param_shapes(params, optimizers, n_rows)
    return n_inactive