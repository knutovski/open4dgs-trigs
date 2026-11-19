"""Motion-guided relocation as a gsplat :class:`Strategy` subclass.

Implements TRiGS §3.4 + Supplement §4: inactive primitives (``σ(α) < τ_α``)
are replaced by clones of active primitives, multinomial-sampled by a
four-term score combining opacity, motion × visibility, and accumulated
2-D / temporal gradients.

This wraps the relocation logic that used to live inside
``TrigsConsolidator._motion_guided_relocation`` so it conforms to the
standard gsplat :class:`gsplat.strategy.base.Strategy` interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
from trigs.compat.device import resolve_device  # macOS port
import torch.nn as nn
from torch import Tensor

from ..core.optimizer_state import build_param_owners, zero_param_state_rows
from gsplat.strategy.base import Strategy


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


class MotionGuidedRelocationStrategy(Strategy):
    """gsplat :class:`Strategy` wrapping TRiGS motion-guided relocation.

    Owns two persistent accumulators (``grad_2d_accum``, ``grad_t_accum``)
    populated by :meth:`accumulate_grads` after every backward pass. When
    :meth:`step_post_backward` fires at the configured cadence, inactive
    primitives are cloned from active ones weighted by a four-term score
    (see :func:`_motion_guided_relocation`), and the accumulators reset.
    """

    def __init__(self, config: MotionGuidedRelocationConfig):
        self.config = config

    # -------------------------------------------------------------------------
    # gsplat Strategy interface
    # -------------------------------------------------------------------------

    def check_sanity(
        self,
        params: Dict[str, nn.Parameter] | nn.ParameterDict,
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        """Validate that TRiGS canonical parameters are present."""
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
        """Allocate the two grad accumulators that drive relocation scores.

        Args:
            n_primitives: Initial primitive count (shape of the accumulators).
            device: Device for the accumulators.
        """
        device = resolve_device(device)  # macOS port: "auto" -> cuda/mps/cpu
        return {
            "grad_2d_accum": torch.zeros(n_primitives, device=device),
            "grad_t_accum": torch.zeros(n_primitives, device=device),
        }

    def step_pre_backward(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        """No-op — relocation only reads gradients after backward."""
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
        """Run motion-guided relocation at the configured cadence.

        Returns:
            Number of primitives relocated this call (0 if the cadence
            condition is not met).
        """
        cfg = self.config
        if not (
            cfg.relocate
            and step >= cfg.relocate_start_iter
            and cfg.relocate_every > 0
            and step % cfg.relocate_every == 0
        ):
            return 0
        n_reloc = _motion_guided_relocation(
            params, optimizers, cfg,
            state["grad_2d_accum"], state["grad_t_accum"],
        )
        # Reset the grad-based cues for the next window.
        state["grad_2d_accum"].zero_()
        state["grad_t_accum"].zero_()
        return n_reloc

    # -------------------------------------------------------------------------
    # Grad accumulation (called by the trainer between backward and zero_grad)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def accumulate_grads(
        self,
        state: Dict[str, Any],
        params: nn.ParameterDict,
        info: Dict[str, Any] | None,
    ) -> None:
        """Accumulate relocation cues from the most recent backward pass.

        Must be called BEFORE ``optimizer.zero_grad()``. No-ops when grads
        are unavailable (zero-loss step).
        """
        m2d = info.get("means2d") if isinstance(info, dict) else None
        if m2d is not None and m2d.grad is not None:
            g = m2d.grad.detach()
            # gsplat returns [C, N, 2]; average across cameras (C=1 here).
            state["grad_2d_accum"] += g.norm(dim=-1).mean(dim=0)
        if params["times"].grad is not None:
            state["grad_t_accum"] += params["times"].grad.detach().abs().squeeze(-1)


# =============================================================================
# Core relocation kernel (TRiGS §3.4 + Suppl §4)
# =============================================================================

@torch.no_grad()
def _motion_guided_relocation(
    params: nn.ParameterDict,
    optimizers: Dict[str, torch.optim.Optimizer],
    config: MotionGuidedRelocationConfig,
    grad_2d_accum: Tensor,
    grad_t_accum: Tensor,
) -> int:
    """Clone active primitives over inactive ones — TRiGS §3.4 + Suppl §4.

    Score (four normalized cues, each normalized over the active set):

        ``p^(α)  = σ(α_i) / Σ_{j∈A} σ(α_j)``                  — opacity
        ``p^(dν) = ‖ν_i‖·exp(s_{t,i}) / Σ_{j∈A} ...``          — motion × visibility
        ``p^(2D) = ‖∇_2D µ_i‖ / Σ_{j∈A} ...``                   — accumulated 2-D grad
        ``p^(t)  = |∂L/∂µ_{t,i}| / Σ_{j∈A} ...``                — accumulated temporal grad
        ``s_i   = p^(α) + p^(dν) + p^(2D) + p^(t)``

    Sources are multinomial-sampled from ``A`` weighted by ``s``. Each sampled
    primitive is cloned over an inactive slot with identical parameters
    except the scale, reduced to ``relocate_scale_factor × S`` (Suppl §4).

    Args:
        grad_2d_accum: [N] accumulated ‖∇_2D µ_i‖ since the last relocation.
        grad_t_accum:  [N] accumulated |∂L/∂µ_{t,i}| since the last relocation.

    Returns:
        Number of primitives relocated this call.
    """
    alpha = torch.sigmoid(params["opacities"].data).reshape(-1)
    inactive_mask = alpha < config.relocate_tau_alpha
    active_mask = ~inactive_mask
    n_inactive = int(inactive_mask.sum().item())
    n_active = int(active_mask.sum().item())
    if n_inactive == 0 or n_active == 0:
        return 0

    # Intrinsic cues.
    nu_base = params["zeta_base_t"].data
    log_sigma_t = params["log_durations"].data.squeeze(-1)
    raw_alpha = alpha
    raw_dnu = nu_base.norm(dim=-1) * torch.exp(log_sigma_t)
    # Grad cues are already magnitudes; just use directly.
    raw_2d = grad_2d_accum
    raw_t = grad_t_accum

    mask_f = active_mask.float()

    def _norm_over_actives(x: Tensor) -> Tensor:
        s = (x * mask_f).sum().clamp(min=1e-12)
        return (x * mask_f) / s

    score = (
        _norm_over_actives(raw_alpha)
        + _norm_over_actives(raw_dnu)
        + _norm_over_actives(raw_2d)
        + _norm_over_actives(raw_t)
    )
    score = score * mask_f
    if score.sum() <= 0:
        return 0
    probs = score / score.sum()

    # Sample sources with replacement.
    src_idx = torch.multinomial(probs, n_inactive, replacement=True)
    dst_idx = torch.nonzero(inactive_mask, as_tuple=False).squeeze(-1)
    n_rows = int(alpha.shape[0])
    owners = build_param_owners(optimizers)
    primitive_items = [
        (name, p) for name, p in params.items()
        if p.data.ndim >= 1 and int(p.data.shape[0]) == n_rows
    ]

    # Clone every tensor parameter, with reduced scale at destinations.
    for name, tensor in primitive_items:
        src_vals = tensor.data[src_idx].clone()
        if name == "scales":
            # scales stored in log-space (exp'd before rasterization);
            # reducing to ``factor × S`` means adding log(factor).
            src_vals = src_vals + math.log(config.relocate_scale_factor)
        tensor.data[dst_idx] = src_vals

        # Reset both sampled-source and overwritten-destination rows; keep
        # scalar optimizer state (e.g. step) intact.
        zero_param_state_rows(owners, tensor, src_idx, n_rows)
        zero_param_state_rows(owners, tensor, dst_idx, n_rows)

    return n_inactive
