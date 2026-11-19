"""Local rigid-twist initialization helpers for TRiGS base motion."""

from __future__ import annotations

import torch
from torch import Tensor

from .quaternion import quaternion_to_rotmat
from .se3 import bezier_quadratic, exp_se3_quat, gauge_fix_omega, tau_from_window


def _skew(r: Tensor) -> Tensor:
    """Return cross-product matrix [r]_x such that [r]_x x = r × x."""
    S = torch.zeros(*r.shape[:-1], 3, 3, device=r.device, dtype=r.dtype)
    S[..., 0, 1] = -r[..., 2]
    S[..., 0, 2] = r[..., 1]
    S[..., 1, 0] = r[..., 2]
    S[..., 1, 2] = -r[..., 0]
    S[..., 2, 0] = -r[..., 1]
    S[..., 2, 1] = r[..., 0]
    return S


@torch.no_grad()
def local_knn_indices(points: Tensor, k: int) -> Tensor:
    """kNN indices within one keyframe point set (self excluded when possible)."""
    N = int(points.shape[0])
    if N <= 0:
        return torch.empty((0, 0), dtype=torch.long, device=points.device)
    if N == 1:
        return torch.zeros((1, 1), dtype=torch.long, device=points.device)

    k_eff = max(1, min(int(k), N - 1))
    try:
        from trigs.compat.ops import knn_points  # macOS port (uses pytorch3d on CUDA if installed)

        res = knn_points(points.unsqueeze(0), points.unsqueeze(0), K=k_eff + 1)
        return res.idx.squeeze(0)[:, 1:].contiguous()
    except ImportError:
        d = torch.cdist(points, points)
        d.fill_diagonal_(float("inf"))
        return d.topk(k_eff, largest=False).indices


@torch.no_grad()
def fit_local_twist(
    means: Tensor,
    v_targets: Tensor,
    anchors: Tensor,
    nn_idx: Tensor,
    *,
    reg: float = 1e-4,
    gauge_eps: float = 1e-8,
    gauge_denom_clamp: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit one local rigid twist per primitive from neighborhood velocities.

    Solves for each primitive i:
        argmin_{ν_i, ω_i} Σ_j w_ij || v_j - ν_i - ω_i × (x_j - a_i) ||²

    with Gaussian spatial weights over same-keyframe neighbors and a Tikhonov
    regularizer. Ill-conditioned solves fall back to:
        ω_i = 0, ν_i = weighted mean_j(v_j).

    The model moves a primitive by a screw (``TrigsTrainer._deform``): rotation
    about the axis through its anchor plus only the axis-parallel part of ν
    (gauge_fix_omega). Keeping the fitted anchor and dropping ν⊥ would change
    every rotating primitive's velocity by ν⊥ -- measured on example capture,
    fast keyframes started 5-6 point spacings off their NPZ path over a
    2.5-frame lifetime. Instead the anchor moves onto the screw axis,
        a* = a + ω × ν / ‖ω‖²,
    where ω × (x - a*) + ν∥ = ν + ω × (x - a) for every x, so each primitive's
    fitted velocity is kept exactly. An axis farther from the anchor than the
    keyframe's own extent is a translation in practice: those rows get ω = 0
    and their fitted velocity at the primitive as ν.

    Returns:
        nu: [N, 3] axis-parallel translation coefficient (full ν when ω = 0).
        omega: [N, 3] angular coefficient.
        anchors: [N, 3] screw-axis anchors.
    """
    N = int(means.shape[0])
    if N == 0:
        z = torch.zeros_like(means)
        return z, z, anchors.clone()
    if N == 1:
        return v_targets.clone(), torch.zeros_like(means), anchors.clone()

    xj = means[nn_idx]  # [N, K, 3]
    vj = v_targets[nn_idx]  # [N, K, 3]
    ai = anchors[:, None, :]  # [N, 1, 3]
    rj = xj - ai  # [N, K, 3]

    dij2 = ((xj - means[:, None, :]) ** 2).sum(dim=-1, keepdim=True)
    sigma2 = dij2.mean(dim=1, keepdim=True).clamp(min=1e-8)
    w = torch.exp(-dij2 / (2.0 * sigma2))  # [N, K, 1]

    K = xj.shape[1]
    I3 = torch.eye(3, device=means.device, dtype=means.dtype).view(1, 1, 3, 3)
    I3 = I3.expand(N, K, -1, -1)
    A = torch.cat([I3, -_skew(rj)], dim=-1)  # [N, K, 3, 6]
    b = vj.unsqueeze(-1)  # [N, K, 3, 1]

    At = A.transpose(-1, -2)  # [N, K, 6, 3]
    w4 = w.unsqueeze(-1)  # [N, K, 1, 1]
    AtWA = (At @ (w4 * A)).sum(dim=1)  # [N, 6, 6]
    AtWb = (At @ (w4 * b)).sum(dim=1)  # [N, 6, 1]

    I6 = torch.eye(6, device=means.device, dtype=means.dtype).unsqueeze(0)
    A_reg = AtWA + reg * I6

    # Stable fallback for singular/ill-conditioned systems.
    if hasattr(torch.linalg, "solve_ex"):
        xi, info = torch.linalg.solve_ex(A_reg, AtWb)
        bad = info != 0
    else:
        xi = torch.linalg.solve(A_reg, AtWb)
        bad = torch.zeros((N,), dtype=torch.bool, device=means.device)
    bad = bad | (~torch.isfinite(xi).all(dim=(1, 2)))

    nu = xi[:, :3, 0]
    omega = xi[:, 3:, 0]

    if bad.any():
        w_sum = w.sum(dim=1).clamp(min=1e-8)
        nu_fallback = (w * vj).sum(dim=1) / w_sum
        omega_fallback = torch.zeros_like(nu_fallback)
        bad_col = bad.unsqueeze(-1)
        nu = torch.where(bad_col, nu_fallback, nu)
        omega = torch.where(bad_col, omega_fallback, omega)

    # nu is the fitted velocity at the anchor; convert to screw form.
    w2 = (omega * omega).sum(dim=-1, keepdim=True)
    rotating = w2 > gauge_eps
    w2_safe = w2.clamp(min=gauge_denom_clamp)
    axis_anchors = anchors + torch.cross(omega, nu, dim=-1) / w2_safe
    nu_par = (nu * omega).sum(dim=-1, keepdim=True) / w2_safe * omega
    extent = (means.max(dim=0).values - means.min(dim=0).values).max()
    translating = rotating & (
        (axis_anchors - anchors).norm(dim=-1, keepdim=True) > extent
    )
    screw = rotating & ~translating
    v_at_point = nu + torch.cross(omega, means - anchors, dim=-1)

    # Below gauge_eps the model treats the row as a pure translation and uses
    # nu whole; write that state exactly (omega = 0, nu = velocity at the row).
    out_nu = torch.where(screw, nu_par, v_at_point)
    out_omega = torch.where(screw, omega, torch.zeros_like(omega))
    out_anchors = torch.where(screw, axis_anchors, anchors)
    return out_nu, out_omega, out_anchors


_CTRL_NAMES = ("zeta_c0_t", "zeta_c1_t", "zeta_c2_t", "zeta_c0_r", "zeta_c1_r", "zeta_c2_r")


def screw_deform_means(
    means: Tensor,
    anchors: Tensor,
    mu_t: Tensor,
    log_sigma_t: Tensor,
    nu_base: Tensor,
    omega_base: Tensor,
    ctrl: dict[str, Tensor],
    t: Tensor,
    *,
    gauge_eps: float = 1e-8,
    gauge_denom_clamp: float = 1e-10,
) -> Tensor:
    """Primitive centres at time ``t``: the position part of
    ``TrigsTrainer._deform`` (base twist + Bézier residual over the
    ±2σ window, gauge-fixed screw about the anchor)."""
    tau = tau_from_window(t, mu_t, log_sigma_t)
    nu = nu_base + bezier_quadratic(tau, ctrl["zeta_c0_t"], ctrl["zeta_c1_t"], ctrl["zeta_c2_t"])
    omega = omega_base + bezier_quadratic(tau, ctrl["zeta_c0_r"], ctrl["zeta_c1_r"], ctrl["zeta_c2_r"])
    a_perp, nu_par = gauge_fix_omega(omega, nu, anchors, eps=gauge_eps, denom_clamp=gauge_denom_clamp)
    dt = t - mu_t
    q_rel, p_par = exp_se3_quat(torch.cat([nu_par * dt, omega * dt], dim=-1).clamp(-1e4, 1e4))
    R = quaternion_to_rotmat(q_rel)
    return (R @ (means - a_perp).unsqueeze(-1)).squeeze(-1) + a_perp + p_par


@torch.no_grad()
def smooth_accelerations(accel: Tensor, has_accel: Tensor, nn_idx: Tensor) -> Tensor:
    """Per-row median of the measured acceleration over the row and its
    neighbours (``nn_idx``, same keyframe), using only rows with a measurement.

    A second difference of 1-frame tracks is noisy per point (example capture:
    77% of points agree in direction one frame later); the neighbourhood
    median keeps the shared motion. Rows with no measured neighbour get 0.
    """
    if int(accel.shape[0]) == 0:
        return accel.clone()
    self_idx = torch.arange(accel.shape[0], device=accel.device).unsqueeze(1)
    idx = torch.cat([self_idx, nn_idx], dim=1)
    vals = accel[idx]
    vals = torch.where(has_accel[idx].unsqueeze(-1), vals, torch.full_like(vals, float("nan")))
    return torch.nan_to_num(torch.nanmedian(vals, dim=1).values, nan=0.0)


def fit_bezier_to_acceleration(
    means: Tensor,
    anchors: Tensor,
    mu_t: Tensor,
    log_sigma_t: Tensor,
    nu_base: Tensor,
    omega_base: Tensor,
    v_target: Tensor,
    a_target: Tensor,
    *,
    n_samples: int = 9,
    iters: int = 300,
    lr: float = 0.02,
    gauge_eps: float = 1e-8,
    gauge_denom_clamp: float = 1e-10,
    row_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Bézier residual controls so each primitive starts on the path
    ``x + v·dt + ½·a·dt²`` over its ±2σ window.

    The model keeps only the axis-parallel part of a translation (the screw
    gauge), so an acceleration across the rotation axis cannot be written
    into the translation controls directly; the controls (translation and
    rotation) are fitted against the model's own deformation instead.
    Base twist and anchors are left as fitted from the velocities.

    ``row_weight`` (n,) replaces the mean over rows with a weighted sum. Rows
    are independent, so weighting each by 1/len(its set) fits several sets in
    one batch with every row's gradient (and Adam step) unchanged.
    """
    n = int(means.shape[0])
    ctrl = {name: torch.zeros(n, 3, device=means.device, dtype=means.dtype) for name in _CTRL_NAMES}
    if n == 0 or iters <= 0:
        return ctrl
    sigma = torch.exp(log_sigma_t)
    offsets = torch.linspace(-2.0, 2.0, int(n_samples), device=means.device, dtype=means.dtype)
    # A translating row (omega below the gauge threshold) must not gain any
    # rotation: past |omega|^2 = gauge_eps the model keeps only the translation
    # along that (arbitrary) axis and the path jumps. Its curve is fitted with
    # the translation controls alone.
    rotating = ((omega_base * omega_base).sum(dim=-1, keepdim=True) > gauge_eps).to(means.dtype)
    with torch.enable_grad():
        params = {k: v.clone().requires_grad_(True) for k, v in ctrl.items()}
        opt = torch.optim.Adam(list(params.values()), lr=lr)
        for _ in range(int(iters)):
            opt.zero_grad(set_to_none=True)
            loss = means.new_zeros(())
            used = {k: (v * rotating if k.endswith("_r") else v) for k, v in params.items()}
            for o in offsets:
                dt = o * sigma
                x = screw_deform_means(
                    means, anchors, mu_t, log_sigma_t, nu_base, omega_base, used, mu_t + dt,
                    gauge_eps=gauge_eps, gauge_denom_clamp=gauge_denom_clamp,
                )
                target = means + v_target * dt + 0.5 * a_target * dt * dt
                err = ((x - target) ** 2).sum(dim=-1)
                loss = loss + (err.mean() if row_weight is None else (err * row_weight).sum())
            loss.backward()
            opt.step()
    return {k: (v * rotating if k.endswith("_r") else v).detach() for k, v in params.items()}


def fit_bezier_to_targets(
    means: Tensor,
    anchors: Tensor,
    mu_t: Tensor,
    log_sigma_t: Tensor,
    nu_base: Tensor,
    omega_base: Tensor,
    dt: Tensor,
    targets: Tensor,
    valid: Tensor,
    *,
    init_ctrl: dict[str, Tensor] | None = None,
    iters: int = 300,
    lr: float = 0.02,
    gauge_eps: float = 1e-8,
    gauge_denom_clamp: float = 1e-10,
    row_weight: Tensor | None = None,
) -> dict[str, Tensor]:
    """Bézier residual controls so each primitive passes through measured
    positions ``targets`` [n, S, 3] at times ``mu_t + dt`` (``dt`` [S] or
    [n, S] seconds), using only entries where ``valid`` [n, S].

    Same model and optimiser as ``fit_bezier_to_acceleration`` (base twist and
    anchors stay fixed), but against positions chained across keyframes, so a
    row's curve can be anchored over a window much wider than its own one-step
    velocity and acceleration reach. ``init_ctrl`` warm-starts the controls.
    Each row's loss is the mean over its valid samples; rows without any are
    left at ``init_ctrl``.
    """
    n = int(means.shape[0])
    ctrl = {name: (init_ctrl[name].clone() if init_ctrl is not None else
                   torch.zeros(n, 3, device=means.device, dtype=means.dtype)) for name in _CTRL_NAMES}
    if n == 0 or iters <= 0:
        return ctrl
    rotating = ((omega_base * omega_base).sum(dim=-1, keepdim=True) > gauge_eps).to(means.dtype)
    w = valid.to(means.dtype)
    per_row = w.sum(dim=1).clamp_min(1.0)
    weight = row_weight if row_weight is not None else torch.full((n,), 1.0 / n, device=means.device, dtype=means.dtype)
    targets = torch.nan_to_num(targets.to(means.dtype))
    with torch.enable_grad():
        params = {k: v.clone().requires_grad_(True) for k, v in ctrl.items()}
        opt = torch.optim.Adam(list(params.values()), lr=lr)
        for _ in range(int(iters)):
            opt.zero_grad(set_to_none=True)
            loss = means.new_zeros(())
            used = {k: (v * rotating if k.endswith("_r") else v) for k, v in params.items()}
            for s in range(int(valid.shape[1])):
                if not bool(valid[:, s].any()):
                    continue
                dt_s = dt[s] if dt.dim() == 1 else dt[:, s].unsqueeze(-1)
                x = screw_deform_means(
                    means, anchors, mu_t, log_sigma_t, nu_base, omega_base, used, mu_t + dt_s,
                    gauge_eps=gauge_eps, gauge_denom_clamp=gauge_denom_clamp,
                )
                err = ((x - targets[:, s]) ** 2).sum(dim=-1) * w[:, s] / per_row
                loss = loss + (err * weight).sum()
            loss.backward()
            opt.step()
    return {k: (v * rotating if k.endswith("_r") else v).detach() for k, v in params.items()}
