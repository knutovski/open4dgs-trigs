"""
Temporal chunk residency for long-sequence training.

This is the LongVolCap-style training-side scaling mechanism: the full model
(all row-aligned parameters) lives in CPU RAM as a plain-tensor "master".
For each temporal chunk, only the rows whose influence intervals intersect
the chunk window — plus static rows — are materialized on the GPU as a
compact ParameterDict. Training (including its optimizer state, gradients,
and autograd graph) is then sized to the chunk, not the sequence. After the
chunk, updated rows are scattered back into the CPU master exactly.

GPU memory per chunk therefore scales with the number of chunk-active rows
rather than the total row count, which is what allows sequence length (and
with it total Gaussian count) to grow past single-GPU limits.

Constraints in this version, enforced by the trainer wiring:
  * row identity must be preserved within a chunk: relocation is fine (it
    rewrites row values in place), delayed pruning is not (it removes rows
    and breaks the sub->master index map);
  * optimizer state is not carried across chunk boundaries (fresh Adam per
    chunk). Dynamic rows have a single writer per round and only owner rows
    are scattered back, so overlap is rendering/supervision context rather
    than a parameter-stitching mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import math

import torch
from trigs.compat.device import f64  # macOS port
from torch import Tensor, nn


def chunk_windows(
    t_max: float,
    chunk_len_s: float,
    overlap_s: float = 0.0,
) -> List[Tuple[float, float]]:
    """Cover ``[0, t_max]`` without creating an oversized final overlap.

    Full windows advance by ``chunk_len_s - overlap_s``.  When the remaining
    tail is shorter than ``chunk_len_s``, the final window is *shortened* and
    starts at the next regular stride position.  The previous implementation
    shifted a full-length final window backwards to ``t_max`` while retaining
    the preceding window; for an 11.65 s sequence with 4.0/0.5 s chunks that
    produced a 3.35 s final overlap instead of 0.5 s.
    """
    t_max = float(max(t_max, 0.0))
    chunk_len_s = float(chunk_len_s)
    if chunk_len_s <= 0:
        raise ValueError("chunk_len_s must be > 0")
    overlap_s = min(max(float(overlap_s), 0.0), 0.9 * chunk_len_s)

    if t_max <= 0.0 or chunk_len_s >= t_max:
        return [(0.0, t_max)]

    stride = chunk_len_s - overlap_s
    windows: List[Tuple[float, float]] = []
    lo = 0.0
    eps = 1e-9
    while lo < t_max - eps:
        hi = min(lo + chunk_len_s, t_max)
        windows.append((lo, hi))
        if hi >= t_max - eps:
            break
        lo += stride

    # Numerical defence: every adjacent pair should overlap by no more than
    # the requested amount (up to floating-point error), and coverage must be
    # monotonic with no duplicate final window.
    deduped: List[Tuple[float, float]] = []
    for lo, hi in windows:
        if deduped and abs(lo - deduped[-1][0]) < eps and abs(hi - deduped[-1][1]) < eps:
            continue
        deduped.append((float(lo), float(hi)))
    return deduped


def temporal_radius_scale(opacity_threshold: float) -> float:
    """Return ``k`` for the Gaussian support radius ``r = k * sigma``."""
    threshold = min(max(float(opacity_threshold), 1e-8), 1.0 - 1e-8)
    return math.sqrt(-2.0 * math.log(threshold))


def allocate_integer_steps(
    total_steps: int,
    weights: Sequence[float],
    *,
    min_each: int = 1,
) -> List[int]:
    """Allocate an exact integer step budget by largest remainder.

    This prevents a short final chunk from receiving the same number of steps
    as a full chunk.  ``min_each`` is honored when the budget permits it.
    """
    n = len(weights)
    if n == 0:
        return []
    total_steps = max(int(total_steps), 0)
    min_each = max(int(min_each), 0)
    base_each = min_each if total_steps >= n * min_each else 0
    remaining = total_steps - n * base_each

    clean = [max(float(w), 0.0) for w in weights]
    weight_sum = sum(clean)
    if weight_sum <= 0.0:
        clean = [1.0] * n
        weight_sum = float(n)

    raw = [remaining * w / weight_sum for w in clean]
    floors = [int(math.floor(x)) for x in raw]
    out = [base_each + x for x in floors]
    leftover = total_steps - sum(out)
    order = sorted(
        range(n),
        key=lambda i: (raw[i] - floors[i], clean[i], -i),
        reverse=True,
    )
    for i in order[:leftover]:
        out[i] += 1
    return out


@torch.no_grad()
def chunk_row_idx(
    times: Tensor,
    log_durations: Tensor,
    t_lo: float,
    t_hi: float,
    *,
    opacity_threshold: float = 0.05,
    extra_margin_s: float = 0.0,
    is_static: Optional[Tensor] = None,
    include_static: bool = True,
) -> Tensor:
    """
    Rows whose influence interval [mu - r, mu + r] intersects [t_lo, t_hi],
    plus static rows. Mirrors the active-query radius definition so a row
    active at any t inside the window is guaranteed to be selected.
    """
    scale = temporal_radius_scale(opacity_threshold)

    mu = times.detach().reshape(-1)
    sigma = log_durations.detach().exp().reshape(-1).clamp_min(1e-6)
    radius = sigma * scale + float(extra_margin_s)

    selected = (mu - radius <= float(t_hi)) & (mu + radius >= float(t_lo))

    if include_static and is_static is not None:
        selected |= is_static.detach().reshape(-1) > 0.5

    return torch.nonzero(selected, as_tuple=False).reshape(-1)


@dataclass
class ChunkResidency:
    """
    CPU-resident master model with exact GPU chunk materialization.

    Row-aligned tensors (first dim == n_rows) are gathered/scattered by row
    index. Non-row-aligned tensors (per-camera offsets, gains) are carried
    whole so chunk-to-chunk learning on them accumulates.
    """

    master: Dict[str, Tensor] = field(default_factory=dict)
    requires_grad: Dict[str, bool] = field(default_factory=dict)
    row_aligned: Dict[str, bool] = field(default_factory=dict)
    n_rows: int = 0

    @classmethod
    @torch.no_grad()
    def from_params(cls, params) -> "ChunkResidency":
        n_rows = int(params["means"].shape[0])
        master: Dict[str, Tensor] = {}
        requires_grad: Dict[str, bool] = {}
        row_aligned: Dict[str, bool] = {}
        for name in list(params.keys()):
            p = params[name]
            master[name] = p.detach().to("cpu").clone()
            requires_grad[name] = bool(p.requires_grad)
            row_aligned[name] = (
                p.dim() >= 1 and int(p.shape[0]) == n_rows
            )
        return cls(
            master=master,
            requires_grad=requires_grad,
            row_aligned=row_aligned,
            n_rows=n_rows,
        )

    @torch.no_grad()
    def materialize(self, idx: Tensor, device) -> "nn.ParameterDict":
        """Compact GPU ParameterDict holding only rows ``idx`` (plus every
        non-row-aligned tensor in full)."""
        idx = idx.detach().reshape(-1).to("cpu", dtype=torch.long)
        if int(idx.numel()) > 0:
            if int(idx.min()) < 0 or int(idx.max()) >= self.n_rows:
                raise IndexError("materialize received an out-of-range row")
        sub = nn.ParameterDict()
        for name, tensor in self.master.items():
            if self.row_aligned[name]:
                data = tensor.index_select(0, idx).to(device)
            else:
                data = tensor.clone().to(device)
            sub[name] = nn.Parameter(
                data, requires_grad=self.requires_grad[name],
            )
        return sub

    @torch.no_grad()
    def absorb(
        self,
        sub_params,
        idx: Tensor,
        *,
        row_write_mask: Optional[Tensor] = None,
    ) -> None:
        """Scatter a trained chunk back into the CPU master.

        ``row_write_mask`` restricts row-aligned writes to the rows owned by
        this chunk.  Frozen context rows are therefore never copied back over
        the authoritative values produced by their owner chunk, eliminating
        the old last-writer-wins seam mechanism.  Non-row-aligned parameters
        (camera offsets/appearance) are still copied in full.
        """
        idx = idx.detach().reshape(-1).to("cpu", dtype=torch.long)
        n_sub_expected = int(idx.numel())
        if row_write_mask is None:
            write_mask = torch.ones(n_sub_expected, dtype=torch.bool)
        else:
            write_mask = row_write_mask.detach().reshape(-1).to("cpu", dtype=torch.bool)
            if int(write_mask.numel()) != n_sub_expected:
                raise RuntimeError(
                    "absorb: row_write_mask length does not match materialized rows"
                )
        write_idx = idx[write_mask]

        for name, tensor in self.master.items():
            if name not in sub_params:
                continue
            value = sub_params[name].detach().to("cpu")
            if self.row_aligned[name]:
                if int(value.shape[0]) != n_sub_expected:
                    raise RuntimeError(
                        f"absorb: '{name}' has {int(value.shape[0])} rows, "
                        f"expected {n_sub_expected}. Row identity was not "
                        "preserved (was pruning enabled inside the chunk?)."
                    )
                if int(write_idx.numel()) > 0:
                    tensor[write_idx] = value[write_mask].to(tensor.dtype)
            else:
                if tuple(value.shape) != tuple(tensor.shape):
                    raise RuntimeError(
                        f"absorb: '{name}' shape changed "
                        f"{tuple(tensor.shape)} -> {tuple(value.shape)}"
                    )
                tensor.copy_(value.to(tensor.dtype))

    @torch.no_grad()
    def full_index(self) -> Tensor:
        return torch.arange(self.n_rows, dtype=torch.long)


class OffsetTBWriter:
    """Shift `step` in add_* calls so per-chunk logging shares one global
    step axis instead of restarting at zero each chunk."""

    _STEPPED = (
        "add_scalar", "add_scalars", "add_image", "add_images",
        "add_histogram", "add_text", "add_figure",
    )

    def __init__(self, writer, offset: int):
        self._writer = writer
        self._offset = int(offset)

    def __getattr__(self, name):
        attr = getattr(self._writer, name)
        if name in self._STEPPED and callable(attr):
            offset = self._offset

            def _wrapped(*args, **kwargs):
                if "global_step" in kwargs and kwargs["global_step"] is not None:
                    kwargs["global_step"] = int(kwargs["global_step"]) + offset
                    return attr(*args, **kwargs)
                if len(args) >= 3 and isinstance(args[2], (int, float)):
                    args = (*args[:2], int(args[2]) + offset, *args[3:])
                return attr(*args, **kwargs)

            return _wrapped
        return attr

@torch.no_grad()
def window_leak_stats(
    params: Mapping[str, Tensor],
    windows: Sequence[Tuple[float, float]],
    t_max: float,
    fps: float,
    *,
    cull_threshold: float,
    alive_opacity: float = 0.005,
    extra_margin_s: float = 0.0,
) -> Dict[str, float]:
    """How far live dynamic rows reach outside the window that owns them.

    Ownership is the window whose interior holds mu_t (overlap midpoints as boundaries), as in
    chunked training. A row "leaks" when its cull-support [mu - r, mu + r] (clipped to the clip)
    extends past its owner window: in a chunked visit those frames are not rendered, so nothing
    in that visit supervises what the row does there. ``mass_outside`` is the opacity-weighted
    share of temporal mass (Gaussian in time, clipped to the clip) outside the owner window.
    Works on any checkpoint, chunked or not, so a chunked run can be compared with an unchunked
    one on the same window layout."""
    mu = params["times"].detach().reshape(-1).cpu().double()
    sigma = params["log_durations"].detach().reshape(-1).cpu().double().exp().clamp_min(1e-6)
    alpha = torch.sigmoid(params["opacities"].detach().reshape(-1).cpu().double())
    keep = alpha >= float(alive_opacity)
    if "is_static" in params:
        keep &= params["is_static"].detach().reshape(-1).cpu() <= 0.5
    mu, sigma, alpha = mu[keep], sigma[keep], alpha[keep]
    n = int(mu.numel())
    if n == 0 or not windows:
        return {"rows": 0.0, "leak_frac": 0.0, "overshoot_p50_frames": 0.0,
                "overshoot_p99_frames": 0.0, "mass_outside": 0.0}
    bounds = []
    for (a_lo, a_hi), (b_lo, b_hi) in zip(windows, windows[1:]):
        o_lo, o_hi = max(a_lo, b_lo), min(a_hi, b_hi)
        bounds.append(0.5 * (o_lo + o_hi) if o_hi >= o_lo else 0.5 * (a_hi + b_lo))
    owner = (torch.bucketize(mu, torch.tensor(bounds, dtype=mu.dtype), right=True)
             if bounds else torch.zeros_like(mu, dtype=torch.long))
    w_lo = torch.tensor([w[0] for w in windows], dtype=mu.dtype)[owner]
    w_hi = torch.tensor([w[1] for w in windows], dtype=mu.dtype)[owner]
    radius = sigma * temporal_radius_scale(cull_threshold) + float(extra_margin_s)
    lo = (mu - radius).clamp(min=0.0)
    hi = (mu + radius).clamp(max=float(t_max))
    overshoot = torch.maximum(w_lo - lo, hi - w_hi).clamp(min=0.0) * float(fps)
    leaking = overshoot > 0.5                                    # more than half a frame
    cdf = lambda x: 0.5 * (1.0 + torch.erf((x - mu) / (math.sqrt(2.0) * sigma)))
    in_clip = (cdf(torch.full_like(mu, float(t_max))) - cdf(torch.zeros_like(mu))).clamp_min(1e-12)
    in_win = (cdf(w_hi) - cdf(w_lo)).clamp(min=0.0)
    outside = (1.0 - in_win / in_clip).clamp(0.0, 1.0)
    q = (torch.quantile(overshoot[leaking].float(), torch.tensor([0.5, 0.99])).tolist()
         if bool(leaking.any()) else [0.0, 0.0])
    return {
        "rows": float(n),
        "leak_frac": float(f64(leaking).mean()),
        "overshoot_p50_frames": float(q[0]),
        "overshoot_p99_frames": float(q[1]),
        "mass_outside": float((alpha * outside).sum() / alpha.sum().clamp_min(1e-12)),
    }
