"""MOLT-08: real-arithmetic observer-error bounds for the strict delay chain.

The interval operations use ordinary float64, not directed rounding. Their
receipts are conditional mathematical diagnostics, never formal certificates.
"""
from __future__ import annotations

import math

import torch

from experiments.molt_goblin_v1.observability_gym import CUT, LOOPS, labels

PAD = 1e-9


def linear_box(weight, lower, upper, bias=None):
    """Coordinate interval image of a linear map, in real arithmetic."""
    pos, neg = weight.clamp_min(0), weight.clamp_max(0)
    lo = pos @ lower + neg @ upper
    hi = pos @ upper + neg @ lower
    if bias is not None:
        lo, hi = lo + bias, hi + bias
    return lo, hi


def multiply_box(al, ah, bl, bh):
    products = torch.stack((al * bl, al * bh, ah * bl, ah * bh))
    return products.amin(0), products.amax(0)


def tanh_derivative_box(lower, upper):
    near = torch.where((lower <= 0) & (upper >= 0), torch.zeros_like(lower),
                       torch.minimum(lower.abs(), upper.abs()))
    far = torch.maximum(lower.abs(), upper.abs())
    return 1 - torch.tanh(far).square(), 1 - torch.tanh(near).square()


def cone_heads(model, x, position):
    """Exactly the same affected readout rows as observe(...)[cone]."""
    for visit in range(CUT + 1, LOOPS + 1):
        j = position + visit - CUT
        if j >= len(x):
            break
        _, mask = labels(x, visit)
        yield j, model.head.weight[mask[j]]


@torch.no_grad()
def global_curvature(model, x, position):
    """Uniform Hessian norm upper bound for the stacked cone observer.

    For F(h)=tanh(Wh+b), ||DF||<=||W|| and
    ||D²F||<=4/(3 sqrt(3)) ||W||² in Euclidean norms.
    Repeated composition uses the chain rule for bilinear derivatives.
    Spectral norms are numerical estimates, not interval-rounded values.
    """
    w = float(torch.linalg.matrix_norm(model.recurrent.weight, ord=2))
    lip, hess, squares = 1.0, 0.0, 0.0
    for _, q in cone_heads(model, x, position):
        hess = w * hess + (4 / (3 * math.sqrt(3))) * w**2 * lip**2
        lip *= w
        if len(q):
            squares += (float(torch.linalg.matrix_norm(q, ord=2)) * hess)**2
    return math.sqrt(squares)


def spectral_segment_bound(tail, residual, radius, curvature):
    """||F(z+d)-F(z+p)|| <= (tail+M*a)||d-p|| for p=P_k d.

    Assumes an upper bound on the singular tail and a uniform Hessian bound
    M on the radius-a support ball. Both endpoints lie in that ball.
    """
    values = (tail, residual, radius, curvature)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("finite nonnegative bounds required")
    return (tail + curvature * radius) * residual


@torch.no_grad()
def directional_interval(model, x, base, position, projected, full, segments):
    """Enclose F(z+full)-F(z+projected) by integrating a derivative box.

    The complete segment is available at map ingestion. After discarding
    the original displacement the bound cannot be recalculated from the
    compressed coefficients alone. Strict predecessor-chain semantics
    are required; this routine must not be used on a general attention block.
    """
    if type(segments) is not int or not 1 <= segments <= 64:
        raise ValueError("segment count outside development bounds")
    if (base.ndim != 2 or x.ndim != 2 or len(base) != len(x)
            or not 0 <= position < len(x)
            or full.shape != base[position].shape or projected.shape != full.shape):
        raise ValueError("invalid intervention chart")
    if any(t.dtype != torch.float64 or not bool(torch.isfinite(t).all())
           for t in (x, base, projected, full)):
        raise ValueError("finite float64 fixture required")
    w = model.recurrent.weight
    difference = full - projected
    start = base[position] + projected
    nodes = [(j, model.input(x[j]), q) for j, q in cone_heads(model, x, position)]
    if not nodes or not any(len(q) for _, _, q in nodes):
        raise ValueError("empty readout cone")
    lower, upper = [], []
    for i in range(segments):
        a = start + (i / segments) * difference
        b = start + ((i + 1) / segments) * difference
        lo, hi = torch.minimum(a, b), torch.maximum(a, b)
        dl, dh = difference, difference
        lows, highs = [], []
        for _, fixed, q in nodes:
            zl, zh = linear_box(w, lo, hi, fixed)
            tl, th = linear_box(w, dl, dh)
            sl, sh = tanh_derivative_box(zl, zh)
            dl, dh = multiply_box(sl, sh, tl, th)
            lo, hi = torch.tanh(zl), torch.tanh(zh)
            if len(q):
                yl, yh = linear_box(q, dl, dh)
                lows.append(yl)
                highs.append(yh)
        lower.append(torch.cat(lows))
        upper.append(torch.cat(highs))
    # Integrate over equal segment lengths. PAD is a disclosed numerical
    # allowance, not a proof of outward floating-point rounding.
    l = torch.stack(lower).mean(0)
    h = torch.stack(upper).mean(0)
    return l - PAD, h + PAD


def stable_signs(approx, lower, upper):
    """Conditional parity with the full-displacement signs, no target labels."""
    if approx.shape != lower.shape or approx.shape != upper.shape:
        raise ValueError("readout shape mismatch")
    if not bool((lower <= upper).all()):
        raise ValueError("inverted interval")
    if not all(bool(torch.isfinite(v).all()) for v in (approx, lower, upper)):
        raise ValueError("nonfinite sign comparison")
    return bool((((approx > PAD) & (approx + lower > PAD))
                 | ((approx < -PAD) & (approx + upper < -PAD))).all())


def stable_symmetric(approx, bound):
    if not math.isfinite(bound) or bound < 0:
        raise ValueError("invalid error bound")
    return stable_signs(approx, torch.full_like(approx, -bound), torch.full_like(approx, bound))
