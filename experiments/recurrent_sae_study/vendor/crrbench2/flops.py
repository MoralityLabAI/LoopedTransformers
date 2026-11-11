"""Analytic inference-FLOP accounting (multiply-add = 2 FLOPs)."""


def loop_flops(T, d, mlp_ratio=4):
    """One pass of the weight-tied block over T tokens."""
    proj = 2 * T * d * d * 4          # q, k, v, o projections
    attn = 2 * 2 * T * T * d          # QK^T and AV
    mlp = 2 * 2 * T * d * mlp_ratio * d
    return proj + attn + mlp


def readout_flops(d, C):
    return 2 * d * C + 5 * d


def controller_flops(d_in, hidden=128, n_actions=3):
    return 2 * (d_in * hidden + hidden * hidden + hidden * n_actions)
