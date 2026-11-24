"""Decompose direct BF16 residual operands under a fixed two-token readout."""
from fractions import Fraction
from math import comb, fsum

from experiments.nan_carry63.algebra import continuous, margins, round_bf16

TERMS = ("attention_input", "attention_write", "mlp_write",
         "attention_add_rounding", "mlp_add_rounding")


def shapley(values):
    """Exact rational marginal contributions for all five named groups."""
    n = len(TERMS)
    if len(values) != 2**n:
        raise ValueError("The rounded-head game requires 32 coalition margins")
    scores = [Fraction(float(value)) for value in values]
    parts = [sum((scores[mask | (1 << i)] - score) /
                 (n * comb(n - 1, mask.bit_count()))
                 for mask, score in enumerate(scores) if not (mask & (1 << i)))
             for i in range(n)]
    if sum(parts) != scores[-1] - scores[0]:
        raise ArithmeticError("Carry coalition scores failed rational Shapley closure")
    return dict(zip(TERMS, map(float, parts), strict=True))


def array(state, np):
    return np.asarray(state.detach().float().numpy(), dtype=np.float64)


def account(left, right, head, tokens, np):
    """Return the signed contrast of the right row against the left row."""
    if len(tokens) != 2 or tokens[0] == tokens[1]:
        raise ValueError("A readout contrast requires two distinct bare label tokens")
    fields = ("attention_input", "attention_write", "pre_mlp", "native_writes", "residual")
    a0, o0, c0, y0, h0 = (array(left[key], np) for key in fields)
    a1, o1, c1, y1, h1 = (array(right[key], np) for key in fields)
    operands = (a1 - a0, o1 - o0, y1 - y0,
                (c1 - (a1 + o1)) - (c0 - (a0 + o0)),
                (h1 - (c1 + y1)) - (h0 - (c0 + y0)))
    h = np.stack((h1, h0))
    np.testing.assert_allclose(h0 + sum(operands), h1, rtol=0, atol=1e-11)
    direction = head.gamma.astype(np.float64) * (head.row(tokens[0]).astype(np.float64) - head.row(tokens[1]))
    scaled, rms, continuous_swing = continuous(h, direction, head.eps, np)
    full = (1 << len(TERMS)) - 1
    states = []
    for mask in range(full + 1):
        # Anchor to the exact native endpoint on each side of the coalition.
        if mask.bit_count() <= len(TERMS) // 2:
            states.append(h0 + sum(value for i, value in enumerate(operands) if mask & (1 << i)))
        else:
            states.append(h1 - sum(value for i, value in enumerate(operands) if not (mask & (1 << i))))
    rounded = round_bf16(np.stack(states), np)
    if not np.array_equal(rounded[[full, 0]], h):
        raise ValueError("Carry coalition changed the captured endpoint")
    scores = margins(head, rounded, tokens, np)
    native_swing = float(scores[-1] - scores[0])
    linear = dict(zip(TERMS, (float(v @ scaled) for v in operands), strict=True))
    linear.update(rms_scale=rms, head_rounding=native_swing - continuous_swing)
    if abs(fsum(linear.values()) - native_swing) > 1e-10:
        raise ArithmeticError("Continuous residual and head accounting did not close")
    return {"native_swing": native_swing, "finite": shapley(scores), "continuous": linear,
            "attention_rounding_delta_l2": float(np.linalg.norm(operands[-2])),
            "mlp_rounding_delta_l2": float(np.linalg.norm(operands[-1]))}


def row(packet, phase, row_index):
    return {key: packet[key][phase, row_index]
            for key in ("attention_input", "attention_write", "pre_mlp", "native_writes", "residual")}
