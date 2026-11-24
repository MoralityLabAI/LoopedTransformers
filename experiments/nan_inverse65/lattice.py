"""Invert the native additive BF16 residual chain on its finite numeric lattice."""
from experiments.nan_readpath37.run import bf16


def lattice(np):
    positive = (np.arange(1, 0x7f80, dtype=np.uint32) << 16).view(np.float32)
    return np.concatenate((-positive[::-1], np.zeros(1, dtype=np.float32), positive))


def add(left, right, np):
    # The registered BF16 operator widens both operands to FP32 before rounding.
    return bf16(np.add(left, right, dtype=np.float32), np)


def forward(left, attention, mlp, np):
    return add(add(left, attention, np), mlp, np)


def boundary(grid, attention, mlp, target, *, upper, np):
    low = np.zeros(target.shape, dtype=np.int32)
    high = np.full(target.shape, len(grid), dtype=np.int32)
    for _ in range(len(grid).bit_length()):
        middle = (low + high) // 2
        sampled = forward(grid[np.minimum(middle, len(grid) - 1)], attention, mlp, np)
        goes_right = (sampled <= target) if upper else (sampled < target)
        goes_right &= middle < len(grid)
        low = np.where(goes_right, middle + 1, low)
        high = np.where(goes_right, high, middle)
    if not np.array_equal(low, high):
        raise ArithmeticError("BF16 lower-bound search did not converge")
    return low


def invert(grid, target, attention, mlp, np):
    """Inclusive numeric input indices and attainable pre-MLP residual endpoints."""
    if (target.shape != attention.shape or target.shape != mlp.shape
            or any(x.dtype != np.float32 or not np.isfinite(x).all()
                   or not np.array_equal(bf16(x, np), x) for x in (target, attention, mlp))):
        raise ValueError("The inverse requires three same-shaped finite BF16 arrays")
    with np.errstate(over="ignore"):
        lo = boundary(grid, attention, mlp, target, upper=False, np=np)
        hi = boundary(grid, attention, mlp, target, upper=True, np=np) - 1
        possible = lo <= hi
        left = grid[np.minimum(lo, len(grid) - 1)]
        right = grid[np.maximum(hi, 0)]
        end_left = forward(left, attention, mlp, np)
        end_right = forward(right, attention, mlp, np)
        if (not np.array_equal(end_left == target, possible)
                or not np.array_equal(end_right == target, possible)):
            raise ArithmeticError("Inverted endpoints disagree with their forward evaluations")
        prev = forward(grid[np.maximum(lo - 1, 0)], attention, mlp, np)
        after = forward(grid[np.minimum(hi + 1, len(grid) - 1)], attention, mlp, np)
        if ((lo > 0) & (prev >= target)).any() or ((hi < len(grid) - 1) & (after <= target)).any():
            raise ArithmeticError("An inverse interval boundary is not minimal")
        return lo, hi, add(left, attention, np), add(right, attention, np)


def test(np):
    grid = lattice(np)
    assert len(grid) == 65279 and np.all(grid[1:] > grid[:-1])
    # Compare binary search with total enumeration, including extreme decoders,
    # cancellation, subnormals, many-to-one cells and unreachable outputs.
    max_finite = grid[-1]
    min_positive = grid[len(grid) // 2 + 1]
    examples = [(0., 1., 2.), (1., -1., 0.), (.5, 1., 1.5),
                (1., -2., min_positive), (max_finite, 0., max_finite),
                (-max_finite, 0., -max_finite), (min_positive, 0., 0.),
                (8., -4., 2.), (0., 128., 1.), (3., 256., -256.)]
    for target, attention, mlp in examples:
        target, attention, mlp = (np.asarray([x], dtype=np.float32) for x in (target, attention, mlp))
        lo, hi, carry_lo, carry_hi = invert(grid, target, attention, mlp, np)
        with np.errstate(over="ignore"):
            exhaustive = np.flatnonzero(forward(grid, attention, mlp, np) == target[0])
        assert np.array_equal(np.arange(lo[0], hi[0] + 1), exhaustive)
        if exhaustive.size:
            endpoints = add(grid[exhaustive[[0, -1]]], attention, np)
            np.testing.assert_array_equal([carry_lo[0], carry_hi[0]], endpoints)
    return len(examples)
