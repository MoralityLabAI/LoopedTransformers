"""Finite product accounting for the retained-native component edit."""
from experiments.nan_carry63.algebra import round_bf16


TERMS = ("state_contrast", "world_gate_difference", "sphere", "rounding")


def route(x, native, gate, v, u, mean, sphere_target, eps, np):
    centered = x @ v - mean
    g = gate - 1.
    adjustment = (g * centered) @ u
    before = native + adjustment
    after = before
    if sphere_target is not None:
        power = np.mean(before * before, axis=1, keepdims=True)
        rms = np.sqrt(power + eps)
        fraction = -np.mean(g, axis=1, keepdims=True)
        desired = (1 - fraction) * rms + fraction * max(sphere_target, eps)
        after = before * np.where(power > eps, desired/rms, 1.)
    if not np.count_nonzero(g):
        # The native kernel returns the captured BF16 bytes for all-one.
        after = native
    rounded = round_bf16(after, np)
    diff = lambda y: y[0] - y[1]
    terms = (
        (g.mean(axis=0) * diff(centered)) @ u,
        (diff(g) * centered.mean(axis=0)) @ u,
        diff(after - before),
        diff(rounded - after),
    )
    np.testing.assert_allclose(terms[0] + terms[1], diff(adjustment), atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(sum(terms), diff(rounded - native), atol=1e-11, rtol=1e-11)
    return rounded, dict(zip(TERMS, terms, strict=True))


def verify(np):
    rng = np.random.default_rng(73023)
    x = round_bf16(rng.normal(size=(2, 6)), np)
    native = round_bf16(rng.normal(size=(2, 10)), np).astype(np.float64)
    v, u = rng.normal(size=(6, 4)), rng.normal(size=(4, 10))
    mean = rng.normal(size=4)
    gate = rng.uniform(size=(2, 4))
    p = np.array([3, 0, 2, 1])
    scale = np.array([.25, 4., .5, 8.])
    for sphere in (None, .8):
        result, terms = route(x, native, gate, v, u, mean, sphere, 1e-6, np)
        other, shifted = route(x, native, gate[:, p], v[:, p] * scale, u[p] / scale[:, None],
                               mean[p] * scale, sphere, 1e-6, np)
        np.testing.assert_array_equal(result, other)
        for name in TERMS:
            np.testing.assert_allclose(terms[name], shifted[name], atol=1e-12, rtol=1e-12)
        noop, noop_terms = route(x, native, np.ones_like(gate), v, u, mean, sphere, 1e-6, np)
        np.testing.assert_array_equal(noop, native)
        assert all(not np.count_nonzero(term) for term in noop_terms.values())
        if sphere is not None:
            assert np.count_nonzero(terms["sphere"])
        assert np.count_nonzero(terms["world_gate_difference"])
        _, control = route(x, native, np.broadcast_to(gate[0], gate.shape), v, u, mean, sphere, 1e-6, np)
        assert not np.count_nonzero(control["world_gate_difference"])
    return True
