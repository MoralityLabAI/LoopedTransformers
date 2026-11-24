"""Fine BF16 support, physical span projection, and rounded edit diagnostics."""
from hashlib import sha256
from math import sqrt

from experiments.nan_lattice53.run import next_neighbors


def basis(vectors, np):
    u, values, _ = np.linalg.svd(vectors, full_matrices=False)
    return u[:, values > 1e-10 * values[0]] if values.size else u


def remove(vector, normals, np):
    orthogonal = basis(normals, np)
    return vector - orthogonal @ (orthogonal.T @ vector)


def candidate(h, writes, raw, gradient, reached, radius, cap, method, seed, np):
    below, above = next_neighbors(h.astype(np.float32), np)
    gap = np.maximum(h - below.astype(np.float64), above.astype(np.float64) - h)
    support = gap <= cap * radius / sqrt(len(h))
    count = int(support.sum())
    if count < 5:
        return None, count
    fixed = np.column_stack((raw, gradient, *reached))[support]
    length = np.linalg.norm(fixed, axis=0)
    fixed = fixed[:, length > 1e-12] / length[length > 1e-12]
    drive = seed[support]
    if method == "screened_bpd":
        span = basis(writes[:, support].T, np)
        unit = span @ remove(span.T @ drive, span.T @ fixed, np)
    elif method == "ambient":
        unit = remove(drive, fixed, np)
    else:
        raise ValueError("Unknown sham family")
    norm = float(np.linalg.norm(unit))
    if norm < 1e-10:
        return None, count
    full = np.zeros_like(h)
    full[support] = unit / norm
    if fixed.size and np.max(abs(fixed.T @ full[support])) > 1e-9:
        raise ValueError("Physical orthogonality did not close")
    return full, count


def drive(tag, namespace, width, np):
    key = sha256(f"{namespace}|{tag}".encode()).digest()
    return np.random.default_rng(int.from_bytes(key[:8], "big")).normal(size=width)


def span_fraction(values, projection, np):
    power = float(values @ values)
    if not power:
        return None
    return float(np.clip(np.linalg.norm(projection.T @ values)**2 / power, 0., 1.))


def verify(cfg, np):
    from experiments.nan_delay39.instrument import prepare

    rng = np.random.default_rng(56)
    h = (rng.integers(20, 150, size=256) / 256).astype(np.float64)
    h[0] = 256.
    writes = rng.normal(size=(8, len(h)))
    raw, normal = rng.normal(size=(2, len(h)))
    first, second = rng.normal(size=(2, len(h)))
    raw[0] = 100.
    vector = drive("fixture", cfg["sham_namespace"], len(h), np)
    radius = float(np.linalg.norm(h)) * .0075
    left, count = candidate(h, writes, raw, normal, (first, second), radius, .25, "screened_bpd", vector, np)
    transform = np.linalg.qr(rng.normal(size=(8, 8)))[0] @ np.diag(np.linspace(.4, 1.8, 8))
    right, other = candidate(h, transform @ writes, raw, normal, (first, second), radius, .25,
                             "screened_bpd", vector, np)
    assert count == other and count < len(h) and left[0] == 0.
    np.testing.assert_allclose(left, right, atol=1e-11, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(left), 1., atol=1e-12, rtol=0)
    assert all(prepare(h, sign * left, .0075, .03, .98, np).eligible for sign in (1, -1))
