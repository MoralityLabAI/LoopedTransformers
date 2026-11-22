"""Physical-unit cross-pass surrogates and signed graph reparameterization."""
from experiments.nan_delay39.graph_host import graph_cells


def orthonormal_bank(writes, np):
    eigenvalues, axes = np.linalg.eigh(writes @ writes.T)
    if eigenvalues[0] <= eigenvalues[-1] * 1e-10:
        raise ValueError("The fixed write bank is numerically rank-deficient")
    root = (axes * np.sqrt(eigenvalues)) @ axes.T
    inverse = (axes / np.sqrt(eigenvalues)) @ axes.T
    return root, inverse, inverse @ writes


def fit(left, right, factor, np):
    gram = left.T @ left
    penalty = factor * np.trace(gram) / gram.shape[0]
    if penalty <= 0:
        raise ValueError("Calibration contains no cross-world signal")
    return np.linalg.solve(gram + penalty * np.eye(gram.shape[0]), left.T @ right)


def head_gradient(mid, direction, eps, np):
    rms = np.sqrt(np.mean(mid**2) + eps)
    return direction / rms - mid * (mid @ direction) / (len(mid) * rms**3)


def gauge_family(root, inverse, rank, rotations, seed, np):
    # Native coordinates are c_white G^-1/2. Every gauge is c_white M, M^-1 U_white.
    order = np.roll(np.eye(rank), 1, axis=1)
    scale = (-1.0)**np.arange(rank) * np.exp2(np.arange(rank) % 5 - 2)
    monomial = order * scale
    yield "native", inverse, root
    yield "reciprocal", inverse @ monomial, np.linalg.inv(monomial) @ root
    yield "orthonormal", np.eye(rank), np.eye(rank)
    rng = np.random.default_rng(seed)
    for i in range(rotations):
        q, r = np.linalg.qr(rng.normal(size=(rank, rank)))
        q *= np.sign(np.diag(r))
        yield f"rotation_{i:02d}", q, q.T


def in_gauge(white_left, white_target, white_writes, transfer, gradient, change, inverse, np):
    writes = inverse @ white_writes
    coefficients = white_left @ change
    target = white_target @ change
    transfer = inverse @ transfer @ change
    norms = np.linalg.norm(writes, axis=1)
    source = coefficients * norms
    middle = transfer * (norms[None, :] / norms[:, None])
    sink = (writes @ gradient) / norms
    graph = graph_cells(source.tolist(), middle.tolist(), sink.tolist())
    return coefficients @ writes, target @ writes, (coefficients @ transfer) @ writes, graph
