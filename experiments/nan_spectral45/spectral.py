"""Small exact vocabulary Gram matrices for a constrained physical write span."""
from experiments.nan_readpath37.run import decode_bf16


def projected_gram(head, vectors, rows_per_chunk, check, np):
    """Compute (W diag(gamma) vectors)^T (W diag(gamma) vectors) / |vocab|."""
    vectors = vectors * head.gamma.astype(np.float64)[:, None]
    gram = np.zeros((vectors.shape[1], vectors.shape[1]))
    stride = 2 * head.width
    head.stream.seek(head.base + head.weight_offset)
    for start in range(0, head.vocab, rows_per_chunk):
        check()
        rows = min(rows_per_chunk, head.vocab - start)
        raw = head.stream.read(rows * stride)
        if len(raw) != rows * stride:
            raise ValueError("Short head block in Gram calculation")
        weights = decode_bf16(raw, np).reshape(rows, head.width).astype(np.float64)
        projected = weights @ vectors
        gram += projected.T @ projected
    return gram / head.vocab


def norm_map(h, basis, eps, np):
    """Coordinates of D RMSNorm(h) as a map into the joined columns [B, h]."""
    rms = np.sqrt(np.mean(h**2) + eps)
    return np.vstack((np.eye(basis.shape[1]) / rms, -(h @ basis)[None, :] / (len(h) * rms**3)))


def dirs(gram, basis, h, margin_gradient, raw_donor, penalty_factor, eps, np):
    push = norm_map(h, basis, eps, np)
    physical_gram = push.T @ gram @ push
    normal = basis.T @ margin_gradient
    if np.linalg.norm(normal) < 1e-12 or np.linalg.norm(raw_donor) < 1e-12:
        raise ValueError("Missing task observer or donor direction in the frozen span")
    _, _, vt = np.linalg.svd(normal.reshape(1, -1), full_matrices=True)
    null = vt[1:].T
    g = null.T @ physical_gram @ null
    eigenvalues, eigenvectors = np.linalg.eigh(g)
    if eigenvalues[0] <= 0 or eigenvalues[1] - eigenvalues[0] <= 1e-12 * eigenvalues[-1]:
        raise ValueError("The audited local minimum direction is numerically unidentified")
    u = null.T @ (basis.T @ raw_donor)
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        raise ValueError("The donor has no direction in the current margin null")
    u /= norm
    least = eigenvectors[:, 0]
    if least @ u < 0:
        least = -least
    lam = penalty_factor * float(np.median(eigenvalues))
    donor = np.linalg.solve(g + lam * np.eye(len(g)), u)
    donor /= np.linalg.norm(donor)
    vectors = {
        "margin_null_donor": u,
        "least_sensitive_null": least,
        "regularized_null_donor": donor,
    }
    physical = {name: (basis @ null) @ v for name, v in vectors.items()}
    predicted = {name: float(np.sqrt(v @ g @ v)) for name, v in vectors.items()}
    return physical, predicted, eigenvalues


def verify(np):
    rng = np.random.default_rng(45)
    n, width, rank = 127, 13, 7
    weights = rng.normal(size=(n, width))
    gamma = rng.uniform(.6, 1.4, size=width)
    h = rng.normal(size=width)
    basis = np.linalg.qr(rng.normal(size=(width, rank)), mode="reduced")[0]
    eps = 1e-6
    design = np.column_stack((basis, h))
    z = weights @ (gamma[:, None] * design)
    compact = z.T @ z / n
    derivative = z @ norm_map(h, basis, eps, np)
    step = 1e-5

    def continuous(states):
        scaled = states * gamma / np.sqrt(np.mean(states**2, axis=1, keepdims=True) + eps)
        return scaled @ weights.T

    dense = (continuous(h + step * basis.T) - continuous(h - step * basis.T)).T / (2 * step)
    np.testing.assert_allclose(derivative, dense, rtol=1e-7, atol=1e-9)
    h_direction = basis @ rng.normal(size=rank)
    grad = basis @ (derivative[3] - derivative[8])
    physical, sensitivity, eigenvalues = dirs(compact, basis, h, grad, h_direction, 1, eps, np)
    for direction in physical.values():
        np.testing.assert_allclose([direction @ direction, direction @ grad], [1, 0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(sensitivity["least_sensitive_null"]**2, eigenvalues[0], atol=1e-12, rtol=1e-12)
    assert sensitivity["least_sensitive_null"] < sensitivity["margin_null_donor"]
    return float(np.max(abs(derivative - dense)))
