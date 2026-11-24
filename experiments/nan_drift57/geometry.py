"""Continuous atlas RMS refinement and rounded pairwise head statistics."""
from experiments.nan_dark44.head import summarize_blocks
from experiments.nan_readpath37.run import bf16, decode_bf16


def refine(matrix, initial, np):
    data = matrix.astype(np.float64)
    gram = (data.T @ data) / len(data)
    del data
    eig, axes = np.linalg.eigh(gram)
    if eig[0] < -1e-12:
        raise ValueError("The vocabulary Gram is not positive semidefinite")
    q = initial.astype(np.float64)
    q /= np.linalg.norm(q)
    limit = float(abs(matrix @ initial).max())
    score = start = float(q @ gram @ q)
    least = axes[:, 0]
    if float(least @ q) < 0.:
        least = -least

    def choose(proposal, current, energy):
        proposal = proposal / np.linalg.norm(proposal)
        value = float(proposal @ gram @ proposal)
        if value < energy and float(abs(matrix @ proposal.astype(np.float32)).max()) <= limit:
            return proposal, value
        return current, energy

    origin = q.copy()
    for weight in (.0625, .125, .25, .5, .75, 1.):
        q, score = choose((1-weight)*origin + weight*least, q, score)
    for step in (.125, .0625, .03125, .015625):
        for _ in range(6):
            slope = gram @ q - score * q
            size = np.linalg.norm(slope)
            if size < 1e-12:
                break
            reached, new = choose(q - step / size * slope, q, score)
            if new >= score:
                break
            q, score = reached, new
    chosen = q.astype(np.float32) if score < start else initial.copy()
    return chosen, {
        "continuous_minimum_rms_in_atlas": float(np.sqrt(max(0., eig[0]))),
        "peak_recipe_continuous_rms": float(np.sqrt(start)),
        "refined_continuous_rms": float(np.sqrt(score)),
        "peak_recipe_continuous_peak": limit,
        "refined_continuous_peak": float(abs(matrix @ chosen).max()),
    }


def pair_moments(z, pairs, np):
    clean, active, sham = pairs.T
    a = z[active].astype(np.float64) - z[clean]
    b = z[sham].astype(np.float64) - z[clean]
    return np.column_stack(((a*a).sum(1), (b*b).sum(1), (a*b).sum(1), ((a-b)**2).sum(1)))


def pair_head(head, states, refs, labels, pairs, rows, check, np):
    hidden = head.norm(states).astype(np.float64)
    stride = 2*head.width
    moments = np.zeros((len(pairs), 4))

    def blocks():
        head.stream.seek(head.base + head.weight_offset)
        for start in range(0, head.vocab, rows):
            check()
            n = min(rows, head.vocab-start)
            payload = head.stream.read(n*stride)
            if len(payload) != n*stride:
                raise ValueError("The pinned head block is incomplete")
            weights = decode_bf16(payload, np).reshape(n, head.width).astype(np.float64)
            z = bf16(hidden @ weights.T, np)
            moments[:] += pair_moments(z, pairs, np)
            yield z

    live = summarize_blocks(blocks(), len(states), refs, labels, np)
    if live["vocab"] != head.vocab:
        raise ValueError("The readout missed vocabulary rows")
    info = []
    for aa, bb, ab, dd in moments:
        scale = np.sqrt(aa*bb)
        ar, br, distance = np.sqrt(np.asarray([aa, bb, dd]) / head.vocab)
        info.append({"active_rms": float(ar), "sham_rms": float(br), "signed_rms_gap": float(br-ar),
                     "head_shift_cosine": float(np.clip(ab/scale, -1, 1)) if scale else None,
                     "head_shift_rms_distance": float(distance)})
    return live, info


def verify(np):
    m = np.array([[1., 0.], [0., .25]], dtype=np.float32)
    _, info = refine(m, np.array([1., 0.], dtype=np.float32), np)
    assert info["continuous_minimum_rms_in_atlas"] == np.sqrt(.03125)
    assert info["refined_continuous_rms"] == info["continuous_minimum_rms_in_atlas"]
    assert info["refined_continuous_peak"] == .25
    sample = np.array([[1., 2., 3.], [2., 1., 3.], [2., 3., 2.]])
    np.testing.assert_array_equal(pair_moments(sample, np.array([[0, 1, 2]]), np), [[2., 3., 0., 5.]])
