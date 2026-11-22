"""Low-rank physical operators and deterministic rank-one alignment."""
from functools import cache


def atom_geometry(a, b, np):
    va, ua = a
    vb, ub = b
    read = (va / np.linalg.norm(va, axis=0)).T @ (vb / np.linalg.norm(vb, axis=0))
    write = (ua / np.linalg.norm(ua, axis=1)[:, None]) @ (ub / np.linalg.norm(ub, axis=1)[:, None]).T
    return read, write, read * write


def assignment(score):
    rank = len(score)

    @cache
    def best(i, used):
        if i == rank:
            return 0., ()
        options = []
        for j in range(rank):
            if not (used & (1 << j)):
                value, peers = best(i + 1, used | (1 << j))
                options.append((abs(float(score[i, j])) + value, (j,) + peers))
        return min(options, key=lambda pair: (-pair[0], pair[1]))
    return best(0, 0)[1]


def cos(a, b, np):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(float(np.sum(a * b)) / den, -1., 1.)) if den > 1e-15 else None


def difference(a, b, np):
    return np.concatenate((a[0], b[0]), axis=1), np.concatenate((a[1], -b[1]), axis=0)


def joint_coordinates(a, b, np):
    ra = a[0].shape[1]
    _, p = np.linalg.qr(np.concatenate((a[0], b[0]), axis=1), mode="reduced")
    _, q = np.linalg.qr(np.concatenate((a[1].T, b[1].T), axis=1), mode="reduced")
    return p[:, :ra] @ q[:, :ra].T, p[:, ra:] @ q[:, ra:].T


def subspace_singular(a, b, np):
    l, r = (np.linalg.qr(x, mode="reduced")[0] for x in (a, b))
    return np.clip(np.linalg.svd(l.T @ r, compute_uv=False), 0, 1).tolist()


def projection_fractions(initial, delta, np):
    v, u = initial
    left = np.linalg.qr(v, mode="reduced")[0]
    right = np.linalg.qr(u.T, mode="reduced")[0]
    physical_core = (left.T @ delta[0]) @ (delta[1] @ right)
    v_unit = v / np.linalg.norm(v, axis=0)
    u_unit = u / np.linalg.norm(u, axis=1)[:, None]
    if not np.allclose(v_unit.T @ v_unit, np.eye(v.shape[1]), atol=1e-5, rtol=0) or not np.allclose(
            u_unit @ u_unit.T, np.eye(u.shape[0]), atol=1e-5, rtol=0):
        raise ValueError("Initial paired SVD rank-one atoms are not mutually orthogonal")
    paired = np.diag((v_unit.T @ delta[0]) @ (delta[1] @ u_unit.T))
    left_coordinates = np.linalg.qr(delta[0], mode="reduced")[1]
    right_coordinates = np.linalg.qr(delta[1].T, mode="reduced")[1]
    total = float(np.linalg.norm(left_coordinates @ right_coordinates.T)**2)
    if total <= 1e-15:
        raise ValueError("The learned update is below the projection resolution floor")
    return {"update_energy_initial_read_write_core": float(np.linalg.norm(physical_core)**2 / total),
            "update_energy_paired_initial_SVD_atoms": float(np.linalg.norm(paired)**2 / total)}


def geometry(a0, a, b0, b, np):
    _, _, init_pair = atom_geometry(a0, b0, np)
    final_r, final_w, final_pair = atom_geometry(a, b, np)
    order_init, order_final = assignment(init_pair), assignment(final_pair)
    a_init, b_init = joint_coordinates(a0, b0, np)
    a_final, b_final = joint_coordinates(a, b, np)
    a_delta, b_delta = difference(a, a0, np), difference(b, b0, np)
    learnt, compared = joint_coordinates(a_delta, b_delta, np)
    rank = len(order_init)
    return {
        "initial_assignment": list(order_init), "final_assignment": list(order_final),
        "assignments_exactly_equal": order_init == order_final,
        "initial_absolute_atom_cosines": [abs(float(init_pair[i, j])) for i, j in enumerate(order_init)],
        "final_absolute_atom_cosines": [abs(float(final_pair[i, j])) for i, j in enumerate(order_final)],
        "signed_final_atom_cosines_at_initial_assignment": [float(final_pair[i, j]) for i, j in enumerate(order_init)],
        "final_absolute_read_cosines": [abs(float(final_r[i, j])) for i, j in enumerate(order_final)],
        "final_absolute_write_cosines": [abs(float(final_w[i, j])) for i, j in enumerate(order_final)],
        "initial_read_span_singular_cosines": subspace_singular(a0[0], b0[0], np),
        "initial_write_span_singular_cosines": subspace_singular(a0[1].T, b0[1].T, np),
        "final_read_span_singular_cosines": subspace_singular(a[0], b[0], np),
        "final_write_span_singular_cosines": subspace_singular(a[1].T, b[1].T, np),
        "initial_operator_cosine": cos(a_init, b_init, np),
        "final_operator_cosine": cos(a_final, b_final, np),
        "update_operator_cosine": cos(learnt, compared, np),
        "relative_ref_update_frobenius": float(np.linalg.norm(learnt) / np.linalg.norm(a_init)),
        "relative_cmp_update_frobenius": float(np.linalg.norm(compared) / np.linalg.norm(b_init)),
        "reference_update_projection": projection_fractions(a0, a_delta, np),
        "compared_update_projection": projection_fractions(b0, b_delta, np),
        "rank": rank,
    }
