"""Synthetic CPU proxy for analytic adapter updates (evidence class: synthetic_cpu).

Tinker returns logprobs and losses but no activations, gradients or adapter tensors, so the
projected-update question is tested on small linear layers here. Nothing in this module is
evidence about Inkling.

Layer W (d_out x d_in). Keep inputs X_keep (k x d_in, rank k < d_in) must map unchanged.
New-task inputs X_new must move toward targets W* x. Each method returns D = dW^T (d_in x d_out).
Projected method: P = I - X_keep^+ X_keep, D = P U, U = (X_new P)^+ R, so X_keep D = 0 exactly.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from . import resources  # noqa: F401
import numpy as np


@dataclass(frozen=True)
class Scenario:
    d_in: int = 192
    d_out: int = 96
    k_keep: int = 96
    m_new: int = 384
    conflict: float = 0.5      # fraction of new-task input energy inside span(X_keep)
    noise: float = 0.01
    offspan_keep: float = 0.0  # held-out keep inputs: fraction of energy off span(X_keep)
    dtype: str = "float64"


def make(s: Scenario, rng: np.random.Generator) -> dict:
    Q, _ = np.linalg.qr(rng.standard_normal((s.d_in, s.d_in)))
    keep_basis, free_basis = Q[:, :s.k_keep], Q[:, s.k_keep:]

    def keep_inputs(n):
        x = rng.standard_normal((n, s.k_keep)) @ keep_basis.T
        if s.offspan_keep:
            off = rng.standard_normal((n, s.d_in - s.k_keep)) @ free_basis.T
            x = np.sqrt(1 - s.offspan_keep) * x + np.sqrt(s.offspan_keep) * off
        return x

    def new_inputs(n):
        a = rng.standard_normal((n, s.k_keep)) @ keep_basis.T
        b = rng.standard_normal((n, s.d_in - s.k_keep)) @ free_basis.T
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b /= np.linalg.norm(b, axis=1, keepdims=True)
        return np.sqrt(s.conflict) * a + np.sqrt(1 - s.conflict) * b

    W = rng.standard_normal((s.d_out, s.d_in)) / np.sqrt(s.d_in)
    # Target edit acts on the new task; low rank, unconstrained direction.
    E = (rng.standard_normal((s.d_out, 8)) @ rng.standard_normal((8, s.d_in))) / s.d_in
    Xk = rng.standard_normal((s.k_keep, s.k_keep)) @ keep_basis.T  # exact keep set, rank k
    Xn, Xn_test = new_inputs(s.m_new), new_inputs(4 * s.m_new)
    R = Xn @ E.T + s.noise * rng.standard_normal((s.m_new, s.d_out))
    dt = np.dtype(s.dtype)
    return {k: v.astype(dt) for k, v in dict(W=W, E=E, Xk=Xk, Xn=Xn, Xn_test=Xn_test,
            Xk_test=keep_inputs(4 * s.k_keep), R=R).items()}


def m_ordinary(d, lr=0.5, steps=200, **_):
    D = np.zeros((d["Xn"].shape[1], d["R"].shape[1]), d["R"].dtype)
    n = d["Xn"].shape[0]
    for _ in range(steps):
        D -= lr * d["Xn"].T @ (d["Xn"] @ D - d["R"]) / n
    return D


def m_ridge(d, lam=10.0, mu=1e-3, **_):
    """Soft preservation (FP-Solve-style): argmin |Xn D - R|^2 + lam |Xk D|^2 + mu |D|^2."""
    Xn, Xk = d["Xn"], d["Xk"]
    A = Xn.T @ Xn + lam * Xk.T @ Xk + mu * np.eye(Xn.shape[1], dtype=Xn.dtype)
    return np.linalg.solve(A, Xn.T @ d["R"])


def m_tsvd(d, rank=8, **_):
    D = np.linalg.pinv(d["Xn"]) @ d["R"]
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    return (U[:, :rank] * S[:rank]) @ Vt[:rank]


def projector(Xk, rcond=None):
    X = Xk.astype(np.float64)
    P = np.eye(X.shape[1]) - np.linalg.pinv(X, rcond=rcond or 1e-10) @ X
    return P.astype(Xk.dtype)


def null_basis(Xk, tol=1e-10):
    """Orthonormal basis N of null(X_keep) from the SVD, so X_keep N ~ 0 to rounding."""
    X = Xk.astype(np.float64)
    _, S, Vt = np.linalg.svd(X, full_matrices=True)
    r = int((S > tol * S.max()).sum())
    return Vt[r:].T.astype(Xk.dtype)


def m_projected(d, rank=None, **_):
    """D = P U computed stably as N (X_new N)^+ R with N an orthonormal null-space basis."""
    N = null_basis(d["Xk"])
    C = np.linalg.pinv(d["Xn"] @ N) @ d["R"]          # coordinates inside null(X_keep)
    if rank is not None:  # low-rank spectral adapter inside range(P)
        Uu, S, Vt = np.linalg.svd(C, full_matrices=False)
        C = (Uu[:, :rank] * S[:rank]) @ Vt[:rank]
    return N @ C


def m_projected_naive(d, **_):
    """Literal P = I - X^+ X, U = pinv(X_new P) R with NumPy default rcond. Diagnostic only:
    X_new P has d_in-k nonzero singular values plus rounding-level ones that default rcond keeps,
    so U can blow up along span(X_keep) and P U leaks."""
    P = projector(d["Xk"])
    return P @ (np.linalg.pinv(d["Xn"] @ P) @ d["R"])


def m_random(d, rank=8, ref=None, rng=None, **_):
    """Matched-rank, matched-Frobenius random control."""
    rng = rng or np.random.default_rng(0)
    D = rng.standard_normal((d["Xn"].shape[1], rank)) @ rng.standard_normal((rank, d["R"].shape[1]))
    scale = np.linalg.norm(ref) / max(np.linalg.norm(D), 1e-30) if ref is not None else 1.0
    return (D * scale).astype(d["R"].dtype)


def metrics(d, D) -> dict:
    W = d["W"]
    keep_out = d["Xk"] @ W.T
    pres = np.linalg.norm(d["Xk"] @ D) / np.linalg.norm(keep_out)
    pres_test = np.linalg.norm(d["Xk_test"] @ D) / np.linalg.norm(d["Xk_test"] @ W.T)
    tgt = d["Xn_test"] @ d["E"].T
    fit = 1.0 - np.linalg.norm(d["Xn_test"] @ D - tgt) ** 2 / np.linalg.norm(tgt) ** 2
    return {"keep_residual_rel": float(pres), "keep_heldout_rel": float(pres_test),
            "new_heldout_r2": float(fit), "update_rank": int(np.linalg.matrix_rank(D.astype(np.float64), tol=1e-8)),
            "update_fro": float(np.linalg.norm(D))}


METHODS = ("ordinary", "ridge", "tsvd", "projected", "projected_naive_pinv", "projected_lowrank", "random_matched")


def run_scenario(s: Scenario, seed: int, rank: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    d = make(s, rng)
    out = {}
    proj = m_projected(d)
    for name, D in (("ordinary", m_ordinary(d)), ("ridge", m_ridge(d)), ("tsvd", m_tsvd(d, rank)),
                    ("projected", proj), ("projected_naive_pinv", m_projected_naive(d)),
                    ("projected_lowrank", m_projected(d, rank)),
                    ("random_matched", m_random(d, rank, ref=m_projected(d, rank), rng=rng))):
        out[name] = metrics(d, D)
    out["nullspace_dim"] = int(s.d_in - np.linalg.matrix_rank(d["Xk"].astype(np.float64)))
    return out


def adaptive_rank(spectra: list[np.ndarray], budget: int) -> list[int]:
    """Greedy rank allocation across modules by marginal squared singular value."""
    alloc = [0] * len(spectra)
    for _ in range(budget):
        gains = [(sp[a] ** 2 if a < len(sp) else -1.0) for sp, a in zip(spectra, alloc)]
        i = int(np.argmax(gains))
        if gains[i] <= 0:
            break
        alloc[i] += 1
    return alloc


def adaptive_vs_uniform(seed: int, modules: int = 4, budget: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    spectra, captured_a, captured_u = [], 0.0, 0.0
    for j in range(modules):
        decay = rng.uniform(0.3, 1.5)
        sp = np.sort(np.exp(-decay * np.arange(32)) * rng.uniform(0.5, 2.0))[::-1]
        spectra.append(sp)
    alloc = adaptive_rank(spectra, budget)
    total = sum(float((sp ** 2).sum()) for sp in spectra)
    captured_a = sum(float((sp[:a] ** 2).sum()) for sp, a in zip(spectra, alloc)) / total
    u = budget // modules
    captured_u = sum(float((sp[:u] ** 2).sum()) for sp in spectra) / total
    return {"allocation": alloc, "captured_adaptive": captured_a, "captured_uniform": captured_u}


# ---- strategy router: choose update method from measurable task properties -------------
def features(d) -> list[float]:
    P = projector(d["Xk"])
    inside = 1.0 - np.linalg.norm(d["Xn"] @ P) ** 2 / np.linalg.norm(d["Xn"]) ** 2
    return [float(inside)]


def utility(m: dict, keep_weight: float = 2.0) -> float:
    return m["new_heldout_r2"] - keep_weight * m["keep_heldout_rel"]


def router_study(train_seeds, test_seeds, conflicts=(0.1, 0.5, 0.9), candidates=("ridge", "projected", "tsvd")):
    """Nearest-centroid router fit on development scenarios, evaluated on fresh seeds."""
    table = []
    for seed in train_seeds:
        for c in conflicts:
            s = Scenario(conflict=c)
            res = run_scenario(s, seed)
            f = features(make(s, np.random.default_rng(seed)))
            best = max(candidates, key=lambda k: utility(res[k]))
            table.append((f, best))
    cents = {}
    for k in candidates:
        pts = [f for f, b in table if b == k]
        if pts:
            cents[k] = np.mean(pts, axis=0)
    test = {"router": [], **{f"always_{k}": [] for k in candidates}, "oracle": [], "random": []}
    rr = np.random.default_rng(99)
    for seed in test_seeds:
        for c in conflicts:
            s = Scenario(conflict=c)
            res = run_scenario(s, seed)
            f = np.array(features(make(s, np.random.default_rng(seed))))
            pick = min(cents, key=lambda k: float(np.linalg.norm(cents[k] - f)))
            test["router"].append(utility(res[pick]))
            for k in candidates:
                test[f"always_{k}"].append(utility(res[k]))
            test["oracle"].append(max(utility(res[k]) for k in candidates))
            test["random"].append(utility(res[candidates[int(rr.integers(len(candidates)))]]))
    return {"train_best_counts": {k: sum(b == k for _, b in table) for k in candidates},
            "centroids": {k: v.tolist() for k, v in cents.items()},
            "test_mean_utility": {k: float(np.mean(v)) for k, v in test.items()},
            "n_test_cases": len(test["router"])}


def study(seeds=range(20), out: Path | None = None) -> dict:
    rows = {}
    for label, s in {"conflict_0.1": Scenario(conflict=0.1), "conflict_0.5": Scenario(conflict=0.5),
                     "conflict_0.9": Scenario(conflict=0.9),
                     "conflict_0.5_float32": Scenario(conflict=0.5, dtype="float32"),
                     "conflict_0.5_offspan_keep_0.2": Scenario(conflict=0.5, offspan_keep=0.2)}.items():
        per = [run_scenario(s, int(seed)) for seed in seeds]
        agg = {}
        for mth in METHODS:
            agg[mth] = {k: {"mean": float(np.mean([p[mth][k] for p in per])),
                            "sd": float(np.std([p[mth][k] for p in per], ddof=1)),
                            "max": float(np.max([p[mth][k] for p in per]))}
                        for k in ("keep_residual_rel", "keep_heldout_rel", "new_heldout_r2")}
        rows[label] = {"scenario": s.__dict__, "nullspace_dim": per[0]["nullspace_dim"], "methods": agg}
    ar = [adaptive_vs_uniform(int(s)) for s in seeds]
    report = {"evidence_class": "synthetic_cpu", "native_inkling_evidence": False,
              "seeds": list(map(int, seeds)), "scenarios": rows,
              "adaptive_rank": {"captured_adaptive_mean": float(np.mean([a["captured_adaptive"] for a in ar])),
                                "captured_uniform_mean": float(np.mean([a["captured_uniform"] for a in ar])),
                                "adaptive_ge_uniform_all": all(a["captured_adaptive"] >= a["captured_uniform"] - 1e-12 for a in ar)},
              "router": router_study(range(100, 110), range(200, 220))}
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=20)
    a = ap.parse_args()
    r = study(range(a.seeds), a.out)
    print(json.dumps({k: {m: v["methods"][m]["keep_residual_rel"]["max"] for m in METHODS}
                      for k, v in r["scenarios"].items()}, indent=1))


if __name__ == "__main__":
    main()
