"""Statistics helpers: AUROC, bootstrap, set overlap, CKA, principal angles, effect sizes."""
from __future__ import annotations

import numpy as np
import torch


def auroc(score, label) -> float:
    """Rank-based AUROC (ties averaged).  Returns NaN if one class is missing."""
    from scipy.stats import rankdata
    s, y = np.asarray(score, dtype=np.float64), np.asarray(label).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auroc_cols(S: np.ndarray, label) -> np.ndarray:
    """AUROC of every column of S [N, F] vs label [N]."""
    from scipy.stats import rankdata
    y = np.asarray(label).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return np.full(S.shape[1], np.nan)
    R = rankdata(S, axis=0)
    return (R[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def bootstrap_ci(fn, n: int, B: int = 500, seed: int = 0, alpha: float = 0.05, cluster: np.ndarray | None = None):
    """fn(idx)->float.  Optional cluster ids -> resample whole clusters (e.g. model seeds)."""
    rng = np.random.default_rng(seed)
    vals = []
    if cluster is None:
        for _ in range(B):
            vals.append(fn(rng.integers(0, n, size=n)))
    else:
        cl = np.unique(cluster)
        groups = [np.where(cluster == c)[0] for c in cl]
        for _ in range(B):
            pick = rng.integers(0, len(cl), size=len(cl))
            vals.append(fn(np.concatenate([groups[i] for i in pick])))
    vals = np.array(vals, dtype=np.float64)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def jaccard_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard of boolean matrices [N, F]; rows with both empty -> NaN."""
    inter = (A & B).sum(1).astype(np.float64)
    union = (A | B).sum(1).astype(np.float64)
    return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def cosine_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    na, nb = np.linalg.norm(A, axis=1), np.linalg.norm(B, axis=1)
    return (A * B).sum(1) / np.maximum(na * nb, 1e-12)


def cohens_d(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / max(sp, 1e-12))


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    p = np.asarray(p)
    n = len(p)
    order = np.argsort(p)
    thr = q * (np.arange(1, n + 1) / n)
    ok = p[order] <= thr
    keep = np.zeros(n, bool)
    if ok.any():
        keep[order[: np.max(np.where(ok)[0]) + 1]] = True
    return keep


# --------------------------------------------------------------- similarity
def linear_cka(X, Y) -> float:
    """Linear CKA between [N, dx] and [N, dy] (rows = same inputs)."""
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.t() @ Y).pow(2).sum()
    xx = (X.t() @ X).pow(2).sum().sqrt()
    yy = (Y.t() @ Y).pow(2).sum().sqrt()
    return float(xy / (xx * yy + 1e-30))


def pca_basis(X, k: int) -> torch.Tensor:
    X = torch.as_tensor(X, dtype=torch.float64)
    X = X - X.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:k].t()          # [d, k]


def principal_angles(X, Y, k: int):
    """Principal angles (degrees) between the top-k PCA subspaces of X and Y (both [N, d], same d)."""
    Qx, Qy = pca_basis(X, k), pca_basis(Y, k)
    s = torch.linalg.svdvals(Qx.t() @ Qy).clamp(-1, 1)
    return torch.rad2deg(torch.acos(s)).numpy()


def subspace_overlap(X, Y, k: int) -> float:
    """mean cos^2 of principal angles = ||Qx^T Qy||_F^2 / k  in [0,1]."""
    Qx, Qy = pca_basis(X, k), pca_basis(Y, k)
    return float((Qx.t() @ Qy).pow(2).sum() / k)


def ridge_map(X, Y, lam: float = 1e-2):
    """Least-squares linear map W with X W + b ~ Y, ridge-regularised.  Returns (W, b) as float32."""
    X = torch.as_tensor(X, dtype=torch.float64)
    Y = torch.as_tensor(Y, dtype=torch.float64)
    mx, my = X.mean(0, keepdim=True), Y.mean(0, keepdim=True)
    Xc, Yc = X - mx, Y - my
    A = Xc.t() @ Xc + lam * Xc.shape[0] * torch.eye(X.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(A, Xc.t() @ Yc)
    return W.float(), (my - mx @ W).float()
