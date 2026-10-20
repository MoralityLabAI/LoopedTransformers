"""Model-agnostic analyses over per-loop hidden states.

Conventions: X is [N, L, d] (examples x loop-states x width); `correct` is [N, K] boolean for the readout after
loops 1..K; loop index k in `X` is the state AFTER loop k (X[:, 0] = input state).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .sae import TopKSAE, recon_stats
from .stats import auroc, jaccard_rows


# ------------------------------------------------------------------ SAE application
@torch.no_grad()
def encode_loops(sae: TopKSAE, X: np.ndarray, loops, device="cpu", chunk=50000):
    """Returns idx [N, nl, k] int16, val [N, nl, k] float16 (top-k codes per requested loop)."""
    N = X.shape[0]
    k = sae.cfg.k
    idx = np.zeros((N, len(loops), k), np.int16)
    val = np.zeros((N, len(loops), k), np.float16)
    for j, l in enumerate(loops):
        for i in range(0, N, chunk):
            x = torch.from_numpy(X[i:i + chunk, l]).float().to(device)
            z = sae.encode(x)
            v, ix = z.topk(k, dim=-1)
            idx[i:i + chunk, j] = ix.cpu().numpy().astype(np.int16)
            val[i:i + chunk, j] = v.cpu().numpy().astype(np.float16)
    return idx, val


def dense_codes(idx, val, n, rows=None, j=None):
    """Dense float32 [R, n] from top-k storage for loop slot j."""
    ix = idx[:, j] if rows is None else idx[rows, j]
    vv = val[:, j] if rows is None else val[rows, j]
    out = np.zeros((ix.shape[0], n), np.float32)
    np.put_along_axis(out, ix.astype(np.int64), vv.astype(np.float32), axis=1)
    return out


def active_bool(idx, val, n, j):
    d = dense_codes(idx, val, n, j=j)
    return d > 0


def loop_recon(sae: TopKSAE, X: np.ndarray, loops, device="cpu"):
    """NMSE (strict: vs the variance of that loop's own states) and FVU vs the SAE training mean, per loop."""
    out = []
    for l in loops:
        x = torch.from_numpy(X[:, l]).float()
        s = recon_stats(sae, x, device=device)
        s2 = recon_stats(sae, x, device=device, ref_mu=sae.mu.cpu())
        out.append({"loop": int(l), "nmse_own": s["nmse"], "fvu_ref": s2["nmse"], "l0": s["l0"],
                    "row_err": s["row_err"].numpy()})
    return out


# ------------------------------------------------------------------ overlap / stability
def adjacent_overlap(idx, val, n, groups: np.ndarray | None = None, seed: int = 0):
    """For consecutive loop slots (j, j+1): per-example Jaccard of active sets, cosine of dense codes,
    carried energy sum(min(z_j, z_j+1))/sum(z_j), plus a label-matched null (partner example drawn from
    the same `groups` id, e.g. same true label and length) so that 'overlap' is not just shared class content."""
    rng = np.random.default_rng(seed)
    N, nl, _ = idx.shape
    if groups is None:
        groups = np.zeros(N, int)
    perm = np.arange(N)
    for g in np.unique(groups):
        m = np.where(groups == g)[0]
        perm[m] = rng.permutation(m)
    res = []
    prev = dense_codes(idx, val, n, j=0)
    for j in range(nl - 1):
        nxt = dense_codes(idx, val, n, j=j + 1)
        A, B = prev > 0, nxt > 0
        jac = jaccard_rows(A, B)
        cos = (prev * nxt).sum(1) / np.maximum(np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt, axis=1), 1e-12)
        carry = np.minimum(prev, nxt).sum(1) / np.maximum(prev.sum(1), 1e-12)
        jac_null = jaccard_rows(A, B[perm])
        cos_null = (prev * nxt[perm]).sum(1) / np.maximum(
            np.linalg.norm(prev, axis=1) * np.linalg.norm(nxt[perm], axis=1), 1e-12)
        res.append({"slot": j, "jacc": jac, "cos": cos, "carry": carry, "jacc_null": jac_null, "cos_null": cos_null})
        prev = nxt
    return res


def feature_frequency(idx, n):
    """P(active) per latent over all (example, loop-slot) in idx."""
    cnt = np.bincount(idx.reshape(-1).astype(np.int64), minlength=n).astype(np.float64)
    return cnt / (idx.shape[0] * idx.shape[1])


# ------------------------------------------------------------------ events
def event_labels(correct: np.ndarray, horizon: int = 4):
    """Per (example, loop t=1..K-horizon) targets (all [N, K']):
         now   : correct at t
         stay  : correct at t..t+H           (retained)
         gain  : ~correct at t and correct at t+H   (more compute helps)
         harm  : correct at t and ~correct at t+H   (more compute hurts)
       Arrays are aligned so that column c <-> loop t=c+1."""
    N, K = correct.shape
    H = horizon
    now = correct[:, : K - H]
    fut = correct[:, H:]
    stay = np.ones_like(now)
    for h in range(H + 1):
        stay &= correct[:, h: K - H + h]
    return {"now": now, "stay": stay, "gain": (~now) & fut, "harm": now & (~fut), "fut": fut}


def first_stable(correct: np.ndarray):
    """First loop (1-idx) from which the answer is correct at every later loop; K+1 if never."""
    N, K = correct.shape
    suffix = np.flip(np.cumprod(np.flip(correct, 1), 1), 1).astype(bool)
    ok = suffix.any(1)
    return np.where(ok, suffix.argmax(1) + 1, K + 1)


def event_taxonomy(correct: np.ndarray):
    """Per example: 'never' (never correct), 'stable' (first-correct stays correct), 'lost' (correct then wrong at end),
    'flicker' (correct at end but wrong somewhere after first correct)."""
    N, K = correct.shape
    anyc = correct.any(1)
    fc = np.where(anyc, correct.argmax(1), K)
    after_wrong = np.array([(~correct[i, fc[i]:]).any() if anyc[i] else False for i in range(N)])
    lab = np.empty(N, object)
    lab[~anyc] = "never"
    lab[anyc & ~after_wrong] = "stable"
    lab[anyc & after_wrong & ~correct[:, -1]] = "lost"
    lab[anyc & after_wrong & correct[:, -1]] = "flicker"
    return lab


# ------------------------------------------------------------------ probes (torch logistic regression)
class Logit:
    """L2 logistic regression with standardisation; torch LBFGS (no sklearn available)."""

    def __init__(self, l2=1e-2, iters=60):
        self.l2, self.iters = l2, iters

    def fit(self, X, y, w=None):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        self.mu, self.sd = X.mean(0, keepdim=True), X.std(0, keepdim=True) + 1e-6
        Xs = (X - self.mu) / self.sd
        self.w = torch.zeros(X.shape[1], requires_grad=True)
        self.b = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], lr=1.0, max_iter=self.iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = Xs @ self.w + self.b
            loss = F.binary_cross_entropy_with_logits(z, y) + self.l2 * (self.w ** 2).sum() / 2
            loss.backward()
            return loss

        opt.step(closure)
        return self

    @torch.no_grad()
    def score(self, X):
        X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return (((X - self.mu) / self.sd) @ self.w + self.b).numpy()


def subsample_rows(n, cap, rng):
    return np.arange(n) if n <= cap else np.sort(rng.choice(n, cap, replace=False))


# ------------------------------------------------------------------ stopping policies
def stop_from_scores(score: np.ndarray, tau: float, t_min: int = 1):
    """score [N, K]: policy stops at first t >= t_min with score >= tau, else at K.  Returns stop index (0-based col)."""
    N, K = score.shape
    ok = score >= tau
    ok[:, : t_min - 1] = False
    any_ = ok.any(1)
    return np.where(any_, ok.argmax(1), K - 1)


def policy_curve(score: np.ndarray, correct: np.ndarray, taus, t_min: int = 1):
    pts = []
    for tau in taus:
        st = stop_from_scores(score, tau, t_min)
        pts.append((float((st + 1).mean()), float(correct[np.arange(len(st)), st].mean()), float(tau)))
    return np.array(pts)         # columns: mean loops, accuracy, tau


def pareto_area(curve: np.ndarray, c_lo: float, c_hi: float, n=200):
    """Area under the upper-envelope accuracy-vs-mean-loops curve on [c_lo, c_hi] (normalised by width)."""
    order = np.argsort(curve[:, 0])
    c, a = curve[order, 0], np.maximum.accumulate(curve[order, 1])
    grid = np.linspace(c_lo, c_hi, n)
    interp = np.interp(grid, c, a, left=a[0] if c_lo < c[0] else a[0], right=a[-1])
    interp = np.where(grid < c[0], 0.0, interp)         # no credit for budgets the policy cannot reach
    return float(interp.mean())


def acc_at_budget(curve: np.ndarray, budget: float):
    ok = curve[:, 0] <= budget + 1e-9
    return float(curve[ok, 1].max()) if ok.any() else float("nan")
