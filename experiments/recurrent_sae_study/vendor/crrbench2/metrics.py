"""Compute / Retain metrics from per-loop predictions.

preds   : int [n, K]   argmax prediction at loops 1..K
y       : int [n]
"""
from __future__ import annotations

import numpy as np


def first_correct(correct):
    """1-indexed first correct loop; K+1 if never correct.  correct: bool [n,K]."""
    n, K = correct.shape
    any_ = correct.any(1)
    fc = correct.argmax(1) + 1
    return np.where(any_, fc, K + 1)


def accuracy_by_loop(correct):
    return correct.mean(0)


def retention_from_first_correct(correct, kstar):
    """P(correct at k*+delta | solved at k*) for delta = 0..K-1 (NaN where no sample has that horizon)."""
    n, K = correct.shape
    out = np.full(K, np.nan)
    solved = kstar <= K
    for dlt in range(K):
        m = solved & (kstar + dlt <= K)
        if m.sum() > 0:
            out[dlt] = correct[m, kstar[m] - 1 + dlt].mean()
    return out


def retention_from_depth(correct, k):
    """P(correct at k+delta | correct at k), delta = 0..K-k (k is 1-indexed)."""
    n, K = correct.shape
    base = correct[:, k - 1]
    if base.sum() == 0:
        return np.full(K - k + 1, np.nan)
    return correct[base][:, k - 1:].mean(0)


def compute_retain_summary(correct, depths=(1, 2, 4, 8, 16, 32)):
    n, K = correct.shape
    kstar = first_correct(correct)
    solved = kstar <= K
    # stable-solve: correct at every loop from k* to K (answer never lost)
    suffix_all = np.flip(np.cumprod(np.flip(correct, 1), 1), 1).astype(bool)  # correct at all loops >= k
    stable = np.array([suffix_all[i, kstar[i] - 1] if solved[i] else False for i in range(n)])
    lost = solved & ~stable
    # one-step forgetting hazard: P(wrong at k+1 | correct at k)
    c0, c1 = correct[:, :-1], correct[:, 1:]
    hazard = float((c0 & ~c1).sum() / max(1, c0.sum()))
    ret_depth = {int(k): retention_from_depth(correct, k) for k in depths if k <= K}
    return {
        "acc_by_loop": accuracy_by_loop(correct),
        "acc_at_depths": {int(k): float(correct[:, k - 1].mean()) for k in depths if k <= K},
        "solved_frac": float(solved.mean()),
        "kstar_median": float(np.median(kstar[solved])) if solved.any() else float("nan"),
        "kstar_mean": float(kstar[solved].mean()) if solved.any() else float("nan"),
        "kstar_hist": np.bincount(kstar, minlength=K + 2)[1:K + 2],
        "stable_solve_frac": float(stable.mean()),
        "lost_given_solved": float(lost.sum() / max(1, solved.sum())),
        "forget_hazard": hazard,
        "retention_from_kstar": retention_from_first_correct(correct, kstar),
        "retention_from_depth": ret_depth,
        "final_acc": float(correct[:, -1].mean()),
        "best_loop_acc": float(correct.mean(0).max()),
    }
