"""Small, dependency-light inference helpers (units of analysis = independent training seeds)."""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import stats as sps


def mean_ci_t(x, level=0.95):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else float("nan")), float("nan"), float("nan")
    se = x.std(ddof=1) / math.sqrt(n)
    t = sps.t.ppf(0.5 + level / 2, n - 1)
    return float(x.mean()), float(x.mean() - t * se), float(x.mean() + t * se)


def signflip_p(diffs, two_sided=True, n_mc=200_000, seed=0):
    """Exact (n<=16) or Monte-Carlo sign-flip permutation test of H0: mean diff = 0 (paired, per seed)."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    if n == 0:
        return float("nan")
    obs = d.mean()
    if n <= 16:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
        means = (signs * d).mean(1)
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(n_mc, n))
        means = (signs * d).mean(1)
    if two_sided:
        return float((np.abs(means) >= abs(obs) - 1e-15).mean())
    return float((means >= obs - 1e-15).mean())


def paired_test(diffs):
    """Summary for per-seed paired differences: mean, t-CI, t-test p, sign-flip p, Wilcoxon p, Cohen's dz."""
    d = np.asarray(diffs, float)
    d = d[~np.isnan(d)]
    n = len(d)
    m, lo, hi = mean_ci_t(d)
    out = {"n": n, "mean": m, "ci_lo": lo, "ci_hi": hi}
    if n >= 2 and d.std(ddof=1) > 0:
        out["t_p"] = float(sps.ttest_1samp(d, 0.0).pvalue)
        out["dz"] = float(m / d.std(ddof=1))
    else:
        out["t_p"], out["dz"] = float("nan"), float("nan")
    out["signflip_p"] = signflip_p(d)
    try:
        out["wilcoxon_p"] = float(sps.wilcoxon(d).pvalue) if n >= 5 and np.any(d != 0) else float("nan")
    except ValueError:
        out["wilcoxon_p"] = float("nan")
    out["frac_pos"] = float((d > 0).mean()) if n else float("nan")
    return out


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (NaN-safe)."""
    p = np.asarray(pvals, float)
    adj = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def cluster_bootstrap_mean(groups, n_boot=5000, seed=0):
    """Two-level bootstrap of a grand mean: resample clusters (e.g. tasks), then items within (e.g. seeds)."""
    rng = np.random.default_rng(seed)
    gs = [np.asarray(g, float)[~np.isnan(np.asarray(g, float))] for g in groups]
    gs = [g for g in gs if len(g)]
    est = np.empty(n_boot)
    for b in range(n_boot):
        ci = rng.integers(0, len(gs), len(gs))
        est[b] = np.mean([gs[c][rng.integers(0, len(gs[c]), len(gs[c]))].mean() for c in ci])
    return float(np.mean([g.mean() for g in gs])), float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))
