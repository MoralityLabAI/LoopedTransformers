"""Controller evaluation: frontiers, budget-matched fixed-depth baseline, calibration, stopping errors."""
from __future__ import annotations

import numpy as np
import torch

from . import controller as C
from .flops import controller_flops, loop_flops, readout_flops
from .metrics import first_correct

LAMBDA_GRID = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.03, 0.06, 0.12, 0.25, 0.5)
THRESH_GRID = (0.34, 0.5, 0.7, 0.85, 0.95, 0.99, 0.999)


def ece(conf, truth, bins=15):
    """Equal-mass ECE.  conf, truth: 1-D arrays."""
    conf, truth = np.asarray(conf, float), np.asarray(truth, float)
    if len(conf) == 0:
        return float("nan"), []
    order = np.argsort(conf)
    parts = np.array_split(order, min(bins, max(1, len(conf) // 20)))
    rel, e = [], 0.0
    for p in parts:
        if len(p) == 0:
            continue
        rel.append((float(conf[p].mean()), float(truth[p].mean()), len(p)))
        e += len(p) / len(conf) * abs(conf[p].mean() - truth[p].mean())
    return float(e), rel


def matched_fixed_curve(correct, kappa):
    """Expected per-sample correctness of the fixed-depth baseline with mean (fractional) depth kappa.

    A fractional depth is realised as a random mixture of floor/ceil depth, whose expected
    accuracy is the linear interpolation between those two fixed depths."""
    K = correct.shape[1]
    kappa = float(np.clip(kappa, 1.0, K))
    lo = int(np.floor(kappa))
    hi = min(lo + 1, K)
    w = kappa - lo
    return (1 - w) * correct[:, lo - 1] + w * correct[:, hi - 1]


def kappa_of(cost_mean, loopF_mean, rF):
    return (cost_mean - rF) / loopF_mean


def stop_stats(tau, correct_ret, kstar, K):
    solvable = kstar <= K
    prem = solvable & (tau < kstar)
    lost = solvable & (tau >= kstar) & ~correct_ret
    ok = correct_ret & solvable & (tau >= kstar)
    return {
        "premature_rate": float(prem.sum() / max(1, solvable.sum())),          # stopped before the first correct loop
        "lost_after_correct_rate": float(lost.sum() / max(1, solvable.sum())),  # was correct earlier, returned wrong
        "excess_loops": float((tau[ok] - kstar[ok]).mean()) if ok.any() else float("nan"),  # loops beyond first-correct
        "futile_loops": float(tau[~solvable].mean()) if (~solvable).any() else float("nan"),  # loops spent on never-solved items
        "overhead_vs_oracle": float((tau - np.minimum(kstar, K)).mean()),
    }


def boot_diff(ctrl_correct, cost_i, correct, loopF_mean, rF, n_boot, rng):
    """Paired bootstrap of (controller acc - budget-matched fixed acc), recomputing the match."""
    n = len(ctrl_correct)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        ix = rng.integers(0, n, size=n)
        kap = kappa_of(cost_i[ix].mean(), loopF_mean, rF)
        diffs[b] = ctrl_correct[ix].mean() - matched_fixed_curve(correct[ix], kap).mean()
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def evaluate_groups(q, T, tok_len, d, groups, lam_grid=LAMBDA_GRID, thr_grid=THRESH_GRID, device="cuda",
                    n_boot=200, seed=0):
    """groups: {name: row indices}.  tok_len[i]: tokens in sample i (for FLOPs)."""
    rng = np.random.default_rng(seed)
    Kc = T.K
    ctrl_F = controller_flops(T.base_dim + C.LOCK_DIM + 1, q.hidden)
    out = {}
    corr_all = T.correct.cpu().numpy()
    maxp_all = T.maxp.cpu().numpy()
    pred_all = T.pred.cpu().numpy()
    y_all = T.y.cpu().numpy()
    for gname, rows in groups.items():
        rows = np.asarray(rows)
        corr = corr_all[rows]
        kstar = first_correct(corr)
        loopF = np.array([loop_flops(int(t), d) for t in tok_len[rows]], float)
        rF = readout_flops(d, T.C)
        loopF_mean = loopF.mean()
        g = {"n": int(len(rows)), "loopF_mean": float(loopF_mean), "readout_F": float(rF), "ctrl_F": float(ctrl_F)}
        # fixed-depth curve
        g["fixed"] = {"k": list(range(1, Kc + 1)), "acc": corr.mean(0).tolist(),
                      "cost": [float((k * loopF).mean() + rF) for k in range(1, Kc + 1)]}
        # oracle halting (returns correct at first-correct loop; unsolved items run to K)
        tau_o = np.minimum(kstar, Kc)
        g["oracle"] = {"acc": float((kstar <= Kc).mean()), "cost": float((tau_o * (loopF + rF)).mean()),
                       "kappa": float(kappa_of((tau_o * (loopF + rF)).mean(), loopF_mean, rF))}
        # confidence-threshold early exit (non-learned adaptive baseline)
        g["conf"] = []
        for th in thr_grid:
            hit = maxp_all[rows] >= th
            tau = np.where(hit.any(1), hit.argmax(1) + 1, Kc)
            ans_ok = corr[np.arange(len(rows)), tau - 1]
            cost_i = tau * (loopF + rF)
            kap = kappa_of(cost_i.mean(), loopF_mean, rF)
            g["conf"].append({"thr": th, "acc": float(ans_ok.mean()), "cost": float(cost_i.mean()), "kappa": float(kap),
                              "matched_fixed_acc": float(matched_fixed_curve(corr, kap).mean()),
                              **stop_stats(tau, ans_ok, kstar, Kc)})
        # learned controller sweep over compute price
        g["ctrl"] = []
        for lam in lam_grid:
            res = C.run_policy(q, T, lam, device=device, idx=torch.as_tensor(rows, device=device))
            tau, okc = res["tau"], res["correct"]
            cost_i = tau * (loopF + rF + ctrl_F)
            kap = kappa_of(cost_i.mean(), loopF_mean, rF)
            base_i = matched_fixed_curve(corr, kap)
            lo, hi = boot_diff(okc.astype(float), cost_i, corr, loopF_mean, rF, n_boot, rng)
            acts = res["acts"]
            vis = acts >= 0
            act_frac = [float((acts == a).sum() / max(1, vis.sum())) for a in (C.CONT, C.PRES, C.HALT)]
            # calibration of Q_halt on visited steps vs raw softmax confidence at the same steps
            qh = res["qhalt"][vis]
            hc = _halt_truth(res, T, rows)[vis]
            e_q, rel_q = ece(np.clip(qh, 0, 1), hc)
            mp = maxp_all[rows][vis]
            cur_ok = corr[vis]  # live-prediction correctness at visited steps
            e_s, rel_s = ece(mp, cur_ok)
            g["ctrl"].append({
                "lam": lam, "acc": float(okc.mean()), "cost": float(cost_i.mean()), "kappa": float(kap),
                "mean_loops": float(tau.mean()), "matched_fixed_acc": float(base_i.mean()),
                "diff": float(okc.mean() - base_i.mean()), "diff_lo": lo, "diff_hi": hi,
                "act_frac": act_frac, "ece_qhalt": e_q, "ece_softmax": e_s,
                "brier_qhalt": float(np.mean((np.clip(qh, 0, 1) - hc) ** 2)) if len(qh) else float("nan"),
                "brier_softmax": float(np.mean((mp - cur_ok) ** 2)) if len(mp) else float("nan"),
                "rel_qhalt": rel_q if lam in (0.004, 0.016) else None,
                "rel_softmax": rel_s if lam in (0.004, 0.016) else None,
                **stop_stats(tau, okc, kstar, Kc),
            })
        out[gname] = g
    return out


def _halt_truth(res, T, rows):
    """For every step of the rollout: would halting right now (with the register state then) return the right answer?"""
    # Reconstruct the register: acts -> latch dynamics, using stored predictions.
    pred = T.pred.cpu().numpy()[rows]
    y = T.y.cpu().numpy()[rows]
    acts = res["acts"]
    n, K = acts.shape
    lock = np.zeros(n, bool)
    lp = np.zeros(n, int)
    truth = np.zeros((n, K))
    for k in range(K):
        cur = np.where(lock, lp, pred[:, k])
        truth[:, k] = (cur == y)
        a = acts[:, k]
        new_lock = np.where(a == C.PRES, True, np.where(a == C.CONT, False, lock))
        set_new = (a == C.PRES) & ~lock
        lp = np.where(set_new, pred[:, k], lp)
        lock = new_lock
    return truth
