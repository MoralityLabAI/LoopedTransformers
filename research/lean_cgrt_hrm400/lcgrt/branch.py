"""Fit the CGRT branch scorer from search traces on TRAINING theorems.

    python -m lcgrt.branch --ckpt D:/lean_cgrt/runs/c1/S_t8_s17 --n 400 --out .../branch.json

CGRT search (with the untrained prior score) is run on training theorems; every child it
scores becomes a row of BRANCH_FEATURES (scores under each context, cross-context
agreement, predicted child value, sibling count, Lean-verified dead siblings). The label is
whether Lean's exploration table says the tactic starts a shortest proof of that goal.
A 9-weight logistic regression is fitted on 80% of theorems and checked on the other 20%.
No evaluation split is touched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .dataset import iter_records
from .evaluate import BUDGETS, load_model
from .lean_repl import LeanRepl
from .search import BRANCH_FEATURES, Scorer, run_searches

INF = 10 ** 9


def optimal_set(rec, key):
    size = rec["size"]
    cost = {t: 1 + sum(size[k] if size[k] is not None else INF for k in new) for t, new in rec["table"][key].items()}
    best = min(cost.values())
    return {t for t, c in cost.items() if c == best}


def fit_logistic(X, y, w0, l2=1e-2, iters=300):
    """Newton's method, L2-regularised toward w0 (the untrained prior score)."""
    w = w0.copy()
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y) + l2 * (w - w0)
        H = (X * (p * (1 - p))[:, None]).T @ X / len(y) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def auc(s, y):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / max(1, pos.sum() * (~pos).sum()))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="lcgrt-v1")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    model, _ = load_model(Path(a.ckpt), a.device)
    recs = {}
    for r in iter_records(a.data, "train"):
        recs[r["tid"]] = r
        if len(recs) >= a.n:
            break
    items = [(t, r["statement"], {"size": r["proof"]["size"]}) for t, r in recs.items()]
    T = model.cfg.fast_steps if model.cfg.arch == "hrm" else None
    contexts = [(T, False), (2 * T, False), (T, True)] if T else [(None, False), (None, True)]
    trace = []
    with LeanRepl() as repl:
        run_searches(repl, items, "cgrt", Scorer(model, a.device, fast_steps=T),
                     budget_fn=lambda m: BUDGETS["b4"](m["size"]), contexts=contexts, trace=trace)
    X, y, grp = [], [], []
    for t in trace:
        rec = recs[t["tid"]]
        for f, tac, key in t["rows"]:
            X.append([f[k] for k in BRANCH_FEATURES]); y.append(float(tac in optimal_set(rec, key)))
            grp.append(t["tid"])
    X, y = np.array(X), np.array(y)
    tids = sorted(set(grp)); cut = set(tids[: int(0.8 * len(tids))])
    tr = np.array([g in cut for g in grp])
    w0 = np.array([{"mean_logp": 1.0, "child_value": 0.5}.get(k, 0.0) for k in BRANCH_FEATURES])
    w = fit_logistic(X[tr], y[tr], w0)
    prior = X[:, 0] + 0.5 * X[:, 4]
    out = {"weights": dict(zip(BRANCH_FEATURES, map(float, w))), "rows": int(len(y)), "positives": int(y.sum()),
           "fit_theorems": len(cut), "check_theorems": len(tids) - len(cut),
           "check_auc_learned": round(auc(X[~tr] @ w, y[~tr]), 4),
           "check_auc_prior_score": round(auc(prior[~tr], y[~tr]), 4),
           "check_auc_mean_logp_only": round(auc(X[~tr, 0], y[~tr]), 4),
           "source": "train split only", "ckpt": a.ckpt, "contexts": contexts}
    o = Path(a.out)
    if o.suffix != ".json":                 # queue job: directory with branch.json + complete.json
        o.mkdir(parents=True, exist_ok=True)
        (o / "branch.json").write_text(json.dumps(out, indent=1) + "\n")
        (o / "complete.json").write_text("{}\n")
    else:
        o.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
