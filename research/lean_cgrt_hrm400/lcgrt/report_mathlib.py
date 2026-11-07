"""c3 analysis (registered before results): Lean-certified one-step completion on Mathlib.

    python -m lcgrt.report_mathlib --eval D:/lean_cgrt/runs/c1/evalM_tlm3_s17_29_43 --out results/c3

Decision rules (LOCAL_SCALING_RESULTS.md, section c3):
  validity  the source tactic itself completes >= 95% of reached states, else the run is invalid
  H1        an encoder beats flat if the Welch 95% CI over seeds of (close@8 arm - close@8 flat),
            model-only, on all held-out states pooled, lies above 0
  H2        a model adds to automation if (model+auto close@8 - auto close@8), success averaged
            over the arm's seeds per state, has a paired-bootstrap 95% CI above 0
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
from pathlib import Path

import numpy as np

from .mathlib_eval import KS, completes

ARM_RE = re.compile(r"^tlmd*_(?P<arm>[A-Za-z]+)_s(?P<seed>\d+)$")


def success(row, policy, k):
    res = row["results"] or {}
    return any(completes(res.get(t)) for t in row["policies"][policy][:k])


def welch(a, b):
    from scipy import stats
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a.mean() - b.mean()
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return {"diff": round(d, 4), "ci95": [round(d, 4), round(d, 4)]}
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = stats.t.ppf(0.975, dof) * se
    return {"diff": round(d, 4), "ci95": [round(d - h, 4), round(d + h, 4)], "dof": round(float(dof), 2)}


def paired_bootstrap(x, y, n=10000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    d = (x[idx] - y[idx]).mean(1)
    return {"diff": round(float((x - y).mean()), 4), "ci95": [round(float(np.quantile(d, .025)), 4),
                                                             round(float(np.quantile(d, .975)), 4)]}


def analyse(eval_dir):
    rows = [json.loads(l) for l in gzip.open(Path(eval_dir) / "states.jsonl.gz", "rt", encoding="utf-8")]
    reached = [r for r in rows if r["reached"]]
    runs = sorted({p for r in rows for p in r["policies"] if ARM_RE.match(p)})
    arms = collections.defaultdict(list)
    for run in runs:
        m = ARM_RE.match(run); arms[m["arm"]].append(run)
    splits = sorted({r["split"] for r in rows})
    out = {"states": len(rows), "reached": len(reached),
           "validity": {"human_close@1": round(np.mean([success(r, "human", 1) for r in reached]), 4)}}
    out["validity"]["valid"] = bool(out["validity"]["human_close@1"] >= 0.95)
    table = {}
    for pol in ["auto"] + runs + [r + "+auto" for r in runs]:
        table[pol] = {sp: {f"close@{k}": round(np.mean([success(r, pol, k) for r in reached if r["split"] == sp]), 4)
                           for k in KS} for sp in splits}
        table[pol]["all"] = {f"close@{k}": round(np.mean([success(r, pol, k) for r in reached]), 4) for k in KS}
    out["table"] = table
    c8 = {run: table[run]["all"]["close@8"] for run in runs}
    out["arms"] = {a: {"runs": rs, "close@8": [c8[r] for r in rs], "mean": round(float(np.mean([c8[r] for r in rs])), 4)}
                   for a, rs in arms.items()}
    if "flatS" in arms:
        out["H1"] = {a: welch([c8[r] for r in rs], [c8[r] for r in arms["flatS"]]) | {}
                     for a, rs in arms.items() if a != "flatS" and len(rs) > 1 and len(arms["flatS"]) > 1}
        for a, w in out["H1"].items():
            w["holds"] = bool(w["ci95"][0] > 0)
    auto = [success(r, "auto", 8) for r in reached]
    out["H2"] = {}
    for a, rs in arms.items():
        mix = [np.mean([success(r, run + "+auto", 8) for run in rs]) for r in reached]
        b = paired_bootstrap(mix, auto); b["holds"] = bool(b["ci95"][0] > 0)
        out["H2"][a] = b
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", default="results/c3")
    a = ap.parse_args(argv)
    res = analyse(a.eval)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("validity", "arms", "H1", "H2") if k in res}, indent=1))


if __name__ == "__main__":
    main()
