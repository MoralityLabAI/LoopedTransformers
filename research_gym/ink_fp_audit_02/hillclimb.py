"""Traced hill climb over analytic adapter-update configurations (evidence class: synthetic_cpu).

State: {method, rank, lam, keep_weight-fixed utility}. Proposals are seeded neighbours. Each
proposal is evaluated on paired development cases (conflict level x dev seed) and accepted
under a frozen continuous rule. Every step, accepted or rejected, is written as a hill-climb
trace. The final incumbent is compared once against the start on fresh held-out seeds.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import resources  # noqa: F401
import numpy as np

from . import bpd_proxy as bp
from . import traces

CRITERIA = {"version": "ink02.hillclimb.bpd.v1", "min_mean_gain": 0.005, "max_family_drop": 0.05,
            "utility": "new_heldout_r2 - 2 * keep_heldout_rel"}
CONFLICTS = (0.1, 0.5, 0.9)
METHODS = ("ridge", "tsvd", "projected", "projected_lowrank")
START = {"method": "ridge", "rank": 8, "lam": 10.0}


def apply(cfg: dict, d: dict):
    m = cfg["method"]
    if m == "ridge":
        return bp.m_ridge(d, lam=cfg["lam"])
    if m == "tsvd":
        return bp.m_tsvd(d, rank=cfg["rank"])
    if m == "projected":
        return bp.m_projected(d)
    if m == "projected_lowrank":
        return bp.m_projected(d, rank=cfg["rank"])
    raise ValueError(m)


def cases(cfg: dict, seeds) -> list[dict]:
    out = []
    for c in CONFLICTS:
        for s in seeds:
            d = bp.make(bp.Scenario(conflict=c), np.random.default_rng(s))
            m = bp.metrics(d, apply(cfg, d))
            out.append({"case_id": f"c{c}-s{s}", "family": f"conflict_{c}",
                        "value": m["new_heldout_r2"] - 2.0 * m["keep_heldout_rel"]})
    return out


def neighbours(cfg: dict, rng: random.Random) -> tuple[dict, str]:
    moves = []
    if cfg["method"] in ("tsvd", "projected_lowrank"):
        moves += [("rank", max(1, cfg["rank"] + d)) for d in (-4, -2, -1, 1, 2, 4)]
        moves += [("rank", max(1, cfg["rank"] // 2))]
    if cfg["method"] == "ridge":
        moves += [("lam", cfg["lam"] * f) for f in (10, 3, 1 / 3, 1 / 10)]
    moves += [("method", m) for m in METHODS if m != cfg["method"]]
    key, val = rng.choice(moves)
    new = {**cfg, key: val}
    return new, f"change {key} {cfg[key]} -> {val}"


def climb(steps: int = 24, dev_seeds=range(300, 306), heldout_seeds=range(400, 420), seed: int = 0,
          out: Path | None = None) -> dict:
    rng = random.Random(seed)
    inc, inc_cases = dict(START), cases(START, dev_seeds)
    tried, rows = {json.dumps(START, sort_keys=True)}, []
    for k in range(steps):
        for _ in range(32):  # redraw until an untried neighbour appears
            prop, change = neighbours(inc, rng)
            key = json.dumps(prop, sort_keys=True)
            if key not in tried:
                break
        else:
            break  # neighbourhood exhausted: local optimum under this move set
        tried.add(key)
        cand = cases(prop, dev_seeds)
        d = traces.continuous_decide(CRITERIA, inc_cases, cand)
        mean = lambda cs: sum(x["value"] for x in cs) / len(cs)
        rows.append(traces.step(
            "bpd_hillclimb", "synthetic_cpu", f"seed{seed}", len(rows),
            {"id": json.dumps(inc, sort_keys=True), "metrics": {"mean_utility": mean(inc_cases)}},
            {"id": key, "change": {"move": change, "config": prop}},
            {"metrics": {"mean_utility": mean(cand), "family_gain": d["family_gain"]}, "paired_n": len(cand)},
            CRITERIA, d["checks"], d["logic"], "accept" if d["accepted"] else "reject",
            next_incumbent=key if d["accepted"] else json.dumps(inc, sort_keys=True)))
        if d["accepted"]:
            inc, inc_cases = prop, cand
    start_h, final_h = cases(START, heldout_seeds), cases(inc, heldout_seeds)
    held = traces.continuous_decide(CRITERIA, start_h, final_h)
    report = {"evidence_class": "synthetic_cpu", "criteria": CRITERIA, "start": START, "final": inc,
              "steps": len(rows), "neighbourhood_exhausted": len(rows) < steps, "accepted": sum(r["decision_by_rule"] == "accept" for r in rows),
              "dev_seeds": list(dev_seeds), "heldout_seeds": list(heldout_seeds),
              "heldout_final_vs_start": {"mean_gain": held["mean_gain"], "family_gain": held["family_gain"],
                                         "passes_rule": held["accepted"]},
              "selection_exposure": "dev seeds reused for every step; held-out seeds read once"}
    if out:
        traces.write(rows, out / f"bpd_hillclimb_seed{seed}.jsonl")
        (out / f"bpd_hillclimb_seed{seed}_report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                                     encoding="utf-8", newline="\n")
    return {"report": report, "rows": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--climbs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for s in range(a.climbs):
        r = climb(a.steps, seed=s, out=a.out)["report"]
        print(json.dumps({"seed": s, "final": r["final"], "accepted": r["accepted"], "steps": r["steps"],
                          "heldout_gain": round(r["heldout_final_vs_start"]["mean_gain"], 4)}))


if __name__ == "__main__":
    main()
