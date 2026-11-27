"""Adaptive driver for DataGoblin MeTTa/ram_hillclimb.py (run with DataGoblin on sys.path).

Uses that module's own make_bundle, fit_candidate, validate_proposal, select_key, OPTIONS,
DEFAULTS and ARMS. The accept rule below restates run_gym's three lines exactly (improved by
select_key and no new violation / unsafe additions); `--verify-against-run-gym` re-runs the
module's DEFAULT_PROPOSALS through run_gym and checks the decisions agree.

Proposers:
  seeded_random      uniform over valid non-no-op edits in the bounded grammar (control)
  coordinate_sweep   cycles fields and tries each untried neighbouring value (heuristic)
  file               reads a proposal per turn from a JSONL queue (e.g. Inky via Tinker)
Standalone script: imports nothing from LoopedTransformers.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")


def load():
    from MeTTa import ram_hillclimb as rh
    return rh


def decide(rh, prev: dict, new: dict) -> tuple[bool, str]:
    safety = (new["new_violation_additions"] > prev["new_violation_additions"]
              or new["unsafe_additions"] > prev["unsafe_additions"])
    improved = rh.select_key(new) > rh.select_key(prev)
    accepted = improved and not safety
    return accepted, ("reject_safety_regression" if safety else
                      "accept_verified_development_gain" if accepted else "reject_no_gain")


def valid_edits(rh, current: dict, arm: str) -> list[tuple[str, object]]:
    return [(f, v) for f, vals in rh.OPTIONS[arm].items() for v in vals if current[arm][f] != v]


class SeededRandom:
    name = "seeded_random"

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def propose(self, rh, current, history, tried):
        arm = self.rng.choice(sorted(rh.ARMS))
        opts = [e for e in valid_edits(rh, current, arm) if (arm, e, json.dumps(current[arm], sort_keys=True)) not in tried]
        if not opts:
            return None
        f, v = self.rng.choice(opts)
        return {"arm": arm, "field": f, "value": v}


class CoordinateSweep:
    name = "coordinate_sweep"

    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.order = None

    def propose(self, rh, current, history, tried):
        if self.order is None:
            self.order = [(a, f) for a in sorted(rh.ARMS) for f in rh.OPTIONS[a]]
            self.rng.shuffle(self.order)
        for _ in range(len(self.order)):
            arm, f = self.order.pop(0)
            self.order.append((arm, f))
            vals = list(rh.OPTIONS[arm][f])
            i = vals.index(current[arm][f])
            for j in (i + 1, i - 1) if self.rng.random() < 0.5 else (i - 1, i + 1):
                if 0 <= j < len(vals):
                    key = (arm, (f, vals[j]), json.dumps(current[arm], sort_keys=True))
                    if key not in tried:
                        return {"arm": arm, "field": f, "value": vals[j]}
        return None


class FileQueue:
    name = "file"

    def __init__(self, path):
        self.rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]

    def propose(self, rh, current, history, tried):
        return {k: v for k, v in self.rows.pop(0).items() if k in ("arm", "field", "value")} if self.rows else None


def climb(rh, proposer, steps: int, seed: int, train_roots: int, dev_roots: int, ram_limit_mib: int) -> list[dict]:
    bundle = rh.make_bundle(train_roots, dev_roots)
    current = {a: dict(rh.DEFAULTS[a]) for a in rh.ARMS}
    best = {a: rh.fit_candidate(a, current[a], bundle["train"], bundle["dev"], seed=seed,
                                ram_limit_mib=ram_limit_mib)[1] for a in rh.ARMS}
    history, tried = [], set()
    for turn in range(steps):
        p = proposer.propose(rh, current, history, tried)
        if p is None:
            break
        arm = p["arm"]
        tried.add((arm, (p["field"], p["value"]), json.dumps(current[arm], sort_keys=True)))
        p = {**p, "source": proposer.name, "model_id": "none"}
        updated = rh.validate_proposal(p, current)
        t0 = time.monotonic()
        _, cand = rh.fit_candidate(arm, updated, bundle["train"], bundle["dev"], seed=seed, ram_limit_mib=ram_limit_mib)
        prev, new = best[arm]["dev_aggregate"], cand["dev_aggregate"]
        accepted, decision = decide(rh, prev, new)
        history.append({"schema_version": "ink02-metta-hillclimb-step-v1", "turn": turn, "arm": arm,
                        "proposer": proposer.name, "proposal": {k: p[k] for k in ("arm", "field", "value")},
                        "incumbent_config": dict(current[arm]), "incumbent_metrics": prev,
                        "candidate_config": updated, "candidate_metrics": new,
                        "candidate_checkpoint_sha256": cand.get("checkpoint_sha256"),
                        "accepted": accepted, "decision": decision, "fit_seconds": time.monotonic() - t0,
                        "select_key_fields": ["solved_and_stopped", "-new_violation_additions", "-unsafe_additions",
                                              "exact_first_edit", "-premature_stops", "-verifier_queries",
                                              "-mean_remaining_canonical_edits", "-mean_inference_iterations"]})
        if accepted:
            current[arm], best[arm] = updated, cand
    return history


def verify_against_run_gym(rh, out: Path, seed: int) -> dict:
    res = rh.run_gym(out / "run_gym_reference", seed=seed, ram_limit_mib=1024)
    ref = [json.loads(x) for x in (out / "run_gym_reference" / "hillclimb_steps.jsonl").read_text(encoding="utf-8").splitlines()]
    mine = climb(rh, FileQueue_from(rh.DEFAULT_PROPOSALS), len(rh.DEFAULT_PROPOSALS), seed, 8, 3, 1024)
    agree = [a["accepted"] == b["accepted"] and a["decision"] == b["decision"] for a, b in zip(ref, mine)]
    return {"steps": len(ref), "agree": all(agree) and len(ref) == len(mine), "reference_accepted": res["accepted_edits"]}


class _ListQueue(FileQueue):
    def __init__(self, rows):
        self.rows = [dict(r) for r in rows]


def FileQueue_from(rows):
    q = _ListQueue(rows)
    q.name = "deterministic_control"
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--proposer", choices=["seeded_random", "coordinate_sweep", "file"], default="seeded_random")
    ap.add_argument("--queue", type=Path)
    ap.add_argument("--climbs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--train-roots", type=int, default=8)
    ap.add_argument("--dev-roots", type=int, default=3)
    ap.add_argument("--ram-limit-mib", type=int, default=1024)
    ap.add_argument("--verify-against-run-gym", action="store_true")
    a = ap.parse_args()
    rh = load()
    a.out.mkdir(parents=True, exist_ok=True)
    if a.verify_against_run_gym:
        v = verify_against_run_gym(rh, a.out, a.seed)
        (a.out / "rule_equivalence.json").write_text(json.dumps(v, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(v))
    for c in range(a.climbs):
        prop = (SeededRandom(a.seed + c) if a.proposer == "seeded_random" else
                CoordinateSweep(a.seed + c) if a.proposer == "coordinate_sweep" else FileQueue(a.queue))
        hist = climb(rh, prop, a.steps, a.seed, a.train_roots, a.dev_roots, a.ram_limit_mib)
        path = a.out / f"metta_{a.proposer}_c{c}.jsonl"
        path.write_text("".join(json.dumps(h, sort_keys=True) + "\n" for h in hist), encoding="utf-8", newline="\n")
        print(json.dumps({"climb": c, "proposer": a.proposer, "steps": len(hist),
                          "accepted": sum(h["accepted"] for h in hist)}))


if __name__ == "__main__":
    main()
