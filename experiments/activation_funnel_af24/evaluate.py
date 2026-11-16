"""AF-24 registered evaluation: public Horn control policies vs frozen AF-23.

The controller receives only PublicHorn and host-known physical atoms. The
scoring harness separately obtains an independently computed proof distance
for the fixed episode budget. All actions receive independent legality and
end-state checks against the actual AF-23 Horn host.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from experiments.activation_funnel_af23 import branching_gym as gym
from . import symbolic as ctl

HERE = Path(__file__).resolve().parent
SPLITS = ("held_depth4", "held_depth5")
SEEDS = gym.SEEDS
ARMS = (*ctl.CORE, *ctl.RANDOM, "fixed", "shortest")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def independent_truth(c):
    """Scorer/auditor only; not visible to the controller during inference."""
    distance = gym.independent_bfs(c)
    if distance != gym.shortest_steps(c):
        raise ValueError("two independent physical Horn BFS results disagree")
    return distance


def exact_oracle_action(graph, state, c, distance):
    """Evaluation-only shortest cost upper control, clearly labeled oracle."""
    if distance is None or distance == 0:
        return None
    apos, _ = gym.physical_maps(c)
    physical_to_logical = {physical: logical for logical, physical in enumerate(apos)}
    current_logical = frozenset(physical_to_logical[x] for x in state)
    d = gym.shortest_steps(c, current_logical)
    if d is None or d == 0:
        raise AssertionError("evaluation-only shortest oracle called off a valid path")
    for slot, _ in graph.enabled(state):
        new_state = graph.fire(state, slot)
        logical = frozenset(physical_to_logical[x] for x in new_state)
        if gym.shortest_steps(c, logical) == d - 1:
            return slot
    raise AssertionError("public shortest oracle failed to find a legal cost decrease")


def episode(c, mode, seed=17, proof_distance=None):
    if mode not in ARMS:
        raise ValueError("unknown registered symbolic arm")
    features = gym.public_features(c)
    graph = ctl.PublicHorn.from_public_features(features)
    known = graph.initial
    if proof_distance is None:
        d = independent_truth(c)
    else:
        d = proof_distance
    budget = (d + 1) if d is not None else 2
    actions = []
    apos, _ = gym.physical_maps(c)
    logical_by_physical = {physical: logical for logical, physical in enumerate(apos)}
    for step in range(budget):
        if graph.goal in known:
            break
        if mode == "shortest":
            action = exact_oracle_action(graph, known, c, d)
            details = {"arm": mode, "action": action, "expanded": 0,
                       "rule_examinations": 0, "closure_passes": 0,
                       "search_exhausted": True, "wall_ns": 0,
                       "oracle_only": True}
        else:
            details = ctl.choose(graph, known, mode, seed=seed, case_id=c.id, step=step)
            action = details["action"]
        before = sorted(known)
        if action is None:
            after = known
            proof = None
        else:
            # Controller execution is checked by independent host-rule ledger
            # using the original logical/physical mapping.
            logical_before = frozenset(logical_by_physical[x] for x in known)
            host_after, proof = gym.sound_fire(c, logical_before, action)
            after = graph.fire(known, action)
            if after != frozenset(apos[x] for x in host_after):
                raise ValueError("symbolic public transition disagrees with host")
        actions.append({"before": before, "after": sorted(after), "action": action,
                        "proof": proof, "work": details})
        known = after
        if action is None or graph.goal in known:
            break
    solved = graph.goal in known
    return {
        "experiment": "AF-24", "case_id": c.id, "split": c.split,
        "seed": seed, "mode": mode,
        "truth": d is not None, "optimal_steps": d, "budget": budget,
        "solved": solved, "premature_stop": bool(
            d is not None and not solved and actions and actions[-1]["action"] is None
        ),
        "first_stop": bool(actions and actions[0]["action"] is None),
        "false_accepted_proof": bool(solved and d is None),
        "initial": sorted(graph.initial), "final": sorted(known),
        "actions": actions,
        "node_expansions": sum(a["work"]["expanded"] for a in actions),
        "rule_examinations": sum(a["work"]["rule_examinations"] for a in actions),
        "closure_passes": sum(a["work"]["closure_passes"] for a in actions),
        "symbolic_wall_ns": sum(a["work"]["wall_ns"] for a in actions),
    }


def summarize(rows):
    positives = [r for r in rows if r["truth"]]
    negatives = [r for r in rows if not r["truth"]]
    if len(positives) != 80 or len(negatives) != 112:
        raise ValueError("AF-23 heldout split balance changed")
    return {
        "cases": len(rows), "provable": len(positives), "unprovable": len(negatives),
        "solved": sum(r["solved"] for r in positives),
        "proof_success": sum(r["solved"] for r in positives) / len(positives),
        "premature_stop": sum(r["premature_stop"] for r in positives),
        "correct_initial_stop": sum(r["first_stop"] for r in negatives),
        "false_accepted_proofs": sum(r["false_accepted_proof"] for r in rows),
        "node_expansions_total": sum(r["node_expansions"] for r in rows),
        "node_expansions_median": median(r["node_expansions"] for r in rows),
        "node_expansions_max": max(r["node_expansions"] for r in rows),
        "rule_examinations_total": sum(r["rule_examinations"] for r in rows),
        "rule_examinations_median": median(r["rule_examinations"] for r in rows),
        "symbolic_wall_ns_total": sum(r["symbolic_wall_ns"] for r in rows),
        "symbolic_wall_ns_median": median(r["symbolic_wall_ns"] for r in rows),
    }


def evaluate(seed, out):
    if seed not in SEEDS:
        raise ValueError("unregistered evaluation seed")
    out = Path(out)
    if out.exists():
        raise FileExistsError("AF-24 evaluation outputs must be fresh")
    out.mkdir(parents=True)
    all_rows = []
    results = {}
    for split in SPLITS:
        cases = gym.cases_for(split)
        distances = [independent_truth(c) for c in cases]
        if sum(d is not None for d in distances) != 80:
            raise ValueError("registered provable case count changed")
        result = {}
        for mode in ARMS:
            rows = [episode(c, mode, seed, d) for c, d in zip(cases, distances)]
            result[mode] = summarize(rows)
            all_rows.extend(rows)
        if result["fixed"]["solved"] != (4 if split == "held_depth4" else 2):
            raise ValueError("AF-23 frozen fixed baseline mismatch")
        if result["shortest"]["solved"] != 80:
            raise ValueError("AF-23 shortest upper control mismatch")
        results[split] = result
    data = "".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n"
                   for r in all_rows).encode()
    (out / "episodes.jsonl").write_bytes(data)
    protocol_hash = digest((HERE / "PROTOCOL.json").read_bytes())
    receipt = {
        "experiment": "AF-24", "seed": seed,
        "original_af23_ci": 43812708916,
        "source_protocol_sha256": protocol_hash,
        "source_manifest_sha256": gym.manifest()["manifest_sha256"],
        "source_branching_grammar_hash": digest(
            (HERE.parent / "activation_funnel_af23/branching_gym.py").read_bytes()
        ),
        "episodes_sha256": digest(data),
        "episode_count": len(all_rows),
        "arm_names": list(ARMS),
        "scores": results,
        "scientific_verdict": "pending_eight_seed_aggregate",
        "model_training": False,
    }
    (out / "evaluation.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    r = evaluate(args.seed, args.out)
    print(json.dumps({"seed": r["seed"], "episodes": r["episode_count"],
                      "fixed_d4": r["scores"]["held_depth4"]["fixed"]["solved"],
                      "fixed_d5": r["scores"]["held_depth5"]["fixed"]["solved"]}))


if __name__ == "__main__":
    main()
