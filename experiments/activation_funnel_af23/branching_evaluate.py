"""AF-23 Track B: held-out direct/VI1/VI2, five random, fixed and oracle.

Every action executes via a legal physical Horn rule; oracle shortest costs
only set the episode budget and audit targets. All neural policies reuse the
same final 6036-parameter checkpoint from the corresponding seed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from . import branching_gym as gym
from . import branching_planner as planner
from . import branching_train as train

NEURAL = ("direct", "vi1", "vi2")
RANDOM = tuple(f"random_legal_{i}" for i in range(1, 6))
BASELINES = ("fixed", *RANDOM, "shortest")
ALL = (*NEURAL, *BASELINES)
HELD = ("held_layout", "held_depth4", "held_depth5")


def neural_rollout(model, cases, modes=NEURAL):
    if any(m not in NEURAL for m in modes) or not modes:
        raise ValueError("invalid AF23 neural evaluation modes")
    modes = tuple(modes)
    initial = [gym.initial_facts(c) for c in cases]
    distances = [gym.shortest_steps(c) for c in cases]
    budgets = [(d + 1 if d is not None else 2) for d in distances]
    states = {m: list(initial) for m in modes}
    done = {m: [False] * len(cases) for m in modes}
    histories = {m: [[] for _ in cases] for m in modes}
    if not budgets:
        raise ValueError("empty evaluation case set")
    for step in range(max(budgets)):
        jobs = [
            (m, i, c, states[m][i])
            for m in modes for i, c in enumerate(cases)
            if not done[m][i] and step < budgets[i]
        ]
        plans = []
        for start in range(0, len(jobs), 32):
            batch = jobs[start:start+32]
            plans.extend(planner.decisions(
                model, [c for _, _, c, _ in batch], [s for _, _, _, s in batch]
            ))
        for (m, i, c, state), plan in zip(jobs, plans):
            action = plan["selected"][m]
            future, proof = gym.sound_fire(c, state, action)
            histories[m][i].append({
                "before": sorted(state), "enabled": list(gym.enabled(c, state)),
                "after": sorted(future), "action": action, "proof": proof,
                "decision_scores": plan["scores"][m],
                "plan_trace": plan["trace"],
            })
            states[m][i] = future
            if action == gym.STOP or gym.GOAL in future:
                done[m][i] = True
    results = {}
    for m in modes:
        rows = []
        for i, c in enumerate(cases):
            known = states[m][i]
            actions = histories[m][i]
            d = distances[i]
            rows.append({
                "mode": m, "case_id": c.id, "split": c.split,
                "initial": sorted(initial[i]), "final": sorted(known),
                "goal_truth": int(d is not None), "optimal_steps": d,
                "budget": budgets[i], "actions": actions,
                "solved": gym.GOAL in known,
                "early_stop": bool(d is not None and gym.GOAL not in known and
                                   any(a["action"] == gym.STOP for a in actions)),
                "host_accepted_false_proof": False,
            })
        results[m] = rows
    return results


def symbolic_rows(cases, mode):
    if mode not in BASELINES:
        raise ValueError("unregistered symbolic baseline")
    rows = []
    for c in cases:
        if mode.startswith("random_legal_"):
            stream = int(mode.rsplit("_", 1)[-1])
            record = gym.symbolic_rollout(c, "random_legal", stream)
        else:
            record = gym.symbolic_rollout(c, mode)
        actions = []
        for action in record["actions"]:
            actions.append({
                **action, "enabled": list(gym.enabled(c, frozenset(action["before"]))),
                "decision_scores": None, "plan_trace": None,
            })
        rows.append({
            "mode": mode, "case_id": c.id, "split": c.split,
            "initial": sorted(gym.initial_facts(c)),
            "final": (
                actions[-1]["after"] if actions else sorted(gym.initial_facts(c))
            ),
            "goal_truth": int(record["truth"]), "optimal_steps": record["optimal_steps"],
            "budget": record["budget"], "actions": actions,
            "solved": record["solved"],
            "early_stop": bool(record["truth"] and not record["solved"]
                               and any(a["action"] == gym.STOP for a in actions)),
            "host_accepted_false_proof": False,
        })
    return rows


def summarize(rows):
    good = [r for r in rows if r["goal_truth"]]
    impossible = [r for r in rows if not r["goal_truth"]]
    def first_stop(row):
        return bool(row["actions"] and row["actions"][0]["action"] == gym.STOP)
    return {
        "cases": len(rows), "provable": len(good), "unprovable": len(impossible),
        "solved": sum(bool(r["solved"]) for r in good),
        "proof_success": (
            sum(bool(r["solved"]) for r in good) / len(good) if good else 1.
        ),
        "initial_premature_stop": sum(first_stop(r) for r in good),
        "initial_correct_stop": sum(first_stop(r) for r in impossible),
        "correct_stop_rate_unprovable": (
            sum(first_stop(r) for r in impossible) / len(impossible)
            if impossible else 1.
        ),
        "false_accepted_proofs": sum(bool(r["host_accepted_false_proof"])
                                      for r in rows),
        "neural_forward_nodes": sum(
            int((step.get("plan_trace") or {}).get("nodes_forward_count", 0))
            for r in rows for step in r["actions"]
        ),
    }


def evaluate(seed, training_dir, out_dir, smoke=False):
    if seed not in train.SEEDS:
        raise ValueError("unregistered seed")
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError("evaluation output must be fresh")
    out_dir.mkdir(parents=True)
    receipt, model = train.load_model(training_dir, seed, smoke=smoke)
    if smoke:
        splits = ("dev",)
    else:
        splits = ("train", "dev", *HELD)
    score = {}
    rows = []
    for split in splits:
        cases = gym.cases_for(split)
        if smoke:
            cases = cases[:16]
        if split in ("train", "dev"):
            results = neural_rollout(model, cases, ("direct",))
        else:
            results = neural_rollout(model, cases)
            for arm in BASELINES:
                results[arm] = symbolic_rows(cases, arm)
        score[split] = {m: summarize(group) for m, group in results.items()}
        for group in results.values():
            rows.extend(group)
    path = out_dir / "episodes.jsonl"
    path.write_text("".join(
        json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows
    ))
    verdict = ("pre_scoring_smoke" if smoke
               else "awaiting_independent_eight_seed_aggregation")
    receipt = {
        "experiment": "AF-23", "track": "B", "seed": seed, "smoke": smoke,
        "train_receipt_sha256": train.digest(Path(training_dir) / "train_receipt.json"),
        "source_sha256": train.source_hashes(),
        "protocol_sha256": train.digest(train.HERE / "PROTOCOL.json"),
        "manifest_sha256": gym.manifest()["manifest_sha256"],
        "episodes_sha256": train.digest(path), "episode_count": len(rows),
        "scores": score, "scientific_verdict": verdict,
        "all_heldout_arms": list(ALL),
    }
    train.write_json(out_dir / "evaluation.json", receipt)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, choices=train.SEEDS, default=17)
    ap.add_argument("--training", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    receipt = evaluate(args.seed, args.training, args.out_dir, args.smoke)
    print(json.dumps({"seed": args.seed, "modes": receipt["all_heldout_arms"],
                      "status": receipt["scientific_verdict"],
                      "splits": list(receipt["scores"])}))


if __name__ == "__main__":
    main()
