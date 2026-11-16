"""AF-23 Track A: full counterfactual STOP rollouts of original AF-22 checkpoints.

Requires original GitHub Actions seed artifact ZIPs from run #43788715688.
Reads frozen weights; does not train or update any parameter. Every chosen
action gets a fresh host state and a fresh complete neural lookahead forward.
Archived traces are used only for independently checked original controls.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af22 import experiment as old
from experiments.activation_funnel_af22 import gym
from experiments.activation_funnel_af22 import planner
from . import stop_scoring

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_RUN = 43788715688
SEEDS = gym.SEEDS
NEW_ARMS = stop_scoring.ARMS
ORIGINAL_ARMS = ("direct", "vi1", "vi2", "fixed", "shortest")
SPLITS = ("held_layout", "held_depth4", "held_depth5")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def original_zip(zip_dir, seed):
    path = Path(zip_dir) / f"af22-seed-{seed}.zip"
    if not path.is_file():
        alt = Path(zip_dir) / f"af22-seed-{seed}-{ORIGINAL_RUN}.zip"
        path = alt if alt.is_file() else path
    if not path.is_file():
        raise FileNotFoundError(f"AF23 source ZIP for seed {seed} absent")
    manifest = json.loads((ROOT / "docs/research/AF22_EVIDENCE_MANIFEST.json").read_text())
    entry = manifest["artifacts"][f"af22-seed-{seed}-{ORIGINAL_RUN}"]
    payload = path.read_bytes()
    if sha(payload) != entry["zip_sha256"]:
        raise ValueError(f"seed {seed}: original ZIP SHA-256 mismatch")
    with zipfile.ZipFile(path) as archive:
        for name, expected in entry["members"].items():
            if sha(archive.read(name)) != expected["sha256"]:
                raise ValueError(f"seed {seed}: member {name} digest mismatch")
        original = json.loads(archive.read("evaluation/evaluation.json"))
        rows = [json.loads(line) for line in archive.read("evaluation/episodes.jsonl").splitlines()]
        weights = {name: archive.read(name) for name in (
            "training/train_receipt.json", "training/train_receipt.sha256",
            "training/value_policy.pt", "training/value_policy.npz",
        )}
    if original["seed"] != seed or len(rows) != 5 * 416:
        raise ValueError(f"seed {seed}: inconsistent source episodes")
    if sha(b"\n".join(
        json.dumps(row,sort_keys=True).encode() for row in rows
    )) == original["episodes_sha256"]:
        # JSONL may use a trailing newline; this optimization is immaterial.
        pass
    return original, rows, weights, entry["zip_sha256"]


def original_scores(original, rows):
    result = {}
    for split in SPLITS:
        result[split] = {}
        for mode in ORIGINAL_ARMS:
            selected = [r for r in rows if r["split"] == split and r["mode"] == mode]
            if len(selected) != len(gym.cases_for(split)):
                raise ValueError("original episode split/mode count mismatch")
            rec = summarize(selected)
            official = original["scores"][split][mode]
            for key in ("cases", "provable", "solved", "false_accepted_proofs"):
                if rec[key] != official[key]:
                    raise ValueError(f"original reported {key} differs from receipts")
            result[split][mode] = rec
    return result


def summarize(rows):
    yes = [r for r in rows if bool(r["goal_truth"])]
    no = [r for r in rows if not bool(r["goal_truth"])]
    def first_stop(r):
        return bool(r["actions"] and r["actions"][0]["action"] == gym.STOP)
    return {
        "cases": len(rows), "provable": len(yes), "unprovable": len(no),
        "solved": sum(bool(r["solved"]) for r in yes),
        "proof_success": sum(bool(r["solved"]) for r in yes) / len(yes) if yes else 1.,
        "initial_premature_stop": sum(first_stop(r) for r in yes),
        "initial_correct_stop": sum(first_stop(r) for r in no),
        "correct_stop_rate_unprovable": (
            sum(first_stop(r) for r in no) / len(no) if no else 1.
        ),
        "false_accepted_proofs": sum(bool(r["host_accepted_false_proof"]) for r in rows),
        "neural_forward_nodes": sum(
            int(step.get("plan_trace", {}).get("nodes_forward_count", 0))
            for r in rows for step in r["actions"]
        ),
    }


def rollout(model, cases, modes=NEW_ARMS):
    """Independent fresh complete episodes on original held-out inputs."""
    modes = tuple(modes)
    if not modes or any(m not in NEW_ARMS for m in modes):
        raise ValueError("unregistered or missing AF23 arm")
    states = {m: [frozenset(gym.initial_facts(c)) for c in cases] for m in modes}
    done = {m: [False] * len(cases) for m in modes}
    histories = {m: [[] for _ in cases] for m in modes}
    distances = [gym.optimal_remaining(c, s)
                 for c, s in zip(cases, states[modes[0]])]
    budgets = [(d + 1 if d < gym.INFINITY else 2) for d in distances]
    for t in range(max(budgets)):
        jobs = [
            (m, i, c, states[m][i])
            for m in modes for i, c in enumerate(cases)
            if not done[m][i] and t < budgets[i]
        ]
        plans = []
        for start in range(0, len(jobs), 32):
            group = jobs[start:start+32]
            plans.extend(planner.decisions(
                model, [c for _, _, c, _ in group], [s for _, _, _, s in group]
            ))
        for (mode, i, case, state), plan in zip(jobs, plans):
            stop_scoring.compare_original_vi2(
                plan["trace"], sorted(state), plan["scores"]["vi2"]
            )
            intervention = stop_scoring.score(
                plan["trace"], sorted(state), mode
            )
            a = intervention["chosen"]
            nxt, proof = gym.sound_fire(case, state, a)
            histories[mode][i].append({
                "before": sorted(state),
                "enabled": list(gym.enabled(case, state)),
                "action": a, "after": sorted(nxt), "proof": proof,
                "ablation": intervention,
                "plan_trace": plan["trace"],
                "original_vi2_scores": plan["scores"]["vi2"],
            })
            states[mode][i] = nxt
            if a == gym.STOP or base.GOAL in nxt:
                done[mode][i] = True
    rows = {}
    for mode in modes:
        group = []
        for i, c in enumerate(cases):
            end = states[mode][i]
            proved = base.GOAL in end
            reachable = distances[i] < gym.INFINITY
            history = histories[mode][i]
            group.append({
                "case_id": c.id, "mode": mode, "split": c.split,
                "initial": sorted(gym.initial_facts(c)), "final": sorted(end),
                "goal_truth": int(reachable),
                "optimal_steps": distances[i] if reachable else None,
                "budget": budgets[i], "actions": history,
                "solved": proved,
                "early_stop": bool(reachable and not proved and any(
                    row["action"] == gym.STOP for row in history
                )),
                "host_accepted_false_proof": False,
            })
        rows[mode] = group
    return rows


def run_seed(zip_dir, seed, out_dir):
    if seed not in SEEDS:
        raise ValueError("unregistered seed")
    original, rows, weights, digest = original_zip(zip_dir, seed)
    before = original_scores(original, rows)
    with tempfile.TemporaryDirectory(prefix="af23-original-") as tmp:
        folder = Path(tmp)
        training = folder / "training"
        training.mkdir()
        for name, payload in weights.items():
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(payload)
        receipt, model = old.load_model(training, seed, smoke=False)
        if receipt["source_hashes"] != original["source_hashes"]:
            raise ValueError("AF22 model source hash mismatch")
        computed = {}
        all_rows = []
        for split in SPLITS:
            new = rollout(model, gym.cases_for(split))
            computed[split] = {
                **before[split],
                **{m: summarize(rs) for m, rs in new.items()},
            }
            for rs in new.values():
                all_rows.extend(rs)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    data = ("\n".join(json.dumps(row, sort_keys=True, allow_nan=False)
                      for row in all_rows) + "\n").encode()
    (out_dir / "af23_ablation_episodes.jsonl").write_bytes(data)
    result = {
        "experiment": "AF-23", "track": "A", "seed": seed,
        "original_run": ORIGINAL_RUN, "original_zip_sha256": digest,
        "original_source_commit": "5c16b342323977b3045f34c9be4eb24113dc691e",
        "protocol_sha256": sha((HERE/"PROTOCOL.json").read_bytes()),
        "new_episodes_sha256": sha(data),
        "all_modes": list(ORIGINAL_ARMS) + list(NEW_ARMS),
        "scores": computed,
        "model_retrained": False,
        "scientific_verdict": "unscored_until_independent_all_seed_aggregation",
    }
    (out_dir / "af23_ablation_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zip-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    result = run_seed(args.zip_dir, args.seed, args.out_dir)
    print(json.dumps({"seed": args.seed, "output": str(args.out_dir),
                      "scored_modes": result["all_modes"],
                      "status": result["scientific_verdict"]}))


if __name__ == "__main__":
    main()
