"""AF-23 Track A independent physical-incidence and score receipt audit.

Reconstructs each legal transition from AF-22's public encoded incidence, not
the inference helper's rule labels. Recomputes every saved numerical STOP
ablation decision and samples exact frozen-model forwards by source hash.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af22 import gym
from experiments.activation_funnel_af22 import experiment as old
from experiments.activation_funnel_af22 import planner
from experiments.activation_funnel_af22 import verify as original_verify
from . import stop_eval, stop_scoring


def independent_episode(c, row):
    rules, initial, goal = original_verify.public_graph(c)
    atom_perm, _ = gym.physical_maps(c.permutation)
    physical = lambda s: frozenset(atom_perm[n] for n in s)
    logical = lambda s: sorted(atom_perm.index(p) for p in s)
    d = original_verify.shortest_bfs(rules, initial, goal)
    if row["case_id"] != c.id or row["split"] != c.split:
        raise ValueError("case identity mismatch")
    if row["mode"] not in stop_scoring.ARMS:
        raise ValueError("wrong evaluation arm")
    if row["initial"] != logical(initial) or row["optimal_steps"] != d:
        raise ValueError("initial/shortest proof receipt mismatch")
    if row["goal_truth"] != int(d is not None):
        raise ValueError("truth label mismatch")
    if row["budget"] != (d + 1 if d is not None else 2):
        raise ValueError("proof budget mismatch")
    state = set(initial)
    ended = False
    visits = 0
    for i, step in enumerate(row["actions"]):
        if step["before"] != logical(state) or ended:
            raise ValueError("unexpected host state or continuation after terminal")
        legal = sorted(a for a, (pre, head) in rules.items()
                       if head not in state and set(pre) <= state)
        if step["enabled"] != legal:
            raise ValueError("physical host legal mask mismatch")
        receipt = stop_scoring.score(
            step["plan_trace"], step["before"], row["mode"]
        )
        if receipt != step["ablation"] or receipt["chosen"] != step["action"]:
            raise ValueError("ablation action or numerical ledger mismatch")
        # Recompute original VI-2 ledger as a separate tamper check.
        planner.check_trace_math(
            c, frozenset(step["before"]), step["plan_trace"],
            {"vi2": step["original_vi2_scores"]}
        )
        visits += receipt["node_count"]
        action = step["action"]
        if action == gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP carries a fabricated proof")
            ended = True
        else:
            if action not in legal:
                raise ValueError("host rejected illegal neural action")
            pre, head = rules[action]
            evidence = step["proof"]
            if (not isinstance(evidence, dict)
                    or evidence.get("rule_slot") != action
                    or evidence.get("head") != atom_perm.index(head)
                    or sorted(evidence.get("premises", []))
                       != sorted(atom_perm.index(p) for p in pre)):
                raise ValueError("Horn proof evidence mismatch")
            state.add(head)
            if goal in state:
                ended = True
        if step["after"] != logical(state):
            raise ValueError("postcondition mismatch")
    if len(row["actions"]) > row["budget"]:
        raise ValueError("proof exceeds shortest+1 budget")
    if row["final"] != logical(state) or row["solved"] != (goal in state):
        raise ValueError("end-state receipt mismatch")
    if (row["early_stop"]
            != bool(d is not None and goal not in state
                    and row["actions"] and row["actions"][-1]["action"] == gym.STOP)):
        raise ValueError("premature STOP receipt mismatch")
    if row["host_accepted_false_proof"] or (goal in state and d is None):
        raise ValueError("false accepted Horn proof")
    return visits


def audit(zip_dir, seed, out_dir, replay_count=3):
    path = Path(out_dir)
    result = json.loads((path / "af23_ablation_results.json").read_text())
    raw = (path / "af23_ablation_episodes.jsonl").read_bytes()
    if result["seed"] != seed or result["new_episodes_sha256"] != stop_eval.sha(raw):
        raise ValueError("AF23 output identity/hash mismatch")
    if result["protocol_sha256"] != stop_eval.sha((stop_eval.HERE / "PROTOCOL.json").read_bytes()):
        raise ValueError("AF23 protocol/source registration mismatch")
    original, _, weights, zip_sha = stop_eval.original_zip(zip_dir, seed)
    if result["original_zip_sha256"] != zip_sha:
        raise ValueError("AF22 source archive identity mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    index = {}
    visits = 0
    for row in rows:
        key = row["mode"], row["split"], row["case_id"]
        if key in index:
            raise ValueError("duplicated scored proof episode")
        index[key] = row
    expected = sum(len(gym.cases_for(s)) for s in stop_eval.SPLITS) * len(stop_eval.NEW_ARMS)
    if len(rows) != expected:
        raise ValueError("missing new-branch episode")
    for split in stop_eval.SPLITS:
        for c in gym.cases_for(split):
            for mode in stop_eval.NEW_ARMS:
                row = index.get((mode, split, c.id))
                if row is None:
                    raise ValueError("missing registered case")
                visits += independent_episode(c, row)
    for split in stop_eval.SPLITS:
        for mode in stop_eval.NEW_ARMS:
            rs = [index[(mode, split, c.id)] for c in gym.cases_for(split)]
            for field in ("cases", "provable", "solved",
                          "initial_premature_stop", "initial_correct_stop",
                          "false_accepted_proofs"):
                if stop_eval.summarize(rs)[field] != result["scores"][split][mode][field]:
                    raise ValueError("metric mismatch")
    replayed = 0
    with tempfile.TemporaryDirectory(prefix="af23-audit-") as temp:
        dest = Path(temp)
        for name, payload in weights.items():
            p = dest / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(payload)
        _, model = old.load_model(dest / "training", seed, smoke=False)
        for split in stop_eval.SPLITS:
            for mode in stop_eval.NEW_ARMS:
                for c in gym.cases_for(split)[:replay_count]:
                    episode = index[(mode, split, c.id)]
                    for step in episode["actions"][:1]:
                        fresh = planner.decisions(
                            model, [c], [frozenset(step["before"])]
                        )[0]
                        if (len(fresh["scores"]["vi2"]) != len(step["original_vi2_scores"]) or
                                max(abs(a - b) for a, b in zip(
                                    fresh["scores"]["vi2"], step["original_vi2_scores"])) > 1e-5):
                            raise ValueError("checkpoint Torch forward replay differs >1e-5")
                        expected = stop_scoring.score(
                            fresh["trace"], step["before"], mode
                        )
                        if expected["chosen"] != step["action"]:
                            raise ValueError("checkpoint neural action replay differs")
                        replayed += 1
    return {
        "experiment": "AF-23", "track": "A", "seed": seed,
        "original_zip_sha256": zip_sha, "verified": True,
        "cases_checked": len(rows), "planner_node_receipts_checked": visits,
        "checkpoint_forward_replays": replayed,
        "false_accepted_proofs": 0,
        "source_mode": "registered original AF22 checkpoints",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True, choices=stop_eval.SEEDS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=3)
    args = parser.parse_args()
    result = audit(args.zip_dir, args.seed, args.out_dir, args.replays)
    path = args.out_dir / "af23_independent_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
