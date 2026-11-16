"""Independent AF-23 Track B public-incidence, Bellman, and checkpoint audit."""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import random

import torch
from . import branching_gym as gym
from . import branching_planner as planner
from . import branching_train as training
from . import branching_evaluate as evaluation


def public_graph(c):
    """Reconstruct physical Horn incidence from encoded model-visible tensors."""
    x = gym.encode(c)
    prem = x.prem_up + x.prem_down
    head = x.head_up + x.head_down
    rules = {}
    for a in range(gym.RULES):
        if x.rule_features[a, 2] < .5:
            continue
        ps = tuple(torch.where(prem[a] > .5)[0].tolist())
        hs = tuple(torch.where(head[a] > .5)[0].tolist())
        if not ps or len(ps) > 2 or len(hs) != 1:
            raise ValueError("invalid public physical incidence")
        rules[a] = (ps, hs[0])
    initial = frozenset(torch.where(x.atom_features[:, 0] > .5)[0].tolist())
    return rules, initial, int(x.read_slots[2])


def shortest_public(rules, start, goal):
    s = sum(1 << x for x in start)
    if s & (1 << goal):
        return 0
    mask = [
        (sum(1 << p for p in prem), 1 << head)
        for prem, head in rules.values()
    ]
    q = deque([(s, 0)])
    seen = {s}
    while q:
        bits, depth = q.popleft()
        for prem, head in mask:
            if prem & bits != prem or head & bits:
                continue
            next_bits = bits | head
            if next_bits & (1 << goal):
                return depth + 1
            if next_bits not in seen:
                seen.add(next_bits)
                q.append((next_bits, depth + 1))
    return None


def validate_episode(c, row):
    rules, initial, goal = public_graph(c)
    mapping, _ = gym.physical_maps(c)
    tophys = lambda known: frozenset(mapping[i] for i in known)
    tological = lambda physical: sorted(mapping.index(i) for i in physical)
    d = shortest_public(rules, initial, goal)
    if (row["case_id"], row["split"]) != (c.id, c.split):
        raise ValueError("episode case identity mismatch")
    if row["initial"] != tological(initial) or row["optimal_steps"] != d:
        raise ValueError("initial facts or shortest length mismatch")
    if row["goal_truth"] != int(d is not None):
        raise ValueError("independent Horn truth mismatch")
    if row["budget"] != (d + 1 if d is not None else 2):
        raise ValueError("incorrect shortest+1 proof budget")
    if len(row["actions"]) > row["budget"]:
        raise ValueError("episode action count exceeds budget")
    mode = row["mode"]
    if mode not in evaluation.ALL and not (
        mode == "direct" and c.split in ("train", "dev")
    ):
        raise ValueError("unknown scored arm")
    rng = None
    if mode.startswith("random_legal_"):
        stream = int(mode.rsplit("_", 1)[-1])
        rng = random.Random(int(gym._sha([c.id, "random_legal", stream])[:16], 16))
    state = set(initial)
    ended = False
    nodes = 0
    for j, step in enumerate(row["actions"]):
        if ended or step["before"] != tological(state):
            raise ValueError("malformed source host state")
        legal = sorted(i for i, (prem, head) in rules.items()
                       if head not in state and set(prem) <= state)
        if step["enabled"] != legal:
            raise ValueError("incorrect physical legal mask")
        action = step["action"]
        if mode in planner.MODES:
            scores = step["decision_scores"]
            receipt = step["plan_trace"]
            if receipt is None or scores is None:
                raise ValueError("missing neural Bellman receipt")
            chosen = planner.check_trace(
                c, frozenset(step["before"]), receipt, {mode: scores}
            )[mode]
            if chosen != action:
                raise ValueError("saved neural rule selection differs")
            nodes += receipt["nodes_forward_count"]
        else:
            if step["decision_scores"] is not None or step["plan_trace"] is not None:
                raise ValueError("symbolic control includes neural value data")
            if mode == "fixed":
                expected = legal[0] if legal else gym.STOP
            elif mode == "shortest":
                if d is None or goal in state:
                    expected = gym.STOP
                else:
                    remaining = shortest_public(rules, state, goal)
                    candidates = [
                        a for a in legal if shortest_public(
                            rules, frozenset(state) | {rules[a][1]}, goal
                        ) == remaining - 1
                    ]
                    expected = candidates[0] if candidates else gym.STOP
            else:
                expected = rng.choice(legal) if legal else gym.STOP
            if expected != action:
                raise ValueError("symbolic fixed/random/oracle action mismatch")
        if action == gym.STOP:
            if step["proof"] is not None:
                raise ValueError("STOP incorrectly claims a physical proof")
            ended = True
        else:
            if action not in legal:
                raise ValueError("illegal rule fired")
            prem, head = rules[action]
            evidence = step["proof"]
            if (not isinstance(evidence, dict)
                    or evidence.get("rule_slot") != action
                    or evidence.get("head") != mapping.index(head)
                    or sorted(evidence.get("premises", [])) != sorted(
                        mapping.index(p) for p in prem
                    )):
                raise ValueError("forged Horn rule evidence")
            state.add(head)
            if goal in state:
                ended = True
        if step["after"] != tological(state):
            raise ValueError("physical postcondition mismatch")
    solved = goal in state
    if row["final"] != tological(state) or row["solved"] != solved:
        raise ValueError("final proof status mismatch")
    if row["early_stop"] != bool(d is not None and not solved and ended):
        raise ValueError("premature STOP accounting mismatch")
    if row["host_accepted_false_proof"] or (solved and d is None):
        raise ValueError("false accepted proof")
    return nodes


def audit(seed, training_dir, eval_dir, out, smoke=False):
    training_dir = Path(training_dir)
    eval_dir = Path(eval_dir)
    original, model = training.load_model(training_dir, seed, smoke)
    scores = json.loads((eval_dir / "evaluation.json").read_text())
    if (scores["experiment"], scores["track"], scores["seed"], scores["smoke"]) != (
        "AF-23", "B", seed, smoke
    ):
        raise ValueError("wrong evaluation identity")
    if (scores["source_sha256"] != training.source_hashes()
            or scores["protocol_sha256"] != training.digest(training.HERE / "PROTOCOL.json")
            or scores["manifest_sha256"] != gym.manifest()["manifest_sha256"]
            or scores["train_receipt_sha256"] != training.digest(
                training_dir / "train_receipt.json"
            )
            or scores["episodes_sha256"] != training.digest(eval_dir / "episodes.jsonl")):
        raise ValueError("source or episode digest mismatch")
    rows = [json.loads(line) for line in (eval_dir / "episodes.jsonl").read_text().splitlines()]
    index = {}
    for row in rows:
        key = row["mode"], row["split"], row["case_id"]
        if key in index:
            raise ValueError("duplicate AF-23 proof trace")
        index[key] = row
    if smoke:
        splits = ("dev",)
    else:
        splits = ("train", "dev", *evaluation.HELD)
    total = 0
    for split in splits:
        cases = gym.cases_for(split)
        if smoke:
            cases = cases[:16]
        modes = ("direct",) if split in ("train", "dev") else evaluation.ALL
        for c in cases:
            for mode in modes:
                row = index.get((mode, split, c.id))
                if row is None:
                    raise ValueError("missing registered episode")
                total += validate_episode(c, row)
        for mode in modes:
            group = [index[(mode, split, c.id)] for c in cases]
            observed = evaluation.summarize(group)
            for k in ("cases", "provable", "solved",
                      "initial_premature_stop", "initial_correct_stop",
                      "false_accepted_proofs"):
                if scores["scores"][split][mode][k] != observed[k]:
                    raise ValueError("aggregate proof score mismatch")
    expected = (16 if smoke else (
        len(gym.cases_for("train")) + len(gym.cases_for("dev")) +
        sum(len(gym.cases_for(s)) for s in evaluation.HELD) * len(evaluation.ALL)
    ))
    if len(rows) != expected:
        raise ValueError("missing saved proof episode")
    replays = 0
    for split in splits:
        cases = gym.cases_for(split)
        if smoke:
            cases = cases[:16]
        for c in cases[:3]:
            for mode in (("direct",) if split in ("train", "dev") else planner.MODES):
                row = index[(mode, split, c.id)]
                for step in row["actions"][:1]:
                    fresh = planner.decisions(model, [c], [frozenset(step["before"])])[0]
                    a = fresh["scores"][mode]
                    b = step["decision_scores"]
                    if any(abs(x - y) > 1e-5 for x, y in zip(a, b)):
                        raise ValueError("checkpoint forward differs from saved neural scores")
                    replays += 1
    result = {
        "experiment": "AF-23", "track": "B", "seed": seed, "smoke": smoke,
        "verified": True, "proof_episodes": len(rows),
        "planner_node_receipts_checked": total,
        "frozen_checkpoint_forward_replays": replays,
        "false_accepted_proofs": 0,
        "training_sha256": training.digest(training_dir / "train_receipt.json"),
        "evaluation_sha256": training.digest(eval_dir / "evaluation.json"),
        "source_sha256": training.source_hashes(),
        "data_manifest_sha256": gym.manifest()["manifest_sha256"],
    }
    training.write_json(out, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, choices=gym.SEEDS, default=17)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    result = audit(args.seed, args.training, args.evaluation, args.out, args.smoke)
    print(json.dumps({"seed": args.seed, "verified": result["verified"],
                      "proof_episodes": result["proof_episodes"],
                      "nodes": result["planner_node_receipts_checked"]}))


if __name__ == "__main__":
    main()
