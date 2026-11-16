#!/usr/bin/env python3
"""Replay AF-22 five-arm and STOP-control counts from original Actions ZIPs.

Usage:
    python docs/research/af22_original_receipt_audit.py /path/to/downloaded/zips

Downloads: GitHub Actions run 43788715688, seed artifacts 17/29/43/61/73/89/103/131.
Requires Python 3.10+ standard library only. No training, GPU, Torch, or imports
from the experiment source. Each original ZIP is checked against GitHub's
published SHA-256; each evaluation's episodes.jsonl is checked against its own
evaluation.json SHA-256 before counting any entries. Outputs audit JSON.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import zipfile

SEEDS = (17, 29, 43, 61, 73, 89, 103, 131)
DEPTHS = ("held_depth4", "held_depth5")
MODES = ("direct", "vi1", "vi2", "fixed", "shortest")
STOP = 14

ZIP_SHA = {
    17: "caf8e248d01692cc6823750081552fcd7037f37364295f79fd43b1845c4d2ec8",
    29: "4c1740864f85c9fc30ff5798cd840e2ae998c5c779ab3ddc3fa77e167db480bc",
    43: "dc37af9fee00df29b583fdd49f7dd3e58277141e81155b0443e4b04c92be0c2a",
    61: "ee6078f5ea8c4807b46757513c153f4ea8bfd3dcd1080a34fd9205f5170b43e7",
    73: "91cafb136324f9524282cd2fcb7c567d9eb63fac0a27c8a4100d1fb985edd6a3",
    89: "3ee4db5943e482e90ce005a900fab14351097f81284a694d049a7e29ee130b36",
    103: "377970d392d57f8aa5d218cfcfc8dffa03305ed09c757027621b4a97a9ab1ac6",
    131: "58765ba877c3da7d8689887196f6f05393295a487aac3b45fbfa29e0a4acdb35",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_seed(directory, seed):
    candidates = (
        directory / f"af22-seed-{seed}.zip",
        directory / f"af22-seed-{seed}-43788715688.zip",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Seed {seed}: missing original Actions ZIP")
    payload = path.read_bytes()
    if sha256(payload) != ZIP_SHA[seed]:
        raise ValueError(f"Seed {seed}: artifact ZIP SHA-256 mismatch")
    with zipfile.ZipFile(path) as z:
        scores = json.loads(z.read("evaluation/evaluation.json"))
        episode_bytes = z.read("evaluation/episodes.jsonl")
    if sha256(episode_bytes) != scores["episodes_sha256"]:
        raise ValueError(f"Seed {seed}: episode member SHA-256 mismatch")
    if scores["seed"] != seed or scores["episode_count"] != 2080:
        raise ValueError(f"Seed {seed}: unexpected evaluation identity/count")
    episodes = [json.loads(line) for line in episode_bytes.splitlines()]
    if len(episodes) != 2080:
        raise ValueError(f"Seed {seed}: unexpected episode rows")
    return scores["scores"], episodes


def root_stop(row):
    if not row["actions"]:
        raise ValueError("Expected at least one action")
    return row["actions"][0]["action"] == STOP


def binding_count(row):
    """Count successor-level STOP minima with at least one legal continuation.

    This measures whether STOP is the interior minimum in a saved *initial*
    VI-2 plan; it does not reroll the policy or score counterfactual outcomes.
    """
    step = row["actions"][0]
    trace = step["plan_trace"]
    nodes = {tuple(n["state"]): n for n in trace["nodes"]}
    root = nodes[tuple(step["before"])]
    bound = compared = 0
    for successor_state in root["edges"].values():
        successor = nodes[tuple(successor_state)]
        continuing = [
            1 + nodes[tuple(target)]["value0"]
            for target in successor["edges"].values()
        ]
        if continuing:
            compared += 1
            if successor["stop_cost"] <= min(continuing):
                bound += 1
    return bound, compared, len(nodes)


def replay(directory):
    scores_by_seed, records = {}, []
    for seed in SEEDS:
        scores, episodes = read_seed(directory, seed)
        scores_by_seed[seed] = scores
        records.extend((seed, row) for row in episodes)

    output = {
        "source_run": 43788715688,
        "seeds": list(SEEDS),
        "hash_verified_original_seed_zips": len(SEEDS),
        "total_episodes": len(records),
        "depths": {},
    }
    for depth in DEPTHS:
        stats = {
            "scores_solved": {
                mode: sum(scores_by_seed[s][depth][mode]["solved"] for s in SEEDS)
                for mode in MODES
            },
            "first_stop": {},
        }
        for mode in MODES:
            roots = [
                row for _, row in records
                if row["mode"] == mode and row["split"] == depth
            ]
            positives = [r for r in roots if r["goal_truth"]]
            negatives = [r for r in roots if not r["goal_truth"]]
            assert len(positives) == 640 and len(negatives) == 896
            stats["first_stop"][mode] = {
                "provable": sum(root_stop(r) for r in positives),
                "unprovable": sum(root_stop(r) for r in negatives),
            }

        # Structural counts are seed-independent: same cases, same fixed policy.
        fixed = [
            row for seed, row in records
            if seed == 17 and row["split"] == depth
            and row["mode"] == "fixed" and row["goal_truth"]
        ]
        stats["enabled_root_count"] = dict(
            sorted(collections.Counter(
                str(len(r["actions"][0]["enabled"])) for r in fixed
            ).items())
        )
        vi2_pos = [
            row for _, row in records
            if row["split"] == depth and row["mode"] == "vi2"
            and row["goal_truth"]
        ]
        branches = [binding_count(row) for row in vi2_pos]
        stats["interior_stop_binding"] = {
            "bound": sum(x[0] for x in branches),
            "continuations": sum(x[1] for x in branches),
            "nodes_examined_initial": sum(x[2] for x in branches),
        }
        output["depths"][depth] = stats

    scores = output["depths"]
    assert [scores[d]["scores_solved"]["fixed"] for d in DEPTHS] == [480, 512]
    assert [scores[d]["scores_solved"]["vi2"] for d in DEPTHS] == [307, 147]
    assert [scores[d]["first_stop"]["vi2"]["provable"] for d in DEPTHS] == [308, 488]
    assert [scores[d]["interior_stop_binding"]["bound"] for d in DEPTHS] == [336, 614]
    assert [scores[d]["interior_stop_binding"]["continuations"] for d in DEPTHS] == [944, 944]
    assert len(records) == 16640
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip_dir", type=Path, help="directory with eight original Actions seed ZIPs")
    args = parser.parse_args()
    print(json.dumps(replay(args.zip_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
