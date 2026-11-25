"""Replay COG-03 receipts using the independent finite Horn interpreter.

This command needs only Python's standard library. It checks every expected
root/arm/seed episode, every charged action and the complete observation prefix.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from .environment import check_step, goal_ancestors, load_tasks, public_observation, sha256


def read_rows(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row):
    return (row["split"], row["root_id"], row["arm"], row["seed"])


def audit(corpus, directory):
    directory = Path(directory)
    artifacts = json.loads((directory / "artifacts.json").read_text())
    for name, expected in artifacts["files"].items():
        if Path(name).name != name or sha256(directory / name) != expected:
            raise ValueError(f"artifact digest mismatch: {name}")
    registration = json.loads((directory / "registration.json").read_text())
    if sha256(corpus) != registration["corpus_sha256"]:
        raise ValueError("corpus bytes changed")
    tasks = load_tasks(corpus)
    task_by_root = {t["root_id"]: t for t in tasks}
    for split, roots in registration["roots"].items():
        if roots != [t["root_id"] for t in tasks if t["split"] == split]:
            raise ValueError("registered split differs from the corpus")
    expected_keys = {(split, root, arm, seed)
                     for split, roots in registration["roots"].items() if split != "train"
                     for root in roots for arm in registration["arms"] for seed in registration["seeds"]}
    episodes = read_rows(directory / "episodes.jsonl")
    if len(episodes) != len(expected_keys) or {key(r) for r in episodes} != expected_keys:
        raise ValueError("missing, duplicate or unexpected root episode")
    attempts = read_rows(directory / "attempts.jsonl.gz")
    grouped = defaultdict(list)
    for row in attempts:
        if key(row) not in expected_keys:
            raise ValueError("attempt belongs to an unexpected episode")
        grouped[key(row)].append(row)
    recomputed = []
    for episode in episodes:
        task = task_by_root[episode["root_id"]]
        world = task["world"]
        known = set(world["facts"])
        needed = goal_ancestors(world)
        history, accepted, rejected, duplicates = [], 0, 0, 0
        calls, inner = 0, 0
        for index, row in enumerate(grouped[key(episode)]):
            if world["goal"] in known:
                raise ValueError("attempt follows a completed goal")
            if row["step"] != index or row["budget"] != registration["budget"] or index >= row["budget"]:
                raise ValueError("step index or budget changed")
            if row["observation"] != public_observation(task, known, history):
                raise ValueError("observation differs from replayed public prefix")
            receipt = check_step(world, known, row["action"])
            if row["receipt"] != receipt or row["verifier_calls"] != 1:
                raise ValueError("checker response or charged cost differs")
            new = receipt["new_fact"]
            if row["new_unique_fact"] != (new is not None):
                raise ValueError("false novel-fact credit")
            if row["goal_relevant_fact"] != (new is not None and new in needed):
                raise ValueError("false goal-relevance credit")
            if row["solved"] != receipt["solved"]:
                raise ValueError("false completed-root credit")
            accepted += receipt["accepted"]
            rejected += not receipt["accepted"]
            duplicates += receipt["code"] == "duplicate_lemma"
            calls += 1
            inner += row["policy_compute"]["inner_passes"]
            history.append({"action": row["action"], **{k: receipt[k] for k in ("accepted", "code", "new_fact")}})
            known = set(receipt["next_known"])
        expected = {"verifier_calls": calls, "solved": world["goal"] in known,
                    "accepted": accepted, "rejected": rejected, "duplicates": duplicates,
                    "unique_verified_facts": len(known - set(world["facts"])),
                    "goal_relevant_verified_facts": len((known - set(world["facts"])) & needed),
                    "final_known": sorted(known), "inner_cell_calls": inner,
                    "task_digest": task["task_digest"]}
        if episode["budget"] != registration["budget"] or any(episode[k] != v for k, v in expected.items()):
            raise ValueError("episode summary differs from replay")
        recomputed.append({**{k: episode[k] for k in ("split", "root_id", "arm", "seed")}, **expected})
    if len(attempts) != artifacts["attempt_rows"] or len(episodes) != artifacts["episode_rows"]:
        raise ValueError("artifact row count differs")
    grouped_metrics = defaultdict(list)
    for episode in recomputed:
        grouped_metrics[(episode["split"], episode["arm"], episode["seed"])].append(episode)
    for row in json.loads((directory / "summary.json").read_text()):
        roots = grouped_metrics.pop((row["split"], row["arm"], row["seed"]))
        calls = sum(r["verifier_calls"] for r in roots)
        completed = sum(r["solved"] for r in roots)
        if (row["roots"] != len(roots) or row["solved"] != completed or row["verifier_calls"] != calls
                or abs(row["accuracy"] - completed / len(roots)) > 1e-12
                or abs(row["goal_relevant_facts_per_call"] - sum(r["goal_relevant_verified_facts"] for r in roots) / calls) > 1e-12):
            raise ValueError("aggregate summary differs from replay")
    if grouped_metrics:
        raise ValueError("summary omitted groups")
    return {"schema_version": "cog03.replay.v1", "verdict": "passed",
            "roots": len({r["root_id"] for r in episodes}), "episodes": len(episodes),
            "attempts": len(attempts), "solved_episodes": sum(r["solved"] for r in episodes),
            "all_expected_episodes_present": True, "public_prefixes_replayed": True,
            "evidence_class": "finite Horn trace replay", "corpus_sha256": sha256(corpus)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.corpus, args.results), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
