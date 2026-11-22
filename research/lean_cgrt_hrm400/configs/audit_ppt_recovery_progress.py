"""Check if committed Lean progress distinguishes otherwise tied full-proof failures."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_branch_recovery import decision
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY
from ppt_gym.ram_limits import TreeBudget
from ppt_gym.recovery_progress import tied_failure_progress


def viable_recovery(observation):
    from ppt_gym.ram_campaign import horn_choice

    selected = horn_choice(canonical(observation["parent"]), observation["parent_menu"])
    return int(selected != observation["explored"])


def audit_seed(session, audit_path):
    from ppt_gym import ram_store

    audited = json.loads(gzip.decompress(audit_path.read_bytes()))
    trial = audited["trials"]["hybrid_recovery_rl"]["trial"]
    if sha256_file(session) != trial["session_sha256"]:
        raise ValueError("progress audit requires the completed routed trajectory ledger")
    witness_worlds = 0
    for pair in audited["witness"]["pairs"].values():
        for row in pair:
            if viable_recovery(row["outcomes"][0]["observation"]) != row["winner"]:
                raise ValueError("the structural branch reference disagrees with the native witness")
            witness_worlds += 1

    counts, rows = {}, []
    with ram_store.open_store(session) as db:
        for row in ram_store.ledger(db):
            episodes = row["episodes"]
            if row["phase"] != "train" or episodes[0]["family"] != RECOVERY_FAMILY:
                continue
            progress = tied_failure_progress(episodes)
            if progress is None:
                continue
            events = [decision(ep) for ep in episodes]
            correct = [event["selected_index"] == viable_recovery(event["input"]) for event in events]
            if sum(correct) != 1:
                raise ValueError("the paired proposal lacks exactly one structurally viable world")
            viable = correct.index(True)
            good, other = progress[viable], progress[1 - viable]
            relation = "aligned" if good > other else "misaligned" if good < other else "tied"
            goals = len(events[0]["input"]["current"]["state"]["goals"])
            family = "pending_goals" if goals > 1 else "single_goal"
            counts.setdefault(family, Counter()).update([relation])
            rows.append({
                "roots": [ep["root"] for ep in episodes], "checks_each": episodes[0]["tactic_checks"],
                "family": family, "discharges": progress, "viable_index": viable, "relation": relation,
            })

    return {
        "seed": audited["summary"]["seed"], "cohort_sha256": audited["summary"]["cohort_sha256"],
        "audit_sha256": sha256_file(audit_path), "session_sha256": trial["session_sha256"],
        "native_witness_worlds_matching_structural_reference": witness_worlds,
        "counts": {family: {name: values[name] for name in ("aligned", "misaligned", "tied")}
                   for family, values in counts.items()},
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", action="append", nargs=2, type=Path, required=True, metavar=("SESSION", "AUDIT"))
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        runs = [audit_seed(*paths) for paths in args.run]
        seeds = {run["seed"] for run in runs}
        if len(seeds) != len(runs) or len({run["cohort_sha256"] for run in runs}) != 1:
            raise ValueError("progress audit requires distinct seeds of one cohort")
        report = {"scope": "logged training pairs with two failures at equal tactic cost",
                  "measure": "counted goal discharges on the surviving Lean tactic prefix",
                  "reference": "shortest public Horn proof; native development witness cross-checked",
                  "runs": sorted(runs, key=lambda item: item["seed"])}
        with args.output.open("x", encoding="utf-8", newline="\n") as out:
            out.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for run in report["runs"]:
            print(run["seed"], run["counts"])
    finally:
        tree.close()


if __name__ == "__main__":
    main()
