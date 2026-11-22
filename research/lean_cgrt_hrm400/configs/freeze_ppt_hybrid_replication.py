"""Freeze a fresh and balanced Horn replication cohort before scoring it."""
import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import SCHEMA, _horn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_cohort", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("excluded", type=Path, nargs="+")
    args = parser.parse_args()
    old = json.loads(args.training_cohort.read_text(encoding="utf-8"))
    if old["schema"] != SCHEMA:
        raise ValueError("unknown original development cohort")
    excluded = {row["lineage"] for row in old["cases"]}
    for path in args.excluded:
        source = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(row["lineage"] for group in source["cohorts"].values() for row in group)
    cases, counts = [], Counter()
    for case in _horn(24029, 0, 192):
        if case.lineage not in excluded and counts[case.family] < 24:
            cases.append(asdict(case))
            excluded.add(case.lineage)
            counts[case.family] += 1
    if counts != {"alt-cycle": 24, "tree-deep": 24}:
        raise ValueError("unable to freeze a balanced disjoint replication cohort")
    result = {**old, "seed": 24029,
              "cases": [row for row in old["cases"] if row["split"] == "train"] + cases,
              "registration": {"purpose": "fresh development replication; no confirmation claim",
                               "primary_population": "alt-cycle",
                               "secondary_population": "tree-deep",
                               "primary": "hybrid actor-critic versus full and frozen-backbone TRM",
                               "min_meaningful_improvement": 0.05,
                               "root_counts": dict(counts), "tactic_checks": 8,
                               "training_cohort_sha256": sha256_file(args.training_cohort),
                               "excluded_cohort_sha256": sorted(sha256_file(path) for path in args.excluded),
                               "model_seeds": [29, 43, 71], "test_families_opened": False}}
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(canonical(result) + "\n")
    print(dict(counts))


if __name__ == "__main__":
    main()
