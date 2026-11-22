"""Combine all registered pending-rule seeds while clustering by the shared layout."""

import argparse
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_pending_alias import compare_arms
from configs.freeze_ppt_branch_recovery import ARMS
from configs.run_ppt_branch_recovery import UNROUTED
from ppt_gym.data import sha256_file


def combine(cohort, seed_study, evidence_paths):
    cohort_sha, seed_study_sha = sha256_file(cohort), sha256_file(seed_study)
    reg = json.loads(cohort.read_bytes())["registration"]
    study = json.loads(seed_study.read_bytes())
    if study["cohort_sha256"] != cohort_sha or reg.get("pending_arrival_ablation") != UNROUTED:
        raise ValueError("the cohort and registered seeds identify different experiments")
    arms = set((*ARMS, UNROUTED))
    outcomes, summaries, hashes = {}, {}, {}
    reference = None
    for path in evidence_paths:
        data = json.loads(gzip.decompress(path.read_bytes()))
        summary, values = data["summary"], data["certified_by_layout"]
        seed = int(summary["seed"])
        roots = {key: sorted(world["root"] for world in pair) for key, pair in data["witness"]["pairs"].items()}
        reference = roots if reference is None else reference
        if (summary["cohort_sha256"] != cohort_sha or summary["seed_study_sha256"] != seed_study_sha
                or seed in summaries or roots != reference or set(values) != arms
                or any(set(rows) != set(roots) for rows in values.values())):
            raise ValueError("unmatched seed, layout, policy, or registration")
        for arm, rows in values.items():
            if any(score not in (0, 1) for score in rows.values()):
                raise ValueError("paired certificate outcomes must be binary")
            if sum(rows.values()) != summary["arms"][arm]["final"]["solved_both_worlds"]:
                raise ValueError("the root-paired outcomes differ from the native audit")
        summaries[seed], outcomes[seed], hashes[str(seed)] = summary, values, sha256_file(path)
    if set(summaries) != set(study["seeds"]):
        raise ValueError("the combined report requires every registered seed")
    return {
        "scope": reg["scope"], "cohort_sha256": cohort_sha, "seed_study_sha256": seed_study_sha,
        "evidence_sha256": hashes, "layouts": len(reference), "seeds": sorted(summaries),
        "native_runs": summaries,
        "comparisons": compare_arms(outcomes, reg["minimum_meaningful_effect"]),
        "confirmation_scored": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "seed_study", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("evidence", type=Path, nargs="+")
    args = parser.parse_args()
    result = combine(args.cohort, args.seed_study, args.evidence)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
