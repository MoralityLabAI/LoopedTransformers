"""Score the shared-observation graph pilot with paired-world units."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from configs.analyze_ppt_hybrid_replication import ARMS, PROTOCOL_FIELDS, read_arm
from ppt_gym.data import sha256_file
from ppt_gym.hybrid_rl.corpus import load


def pair_counts(stage, clusters):
    counts = [0, 0, 0]
    for roots in clusters.values():
        counts[sum(stage[root]["certified"] for root in roots)] += 1
    return {"solved_neither_world": counts[0], "solved_one_world": counts[1],
            "solved_both_worlds": counts[2], "proofs": counts[1] + 2 * counts[2]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    registration = json.loads(args.cohort.read_bytes())["registration"]
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    clusters = defaultdict(list)
    for root, case in cases.items():
        clusters[case.cluster].append(root)
    if len(clusters) != registration["dev_alias_clusters"] or any(len(roots) != 2 for roots in clusters.values()):
        raise ValueError("incomplete paired-world pilot")
    seed, = registration["model_seeds"]
    trials, protocol = {}, None
    cohort_sha = sha256_file(args.cohort)
    for arm, (arch, frozen) in ARMS.items():
        trial = read_arm(args.sessions / f"{seed}-{arm}.sqlite", cases, cohort_sha)
        manifest = trial["manifest"]
        if (manifest["architecture"], manifest["head_only"], manifest["seed"], manifest.get("graph_context")) != (
            arch, frozen, seed, True
        ):
            raise ValueError("trial does not match the registered shared-graph arm")
        identity = tuple(manifest[field] for field in PROTOCOL_FIELDS)
        if protocol is None:
            protocol = identity
        if identity != protocol:
            raise ValueError("graph arms did not use matching run budgets and sources")
        trials[arm] = trial
    if len({trial["training_exposure_sha256"] for trial in trials.values()}) != 1:
        raise ValueError("graph arms did not use matching training roots")

    counts = {arm: {phase: pair_counts(stage, clusters) for phase, stage in trial["stages"].items()}
              for arm, trial in trials.items()}
    pairs = {}
    for arm in ("trm", "head-only"):
        evidence = []
        for group, roots in clusters.items():
            hybrid = [trials["hybrid"]["stages"]["final"][root]["certified"] for root in roots]
            control = [trials[arm]["stages"]["final"][root]["certified"] for root in roots]
            evidence.append({"pair_id": group, "roots": roots, "candidate": hybrid, "baseline": control,
                             "paired_world_delta": sum(hybrid) - sum(control)})
        pairs[arm] = evidence

    report = {"scope": "one-seed development mechanism pilot", "cohort_sha256": cohort_sha,
              "registered_pairs": len(clusters), "registered_worlds": len(cases), "counts": counts,
              "comparisons": pairs, "trials": trials}
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for arm, stages in counts.items():
        print(arm, stages["initial"], stages["final"])


if __name__ == "__main__":
    main()
