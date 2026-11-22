"""Verify the routed and unrouted controls on the pending-rule development cohort."""

import argparse
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_branch_recovery import audit
from configs.freeze_ppt_branch_recovery import ARMS
from configs.run_ppt_branch_recovery import UNROUTED
from ppt_gym.data import canonical


def paired_comparisons(evidence, minimum_effect):
    from spiral_ln.evaluation_finite_sample import evaluate_manifest

    candidate = ARMS[0]
    controls = (*ARMS[1:], UNROUTED)
    stages = {arm: trial["trial"]["stages"]["final"] for arm, trial in evidence["trials"].items()}
    groups = {cluster: [row["root"] for row in worlds]
              for cluster, worlds in evidence["witness"]["pairs"].items()}
    seed = evidence["summary"]["seed"]
    outcomes = {
        arm: {cluster: int(all(stage[root]["certified"] for root in roots))
              for cluster, roots in groups.items()}
        for arm, stage in stages.items()
    }
    comparisons = {}
    for control in controls:
        rows = [{"case_id": f"{seed}:{cluster}", "cluster_id": cluster,
                 "candidate": outcomes[candidate][cluster], "baseline": outcomes[control][cluster]}
                for cluster in groups]
        result = evaluate_manifest({
            "schema_version": 1, "experiment_id": f"ppt-pending-alias-{seed}-{control}",
            "target_population": "development pairs with balanced pending-rule support",
            "evaluation_mode": "sampled", "alpha": "0.05", "planned_comparisons": len(controls),
            "minimum_effect": str(minimum_effect),
            "hypothesis_preregistered": True, "sealed_readout": False, "clusters_independent": False,
            "exchangeable_pair_labels": False, "cases": rows,
        })
        differences = [row["candidate"] - row["baseline"] for row in rows]
        result.update(layout_wins=differences.count(1), layout_losses=differences.count(-1),
                      layout_ties=differences.count(0))
        comparisons[control] = result
    return comparisons, outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "initialization", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed-study", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    registration = json.loads(args.cohort.read_bytes())["registration"]
    if registration.get("pending_arrival_ablation") != UNROUTED:
        parser.error("expected the registered pending-arrival comparison")
    args.probe = None
    summary, evidence = audit(args)
    summary["comparisons"], evidence["certified_by_layout"] = paired_comparisons(
        evidence, registration["minimum_meaningful_effect"]
    )
    with args.summary.open("x", encoding="utf-8", newline="\n") as out:
        out.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    with args.evidence.open("xb") as out:
        out.write(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for arm, values in summary["arms"].items():
        print(arm, values["final"]["solved_both_worlds"], values["final"]["proofs"])


if __name__ == "__main__":
    main()
