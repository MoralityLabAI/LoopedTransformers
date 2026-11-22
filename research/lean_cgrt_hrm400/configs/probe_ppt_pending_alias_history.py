"""Apply history-only recovery to audited pending-rule and proof-progress checkpoints."""

import argparse
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_branch_recovery import verify_witness
from configs.freeze_ppt_branch_recovery import ARMS
from configs.probe_ppt_history_only import evaluate_policies
from configs.run_ppt_branch_recovery import PROGRESS_ARMS, UNROUTED, registered_progress
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_limits import TreeBudget


def certificate_changes(audited, result, seed, arms):
    changes = {}
    for arm in arms:
        pristine = audited["certified_by_layout"][arm]
        roots = result["roots"][str(seed)][arm]
        history = {
            cluster: int(all(roots[world["root"]]["certified"] for world in worlds))
            for cluster, worlds in audited["witness"]["pairs"].items()
        }
        differences = [history[cluster] - pristine[cluster] for cluster in history]
        changes[arm] = {
            "pristine_certified_pairs": sum(pristine.values()),
            "history_only_certified_pairs": sum(history.values()),
            "history_only_gains": differences.count(1),
            "history_only_losses": differences.count(-1),
            "unchanged": differences.count(0),
        }
    return changes


def evaluate(args, tree):
    cohort_sha = sha256_file(args.cohort)
    registration = json.loads(args.cohort.read_bytes())["registration"]
    audited = json.loads(gzip.decompress(args.audit.read_bytes()))
    summary = audited["summary"]
    seed = summary["seed"]
    progress = registered_progress(args.cohort, args.progress_study, seed)
    study_sha = sha256_file(args.progress_study) if progress else None
    arms = PROGRESS_ARMS if progress else (*ARMS, UNROUTED)
    if (summary["cohort_sha256"] != cohort_sha
            or summary.get("progress_study_sha256") != study_sha
            or registration.get("pending_arrival_ablation") != UNROUTED
            or set(audited["trials"]) != set(arms)):
        raise ValueError("history-only recovery requires the complete audit of the registered comparison")
    cases = {case.root: case for case in load(args.cohort)["dev"]}
    witness_file = args.sessions / "native-witness.json"
    reference, _, winners = verify_witness(witness_file, cases, cohort_sha, registration)
    if reference != audited["witness"]:
        raise ValueError("the proof witness differs from the audited run")
    witness = {}
    pending_goals = 3 if progress else 2
    for cluster, pair in reference["pairs"].items():
        for row in pair:
            root = row["root"]
            public = row["outcomes"][0]["observation"]
            if (public != row["outcomes"][1]["observation"]
                    or len(public["current"]["state"]["goals"]) != pending_goals):
                raise ValueError("pending-rule witnesses disagree on the recovery state")
            witness[root] = (cluster, public, winners[root])
    label = str(seed)
    hybrid = [arm for arm in arms if arm.startswith("hybrid_")]
    expected = {
        "previous_cohort_sha256": cohort_sha,
        "source_session_sha256": {label: {arm: audited["trials"][arm]["trial"]["session_sha256"] for arm in hybrid}},
    }
    result = evaluate_policies(tree, expected, cases, witness, [(seed, args.sessions)], hybrid,
                               native=args.native, reuse_identical=args.native)
    if any(result["weights_sha256"][label][arm] != audited["trials"][arm]["trial"]["final_weights_sha256"]
           for arm in hybrid):
        raise ValueError("history-only control did not use the audited final weights")
    report = {
        "scope": "pending-rule development control with direct parent/current score zeroed",
        "cohort_sha256": cohort_sha, "audit_sha256": sha256_file(args.audit),
        "witness_sha256": sha256_file(witness_file), "seed": seed, **result,
    }
    if progress:
        report["scope"] = "three-obligation development control with direct parent/current score zeroed"
        report["progress_study_sha256"] = study_sha
        if args.native:
            report["paired_certificate_changes"] = certificate_changes(audited, result, seed, hybrid)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "audit", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--progress-study", type=Path)
    args = parser.parse_args()
    tree = TreeBudget(2, 2)
    try:
        report = evaluate(args, tree)
        with args.output.open("xb") as handle:
            handle.write(gzip.compress(canonical(report).encode("utf-8"), mtime=0))
    finally:
        tree.close()


if __name__ == "__main__":
    main()
