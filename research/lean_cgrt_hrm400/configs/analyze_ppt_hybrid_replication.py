"""Audit the registered multi-seed hybrid replication at the theorem-root level."""
import argparse
from collections import Counter
import json
from pathlib import Path

from configs.analyze_ppt_hybrid_rl import compare, read_trial
from ppt_gym import ram_store
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_learning import tensor_hash

PROTOCOL_FIELDS = (
    "batch", "cap", "cohort", "cpu_seconds", "entry_count", "lr", "mathcan_archive_sha256",
    "max_macs", "objective", "rollouts", "source_sha256", "threads", "torch", "train_count",
    "verifier", "width",
)
ARMS = {"hybrid": ("hybrid", False), "trm": ("trm", False), "head-only": ("trm", True)}


def read_arm(path, cases, cohort_sha):
    trial = read_trial(path, cases, cohort_sha)
    exposure = []
    expected = None
    with ram_store.open_store(path) as db:
        for row in ram_store.ledger(db):
            if expected is not None and row["weights_before"] != expected:
                raise ValueError("checkpoint weights disagree with preceding episode")
            expected = row["weights_after"]
            if row["phase"] in ("train", "mathcan"):
                exposure.append([row["phase"], [ep["root"] for ep in row["episodes"]]])
            for ep in row["episodes"]:
                cert = ep.get("certificate")
                if ep["verified_closed"] and (cert is None or digest(cert["source"]) != cert["source_digest"]):
                    raise ValueError("missing or changed standalone certificate source")
        if tensor_hash(ram_store.checkpoint(db)["model"]) != expected:
            raise ValueError("stored final weights disagree with the ledger")
    trial["training_exposure_sha256"] = digest(exposure)
    trial["final_weights_sha256"] = expected
    trial["session_sha256"] = sha256_file(path)
    return trial


def paired_rows(trials, control, cases, phase="final"):
    return [
        {"case_id": f"{seed}:{root}", "cluster_id": case.cluster, "root": root,
         "seed": seed, "family": case.family,
         "candidate": arms["hybrid"]["stages"][phase][root]["certified"],
         "baseline": arms[control]["stages"][phase][root]["certified"]}
        for seed, arms in trials.items()
        for root, case in cases.items()
    ]


def fsi(rows, control, registration):
    from spiral_ln.evaluation_finite_sample import evaluate_manifest

    primary = registration["primary_population"]
    return evaluate_manifest({
        "schema_version": 1, "experiment_id": f"ppt-hybrid-replication-{control}",
        "target_population": f"{primary} development roots, conditional on the registered model seeds",
        "evaluation_mode": "sampled", "alpha": "0.05", "planned_comparisons": 2,
        "minimum_effect": str(registration["min_meaningful_improvement"]),
        "hypothesis_preregistered": True, "sealed_readout": False,
        "clusters_independent": False, "exchangeable_pair_labels": False,
        "cases": [row for row in rows if row["family"] == primary],
    })


def audit(cohort, directory):
    registration = json.loads(cohort.read_bytes())["registration"]
    cases = {case.root: case for case in load(cohort)["dev"]}
    if dict(Counter(case.family for case in cases.values())) != registration["root_counts"]:
        raise ValueError("registered population counts changed")
    cohort_sha = sha256_file(cohort)
    match = None
    trials = {}
    for seed in registration["model_seeds"]:
        arms = {}
        for label, (arch, head) in ARMS.items():
            result = read_arm(directory / f"{seed}-{label}.sqlite", cases, cohort_sha)
            spec = result["manifest"]
            if (spec["architecture"], spec["head_only"], spec["seed"]) != (arch, head, seed):
                raise ValueError(f"unexpected architecture or seed for {seed}-{label}")
            common = (*[spec[key] for key in PROTOCOL_FIELDS], spec.get("graph_context", False))
            if match is None:
                match = common
            if common != match:
                raise ValueError("replication arms used different budgets, code or source inputs")
            arms[label] = result
        if len({arm["training_exposure_sha256"] for arm in arms.values()}) != 1:
            raise ValueError("paired arms saw different training root schedules")
        for root, row in arms["trm"]["stages"]["initial"].items():
            baseline = arms["head-only"]["stages"]["initial"][root]
            if any(row[name] != baseline[name] for name in ("certified", "macs", "tactic_checks", "tactics")):
                raise ValueError("initial TRM trajectories differ across the two controls")
        trials[seed] = arms
    comparisons = {}
    for control in ("trm", "head-only"):
        rows = paired_rows(trials, control, cases)
        comparisons[control] = {"initial": paired_rows(trials, control, cases, "initial"),
                                "final": rows, "primary_fsi": fsi(rows, control, registration)}
    per_seed = {
        seed: {
            "before_after": {label: compare(arm["stages"]["final"], arm["stages"]["initial"], cases)
                             for label, arm in arms.items()},
            "final": {label: compare(arms["hybrid"]["stages"]["final"], arm["stages"]["final"], cases)
                      for label, arm in arms.items() if label != "hybrid"},
        }
        for seed, arms in trials.items()
    }
    return {"scope": "development replication", "cohort_sha256": cohort_sha,
            "registration": registration, "comparisons": comparisons, "per_seed": per_seed,
            "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("sessions", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit(args.cohort, args.sessions)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for control, item in report["comparisons"].items():
        score = item["primary_fsi"]
        print(control, score["candidate_correct"], score["baseline_correct"], score["n_cases"],
              score["n_clusters"], score["verdict"])


if __name__ == "__main__":
    main()
