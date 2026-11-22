"""Audit native rewrite model comparisons and report paired development outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path

from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_root_adaptation import source_exposure
from configs.freeze_ppt_bpd_rewrite import ARMS, registered_exploration
from configs.probe_ppt_bpd_rewrite_revisits import revisit_count
from ppt_gym import ram_store
from ppt_gym.data import digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.rewrite import BPD_REWRITE_FAMILY


def rewrite_training(db, expected_rollouts, *, exploration=False):
    counts = Counter(rollouts=0, certificates=0, state_revisits=0, cpu_censored=0,
                     certificate_free_parameter_updates=0)
    milestones = []
    for row in ram_store.ledger(db):
        if row["phase"] != "train":
            continue
        episodes = row["episodes"]
        rewrites = [episode for episode in episodes if episode["family"] == BPD_REWRITE_FAMILY]
        if not rewrites:
            continue
        all_within_budget = all(episode["within_cpu_budget"] for episode in episodes)
        any_certificate = any(episode["verified_closed"] for episode in episodes)
        if all_within_budget and not any_certificate:
            counts["certificate_free_parameter_updates"] += row["weights_before"] != row["weights_after"]
        for episode in rewrites:
            if (bool(episode.get("rewrite_exploration")) != exploration
                    or (exploration and not episode["native_replay"]["replay_passed"])):
                raise ValueError("training rewrite exploration is missing authoritative replay")
            counts.update(rollouts=1, certificates=int(episode["verified_closed"] and episode["within_cpu_budget"]),
                          state_revisits=revisit_count(episode), cpu_censored=int(not episode["within_cpu_budget"]))
            if exploration:
                counts["masked_decisions"] += sum("rewrite_choices" in event for event in episode["events"])
            if counts["rollouts"] % 32 == 0:
                milestones.append({key: counts[key] for key in ("rollouts", "certificates", "state_revisits")})
    if counts["rollouts"] != expected_rollouts:
        raise ValueError("rewrite training did not cover the registered rollouts")
    return {**counts, "milestones": milestones}


def audit(cohort, reference_path, sessions, initialization, exploration_study=None, *, seed=151, replication_study=None):
    cohort_sha = sha256_file(cohort)
    reference_sha = sha256_file(reference_path)
    reference = json.loads(reference_path.read_bytes())
    resources = json.loads((sessions / "campaign-resource-audit.json").read_bytes())
    exploration = registered_exploration(cohort, exploration_study, seed=seed, replication=replication_study)
    study_sha = sha256_file(exploration_study) if exploration else None
    replication_sha = sha256_file(replication_study) if replication_study else None
    if (resources["cohort"] != cohort_sha or resources["reference_sha256"] != reference_sha
            or resources.get("exploration_study_sha256") != study_sha
            or resources.get("replication_study_sha256") != replication_sha
            or resources.get("model_seed", 151) != seed
            or resources["statuses"] != dict.fromkeys(ARMS, 0) or reference["cohort_sha256"] != cohort_sha):
        raise ValueError("the resource receipt does not cover the registered rewrite run")
    data = load(cohort)
    cases = {case.root: case for case in data["dev"]}
    training = {case.root: case for case in data["train"]}
    groups = {root: case.cluster for root, case in cases.items()}
    positive = {root for root, row in reference["rows"].items() if row["certified"]}
    if set(reference["rows"]) != set(cases) or len(positive) != 8:
        raise ValueError("the native reference does not cover all development declarations")
    for root in positive:
        row = reference["rows"][root]
        certificate = row["certificate"]
        if (not certificate["passed"] or certificate["axioms"] or row["checks"] > 8
                or not row["replay"]["replay_passed"] or digest(certificate["source"]) != certificate["source_digest"]):
            raise ValueError("a reference proof is missing its native certificate")

    trials, exposure_hashes, protocol = {}, set(), None
    for arm in ARMS:
        architecture, objective = arm.split("_", 1)
        path = sessions / f"{arm}.sqlite"
        trial = read_arm(path, cases, cohort_sha)
        manifest = trial["manifest"]
        source_name = "hybrid_imitation" if architecture == "hybrid" else "trm"
        source = initialization / f"{seed}-{source_name}.sqlite"
        if (manifest["architecture"] != architecture or manifest["seed"] != seed
                or manifest["head_only"] != (objective == "calibration")
                or manifest["objective"] != ("imitation" if objective == "imitation" else "rl")
                or not manifest["graph_context"] or manifest["premise_interaction"] != (architecture == "hybrid")
                or manifest.get("rewrite_exploration") != exploration
                or manifest["initialization"]["session_sha256"] != sha256_file(source)):
            raise ValueError("an arm used a different architecture, objective, or initialization")
        common = tuple(manifest[key] for key in PROTOCOL_FIELDS if key != "objective")
        if protocol is not None and common != protocol:
            raise ValueError("rewrite arms used different protocol settings or code")
        protocol = common
        exposure_hashes.add(trial["training_exposure_sha256"])
        with ram_store.open_store(path) as db:
            exposures = source_exposure(db, training)
            expected_rollouts = sum(c.family == BPD_REWRITE_FAMILY for c in training.values()) * manifest["rollouts"]
            trial["rewrite_training"] = rewrite_training(db, expected_rollouts, exploration=bool(exploration))
        if (not exposures.get("mathcan_ram08_audited_train")
                or any(source not in exposures for source in {case.source for case in training.values()})):
            raise ValueError("a registered training source is missing")
        for stage in trial["stages"].values():
            if any(row["certified"] for root, row in stage.items() if root not in positive):
                raise ValueError("a policy closure contradicted the native BPD reference")
        trial["source_exposures"] = exposures
        trials[arm] = trial
    if len(exposure_hashes) != 1:
        raise ValueError("the rewrite arms consumed different training cases")

    counts = {arm: {phase: sum(row["certified"] for row in stage.values())
                    for phase, stage in trial["stages"].items()} for arm, trial in trials.items()}
    primary = trials["hybrid_rl"]["stages"]["final"]
    paired = {}
    for control in ARMS[1:]:
        final = trials[control]["stages"]["final"]
        by_root = {root: primary[root]["certified"] - final[root]["certified"] for root in positive}
        paired[control] = {
            "wins": sum(value == 1 for value in by_root.values()),
            "losses": sum(value == -1 for value in by_root.values()),
            "ties": sum(value == 0 for value in by_root.values()),
            "certificate_delta_by_group": {
                cluster: sum(value for root, value in by_root.items() if groups[root] == cluster)
                for cluster in set(groups.values())
            },
        }
    return {"scope": "original BPD rewrite development", "cohort_sha256": cohort_sha, "seed": seed,
            "reference_sha256": reference_sha, "native_reference_certificates": len(positive),
            "exploration_study_sha256": study_sha, "replication_study_sha256": replication_sha,
            "development_declarations": len(cases), "structural_groups": len(set(groups.values())),
            "certificates": counts, "paired_positive_outcomes": paired,
            "trials": trials, "memory": resources["memory"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "reference", "sessions", "initialization", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--exploration-study", type=Path)
    parser.add_argument("--replication-study", type=Path)
    parser.add_argument("--seed", type=int, default=151)
    args = parser.parse_args()
    report = audit(args.cohort, args.reference, args.sessions, args.initialization, args.exploration_study,
                   seed=args.seed, replication_study=args.replication_study)
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(report, sort_keys=True, indent=2) + "\n")
    for arm, counts in report["certificates"].items():
        print(arm, counts, flush=True)


if __name__ == "__main__":
    main()
