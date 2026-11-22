"""Audit additional training seeds on the original paired development layouts."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_transport_pilot import audit_proposals
from configs.check_ppt_conjunction_pilot import ProofLengthWitness
from configs.run_ppt_conjunction_pilot import ARMS
from ppt_gym import ram_store
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_campaign import configure


def first_choices(path, witness):
    with ram_store.open_store(path) as db:
        rows = {}
        for row in ram_store.ledger(db):
            if row["phase"] != "final":
                continue
            ep, = row["episodes"]
            event = ep["events"][0]
            targets, _ = witness(canonical(event["input"]), event["candidate_menu"], ep["depth"])
            if len(targets) != 2 or int((targets == -8).sum()) != 1:
                raise ValueError("expected one eight-check first action")
            rows[ep["root"]] = int(targets.argmax()) == event["selected_index"]
    return rows


def audit_seed(seed, folder, cases, clusters, train, cohort_sha):
    resource = json.loads((folder / "campaign-resource-audit.json").read_bytes())
    if resource["cohort"] != cohort_sha or resource["statuses"] != {arm: 0 for arm in ARMS}:
        raise ValueError("incomplete matched seed-replication campaign")
    trials, exposure, schedule, protocol = {}, set(), set(), None
    witness = ProofLengthWitness()
    for arm, theory in ARMS.items():
        path = folder / f"{seed}-{arm}.sqlite"
        trial = read_arm(path, cases, cohort_sha)
        model = trial["manifest"]
        tested = tuple(model[key] for key in ("architecture", "objective", "head_only",
                                              "residual_fact_transport", "premise_interaction"))
        if (tested != theory or model["seed"] != seed or not model["graph_context"] or
                model["alias_training"] != "contrast-search" or set(trial["stages"]) != {"initial", "final"}):
            raise ValueError("unregistered policy in the seed replication")
        arm_protocol = tuple(model[key] for key in PROTOCOL_FIELDS if key != "objective")
        protocol = arm_protocol if protocol is None else protocol
        if protocol != arm_protocol:
            raise ValueError("replication arms used different budgets or source code")
        exposure.add(trial["training_exposure_sha256"])
        proposal = audit_proposals(path, train, model["rollouts"], require_both=True)
        schedule.add(proposal["schedule_sha256"])
        roots = first_choices(path, witness)
        final = trial["stages"]["final"]
        if set(roots) != set(cases) or any(final[key]["certified"] and not correct for key, correct in roots.items()):
            raise ValueError("native proof contradicts first-choice witness")
        trial["feasible_first"] = roots
        trial["proposal"] = proposal
        trials[arm] = trial
    if len(exposure) != 1 or len(schedule) != 1:
        raise ValueError("arms saw different training roots or root proposals")
    summary = {}
    for arm, trial in trials.items():
        summary[arm] = {phase: pair_counts(stage, clusters) for phase, stage in trial["stages"].items()}
        roots = trial["feasible_first"]
        summary[arm]["final"].update(
            feasible_first=sum(roots.values()),
            failed_after_feasible_first=sum(correct and not trial["stages"]["final"][root]["certified"]
                                           for root, correct in roots.items()),
        )
    return summary, {"resource": resource, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "registration", "source_workdir", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    study = json.loads(args.registration.read_bytes())
    cohort_sha = sha256_file(args.cohort)
    if study["pilot_cohort_sha256"] != cohort_sha or study["arms"] != list(ARMS):
        raise ValueError("wrong source cohort or control family")
    data = load(args.cohort)
    cases = {case.root: case for case in data["dev"]}
    train = {case.root: case for case in data["train"]}
    clusters = defaultdict(list)
    for root, case in cases.items():
        clusters[case.cluster].append(root)
    if len(clusters) != 24 or set(map(len, clusters.values())) != {2}:
        raise ValueError("wrong paired development layouts")
    configure(2, 0)
    sources = {103: args.source_workdir / "hybrid-conjunction-pilot"}
    sources.update({seed: args.source_workdir / f"hybrid-conjunction-seed-{seed}" for seed in study["seeds"]})
    summary = {"scope": study["scope"], "registration_sha256": sha256_file(args.registration),
               "cohort_sha256": cohort_sha, "layout_pairs": len(clusters), "seeds": {}}
    evidence = {}
    for seed, sessions in sources.items():
        summary["seeds"][seed], evidence[seed] = audit_seed(seed, sessions, cases, clusters, train, cohort_sha)
        for arm, result in summary["seeds"][seed].items():
            print(seed, arm, result["final"])
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))


if __name__ == "__main__":
    main()
