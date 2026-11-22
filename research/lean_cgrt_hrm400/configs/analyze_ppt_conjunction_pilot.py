"""Audit matched Lean conjunction choices, frozen controls and verifier rewards."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from configs.analyze_ppt_graph_alias_pilot import fact_switch, pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_transport_pilot import audit_proposals, summarize_fact_switch
from configs.run_ppt_conjunction_pilot import ARMS
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_campaign import configure


def audit_intervention(path, original, cohort_sha):
    report = json.loads(path.read_bytes())
    if (report["session_sha256"] != original["session_sha256"]
            or report["weights_sha256"] != original["final_weights_sha256"] or report["cohort"] != cohort_sha):
        raise ValueError("intervention used a different cohort or checkpoint")
    rows = report["roots"]
    if set(rows) != set(original["stages"]["final"]):
        raise ValueError("incomplete intervention cohort")
    clusters = defaultdict(list)
    for root, row in rows.items():
        clusters[row["cluster"]].append(root)
    counts = {}
    for arm in ("pristine", "ablate-interaction", "control-common-shift"):
        stage, changed = {}, 0
        for root, row in rows.items():
            event = row["arms"][arm]
            pristine = row["arms"]["pristine"]
            if (event["checks"] > 8 or not event["native_replay"]["replay_passed"]
                    or event["bank_reads_edited"] != int(arm != "pristine")):
                raise ValueError("intervention did not replay within the first-decision dose")
            if event["certified"]:
                cert = event["standalone_certificate"]
                if not (cert["passed"] and cert["axioms"] == [] and digest(cert["source"]) == cert["source_digest"]):
                    raise ValueError("intervention success lacks an independent certificate")
            stage[root] = {"certified": int(event["certified"])}
            changed += event["path"] != pristine["path"]
        counts[arm] = {**pair_counts(stage, clusters), "changed_paths": changed}
    return counts, report


def audit(args):
    registry = json.loads(args.cohort.read_bytes())["registration"]
    groups = load(args.cohort)
    cases = {case.root: case for case in groups["dev"]}
    train = {case.root: case for case in groups["train"]}
    clusters = defaultdict(list)
    for root, case in cases.items():
        clusters[case.cluster].append(root)
    if (len(clusters) != registry["dev_alias_clusters"] or any(len(roots) != 2 for roots in clusters.values())
            or registry["controls"] != list(ARMS)[1:]):
        raise ValueError("incomplete registered study")
    seed, = registry["model_seeds"]
    native_batches = min(registry["updates"], (len(train) + 1) // 2)
    cohort_sha = sha256_file(args.cohort)
    trials, proposals, common = {}, {}, None
    for arm, expected in ARMS.items():
        path = args.sessions / f"{seed}-{arm}.sqlite"
        trial = read_arm(path, cases, cohort_sha)
        model = trial["manifest"]
        kind = tuple(model[key] for key in ("architecture", "objective", "head_only",
                                            "residual_fact_transport", "premise_interaction"))
        if (kind != expected or model["seed"] != seed or not model["graph_context"]
                or model["alias_training"] != "contrast-search" or model["lr"] != registry["lr"]
                or model["rollouts"] != registry["rollouts"] or model["train_count"] != min(2 * native_batches, len(train))
                or set(trial["stages"]) != {"initial", "final"}):
            raise ValueError("session differs from its registered arm")
        protocol = tuple(model[key] for key in PROTOCOL_FIELDS if key != "objective")
        common = protocol if common is None else common
        if common != protocol:
            raise ValueError("policy arms did not share an execution budget and training source")
        proposals[arm] = audit_proposals(path, train, model["rollouts"], require_both=True)
        trials[arm] = trial
    if (len({trial["training_exposure_sha256"] for trial in trials.values()}) != 1
            or len({entry["schedule_sha256"] for entry in proposals.values()}) != 1):
        raise ValueError("control arms saw different source exposures or first-action proposals")
    for reference, control in (("hybrid", "hybrid_mean"), ("hybrid", "hybrid_frozen"),
                               ("hybrid", "hybrid_imitation"), ("trm", "trm_frozen")):
        left, right = (trials[name]["stages"]["initial"] for name in (reference, control))
        if any(left[root]["tactics"] != right[root]["tactics"] for root in cases):
            raise ValueError("matched arms used different initial greedy proof trajectories")

    configure(2, seed)
    all_margins = {
        arm: fact_switch(args.sessions / f"{seed}-{arm}.sqlite", clusters, kind[0], trials[arm]["manifest"]["width"])
        for arm, kind in ARMS.items()
    }
    strata = {"all": clusters, "held_out_arity_four": {
        key: roots for key, roots in clusters.items() if registry["shapes_by_cluster"][key][0] == 4}}
    counts = {
        arm: {phase: {key: pair_counts(stage, subset) for key, subset in strata.items()}
              for phase, stage in trial["stages"].items()}
        for arm, trial in trials.items()
    }
    witness = json.loads(args.witness.read_bytes())
    if (witness["cohort_sha256"] != cohort_sha or set(witness["pairs"]) != set(clusters)
            or any({row["root"] for row in pair} != set(clusters[key]) for key, pair in witness["pairs"].items())):
        raise ValueError("native structural witness does not cover the full development cohort")
    outcome, intervention = audit_intervention(args.intervention, trials["hybrid"], cohort_sha)
    resource = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    if resource["cohort"] != cohort_sha or resource["statuses"] != {arm: 0 for arm in ARMS}:
        raise ValueError("one or more registered campaign arms failed")
    summary = {
        "scope": registry["scope"], "cohort_sha256": cohort_sha, "n_pairs": len(clusters), "n_worlds": len(cases),
        "native_training_batches": native_batches,
        "counts": counts, "fact_switch": {name: summarize_fact_switch(row) for name, row in all_margins.items()},
        "matched_proposals": proposals, "intervention": outcome, "resource": resource,
        "max_witness_mean_gap": max(abs(row["linear_margin"]) for pair in witness["pairs"].values() for row in pair),
        "parameters": {arm: {key: trial["manifest"][key] for key in ("parameters", "optimizer_parameters")}
                       for arm, trial in trials.items()},
        "training": {arm: trial["training"] for arm, trial in trials.items()},
    }
    return summary, {"summary": summary, "factorial_witness": witness, "intervention": intervention,
                     "margins": all_margins, "trials": trials}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "witness", "intervention", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary, evidence = audit(args)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for arm, phase in summary["counts"].items():
        print(arm, phase["initial"]["all"], phase["final"]["all"])
    print("intervention", summary["intervention"])


if __name__ == "__main__":
    main()
