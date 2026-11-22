"""Audit matched imitation warm starts, frozen continuation and native mixed-source outcomes."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import torch

from configs.analyze_ppt_conjunction_seed_replication import first_choices
from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_transport_pilot import audit_proposals
from configs.check_ppt_conjunction_pilot import ProofLengthWitness
from configs.freeze_ppt_root_adaptation import ARMS, SEEDS
from ppt_gym import ram_store
from ppt_gym.data import canonical, sha256_file
from ppt_gym.hybrid_rl.corpus import load
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash


def audit_narrow_updates(db, original, calibration):
    current = ram_store.checkpoint(db)["model"]
    if tensor_hash({name: current[name] for name in original}) != tensor_hash(original):
        raise ValueError("first-action adaptation changed a warm-start continuation weight")
    if calibration:
        for name, initial in original.items():
            if name.startswith("graph.transport.") and name != "graph.transport.action.weight":
                copied = current["graph.root_transport." + name.removeprefix("graph.transport.")]
                if not torch.equal(copied, initial):
                    raise ValueError("readout calibration changed the copied recurrent body")
    return float(current["graph.root_gain"])


def source_exposure(db, training):
    by_source = defaultdict(Counter)
    for row in ram_store.ledger(db):
        if row["phase"] == "train":
            for ep in row["episodes"]:
                case = training[ep["root"]]
                counts = by_source[case.source]
                counts["episodes"] += 1
                if ep["verified_closed"] and ep["within_cpu_budget"]:
                    cert = ep["certificate"]
                    if not (ep["live_kernel"] and cert["passed"] and cert["axioms"] == []):
                        raise ValueError("mixed native training success did not independently compile")
                    counts["native_certificates"] += 1
        elif row["phase"] == "mathcan":
            for ep in row["episodes"]:
                counts = by_source[ep["source"]]
                counts["episodes"] += 1
                counts["archived_closures"] += int(ep["archived_closure"])
    return {source: dict(counts) for source, counts in by_source.items()}


def analyze(args):
    doc = json.loads(args.cohort.read_bytes())
    reg = doc["registration"]
    grouped = load(args.cohort)
    train = {case.root: case for case in grouped["train"]}
    cases = {case.root: case for case in grouped["dev"]}
    expected = {key for key, case in train.items() if case.source == "evalmath_lg01"}
    if (tuple(reg["model_seeds"]) != SEEDS or tuple(reg["controls"]) != ARMS[1:]
            or expected != set(reg["evalmath_train_roots"]) or not expected):
        raise ValueError("warm-start registry or EvalMath source membership changed")
    clusters = defaultdict(list)
    for root, case in cases.items():
        clusters[case.cluster].append(root)
    if set(map(len, clusters.values())) != {2}:
        raise ValueError("warm-start evaluation requires two worlds per layout")
    cohort_sha = sha256_file(args.cohort)
    resources = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    if resources["cohort"] != cohort_sha or resources["statuses"] != {f"{s}-{a}": 0 for s in SEEDS for a in ARMS}:
        raise ValueError("warm-start campaign is incomplete")
    configure(2, 0)
    witness = ProofLengthWitness()
    seeds, raw = {}, {}
    for seed in SEEDS:
        folder = "hybrid-conjunction-pilot" if seed == 103 else f"hybrid-conjunction-seed-{seed}"
        source = args.initialization_work / folder / f"{seed}-hybrid_imitation.sqlite"
        initial = read_arm(source, cases, reg["initialization_cohort_sha256"])
        with ram_store.open_store(source) as db:
            weights = ram_store.checkpoint(db)["model"]
        trials, protocols, exposures, schedules = {}, None, set(), set()
        seeds[seed] = {"imitation_before": pair_counts(initial["stages"]["final"], clusters)}
        for arm in ARMS:
            path = args.sessions / f"{seed}-{arm}.sqlite"
            trial = read_arm(path, cases, cohort_sha)
            m = trial["manifest"]
            optimization = ("imitation" if arm == "root_imitation" else "rl", arm == "root_calibration",
                            arm != "warm_full_rl")
            if ((m["objective"], m["head_only"], m["root_fact_adapter"]) != optimization
                    or m["seed"] != seed or not m["premise_interaction"] or m["architecture"] != "hybrid"
                    or m["alias_training"] != "contrast-search"
                    or m["initialization"]["session_sha256"] != initial["session_sha256"]
                    or m["initialization"]["imitation_weights_sha256"] != initial["final_weights_sha256"]):
                raise ValueError("warm-start arm differs from the registered objective or initialization")
            shared = tuple(m[field] for field in PROTOCOL_FIELDS if field != "objective")
            protocols = shared if protocols is None else protocols
            if shared != protocols:
                raise ValueError("warm-start arms used different source and execution budgets")
            original = initial["stages"]["final"]
            before, after = trial["stages"]["initial"], trial["stages"]["final"]
            if any(before[key]["tactics"] != original[key]["tactics"] for key in cases):
                raise ValueError("warm-start policy did not reproduce its imitation trajectory")
            exposures.add(trial["training_exposure_sha256"])
            proposal = audit_proposals(path, train, m["rollouts"], require_both=True)
            schedules.add(proposal["schedule_sha256"])
            roots = first_choices(path, witness)
            if set(roots) != set(cases) or any(after[key]["certified"] and not correct for key, correct in roots.items()):
                raise ValueError("native proof contradicted first-choice witness")
            with ram_store.open_store(path) as db:
                gain = audit_narrow_updates(db, weights, m["head_only"]) if m["root_fact_adapter"] else None
                training = source_exposure(db, train)
            if training["evalmath_lg01"]["episodes"] != len(expected) * m["rollouts"]:
                raise ValueError("some registered EvalMath declarations were omitted from training")
            changed_first = sum(before[key]["tactics"][:1] != after[key]["tactics"][:1] for key in cases)
            changed_path = sum(before[key]["tactics"] != after[key]["tactics"] for key in cases)
            if m["root_fact_adapter"] and changed_first != changed_path:
                raise ValueError("continuation changed despite preserving the selected first tactic")
            trial.update(source_exposure=training, matched_proposals=proposal, feasible_first=roots)
            trials[arm] = trial
            seeds[seed][arm] = {
                **pair_counts(after, clusters), "trainable": m["optimizer_parameters"], "parameters": m["parameters"],
                "changed_first": changed_first, "changed_paths": changed_path,
                "feasible_first": sum(roots.values()), "failed_after_feasible_first": sum(
                    correct and not after[key]["certified"] for key, correct in roots.items()),
                "root_gain_parameter": gain, "source_exposure": training,
            }
        if len(exposures) != 1 or len(schedules) != 1:
            raise ValueError("warm-start arms did not see identical roots and root proposals")
        raw[seed] = {"imitation_before": initial, "trials": trials}
    summary = {"scope": reg["scope"], "layout_pairs": len(clusters), "cohort_sha256": cohort_sha, "seeds": seeds,
               "source_sha256": doc["source_sha256"], "resources": resources, "confirmation_scored": False}
    return summary, {"summary": summary, "seeds": raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "initialization_work", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary, raw = analyze(args)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(raw).encode("utf-8"), mtime=0))
    for seed, arms in summary["seeds"].items():
        print(seed, {arm: (row["proofs"], row["solved_both_worlds"]) for arm, row in arms.items()})


if __name__ == "__main__":
    main()
