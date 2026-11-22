"""Audit shared recovery proposals, native proof outcomes and intervention dose."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import torch

from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_root_adaptation import source_exposure
from configs.freeze_ppt_branch_recovery import ARMS, PROOF_CHECKS
from configs.probe_ppt_branch_recovery import MODES
from configs.run_ppt_branch_recovery import registered_seed
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, load
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash


def checked_certificate(certificate):
    return (certificate is not None and certificate["passed"] and certificate["axioms"] == []
            and digest(certificate["source"]) == certificate["source_digest"])


def verify_witness(path, cases, cohort_sha, reg):
    reference = json.loads(path.read_bytes())
    if reference["cohort_sha256"] != cohort_sha or len(reference["pairs"]) != 24:
        raise ValueError("incomplete native recovery reference")
    winners, groups = {}, {}
    for cluster, pair in reference["pairs"].items():
        a, b = pair
        if a["winner"] == b["winner"]:
            raise ValueError("the reference worlds did not swap the recovery decision")
        groups[cluster] = [row["root"] for row in pair]
        for row in pair:
            root, winner = row["root"], row["winner"]
            yes, no = row["outcomes"][winner], row["outcomes"][1 - winner]
            if (cases[root].cluster != cluster or yes["decision"] != winner or yes["checks"] != PROOF_CHECKS + winner
                    or not yes["success"] or no["success"] or not checked_certificate(yes["certificate"])
                    or any(not x["native"]["replay_passed"] for x in (yes, no))):
                raise ValueError("invalid reference certificate or recovery outcome")
            winners[root] = winner
    if set(winners) != set(cases):
        raise ValueError("reference roots differ from the development cohort")
    partitions = {name: {digest(["early-branch-recovery-v1", shape]) for shape in pool}
                  for name, pool in {**reg["structural_pairs"], "reserved": reg["reserved_pairs"]}.items()}
    if (partitions["dev"] != set(groups) or len(partitions["reserved"]) != 192
            or partitions["train"] & partitions["dev"]
            or partitions["reserved"] & (partitions["train"] | partitions["dev"])):
        raise ValueError("recovery structural partitions overlap")
    return reference, groups, winners


def decision(ep):
    record, = (e for e in ep["events"] if e["kind"] == "recovery")
    first = ep["events"][0]
    if (first["kind"] != "tactic" or first["selected_index"] != 0 or record["checks_used"] != 1
            or record["input"]["explored"] != 0):
        raise ValueError("recovery policy departed from the shared speculative first tactic")
    return record


def read_recovery_ledger(db, train, winners, seed):
    phases = {"initial": {}, "final": {}}
    correct = {"initial": {}, "final": {}}
    batches = []
    counts = Counter()
    for row in ram_store.ledger(db):
        phase, episodes = row["phase"], row["episodes"]
        if phase in phases:
            ep, = episodes
            record = decision(ep)
            phases[phase][ep["root"]] = record["selected_index"]
            correct[phase][ep["root"]] = record["selected_index"] == winners[ep["root"]]
        elif phase == "train" and episodes[0]["family"] == RECOVERY_FAMILY:
            left, right = episodes
            a, b = (decision(ep) for ep in (left, right))
            cluster = train[left["root"]].cluster
            choice = int(digest([seed, cluster, "recovery-proposal"])[0], 16) % 2
            if (left["root"] == right["root"] or train[right["root"]].cluster != cluster
                    or a["selected_index"] != choice or b["selected_index"] != choice
                    or a["input"]["parent_menu"] != b["input"]["parent_menu"]
                    or not all(ep["within_cpu_budget"] for ep in episodes)):
                raise ValueError("recovery worlds did not receive the same registered decision")
            successes = sum(ep["verified_closed"] for ep in episodes)
            updated = row["weights_before"] != row["weights_after"]
            counts.update(training_pairs=1, native_trials=2, certificates=successes, updated_pairs=updated,
                          pairs_without_certificate=int(successes == 0),
                          updates_without_certificate=int(successes == 0 and updated))
            batches.append({"cluster": cluster, "choice": choice, "certificates": successes})
    if any(set(items) != set(winners) for items in phases.values()) or not batches:
        raise ValueError("some development worlds or recovery training cases are missing")
    return phases, correct, batches, dict(counts)


def audit_probe(path, trial, groups, cohort_sha):
    data = json.loads(gzip.decompress(path.read_bytes()))
    if (data["session_sha256"] != trial["session_sha256"] or data["weights_sha256"] != trial["final_weights_sha256"]
            or data["cohort_sha256"] != cohort_sha or set(data["roots"]) != set(trial["stages"]["final"])):
        raise ValueError("recovery edit does not identify the frozen model")
    counts = {}
    for mode in MODES:
        stage, decisions = {}, 0
        for root, trace in data["roots"].items():
            row = trace[mode]
            if (not row["native_replay"]["replay_passed"] or row["hits"] != int(mode != "pristine")
                    or (row["success"] and not checked_certificate(row["certificate"]))):
                raise ValueError("recovery edit failed proof verification or exact dose")
            stage[root] = {"certified": int(row["success"])}
            decisions += row["choice"] != trace["pristine"]["choice"]
        counts[mode] = {**pair_counts(stage, groups), "changed_decisions": decisions}
    return counts, data


def audit(args):
    configure(2, 0)
    doc = json.loads(args.cohort.read_bytes())
    reg = doc["registration"]
    cohort_sha = sha256_file(args.cohort)
    seed = registered_seed(args.cohort, args.seed_study, args.seed)
    data = load(args.cohort)
    cases = {case.root: case for case in data["dev"]}
    training = {case.root: case for case in data["train"]}
    reference, groups, winners = verify_witness(args.sessions / "native-witness.json", cases, cohort_sha, reg)
    shifted_keys = {digest(["early-branch-recovery-v1", shape]) for shape in reg["structural_pairs"]["dev"]
                    if all(branch[0] == 4 for branch in shape)}
    shifted = {key: groups[key] for key in shifted_keys}
    if len(shifted) != 12:
        raise ValueError("missing four-premise recovery controls")
    resources = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    if resources["cohort"] != cohort_sha or resources["statuses"] != dict.fromkeys(ARMS, 0):
        raise ValueError("some registered recovery arms did not complete")
    sources = {}
    for arch, old in (("hybrid", "hybrid_imitation"), ("trm", "trm")):
        source = args.initialization / f"{seed}-{old}.sqlite"
        with ram_store.open_store(source) as db:
            sources[arch] = (sha256_file(source), ram_store.checkpoint(db)["model"])
    studies, raw, initial_policies, exposures, proposals = {}, {}, {}, set(), set()
    protocol = None
    for arm in ARMS:
        architecture = "hybrid" if arm.startswith("hybrid_") else "trm"
        path = args.sessions / (arm + ".sqlite")
        trial = read_arm(path, cases, cohort_sha)
        m = trial["manifest"]
        source_sha, weights = sources[architecture]
        initkey = "imitation_weights_sha256" if architecture == "hybrid" else "source_weights_sha256"
        if (m["architecture"] != architecture or m["seed"] != seed or not m.get("branch_recovery")
                or not m.get("prefer_direct_fact") or m["head_only"] != arm.endswith("_calibration")
                or m["objective"] != ("imitation" if arm == "hybrid_imitation" else "rl")
                or m["initialization"] != {"session_sha256": source_sha, initkey: tensor_hash(weights)}):
            raise ValueError("unregistered recovery policy or warm checkpoint")
        common = tuple(m[key] for key in PROTOCOL_FIELDS if key != "objective")
        protocol = common if protocol is None else protocol
        if common != protocol:
            raise ValueError("recovery arms did not use the same protocol")
        with ram_store.open_store(path) as db:
            phases, correct, batches, train_counts = read_recovery_ledger(db, training, winners, seed)
            mixed = source_exposure(db, training)
            final_weights = ram_store.checkpoint(db)["model"]
        initial = {root: (phases["initial"][root], row["tactics"])
                   for root, row in trial["stages"]["initial"].items()}
        if initial != initial_policies.setdefault(architecture, initial):
            raise ValueError("same-architecture recovery arms did not begin with the same policy decisions")
        if arm == "hybrid_calibration":
            for name, value in weights.items():
                if name.startswith("graph.transport.") and name != "graph.transport.action.weight":
                    clone = name.replace("graph.transport.", "recovery_head.transport.")
                    if not torch.equal(final_weights[clone], value):
                        raise ValueError("readout calibration changed recurrent recovery weights")
            if torch.count_nonzero(final_weights["recovery_head.transport.arrival"]):
                raise ValueError("readout calibration changed the recovery arrival projection")
        exposures.add(trial["training_exposure_sha256"])
        proposals.add(digest([[row["cluster"], row["choice"]] for row in batches]))
        study = {}
        for phase, stage in trial["stages"].items():
            if any(event["certified"] and not correct[phase][r] for r, event in stage.items()):
                raise ValueError("recorded certificate contradicted the native recovery reference")
            study[phase] = {**pair_counts(stage, groups), "four_premises": pair_counts(stage, shifted),
                            "correct_decisions": sum(correct[phase].values()),
                            "early_abandons": sum(phases[phase].values())}
        studies[arm] = {**study, "training": train_counts, "mixed_sources": mixed,
                        "total_parameters": m["parameters"], "trainable_parameters": m["optimizer_parameters"]}
        raw[arm] = {"trial": trial, "decisions": phases, "correct": correct, "training_proposals": batches}
    if len(exposures) != 1 or len(proposals) != 1:
        raise ValueError("recovery arms used different proposal schedules or source exposure")
    probe, probe_raw = audit_probe(args.probe, raw[ARMS[0]]["trial"], groups, cohort_sha)
    study_sha = sha256_file(args.seed_study) if args.seed_study else None
    summary = {"scope": reg["scope"], "cohort_sha256": cohort_sha, "seed": seed, "pairs": len(groups),
               "seed_study_sha256": study_sha,
               "arms": studies, "intervention": probe, "memory": resources["memory"], "confirmation_scored": False}
    return summary, {"summary": summary, "witness": reference, "trials": raw, "intervention": probe_raw}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "initialization", "probe", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--seed-study", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    summary, evidence = audit(args)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for name, row in summary["arms"].items():
        print(name, row["final"], row["training"])
    print("recovery probe", summary["intervention"])


if __name__ == "__main__":
    main()
