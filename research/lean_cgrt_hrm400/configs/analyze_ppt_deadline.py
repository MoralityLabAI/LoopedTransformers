"""Audit on-time native proofs, counterfactual root choices and receipt exposure."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import torch

from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.analyze_ppt_hybrid_replication import PROTOCOL_FIELDS, read_arm
from configs.analyze_ppt_root_adaptation import source_exposure
from configs.analyze_ppt_transport_pilot import audit_proposals
from configs.freeze_ppt_deadline import ARMS, DEADLINE, MAX_CHECKS, SEED
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import DEADLINE_FAMILY, load
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash


def validate_witness(path, cohort_sha, eval_cases, registration):
    report = json.loads(path.read_bytes())
    clusters, lookup = {}, {}
    if report["cohort_sha256"] != cohort_sha or len(report["pairs"]) != 24:
        raise ValueError("incomplete native proof-cost reference")
    for cluster, pair in report["pairs"].items():
        a, b = pair
        if a["menu"] != b["menu"] or a["early_first"] == b["early_first"]:
            raise ValueError("native witness did not swap both on-time tactics")
        for row in pair:
            if (eval_cases[row["root"]].cluster != cluster or sorted(row["checks"]) != [DEADLINE, MAX_CHECKS]
                    or row["checks"][row["early_first"]] != DEADLINE or len(row["certificates"]) != 2
                    or not all(c["passed"] and c["axioms"] == [] and digest(c["source"]) == c["source_digest"]
                               for c in row["certificates"])
                    or not all(replay["replay_passed"] for replay in row["replay"])):
                raise ValueError("native deadline witness lacks paired certificates")
            lookup[row["root"]] = row
        clusters[cluster] = [row["root"] for row in pair]
    if set(lookup) != set(eval_cases):
        raise ValueError("reference worlds differ from the registered development layouts")
    memberships = {name: {digest(["proof-deadline-v1", shape]) for shape in pool}
                   for name, pool in {**registration["structural_pairs"], "reserved": registration["reserved_pairs"]}.items()}
    if (memberships["dev"] != set(clusters) or len(memberships["reserved"]) != 480
            or memberships["train"] & memberships["dev"]
            or memberships["reserved"] & (memberships["train"] | memberships["dev"])):
        raise ValueError("structural partitions overlap")
    return clusters, lookup, report


def training_deadlines(db):
    counts = Counter()
    for row in ram_store.ledger(db):
        if row["phase"] != "train" or row["episodes"][0]["family"] != DEADLINE_FAMILY:
            continue
        changed = row["weights_before"] != row["weights_after"]
        counts["deadline_batches"] += 1
        counts["changed_deadline_batches"] += changed
        episodes = [ep for ep in row["episodes"] if not ep.get("reward_censored")]
        on_time = sum(ep["verified_closed"] and ep["within_cpu_budget"] and ep["tactic_checks"] <= DEADLINE
                      for ep in episodes)
        late = sum(ep["verified_closed"] and ep["within_cpu_budget"] and ep["tactic_checks"] > DEADLINE
                   for ep in episodes)
        counts.update(on_time_certificates=on_time, late_certificates=late, total=len(episodes))
        if not on_time:
            counts["batches_without_on_time_proof"] += 1
            counts["changed_without_on_time_proof"] += changed
    return dict(counts)


def stage_counts(db, trial, clusters, witness, unseen_arity):
    first = {}
    for row in ram_store.ledger(db):
        if row["phase"] == "final":
            episode, = row["episodes"]
            event = episode["events"][0]
            expected = witness[episode["root"]]
            if event["candidate_menu"] != expected["menu"]:
                raise ValueError("policy and witness saw different first tactics")
            first[episode["root"]] = event["selected_index"] == expected["early_first"]
    counts = {}
    for phase, rows in trial["stages"].items():
        on_time = {root: {"certified": int(row["certified"] and row["tactic_checks"] <= DEADLINE)}
                   for root, row in rows.items()}
        counts[phase] = {"on_time": pair_counts(on_time, clusters), "full_budget": pair_counts(rows, clusters),
                         "on_time_three_premise": pair_counts(on_time, unseen_arity)}
    if any(row["certified"] and row["tactic_checks"] <= DEADLINE and not first[root]
           for root, row in trial["stages"]["final"].items()):
        raise ValueError("on-time native proof contradicts the paired cost witness")
    counts["final"]["early_first"] = sum(first.values())
    return counts, first


def proof_path_audit(path, trial, clusters, cohort_sha):
    report = json.loads(path.read_bytes())
    if (report["cohort"] != cohort_sha or report["session_sha256"] != trial["session_sha256"]
            or report["weights_sha256"] != trial["final_weights_sha256"]
            or set(report["roots"]) != set(trial["stages"]["final"])):
        raise ValueError("intervention does not identify the original deadline checkpoint")
    summary = {}
    for mode in ("pristine", "ablate-arrival", "control-common-shift"):
        stage, changed = {}, 0
        for root, modes in report["roots"].items():
            row = modes["arms"][mode]
            if not row["native_replay"]["replay_passed"] or row["bank_reads_edited"] != int(mode != "pristine"):
                raise ValueError("intervention did not preserve the registered root-only dose")
            if row["certified"]:
                cert = row["standalone_certificate"]
                if not (cert["passed"] and cert["axioms"] == [] and digest(cert["source"]) == cert["source_digest"]):
                    raise ValueError("invalid intervention certificate")
            stage[root] = {"certified": int(row["certified"] and row["checks"] <= DEADLINE)}
            changed += row["path"] != modes["arms"]["pristine"]["path"]
        summary[mode] = {**pair_counts(stage, clusters), "changed_paths": changed}
    return summary, report


def audit(args):
    doc = json.loads(args.cohort.read_bytes())
    reg = doc["registration"]
    cohort_sha = sha256_file(args.cohort)
    corpus = load(args.cohort)
    train = {case.root: case for case in corpus["train"]}
    dev = {case.root: case for case in corpus["dev"]}
    clusters, witness, native = validate_witness(args.sessions / "native-witness.json", cohort_sha, dev, reg)
    unseen_keys = {digest(["proof-deadline-v1", shape]) for shape in reg["structural_pairs"]["dev"]
                   if len(shape[0][1]) == 3}
    unseen = {key: clusters[key] for key in unseen_keys}
    if len(unseen) != 12:
        raise ValueError("wrong number of three-premise development pairs")
    resource = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    if resource["cohort"] != cohort_sha or resource["statuses"] != dict.fromkeys(ARMS, 0):
        raise ValueError("deadline campaign did not complete all registered arms")
    configure(2, 0)
    sources = {}
    for arch, arm in (("hybrid", "hybrid_imitation"), ("trm", "trm")):
        path = args.initialization / f"{SEED}-{arm}.sqlite"
        with ram_store.open_store(path) as db:
            weights = ram_store.checkpoint(db)["model"]
        sources[arch] = (sha256_file(path), tensor_hash(weights), weights)
    trials, results, protocol, proposals, exposure = {}, {}, None, set(), set()
    for arm in ARMS:
        path = args.sessions / (arm + ".sqlite")
        trial = read_arm(path, dev, cohort_sha)
        m = trial["manifest"]
        arch = "trm" if arm.startswith("trm_") else "hybrid"
        src_sha, weight_sha, source_weights = sources[arch]
        key = "imitation_weights_sha256" if arch == "hybrid" else "source_weights_sha256"
        expected_deadline = None if arm == "all_proofs_imitation" else DEADLINE
        has_arrival = arch == "hybrid" and arm != "terminal_readout_rl"
        objective = "imitation" if arm.endswith("_imitation") else "rl"
        if (m["architecture"] != arch or m["objective"] != objective or m["seed"] != SEED
                or not m["graph_context"] or m["head_only"] != arm.endswith("_calibration")
                or m["root_fact_adapter"] != (arch == "hybrid") or m.get("arrival_transport", False) != has_arrival
                or m.get("proof_deadline") != expected_deadline or m["alias_training"] != "contrast-search"
                or m["initialization"] != {"session_sha256": src_sha, key: weight_sha}):
            raise ValueError("unregistered deadline arm or initialization")
        common = tuple(m[key] for key in PROTOCOL_FIELDS if key != "objective")
        protocol = common if protocol is None else protocol
        if protocol != common:
            raise ValueError("deadline arms ran different tactic or resource protocols")
        exposure.add(trial["training_exposure_sha256"])
        proposal = audit_proposals(path, train, m["rollouts"], require_both=True)
        proposals.add(proposal["schedule_sha256"])
        with ram_store.open_store(path) as db:
            counts, first = stage_counts(db, trial, clusters, witness, unseen)
            outcomes = training_deadlines(db)
            mixed = source_exposure(db, train)
            weights = ram_store.checkpoint(db)["model"]
            if arch == "hybrid" and tensor_hash({n: weights[n] for n in source_weights}) != weight_sha:
                raise ValueError("hybrid deadline arm changed continuation weights")
            if arm == "readout_calibration" and torch.count_nonzero(weights["graph.root_transport.arrival"]):
                raise ValueError("readout calibration changed the frozen arrival projection")
        trial.update(first=first, deadline_training=outcomes, mixed_sources=mixed, proposal=proposal)
        trials[arm] = trial
        results[arm] = {"proofs": counts, "deadline_training": outcomes, "mixed_sources": mixed,
                        "trainable_parameters": m["optimizer_parameters"], "total_parameters": m["parameters"]}
    if len(proposals) != 1 or len(exposure) != 1:
        raise ValueError("the deadline arms did not share roots and proposed first tactics")
    interventions, raw_probe = proof_path_audit(args.probe, trials["arrival_rl"], clusters, cohort_sha)
    report = {"scope": reg["scope"], "seed": SEED, "cohort_sha256": cohort_sha, "pairs": len(clusters),
              "proof_deadline": DEADLINE, "arms": results, "intervention": interventions,
              "memory": resource["memory"], "confirmation_scored": False}
    return report, {"report": report, "trials": trials, "native_witness": native, "arrival_probe": raw_probe}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "sessions", "initialization", "probe", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary, raw = audit(args)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(raw).encode("utf-8"), mtime=0))
    for arm, item in summary["arms"].items():
        print(arm, item["proofs"]["final"], item["deadline_training"])
    print("root intervention", summary["intervention"])


if __name__ == "__main__":
    main()
