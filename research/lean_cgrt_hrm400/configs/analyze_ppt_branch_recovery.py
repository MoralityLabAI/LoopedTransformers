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
from configs.freeze_ppt_pending_alias import GENERATOR as PENDING_GENERATOR
from configs.probe_ppt_branch_recovery import MODES
from configs.run_ppt_branch_recovery import PROGRESS_ARMS, TERMINAL_CONTROLS, UNROUTED, registered_progress, registered_seed
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file
from ppt_gym.hybrid_rl.corpus import RECOVERY_FAMILY, load
from ppt_gym.hybrid_rl.mathcan import ARCHIVE_SHA
from ppt_gym.ram_campaign import configure
from ppt_gym.ram_learning import tensor_hash
from ppt_gym.recovery_progress import tied_failure_preference


def checked_certificate(certificate):
    return (certificate is not None and certificate["passed"] and certificate["axioms"] == []
            and digest(certificate["source"]) == certificate["source_digest"])


def verify_witness(path, cases, cohort_sha, reg):
    reference = json.loads(path.read_bytes())
    if (reference["cohort_sha256"] != cohort_sha
            or len(reference["pairs"]) != len(reg["structural_pairs"]["dev"])):
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
    routed = reg.get("pending_arrival_ablation") == UNROUTED
    generator = PENDING_GENERATOR if routed else "early-branch-recovery-v1"
    development_generator = reg.get("development_layout_namespace", generator)
    partitions = {name: {digest([development_generator if name == "dev" else generator, shape]) for shape in pool}
                  for name, pool in {**reg["structural_pairs"], "reserved": reg.get("reserved_pairs", [])}.items()}
    if (partitions["dev"] != set(groups) or len(partitions["reserved"]) != (0 if routed else 192)
            or partitions["train"] & partitions["dev"]
            or partitions["reserved"] & (partitions["train"] | partitions["dev"])):
        raise ValueError("recovery structural partitions overlap")
    return reference, groups, winners


def decision(ep, *, allow_censored=False):
    records = [e for e in ep["events"] if e["kind"] == "recovery"]
    if (not records and allow_censored and not ep["within_cpu_budget"]
            and ep["status"] == "compute_budget_exhausted"):
        if ep["events"] and (ep["events"][0]["kind"] != "tactic" or ep["events"][0]["selected_index"] != 0):
            raise ValueError("censored search did not start with the registered tactic")
        return None
    record, = records
    first = ep["events"][0]
    if (first["kind"] != "tactic" or first["selected_index"] != 0 or record["checks_used"] != 1
            or record["input"]["explored"] != 0):
        raise ValueError("recovery policy departed from the shared speculative first tactic")
    return record


def recovery_feedback(episodes, records, objective, update, changed, *, progress_credit=False):
    goal_counts = [len(record["input"]["current"]["state"]["goals"]) for record in records]
    if goal_counts[0] != goal_counts[1] or not goal_counts[0]:
        raise ValueError("paired recovery worlds disagree on whether a goal remains pending")
    family = "pending_goals" if goal_counts[0] > 1 else "single_goal"
    certificates = sum(ep["verified_closed"] for ep in episodes)
    # The learner uses one unit for a certificate and a cost of 1/40 for every check.
    returns = [40 * int(ep["verified_closed"]) - ep["tactic_checks"] for ep in episodes]
    unequal = returns[0] != returns[1]
    if update is None or update.get("recovery_pairs") != 1:
        raise ValueError("a budget-valid recovery pair lacks a recovery update receipt")
    preference = tied_failure_preference(episodes) if progress_credit else 0
    if update.get("progress_preference") != (preference if progress_credit else None):
        raise ValueError("the recovery update used a different partial-progress signal")
    contrast = unequal or preference != 0
    expected_decisions = 2 if objective == "rl" or preference else certificates
    if update["decisions"] != expected_decisions:
        raise ValueError("the learner used an unexpected number of training decisions")
    counts = {
        "pairs": 1, "parameter_updates": int(changed), "without_certificates": int(certificates == 0),
        "with_one_certificate": int(certificates == 1),
        "pairs_with_unequal_returns": int(unequal),
    }
    if progress_credit:
        counts["direct_progress_preferences"] = int(preference != 0)
    if objective == "rl":
        # Equal returns make the recovery policy term zero. The critic and optimizer can still move.
        counts["direct_policy_contrast_without_certificate"] = int(contrast and certificates == 0)
        counts["direct_policy_contrast_with_an_uncertified_world"] = int(contrast and certificates < 2)
        tied_update = "parameter_updates_without_policy_contrast" if progress_credit else "equal_return_parameter_updates"
        counts[tied_update] = int(not contrast and changed)
    return family, counts


def read_recovery_ledger(db, train, winners, seed, *, feedback_objective=None, progress_credit=False):
    phases = {"initial": {}, "final": {}}
    correct = {"initial": {}, "final": {}}
    batches = []
    counts = Counter()
    feedback = {}
    cpu_censored = {"initial": {}, "final": {}}
    routed = feedback_objective is not None
    for row in ram_store.ledger(db):
        phase, episodes = row["phase"], row["episodes"]
        if phase in phases:
            ep, = episodes
            record = decision(ep, allow_censored=routed)
            selected = record["selected_index"] if record is not None else None
            phases[phase][ep["root"]] = selected
            correct[phase][ep["root"]] = selected == winners[ep["root"]]
            if not ep["within_cpu_budget"]:
                cpu_censored[phase][ep["root"]] = ep["status"]
        elif phase == "train" and episodes[0]["family"] == RECOVERY_FAMILY:
            left, right = episodes
            records = [decision(ep, allow_censored=routed) for ep in episodes]
            observed = [record for record in records if record is not None]
            cluster = train[left["root"]].cluster
            choice = int(digest([seed, cluster, "recovery-proposal"])[0], 16) % 2
            within_cpu = all(ep["within_cpu_budget"] for ep in episodes)
            if (left["root"] == right["root"] or train[right["root"]].cluster != cluster
                    or any(record["selected_index"] != choice for record in observed)
                    or (len(observed) == 2 and observed[0]["input"]["parent_menu"] != observed[1]["input"]["parent_menu"])
                    or (not routed and not within_cpu)):
                raise ValueError("recovery worlds did not receive the same registered decision")
            successes = sum(ep["verified_closed"] for ep in episodes)
            updated = row["weights_before"] != row["weights_after"]
            counts.update(training_pairs=1, native_trials=2, certificates=successes, updated_pairs=updated,
                          pairs_without_certificate=int(successes == 0),
                          updates_without_certificate=int(successes == 0 and updated))
            batches.append({"cluster": cluster, "choice": choice, "certificates": successes})
            if routed and within_cpu:
                family, conditions = recovery_feedback(episodes, records, feedback_objective, row["update"], updated,
                                                       progress_credit=progress_credit)
                feedback.setdefault(family, Counter()).update(conditions)
            elif routed:
                if updated or row["update"] != {"decisions": 0, "loss": None, "recovery_pairs": 0}:
                    raise ValueError("a CPU-censored recovery batch received a policy update")
                counts["cpu_censored_recovery_pairs"] += 1
    if any(set(items) != set(winners) for items in phases.values()) or not batches:
        raise ValueError("some development worlds or recovery training cases are missing")
    training = dict(counts)
    if routed:
        training["feedback_by_goal_count"] = {family: dict(values) for family, values in feedback.items()}
    return phases, correct, batches, training, cpu_censored


def check_initial_choices(reference, current, *, allow_censored):
    prefixes = 0
    for root, first in reference.items():
        other = current[root]
        if first[:2] == other[:2]:
            continue
        short, long = sorted((first, other), key=lambda row: (len(row[1]), row[0] is not None))
        if (not allow_censored or short[2] != "compute_budget_exhausted"
                or short[1] != long[1][:len(short[1])] or (short[0] is not None and short[0] != long[0])):
            raise ValueError("same-architecture recovery arms disagreed before any CPU censor")
        prefixes += 1
    return prefixes


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
    progress_study = getattr(args, "progress_study", None)
    progress = registered_progress(args.cohort, progress_study, seed)
    progress_sha = json.loads(progress_study.read_bytes())["progress_signal_sha256"] if progress else None
    data = load(args.cohort)
    cases = {case.root: case for case in data["dev"]}
    training = {case.root: case for case in data["train"]}
    reference, groups, winners = verify_witness(args.sessions / "native-witness.json", cases, cohort_sha, reg)
    routed = reg.get("pending_arrival_ablation") == UNROUTED
    arms = PROGRESS_ARMS if progress else (*ARMS, UNROUTED) if routed else ARMS
    layouts = reg["structural_pairs"]["dev"]
    if routed:
        generator = reg.get("development_layout_namespace", PENDING_GENERATOR)
        mixed = {digest([generator, shape]) for shape in layouts if {branch[0] for branch in shape} == {2, 3}}
        if len(mixed) != reg["counts"]["mixed_arity_development_pairs"]:
            raise ValueError("missing the registered mixed-arity pending-goal layouts")
        strata = {"mixed_arity": mixed, "both_three_premises": set(groups) - mixed}
    else:
        shifted = {digest(["early-branch-recovery-v1", shape]) for shape in layouts
                   if all(branch[0] == 4 for branch in shape)}
        if len(shifted) != 12:
            raise ValueError("missing four-premise recovery controls")
        strata = {"four_premises": shifted}
    resources = json.loads((args.sessions / "campaign-resource-audit.json").read_bytes())
    if resources["cohort"] != cohort_sha or resources["statuses"] != dict.fromkeys(arms, 0):
        raise ValueError("some registered recovery arms did not complete")
    sources = {}
    for arch, old in (("hybrid", "hybrid_imitation"), ("trm", "trm")):
        source = args.initialization / f"{seed}-{old}.sqlite"
        with ram_store.open_store(source) as db:
            sources[arch] = (sha256_file(source), ram_store.checkpoint(db)["model"])
    studies, raw, initial_policies, exposures, proposals = {}, {}, {}, set(), set()
    protocol = None
    for arm in arms:
        architecture = "hybrid" if arm.startswith("hybrid_") else "trm"
        path = args.sessions / (arm + ".sqlite")
        trial = read_arm(path, cases, cohort_sha)
        m = trial["manifest"]
        source_sha, weights = sources[architecture]
        expected_router = routed and architecture == "hybrid" and arm != UNROUTED
        initkey = "imitation_weights_sha256" if architecture == "hybrid" else "source_weights_sha256"
        if (m["architecture"] != architecture or m["seed"] != seed or not m.get("branch_recovery")
                or not m.get("prefer_direct_fact") or m["head_only"] != arm.endswith("_calibration")
                or m["objective"] != ("imitation" if arm.endswith("_imitation") else "rl")
                or bool(m.get("pending_arrival_router", False)) != expected_router
                or m.get("recovery_progress_credit") != (progress_sha if arm not in TERMINAL_CONTROLS else None)
                or m["initialization"] != {"session_sha256": source_sha, initkey: tensor_hash(weights)}):
            raise ValueError("unregistered recovery policy or warm checkpoint")
        common = tuple(m[key] for key in PROTOCOL_FIELDS if key != "objective")
        protocol = common if protocol is None else protocol
        if common != protocol:
            raise ValueError("recovery arms did not use the same protocol")
        with ram_store.open_store(path) as db:
            phases, correct, batches, train_counts, cpu_censored = read_recovery_ledger(
                db, training, winners, seed, feedback_objective=m["objective"] if routed else None,
                progress_credit=bool(m.get("recovery_progress_credit"))
            )
            mixed = source_exposure(db, training)
            final_weights = ram_store.checkpoint(db)["model"]
        if routed:
            expected_pairs = {case.cluster for case in training.values() if case.family == RECOVERY_FAMILY}
            realized_pairs = [row["cluster"] for row in batches]
            if len(realized_pairs) != len(expected_pairs) or set(realized_pairs) != expected_pairs:
                raise ValueError("routed training did not cover each registered recovery pair exactly once")
            expected_native = Counter()
            for case in training.values():
                expected_native[case.source] += 1 if case.family == RECOVERY_FAMILY else m["rollouts"]
            archive = mixed.get("mathcan_ram08_audited_train", {})
            if (m["mathcan_archive_sha256"] != ARCHIVE_SHA or not archive.get("episodes")
                    or any(mixed.get(source, {}).get("episodes") != count
                           for source, count in expected_native.items())):
                raise ValueError("routed training is missing registered native or archived source exposures")
        initial = {root: (phases["initial"][root], row["tactics"], cpu_censored["initial"].get(root))
                   for root, row in trial["stages"]["initial"].items()}
        initial_prefixes = check_initial_choices(initial_policies.setdefault(architecture, initial), initial,
                                                 allow_censored=routed)
        if arm == "hybrid_calibration":
            for name, value in weights.items():
                if name.startswith("graph.transport.") and name != "graph.transport.action.weight":
                    clone = name.replace("graph.transport.", "recovery_head.transport.")
                    if not torch.equal(final_weights[clone], value):
                        raise ValueError("readout calibration changed recurrent recovery weights")
            if torch.count_nonzero(final_weights["recovery_head.transport.arrival"]):
                raise ValueError("readout calibration changed the recovery arrival projection")
        gate = {}
        if expected_router:
            weight, bias = (final_weights[f"recovery_head.pending_arrival.{name}"] for name in ("weight", "bias"))
            if not torch.isfinite(weight).all() or not torch.isfinite(bias).all():
                raise ValueError("pending arrival controller has nonfinite weights")
            if arm == "hybrid_calibration" and (torch.count_nonzero(weight) or float(bias.item()) != -2.):
                raise ValueError("readout calibration changed the frozen pending-arrival router")
            gate = {"weight": weight.tolist(), "bias": bias.tolist()}
        exposures.add(trial["training_exposure_sha256"])
        proposals.add(digest([[row["cluster"], row["choice"]] for row in batches]))
        study = {}
        for phase, stage in trial["stages"].items():
            if any(event["certified"] and not correct[phase][r] for r, event in stage.items()):
                raise ValueError("recorded certificate contradicted the native recovery reference")
            by_stratum = {name: pair_counts(stage, {key: groups[key] for key in selected})
                          for name, selected in strata.items()}
            study[phase] = {**pair_counts(stage, groups), **by_stratum,
                            "correct_decisions": sum(correct[phase].values()),
                            "early_abandons": sum(choice == 1 for choice in phases[phase].values())}
            if routed:
                study[phase]["cpu_censored_worlds"] = len(cpu_censored[phase])
        studies[arm] = {**study, "training": train_counts, "mixed_sources": mixed,
                        "total_parameters": m["parameters"], "trainable_parameters": m["optimizer_parameters"]}
        if routed:
            studies[arm]["initial_cpu_truncated_prefixes"] = initial_prefixes
        if expected_router:
            studies[arm]["pending_arrival_parameters"] = gate
        raw[arm] = {"trial": trial, "decisions": phases, "correct": correct, "training_proposals": batches}
        if routed:
            raw[arm]["cpu_censored_evaluation"] = cpu_censored
    if len(exposures) != 1 or len(proposals) != 1:
        raise ValueError("recovery arms used different proposal schedules or source exposure")
    probe, probe_raw = audit_probe(args.probe, raw[ARMS[0]]["trial"], groups, cohort_sha) if args.probe else ({}, {})
    study_sha = sha256_file(args.seed_study) if args.seed_study else None
    summary = {"scope": reg["scope"], "cohort_sha256": cohort_sha, "seed": seed, "pairs": len(groups),
               "seed_study_sha256": study_sha,
               "arms": studies, "intervention": probe, "memory": resources["memory"], "confirmation_scored": False}
    if "attempts" in resources:
        summary["resource_attempts"] = resources["attempts"]
    if progress:
        summary["progress_study_sha256"] = sha256_file(progress_study)
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
