"""Audit frozen calibration outcomes and plan resolution for the sealed finite domain."""
import argparse
from collections import defaultdict
from fractions import Fraction
import gzip
import json
from math import ceil, comb, log, sqrt
from pathlib import Path
import random
from statistics import mean

from configs.analyze_ppt_graph_alias_pilot import pair_counts
from configs.check_ppt_conjunction_pilot import ProofLengthWitness
from configs.eval_ppt_conjunction_calibration import read_calibration
from configs.freeze_ppt_conjunction_followup import INTERVENTIONS, domain
from configs.run_ppt_conjunction_pilot import ARMS
from ppt_gym import ram_store
from ppt_gym.data import canonical, digest, sha256_file


def read_arm(path, arm, cases, registry_sha, data_sha, root_witness):
    rows, raw = {}, {}
    with ram_store.open_store(path) as db:
        identity = ram_store.read_manifest(db)
        if (identity["arm"], identity["registry_sha256"], identity["calibration_sha256"]) != (
            arm, registry_sha, data_sha
        ):
            raise ValueError("calibration arm has wrong provenance")
        for row, case in zip(ram_store.ledger(db), cases, strict=True):
            ep = row["episode"]
            if (ep["root"] != case.root or ep["split"] != "calibration" or ep["tactic_checks"] > 8
                    or ep["model_macs"] > identity["max_macs"] or not ep["native_replay"]["replay_passed"]):
                raise ValueError("calibration ledger failed custody, budget or native replay audit")
            if arm in INTERVENTIONS and ep["interaction_reads_edited"] != 1:
                raise ValueError("the root-only intervention did not reach exactly one readout")
            success = ep["verified_closed"] and ep["within_cpu_budget"]
            cert = ep["certificate"]
            if success and not (cert["passed"] and cert["axioms"] == [] and digest(cert["source"]) == cert["source_digest"]):
                raise ValueError("calibration success is missing its standalone certificate")
            first = ep["events"][0]
            scores, _ = root_witness(canonical(first["input"]), first["candidate_menu"], ep["depth"])
            if len(scores) != 2 or sorted(scores.tolist())[-1] != -8 or int((scores == -8).sum()) != 1:
                raise ValueError("support witness did not identify a unique budget-feasible first action")
            feasible_first = first["selected_index"] == int(scores.argmax())
            if success and not feasible_first:
                raise ValueError("native success contradicted the first-step witness")
            rows[case.root] = {"certified": int(success),
                               "feasible_first": feasible_first,
                               "path": [event["tactic"] for event in ep["events"] if event["kind"] == "tactic"]}
            raw[case.root] = {"tactic_checks": ep["tactic_checks"], "model_macs": ep["model_macs"],
                              "native_replay": ep["native_replay"], "certificate": cert}
    return identity, rows, raw


def sham_quantile_upper_bound(observations, population, quantile=Fraction(95, 100), confidence=Fraction(95, 100)):
    """Order statistic for uniformly sampled distinct shapes, using hypergeometric coverage."""
    below_quantile = ceil(quantile * population) - 1
    n = len(observations)
    target = confidence * comb(population, n)
    cumulative = 0
    ordered = sorted(observations)
    for j, value in enumerate(ordered):
        cumulative += comb(below_quantile, j) * comb(population - below_quantile, n - j)
        if cumulative >= target:
            return value, j + 1
    return None, None


def audit(args):
    registry, cases = read_calibration(args.registry, args.input)
    pool_size = registry["domain_size"]
    admissible = domain()
    expected = registry["calibration"]["pairs"] + registry["confirmation"]["pairs"]
    if (len(admissible) != pool_size
            or canonical(random.Random(registry["domain_seed"]).sample(admissible, len(expected))) != canonical(expected)):
        raise ValueError("calibration and confirmation layouts differ from the frozen draws")
    calibration_clusters = {digest(branches) for branches in registry["calibration"]["pairs"]}
    confirm_clusters = {digest(branches) for branches in registry["confirmation"]["pairs"]}
    if (len(calibration_clusters) != len(cases) // 2 or calibration_clusters & confirm_clusters
            or len(confirm_clusters) != len(registry["confirmation"]["pairs"])):
        raise ValueError("calibration and confirmation structural pairs overlap or repeat")
    reg_sha, data_sha = sha256_file(args.registry), sha256_file(args.input)
    resources = json.loads((args.sessions / "calibration-resources.json").read_bytes())
    arms = [*ARMS, *INTERVENTIONS]
    if (resources["registry_sha256"] != reg_sha or resources["calibration_sha256"] != data_sha
            or resources["statuses"] != {arm: 0 for arm in arms}):
        raise ValueError("incomplete or unregistered calibration campaign")
    clusters = defaultdict(list)
    for case in cases:
        clusters[case.cluster].append(case.root)
    if set(map(len, clusters.values())) != {2}:
        raise ValueError("each calibration pair must contain two worlds")
    witness = ProofLengthWitness()
    entries = {arm: read_arm(args.sessions / (arm + ".sqlite"), arm, cases, reg_sha, data_sha, witness)
               for arm in arms}
    candidate_id, candidate, _ = entries["hybrid"]
    if any(entries[arm][0]["checkpoint_sha256"] != candidate_id["checkpoint_sha256"] for arm in INTERVENTIONS):
        raise ValueError("the intervention and sham must use the frozen candidate weights")
    if any(entries[arm][0]["verifier"] != candidate_id["verifier"] for arm in arms):
        raise ValueError("frozen arms did not use the same native Lean verifier")
    threshold = registry["minimum_effect"]
    effects, proof_counts = {}, {}
    for arm, (_, rows, _) in entries.items():
        proof_counts[arm] = {**pair_counts(rows, clusters),
                             "correct_root_choices": sum(row["feasible_first"] for row in rows.values()),
                             "failed_after_correct_root": sum(row["feasible_first"] and not row["certified"]
                                                             for row in rows.values()),
                             "changed_paths": sum(row["path"] != candidate[root]["path"] for root, row in rows.items())}
        effects[arm] = {key: int(all(rows[root]["certified"] for root in roots)) for key, roots in clusters.items()}
    differences = {arm: {key: value - effects[arm][key] for key, value in effects["hybrid"].items()}
                   for arm in ARMS if arm != "hybrid"}
    differences["interaction_removal"] = {key: effects["control-common-shift"][key] - effects["ablate-interaction"][key]
                                          for key in clusters}
    sham = {key: abs(effects["control-common-shift"][key] - effects["hybrid"][key]) for key in clusters}
    sham_floor, order_stat = sham_quantile_upper_bound(list(sham.values()), pool_size)
    n_confirm = len(confirm_clusters)
    radius = sqrt(2 * log(registry["planned_comparisons"] / registry["alpha"]) / n_confirm)
    variation = max(threshold, sham_floor or 0, 1 / n_confirm)
    effect_means = {arm: mean(values.values()) for arm, values in differences.items()}
    ready = sham_floor is not None and min(effect_means.values()) - radius > variation
    summary = {"registry_sha256": reg_sha, "calibration_sha256": data_sha, "resources": resources,
               "calibration_pairs": len(clusters), "finite_domain_size": pool_size, "proof_counts": proof_counts,
               "paired_effects": effect_means, "observed_worst_sham": max(sham.values()),
               "upper_95pct_sham_quantile": sham_floor, "sham_order_statistic_rank": order_stat,
               "variation_floor": variation, "registered_confirmation_pairs": n_confirm,
               "prospective_hoeffding_radius": radius, "calibration_supports_confirmation": ready,
               "confirmation_scored": False,
               "scope": "one frozen training seed on a uniform sample of distinct composite Horn layouts"}
    return summary, {"summary": summary, "paired_effects_by_cluster": differences,
                     "matched_sham_by_cluster": sham, "arms": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("registry", "input", "sessions", "summary", "evidence"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    summary, evidence = audit(args)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.evidence.write_bytes(gzip.compress(canonical(evidence).encode("utf-8"), mtime=0))
    for arm, count in summary["proof_counts"].items():
        print(arm, count)
    print("prospective resolution", summary["prospective_hoeffding_radius"],
          "supports confirmation", summary["calibration_supports_confirmation"])


if __name__ == "__main__":
    main()
