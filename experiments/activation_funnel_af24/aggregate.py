"""AF-24 registered eight-seed gate vs exact original AF-23 evidence.

Requires original aggregated JSON extracted from its checksum-verified official
GitHub Actions artifact ZIP (run 43812708916), plus eight independently
audited symbolic seed receipts. A single named graph-aware policy must match
the neural source at both depths in >=6 of 8 seeds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

from experiments.activation_funnel_af23 import branching_gym as gym
from . import evaluate as ex
from . import symbolic as ctl

DEPTHS = ex.SPLITS
MATCH_ALLOWANCE = 4  # 5 percentage points of 80 provable cases
REQUIRED_SEEDS = 6
PRIMARY = ("ancestor_greedy", "closure_direct", "bfs_cap4",
           "bfs_cap16", *ctl.RANDOM)
CI_RUN = 43812708916
EXPECTED_ORIGINAL = "8e52cd0cd61866b35bd9f39ce7c42e62f80398a79b26b63d4360253cb6486d31"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def aggregate(root, original_file):
    original_bytes = Path(original_file).read_bytes()
    if sha(original_bytes) != EXPECTED_ORIGINAL:
        raise ValueError("original AF23 official results checksum mismatch")
    original = json.loads(original_bytes)
    if (original["experiment"], original["track"],
            original["scientific_verdict"], original["both_depths_pass"]) != (
            "AF-23", "B", "PASS", 6):
        raise ValueError("not the original source-verified AF23 PASS")
    if original["branching_manifest_sha256"] != gym.manifest()["manifest_sha256"]:
        raise ValueError("graph manifest drift from original results")
    seeds = {}
    failures = []
    pooled = {
        depth: {arm: {"solved": 0, "provable": 0, "unprovable": 0,
                      "premature_stop": 0, "correct_initial_stop": 0,
                      "false_accepted_proofs": 0,
                      "node_expansions_total": 0, "rule_examinations_total": 0}
                for arm in ex.ARMS}
        for depth in DEPTHS
    }
    for seed in ex.SEEDS:
        folder = Path(root) / f"af24-seed-{seed}"
        pp = folder / "evaluation.json"
        aa = folder / "independent_audit.json"
        ee = folder / "episodes.jsonl"
        if not (pp.is_file() and aa.is_file() and ee.is_file()):
            failures.append(seed)
            continue
        record = json.loads(pp.read_text())
        check = json.loads(aa.read_text())
        if (record["seed"] != seed or check["seed"] != seed
            or not check["verified"] or check["false_accepted_proofs"] != 0
            or check["evaluation_sha256"] != sha(pp.read_bytes())
            or check["episode_sha256"] != sha(ee.read_bytes())
            or record["episodes_sha256"] != check["episode_sha256"]
            or record["original_af23_ci"] != CI_RUN
            or record["arm_names"] != list(ex.ARMS)
            or record["source_protocol_sha256"] != sha((ex.HERE/"PROTOCOL.json").read_bytes())
            or record["source_manifest_sha256"] != original["branching_manifest_sha256"]):
            raise ValueError(f"seed {seed}: source or independent audit discrepancy")
        old = original["seed_results"][str(seed)]["depths"]
        d4_neural = original["pooled"]["held_depth4"]["direct"]["solved"] // 8
        if d4_neural != 80:
            raise ValueError("original AF23 per-seed direct depth4 not saturated")
        d5_neural = old["held_depth5"]["vi2_solved"]
        per_arm = {}
        for arm in ex.ARMS:
            d4 = record["scores"]["held_depth4"][arm]
            d5 = record["scores"]["held_depth5"][arm]
            for depth, rec in (("held_depth4", d4), ("held_depth5", d5)):
                if (rec["provable"], rec["unprovable"]) != (80,112):
                    raise ValueError("incorrect registered positive/negative cohort")
                for key in pooled[depth][arm]:
                    pooled[depth][arm][key] += rec[key]
            matched = bool(
                d4["solved"] >= d4_neural - MATCH_ALLOWANCE
                and d5["solved"] >= d5_neural - MATCH_ALLOWANCE
                and not d4["false_accepted_proofs"]
                and not d5["false_accepted_proofs"]
            )
            per_arm[arm] = {
                "matched_at_both_depths": matched,
                "depth4_solved": d4["solved"], "depth5_solved": d5["solved"],
                "depth4_direct_neural": d4_neural,
                "depth5_vi2_neural": d5_neural,
                "depth4_rule_examinations": d4["rule_examinations_total"],
                "depth5_rule_examinations": d5["rule_examinations_total"],
                "depth4_expansions": d4["node_expansions_total"],
                "depth5_expansions": d5["node_expansions_total"],
                "correct_stop_negative": [
                    d4["correct_initial_stop"],d5["correct_initial_stop"]
                ],
            }
        if per_arm["fixed"]["depth4_solved"] != 4 or per_arm["fixed"]["depth5_solved"] != 2:
            raise ValueError("frozen first-legal controls fail parity")
        if per_arm["shortest"]["depth4_solved"] != 80 or per_arm["shortest"]["depth5_solved"] != 80:
            raise ValueError("frozen shortest upper bound fails parity")
        seeds[str(seed)] = {
            "arm_comparisons": per_arm,
            "original_af23_vi2_depth5": d5_neural,
            "original_af23_direct_depth4": d4_neural,
            "episodes_audited": check["episode_count"],
        }
    matched = {
        arm: sum(x["arm_comparisons"][arm]["matched_at_both_depths"]
                 for x in seeds.values())
        for arm in PRIMARY
    }
    if failures:
        verdict = "INCONCLUSIVE"
    elif any(n >= REQUIRED_SEEDS for n in matched.values()):
        verdict = "SYMBOLIC_CONTROL_MATCH"
    else:
        verdict = "NO_SYMBOLIC_CONTROL_MATCH"
    return {
        "experiment": "AF-24", "track": "graph_aware_controls",
        "registered_verdict": verdict,
        "original_af23_scientific_verdict": "PASS_6_of_8_vs_original_fixed_random",
        "neural_search_advantage_over_public_structure": (
            "UNSUPPORTED" if verdict == "SYMBOLIC_CONTROL_MATCH"
            else "NOT_DETERMINED_BY_THESE_CONTROLS"
        ),
        "min_matches_required": REQUIRED_SEEDS,
        "allowed_shortfall_per_depth": MATCH_ALLOWANCE,
        "source_protocol_sha256": sha((ex.HERE/"PROTOCOL.json").read_bytes()),
        "source_af23_aggregate_sha256": EXPECTED_ORIGINAL,
        "source_af23_ci_run": CI_RUN,
        "missing_seeds": failures, "per_arm_paired_matches": matched,
        "seed_results": seeds, "pooled": pooled,
        "scope_note": "Public 15-atom/14-rule synthetic Horn grammar only; no claim of equal-latency neural vs symbolic inference.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    a = parser.parse_args()
    value = aggregate(a.root, a.original)
    a.out.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({
        "registered_verdict": value["registered_verdict"],
        "per_arm_paired_matches": value["per_arm_paired_matches"],
        "missing_seeds": value["missing_seeds"],
    }))


if __name__ == "__main__":
    main()
