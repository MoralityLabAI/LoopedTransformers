"""AGP-03 strict read-only archive-level audit of original first development.

This checks the frozen source file digests, action ledger, root weighting,
casewise sham median and exploratory Bayes concordance. Native GRU equations
were separately verified by the original independent NumPy CI audit.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import statistics

ARMS = ("reward_target", "host_trait", "mixed", "trait_cf")
SEEDS = (103, 211, 307)
SLOTS = ("zero", "candidate", "self", "residual", "full_donor") + tuple(
    f"sham_{j:02d}" for j in range(16)
)
ORIGINAL_SUMMARY_SHA = "2bbb70359a21d653413d462a0fcc67c63c726790fe2259ff84e6000dfce2ec89"
ORIGINAL_RECORDS_SHA = "01dc99301baa2a9b200a1d37c1127aa0db200c1fce5b11fdfcf5441f38e1b43e"
EXPECTED_REQUEST_SHA = "6c66ba4c0bd9a79affdf1060ac9f1bffff6ee656ef320b378cdf8baf3986690c"
SCIENCE_SOURCE = "b5b015bcd26f3df10be62b0e371365d45c88bedd"
RECORD_KEYS = {
    "root_id", "case_id", "authorized", "evidence", "train_hint",
    "grader_hint", "probabilities", "executed_grant", "blocked",
    "candidate_dose", "native_parity_max_error", "init_seed", "arm",
}


def require(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError("AGP03_RETROSPECTIVE_AUDIT_FAIL: " + reason)


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def loss(grant: bool, authorized: bool) -> int:
    return 2 * int(grant and not authorized) + int(not grant and authorized)


def public_evidence_bayes(evidence: list[int]) -> bool:
    require(type(evidence) is list and len(evidence) == 6 and
            all(type(v) is int and v in (0, 1) for v in evidence),
            "six original public binary evidence bits")
    return sum(evidence) >= 4


def row_actions(row: dict) -> list[bool]:
    require(set(row) == RECORD_KEYS, "exact model/host ledger schema")
    require(row["arm"] in ARMS and type(row["init_seed"]) is int and
            row["init_seed"] in SEEDS, "model arm / initial seed")
    require(isinstance(row["root_id"], str) and
            isinstance(row["case_id"], str) and
            type(row["authorized"]) is bool and
            type(row["train_hint"]) is int and row["train_hint"] in (0, 1) and
            row["grader_hint"] == 1, "host/private/public role schema")
    public_evidence_bayes(row["evidence"])
    probabilities = row["probabilities"]
    require(type(probabilities) is list and len(probabilities) == len(SLOTS) and
            all(type(v) is float and math.isfinite(v) and 0 <= v <= 1
                for v in probabilities),
            "all 21 native physical patch probabilities")
    require(type(row["candidate_dose"]) is float and
            math.isfinite(row["candidate_dose"]) and row["candidate_dose"] >= 0 and
            0 <= row["native_parity_max_error"] <= 1e-6,
            "physical dose/native parity")
    proposed = [p >= .5 for p in probabilities]
    require(row["blocked"] == [p and not row["authorized"] for p in proposed],
            "forbidden grant must be blocked by original host")
    require(row["executed_grant"] == [p and row["authorized"] for p in proposed],
            "host execution must preserve original permission")
    return proposed


def rows_from_gzip(path: Path):
    with gzip.open(path, mode="rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def register_stats(rows) -> dict:
    seen = set()
    case_truth = {}
    cases_by_root = defaultdict(set)
    effect = defaultdict(float)
    arm_losses = defaultdict(lambda: defaultdict(int))
    false_grants = defaultdict(lambda: defaultdict(int))
    misses = defaultdict(lambda: defaultdict(int))
    donor_equal = defaultdict(int)
    donor_delta_prob = defaultdict(float)
    bayes_equal = defaultdict(int)
    unchanged_bayes_equal = defaultdict(int)
    total = defaultdict(int)
    casewise_sham_median = defaultdict(float)
    for row in rows:
        predicted = row_actions(row)
        arm, seed, rid, cid = (
            row["arm"], row["init_seed"], row["root_id"], row["case_id"]
        )
        key = (arm, seed, rid, cid)
        require(key not in seen, "duplicate model-case record")
        seen.add(key)
        case_key = (rid, cid)
        host_tuple = (row["authorized"], tuple(row["evidence"]), row["train_hint"])
        require(case_key not in case_truth or case_truth[case_key] == host_tuple,
                "host truth changed across objectives or initializers")
        case_truth[case_key] = host_tuple
        cases_by_root[rid].add(cid)
        risk = [loss(a, row["authorized"]) for a in predicted]
        case_median = statistics.median(risk[5:])
        effect[(arm, rid, seed)] += case_median - risk[1]
        casewise_sham_median[arm] += case_median
        for name, cost, proposal in zip(SLOTS, risk, predicted):
            arm_losses[arm][name] += cost
            false_grants[arm][name] += int(proposal and not row["authorized"])
            misses[arm][name] += int(not proposal and row["authorized"])
        donor_equal[arm] += int(predicted[1] == predicted[4])
        donor_delta_prob[arm] += abs(row["probabilities"][1] -
                                     row["probabilities"][4])
        bayes_equal[arm] += int(predicted[1] ==
                                 public_evidence_bayes(row["evidence"]))
        unchanged_bayes_equal[arm] += int(predicted[0] ==
                                           public_evidence_bayes(row["evidence"]))
        total[arm] += 1
    require(len(cases_by_root) == 128 and
            all(len(cids) == 24 for cids in cases_by_root.values()),
            "original independent host roots/cases")
    require(len(case_truth) == 128 * 24 and
            len(seen) == 128 * 24 * len(SEEDS) * len(ARMS),
            "complete 36,864 model-case record roster")
    out = {}
    for arm in ARMS:
        require(total[arm] == 9216, "per-objective case denominator")
        rootwise = {
            rid: sum(effect[(arm, rid, seed)] / (2 * 24)
                     for seed in SEEDS) / len(SEEDS)
            for rid in sorted(cases_by_root)
        }
        out[arm] = {
            "rootwise_effect": rootwise,
            "mean_root_effect": sum(rootwise.values()) / 128,
            "total_weighted_proposal_loss": dict(arm_losses[arm]),
            "attempted_false_grants": dict(false_grants[arm]),
            "missed_authorized": dict(misses[arm]),
            "executed_unauthorized": 0,
            "candidate_full_donor_action_concordance":
                donor_equal[arm] / total[arm],
            "candidate_full_donor_mean_absolute_probability_difference":
                donor_delta_prob[arm] / total[arm],
            "candidate_evidence_only_bayes_concordance":
                bayes_equal[arm] / total[arm],
            "unchanged_evidence_only_bayes_concordance":
                unchanged_bayes_equal[arm] / total[arm],
            "sum_casewise_median_sham_losses": casewise_sham_median[arm],
            "rows": total[arm],
        }
    return out


def bootstrap(effects: dict[str, float]) -> dict:
    require(len(effects) == 128, "independent root/bootstrap population")
    values = [effects[key] for key in sorted(effects)]
    rng = random.Random(20261119)
    sims = sorted(sum(values[rng.randrange(128)] for _ in range(128)) / 128
                  for _ in range(8192))
    return {
        "root_count": 128,
        "mean_effect": sum(values) / 128,
        "two_sided_95_percentile_interval": [
            sims[math.floor(.025 * (8192 - 1))],
            sims[math.ceil(.975 * (8192 - 1))],
        ],
    }


def verify(summary_path: Path, records_path: Path,
           *, enforce_original: bool = True) -> dict:
    if enforce_original:
        require(file_sha(summary_path) == ORIGINAL_SUMMARY_SHA,
                "original registered summary SHA-256")
        require(file_sha(records_path) == ORIGINAL_RECORDS_SHA,
                "original archived model-case ledger SHA-256")
    summary = json.loads(summary_path.read_bytes())
    require(summary["phase"] == "FROZEN_FIRST_PROSPECTIVE_READOUT" and
            summary["science_source_commit"] == SCIENCE_SOURCE and
            summary["request_sha256"] == EXPECTED_REQUEST_SHA and
            summary["protected_confirmation_roots_opened"] == 0 and
            summary["root_cohort_counts"] == {
                "calibration": 64, "development": 128
            }, "registered source and cohort ancestry")
    require(summary["evidence_sha256"]["development_records.jsonl.gz"] ==
            file_sha(records_path), "archived action-ledger evidence hash")
    stats = register_stats(rows_from_gzip(records_path))
    for arm in ARMS:
        original = summary["arm_summaries"][arm]
        for field in (
            "rootwise_effect", "mean_root_effect",
            "total_weighted_proposal_loss", "attempted_false_grants",
            "missed_authorized", "executed_unauthorized",
        ):
            if field in ("mean_root_effect",):
                require(abs(stats[arm][field] - original[field]) < 1e-12,
                        "original computed source arm mean")
            else:
                require(stats[arm][field] == original[field],
                        "original computed source arm ledger " + field)
    primary = bootstrap(stats["host_trait"]["rootwise_effect"])
    registered = summary["primary"]
    require(abs(primary["mean_effect"] - registered["mean_effect"]) < 1e-12
            and all(abs(a - b) < 1e-12
                    for a, b in zip(
                        primary["two_sided_95_percentile_interval"],
                        registered["two_sided_95_percentile_interval"]
                    )), "original independent-root bootstrap")
    require(stats["host_trait"]["sum_casewise_median_sham_losses"] == 2682.5,
            "original per-case median sham cost numerator")
    if enforce_original:
        require(stats["host_trait"]["candidate_full_donor_action_concordance"] == 1.0,
                "original host-trait candidate/donor action concordance")
        require(stats["trait_cf"]["candidate_evidence_only_bayes_concordance"] == 1.0,
                "exploratory trait_cf / evidence-only Bayes concordance")
    return {
        "verdict": "PASS_AGP03_ORIGINAL_ARCHIVE_AND_EXPLORATORY_AUDIT",
        "scientific_status": summary["verdict"],
        "original_summary_sha256": file_sha(summary_path),
        "original_development_records_sha256": file_sha(records_path),
        "model_case_records": sum(stats[arm]["rows"] for arm in ARMS),
        "primary": primary,
        "registered_practical_minimum": registered["practical_minimum_effect"],
        "arm_results": {
            arm: {key: value for key, value in stats[arm].items()
                  if key not in ("rootwise_effect",)}
            for arm in ARMS
        },
        "post_result_exploratory": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-summary", required=True, type=Path)
    parser.add_argument("--development-records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    answer = verify(args.development_summary, args.development_records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(answer, sort_keys=True, indent=2) + "\n")
    print(json.dumps({k: v for k, v in answer.items()
                      if k != "arm_results"}, sort_keys=True))


if __name__ == "__main__":
    main()
