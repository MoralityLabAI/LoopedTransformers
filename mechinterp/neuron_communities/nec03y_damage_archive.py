"""NEC-03Y: archived NEC-03X physical-null functional damage audit.

Pure Python standard library. No new model inference, source ranking, sham
search, model/SAE weights or new labels. Replays 8 immutable root archives
(16 native teacher arms, 4096 previously scored source-training cases)
and independently recomputes baseline-to-intervention correctness flips.

All quantitative descriptions are retrospective in-sample diagnostics.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha1, sha256
import csv
import json
import math
from pathlib import Path

SCHEMA = "NEC-03Y/read-only-source-physical-sham-functional-damage/v1"
LOCK_NAME = "NEC03Y_ARCHIVE_LOCK.json"
PANEL = tuple(range(40, 48))
ROOTS = tuple(range(8))
QUALIFIED = (0, 1, 2, 4, 6, 7)
ZERO = (3, 5)
ARMS = ("composed", "additive")
MODES = (
    "clean", "hubs", "native_nonhub", "full_affine", "contrib16_only",
    "continuous4_source", "continuous4_sham",
    "discrete4_source", "discrete4_sham",
)
INTERVENTIONS = tuple(m for m in MODES if m.endswith(("_source", "_sham")))
EPS = 1e-7
QUALITY_FIELDS = (
    "angle_gap", "cohesion_gap", "cross_group_absolute_gram_overlap",
    "physical_source_control_output_cosine", "joint_score",
    "source_physical_output_norm", "control_physical_output_norm",
)
DOSE_FIELDS = (
    "raw_source_output_L2", "raw_sham_output_L2", "shared_output_L2",
    "source_gain", "sham_gain", "dose_match_within_1pct", "positive_matched_dose",
)
COSINE_BINS = ("below_025", "025_to_050", "050_to_075", "075_and_up")
SUPPORT_BINS = ("zero", "one_to_49", "50_to_199", "200_to_495")
ANGLE_BINS = ("below_005", "005_to_015", "015_to_025", "above_025")
DOSE_BINS = ("zero", "up_to_01", "01_to_1", "one_and_up")


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def source_case_bytes(rows):
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in rows
    )


def check_numeric(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AssertionError("bad archived numeric "+field)
    return float(value)


def verify_lock():
    path = Path(__file__).with_name(LOCK_NAME)
    cfg = read(path)
    exact = {
        "schema": "NEC-03Y/frozen-retrospective-damage-and-quality/v1",
        "original_nec03x_run": 44975316374,
        "original_root_indices": list(ROOTS),
        "old_source_training_groups": list(PANEL),
        "source_cases_per_native_teacher": 256,
        "original_paired_source_qualified_roots": list(QUALIFIED),
        "original_paired_source_zero_ITT_roots": list(ZERO),
        "paired_native_teacher_arms": list(ARMS),
        "original_binary_intervention_modes": list(MODES),
        "original_primary_ITT_effect": -5 / 2048,
        "old_exploratory_signflip_p": 1.0,
        "same_old_source_sham_family_only": True,
        "correctness_tie_policy": "logit0 >= logit1 predicts class0",
        "source_damage_basis": "full_affine saved correctness -> intervention saved correctness",
        "null_inertness_denominator": "full_affine-correct AND original full-trajectory matched-control-valid",
        "cosine_bins": list(COSINE_BINS),
        "qualified_candidate_bins": list(SUPPORT_BINS),
        "angular_gap_bins": list(ANGLE_BINS),
        "physical_shared_dose_bins": list(DOSE_BINS),
        "posthoc_tests_computed": False,
        "new_native_or_SAE_model_forwards": 0,
        "new_native_or_SAE_optimizer_updates": 0,
        "new_sham_candidates_optimized": 0,
        "new_development_eval_groups_scored": 0,
        "new_reserved_groups_scored": 0,
        "new_recruited_worlds_scored": 0,
        "new_five_input_embargoed_worlds_scored": 0,
        "new_four_input_confirmation_worlds_scored": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
    }
    if any(cfg.get(k) != v for k, v in exact.items()):
        raise AssertionError("archival NEC03Y method or original cohort changed")
    if set(cfg["original_artifact_ids"]) != set(map(str, ROOTS)) | {"aggregate"}:
        raise AssertionError("one of nine parent source artifacts missing")
    if len(cfg["original_aggregate_sha256"]) != 64:
        raise AssertionError("original NEC03X eight-world aggregate unpinned")
    if set(cfg["original_root_pins"]) != set(map(str, ROOTS)):
        raise AssertionError("original NEC03X eight-world root digest list altered")
    for i in ROOTS:
        pins = cfg["original_root_pins"][str(i)]
        if set(pins) != {
            "world_sha256", "original_lock_sha256",
            "report_sha256", "independent_audit_sha256",
            "composed_jsonl_sha256", "additive_jsonl_sha256"
        } or any(len(v) != 64 for v in pins.values()):
            raise AssertionError("source root/hash manifest malformed")
    repository = Path(__file__).resolve().parents[2]
    for name, pin in cfg["git_blob_sha1"].items():
        raw = (repository / name).read_bytes()
        digest = sha1(b"blob " + str(len(raw)).encode() + b"\x00" + raw).hexdigest()
        if digest != pin:
            raise AssertionError("NEC03Y archived damage method changed "+name)
    return cfg, file_sha(path)


def classify_correct(pair, truth):
    if (not isinstance(pair, list) or len(pair) != 2
        or type(truth) != int or truth not in (0, 1)):
        raise AssertionError("binary logits or original Boolean truth label invalid")
    a = check_numeric(pair[0], "binary logit0")
    b = check_numeric(pair[1], "binary logit1")
    return int(b > a) == truth, int(b > a)


def transition(base, intervention):
    if base and intervention:
        return "remained_correct"
    if base and not intervention:
        return "correct_to_wrong"
    if not base and intervention:
        return "wrong_to_correct"
    return "remained_wrong"


def blank_transition():
    return Counter({
        "correct_to_wrong": 0, "wrong_to_correct": 0,
        "remained_correct": 0, "remained_wrong": 0
    })


def transitions_summary(rows, mode, only_valid=False):
    tally = blank_transition()
    baseline_valid = 0
    for row in rows:
        valid = row["original_control_available"][(
            "continuous" if mode.startswith("continuous") else "discrete"
        )]
        if only_valid and not valid:
            continue
        baseline = row["correct_by_mode"]["full_affine"]
        alternative = row["correct_by_mode"][mode]
        tally[transition(baseline, alternative)] += 1
        baseline_valid += int(baseline)
    return {
        "case_count": sum(tally.values()),
        "full_affine_correct_baseline": baseline_valid,
        **{key: tally[key] for key in (
            "correct_to_wrong", "wrong_to_correct",
            "remained_correct", "remained_wrong"
        )},
        "sham_or_source_correct": tally["remained_correct"] + tally["wrong_to_correct"],
        "net_correctness_change": tally["wrong_to_correct"] - tally["correct_to_wrong"],
        "lost_correct_rate_given_correct_baseline": (
            tally["correct_to_wrong"] / baseline_valid if baseline_valid else None
        ),
    }


def cosine_bin(v):
    v = check_numeric(v, "original source/sham output cosine")
    if v < .25: return "below_025"
    if v < .5: return "025_to_050"
    if v < .75: return "050_to_075"
    return "075_and_up"


def support_bin(n):
    if type(n) != int or not 0 <= n <= 495:
        raise AssertionError("number of feasible original 495 shams malformed")
    if n == 0: return "zero"
    if n < 50: return "one_to_49"
    if n < 200: return "50_to_199"
    return "200_to_495"


def angle_bin(v):
    v = check_numeric(v, "angle gap")
    if v < .05: return "below_005"
    if v < .15: return "005_to_015"
    if v <= .25: return "015_to_025"
    return "above_025"


def dose_bin(v):
    v = check_numeric(v, "original 32D physical dose")
    if v <= 1e-6: return "zero"
    if v < .1: return "up_to_01"
    if v < 1.: return "01_to_1"
    return "one_and_up"


def validate_original_parent(parent):
    if (
        parent["schema"] != "NEC-03X/eight-original-source-world-continuous-physical-null-ITT/v1"
        or parent["original_eight_root_ITT_count"] != 8
        or parent["independently_audited_source_root_count"] != 8
        or not parent["all_eight_original_neural_source_audits_passed"]
        or parent["original_eligible_paired_source_roots"] != list(QUALIFIED)
        or parent["historical_source_ineligible_ITT_zeros"] != list(ZERO)
        or parent["original_source_training_groups"] != list(PANEL)
        or parent["original_source_cases_per_native_teacher_arm"] != 256
        or parent["one_primary_macro_correctness_effect"] != -5 / 2048
        or parent["exploratory_root_exact_two_sided_signflip_p"] != 1
        or parent["complete_source_only_pilot_resolution_gate"] is not False
        or parent["source_only_is_population_confirmation"] is not False
        or parent["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
        or len(parent["roots"]) != 8
        or any(parent[key] != 0 for key in (
            "new_development_eval_groups_scored",
            "reserved_groups_scored", "new_recruited_worlds_scored",
            "five_input_embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"
        ))
    ):
        raise AssertionError("old eight-world independent source study changed")


def validate_source_receipt_shape(item, dose, mask, *, positive_topk):
    if positive_topk:
        source = item["source_ids"]
        target = item["continuous"]
        disc = item["discrete"]
        if (len(source) != 4 or len(set(source)) != 4 or
            target is None or disc is None or
            target["candidate_count"] != 495 or
            target["solver_iterations"] != 32 or
            target["solver_restarts"] != 1 or
            target["solver_max_relative_coefficient_multiplier"] != 3 or
            target["source_ids"] != source):
            raise AssertionError("frozen source quartet or continuous solver changed")
        sham = target["selected_sham_ids"]
        if (len(sham) != 4 or len(set(sham)) != 4
            or bool(set(source) & set(sham))
            or not set(source + sham) <= set(item["active_ids"])
            or disc["original_candidates"] != 495
            or set(disc["sham_ids"]) & set(source)):
            raise AssertionError("archived source/sham quartet not disjoint")
        if any(type(x) != int or not 0 <= x < 128 for x in source + sham):
            raise AssertionError("invalid original source or sham active code ID")
        raw = target["original_coefficients"]
        upper = target["coefficient_upper_bounds"]
        fitted = target["optimized_coefficients"]
        if not (len(raw) == len(upper) == len(fitted) == 4):
            raise AssertionError("sham optimizer coefficient dimension changed")
        for z, u, x in zip(raw, upper, fitted):
            z, u, x = (check_numeric(v, "continuous coefficient") for v in (z, u, x))
            if (z <= 0 or abs(u - 3*z) > 1e-4 * max(1., u)
                or x < -1e-6 or x > u + 1e-5 * max(1., u)):
                raise AssertionError("coefficient violates original nonnegative box")
        if (target["selected_candidate_quality_valid"] is not True
            and target["selected_candidate_quality_valid"] is not False):
            raise AssertionError("invalid original continuous physical quality boolean")
        if not 0 <= target["qualified_candidate_count"] <= 495:
            raise AssertionError("qualified original sham support count outside 495")
        if not 0 <= target["selected_candidate_index"] < 495:
            raise AssertionError("original selected support index outside 495")
        for key in QUALITY_FIELDS:
            check_numeric(target[key], "saved "+key)
        if abs(target["solver_optimal_objective"] - target["joint_score"]) > 1e-5:
            raise AssertionError("original source sham optimization score changed")
        if abs(
            target["joint_score"] - (
                target["angle_gap"] + .25 * target["cohesion_gap"] +
                .25 * (1 - target["physical_source_control_output_cosine"])
            )
        ) > 1e-4:
            raise AssertionError("source sham physical similarity objective changed")
        if target["selected_candidate_quality_valid"]:
            if not (
                target["angle_gap"] <= .250001 and
                target["cohesion_gap"] <= .750001 and
                target["cross_group_absolute_gram_overlap"] >= .119999 and
                target["physical_source_control_output_cosine"] >= .249999 and
                target["source_physical_output_norm"] > 1e-6 and
                target["control_physical_output_norm"] > 1e-6
            ):
                raise AssertionError("unqualified physical sham declared high quality")
    else:
        if item["source_ids"] or item["continuous"] is not None or item["discrete"] is not None:
            raise AssertionError("unsupported source input manufactured a sham")
    for k in DOSE_FIELDS:
        if len(dose[k]) != 32:
            raise AssertionError("complete physical dose receipt absent "+k)
    n0 = check_numeric(dose["raw_source_output_L2"][mask], "source dose")
    n1 = check_numeric(dose["raw_sham_output_L2"][mask], "sham dose")
    shared = check_numeric(dose["shared_output_L2"][mask], "shared dose")
    g0 = check_numeric(dose["source_gain"][mask], "source attenuation")
    g1 = check_numeric(dose["sham_gain"][mask], "sham attenuation")
    if (min(n0, n1, shared, g0, g1) < 0 or
        max(g0, g1) > 1.000001 or
        abs(shared - min(n0, n1)) > max(1e-5, shared*1e-5)):
        raise AssertionError("original 32D output physical source/sham dose altered")
    if (dose["positive_matched_dose"][mask] is not (shared > 1e-6)
        or type(dose["dose_match_within_1pct"][mask]) is not bool):
        raise AssertionError("positive-dose or 1% matcher receipt changed")
    if shared > 1e-6 and (
        abs(g0*n0 - shared) > max(1e-4, shared * .0101) or
        abs(g1*n1 - shared) > max(1e-4, shared * .0101)
    ):
        raise AssertionError("registered 32D physical gains differ from shared output dose")


def case_geometry(record, mask, valid):
    sites = record["site_geometry"]
    continuous = []
    for visit in sorted(sites, key=int):
        site = sites[visit]
        if (len(site["case_geometry"]) != 32 or
            set(site) != {
                "case_geometry", "continuous_pair_dose", "discrete_pair_dose",
                "full_affine_32d_output_dose", "full_affine_32d_reconstruction_error",
                "graph_hubs", "native_selected_units"
            }):
            raise AssertionError("source native 12-visit physical site receipt incomplete")
        for key in ("continuous_pair_dose", "discrete_pair_dose"):
            validate_source_receipt_shape(
                site["case_geometry"][mask],
                site[key], mask,
                positive_topk=bool(site["case_geometry"][mask]["full_topk_support"])
            )
        item = site["case_geometry"][mask]
        active = item["active_ids"]
        if (type(item["full_topk_support"]) is not bool or
            item["full_topk_support"] != (len(active) == 16) or
            len(active) != len(set(active))):
            raise AssertionError("original TopK16 active feature support changed")
        dose = site["continuous_pair_dose"]
        info = item["continuous"]
        if info is not None:
            physical_qualified = bool(info["selected_candidate_quality_valid"])
            physical_dose = bool(
                dose["positive_matched_dose"][mask] and
                dose["dose_match_within_1pct"][mask]
            )
            continuous.append({
                "visit": int(visit),
                "matched_quality": physical_qualified,
                "matched_dose": physical_dose,
                "qualified_candidate_count": info["qualified_candidate_count"],
                "angle_gap": info["angle_gap"],
                "cohesion_gap": info["cohesion_gap"],
                "cross_gram": info["cross_group_absolute_gram_overlap"],
                "source_control_output_cosine":
                    info["physical_source_control_output_cosine"],
                "raw_source_output_L2": dose["raw_source_output_L2"][mask],
                "raw_sham_output_L2": dose["raw_sham_output_L2"][mask],
                "shared_output_L2": dose["shared_output_L2"][mask],
                "source_gain": dose["source_gain"][mask],
                "sham_gain": dose["sham_gain"][mask],
                "coeff_zero": sum(
                    abs(check_numeric(v, "coefficient")) < 1e-6
                    for v in info["optimized_coefficients"]
                ),
                "coeff_upper_saturated": sum(
                    abs(check_numeric(x, "fitted coefficient") - check_numeric(u, "upper bound"))
                    <= 1e-4 * max(1., u)
                    for x, u in zip(
                        info["optimized_coefficients"],
                        info["coefficient_upper_bounds"]
                    )
                ),
            })
        elif item["full_topk_support"]:
            raise AssertionError("positive original TopK16 source has missing sham geometry")
    actual = bool(continuous) and len(continuous) == len(sites) and all(
        s["matched_quality"] and s["matched_dose"] for s in continuous
    )
    if actual != valid:
        raise AssertionError("rederived original continuous sham all-visit availability differs")
    return {
        "selected_site_count": len(sites),
        "qualified_candidate_min": min(
            (v["qualified_candidate_count"] for v in continuous), default=None
        ),
        "worst_angle_gap": max((v["angle_gap"] for v in continuous), default=None),
        "worst_cohesion_gap": max((v["cohesion_gap"] for v in continuous), default=None),
        "worst_cross_gram": min((v["cross_gram"] for v in continuous), default=None),
        "worst_source_control_cosine": min(
            (v["source_control_output_cosine"] for v in continuous), default=None
        ),
        "smallest_shared_output_L2": min(
            (v["shared_output_L2"] for v in continuous), default=None
        ),
        "smallest_sham_gain": min(
            (v["sham_gain"] for v in continuous), default=None
        ),
        "coefficient_zero_count": sum(v["coeff_zero"] for v in continuous),
        "coefficient_upper_saturation_count": sum(
            v["coeff_upper_saturated"] for v in continuous
        ),
        "site_records": continuous,
    }


def validate_root(root_index, directory, old_root, pins):
    report_file = directory / "nec03x_source_root_report.json"
    audit_file = directory / "independent_nec03x_continuous_source_audit.json"
    if (file_sha(report_file) != pins["report_sha256"] or
        file_sha(audit_file) != pins["independent_audit_sha256"]):
        raise AssertionError("original NEC03X root report or neural audit SHA changed")
    report, audit = read(report_file), read(audit_file)
    if (
        report["schema"] != "NEC-03X/source-only-continuous-disjoint-physical-null/v1"
        or audit["schema"] != "NEC-03X/independent-box-solver-and-continuous-physical-sham-replay/v1"
        or report["world_index"] != root_index or audit["world_index"] != root_index
        or report["world_id"] != f"dev5-{root_index:03d}"
        or old_root["world_index"] != root_index
        or report["world_sha256"] != pins["world_sha256"]
        or audit["world_sha256"] != pins["world_sha256"]
        or old_root["world_sha256"] != pins["world_sha256"]
        or report["source_lock_sha256"] != pins["original_lock_sha256"]
        or audit["source_lock_sha256"] != pins["original_lock_sha256"]
        or old_root["source_root_report_sha256"] != file_sha(report_file)
        or old_root["independent_source_audit_sha256"] != file_sha(audit_file)
        or audit["original_root_report_sha256"] != file_sha(report_file)
        or old_root["independent_audit_complete"] is not True
        or report["source_training_groups_scored"] != list(PANEL)
        or report["source_cases_per_model_arm"] != 256
        or audit["source_cases_per_arm"] != 256
        or report["original_paired_source_eligible"] != (root_index in QUALIFIED)
        or report["original_ITT_zero"] != (root_index in ZERO)
        or report["model_or_SAE_optimizer_updates"] != 0
        or set(report["arms"]) != set(ARMS)
        or set(audit["arms"]) != set(ARMS)
        or any(report[k] != 0 or audit[k] != 0 for k in (
            "new_development_eval_groups_scored",
            "reserved_groups_scored", "new_recruited_worlds_scored",
            "five_input_embargoed_worlds_scored",
            "old_four_input_confirmation_worlds_scored"
        ))
        or report["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
        or audit["FSI_01_gate"] != "REDESIGN_OR_PILOT_ONLY"
    ):
        raise AssertionError("original source population, report and independent model audit differ")
    if not all(audit[x] for x in (
        "frozen_original_native_and_SAE_checkpoint_ancestry_verified",
        "all_495_disjoint_continuous_supports_and_bounded_32_step_solver_replayed",
        "all_source_coherent_quartets_and_control_quality_gates_recomputed",
        "original_secondary_discrete_null_independently_replayed",
        "original_32d_output_L2_doses_and_nonamplifying_gains_verified",
        "all_original_256_training_source_labels_per_teacher_recomputed",
        "all_nine_12_visit_neural_recurrent_modes_independently_replayed"
    )):
        raise AssertionError("original independent NEC03X model/solver audit was incomplete")

    roots = {}
    case_rows = {}
    for arm in ARMS:
        old_arm = report["arms"][arm]
        audit_arm = audit["arms"][arm]
        parent_arm = old_root["arms"][arm]
        fname = f"{arm}.nec03x_continuous_source.jsonl"
        raw_file = directory / fname
        if (
            old_arm["raw_file"] != fname or
            old_arm["raw_sha256"] != pins[f"{arm}_jsonl_sha256"] or
            file_sha(raw_file) != old_arm["raw_sha256"] or
            audit_arm["raw_sha256"] != old_arm["raw_sha256"] or
            old_arm["correct_counts"] != audit_arm["correct_counts"] or
            old_arm["correct_counts"] != parent_arm["correct_counts"] or
            old_arm["matched_continuous_inputs"] != audit_arm["matched_continuous_inputs"] or
            old_arm["matched_continuous_inputs"] != parent_arm["matched_continuous_inputs"] or
            old_arm["matched_discrete_inputs"] != audit_arm["matched_discrete_inputs"] or
            old_arm["matched_discrete_inputs"] != parent_arm["matched_discrete_inputs"] or
            old_arm["source_model_sha256"] != audit_arm["source_model_sha256"] or
            old_arm["source_sae_sha256"] != audit_arm["SAE_checkpoint_sha256"] or
            old_arm["source_rank_sha256"] != audit_arm["source_rank_sha256"] or
            old_arm["source_group_indices"] != list(PANEL) or
            old_arm["source_cases"] != 256
        ):
            raise AssertionError("original root/arm raw source outcome SHA or audited counts differ")
        records = [json.loads(line) for line in raw_file.read_text().splitlines()]
        if len(records) != 8:
            raise AssertionError("original source window missing full group")
        truth_ids = set()
        counted = Counter()
        mask_count = Counter()
        cases = []
        for g, record in zip(PANEL, records):
            if (
                record["group"] != g or record["arm"] != arm
                or record["schema"] != "NEC-03X/source-only-continuous-disjoint-physical-null/v1"
                or record["focal_masks"] != list(range(32))
                or record["teacher_labels_used_for_sham_selection"] is not False
                or set(record["logits"]) != set(MODES)
                or set(record["original_predictions"]) != set(MODES)
                or len(record["source_input_histories"]) != 32
                or len(record["original_source_teacher_labels"]) != 32
                or len(record["continuous_available_all_visits"]) != 32
                or len(record["discrete_available_all_visits"]) != 32
            ):
                raise AssertionError("original source group, interventions or 32 masks changed")
            histories = record["source_input_histories"]
            if len(set(map(tuple, histories))) != 32:
                raise AssertionError("one original source group has duplicate masks")
            for mask in range(32):
                hist = histories[mask]
                if (
                    len(hist) != 12 or
                    any(type(bit) != int or bit not in (0, 1) for bit in hist)
                    or type(record["original_source_teacher_labels"][mask]) != int
                ):
                    raise AssertionError("source mask historical input or teacher label malformed")
                key = (g, mask)
                if key in truth_ids:
                    raise AssertionError("duplicate original source group/mask")
                truth_ids.add(key)
                y = record["original_source_teacher_labels"][mask]
                correctness = {}
                for mode in MODES:
                    if (
                        len(record["logits"][mode]) != 32 or
                        len(record["original_predictions"][mode]) != 32
                    ):
                        raise AssertionError("incomplete original native logits/predictions")
                    correct, predicted = classify_correct(record["logits"][mode][mask], y)
                    if predicted != record["original_predictions"][mode][mask]:
                        raise AssertionError("original full recurrent argmax differs from recorded result")
                    counted[mode] += int(correct)
                    correctness[mode] = correct
                cont_valid = record["continuous_available_all_visits"][mask]
                disc_valid = record["discrete_available_all_visits"][mask]
                if type(cont_valid) != bool or type(disc_valid) != bool:
                    raise AssertionError("old physical sham availability not Boolean")
                for pair, valid in (
                    ("continuous4", cont_valid),
                    ("discrete4", disc_valid)
                ):
                    if not valid:
                        s = record["logits"][f"{pair}_source"][mask]
                        h = record["logits"][f"{pair}_sham"][mask]
                        baseline = record["logits"]["full_affine"][mask]
                        if max(abs(s[i] - h[i]) for i in range(2)) > 1e-5:
                            raise AssertionError("invalid source/sham path gave nonzero contrast")
                        if max(abs(s[i] - baseline[i]) for i in range(2)) > 1e-5:
                            raise AssertionError("invalid source/sham diverged from full affine background")
                mask_count["continuous"] += int(cont_valid)
                mask_count["discrete"] += int(disc_valid)
                geom = case_geometry(record, mask, cont_valid)
                cases.append({
                    "world_index": root_index,
                    "world_id": report["world_id"],
                    "arm": arm, "source_group": g, "focal_mask": mask,
                    "source_input_history": hist,
                    "original_teacher_truth": y,
                    "original_ITT_eligible_world": root_index in QUALIFIED,
                    "original_ITT_zero_world": root_index in ZERO,
                    "original_control_available": {
                        "continuous": cont_valid, "discrete": disc_valid
                    },
                    "correct_by_mode": correctness,
                    "transition_vs_full_affine": {
                        m: transition(correctness["full_affine"], correctness[m])
                        for m in INTERVENTIONS
                    },
                    "physical_null_geometry": geom,
                })
        if len(truth_ids) != 256:
            raise AssertionError("old source group/mask coverage incomplete")
        if (
            dict(counted) != old_arm["correct_counts"]
            or mask_count["continuous"] != old_arm["matched_continuous_inputs"]
            or mask_count["discrete"] != old_arm["matched_discrete_inputs"]
        ):
            raise AssertionError("recomputed original 9-mode model outcomes differ from independent audit")
        report_modes = {
            mode: {
                "all_256_source_cases": transitions_summary(cases, mode, False),
                "original_valid_matched_inputs_only": transitions_summary(cases, mode, True),
            } for mode in INTERVENTIONS
        }
        inertness = report_modes["continuous4_sham"]["original_valid_matched_inputs_only"]
        roots[arm] = {
            "archived_correct_counts": dict(counted),
            "source_cases": 256,
            "original_continuous_valid_cases": mask_count["continuous"],
            "original_discrete_valid_cases": mask_count["discrete"],
            "old_root_raw_sha256": file_sha(raw_file),
            "damage_by_mode": report_modes,
            "sham_functional_inertness_valid": {
                "baseline_correct_matched_cases": inertness["full_affine_correct_baseline"],
                "baseline_correct_to_wrong": inertness["correct_to_wrong"],
                "baseline_wrong_to_correct": inertness["wrong_to_correct"],
                "rate_given_baseline_correct":
                    inertness["lost_correct_rate_given_correct_baseline"],
            },
        }
        case_rows[arm] = cases
    if [c["source_input_history"] for c in case_rows["composed"]] != [
        c["source_input_history"] for c in case_rows["additive"]
    ]:
        raise AssertionError("old paired teacher histories diverged within original root")
    root_effect = (
        roots["composed"]["archived_correct_counts"]["continuous4_sham"] -
        roots["composed"]["archived_correct_counts"]["continuous4_source"] -
        roots["additive"]["archived_correct_counts"]["continuous4_sham"] +
        roots["additive"]["archived_correct_counts"]["continuous4_source"]
    ) / 256
    if (
        abs(root_effect - report["raw_continuous_primary_root_DID"]) > 1e-12
        or abs(
            (root_effect if root_index in QUALIFIED else 0) -
            old_root["root_primary_ITT_DID"]
        ) > 1e-12
    ):
        raise AssertionError("historical original eight-world source primary changed")
    return {
        "world_index": root_index,
        "world_id": report["world_id"],
        "world_sha256": report["world_sha256"],
        "original_root_report_sha256": file_sha(report_file),
        "original_independent_neural_audit_sha256": file_sha(audit_file),
        "historical_primary_root_effect": (
            root_effect if root_index in QUALIFIED else 0
        ),
        "historical_source_ineligible_ITT_zero": root_index in ZERO,
        "arms": roots,
    }, case_rows["composed"] + case_rows["additive"]


def fixed_quality_strata(rows):
    """Only originally valid shams with already-observed task outcomes.

    The bins are fixed in the source-lock. Every stratum retains root/arm
    identity and baseline-correct denominator, avoiding pooled pseudo-replication.
    """
    result = {}
    for axis, finder, labeler in (
        ("output_cosine", "worst_source_control_cosine", cosine_bin),
        ("feasible_supports", "qualified_candidate_min", support_bin),
        ("angle_gap", "worst_angle_gap", angle_bin),
        ("shared_output_dose", "smallest_shared_output_L2", dose_bin),
    ):
        strata = {}
        for row in rows:
            if not row["original_control_available"]["continuous"]:
                continue
            val = row["physical_null_geometry"][finder]
            if val is None:
                raise AssertionError("valid physical null missing source quality metric")
            key = labeler(val)
            bucket = strata.setdefault(key, {
                "cases": 0, "full_affine_correct": 0,
                "sham_correct_to_wrong": 0, "source_correct_to_wrong": 0,
                "sham_wrong_to_correct": 0, "source_wrong_to_correct": 0,
            })
            bucket["cases"] += 1
            bucket["full_affine_correct"] += int(row["correct_by_mode"]["full_affine"])
            for prefix in ("sham", "source"):
                mode = f"continuous4_{prefix}"
                t = row["transition_vs_full_affine"][mode]
                bucket[f"{prefix}_correct_to_wrong"] += int(t == "correct_to_wrong")
                bucket[f"{prefix}_wrong_to_correct"] += int(t == "wrong_to_correct")
        result[axis] = strata
    return result


def audit(root: Path, old_aggregate: Path, out: Path):
    cfg, lock_sha = verify_lock()
    if file_sha(old_aggregate) != cfg["original_aggregate_sha256"]:
        raise AssertionError("old independently audited eight-root aggregate SHA changed")
    parent = read(old_aggregate)
    validate_original_parent(parent)
    expected = {f"nec03x-source-root-{i}" for i in ROOTS}
    found = {path.name for path in root.iterdir() if path.is_dir()}
    if found != expected:
        raise AssertionError("unexpected/missing root, or held-out source folder entered")
    summaries, cases = [], []
    for i in ROOTS:
        row, c = validate_root(
            i, root / f"nec03x-source-root-{i}",
            parent["roots"][i], cfg["original_root_pins"][str(i)]
        )
        summaries.append(row)
        cases.extend(c)
    if len(summaries) != 8 or len(cases) != 4096:
        raise AssertionError("8 roots×2 native teacher arms×256 source cases not complete")
    original_counts = {
        arm: {
            mode: sum(row["arms"][arm]["archived_correct_counts"][mode]
                      for row in summaries)
            for mode in MODES
        }
        for arm in ARMS
    }
    if original_counts != parent["source_correct_counts_by_native_teacher"]:
        raise AssertionError("recomputed old source model correct counts differ from aggregate")
    old_primary = [
        row["historical_primary_root_effect"] for row in summaries
    ]
    if (
        any(abs(a-b) > 1e-12 for a,b in zip(
            old_primary, parent["root_primary_continuous_disjoint_sham_effects"]
        ))
        or abs(sum(old_primary)/8 + 5/2048) > 1e-12
    ):
        raise AssertionError("registered root-level NEC03X source correctness ITT changed")

    # Descriptive counts at root, arm and eligible-world cluster levels.
    pooled = {}
    for qualified_only in (False, True):
        key = "six_original_eligible_roots" if qualified_only else "all_eight_original_roots"
        pooled[key] = {}
        chosen = [s for s in summaries
                  if not qualified_only or s["world_index"] in QUALIFIED]
        for arm in ARMS:
            totals = {
                "original_world_roots": len(chosen),
                "source_cases": len(chosen)*256,
                "original_continuous_valid_cases": sum(
                    s["arms"][arm]["original_continuous_valid_cases"] for s in chosen
                ),
                "original_discrete_valid_cases": sum(
                    s["arms"][arm]["original_discrete_valid_cases"] for s in chosen
                ),
                "baseline_correct_cases": sum(
                    s["arms"][arm]["archived_correct_counts"]["full_affine"] for s in chosen
                ),
                "transitions": {},
            }
            for mode in INTERVENTIONS:
                mode_tally = {k: 0 for k in (
                    "case_count", "full_affine_correct_baseline",
                    "correct_to_wrong", "wrong_to_correct",
                    "remained_correct", "remained_wrong",
                    "sham_or_source_correct", "net_correctness_change"
                )}
                for s in chosen:
                    m = s["arms"][arm]["damage_by_mode"][mode]["original_valid_matched_inputs_only"]
                    for k in mode_tally:
                        mode_tally[k] += m[k]
                mode_tally["lost_correct_rate_given_correct_baseline"] = (
                    mode_tally["correct_to_wrong"] / mode_tally["full_affine_correct_baseline"]
                    if mode_tally["full_affine_correct_baseline"] else None
                )
                totals["transitions"][mode] = mode_tally
            pooled[key][arm] = totals

    by_world_arm = {}
    for world in ROOTS:
        by_world_arm[str(world)] = {}
        for arm in ARMS:
            rows = [c for c in cases if c["world_index"] == world and c["arm"] == arm]
            by_world_arm[str(world)][arm] = fixed_quality_strata(rows)

    raw = source_case_bytes(cases)
    out.mkdir(parents=True, exist_ok=True)
    (out/"nec03y_archived_casewise_functional_damage.jsonl").write_bytes(raw)
    result = {
        "schema": SCHEMA,
        "original_nec03x_actions_run": 44975316374,
        "frozen_nec03y_lock_sha256": lock_sha,
        "old_aggregate_sha256": file_sha(old_aggregate),
        "old_independent_root_neural_audits_verified": True,
        "all_original_root_report_audit_and_raw_sha256_verified": True,
        "all_original_nine_mode_source_predictions_recomputed": True,
        "original_eight_world_roots": list(ROOTS),
        "historical_source_ineligible_ITT_zeros": list(ZERO),
        "original_source_training_groups": list(PANEL),
        "original_native_teacher_arms": list(ARMS),
        "original_case_arm_count": 4096,
        "historical_nec03x_root_primary_effects": old_primary,
        "historical_nec03x_primary_correctness_effect": -5/2048,
        "historical_exploratory_signflip_p": 1,
        "historical_fsi_01_source_only_resolution_passed": False,
        "new_inferential_tests": 0,
        "new_model_forwards": 0,
        "new_source_or_sham_interventions": 0,
        "new_sham_candidates_selected": 0,
        "new_native_or_sae_training_updates": 0,
        "new_development_eval_groups_scored": 0,
        "new_reserved_context_groups_scored": 0,
        "new_recruited_worlds_scored": 0,
        "new_five_input_embargoed_worlds_scored": 0,
        "new_four_input_confirmation_worlds_scored": 0,
        "FSI_01_gate": "REDESIGN_OR_PILOT_ONLY",
        "original_correct_counts": original_counts,
        "pooled_original_source_functional_damage": pooled,
        "original_root_functional_damage": summaries,
        "within_original_root_arm_fixed_quality_strata": by_world_arm,
        "retrospective_note": (
            "All source cases previously trained, repeated masks and sites correlated; "
            "physical sham admissibility and functional damage never constitute a new "
            "calibrated causal null or population inference."
        ),
    }
    result["archived_case_receipt_sha256"] = sha256(raw).hexdigest()
    (out/"nec03y_readonly_functional_damage_aggregate.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False)+"\n"
    )
    with (out/"nec03y_root_damage_summary.csv").open("w", newline="") as handle:
        writer=csv.writer(handle)
        writer.writerow([
            "world_index", "arm", "source_cases", "matched_continuous_cases",
            "full_affine_correct", "valid_baseline_correct",
            "continuous_sham_correct_to_wrong", "continuous_sham_wrong_to_correct",
            "continuous_source_correct_to_wrong", "continuous_source_wrong_to_correct",
            "sham_damage_rate_given_valid_correct_baseline", "original_root_ITT_effect"
        ])
        for row in summaries:
            for arm in ARMS:
                a = row["arms"][arm]
                sham = a["damage_by_mode"]["continuous4_sham"]["original_valid_matched_inputs_only"]
                src = a["damage_by_mode"]["continuous4_source"]["original_valid_matched_inputs_only"]
                writer.writerow([
                    row["world_index"], arm, 256,
                    a["original_continuous_valid_cases"], a["archived_correct_counts"]["full_affine"],
                    sham["full_affine_correct_baseline"], sham["correct_to_wrong"],
                    sham["wrong_to_correct"], src["correct_to_wrong"], src["wrong_to_correct"],
                    sham["lost_correct_rate_given_correct_baseline"],
                    row["historical_primary_root_effect"]
                ])
    print(
        "NEC03Y_ARCHIVED_FUNCTIONAL_DAMAGE_AUDITED_PASS",
        "root_audits", len(summaries), "case_arms", len(cases),
        "historical_ITT", -5/2048,
        "additive_qualified_sham_flips",
        pooled["six_original_eligible_roots"]["additive"]["transitions"]["continuous4_sham"]["correct_to_wrong"]
    )
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--original-aggregate",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    audit(args.root,args.original_aggregate,args.out)


if __name__=="__main__":
    main()
