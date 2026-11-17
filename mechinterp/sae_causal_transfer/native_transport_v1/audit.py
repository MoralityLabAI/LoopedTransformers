"""LS-SAE09 independent standard-library check of transport summary receipts.

Replays the original LS-SAE08 untouched source/geometry/81x9 logits and
independently checks prediction flips, boundary ratios, norm decompositions,
fixed classification gates and complete signed-pair coverage. This numerical
auditor cannot independently reconstruct high-dimensional hidden tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as A8
from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as F8

SCHEMA = "LS-SAE09/native-ACT3-6-12-hidden-transport-v1"
PROTOCOL = "94477c21425509c896112eda1dde1af802ce0659"
CHECKPOINT = A8.CHECKPOINT
SOURCE_NPZ = A8.SOURCE_NPZ
SOURCE_MANIFEST = A8.SOURCE_MANIFEST
FIT_ARRAY = A8.FIT_ARRAY
PHYSICAL = "14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb"
FULL_SHA = "69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
PILOT_SHA = "3b26bec178fdea4a2120001b72f3c035a91e728642adf635b9b42af985a05824"
ARMS = ("state_power", "logit_power", "sae", "random")
RADII = ("0.20", "2.00", "8.00")
HORIZONS = ("ACT3", "ACT6", "ACT12")
TOL = 2e-4  # Registered same-run centered logit/hidden identity.
ARCHIVE_TOL = 1e-3  # Dated A1 amendment; runner requires all81 argmax identity.


def close(x, y, atol=3e-5, rtol=1e-5):
    return math.isfinite(float(x)) and math.isfinite(float(y)) and (
        abs(float(x) - float(y)) <= atol + rtol * max(abs(float(x)), abs(float(y)))
    )


def close_square(a, b, c):
    # Norm identities, with FP32 rounding allowed by the signed continuation.
    return abs(float(a)**2 - float(b)**2 - float(c)**2) <= (
        2e-5 * (1 + float(a)**2)
    )


def original_decision_stats(raw, source):
    """Independent scalar implementation; compare original 81x9 saved logits."""
    clean = raw[0]
    changed = raw[1]
    blanks = [i for i, x in enumerate(source["x"]) if x == 0]
    target = source["target"]
    clean_top = [max(range(9), key=lambda k: clean[i][k]) for i in range(81)]
    edit_top = [max(range(9), key=lambda k: changed[i][k]) for i in range(81)]
    flips = sum(clean_top[i] != edit_top[i] for i in blanks)
    correct = sum((edit_top[i] + 1) == target[i] for i in blanks)
    pressure = []
    zero = 0
    margin_min = math.inf
    fro2 = 0.0
    for i in range(81):
        diff = [changed[i][j] - clean[i][j] for j in range(9)]
        dm = sum(diff)/9
        fro2 += sum((d-dm)**2 for d in diff)
        if i not in blanks:
            continue
        p = clean_top[i]
        for j in range(9):
            if j == p:
                continue
            gap = clean[i][p] - clean[i][j]
            margin_min = min(margin_min, gap)
            if gap <= 1e-12:
                zero += 1
                continue
            pressure.append((diff[j]-diff[p])/gap)
    return {
        "blank_flip_count": flips, "blank_correct": correct,
        "zero_clean_margin_pairs": zero,
        "boundary_pressure_max": max(pressure) if pressure else None,
        "minimum_clean_blank_margin": margin_min,
        "centered_logit_fro": math.sqrt(fro2)
    }


def check_case(reported, old, source, rank, smin, smax):
    if (reported["seed"] != source["seed"] or
        reported["visible_goal_position"] != F8.median_blank(source["x"]) or
        reported["original_source_zH_norm"] <= 0 or
        not close(reported["original_source_zH_norm"],
                  old["original_source_zH_token_norm"], atol=TOL) or
        reported["direction_status"] != old["source_direction_status"] or
        reported["visible_structural_difficulty"] != source["visible_structural_difficulty"] or
        set(reported["horizons"]) != set(HORIZONS)):
        raise ValueError("transport source identity / original ACT1 status altered")

    expected = [(arm, radius, sign)
                for arm in ARMS if old["source_direction_status"][arm]
                for radius in RADII for sign in ("plus", "minus")]
    for horizon in HORIZONS:
        rec = reported["horizons"][horizon]
        if not 0 <= rec["original_clean_logit_max_absolute_drift"] <= ARCHIVE_TOL:
            raise ValueError("clean original model replay drift exceeded registration")
        raw = old["native_81x9_logits"][horizon]
        blanks = [i for i, x in enumerate(source["x"]) if not x]
        actual_clean_correct = sum(
            1 + max(range(9), key=lambda j: raw["clean"][i][j]) == source["target"][i]
            for i in blanks
        )
        if rec["clean_blank_correct"] != actual_clean_correct:
            raise ValueError("original clean Sudoku prediction differs")
        got = [(v["arm"], v["radius"], v["sign"]) for v in rec["variants"]]
        if got != expected:
            raise ValueError("registered arms/radii/signs/order or null status altered")
        for v in rec["variants"]:
            for key in ("h_global_l2","h_gain","l_global_l2","l_gain",
                        "h_source_site_l2","h_non_site_l2","h_blank_l2",
                        "h_given_l2","h_board_l2","h_readout_l2","h_null_l2",
                        "readout_fraction","centered_logit_fro",
                        "centered_readout_reconstruction_error",
                        "original_logits_max_absolute_drift"):
                if not math.isfinite(v[key]) or v[key] < 0:
                    raise ValueError("nonfinite or negative transport numerical metric: "+key)
            if v["readout_fraction"] > 1 + 1e-5:
                raise ValueError("rowspace fraction exceeds unity")
            if (v["original_logits_max_absolute_drift"] > ARCHIVE_TOL or
                v["centered_readout_reconstruction_error"] > TOL):
                raise ValueError("original logits changed or readout linearity failed")
            dose = float(v["radius"])
            if not close(v["h_gain"], v["h_global_l2"]/dose) or not close(
                v["l_gain"], v["l_global_l2"]/dose):
                raise ValueError("original physical radius transport ratio changed")
            if not close_square(v["h_global_l2"],
                                v["h_source_site_l2"],v["h_non_site_l2"]):
                raise ValueError("edited-site vs full-state transport norm mismatch")
            if not close_square(v["h_board_l2"],
                                v["h_blank_l2"],v["h_given_l2"]):
                raise ValueError("blank/given transport norm mismatch")
            if not close_square(v["h_board_l2"],
                                v["h_readout_l2"],v["h_null_l2"]):
                raise ValueError("readout/nullspace orthogonal norm mismatch")
            frac=(v["h_readout_l2"]/v["h_board_l2"]
                  if v["h_board_l2"]>1e-12 else 0)
            if not close(frac,v["readout_fraction"]):
                raise ValueError("readout rowspace fraction miscomputed")

            raw_pair = (raw["clean"],raw[v["arm"]][v["radius"]][v["sign"]])
            original = original_decision_stats(raw_pair, source)
            if (v["blank_flip_count"] != original["blank_flip_count"] or
                v["blank_correct"] != original["blank_correct"] or
                v["zero_clean_margin_pairs"] != original["zero_clean_margin_pairs"]):
                raise ValueError("saved original LS-SAE08 digit flips/correctness differ")
            if not close(v["minimum_clean_blank_margin"],
                         original["minimum_clean_blank_margin"], atol=ARCHIVE_TOL):
                raise ValueError("saved original clean output margin changed")
            # Each logit in a 9-way row can move up to the replay tolerance.
            # Frobenius drift of centered difference <= 2*sqrt(81*9)*TOL.
            if not close(v["centered_logit_fro"],original["centered_logit_fro"],
                         atol=2*math.sqrt(81*9)*ARCHIVE_TOL):
                raise ValueError("centered finite logit displacement differs")
            if original["boundary_pressure_max"] is not None:
                if v["boundary_pressure_max"] is None:
                    raise ValueError("signed boundary pressure omitted")
                m=original["minimum_clean_blank_margin"]
                pressure_error=4*ARCHIVE_TOL/max(m,1e-12)
                if not close(v["boundary_pressure_max"],
                             original["boundary_pressure_max"],
                             atol=pressure_error):
                    raise ValueError("signed decision boundary pressure changed")

            # Rowspace output obeys sigma_min ||P_W dh|| <= ||W_c dh|| <= sigma_max ||P_W dh||.
            if v["centered_logit_fro"] < smin*v["h_readout_l2"]-2*TOL:
                raise ValueError("readout rowspace lower singular bound violated")
            if v["centered_logit_fro"] > smax*v["h_readout_l2"]+2*TOL:
                raise ValueError("readout rowspace upper singular bound violated")

            declared = {
              "contracted":v["h_gain"]<0.1,
              "persistent_output_silent":v["h_gain"]>=1.0 and v["blank_flip_count"]==0,
              "readout_small":v["h_board_l2"]>1e-8 and v["readout_fraction"]<0.1
            }
            if v["classification"] != declared:
                raise ValueError("posthoc mechanism classification changed")


def audit(record, baseline, baseline_sha):
    A8.audit(baseline)
    expected_mode = baseline["mode"]
    if (record["schema"] != SCHEMA or
        record["preregistration_commit"] != PROTOCOL or
        record["mode"] != expected_mode or
        baseline_sha != (PILOT_SHA if expected_mode == "pilot" else FULL_SHA) or
        record["source_ls_sae08_json_sha256"] != baseline_sha or
        record["source_checkpoint_sha256"] != CHECKPOINT or
        record["source_calibration_npz_sha256"] != SOURCE_NPZ or
        record["source_calibration_manifest_sha256"] != SOURCE_MANIFEST or
        record["source_fit_array_sha256"] != FIT_ARRAY or
        record["physical_dictionary_sha256"] != PHYSICAL or
        record["horizons"] != list(HORIZONS) or
        record["arms"] != list(ARMS) or record["radii"] != list(RADII) or
        record["max_original_logit_replay_drift"] != ARCHIVE_TOL or
        record["same_run_readout_identity_tolerance"] != TOL):
        raise ValueError("registered LS-SAE09 source/geometry/dose settings differ")
    rank=record["readout_centered_rank"]
    singular=record["readout_nonzero_singular_values"]
    if rank < 1 or rank > 8 or len(singular) != rank:
        raise ValueError("centered 9-digit readout rank invalid")
    if (not all(math.isfinite(v) and v>0 for v in singular) or
        any(singular[i+1] > singular[i] for i in range(rank-1))):
        raise ValueError("centered digit readout singular values invalid")
    groups=("pilot_52","pilot_54") if expected_mode=="pilot" else ("test_52","test_54")
    if set(record["families"]) != set(groups):
        raise ValueError("source cohort list changed")
    for group in groups:
        original=baseline["families"][group]
        roster=F8.roster(group)
        entries=record["families"][group]["per_puzzle"]
        if len(entries) != len(roster) or roster != original["source_roster"]:
            raise ValueError("original source roster or denominator changed")
        for source, old, current in zip(roster,original["per_puzzle"],entries):
            check_case(current,old,source,rank,singular[-1],singular[0])
    counts={g:len(record["families"][g]["per_puzzle"]) for g in groups}
    return {
      "schema":"LS-SAE09/independent-summary-and-original-logit-replay-v1",
      "original_ls_sae08_audit_replayed":True,
      "exact_original_preheldout_source_seal":True,
      "all_registered_signed_physical_radii_and_ACT3_6_12_checked":True,
      "original_output_digits_flips_and_margin_pressure_replayed":True,
      "transport_geometry_and_readout_orthogonal_norms_checked":True,
      "readout_singular_value_bounds_checked":True,
      "model_reexecuted_by_auditor":False,
      "limitation":"Original 81x9 logits independently re-scored; hidden-state summary norms are internally checked and require a second model execution to independently reconstruct.",
      "cases":counts,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    old_bytes=a.baseline.read_bytes()
    result=audit(json.loads(a.input.read_text()),json.loads(old_bytes),
                 hashlib.sha256(old_bytes).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
