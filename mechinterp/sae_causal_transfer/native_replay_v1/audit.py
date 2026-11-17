"""Independent stdlib scalar audit of LS-SAE10 and frozen LS-SAE08 81x9 outputs.

This module does not call the new neural runner or re-execute the checkpoint.
All original/new full-grid logits are saved, so drift and argmax can be replayed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from mechinterp.sae_causal_transfer.native_hard_task_v2 import audit as OLD

SCHEMA = "LS-SAE10/retrospective-original-numeric-replay-forensics-v1"
PROTOCOL = "2c9a5db2ca39f82f94649df66aea6907a80d47ea"
ORIGINAL_FULL = "69c4cd1a2e09760ed2d463e598550bdd1da42fb1bc0ad83a8a95825c47bea7f9"
CHECKPOINT = OLD.CHECKPOINT
SENTINELS = (("test_52",0),("test_54",3))
MODES = ("canonical_a","canonical_b","instrumented","single_source")
HORIZONS = ("ACT3","ACT6","ACT12")
RADII = ("0.20","2.00","8.00")
ARMS = ("state_power","logit_power","sae","random")


def approx(a,b,atol=1e-6,rtol=1e-8):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and (
        abs(float(a)-float(b)) <= atol+rtol*max(abs(float(a)),abs(float(b)))
    )


def rows_ok(rows):
    if not isinstance(rows,list) or len(rows)!=81:
        raise ValueError("native output needs all 81 digit rows")
    for row in rows:
        if not isinstance(row,list) or len(row)!=9 or not all(
            isinstance(x,(float,int)) and math.isfinite(x) for x in row
        ):
            raise ValueError("native output has invalid 9-digit logits")


def all_variants(raw,eligible):
    if set(raw) != set(HORIZONS):
        raise ValueError("native ACT3/ACT6/ACT12 coverage changed")
    for horizon in HORIZONS:
        block=raw[horizon]
        expected={"clean"}|{a for a in ARMS if eligible[a]}
        if set(block)!=expected:
            raise ValueError("native original arm/status coverage changed")
        rows_ok(block["clean"])
        yield horizon,"clean",block["clean"]
        for arm in ARMS:
            if not eligible[arm]:
                continue
            if set(block[arm])!=set(RADII):
                raise ValueError("original physical radius coverage changed")
            for radius in RADII:
                if set(block[arm][radius])!={"plus","minus"}:
                    raise ValueError("missing original physical dose sign")
                for sign in ("plus","minus"):
                    array=block[arm][radius][sign]
                    rows_ok(array)
                    yield horizon,f"{arm}/{radius}/{sign}",array


def lookup(raw,horizon,name):
    if name=="clean":
        return raw[horizon]["clean"]
    arm,radius,sign=name.split("/")
    return raw[horizon][arm][radius][sign]


def independent_stats(left,right,blanks):
    rows_ok(left);rows_ok(right)
    max_abs=0.
    centered_square=0.
    digit_changed=0
    blank_changed=0
    missing=set(blanks)
    for i in range(81):
        a,b=left[i],right[i]
        delta=[b[j]-a[j] for j in range(9)]
        max_abs=max(max_abs,max(abs(d) for d in delta))
        mean=sum(delta)/9
        centered_square+=sum((d-mean)**2 for d in delta)
        p=max(range(9),key=lambda j:a[j])
        q=max(range(9),key=lambda j:b[j])
        if p!=q:
            digit_changed+=1
            if i in missing:
                blank_changed+=1
    return {
        "max_absolute_drift":max_abs,
        "centered_drift_rms":math.sqrt(centered_square/(81*9)),
        "all81_argmax_changes":digit_changed,
        "blank_argmax_changes":blank_changed,
    }


def check_summary(summary,computed):
    if set(summary)!=set(computed):
        raise ValueError("reported drift statistics are incomplete")
    for field,expected in computed.items():
        reported=summary[field]
        if field.endswith("changes"):
            if reported!=expected:
                raise ValueError("saved original digit flips differ")
        elif not approx(reported,expected):
            raise ValueError("saved original/full-grid drift differs: "+field)


def verify_anchor(source_stats):
    if set(source_stats)!={"zh","zl","x_emb"}:
        raise ValueError("source ACT1 tensor checkpoint evidence incomplete")
    for data in source_stats.values():
        if set(data)!={"paired_sha256","single_sha256",
                       "max_abs_paired_minus_single","fro_paired_minus_single"}:
            raise ValueError("source paired/single extraction receipt missing")
        for key in ("paired_sha256","single_sha256"):
            if not isinstance(data[key],str) or len(data[key])!=64:
                raise ValueError("invalid ACT1 fingerprint")
            int(data[key],16)
        if not (0<=data["max_abs_paired_minus_single"]<=data["fro_paired_minus_single"]+1e-7):
            raise ValueError("source batch-size scalar norm identity failed")


def audit(report,original,source_sha):
    OLD.audit(original)
    if (source_sha!=ORIGINAL_FULL or
        report["schema"]!=SCHEMA or
        report["preregistration_commit"]!=PROTOCOL or
        report["source_ls_sae08_sha256"]!=ORIGINAL_FULL or
        report["source_checkpoint_sha256"]!=CHECKPOINT or
        report["mode"]!="retrospective_two_sentinels" or
        report["sentinels"]!=[{"family":f,"index":i} for f,i in SENTINELS] or
        report["modes"]!=list(MODES) or
        report["horizons"]!=list(HORIZONS) or
        report["physical_radii"]!=list(RADII) or
        report["direction_arm_order"]!=list(ARMS) or
        report["cpu_threads"]!=4 or
        len(report["cases"])!=2):
        raise ValueError("frozen LS-SAE10 registration or original source changed")
    max_by_sentinel={}
    same_process_identical={}
    for case,(family,index) in zip(report["cases"],SENTINELS):
        source=original["families"][family]["source_roster"][index]
        old=original["families"][family]["per_puzzle"][index]
        expected_status=old["source_direction_status"]
        expected_position=old["visible_goal_position"]
        if (case["family"]!=family or case["index"]!=index or
            case["seed"]!=source["seed"] or
            case["visible_goal_position"]!=expected_position or
            case["blank_count"]!=source["x"].count(0) or
            case["direction_status"]!=expected_status or
            set(case["modes"])!=set(MODES) or
            set(case["comparisons"])!=set(MODES)):
            raise ValueError("original LS-SAE08 puzzle, site or arms altered")
        verify_anchor(case["source_anchor_comparison"])
        blanks=[i for i,v in enumerate(source["x"]) if v==0]
        archived=old["native_81x9_logits"]
        historical_keys=[(h,k) for h,k,_ in all_variants(archived,expected_status)]
        for mode in MODES:
            raw=case["modes"][mode]
            keys=[(h,k) for h,k,_ in all_variants(raw,expected_status)]
            if keys!=historical_keys:
                raise ValueError("original source edit grid/order differs")
            saved=case["comparisons"][mode]
            if set(saved)!={"vs_historical","vs_canonical_a"}:
                raise ValueError("within-run and historical comparisons missing")
            for comparison,reference in (
                ("vs_historical",archived),
                ("vs_canonical_a",case["modes"]["canonical_a"]),
            ):
                saved_h=saved[comparison]
                if set(saved_h)!=set(HORIZONS):
                    raise ValueError("original ACT horizon comparisons missing")
                for horizon in HORIZONS:
                    found={name for h,name in historical_keys if h==horizon}
                    if set(saved_h[horizon])!=found:
                        raise ValueError("expected all native signed variants in comparison")
                    for name in found:
                        left=lookup(reference,horizon,name)
                        right=lookup(raw,horizon,name)
                        check_summary(saved_h[horizon][name],independent_stats(left,right,blanks))
        maxima={
            mode:max(item["max_absolute_drift"]
                     for by_h in case["comparisons"][mode]["vs_historical"].values()
                     for item in by_h.values())
            for mode in MODES
        }
        if set(case["max_original_drift_by_mode"])!=set(MODES) or any(
            not approx(maxima[k],case["max_original_drift_by_mode"][k]) for k in MODES
        ):
            raise ValueError("reported original drift maxima differ")
        hist_states=case["instrumented_clean_state"]
        if set(hist_states)!=set(HORIZONS):
            raise ValueError("instrumented hidden-state horizon fingerprints missing")
        for row in hist_states.values():
            if set(row)!={"clean_zh_sha256","clean_zl_sha256","clean_zh_l2","clean_zl_l2"}:
                raise ValueError("instrumented clean hidden-state scalars omitted")
            if any(not math.isfinite(row[k]) or row[k]<=0 for k in ("clean_zh_l2","clean_zl_l2")):
                raise ValueError("invalid model hidden-state clean norm")
            for k in ("clean_zh_sha256","clean_zl_sha256"):
                if len(row[k])!=64:raise ValueError("invalid clean native state digest")
                int(row[k],16)
        max_by_sentinel[family]=maxima
        same_process_identical[family]=all(
            item["max_absolute_drift"]==0
            for by_h in case["comparisons"]["canonical_b"]["vs_canonical_a"].values()
            for item in by_h.values()
        )
    return {
        "schema":"LS-SAE10/independent-stdlib-native-output-drift-replay-v1",
        "original_LS_SAE08_source_and_numeric_audit_passed":True,
        "two_registered_retrospective_sentinels_verified":True,
        "all_source_arms_doses_signs_horizons_present":True,
        "all_new_native_81x9_logits_and_drift_metrics_recomputed":True,
        "original_sudoku_prediction_argmax_comparison_verified":True,
        "same_process_canonical_a_equals_b_bitwise":same_process_identical,
        "max_historical_logit_drift":max_by_sentinel,
        "scientific_gate":"diagnostic_only_no_LS_SAE09_acceptance",
        "limitations":"Native hidden tensors are saved only as hashes and norms; independent audit checks the saved logits but does not rerun the checkpoint or independently reconstruct the hidden states.",
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--baseline",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    b=a.baseline.read_bytes()
    result=audit(json.loads(a.input.read_text()),json.loads(b),hashlib.sha256(b).hexdigest())
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
