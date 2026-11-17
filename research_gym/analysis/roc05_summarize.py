"""ROC-05A independent six-seed source-lock and paired family-transfer audit.

Pure standard library, accepts only six complete frozen model receipts
and twelve dev prediction tables. The answer uses seed-level sign flips,
never treating 256 within-seed dev cases as independent models.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean

SEEDS=(20261117231,20261117233,20261117239,20261117241,20261117251,20261117257)
ARMS=("low_affine4","high_permutation512")
EXPECTED=256


def exact_p(d):
    observed=mean(d)
    means=[mean(a*b for a,b in zip(signs,d))
           for signs in itertools.product((-1.,1.),repeat=len(d))]
    return sum(v>=observed-1e-15 for v in means),len(means)


def audit(folder:Path):
    receipts=[]
    for seed in SEEDS:
        path=folder/f"roc05_S{seed}_receipt.json"
        receipt=json.loads(path.read_text())
        if receipt["seed"]!=seed or receipt["steps"]!=1920:
            raise ValueError("seed/optimizer budget mismatch")
        if not receipt["identical_start"] or not receipt["identical_sample_indices"]:
            raise ValueError("unequal paired budget")
        if receipt["original_roc02_sealed_evaluated"] or receipt["roc05_reserved_test_evaluated"]:
            raise ValueError("sealed use")
        if set(receipt["final_arms"])!=set(ARMS):
            raise ValueError("missing comparator arm")
        assert len(receipt["milestones"])==8
        outcomes={}
        ids={}
        for arm in ARMS:
            path=folder/f"roc05_{arm}_S{seed}_development_predictions.csv"
            with path.open(newline="") as f:
                rows=list(csv.DictReader(f))
            if len(rows)!=EXPECTED or [int(r["dev_index"]) for r in rows]!=list(range(EXPECTED)):
                raise AssertionError("missing or duplicated dev rows")
            ids[arm]=[(r["source_id"],r["original_true"],r["swapped_true"]) for r in rows]
            if any(r["original_true"]==r["swapped_true"] for r in rows):
                raise AssertionError("invalid unchanged counterfactual")
            base=sum(int(r["original_correct"]) for r in rows)
            whole=sum(int(r["whole_table_exact"]) for r in rows)
            swapped=sum(int(r["swapped_correct"]) for r in rows)
            both=sum(int(r["both_original_and_swap_correct"]) for r in rows)
            eligible=[r for r in rows if int(r["nuisance_eligible"])]
            invariance=sum(int(r["nuisance_prediction_stable"]) for r in eligible)
            stats=receipt["final_arms"][arm]["development"]
            if abs(stats["accuracy"]-base/EXPECTED)>1e-10:
                raise AssertionError("accuracy inconsistent with per-case predictions")
            if abs(stats["whole_table_exact_accuracy"]-whole/EXPECTED)>1e-10:
                raise AssertionError("whole-table rate inconsistent")
            controls=receipt["causal_dev_controls"][arm]
            if (controls["nuisance_eligible_count"]!=len(eligible)
                or abs(controls["swapped_rule_accuracy"]-swapped/EXPECTED)>1e-10
                or abs(controls["joint_base_and_swapped_accuracy"]-both/EXPECTED)>1e-10):
                raise AssertionError("legal control receipt mismatch")
            outcomes[arm]={
                "dev_accuracy":base/EXPECTED,
                "dev_correct":base,
                "dev_whole_table_exact":whole/EXPECTED,
                "dev_whole_table_count":whole,
                "dev_rowwise_accuracy":stats["row_reconstruction_accuracy"],
                "dev_cross_entropy":stats["cross_entropy"],
                "train_accuracy":receipt["final_arms"][arm]["train"]["accuracy"],
                "legal_row_swap_accuracy":swapped/EXPECTED,
                "joint_original_swap_accuracy":both/EXPECTED,
                "nuisance_eligible":len(eligible),
                "nuisance_invariance":invariance/len(eligible) if eligible else None,
            }
        if ids[ARMS[0]]!=ids[ARMS[1]]:
            raise AssertionError("dev cases not paired")
        delta=outcomes[ARMS[1]]["dev_accuracy"]-outcomes[ARMS[0]]["dev_accuracy"]
        if abs(delta-receipt["primary_dev_accuracy_difference"])>1e-10:
            raise AssertionError("reported primary effect mismatches rows")
        receipts.append({
            "seed":seed,"primary_difference":delta,
            "source_manifest_sha256":receipt["source_manifest_sha256"],
            "arms":outcomes,
        })
    if len({r["source_manifest_sha256"] for r in receipts})!=1:
        raise AssertionError("source files differ across seeds")
    differences=[r["primary_difference"] for r in receipts]
    extreme,total=exact_p(differences)
    positive=sum(d>0 for d in differences)
    task_competence=sum(r["arms"]["high_permutation512"]["dev_accuracy"]>=.70 for r in receipts)
    table_competence=sum(r["arms"]["high_permutation512"]["dev_whole_table_exact"]>=.50 for r in receipts)
    primary_pass=(positive>=5 and mean(differences)>=.20 and extreme/total<=.05)
    competence_pass=task_competence>=5 and table_competence>=5
    return {
        "study":"ROC-05A-independent-rule-pool-and-model-seed-replication-v1",
        "scope":"development-only; ROC-05 reserved test and ROC-02 sealed6/7 untouched",
        "original_roc02_sealed_evaluated":False,"roc05_reserved_test_evaluated":False,
        "model_seed_count":len(receipts),"models_trained":len(receipts)*2,
        "dev_cases_per_seed":256,
        "paired_seed_results":receipts,
        "mean_accuracy_gain_high_minus_low":mean(differences),
        "positive_seed_count":positive,
        "exact_signflip_extreme_assignments":extreme,
        "exact_signflip_total":total,
        "exact_one_sided_p":extreme/total,
        "primary_thresholds":{"positive_seeds":5,"mean_gain":.20,"p":.05},
        "primary_gate":"PASS" if primary_pass else "FAIL",
        "competence_thresholds":{"task_dev_accuracy":.70,"table_dev_exact":.50,"seeds":5},
        "high_task_gate_seed_count":task_competence,
        "high_table_gate_seed_count":table_competence,
        "competence_gate":"PASS" if competence_pass else "FAIL",
        "mean_dev_accuracy":{
            arm:mean(row["arms"][arm]["dev_accuracy"] for row in receipts)
            for arm in ARMS
        },
        "mean_dev_full_table_accuracy":{
            arm:mean(row["arms"][arm]["dev_whole_table_exact"] for row in receipts)
            for arm in ARMS
        },
        "mean_legal_swap_accuracy":{
            arm:mean(row["arms"][arm]["legal_row_swap_accuracy"] for row in receipts)
            for arm in ARMS
        },
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--receipt-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.receipt_dir)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="paired_seed_results"},
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
