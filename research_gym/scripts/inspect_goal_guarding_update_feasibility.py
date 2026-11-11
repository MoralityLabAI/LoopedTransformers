"""GG-04 post-hoc feasibility audit; leaves the registered experiment frozen.

The original GG-04 preregistered calibration guard rejected its first full
pilot. This script records the conditional real-SGD effects for BOTH seeds
under the original six-cue sealed gold panel, and a separate exploratory
four-cue alternative. It neither trains an actor nor changes GG-04's gate.
Gold labels remain exactly calculated by the same independent Bayes oracle.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from statistics import median
import json
import math

import torch

from research_gym.interp.goal_guarding_causal import make_panel,train_cell
from research_gym.interp.goal_guarding_actual_updates import (
    SealedGoldAuditor,query_splits,_finite_effect_log,
    BASE_COMMIT,OPTIMIZER_LR,REPORT_GRAD_CLIP,
)


def _summary(effects):
    gaps=[x.damage_gap for x in effects]
    return {
        "n":len(gaps),
        "median_flip_minus_truth":float(median(gaps)),
        "mean_flip_minus_truth":sum(gaps)/len(gaps),
        "fraction_positive_gap":sum(x>0 for x in gaps)/len(gaps),
        "fraction_negative_gap":sum(x<0 for x in gaps)/len(gaps),
        "min_gap":min(gaps),
        "max_gap":max(gaps),
        "mean_truth_gold_delta":sum(x.gold_loss_delta_truth for x in effects)/len(effects),
        "mean_flip_gold_delta":sum(x.gold_loss_delta_flip for x in effects)/len(effects),
        "prediction_mae":sum(
            abs(x.gold_loss_delta_truth-x.linear_delta_truth)+
            abs(x.gold_loss_delta_flip-x.linear_delta_flip)
            for x in effects)/(2*len(effects)),
    }


def run_diagnostic(out:Path,artifacts:Path)->dict:
    torch.set_num_threads(1)
    rows=query_splits()["calibration"]
    artifacts.mkdir(parents=True,exist_ok=True)
    result={
        "protocol":"RMP-GG-04-CALIBRATION-POSTHOC-DIAGNOSTIC",
        "source_main_commit":BASE_COMMIT,
        "registered_gg04_update_lr":OPTIMIZER_LR,
        "registered_gg04_gradient_norm_clip":REPORT_GRAD_CLIP,
        "evaluated_data_split":"GG04 calibration only (24 frozen unique queries)",
        "registered_original_six_cue_gold_seed":68001,
        "exploratory_four_cue_gold_seed":69001,
        "original_GG04_calibration_guard_intact":True,
        "posthoc_not_primary_locked_result":True,
        "results":[],
    }
    for seed in (11,17):
        model,record=train_cell(seed,steps=400,save_dir=artifacts)
        per_gold={}
        for name,visits,source in (
            ("registered_six_cue",6,68001),
            ("exploratory_four_cue",4,69001),
        ):
            auditor=SealedGoldAuditor(model,make_panel(128,visits,source))
            if not auditor.check_no_update_identity():
                raise AssertionError("physical no-update identity lost")
            effects=tuple(auditor.measure(row) for row in rows)
            ledger=artifacts/f"gg04_diagnostic_seed{seed}_{name}.json"
            ledger.write_text(json.dumps({
                "name":name,
                "seed":seed,
                "gold_baseline":auditor.baseline_loss,
                "query_effects":_finite_effect_log(rows,effects)
            },indent=2,sort_keys=True,allow_nan=False)+"\n")
            per_gold[name]={
                "pre_update_gold_ce":auditor.baseline_loss,
                "statistics":_summary(effects),
                "ledger":ledger.name
            }
        result["results"].append({
            "seed":seed,"base_record":record,
            "gold_conditions":per_gold,
        })
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2,sort_keys=True,
                              allow_nan=False)+"\n")
    return result


def main()->None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("output/gg04/diagnostic.json"))
    parser.add_argument("--artifact-dir",type=Path,default=Path("output/gg04/diagnostic"))
    args=parser.parse_args()
    result=run_diagnostic(args.out,args.artifact_dir)
    print(json.dumps({
        "protocol":result["protocol"],
        "results":[{
            "seed":entry["seed"],
            "registered_six_cue":entry["gold_conditions"][
                "registered_six_cue"]["statistics"],
            "exploratory_four_cue":entry["gold_conditions"][
                "exploratory_four_cue"]["statistics"],
        } for entry in result["results"]]
    },indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
