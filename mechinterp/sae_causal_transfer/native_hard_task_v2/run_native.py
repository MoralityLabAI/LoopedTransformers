"""LS-SAE08 exact-source-admitted 52/54-hole native causal Sudoku campaign.

Reuses the frozen LS-SAE07 physical edit runner/scorer, after a NEW prospectively
registered model-blind first-admissible 32-candidate source-grid algorithm.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as SOURCEFIT
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as FITPACK
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as S6
from mechinterp.sae_causal_transfer.native_hard_task_v1 import run_native as NATIVE
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as SCORE
from mechinterp.sae_causal_transfer.native_hard_task_v2 import families as FAMILY

SCHEMA="LS-SAE08/model-blind-admitted-hard-native-causal-task-v1"
PROTOCOL_COMMIT="6b38c3dcf65dbc3a812e0f7c63b79dc14c6893fe"


def run(*,pilot=False,source_dir,out=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    source_dir=Path(source_dir)
    fit,manifest,manifest_sha=FITPACK.load_pack(
        source_dir/"frozen_calibration.npz",
        source_dir/"frozen_calibration_manifest.json")
    if (manifest_sha!=S6.MANIFEST_SHA or
        manifest["npz_file_sha256"]!=S6.NPZ_SHA
        or fit.fingerprint!=S6.FIT_SHA
        or manifest["physical_dictionary_sha256"]!=S6.PHYSICAL_SHA):
        raise ValueError("original preheldout source PCA/SAE was changed")
    model,sha=TRM.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if sha!=S6.MODEL_SHA:
        raise ValueError("original pretrained checkpoint source byte SHA mismatch")
    model.eval().requires_grad_(False)
    manifests=FAMILY.full_manifest()
    names=("pilot_52","pilot_54") if pilot else ("test_52","test_54")
    original={}
    for key in names:
        rosters=FAMILY.roster(key)
        _,anchors=SOURCEFIT.stage_one(model,rosters,retain_zl=True)
        results=[]
        for index,(row,anchor) in enumerate(zip(rosters,anchors)):
            case=NATIVE.one_case(model,fit,anchor,
                                 key.replace("pilot_","test_"),row,index)
            results.append(case)
            act6=case["task_score"]["ACT6"]
            a=act6["state_power"]["8.00"]
            b=act6["random"]["8.00"]
            print("LS_SAE08_CASE",key,index,
                  "source_seed",row["seed"],
                  "source_attempt",row["candidate_attempt_index"],
                  "power_flips",a["symmetric_blank_prediction_flip_score"] if a else None,
                  "random_flips",b["symmetric_blank_prediction_flip_score"],
                  "clean_correct",b["clean"]["blank_correct"],flush=True)
        stats=SCORE.family_score(results,key.replace("pilot_","test_"))
        original[key]={"source_roster":rosters,"per_puzzle":results,"statistics":stats}
        print("LS_SAE08_PRIMARY",key,json.dumps(stats,sort_keys=True),flush=True)
    payload={
      "schema":SCHEMA,"preregistration_commit":PROTOCOL_COMMIT,
      "source_model_file_sha256":sha,
      "original_calibration_npz_file_sha256":S6.NPZ_SHA,
      "original_calibration_manifest_sha256":manifest_sha,
      "original_calibration_fit_array_sha256":fit.fingerprint,
      "mode":"pilot" if pilot else "full",
      "all_source_candidate_roster_sha256":manifests,
      "source_candidate_budget_per_slot":FAMILY.MAX_CANDIDATES,
      "full_protocol":not pilot,
      "horizons":[3,6,12],
      "physical_radii":list(SCORE.RADII),
      "direction_arm_order":list(SCORE.ARMS),
      "state_power_updates":4,"logit_power_updates":4,
      "families":original,
      "registered_success_both_families":bool(
        not pilot and all(original[x]["statistics"]["registered_success"]
                          for x in ("test_52","test_54"))),
      "limits":"FP32 native JVP source and full 81x9 output numerical receipts independently replayed; not a second pretrained inference or formal interval-certified proof."
    }
    if out is not None:
        out=Path(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--pilot",action="store_true")
    args=p.parse_args()
    r=run(pilot=args.pilot,source_dir=args.source_dir,out=args.out)
    print(json.dumps({"out":str(args.out),"pilot":args.pilot,
                      "scientific_gate":r["registered_success_both_families"],
                      "family":{k:v["statistics"] for k,v in r["families"].items()}},
                     indent=2))


if __name__=="__main__":
    main()
