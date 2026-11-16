"""LS-SAE07 pretrained native hard-Sudoku 81-cell finite causal intervention.

Extends LS-SAE06 source-frozen four-arm source direction selection, preserving
the exact preheldout PCA/SAE artifact and recording all 81 Sudoku logits at
ACT3/ACT6/ACT12 under +/-0.20, +/-2, +/-8 physical token edits.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as SOURCEFIT
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as FITPACK
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import run_native as BASE
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import scoring as ORIGINAL_SCORE
from mechinterp.sae_causal_transfer.native_hard_task_v1 import families as FAMILY
from mechinterp.sae_causal_transfer.native_hard_task_v1 import scoring as SCORE

SCHEMA="LS-SAE07/source-pinned-hard-unique-Sudoku-causal-task-v1"
PROTOCOL_COMMIT="2641f05d5cc9a1d6c7359543c44693b151d3c3cf"
CHECKPOINT_SHA=BASE.MODEL_SHA
NPZ_SHA=BASE.NPZ_SHA
MANIFEST_SHA=BASE.MANIFEST_SHA
FIT_SHA=BASE.FIT_SHA
PHYSICAL_SHA=BASE.PHYSICAL_SHA


def native_all_outputs(model,anchor,position,directions):
    sequence=[("clean",None,None,0)]
    for arm in SCORE.ARMS:
        if directions[arm] is None:
            continue
        for r in SCORE.RADII:
            for sign in ("plus","minus"):
                sequence.append((arm,r,sign,+1 if sign=="plus" else -1))
    token=model.puzzle_emb_len+position
    n=len(sequence)
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        zh=anchor["zh"].detach().repeat(n,1,1)
        zl=anchor["zl"].detach().repeat(n,1,1)
        emb=anchor["x_emb"].detach().repeat(n,1,1)
        for index,(arm,r,sign,direction_sign) in enumerate(sequence[1:],1):
            unit=torch.tensor(directions[arm],device=zh.device,dtype=zh.dtype)
            zh[index,token,:]+=r*direction_sign*unit
        out={}
        for step in range(1,max(SCORE.HORIZONS)):
            zh,zl=model.act_step(zh,zl,emb)
            if step not in (2,5,11):
                continue
            horizon="ACT"+str(step+1)
            data=model.digit_logits(zh).detach().double().cpu().numpy()
            if data.shape!=(n,81,9) or not np.all(np.isfinite(data)):
                raise ArithmeticError("native full81 logits missing or nonfinite")
            raw={"clean":data[0].tolist()}
            for index,(arm,r,sign,_) in enumerate(sequence[1:],1):
                raw.setdefault(arm,{}).setdefault(f"{r:.2f}",{})[sign]=data[index].tolist()
            out[horizon]=raw
    if set(out)!={"ACT3","ACT6","ACT12"}:
        raise AssertionError("native ACT3/6/12 response grid altered")
    return out


def one_case(model,fit,anchor,family,row,puzzle_index):
    position=FAMILY.median_blank(row["x"])
    token=model.puzzle_emb_len+position
    source_norm=float(torch.linalg.vector_norm(anchor["zh"][0,token].float()))
    if source_norm<=1e-12 or not np.isfinite(source_norm):
        raise ArithmeticError("source ACT1 token has invalid original state norm")

    # Exact LS-SAE06 label-blind search. No access to Sudoku solution.
    directions,metadata=BASE.freeze_source_directions(
        model,anchor,fit,family,row["seed"],position)
    # Freeze actual all-board model outputs for all arms/doses/horizons first.
    logits=native_all_outputs(model,anchor,position,directions)
    scored={}
    for horizon in ("ACT3","ACT6","ACT12"):
        per_arm={}
        for arm in SCORE.ARMS:
            per_radius={}
            for r in SCORE.RADII:
                key=f"{r:.2f}"
                raw=logits[horizon].get(arm,{}).get(key)
                if raw is None:
                    per_radius[key]=None
                    continue
                item=ORIGINAL_SCORE.signed_pair_effect(
                    logits[horizon]["clean"],raw["plus"],raw["minus"],
                    row["x"],row["target"],position)
                item["centered_logit_effect_per_radius"]=item["mean_centered_logit_effect_unscaled"]/r
                item["centered_logit_symmetric_secant_per_radius"]=item["centered_logit_sym_secant_unscaled"]/r
                if horizon=="ACT3" and r==.2:
                    jvp=np.asarray(metadata["geometry"][arm]["centered_digit_logit_jvp_81x9"])
                    sec=ORIGINAL_SCORE.center(
                        np.asarray(raw["plus"])-np.asarray(raw["minus"]))/(2*r)
                    error=float(np.linalg.norm(sec-jvp))
                    item["centered_logit_secant_minus_jvp_fro"]=error
                    item["centered_logit_secant_minus_jvp_relative"]=error/(1+float(np.linalg.norm(jvp)))
                per_radius[key]=item
            per_arm[arm]=per_radius
        scored[horizon]=per_arm
    return {
      "puzzle_index":puzzle_index,
      "seed":row["seed"],
      "visible_goal_position":position,
      "source_calibration_locked":True,
      "original_source_zH_token_norm":source_norm,
      "radius_over_clean_source_norm":{
          f"{r:.2f}":r/source_norm for r in SCORE.RADII},
      "direction_metadata":metadata,
      "source_direction_status":{
          arm:directions[arm] is not None for arm in SCORE.ARMS},
      "native_81x9_logits":logits,
      "task_score":scored
    }


def run(*,pilot=False,source_dir,out=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    source_dir=Path(source_dir)
    fit,manifest,manifest_sha=FITPACK.load_pack(
        source_dir/"frozen_calibration.npz",
        source_dir/"frozen_calibration_manifest.json")
    if (manifest_sha!=MANIFEST_SHA or manifest["npz_file_sha256"]!=NPZ_SHA
        or fit.fingerprint!=FIT_SHA
        or manifest["physical_dictionary_sha256"]!=PHYSICAL_SHA):
        raise ValueError("original LS-SAE04 immutable calibration source changed")
    model,sha=TRM.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if sha!=CHECKPOINT_SHA:
        raise ValueError("original pretrained 7M TRM checkpoint source SHA mismatch")
    model.eval().requires_grad_(False)

    manifests=FAMILY.full_manifest()
    selected=("pilot_52","pilot_54") if pilot else ("test_52","test_54")
    observed={}
    for family in selected:
        rows=FAMILY.roster(family)
        _,anchors=SOURCEFIT.stage_one(model,rows,retain_zl=True)
        rec=[]
        for i,(row,anchor) in enumerate(zip(rows,anchors)):
            case=one_case(model,fit,anchor,family.replace("pilot_","test_"),row,i)
            rec.append(case)
            primary=case["task_score"]["ACT6"]
            a=primary["state_power"]["8.00"]
            b=primary["random"]["8.00"]
            print("LS_SAE07_CASE",family,i,
                  a["symmetric_blank_prediction_flip_score"] if a else None,
                  b["symmetric_blank_prediction_flip_score"],
                  "clean_correct",b["clean"]["blank_correct"],flush=True)
        stats=SCORE.family_score(rec,family.replace("pilot_","test_"))
        observed[family]={"source_roster":rows,"per_puzzle":rec,"statistics":stats}
        print("LS_SAE07_FAMILY",family,json.dumps(stats,sort_keys=True),flush=True)

    result={
        "schema":SCHEMA,"preregistration_commit":PROTOCOL_COMMIT,
        "mode":"pilot" if pilot else "full",
        "source_checkpoint_sha256":sha,
        "original_preheldout_calibration_npz_sha256":NPZ_SHA,
        "source_calibration_manifest_sha256":manifest_sha,
        "source_fit_array_sha256":fit.fingerprint,
        "all_registered_roster_sha256":manifests,
        "full_protocol":not pilot,
        "original_source_horizon_ACT":list(SCORE.HORIZONS),
        "physical_radii":list(SCORE.RADII),
        "arms":list(SCORE.ARMS),
        "state_power_updates":4,"logit_power_updates":4,
        "families":observed,
        "registered_success_both_families":bool(
            not pilot and all(observed[k]["statistics"]["registered_success"]
                              for k in ("test_52","test_54"))),
        "scope":"Original pretrained TRM finite physical source edits; all81 actual logits and exact Sudoku verification; independent stdlib replay, not independent neural execution."
    }
    if out is not None:
        out=Path(out)
        out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--pilot",action="store_true")
    a=parser.parse_args()
    r=run(pilot=a.pilot,source_dir=a.source_dir,out=a.out)
    print(json.dumps({"mode":r["mode"],"success":r["registered_success_both_families"],
        "families":{f:x["statistics"] for f,x in r["families"].items()}},indent=2))


if __name__=="__main__":
    main()
