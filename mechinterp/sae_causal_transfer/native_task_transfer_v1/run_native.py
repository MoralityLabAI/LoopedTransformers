"""LS-SAE06 native full-81 logit causal intervention and exact task scoring.

The structural and centered-logit power directions are fitted to source
geometry only. Ground-truth Sudoku target digits are introduced AFTER all
input directions and model forward variants have been frozen.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from mechinterp.loopscope02.scripts import trm_sudoku as TRM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as SOURCEFIT
from mechinterp.sae_causal_transfer.native_jvp_v2 import fit_pack as FITPACK
from mechinterp.sae_causal_transfer.native_singular_v1 import run_native as S5
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import families as FAMILY
from mechinterp.sae_causal_transfer.native_task_transfer_v1 import scoring as SCORE

PROTOCOL_COMMIT="3b1bec465b89c552b93cd413593045bf61d4cec0"
SCHEMA="LS-SAE06/complete-81-sudoku-causal-task-transfer-v1"
MODEL_SHA="932b5e0df4e90021c21a064d4e716154086c46fda7afa676ea294484a30e38c3"
NPZ_SHA="073b7771676e44eb934c50bdf1e7639ffeadbdff12998979cfe4c43e42b2da25"
MANIFEST_SHA="cb44b910f94210745b7a7df304e575d5a194ee8cd7bec51a88281fccc95bf33a"
FIT_SHA="83dacde8f13926fc41fc48cb17bcfba8c399538754efb7104e41e94bd93797ec"
PHYSICAL_SHA="14dc1a8b4130a1cbbf215e97a4a0d1c4004d23ded319546b6bf35ddc3238f5eb"
POWER_STEPS=4


def sha_seed(*items):
    b="LS-SAE06|"+"|".join(str(x) for x in items)
    return int.from_bytes(hashlib.sha256(b.encode()).digest()[:8],"big") & ((1<<63)-1)


def logit_map(model,anchor,token,site):
    zh=S5.frozen_output_from_site(model,anchor,token,site)
    logits=model.digit_logits(zh)
    return logits-logits.mean(dim=-1,keepdim=True)


def logit_power_direction(model,anchor,token,family,seed):
    """Four label-blind JtJ updates on all 81×9 centered predicted logits."""
    site=anchor["zh"].detach()[0,token].clone()
    u=S5.normed(np.random.default_rng(
        sha_seed(family,seed,"logit-power-start")).normal(size=512))
    if u is None:
        return None,[{"status":"null_initial_vector"}]
    progress=[]
    for t in range(POWER_STEPS):
        v=torch.tensor(u,dtype=site.dtype,device=site.device)
        with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
            _,tangent=torch.func.jvp(
                lambda h:logit_map(model,anchor,token,h),
                (site.detach(),),(v,))
        if tangent.shape!=(1,81,9) or not bool(torch.isfinite(tangent).all()):
            raise ArithmeticError("native all-81 centered logit JVP nonfinite")
        gain=float(torch.linalg.vector_norm(tangent))
        if gain<=1e-12:
            progress.append({"step":t,"status":"zero_logit_jvp",
                             "input_direction_512":u.tolist(),"output_gain":gain})
            return None,progress
        with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
            h=site.detach().clone().requires_grad_(True)
            output=logit_map(model,anchor,token,h)
            w=torch.autograd.grad(output,h,grad_outputs=tangent.detach())[0]
        wd=w.detach().double().cpu().numpy()
        wn=float(np.linalg.norm(wd))
        if not math.isfinite(wn) or wn<=1e-12:
            progress.append({"step":t,"status":"zero_logit_vjp",
                             "input_direction_512":u.tolist(),"output_gain":gain})
            return None,progress
        progress.append({
          "step":t,"status":"eligible",
          "input_direction_512":u.tolist(),
          "output_gain":gain,"reverse_gradient_norm":wn,
          "reverse_gradient_512":wd.tolist(),
          "jvp_dot_vjp":float(u@wd),"squared_gain":gain*gain
        })
        u=wd/wn
    return u,progress


def forward_geometry(model,anchor,token,unit):
    if unit is None:
        return None
    site=anchor["zh"].detach()[0,token].clone()
    v=torch.tensor(unit,dtype=site.dtype,device=site.device)
    with torch.enable_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        _,(hidden,centered_logits)=torch.func.jvp(
            lambda h:(S5.frozen_output_from_site(model,anchor,token,h),
                      logit_map(model,anchor,token,h)),
            (site.detach(),),(v,))
    z=hidden.detach().double().cpu().numpy()[0]
    lg=centered_logits.detach().double().cpu().numpy()[0]
    norms=np.linalg.norm(z,axis=-1)
    if (z.shape!=(97,512) or lg.shape!=(81,9)
        or not np.all(np.isfinite(z)) or not np.all(np.isfinite(lg))):
        raise ArithmeticError("native full-state and 81-cell logit JVP mismatch")
    if float(np.max(np.abs(lg.mean(axis=-1))))>2e-5:
        raise ArithmeticError("per-cell additive-logit gauge not removed")
    return {
      "direction_512":np.asarray(unit).tolist(),
      "all97_hidden_tangent_norms":norms.tolist(),
      "full_hidden_jvp_gain":float(np.linalg.norm(norms)),
      "goal_hidden_jvp_gain":float(norms[token]),
      "centered_digit_logit_jvp_81x9":lg.tolist(),
      "centered_digit_logit_jvp_gain":float(np.linalg.norm(lg))
    }


def freeze_source_directions(model,anchor,fit,family,seed,goal):
    token=model.puzzle_emb_len+goal
    state,progress_state=S5.power_direction(model,anchor,token,family,seed)
    lp,progress_logit=logit_power_direction(model,anchor,token,family,seed)
    sae,head,meta=S5.native_head_and_sae(model,anchor,fit,goal)
    random_unit=S5.normed(np.random.default_rng(
        sha_seed(family,seed,"random-final")).normal(size=512))
    if random_unit is None or state is None or lp is None:
        # Keep the original puzzle in the denominator with explicit abstention.
        pass
    directions={"state_power":state,"logit_power":lp,
                "sae":sae,"random":random_unit}
    geometry={name:forward_geometry(model,anchor,token,u)
              for name,u in directions.items()}
    return directions,{
       "original_sae_source":meta,
       "state_power_transcript":progress_state,
       "logit_power_transcript":progress_logit,
       "geometry":geometry
    }


def native_full_board_variants(model,anchor,goal,directions):
    """Equal physical dose, unchanged source zh/zl and every other source token."""
    ordering=[("clean",None,None,None)]
    for name in SCORE.ARMS:
        if directions[name] is None:
            continue
        for r in SCORE.RADII:
            for sign in ("plus","minus"):
                ordering.append((name,r,sign,1.0 if sign=="plus" else -1.0))
    token=model.puzzle_emb_len+goal
    n=len(ordering)
    with torch.no_grad(),sdpa_kernel(backends=[SDPBackend.MATH]):
        zh=anchor["zh"].detach().repeat(n,1,1)
        zl=anchor["zl"].detach().repeat(n,1,1)
        emb=anchor["x_emb"].detach().repeat(n,1,1)
        for idx,(name,r,sign,sgn) in enumerate(ordering[1:],1):
            v=torch.tensor(directions[name],device=zh.device,dtype=zh.dtype)
            zh[idx,token,:]+=sgn*r*v
        out={}
        for step in range(1,6):
            zh,zl=model.act_step(zh,zl,emb)
            if step in (2,5):
                label="ACT3" if step==2 else "ACT6"
                logits=model.digit_logits(zh).detach().double().cpu().numpy()
                if logits.shape!=(n,81,9) or not np.all(np.isfinite(logits)):
                    raise ArithmeticError("missing full 81×9 native Sudoku digit logits")
                record={"clean":logits[0].tolist()}
                for idx,(name,r,sign,_) in enumerate(ordering[1:],1):
                    record.setdefault(name,{}).setdefault(f"{r:.2f}",{})[sign]=logits[idx].tolist()
                out[label]=record
    return out


def score_case(model,fit,anchor,family,row,puzzle_index):
    goal=FAMILY.goal_from_visible_mask(row["x"])
    # Direction search is source-blind, without reading any target/solution.
    direction,src=freeze_source_directions(model,anchor,fit,family,row["seed"],goal)
    # All actual finite model outputs are frozen before task correctness/margins.
    logits=native_full_board_variants(model,anchor,goal,direction)
    scores={}
    for horizon in ("ACT3","ACT6"):
        clean=logits[horizon]["clean"]
        per_arm={}
        for arm in SCORE.ARMS:
            per_radius={}
            for r in SCORE.RADII:
                key=f"{r:.2f}"
                raw=logits[horizon].get(arm,{}).get(key)
                if raw is None:
                    per_radius[key]=None
                    continue
                item=SCORE.signed_pair_effect(clean,raw["plus"],raw["minus"],
                                               row["x"],row["target"],goal)
                item["centered_logit_effect_per_radius"]=item["mean_centered_logit_effect_unscaled"]/r
                item["centered_logit_symmetric_secant_per_radius"]=item["centered_logit_sym_secant_unscaled"]/r
                if horizon=="ACT3" and r==.05:
                    target=np.asarray(src["geometry"][arm]["centered_digit_logit_jvp_81x9"])
                    p=np.asarray(raw["plus"]);m=np.asarray(raw["minus"])
                    sec=SCORE.center(p-m)/(2*r)
                    item["centered_logit_secant_minus_jvp_fro"]=float(np.linalg.norm(sec-target))
                    item["centered_logit_secant_minus_jvp_relative"]=float(np.linalg.norm(sec-target)/(1+np.linalg.norm(target)))
                per_radius[key]=item
            per_arm[arm]=per_radius
        scores[horizon]=per_arm
    return {
       "puzzle_index":puzzle_index,"seed":row["seed"],"visible_goal_position":goal,
       "source_calibration_locked":True,
       "direction_metadata":src,
       "source_direction_status":{a:direction[a] is not None for a in SCORE.ARMS},
       "native_81x9_logits":logits,
       "task_score":scores
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
        raise ValueError("LS-SAE04 source-fit-only calibration SHA lock changed")
    model,sha=TRM.load_trm_checkpoint("gaoxin492","cpu",dtype=torch.float32)
    if sha!=MODEL_SHA:
        raise ValueError("native pretrained original model SHA changed")
    model.eval().requires_grad_(False)
    families=FAMILY.manifest()  # source split and cross-experiment disjointness
    names=("pilot_30","pilot_34") if pilot else ("test_30","test_34")
    observed={}
    for kind in names:
        fam=kind.replace("pilot_","test_")
        rows=FAMILY.roster(kind)
        _,anchors=SOURCEFIT.stage_one(model,rows,retain_zl=True)
        cases=[]
        for i,(r,anchor) in enumerate(zip(rows,anchors)):
            one=score_case(model,fit,anchor,fam,r,i)
            cases.append(one)
            d=one["task_score"]["ACT6"]
            print("LS_SAE06_PUZZLE",kind,i,
                  d["state_power"]["0.20"]["symmetric_blank_prediction_flip_score"]
                  if d["state_power"]["0.20"] else None,
                  d["random"]["0.20"]["symmetric_blank_prediction_flip_score"],flush=True)
        stats=SCORE.score_families(cases,fam)
        observed[kind]={"source_roster":rows,"per_puzzle":cases,"statistics":stats}
        print("LS_SAE06_FAMILY",kind,json.dumps(stats,sort_keys=True),flush=True)
    result={
       "schema":SCHEMA,"original_preregistration_commit":PROTOCOL_COMMIT,
       "mode":"pilot" if pilot else "full",
       "source_checkpoint_file_sha256":sha,
       "source_calibration_npz_file_sha256":NPZ_SHA,
       "source_calibration_manifest_sha256":manifest_sha,
       "source_calibration_fit_sha256":fit.fingerprint,
       "all_fresh_roster_sha256":families,
       "full_protocol":not pilot,
       "horizons":list(SCORE.HORIZONS),
       "radii":list(SCORE.RADII),
       "arms":list(SCORE.ARMS),
       "state_power_updates":POWER_STEPS,
       "logit_power_updates":POWER_STEPS,
       "families":observed,
       "registered_success_both_families":bool(not pilot and
          all(observed[n]["statistics"]["registered_success"] for n in ("test_30","test_34"))),
       "scope":"Native FP32 full 81×9 logits and exact Sudoku verified correctness from pretrained TRM, with source labels absent from direction selection. Independent stdlib receipt replay does not rerun pretrained model."
    }
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--source-dir",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--pilot",action="store_true")
    args=parser.parse_args()
    result=run(pilot=args.pilot,source_dir=args.source_dir,out=args.out)
    print(json.dumps({"mode":result["mode"],"both_family_gate":result["registered_success_both_families"],
                      "stats":{k:v["statistics"] for k,v in result["families"].items()}},indent=2))


if __name__=="__main__":
    main()
