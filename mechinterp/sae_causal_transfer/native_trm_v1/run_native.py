"""LS-SAE02 native source-locked TRM zH causal effects and recurrent gradients."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch

from mechinterp.loopscope02.scripts import trm_sudoku as TS
from mechinterp.sae_causal_transfer.observer import plan_edit
from mechinterp.sae_causal_transfer.native_trm_v1 import families as FAM
from mechinterp.sae_causal_transfer.native_trm_v1 import fit as FIT

MODEL_TAG="gaoxin492"
MODEL_SHA=TS.TRM_CHECKPOINTS[MODEL_TAG]["sha256"]
DEPTHS=(0,1,2)
GOALS=2
RANDOM_OBSERVERS=128
BOOTSTRAPS=2000


def token_alignment(v,g):
    v=np.asarray(v,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
    den=float(np.linalg.norm(v)*np.linalg.norm(g))
    return abs(float(v@g))/den if den>1e-12 else None


def random_observers(g,feature_vector,task_vector,family,seed,goal_slot,depth):
    """128 independent rank-one controls separate from sham candidates."""
    matched=token_alignment(feature_vector,g)
    task=token_alignment(task_vector,g)
    if matched is None:
        return {"sae_alignment":None,"task_only_alignment":task,
                "random_alignments":[],"rank_p":None}
    null=[]
    for j in range(RANDOM_OBSERVERS):
        material=f"LS-SAE02|{family}|{seed}|{goal_slot}|{depth}|{j}"
        digest=hashlib.sha256(material.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(digest[:8],"big")&((1<<63)-1))
        vec=rng.normal(size=len(g))
        null.append(token_alignment(vec,g))
    p=(1+sum(x is not None and x>=matched-1e-15 for x in null))/(RANDOM_OBSERVERS+1)
    return {"sae_alignment":matched,"task_only_alignment":task,
            "random_alignments":null,"rank_p":p}


def four_branch_logits(model,anchor,position,fit,plan):
    """Batched clean/kill/baseline/best native continuation, fixed zL at injection."""
    torch_device=anchor["zh"].device
    zh=anchor["zh"].detach().repeat(4,1,1)
    zl=anchor["zl"].detach().repeat(4,1,1)
    emb=anchor["x_emb"].detach().repeat(4,1,1)
    edit_token=model.puzzle_emb_len+position
    reductions=[np.zeros(fit.basis.shape[1]),
                plan["kill"],plan["baseline"],plan["best"]]
    edits=np.stack([r@fit.basis.T for r in reductions])
    displacement=torch.tensor(edits,dtype=zh.dtype,device=torch_device)
    with torch.no_grad():
        zh[:,edit_token,:]-=displacement
        outputs=[model.digit_logits(zh)[:,position,:].float().cpu()]
        for _ in range(max(DEPTHS)):
            zh,zl=model.act_step(zh,zl,emb)
            outputs.append(model.digit_logits(zh)[:,position,:].float().cpu())
    return outputs


def native_gradients(model,anchor,position,truth,clean_logits,fit):
    """Full BPTT from clean step-one zH, with no detached recurrence gradient."""
    token=model.puzzle_emb_len+position
    anchor_h=anchor["zh"].detach().clone().requires_grad_(True)
    zl=anchor["zl"].detach().clone()
    xemb=anchor["x_emb"].detach()
    hidden=[anchor_h]
    with torch.enable_grad():
        zh=anchor_h
        for _ in range(max(DEPTHS)):
            zh,zl=model.act_step(zh,zl,xemb)
            hidden.append(zh)
        results={}
        for k in DEPTHS:
            rivals=clean_logits[k][0].clone()
            rivals[truth]=-float("inf")
            competitor=int(torch.argmax(rivals))
            logits=model.digit_logits(hidden[k])[0,position]
            m=logits[truth]-logits[competitor]
            g=torch.autograd.grad(m,anchor_h,retain_graph=(k!=DEPTHS[-1]))[0]
            projected=g[0,token].detach().double().cpu().numpy()@fit.basis
            if not np.all(np.isfinite(projected)):
                raise ArithmeticError("native unrolled autograd gradient is nonfinite")
            results[k]=(projected,competitor)
    return results


def score_case(model,anchor,position,truth,fit,plan,family,seed,slot):
    """Exactly one token/ACT1 anchor, measured at fixed registered depths."""
    logits=four_branch_logits(model,anchor,position,fit,plan)
    gradients=native_gradients(model,anchor,position,truth,logits,fit)
    rows={}
    head=model.lm_head.weight.detach().double().cpu().numpy()
    edit_direction=np.asarray(plan["kill"])-np.asarray(plan["best"])
    for k in DEPTHS:
        v=logits[k].double().numpy()
        y=int(truth);gradient,rival=gradients[k]
        margins=v[:,y]-v[:,rival]
        d=float(margins[3]-margins[1])
        first=float(gradient@edit_direction)
        winner=int(np.argmax(v[0]))
        competitor=int(np.argsort(-v[0],kind="stable")[1])
        task_direction=(head[winner+2]-head[competitor+2])@fit.basis
        score=random_observers(gradient,plan["kill"],task_direction,
                              family,seed,slot,k)
        if score["rank_p"] is not None and len(score["random_alignments"])!=RANDOM_OBSERVERS:
            raise AssertionError("random observer budget changed")
        clean_correct=winner==y
        rows[str(k)]={
            "depth":k,"true_digit_zero_based":y,"clean_rival_digit_zero_based":rival,
            "predicted_clean_digit_zero_based":winner,
            "clean_logits_9":v[0].tolist(),
            "predicted_digits_by_arm":[int(np.argmax(v[arm])) for arm in range(4)],
            "task_only_direction_48":task_direction.tolist(),
            "margins":{"clean":float(margins[0]),"kill":float(margins[1]),
                       "baseline":float(margins[2]),"best":float(margins[3])},
            "kill_minus_best_margin_loss":d,
            "kill_minus_baseline_margin_loss":float(margins[2]-margins[1]),
            "predicted_first_order_gradient_effect":first,
            "nonlinear_residual":d-first,
            "relative_nonlinear_residual":abs(d-first)/(1+abs(d)),
            "gradient_48":gradient.tolist(),
            "clean_correct":bool(clean_correct),
            "kill_clean_correct_to_wrong":bool(clean_correct and np.argmax(v[1])!=y),
            "best_clean_correct_to_wrong":bool(clean_correct and np.argmax(v[3])!=y),
            "baseline_clean_correct_to_wrong":bool(clean_correct and np.argmax(v[2])!=y),
            "random_rank1_controls":score,
        }
    return rows


def summarize_family(rows,family):
    counts=[];sums=[];flips_k=flips_b=0
    for row in rows:
        ds=[]
        for g in row["goals"]:
            if not g["eligible"]:
                continue
            v=g["depths"]["2"]
            ds.append(v["kill_minus_best_margin_loss"])
            flips_k+=int(v["kill_clean_correct_to_wrong"])
            flips_b+=int(v["best_clean_correct_to_wrong"])
        counts.append(len(ds));sums.append(sum(ds))
    total=sum(counts);n_active=sum(c>0 for c in counts)
    rng=random.Random(202619130+len(family))
    values=[]
    for _ in range(BOOTSTRAPS):
        ids=[rng.randrange(len(rows)) for _ in rows]
        denominator=sum(counts[j] for j in ids)
        if denominator>0:
            values.append(sum(sums[j] for j in ids)/denominator)
    values.sort()
    valid=len(values)==BOOTSTRAPS
    interval=[values[50],values[1949]] if valid else None
    passed=(len(rows)==16 and total>=12 and n_active>=8 and
            valid and interval is not None and interval[0]>0 and flips_k>flips_b)
    means={}
    for k in DEPTHS:
        vals=[g["depths"][str(k)] for r in rows for g in r["goals"] if g["eligible"]]
        means[str(k)]={
           "mean_kill_minus_best":sum(v["kill_minus_best_margin_loss"] for v in vals)/len(vals) if vals else None,
           "mean_gradient_prediction":sum(v["predicted_first_order_gradient_effect"] for v in vals)/len(vals) if vals else None,
           "mean_relative_nonlinear_residual":sum(v["relative_nonlinear_residual"] for v in vals)/len(vals) if vals else None,
           "median_rank_p":float(np.median([v["random_rank1_controls"]["rank_p"] for v in vals
                   if v["random_rank1_controls"]["rank_p"] is not None])) if any(v["random_rank1_controls"]["rank_p"] is not None for v in vals) else None,
        }
    return {"eligible_goals":total,"eligible_puzzles":n_active,
            "selected_goals":len(rows)*GOALS,
            "paired_mean_depth2":sum(sums)/total if total else None,
            "puzzle_bootstrap95":interval,
            "valid_bootstrap_resamples":len(values),
            "kill_clean_correct_to_wrong":flips_k,
            "best_clean_correct_to_wrong":flips_b,
            "registered_success":bool(passed),
            "by_depth":means}


def run(*,pilot=False,out=None):
    torch.set_num_threads(4)
    torch.manual_seed(71)
    torch.set_grad_enabled(False)
    rosters,manifests=FAM.build_rosters(pilot=pilot)
    # Check all registered new full solutions/visible inputs before any model state access.
    assert len({tuple(row["target"]) for rows in rosters.values() for row in rows})==sum(map(len,rosters.values()))
    model,sha=TS.load_trm_checkpoint(MODEL_TAG,"cpu",dtype=torch.float32)
    if sha!=MODEL_SHA or model.hidden_size!=512:
        raise ValueError("source checkpoint fingerprint or model shape changed")
    model.requires_grad_(False).eval()
    fit_states,_=FIT.stage_one(model,rosters["fit_30"])
    fit=FIT.fit_only(fit_states)
    fit_receipt=FIT.frozen_fit_receipt(fit,manifests["fit_30"],sha)
    print("LS_SAE02_CALIBRATION_FROZEN",json.dumps(fit_receipt,sort_keys=True),flush=True)
    # Map diagnostics are held separate and never refit source-normalizer or SAE.
    diag_states,_=FIT.stage_one(model,rosters["map_30"])
    diag=FIT.fit_support_diagnostics(fit,diag_states)
    print("LS_SAE02_SUPPORT_DIAGNOSTIC_FROZEN",diag["never_active_atoms"],flush=True)
    result={}
    for family in ("test_30","test_34"):
        states,anchors=FIT.stage_one(model,rosters[family],retain_zl=True)
        family_rows=[]
        for i,row in enumerate(rosters[family]):
            selected=FAM.visible_goals(row["x"])
            goal_rows=[]
            for slot,pos in enumerate(selected):
                reduced=fit.encode(states[i,pos])
                case_key=f"LS-SAE02|{family}|{row['seed']}|{slot}"
                from mechinterp.sae_causal_transfer.observer import encode_top4
                code,_=encode_top4(reduced,fit.physical_rows,fit.physical_bias)
                vals=np.sort(code)[::-1]
                top=int(np.argmax(code))
                if vals[0]<=0 or vals[0]-vals[1]<=1e-8:
                    plan={"eligible":False,"reason":"source_top_one_not_strict"}
                else:
                    plan=plan_edit(reduced,fit.physical_rows,fit.physical_bias,
                                   fit.physical_decoder,top,case_key=case_key)
                receipt={"position":int(pos),"feature":top,"eligible":bool(plan["eligible"]),
                         "reason":plan["reason"],"case_key":case_key}
                if plan["eligible"]:
                    receipt["physical_edit_48"]={
                        name:plan[name].tolist() for name in ("kill","best","baseline")}
                    receipt["geometry"]=plan["diagnostics"]
                    # Only now access the heldout target digit or continuation output.
                    receipt["depths"]=score_case(model,anchors[i],pos,
                        row["target"][pos]-1,fit,plan,family,row["seed"],slot)
                goal_rows.append(receipt)
            family_rows.append({"puzzle_index":i,"seed":row["seed"],
                                "goal_positions":list(selected),"goals":goal_rows})
        stats=summarize_family(family_rows,family)
        result[family]={"per_puzzle":family_rows,"statistics":stats}
        print("LS_SAE02_HELDOUT",family,json.dumps(stats,sort_keys=True),flush=True)
    report={
      "schema":"LS-SAE02/native-TRM-actual-autograd-depth-gradient-v1",
      "preregistration_commit":"3e5e29df0886c3754c2e5b9a07050953444f860b",
      "checkpoint_tag":MODEL_TAG,"checkpoint_file_sha256":sha,
      "mode":"pilot" if pilot else "full",
      "floating_dtype":"float32",
      "source_rosters":rosters,"source_roster_sha256":manifests,
      "fit_receipt":fit_receipt,"map_diagnostics":diag,
      "test":result,"depth_grid":list(DEPTHS),
      "random_rank1_observer_budget":RANDOM_OBSERVERS,
      "full_protocol":not pilot,
      "registered_success_both_families":bool(
          not pilot and all(result[k]["statistics"]["registered_success"] for k in ("test_30","test_34"))),
      "limitation":"Trained LoopedTRM within-checkpoint FP32 48D PCA SAE intervention; not cross-checkpoint identity, unique mechanism, or Arb-certified."}
    if out is not None:
        out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
        out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    r=run(pilot=a.pilot,out=a.out)
    print(json.dumps({"out":str(a.out),"full":r["full_protocol"],
                      "passed":r["registered_success_both_families"],
                      "covered":{k:v["statistics"]["eligible_goals"] for k,v in r["test"].items()}},indent=2))


if __name__=="__main__":
    main()
