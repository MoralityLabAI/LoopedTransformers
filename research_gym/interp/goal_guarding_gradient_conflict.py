"""GG-05: exact-autograd first-order conflict versus finite-step SGD effects.

Two previously inspected heldout query sets are used solely as the *fixed*
48-query comparison roster. Fresh 4/6-cue gold panels are primary; prior
GG-04B gold panels are separately labelled historical replication.

The native GG-01 tied recurrent model, report loss, clipping and physical
SGD step are preserved. Dot products are accumulated in float64 for
diagnostics; native gradients and SGD weights are float32, and differences
of finite CE values are measured using actual native PyTorch forwards.

No claims about an agent knowing or changing its own optimizer follow.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any
import json
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, make_panel, train_cell, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    Query, query_splits, _tensor_rows, _rows_hash,
)
from research_gym.interp.goal_guarding_actual_updates_b import _panel_hash


BASE_SHA="15512fb3b85c988442372c242317e82fe726372e"
MODEL_SEEDS=(11,17)
LR_GRID=(0.0125,0.025,0.05)
SIGN_EPS=1e-7
GOLD_SPECS=(
    ("fresh4",4,69002),("fresh6",6,68002),
    ("old4",4,69001),("old6",6,68001),
)
GOLD_EXAMPLES=128
CLIP=1.0


def key_eta(lr:float)->str:
    return f"{lr:.4f}"


def sign(value:float, epsilon:float=SIGN_EPS)->int:
    if not math.isfinite(value) or not math.isfinite(epsilon) or epsilon<0:
        raise ValueError("sign requires finite value and nonnegative tolerance")
    if value>epsilon: return 1
    if value < -epsilon: return -1
    return 0


def mixture_threshold(first:float,second:float,
                      epsilon:float=SIGN_EPS)->float | None:
    """Gold-mix weight lambda where (1-lambda)*first+lambda*second=0."""
    if sign(first,epsilon)*sign(second,epsilon)!=-1:
        return None
    point=first/(first-second)
    if not 0<point<1:
        raise AssertionError("opposing signs need an interior mixture point")
    return point


def dot64(a:tuple[Tensor,...],b:tuple[Tensor,...])->float:
    if len(a)!=len(b) or any(x.shape!=y.shape for x,y in zip(a,b)):
        raise ValueError("gradient tuple shape mismatch")
    return sum(float((x.detach().double()*y.detach().double()).sum())
               for x,y in zip(a,b))


def norm64(a:tuple[Tensor,...])->float:
    return math.sqrt(max(0.0,dot64(a,a)))


def cosine64(a:tuple[Tensor,...],b:tuple[Tensor,...])->float | None:
    norm=norm64(a)*norm64(b)
    if norm<=1e-20: return None
    return max(-1.0,min(1.0,dot64(a,b)/norm))


def subtract(a:tuple[Tensor,...],b:tuple[Tensor,...])->tuple[Tensor,...]:
    if len(a)!=len(b):raise ValueError("gradient tuple mismatch")
    return tuple(x-y for x,y in zip(a,b))


def gradient(model:nn.Module,loss:Tensor,*,clip:float | None = None
             )->tuple[tuple[Tensor,...],float]:
    """Use autograd over the entire native parameter roster."""
    model.zero_grad(set_to_none=True)
    loss.backward()
    raw=float(nn.utils.clip_grad_norm_(
        model.parameters(),clip if clip is not None else float("inf")))
    grad=tuple(p.grad.detach().clone() if p.grad is not None
               else torch.zeros_like(p) for p in model.parameters())
    model.zero_grad(set_to_none=True)
    if not math.isfinite(raw):
        raise ValueError("gradient norm is nonfinite")
    return grad,raw


def panel_ce(model:EvaluationBeliefLooper,panel)->Tensor:
    return F.cross_entropy(
        model(panel.observations,panel.context)[0][:,-1,:],
        panel.labels,
    )


def panel_gradients(base:EvaluationBeliefLooper,panels:dict[str,Any]
                    )->tuple[dict[str,tuple[Tensor,...]],dict[str,float]]:
    gradients={}
    losses={}
    for name,panel in panels.items():
        loss=panel_ce(base,panel)
        losses[name]=float(loss.detach())
        gradients[name],_=gradient(base,loss)
    return gradients,losses


def query_gradients(base:EvaluationBeliefLooper,row:Query
                    )->tuple[dict[str,tuple[Tensor,...]],dict[str,float]]:
    bits,context,_=_tensor_rows((row,))
    gradients={}
    norms={}
    for arm,report in (("truth",row.label),("flip",1-row.label)):
        loss=F.cross_entropy(base(bits,context)[0][:,-1,:],
                             torch.tensor([report],dtype=torch.long))
        gradients[arm],norms[arm]=gradient(base,loss,clip=CLIP)
        if norm64(gradients[arm])>CLIP+1e-5:
            raise AssertionError("registered report clipping failed")
    return gradients,norms


def physical_step_scores(base:EvaluationBeliefLooper,
                         report_grad:tuple[Tensor,...],lr:float,
                         panels:dict[str,Any],baseline_hash:str
                         )->dict[str,float]:
    """Perform an actual fresh SGD optimizer step, then independently score."""
    if lr not in LR_GRID:
        raise ValueError("unregistered learning rate")
    trial=deepcopy(base).eval()
    if _weights_digest(trial)!=baseline_hash:
        raise AssertionError("native starting checkpoint changed")
    trial.zero_grad(set_to_none=True)
    for param,g in zip(trial.parameters(),report_grad):
        param.grad=g.detach().clone()
    opt=torch.optim.SGD(trial.parameters(),lr=lr,
                        weight_decay=0.0,momentum=0.0)
    opt.step()
    if _weights_digest(trial)==baseline_hash:
        raise AssertionError("physical SGD failed to alter weights")
    with torch.no_grad():
        scores={name:float(panel_ce(trial,panel))
                for name,panel in panels.items()}
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("nonfinite SGD successor CE")
    return scores


def physical_report(base:EvaluationBeliefLooper,
                    row:Query,panels:dict[str,Any],
                    gold_grads:dict[str,tuple[Tensor,...]],
                    baselines:dict[str,float],base_hash:str
                    )->dict[str,Any]:
    report_grads,norms=query_gradients(base,row)
    different=subtract(report_grads["flip"],report_grads["truth"])
    result={
        "uid":row.uid,
        "exact_gold_label":row.label,
        "prior":str(row.prior),
        "ratio":str(row.ratio),
        "bits":list(row.bits),
        "raw_report_gradient_norms":norms,
        "clipped_report_gradient_norms":{
            a:norm64(g) for a,g in report_grads.items()
        },
        "truth_flip_direction_cosine":cosine64(
            report_grads["truth"],report_grads["flip"]),
        "direction_gap_norm":norm64(different),
        "panel_readouts":{},
    }
    for lr in LR_GRID:
        first_order={
            name:-lr*dot64(gold_grads[name],different)
            for name in panels
        }
        truth=physical_step_scores(base,report_grads["truth"],
                                   lr,panels,base_hash)
        flip=physical_step_scores(base,report_grads["flip"],
                                  lr,panels,base_hash)
        result["panel_readouts"][key_eta(lr)]={
            name:{
                "truth_post_sgd_ce":truth[name],
                "flip_post_sgd_ce":flip[name],
                "truth_delta":truth[name]-baselines[name],
                "flip_delta":flip[name]-baselines[name],
                "gap":flip[name]-truth[name],
                "linear_gap":first_order[name],
                "curvature_residual":(
                    flip[name]-truth[name]-first_order[name]),
            } for name in panels
        }
    return result


def classify_pair(four:dict[str,float],six:dict[str,float],
                  *,epsilon:float=SIGN_EPS)->dict[str,Any]:
    """Decision labels follow the registered exact finite/linear sign rules."""
    f4,f6=four["gap"],six["gap"]
    l4,l6=four["linear_gap"],six["linear_gap"]
    s4,s6=sign(f4,epsilon),sign(f6,epsilon)
    t4,t6=sign(l4,epsilon),sign(l6,epsilon)
    actual_reversal=(s4*s6==-1)
    first_explains=actual_reversal and s4==t4 and s6==t6
    return {
        "measured_signs":[s4,s6],
        "first_order_signs":[t4,t6],
        "measured_reversal":actual_reversal,
        "linear_reversal":t4*t6==-1,
        "reversal_first_order_explained":first_explains,
        "reversal_curvature_implicated":actual_reversal and not first_explains,
        "critical_finite_gold_mix":mixture_threshold(f4,f6,epsilon),
        "critical_linear_gold_mix":mixture_threshold(l4,l6,epsilon),
    }


def _percent(n:int,d:int)->float | None:
    return n/d if d else None


def summarize_rows(rows:list[dict[str,Any]],
                   four:str,six:str)->dict[str,Any]:
    if not rows:
        raise ValueError("cannot summarize empty cohort")
    full=key_eta(LR_GRID[-1])
    entries=[]
    for row in rows:
        data=row["panel_readouts"][full]
        entries.append(classify_pair(data[four],data[six]))
    n=len(rows)
    reversal=sum(e["measured_reversal"] for e in entries)
    linear=sum(e["linear_reversal"] for e in entries)
    explained=sum(e["reversal_first_order_explained"] for e in entries)
    curvature=sum(e["reversal_curvature_implicated"] for e in entries)
    ambiguous=sum(0 in e["measured_signs"] for e in entries)
    if explained+curvature!=reversal:
        raise AssertionError("reversal classification incomplete")
    critical=[e["critical_finite_gold_mix"] for e in entries
              if e["critical_finite_gold_mix"] is not None]
    linear_critical=[e["critical_linear_gold_mix"] for e in entries
                     if e["critical_linear_gold_mix"] is not None]
    properties={}
    for name in (four,six):
        full_data=[r["panel_readouts"][full][name] for r in rows]
        residuals=[abs(d["curvature_residual"]) for d in full_data]
        agreement=sum(
            sign(d["gap"])!=0 and
            sign(d["gap"])==sign(d["linear_gap"])
            for d in full_data)
        nonzero=sum(sign(d["gap"])!=0 for d in full_data)
        linear_zeros=sum(sign(d["linear_gap"])==0 for d in full_data)
        stable=changed=rate_ambiguous=0
        for row in rows:
            signs=[sign(row["panel_readouts"][key_eta(lr)][name]["gap"])
                   for lr in LR_GRID]
            if 0 in signs:
                rate_ambiguous+=1
            elif len(set(signs))==1:
                stable+=1
            else:
                changed+=1
        quadratic=[]
        for row in rows:
            half=row["panel_readouts"][key_eta(LR_GRID[1])][name][
                "curvature_residual"]
            whole=row["panel_readouts"][full][name]["curvature_residual"]
            if abs(whole)>SIGN_EPS and math.isfinite(half):
                quadratic.append(half/whole)
        properties[name]={
            "finite_nonzero":nonzero,
            "linear_zero_or_ambiguous":linear_zeros,
            "finite_linear_sign_agreement_count":agreement,
            "finite_linear_sign_agreement_over_nonzero":_percent(agreement,nonzero),
            "median_abs_curvature_residual":float(median(residuals)),
            "mean_abs_curvature_residual":sum(residuals)/n,
            "max_abs_curvature_residual":max(residuals),
            "median_abs_measured_gap":float(median(abs(d["gap"]) for d in full_data)),
            "sign_stable_three_rates":stable,
            "sign_changed_three_rates":changed,
            "sign_ambiguous_three_rates":rate_ambiguous,
            "median_half_to_full_residual_ratio":(
                float(median(quadratic)) if quadratic else None),
            "residual_ratio_samples":len(quadratic),
        }
    return {
        "cohort_size":n,
        "primary_eta":LR_GRID[-1],
        "classification_epsilon":SIGN_EPS,
        "measured_reversal_count":reversal,
        "measured_reversal_fraction":reversal/n,
        "first_order_reversal_count":linear,
        "first_order_explained_measured_reversal_count":explained,
        "finite_residual_implicated_measured_reversal_count":curvature,
        "finite_ambiguous_count":ambiguous,
        "first_order_supported_at_least_half_if_reversals_ge_8":
            reversal>=8 and explained*2>=reversal,
        "curvature_sensitive_at_least_8":curvature>=8,
        "critical_finite_gold_mix_count":len(critical),
        "median_critical_finite_gold_mix":(
            float(median(critical)) if critical else None),
        "critical_linear_gold_mix_count":len(linear_critical),
        "median_critical_linear_gold_mix":(
            float(median(linear_critical)) if linear_critical else None),
        "per_panel":properties,
    }


def run_gg05(*,output:Path,artifacts:Path,
             seeds:tuple[int,int]=MODEL_SEEDS,
             train_steps:int=400)->dict[str,Any]:
    if seeds!=MODEL_SEEDS or train_steps!=400:
        raise ValueError("GG-05 requires frozen seed roster and 400 steps")
    torch.set_num_threads(1)
    rows=query_splits()["heldout"]
    panels={name:make_panel(GOLD_EXAMPLES,depth,seed)
            for name,depth,seed in GOLD_SPECS}
    artifacts.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    root={
        "protocol":"RMP-GG-05",
        "source_main_commit":BASE_SHA,
        "experiment_registered_before_new_gold_readout":True,
        "known_gg04b_heldout_query_roster_reused":True,
        "query_manifest_sha256":_rows_hash(rows),
        "query_count":len(rows),
        "gold_panel_hashes":{name:_panel_hash(panel)
                             for name,panel in panels.items()},
        "gold_panel_definitions":{
            name:{"visits":depth,"seed":seed,"examples":GOLD_EXAMPLES,
                  "is_new_scoring_panel":name.startswith("fresh")}
            for name,depth,seed in GOLD_SPECS},
        "learning_rates":[*LR_GRID],
        "global_report_gradient_clip":CLIP,
        "finite_sign_epsilon":SIGN_EPS,
        "actual_native_SGD":True,
        "actual_model_introspection_claim":False,
        "results":[],
    }
    for seed in seeds:
        base,training=train_cell(seed,steps=train_steps,save_dir=artifacts)
        base.eval()
        start_hash=_weights_digest(base)
        gold_grads,baselines=panel_gradients(base,panels)
        gold_cosine={
            "fresh4_fresh6":cosine64(gold_grads["fresh4"],gold_grads["fresh6"]),
            "old4_old6":cosine64(gold_grads["old4"],gold_grads["old6"]),
        }
        if any(x is None for x in gold_cosine.values()):
            raise ValueError("gold gradient has degenerate norm")
        observations=[]
        for row in rows:
            observations.append(physical_report(base,row,panels,
                         gold_grads,baselines,start_hash))
        if start_hash!=_weights_digest(base):
            raise AssertionError("GG-05 task source checkpoint mutated")
        # Actual gold-gradient dot directions for every query.
        alignment={}
        for four,six,tag in (
            ("fresh4","fresh6","fresh"),("old4","old6","historical")):
            diag=[classify_pair(
                r["panel_readouts"][key_eta(LR_GRID[-1])][four],
                r["panel_readouts"][key_eta(LR_GRID[-1])][six])
                for r in observations]
            alignment[tag]={
                "nonzero_opposed_report_projections":sum(
                    a["linear_reversal"] for a in diag),
                "total_registered_reports":len(rows),
            }
        summaries={
            "fresh":summarize_rows(observations,"fresh4","fresh6"),
            "historical":summarize_rows(observations,"old4","old6"),
        }
        rows_path=artifacts/f"gg05_seed{seed}_physical_decomposition.json"
        rows_path.write_text(json.dumps({
            "seed":seed,"source_model_sha256":start_hash,
            "gold_panel_hashes":root["gold_panel_hashes"],
            "query_manifest_sha256":root["query_manifest_sha256"],
            "baseline_gold_cross_entropy":baselines,
            "physical_query_readouts":observations,
        },sort_keys=True,indent=2,allow_nan=False)+"\n")
        root["results"].append({
            "seed":seed,
            "base_training":training,
            "base_weight_sha256":start_hash,
            "gold_baseline_loss":baselines,
            "gold_gradient_cosines":gold_cosine,
            "report_direction_alignments":alignment,
            "fresh_gold_summary":summaries["fresh"],
            "historical_gold_summary":summaries["historical"],
            "total_physical_sgd_steps":2*len(rows)*len(LR_GRID),
            "total_finite_gold_panel_evaluations":
                2*len(rows)*len(LR_GRID)*len(panels),
            "source_weight_unchanged":True,
            "decomposition_filename":rows_path.name,
            "decomposition_sha256":sha256(rows_path.read_bytes()).hexdigest(),
        })
    fresh=root["results"]
    root["first_order_conflict_two_seed_gate"]=all(
        r["fresh_gold_summary"][
            "first_order_supported_at_least_half_if_reversals_ge_8"]
        for r in fresh
    )
    root["any_curvature_sensitive_seed"]=any(
        r["fresh_gold_summary"]["curvature_sensitive_at_least_8"]
        for r in fresh
    )
    output.write_text(json.dumps(root,sort_keys=True,indent=2,
                                 allow_nan=False)+"\n")
    return root
