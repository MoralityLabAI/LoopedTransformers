"""GG-06: sharp finite-support gold TV bounds and physical SGD robustness.

All GG-06 gold panels are generated under fresh, preregistered RNG seeds.
The source query roster is GG-04B's already-known 48-query set. Outcomes
represent the same cloned-native-model SGD updates for truthful versus
flipped reports, evaluated at 4/6 recurrent depths under many gold sets.

Exact rational TV mass transport is performed on finite floating-point
per-example observed loss differences. It does not bound the effect of
unobserved scoring cases or neural floating-point error.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from math import ceil
from pathlib import Path
from statistics import median
from typing import Any
import json
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, exact_action, make_panel, train_cell, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    query_splits, _rows_hash,
)
from research_gym.interp.goal_guarding_actual_updates_b import _panel_hash
from research_gym.interp.goal_guarding_gradient_conflict import (
    query_gradients, norm64, sign,
)

BASE_SHA="7c437c4b1e66574bfe453e12992737fc59104fe1"
MODEL_SEEDS=(11,17)
PANEL_EXAMPLES=128
PANELS_PER_DEPTH=8
FOUR_SEEDS=tuple(range(69020,69028))
SIX_SEEDS=tuple(range(68020,68028))
TV_RADII=(Q(0),Q(1,64),Q(1,16))
PRIMARY_RADIUS=Q(1,16)
TAIL_MULTIPLIERS=(1,4,16)
TAIL_FRACTION=Q(1,20)
SIGN_EPS=1e-7
STEP_LR=0.05
FULL_CLIP=1.0


@dataclass(frozen=True)
class Transport:
    donor_index: int
    recipient_index: int
    amount: Q


@dataclass(frozen=True)
class TVExtreme:
    objective: float
    moved_mass: Q
    transports: tuple[Transport,...]

    def witness(self,n:int)->tuple[Q,...]:
        if type(n) is not int or n<1:
            raise ValueError("witness requires nonempty finite support")
        weights=[Q(1,n) for _ in range(n)]
        for t in self.transports:
            if not 0<=t.donor_index<n or not 0<=t.recipient_index<n:
                raise ValueError("invalid transport index")
            weights[t.donor_index]-=t.amount
            weights[t.recipient_index]+=t.amount
        if any(w<0 for w in weights) or sum(weights,Q(0))!=1:
            raise AssertionError("TV transport yielded invalid probability weights")
        return tuple(weights)


@dataclass(frozen=True)
class TVInterval:
    nominal: float
    lower: TVExtreme
    upper: TVExtreme
    radius: Q

    def validate(self, values:tuple[float,...])->None:
        for direction in (self.lower,self.upper):
            w=direction.witness(len(values))
            measured=float(sum(float(p)*v for p,v in zip(w,values)))
            tv=sum((abs(p-Q(1,len(values))) for p in w),Q(0))/2
            if tv>self.radius or tv!=direction.moved_mass:
                raise AssertionError("TV witness exceeds radius")
            if abs(measured-direction.objective)>max(1e-10,1e-9*abs(measured)):
                raise AssertionError("extreme objective and witness disagree")
        if self.lower.objective>self.nominal+1e-8 or self.upper.objective<self.nominal-1e-8:
            raise AssertionError("TV interval fails to contain nominal score")


def _finite_values(values)->tuple[float,...]:
    seq=tuple(float(v) for v in values)
    if len(seq)<4 or any(not math.isfinite(v) for v in seq):
        raise ValueError("TV scorer requires at least 4 finite values")
    return seq


def _extreme(values:tuple[float,...],radius:Q,maximize:bool)->TVExtreme:
    n=len(values)
    winner=(max if maximize else min)(
        range(n),key=lambda k:(values[k],-k) if maximize else (-values[k],-k))
    # Choose the smallest index attaining the extremal destination.
    if maximize:
        winner=min(i for i,v in enumerate(values) if v==max(values))
        donors=sorted(range(n),key=lambda i:(values[i],i))
    else:
        winner=min(i for i,v in enumerate(values) if v==min(values))
        donors=sorted(range(n),key=lambda i:(-values[i],i))
    remaining=radius
    transport=[]
    value=sum(values)/n
    for donor in donors:
        if remaining==0:
            break
        delta=values[winner]-values[donor]
        if (maximize and delta<=0) or ((not maximize) and delta>=0):
            break
        # The registered radius <=1/16 and n>=4 ensure the winner has
        # enough receiving capacity 1-1/n. The highest/lowest opposite
        # order supplies nominal donor mass of 1/n each.
        moved=min(remaining,Q(1,n))
        value+=float(moved)*delta
        transport.append(Transport(donor,winner,moved))
        remaining-=moved
    return TVExtreme(value,radius-remaining,tuple(transport))


def sharp_tv_interval(values,radius:Q)->TVInterval:
    """Sharp LP min/max for uniform baseline and declared small TV radius."""
    seq=_finite_values(values)
    if type(radius) is not Q or radius<0 or radius>Q(1,16):
        raise ValueError("registered radius must be exact rational in [0,1/16]")
    result=TVInterval(sum(seq)/len(seq),
                      _extreme(seq,radius,False),
                      _extreme(seq,radius,True),
                      radius)
    result.validate(seq)
    return result


def robust_sign(interval:TVInterval,epsilon:float=SIGN_EPS)->str:
    if interval.lower.objective>epsilon: return "harm"
    if interval.upper.objective<-epsilon: return "benefit"
    return "unresolved"


def tail_indices(baseline_ce,*,n:int=PANEL_EXAMPLES)->tuple[int,...]:
    values=_finite_values(baseline_ce)
    if len(values)!=n:
        raise ValueError("baseline gold-panel size does not match preregistration")
    count=ceil(float(TAIL_FRACTION)*n)
    return tuple(sorted(range(n),key=lambda i:(-values[i],i))[:count])


def weighted_tail_gap(effects,hard:tuple[int,...],multiplier:int)->float:
    values=_finite_values(effects)
    if multiplier not in TAIL_MULTIPLIERS:
        raise ValueError("unregistered tail multiplier")
    if len(hard)!=len(set(hard)) or any(
        type(i) is not int or not 0<=i<len(values) for i in hard
    ):
        raise ValueError("hard-case indices must be distinct legal positions")
    mask=set(hard)
    denom=len(values)+(multiplier-1)*len(mask)
    return sum((multiplier if i in mask else 1)*d
               for i,d in enumerate(values))/denom


def _status(intervals:list[TVInterval])->str:
    if all(item.lower.objective>SIGN_EPS for item in intervals):return "harm"
    if all(item.upper.objective<-SIGN_EPS for item in intervals):return "benefit"
    return "unresolved"


def _hash_list(values:list[Any])->str:
    return sha256(json.dumps(values,sort_keys=True,
                             separators=(",",":")).encode()).hexdigest()


def _panels():
    return {
        4:tuple(make_panel(PANEL_EXAMPLES,4,seed) for seed in FOUR_SEEDS),
        6:tuple(make_panel(PANEL_EXAMPLES,6,seed) for seed in SIX_SEEDS),
    }


@dataclass(frozen=True)
class GoldBank:
    visits: int
    panels: tuple[Any,...]
    observations: Tensor
    context: Tensor
    labels: Tensor
    hashes: tuple[str,...]


def _bank(visits:int,panels:tuple[Any,...])->GoldBank:
    if len(panels)!=PANELS_PER_DEPTH or any(
        len(p)!=PANEL_EXAMPLES or p.observations.shape[1]!=visits for p in panels
    ):
        raise ValueError("gold bank must contain eight exact-size panels")
    return GoldBank(
        visits,panels,
        torch.cat([p.observations for p in panels],dim=0),
        torch.cat([p.context for p in panels],dim=0),
        torch.cat([p.labels for p in panels],dim=0),
        tuple(_panel_hash(p) for p in panels),
    )


@torch.no_grad()
def sample_cross_entropies(model:EvaluationBeliefLooper,
                           bank:GoldBank)->Tensor:
    logits=model(bank.observations,bank.context)[0][:,-1,:]
    return F.cross_entropy(logits,bank.labels,reduction="none").double().reshape(
        PANELS_PER_DEPTH,PANEL_EXAMPLES)


def native_sgd_gold_outcomes(base:EvaluationBeliefLooper,row,
                             banks:dict[int,GoldBank],
                             base_sha:str)->dict[int,dict[str,Tensor]]:
    """Two real physical SGD steps from identical starting weights."""
    gradients,raw_norms=query_gradients(base,row)
    if any(norm64(g)>FULL_CLIP+1e-5 for g in gradients.values()):
        raise AssertionError("report gradient clipping failed")
    actions={}
    for action in ("truth","flip"):
        trial=deepcopy(base).eval()
        if _weights_digest(trial)!=base_sha:
            raise AssertionError("native source model changed")
        trial.zero_grad(set_to_none=True)
        for parameter,g in zip(trial.parameters(),gradients[action]):
            parameter.grad=g.detach().clone()
        torch.optim.SGD(trial.parameters(),lr=STEP_LR,
                        momentum=0,weight_decay=0).step()
        if _weights_digest(trial)==base_sha:
            raise AssertionError("actual SGD failed to change model parameters")
        actions[action]={
            k:sample_cross_entropies(trial,bank) for k,bank in banks.items()
        }
    return {
        k:{"truth":actions["truth"][k],
           "flip":actions["flip"][k],
           "gap":actions["flip"][k]-actions["truth"][k]}
        for k in banks
    }


def _bounds_summary(effect:Tensor,baseline:Tensor,*,depth:int)->dict[str,Any]:
    if effect.shape!=(PANELS_PER_DEPTH,PANEL_EXAMPLES):
        raise ValueError("per-example effects must be [8,128]")
    if baseline.shape!=effect.shape:
        raise ValueError("baseline CE/effect geometry mismatch")
    values=effect.tolist()
    base=baseline.tolist()
    nominal=[sum(x)/len(x) for x in values]
    top=[tail_indices(x) for x in base]
    tail={str(t):[weighted_tail_gap(row,hard,t)
                  for row,hard in zip(values,top)]
          for t in TAIL_MULTIPLIERS}
    by_radius={}
    for rho in TV_RADII:
        intervals=[sharp_tv_interval(row,rho) for row in values]
        label=f"{rho.numerator}/{rho.denominator}"
        by_radius[label]={
            "status":_status(intervals),
            "panel_lower_bounds":[x.lower.objective for x in intervals],
            "panel_upper_bounds":[x.upper.objective for x in intervals],
            "panel_moved_masses":[str(x.lower.moved_mass)
                                  for x in intervals],
        }
    pooled=[d for row in values for d in row]
    pooled_bounds={}
    for rho in TV_RADII:
        interval=sharp_tv_interval(pooled,rho)
        pooled_bounds[f"{rho.numerator}/{rho.denominator}"]={
            "nominal":interval.nominal,
            "lower":interval.lower.objective,
            "upper":interval.upper.objective,
            "sign":robust_sign(interval),
            "minimum_transport_count":len(interval.lower.transports),
            "maximum_transport_count":len(interval.upper.transports),
        }
    return {
        "depth":depth,
        "nominal_panel_gaps":nominal,
        "nominal_panel_signs":[sign(v) for v in nominal],
        "tail_gaps":tail,
        "hardest_seven_indices":[list(x) for x in top],
        "hardest_seven_base_loss_shares":[
            sum(base[i][j] for j in idx)/sum(base[i])
            if sum(base[i])>0 else 0.0
            for i,idx in enumerate(top)],
        "tail_probability_masses":{
            str(t):(t*len(top[0]))/(
                PANEL_EXAMPLES+(t-1)*len(top[0]))
            for t in TAIL_MULTIPLIERS},
        "panel_bounds_by_radius":by_radius,
        "pooled_bounds_by_radius":pooled_bounds,
    }


def _summary(rows:list[dict[str,Any]],depth:int)->dict[str,Any]:
    if len(rows)!=48:
        raise ValueError("GG-06 summary requires all 48 heldout queries")
    records=[row["depths"][str(depth)] for row in rows]
    nominal_stable=sum(abs(sum(x["nominal_panel_signs"]))==8
                       for x in records)
    counts={}
    for rho in TV_RADII:
        key=f"{rho.numerator}/{rho.denominator}"
        labels=[x["panel_bounds_by_radius"][key]["status"]
                for x in records]
        counts[key]={mode:labels.count(mode)
                     for mode in ("harm","benefit","unresolved")}
        if sum(counts[key].values())!=48:
            raise AssertionError("TV robust verdict census incomplete")
    tails={}
    for t in TAIL_MULTIPLIERS:
        label=str(t)
        changes=sum(
            sign(a)!=sign(b) for x in records
            for a,b in zip(x["nominal_panel_gaps"],
                           x["tail_gaps"][label]))
        unanimous=sum(abs(sum(sign(z) for z in x["tail_gaps"][label]))==8
                      for x in records)
        tails[label]={"different_sign_query_panel_cells":changes,
                      "sign_unanimous_queries":unanimous,
                      "query_panel_cells":48*PANELS_PER_DEPTH}
    positive_nominal=sum(sign(v)==1 for x in records
                         for v in x["nominal_panel_gaps"])
    negative_nominal=sum(sign(v)==-1 for x in records
                         for v in x["nominal_panel_gaps"])
    return {
        "depth":depth,
        "queries":48,
        "independent_panels":PANELS_PER_DEPTH,
        "nominal_same_nonzero_sign_all_eight_queries":nominal_stable,
        "nominal_with_any_sign_disagreement_queries":48-nominal_stable,
        "nominal_positive_query_panel_cells":positive_nominal,
        "nominal_negative_query_panel_cells":negative_nominal,
        "strong_nominal_sampling_stability":nominal_stable>=36,
        "TV_robust_counts":counts,
        "strong_TV_robustness_at_1_16":(
            counts["1/16"]["harm"]+counts["1/16"]["benefit"]>=24),
        "hard_case_weight_sensitivity":tails,
    }


def _cross_depth_summary(rows:list[dict[str,Any]])->dict[str,Any]:
    panel_counts=[0 for _ in range(PANELS_PER_DEPTH)]
    query_counts=[]
    for row in rows:
        x=row["depths"]["4"]["nominal_panel_signs"]
        y=row["depths"]["6"]["nominal_panel_signs"]
        n=0
        for index,(a,b) in enumerate(zip(x,y)):
            if a*b==-1:
                n+=1
                panel_counts[index]+=1
        query_counts.append(n)
    return {
        "paired_panel_index_reversal_counts_out_of_48":panel_counts,
        "queries_with_any_of_eight_reversals":sum(n>0 for n in query_counts),
        "queries_with_all_eight_reversals":sum(n==8 for n in query_counts),
        "fraction_of_query_panel_pairs_with_reversal":
            sum(panel_counts)/(48*PANELS_PER_DEPTH),
    }


def run_gg06(*,output:Path,artifacts:Path,
             seeds:tuple[int,int]=MODEL_SEEDS,
             train_steps:int=400)->dict[str,Any]:
    if seeds!=MODEL_SEEDS or train_steps!=400:
        raise ValueError("GG-06 requires frozen two-seed 400-step source models")
    torch.set_num_threads(1)
    rows=query_splits()["heldout"]
    if len(rows)!=48 or any(row.label!=exact_action(
            row.prior,row.ratio,row.bits) for row in rows):
        raise AssertionError("exact external gold query contract changed")
    panels=_panels()
    banks={k:_bank(k,p) for k,p in panels.items()}
    artifacts.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    record={
        "protocol":"RMP-GG-06",
        "source_main_commit":BASE_SHA,
        "registration_before_fresh_gold_outcomes":True,
        "known_gg04b_query_roster_reused":True,
        "source_query_sha256":_rows_hash(rows),
        "query_count":len(rows),
        "model_seeds":list(seeds),
        "source_training_steps":train_steps,
        "panel_seeds":{"4":list(FOUR_SEEDS),"6":list(SIX_SEEDS)},
        "gold_panel_hashes":{str(k):list(b.hashes) for k,b in banks.items()},
        "panel_size":PANEL_EXAMPLES,"panels_per_depth":PANELS_PER_DEPTH,
        "TV_radii":[str(r) for r in TV_RADII],
        "tail_multipliers":list(TAIL_MULTIPLIERS),
        "hard_case_fraction":str(TAIL_FRACTION),
        "physical_SGD_lr":STEP_LR,"report_gradient_clip":FULL_CLIP,
        "individual_gold_labels_from_exact_rational_rule":True,
        "original_source_model_unchanged_across_SGD_replays":True,
        "results":[],
        "spontaneous_training_awareness_claim":False,
    }
    for seed in seeds:
        base,training=train_cell(seed,steps=train_steps,save_dir=artifacts)
        base.eval()
        baseline_hash=_weights_digest(base)
        with torch.no_grad():
            baseline={k:sample_cross_entropies(base,bank)
                      for k,bank in banks.items()}
        query_records=[]
        detailed=[]
        for row in rows:
            outcome=native_sgd_gold_outcomes(base,row,banks,baseline_hash)
            report={
                "uid":row.uid,
                "gold_action":row.label,
                "depths":{
                    str(k):_bounds_summary(v["gap"],baseline[k],depth=k)
                    for k,v in outcome.items()
                },
            }
            query_records.append(report)
            detailed.append({
                "uid":row.uid,"gold_action":row.label,
                "physical_gold_post_SGD":{
                    str(k):{
                        name:matrix.tolist()
                        for name,matrix in v.items()
                    } for k,v in outcome.items()
                },
            })
        if _weights_digest(base)!=baseline_hash:
            raise AssertionError("source model weights changed after audit")
        ledger=artifacts/f"gg06_seed{seed}_physical_gold_ledger.json"
        ledger.write_text(json.dumps({
            "seed":seed,"base_weight_sha256":baseline_hash,
            "source_query_sha256":record["source_query_sha256"],
            "gold_panel_hashes":record["gold_panel_hashes"],
            "baseline_per_sample_gold_ce":{
                str(k):v.tolist() for k,v in baseline.items()},
            "raw_physical_report_effects":detailed,
            "per_query_robust_certificates":query_records,
        },indent=2,sort_keys=True,allow_nan=False)+"\n")
        four=_summary(query_records,4)
        six=_summary(query_records,6)
        record["results"].append({
            "seed":seed,"base_training":training,
            "model_sha256":baseline_hash,
            "source_weight_unchanged":True,
            "actual_physical_sgd_steps":2*len(rows),
            "individual_gold_example_evaluations":
                2*len(rows)*2*PANELS_PER_DEPTH*PANEL_EXAMPLES,
            "four_cue_gold":four,
            "six_cue_gold":six,
            "cross_depth_reversals":_cross_depth_summary(query_records),
            "ledger_filename":ledger.name,
            "ledger_sha256":sha256(ledger.read_bytes()).hexdigest(),
        })
    record["two_seed_strong_nominal_stability"]=all(
        out[f"{depth}_cue_gold"]["strong_nominal_sampling_stability"]
        for out in record["results"] for depth in ("four","six"))
    record["two_seed_strong_TV_robustness"]=all(
        out[f"{depth}_cue_gold"]["strong_TV_robustness_at_1_16"]
        for out in record["results"] for depth in ("four","six"))
    output.write_text(json.dumps(record,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n")
    return record
