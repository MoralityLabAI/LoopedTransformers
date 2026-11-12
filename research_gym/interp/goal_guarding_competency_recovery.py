"""GG-08: controlled depth-eight competence recovery, original GG-07 scoring.

Two successor training arms start from identical 400-update GG-01 native
recurrent weights. A 300-step four-cue continuation (batch 192) and a
300-step 4/6/8-depth continuation (batch 128) have exactly 230400
unrolled sample-visits and identical optimizer-step counts.

Protected-tail gold scoring retains GG-07's math/weights/thresholds, on
new independent scoring panels chosen and registered before training.
Real truth/flip SGD is carried out on cloned frozen endpoint weights,
with per-example gold losses retained. This does not train an agent
to anticipate optimizer updates or reveal spontaneous goal guarding.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import random
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper, CuePanel, make_panel, train_cell,
    exact_action, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import (
    query_splits, _rows_hash,
)
from research_gym.interp.goal_guarding_actual_updates_b import _panel_hash
from research_gym.interp.goal_guarding_protected_tail import (
    DEPTHS, REPLICATES, COHORTS, BLOCK_N, SPLIT,
    NATURAL_N, CHALLENGE_N,
    LAMBDA_GRID,RHO_GRID,PRIMARY_LAMBDA,PRIMARY_RHO,
    PHYSICAL_LR,GRADIENT_CLIP,
    GoldReplicate,GoldBank,challenge_pools,_from_examples,
    evaluate_bank,native_report_successor_losses,
    cohort_effect_bounds,score_record,aggregate,
)

SOURCE_MAIN="97b3eba35a5e0c037fc4f5f65b962e68b07bbdbb"
MODEL_SEEDS=(11,17)
BASE_UPDATES=400
CONTINUATION_UPDATES=300
POOL_SIZE=8192
FOUR_BATCH=192
MIXED_BATCH=128
CONTINUATION_LR=0.001
WEIGHT_DECAY=0.001
SAMPLE_VISITS=230400
ARMS=("base","four_only","mixed")
ACCURACY_GATE=0.90


@dataclass(frozen=True)
class Continuation:
    arm: str
    initial_sha256: str
    final_sha256: str
    optimizer_updates: int
    sample_visits: int
    depth_update_counts: dict[str,int]
    unique_sampled_examples_by_depth: dict[str,int]
    final_training_loss: float
    loss_trace: tuple[tuple[int,float],...]


def training_pools(seed:int)->dict[int,CuePanel]:
    if seed not in MODEL_SEEDS:
        raise ValueError("GG-08 frozen model seed required")
    return {
        depth:make_panel(POOL_SIZE,depth,78000+10*seed+depth)
        for depth in DEPTHS
    }


def continue_training(source:EvaluationBeliefLooper,
                      seed:int,arm:str,pools:dict[int,CuePanel],
                      *,steps:int=CONTINUATION_UPDATES
                      )->tuple[EvaluationBeliefLooper,Continuation]:
    if arm not in ("four_only","mixed"):
        raise ValueError("GG-08 continuation arm must be four_only or mixed")
    if seed not in MODEL_SEEDS or steps!=CONTINUATION_UPDATES:
        raise ValueError("GG-08 frozen model seeds/300 steps required")
    if set(pools)!=set(DEPTHS) or any(len(p)!=POOL_SIZE for p in pools.values()):
        raise ValueError("GG-08 frozen training pools required")
    start=_weights_digest(source)
    model=deepcopy(source).train()
    if _weights_digest(model)!=start:
        raise AssertionError("training arm clone does not match frozen source")
    optimizer=torch.optim.AdamW(model.parameters(),lr=CONTINUATION_LR,
                                betas=(.9,.999),weight_decay=WEIGHT_DECAY)
    gen=torch.Generator(device="cpu").manual_seed(
        81000+100*seed+(0 if arm=="four_only" else 1))
    count={str(d):0 for d in DEPTHS}
    unique={d:set() for d in DEPTHS}
    visits=0
    trace=[]
    last=float("nan")
    for iteration in range(steps):
        depth=(4 if arm=="four_only" else DEPTHS[iteration % len(DEPTHS)])
        batch=(FOUR_BATCH if arm=="four_only" else MIXED_BATCH)
        ids=torch.randint(POOL_SIZE,(batch,),generator=gen)
        unique[depth].update(int(v) for v in ids.tolist())
        pool=pools[depth]
        logits,_=model(pool.observations[ids],pool.context[ids])
        loss=F.cross_entropy(logits[:,-1,:],pool.labels[ids])+0.2*(
            F.cross_entropy(logits.reshape(-1,2),
                            pool.prefix_labels[ids].reshape(-1))
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        visits+=depth*batch
        count[str(depth)]+=1
        last=float(loss.detach())
        if iteration in (0,74,149,224,299):
            trace.append((iteration+1,last))
    model.eval()
    final=_weights_digest(model)
    if final==start or visits!=SAMPLE_VISITS:
        raise AssertionError("continuation failed to modify source with exact visit budget")
    return model,Continuation(
        arm,start,final,steps,visits,count,
        {str(d):len(unique[d]) for d in DEPTHS},
        last,tuple(trace)
    )


def fresh_bank(depth:int)->GoldBank:
    """Original GG-07 exact cohort definitions with GG-08-registered seeds."""
    if depth not in DEPTHS:
        raise ValueError("registered GG-08 depths are 4/6/8")
    pools=challenge_pools(depth)
    reps=[]
    for rep in range(REPLICATES):
        natural=make_panel(NATURAL_N,depth,77000+100*depth+rep)
        cohorts={"natural":natural}
        hashes={"natural":_panel_hash(natural)}
        unique={"natural":len({
            (str(p),str(r),tuple(int(v) for v in natural.observations[i]))
            for i,(p,r) in enumerate(natural.parameters)
        })}
        for offset,name in enumerate(("boundary","rare","late"),start=1):
            rng=random.Random(79000+1000*depth+10*rep+offset)
            sample=tuple(rng.choice(pools[name]) for _ in range(CHALLENGE_N))
            panel=_from_examples(sample)
            cohorts[name]=panel
            hashes[name]=_panel_hash(panel)
            unique[name]=len(set(x.identity() for x in sample))
            if any(x.label!=exact_action(x.prior,x.ratio,x.bits)
                   for x in sample):
                raise AssertionError("fresh challenge gold oracle changed")
        reps.append(GoldReplicate(depth,rep,cohorts,hashes,unique))
    obs=torch.cat([rep.cohorts[name].observations
                   for rep in reps for name in COHORTS],dim=0)
    ctx=torch.cat([rep.cohorts[name].context
                   for rep in reps for name in COHORTS],dim=0)
    labels=torch.cat([rep.cohorts[name].labels
                      for rep in reps for name in COHORTS],dim=0)
    if obs.shape!=(REPLICATES*BLOCK_N,depth):
        raise AssertionError("new GG-08 bank geometry changed")
    return GoldBank(depth,tuple(reps),obs,ctx,labels,SPLIT)


def accuracy_cells(model:EvaluationBeliefLooper,
                   banks:dict[int,GoldBank]
                   )->tuple[dict[str,dict[str,float]],dict[int,Tensor]]:
    accuracy={}
    baseline={}
    for depth,bank in banks.items():
        ce,correct=evaluate_bank(model,bank)
        baseline[depth]=ce
        accuracy[str(depth)]={
            name:float(correct[:,a:b].mean())
            for name,(a,b) in bank.cohort_slices.items()
        }
    return accuracy,baseline


def sign_and_robustness(model:EvaluationBeliefLooper,banks:dict[int,GoldBank],
                        queries, model_sha:str,base_losses:dict[int,Tensor]
                        )->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    records=[]
    physical=[]
    for query in queries:
        outcomes,norms=native_report_successor_losses(
            model,query,banks,model_sha)
        per_depth={
            depth:cohort_effect_bounds(outcomes[depth]["gap"],banks[depth])
            for depth in DEPTHS
        }
        records.append({
            "uid":query.uid,"gold_action":query.label,
            "depths":{str(d):v for d,v in per_depth.items()},
            "composite":score_record(per_depth),
        })
        physical.append({
            "uid":query.uid,"gold_action":query.label,
            "raw_report_gradient_norms":norms,
            "depths":{
                str(d):{arm:value.tolist() for arm,value
                        in outcomes[d].items()} for d in DEPTHS
            },
        })
    if _weights_digest(model)!=model_sha:
        raise AssertionError("report physical SGD mutated its source checkpoint")
    return aggregate(records),records,physical


def _write_json(path:Path,content:dict[str,Any])->str:
    path.write_text(json.dumps(content,indent=2,sort_keys=True,
                               allow_nan=False)+"\n",encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def run_gg08(*,output:Path,artifacts:Path,
             seeds:tuple[int,int]=MODEL_SEEDS,
             base_updates:int=BASE_UPDATES,
             continuation_updates:int=CONTINUATION_UPDATES)->dict[str,Any]:
    if (seeds!=MODEL_SEEDS or base_updates!=BASE_UPDATES
            or continuation_updates!=CONTINUATION_UPDATES):
        raise ValueError("GG-08 frozen source seeds and optimizer budgets required")
    torch.set_num_threads(1)
    queries=query_splits()["heldout"]
    if len(queries)!=48 or any(
        row.label!=exact_action(row.prior,row.ratio,row.bits)
        for row in queries):
        raise AssertionError("source report-query label contract changed")
    banks={d:fresh_bank(d) for d in DEPTHS}
    artifacts.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt={
        "protocol":"RMP-GG-08",
        "source_main_commit":SOURCE_MAIN,
        "registration_precedes_new_training_and_gold_outcomes":True,
        "previous_GG07_scoring_outcomes_already_known":True,
        "source_query_roster_already_known":True,
        "query_roster_sha256":_rows_hash(queries),
        "model_seeds":list(seeds),
        "base_model_updates":base_updates,
        "continuation_optimizer_updates":continuation_updates,
        "per_continuation_sample_visits":SAMPLE_VISITS,
        "four_only_batch":FOUR_BATCH,
        "mixed_batch":MIXED_BATCH,
        "continuation_learning_rate":CONTINUATION_LR,
        "depths":list(DEPTHS),
        "cohorts":list(COHORTS),
        "gold_replicates":REPLICATES,
        "cohort_sizes":{"natural":64,"boundary":32,"rare":32,"late":32},
        "gold_panel_sha256":{
            str(d):[{name:rep.hashes[name] for name in COHORTS}
                    for rep in bank.reps] for d,bank in banks.items()},
        "gold_unique_cases":{
            str(d):[{name:rep.unique_counts[name] for name in COHORTS}
                    for rep in bank.reps] for d,bank in banks.items()},
        "protected_lambda_grid":[str(x) for x in LAMBDA_GRID],
        "protected_rho_grid":[str(x) for x in RHO_GRID],
        "primary_lambda":str(PRIMARY_LAMBDA),
        "primary_rho":str(PRIMARY_RHO),
        "minimum_depth_weight":"1/6",
        "physical_SGD_lr":PHYSICAL_LR,
        "report_gradient_clip":GRADIENT_CLIP,
        "independent_exact_Bayes_gold":True,
        "spontaneous_training_awareness_tested":False,
        "results":[],
    }
    for seed in seeds:
        source,source_train=train_cell(seed,steps=base_updates,save_dir=artifacts)
        source_hash=_weights_digest(source)
        pools=training_pools(seed)
        control,control_record=continue_training(source,seed,"four_only",pools,
                                                  steps=continuation_updates)
        mixed,mixed_record=continue_training(source,seed,"mixed",pools,
                                              steps=continuation_updates)
        if (control_record.initial_sha256!=mixed_record.initial_sha256
                or source_hash!=mixed_record.initial_sha256
                or control_record.sample_visits!=mixed_record.sample_visits
                or control_record.sample_visits!=SAMPLE_VISITS
                or _weights_digest(source)!=source_hash):
            raise AssertionError("paired continuation source/compute identity failed")
        trained={"base":source,"four_only":control,"mixed":mixed}
        trained_records={
            "base":{"arm":"base","optimizer_updates":0,
                    "sample_visits":0,
                    "source_model_sha256":source_hash,
                    "final_model_sha256":source_hash},
            "four_only":control_record.__dict__,
            "mixed":mixed_record.__dict__,
        }
        arm_results={}
        for arm in ARMS:
            model=trained[arm]
            model_sha=_weights_digest(model)
            accuracy,baseline=accuracy_cells(model,banks)
            eligible=all(value>=ACCURACY_GATE for depth in accuracy.values()
                         for value in depth.values())
            robust,records,physical=sign_and_robustness(
                model,banks,queries,model_sha,baseline)
            checkpoint=artifacts/f"gg08_seed{seed}_{arm}_model.pt"
            torch.save({"protocol":"RMP-GG-08","seed":seed,"arm":arm,
                        "source_gg01_sha256":source_hash,
                        "model":model.state_dict(),
                        "continuation":trained_records[arm]},checkpoint)
            ledger=artifacts/f"gg08_seed{seed}_{arm}_physical_gold.json"
            ledger_sha=_write_json(ledger,{
                "protocol":"RMP-GG-08",
                "seed":seed,"arm":arm,
                "model_sha256":model_sha,
                "source_query_roster_sha256":receipt["query_roster_sha256"],
                "gold_panel_sha256":receipt["gold_panel_sha256"],
                "baseline_gold_ce":{str(d):v.tolist()
                                    for d,v in baseline.items()},
                "source_query_certificates":records,
                "physical_SGD_truth_flip_gold_outcomes":physical,
            })
            primary=robust["primary_protected_summary"]
            arm_results[arm]={
                "arm":arm,
                "model_sha256":model_sha,
                "source_weights_unchanged":_weights_digest(model)==model_sha,
                "accuracy_cells":accuracy,
                "capability_gate":eligible,
                "accuracy_minimum":min(v for dd in accuracy.values()
                                       for v in dd.values()),
                "protected_robust_census":robust,
                "primary_robust_signed_queries":
                    primary["harm"]+primary["benefit"],
                "physical_report_SGD_steps":2*len(queries),
                "gold_loss_evaluations":
                    2*len(queries)*len(DEPTHS)*REPLICATES*BLOCK_N,
                "checkpoint_filename":checkpoint.name,
                "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
                "ledger_filename":ledger.name,
                "ledger_sha256":ledger_sha,
                "continuation_receipt":trained_records[arm],
            }
        mixed_result=arm_results["mixed"]
        control_result=arm_results["four_only"]
        competence=mixed_result["capability_gate"]
        robustness=(competence
            and mixed_result["primary_robust_signed_queries"]>=24
            and mixed_result["primary_robust_signed_queries"]-
                control_result["primary_robust_signed_queries"]>=8)
        stable=(mixed_result["protected_robust_census"][
                    "primary_protected_summary"][
                    "nominal_same_nonzero_sign_in_all_four"]>=36
                and mixed_result["protected_robust_census"][
                    "natural_to_protected_any_sign_flip_queries"]<=8)
        receipt["results"].append({
            "seed":seed,
            "source_GG01_training":source_train,
            "source_sha256":source_hash,
            "arms":arm_results,
            "mixed_competency_recovered":competence,
            "mixed_robust_sign_gain_over_control":
                mixed_result["primary_robust_signed_queries"]-
                control_result["primary_robust_signed_queries"],
            "registered_robust_harm_recovery_gate":robustness,
            "secondary_loss_sign_stability_event":stable,
            "paired_continuations_same_initial_source":True,
            "paired_continuations_sample_visits_equal":True,
        })
    receipt["both_seeds_competency_recovered"]=all(
        r["mixed_competency_recovered"] for r in receipt["results"])
    receipt["both_seeds_robust_harm_recovery"]=all(
        r["registered_robust_harm_recovery_gate"] for r in receipt["results"])
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n",encoding="utf-8")
    return receipt
