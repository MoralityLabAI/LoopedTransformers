"""GG-09: exact two-state HMM Bayes transfer and physical SGD on heldout grammar.

A world changes between observations with probability 1/4. All task labels
are derived by exact rational forward filtering, independently of neural
predictions. This grammar differs from GG-01's fixed-world hypothesis.
Source GG-08 and paired 240-step static/Markov continuations share the
same tied native decoder weights and use disjoint support IDs for new
GG-09 continuation training versus its gold/report holdout.

The product-TV scorer is reused unchanged from GG-07/08. Its certificates
apply to observed gold support; the experiment does not measure autonomous
training awareness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any
import json
import math
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    PRIORS, RATIOS, CuePanel, EvaluationBeliefLooper,
    exact_action, train_cell, _weights_digest,
)
from research_gym.interp.goal_guarding_actual_updates import Query
from research_gym.interp.goal_guarding_actual_updates_b import _panel_hash
from research_gym.interp.goal_guarding_competency_recovery import (
    training_pools as gg08_training_pools,
    continue_training as gg08_continue_training,
)
from research_gym.interp.goal_guarding_protected_tail import (
    DEPTHS, REPLICATES, COHORTS, SPLIT, BLOCK_N,
    NATURAL_N, CHALLENGE_N,
    LAMBDA_GRID, RHO_GRID, PRIMARY_LAMBDA, PRIMARY_RHO,
    GoldReplicate, GoldBank, evaluate_bank,
    native_report_successor_losses, cohort_effect_bounds,
    score_record, aggregate,
)

SOURCE_MAIN="ffee28c83ab8320aa8749b567280b23780dc2a78"
MODEL_SEEDS=(11,17)
SWITCH=Q(1,4)
STATIC=Q(0)
BASE_STEPS=400
GG08_MIXED_STEPS=300
TRANSFER_STEPS=240
TRANSFER_BATCH=128
POOL_SIZE=8192
TRANSFER_LR=.001
TRANSFER_WD=.001
GOLD_ACCURACY_GATE=.9
ARMS=("gg08_source","static_control","markov_transfer")
TOTAL_SAMPLE_VISITS=TRANSFER_BATCH*80*sum(DEPTHS)
EXPECTED_DIFFERENCES={4:35,6:164,8:727}


def forward_masses(prior:Q,bits:tuple[int,...],
                   switch:Q=SWITCH)->tuple[Q,Q]:
    """Exact HMM posterior joint masses after the last emission.

    State E has emission P(1)=3/4; D has P(1)=1/4. Switching occurs
    only between visits, never before the first observation.
    """
    if type(prior) is not Q or not Q(0)<prior<Q(1):
        raise ValueError("prior must be an exact interior rational")
    if type(switch) is not Q or not Q(0)<=switch<=Q(1,2):
        raise ValueError("transition must be a rational probability <= 1/2")
    a,b=prior,1-prior
    for i,bit in enumerate(bits):
        if type(bit) is not int or bit not in (0,1):
            raise ValueError("binary evidence tokens required")
        if i:
            a,b=(1-switch)*a+switch*b, switch*a+(1-switch)*b
        a*=Q(3,4) if bit else Q(1,4)
        b*=Q(1,4) if bit else Q(3,4)
    return a,b


def markov_action(prior:Q,ratio:Q,bits:tuple[int,...],
                  switch:Q=SWITCH)->int:
    if type(ratio) is not Q or ratio<=0:
        raise ValueError("ratio must be a positive exact rational")
    a,b=forward_masses(prior,bits,switch)
    return int(ratio*a>b)


def _id(prior:Q,ratio:Q,bits:tuple[int,...])->str:
    item={"tag":"gg09.markov-input.v1","prior":str(prior),
          "ratio":str(ratio),"bits":list(bits)}
    return sha256(json.dumps(item,sort_keys=True,
           separators=(",",":")).encode()).hexdigest()


@dataclass(frozen=True)
class TaskExample:
    uid: str
    prior: Q
    ratio: Q
    bits: tuple[int,...]
    label: int
    prefix_labels: tuple[int,...]
    likelihood: Q
    margin: float

    @property
    def key(self)->tuple[str,str,tuple[int,...]]:
        return str(self.prior),str(self.ratio),self.bits


@lru_cache(maxsize=None)
def task_grammar(depth:int,switch:Q=SWITCH)->tuple[TaskExample,...]:
    if depth not in DEPTHS or switch not in (STATIC,SWITCH):
        raise ValueError("registered grammar depth or switching probability")
    result=[]
    for prior,ratio in product(PRIORS,RATIOS):
        for bits in product((0,1),repeat=depth):
            bits=tuple(bits)
            a,b=forward_masses(prior,bits,switch)
            result.append(TaskExample(
                _id(prior,ratio,bits),prior,ratio,bits,int(ratio*a>b),
                tuple(markov_action(prior,ratio,bits[:k],switch)
                      for k in range(1,depth+1)),
                a+b, abs(math.log(float(ratio*a/b)))))
    result.sort(key=lambda row:row.uid)
    if len(result)!=9*(2**depth) or len({r.uid for r in result})!=len(result):
        raise AssertionError("finite HMM grammar changed")
    return tuple(result)


def task_split(depth:int,switch:Q=SWITCH
               )->dict[str,tuple[TaskExample,...]]:
    rows=task_grammar(depth,switch)
    cut=2*len(rows)//3
    return {"training":rows[:cut],"heldout":rows[cut:]}


def mismatch_census()->dict[str,dict[str,int]]:
    result={}
    for depth in DEPTHS:
        old=task_grammar(depth,STATIC)
        new=task_grammar(depth,SWITCH)
        if tuple(r.uid for r in old)!=tuple(r.uid for r in new):
            raise AssertionError("static/Markov input ID mismatch")
        count=sum(a.label!=b.label for a,b in zip(old,new))
        if count!=EXPECTED_DIFFERENCES[depth]:
            raise AssertionError("exact HMM grammar disagreement count changed")
        result[str(depth)]={"total":len(old),"disagreements":count}
    return result


def as_panel(rows:tuple[TaskExample,...])->CuePanel:
    if not rows or len({len(x.bits) for x in rows})!=1:
        raise ValueError("task panel needs one positive depth")
    return CuePanel(
        torch.tensor([x.bits for x in rows],dtype=torch.long),
        torch.tensor([
            (math.log(float(x.prior/(1-x.prior))),
             math.log(float(x.ratio))) for x in rows
        ],dtype=torch.float32),
        torch.tensor([x.label for x in rows],dtype=torch.long),
        torch.tensor([x.prefix_labels for x in rows],dtype=torch.long),
        tuple((x.prior,x.ratio) for x in rows),
    )


def pool(seed:int,depth:int,switch:Q)->CuePanel:
    """Exact gold training examples drawn from split-restricted support."""
    if seed not in MODEL_SEEDS:
        raise ValueError("registered GG-09 model seed required")
    if switch not in (STATIC,SWITCH):
        raise ValueError("only static/Markov control grammars permitted")
    candidates=task_split(depth,switch)["training"]
    armcode=0 if switch==STATIC else 1
    rng=random.Random(94000+100*seed+10*depth+armcode)
    chosen=tuple(rng.choices(candidates,
                 weights=[float(x.likelihood) for x in candidates],
                 k=POOL_SIZE))
    train_ids={x.uid for x in candidates}
    if any(row.uid not in train_ids for row in chosen):
        raise AssertionError("training/heldout leakage")
    return as_panel(chosen)


def task_pools(seed:int,switch:Q)->dict[int,CuePanel]:
    return {depth:pool(seed,depth,switch) for depth in DEPTHS}


@dataclass(frozen=True)
class TrainingReceipt:
    task: str
    model_seed: int
    initial_sha256: str
    final_sha256: str
    updates: int
    total_sample_visits: int
    update_counts_by_depth: dict[str,int]
    unique_pool_indices_by_depth: dict[str,int]
    loss_trace: tuple[tuple[int,float],...]


def continue_task(model:EvaluationBeliefLooper,seed:int,switch:Q,
                  pools:dict[int,CuePanel],*,steps:int=TRANSFER_STEPS
                  )->tuple[EvaluationBeliefLooper,TrainingReceipt]:
    if seed not in MODEL_SEEDS or steps!=TRANSFER_STEPS:
        raise ValueError("GG-09 frozen seed and 240 continuation steps required")
    if switch not in (STATIC,SWITCH) or set(pools)!=set(DEPTHS) or any(
        len(x)!=POOL_SIZE for x in pools.values()
    ):
        raise ValueError("registered static/Markov training pools required")
    start=_weights_digest(model)
    target=deepcopy(model).train()
    if _weights_digest(target)!=start:
        raise AssertionError("continuation did not clone identical source weights")
    opt=torch.optim.AdamW(target.parameters(),lr=TRANSFER_LR,
                          betas=(.9,.999),weight_decay=TRANSFER_WD)
    code=0 if switch==STATIC else 1
    gen=torch.Generator(device="cpu").manual_seed(97000+100*seed+code)
    sampled={depth:set() for depth in DEPTHS}
    counts={str(d):0 for d in DEPTHS}
    visits=0
    history=[]
    for step in range(steps):
        depth=DEPTHS[step%len(DEPTHS)]
        ids=torch.randint(POOL_SIZE,(TRANSFER_BATCH,),generator=gen)
        sampled[depth].update(int(z) for z in ids.tolist())
        x=pools[depth]
        logits,_=target(x.observations[ids],x.context[ids])
        loss=F.cross_entropy(logits[:,-1,:],x.labels[ids])
        loss=loss+0.2*F.cross_entropy(logits.reshape(-1,2),
                                     x.prefix_labels[ids].reshape(-1))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(target.parameters(),1)
        opt.step()
        counts[str(depth)]+=1
        visits+=TRANSFER_BATCH*depth
        if step in (0,59,119,179,239):
            history.append((step+1,float(loss.detach())))
    target.eval()
    end=_weights_digest(target)
    if visits!=TOTAL_SAMPLE_VISITS or end==start:
        raise AssertionError("continuation optimizer budget or weight change failed")
    task_name="static" if switch==STATIC else "markov"
    return target,TrainingReceipt(
        task_name,seed,start,end,steps,visits,counts,
        {str(d):len(sampled[d]) for d in DEPTHS},tuple(history))


def _heldout_pools(depth:int)->dict[str,tuple[TaskExample,...]]:
    sample=task_split(depth,SWITCH)["heldout"]
    size=len(sample)//4
    boundary=tuple(sorted(sample,key=lambda x:(x.margin,x.key))[:size])
    rare=tuple(sorted(sample,key=lambda x:(x.likelihood,x.key))[:size])
    late=tuple(x for x in sample if
               markov_action(x.prior,x.ratio,x.bits[:depth//2])!=x.label)
    if not all((boundary,rare,late)):
        raise ValueError("heldout challenge pool empty")
    return {"natural":sample,"boundary":boundary,"rare":rare,"late":late}


def markov_gold_bank(depth:int)->GoldBank:
    if depth not in DEPTHS:
        raise ValueError("registered score depth required")
    pools=_heldout_pools(depth)
    reps=[]
    for rep in range(REPLICATES):
        cohorts={}
        hashes={}
        unique={}
        for offset,name in enumerate(COHORTS):
            candidates=pools[name]
            if name=="natural":
                rng=random.Random(95000+100*depth+rep)
                drawn=tuple(rng.choices(
                    candidates,
                    weights=[float(x.likelihood) for x in candidates],
                    k=NATURAL_N))
            else:
                rng=random.Random(96000+1000*depth+10*rep+offset)
                drawn=tuple(rng.choice(candidates)
                            for _ in range(CHALLENGE_N))
            panel=as_panel(drawn)
            cohorts[name]=panel
            hashes[name]=_panel_hash(panel)
            unique[name]=len({x.uid for x in drawn})
            if not set(x.uid for x in drawn) <= set(x.uid for x in candidates):
                raise AssertionError("new HMM gold pool/source membership changed")
        reps.append(GoldReplicate(depth,rep,cohorts,hashes,unique))
    obs=torch.cat([rep.cohorts[name].observations
                   for rep in reps for name in COHORTS],dim=0)
    ctx=torch.cat([rep.cohorts[name].context
                   for rep in reps for name in COHORTS],dim=0)
    label=torch.cat([rep.cohorts[name].labels
                     for rep in reps for name in COHORTS],dim=0)
    if obs.shape!=(BLOCK_N*REPLICATES,depth):
        raise AssertionError("HMM gold bank shape changed")
    return GoldBank(depth,tuple(reps),obs,ctx,label,SPLIT)


def _static_labels_for_bank(bank:GoldBank)->Tensor:
    result=[]
    for rep in bank.reps:
        for name in COHORTS:
            cohort=rep.cohorts[name]
            for i,(prior,ratio) in enumerate(cohort.parameters):
                bits=tuple(int(b) for b in cohort.observations[i])
                result.append(exact_action(prior,ratio,bits))
    result=torch.tensor(result,dtype=torch.long)
    if result.shape!=bank.labels.shape:
        raise AssertionError("static scoring labels differ in shape")
    return result


@torch.no_grad()
def ability(model:EvaluationBeliefLooper,banks:dict[int,GoldBank]
            )->tuple[dict[str,dict[str,float]],dict[str,dict[str,float]],
                     dict[str,dict[str,float]],dict[int,Tensor]]:
    markov={}
    static={}
    oracle_gap={}
    baseline_ce={}
    for depth,bank in banks.items():
        logits=model(bank.observations,bank.contexts)[0][:,-1,:]
        prediction=logits.argmax(-1)
        labels_old=_static_labels_for_bank(bank)
        markov_correct=(prediction==bank.labels).float().reshape(REPLICATES,BLOCK_N)
        static_correct=(prediction==labels_old).float().reshape(REPLICATES,BLOCK_N)
        disagreement=(labels_old!=bank.labels).float().reshape(REPLICATES,BLOCK_N)
        baseline_ce[depth]=F.cross_entropy(logits,bank.labels,
                           reduction="none").double().reshape(REPLICATES,BLOCK_N)
        markov[str(depth)]={}
        static[str(depth)]={}
        oracle_gap[str(depth)]={}
        for name,(a,b) in bank.cohort_slices.items():
            markov[str(depth)][name]=float(markov_correct[:,a:b].mean())
            static[str(depth)][name]=float(static_correct[:,a:b].mean())
            oracle_gap[str(depth)][name]=float(disagreement[:,a:b].mean())
    return markov,static,oracle_gap,baseline_ce


def _hmm_report_queries()->tuple[Query,...]:
    held=task_split(4,SWITCH)["heldout"]
    if len(held)!=48:
        raise AssertionError("HMM report source roster must have 48 rows")
    return tuple(Query(x.uid,x.prior,x.ratio,x.bits,x.label) for x in held)


def _write_json(path:Path,obj:dict[str,Any])->str:
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,
                              allow_nan=False)+"\n",encoding="utf-8")
    return sha256(path.read_bytes()).hexdigest()


def _audit(model:EvaluationBeliefLooper,banks:dict[int,GoldBank],
           queries:tuple[Query,...],sha:str
           )->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    certificates=[]
    physical=[]
    for query in queries:
        outputs,norms=native_report_successor_losses(model,query,banks,sha)
        by_depth={d:cohort_effect_bounds(outputs[d]["gap"],banks[d])
                  for d in DEPTHS}
        certificates.append({
            "uid":query.uid,"markov_label":query.label,
            "depths":{str(d):x for d,x in by_depth.items()},
            "composite":score_record(by_depth),
        })
        physical.append({
            "uid":query.uid,"markov_label":query.label,
            "raw_report_gradient_norms":norms,
            "depths":{str(d):{name:v.tolist() for name,v in outputs[d].items()}
                      for d in DEPTHS},
        })
    if _weights_digest(model)!=sha:
        raise AssertionError("source mutated during real SGD evaluation")
    return aggregate(certificates),certificates,physical


def run_gg09(*,output:Path,artifacts:Path,
             seeds:tuple[int,int]=MODEL_SEEDS,
             transfer_steps:int=TRANSFER_STEPS,
             base_steps:int=BASE_STEPS,
             gg08_steps:int=GG08_MIXED_STEPS)->dict[str,Any]:
    if (seeds!=MODEL_SEEDS or transfer_steps!=TRANSFER_STEPS
            or base_steps!=BASE_STEPS or gg08_steps!=GG08_MIXED_STEPS):
        raise ValueError("GG-09 frozen two-seed and optimizer schedule required")
    torch.set_num_threads(1)
    discrepancies=mismatch_census()
    banks={depth:markov_gold_bank(depth) for depth in DEPTHS}
    queries=_hmm_report_queries()
    split_hashes={
        str(d):{
            tag:sha256("".join(x.uid for x in task_split(d,SWITCH)[tag]).encode()
                        ).hexdigest()
            for tag in ("training","heldout")}
        for d in DEPTHS
    }
    if any({x.uid for x in task_split(d,SWITCH)["training"]} &
           {x.uid for x in task_split(d,SWITCH)["heldout"]}
           for d in DEPTHS):
        raise AssertionError("train/test grammar ID overlap")
    artifacts.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    record={
        "protocol":"RMP-GG-09",
        "registered_source_main_sha":SOURCE_MAIN,
        "preregistered_new_grammar":True,
        "known_binary_Bayes_source_checkpoints":True,
        "HMM_switch_probability":"1/4",
        "gold_oracle_exact_rational":True,
        "input_grammar_mismatch_census":discrepancies,
        "train_holdout_identity_hashes":split_hashes,
        "split_counts":{str(d):{tag:len(task_split(d,SWITCH)[tag])
                    for tag in ("training","heldout")} for d in DEPTHS},
        "model_seeds":list(seeds),
        "source_training_updates":[base_steps,gg08_steps],
        "new_continuation_updates":transfer_steps,
        "matched_continuation_sample_visits":TOTAL_SAMPLE_VISITS,
        "new_train_pool_size":POOL_SIZE,
        "new_train_batch_size":TRANSFER_BATCH,
        "scoring_depths":list(DEPTHS),
        "scoring_cohorts":list(COHORTS),
        "scoring_replicates":REPLICATES,
        "scoring_examples_per_replicate":BLOCK_N,
        "gold_panel_sha256":{
            str(d):[{k:rep.hashes[k] for k in COHORTS} for rep in bank.reps]
            for d,bank in banks.items()},
        "gold_panel_unique_counts":{
            str(d):[{k:rep.unique_counts[k] for k in COHORTS}
                    for rep in bank.reps] for d,bank in banks.items()},
        "protected_lambda_grid":[str(x) for x in LAMBDA_GRID],
        "protected_rho_grid":[str(x) for x in RHO_GRID],
        "protected_primary":["1/2","1/64"],
        "minimum_gold_depth_weight":"1/6",
        "native_physical_sgd_lr":.05,
        "native_full_gradient_clip":1.0,
        "model_knows_training_status":False,
        "results":[],
    }
    for seed in seeds:
        base,source_train=train_cell(seed,steps=base_steps,save_dir=artifacts)
        base_hash=_weights_digest(base)
        gg08_pools=gg08_training_pools(seed)
        source,source_trace=gg08_continue_training(
            base,seed,"mixed",gg08_pools,steps=gg08_steps)
        gg08_hash=_weights_digest(source)
        task_static=task_pools(seed,STATIC)
        task_hmm=task_pools(seed,SWITCH)
        fixed,stat_trace=continue_task(source,seed,STATIC,task_static,
                                       steps=transfer_steps)
        dynamic,mark_trace=continue_task(source,seed,SWITCH,task_hmm,
                                         steps=transfer_steps)
        if _weights_digest(source)!=gg08_hash or (
            stat_trace.initial_sha256!=gg08_hash
            or mark_trace.initial_sha256!=gg08_hash
            or stat_trace.total_sample_visits!=mark_trace.total_sample_visits
            or stat_trace.total_sample_visits!=TOTAL_SAMPLE_VISITS):
            raise AssertionError("matched transfer source/visit contract failed")
        models={"gg08_source":source,"static_control":fixed,
                "markov_transfer":dynamic}
        source_history={"base":source_train,
                        "gg08_mixed":source_trace.__dict__,
                        "static_continuation":stat_trace.__dict__,
                        "markov_continuation":mark_trace.__dict__}
        arms={}
        for arm in ARMS:
            model=models[arm]
            model_sha=_weights_digest(model)
            mark_ability,static_ability,label_gap,baseline=ability(model,banks)
            minimum=min(x for d in mark_ability.values() for x in d.values())
            capability=minimum>=GOLD_ACCURACY_GATE
            robustness,certs,physical=_audit(model,banks,queries,model_sha)
            checkpoint=artifacts/f"gg09_seed{seed}_{arm}.pt"
            torch.save({
                "protocol":"RMP-GG-09","seed":seed,"arm":arm,
                "source_GG08_sha256":gg08_hash,
                "model_state_dict":model.state_dict(),
                "source_history":source_history,
            },checkpoint)
            ledger=artifacts/f"gg09_seed{seed}_{arm}_physical.json"
            ledger_sha=_write_json(ledger,{
                "protocol":"RMP-GG-09","seed":seed,"arm":arm,
                "source_model_sha256":model_sha,
                "input_split_hashes":split_hashes,
                "gold_panel_sha256":record["gold_panel_sha256"],
                "baseline_markov_ce":{str(d):b.tolist() for d,b in baseline.items()},
                "per_query_certificates":certs,
                "physical_truth_flip_outcomes":physical,
            })
            signed=robustness["primary_protected_summary"]
            arms[arm]={
                "arm":arm,
                "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
                "checkpoint_filename":checkpoint.name,
                "physical_ledger_filename":ledger.name,
                "physical_ledger_sha256":ledger_sha,
                "model_sha256":model_sha,
                "source_weight_unchanged":_weights_digest(model)==model_sha,
                "markov_accuracy_cells":mark_ability,
                "static_accuracy_on_same_inputs":static_ability,
                "exact_oracle_disagreement_cells":label_gap,
                "minimum_markov_accuracy":minimum,
                "markov_capability_eligible":capability,
                "protected_gold":robustness,
                "primary_robust_signed_queries":signed["harm"]+signed["benefit"],
                "native_physical_sgd_steps":2*len(queries),
                "individual_post_sgd_gold_losses":
                    2*len(queries)*len(DEPTHS)*REPLICATES*BLOCK_N,
            }
        candidate=arms["markov_transfer"]
        control=arms["static_control"]
        acc_gain=candidate["minimum_markov_accuracy"]-control[
            "minimum_markov_accuracy"]
        signed_gain=(candidate["primary_robust_signed_queries"]-
                     control["primary_robust_signed_queries"])
        competence=candidate["markov_capability_eligible"] and acc_gain>=.05
        robustness=(competence and
                    candidate["primary_robust_signed_queries"]>=24 and
                    signed_gain>=8)
        record["results"].append({
            "seed":seed,
            "original_GG01_sha256":base_hash,
            "starting_GG08_mixed_sha256":gg08_hash,
            "source_and_training_receipts":source_history,
            "paired_transfer_source_weights_identical":True,
            "paired_transfer_sample_visits_identical":True,
            "arms":arms,
            "markov_min_accuracy_gain_over_static_control":acc_gain,
            "markov_robust_sign_gain_over_static_control":signed_gain,
            "registered_markov_competency_transfer_gate":bool(competence),
            "registered_markov_protected_harm_transfer_gate":bool(robustness),
        })
    record["two_seed_markov_competency_transfer"]=all(
        x["registered_markov_competency_transfer_gate"] for x in record["results"])
    record["two_seed_markov_protected_harm_transfer"]=all(
        x["registered_markov_protected_harm_transfer_gate"]
        for x in record["results"])
    output.write_text(json.dumps(record,sort_keys=True,indent=2,
                                 allow_nan=False)+"\n",encoding="utf-8")
    return record
