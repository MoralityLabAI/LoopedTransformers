"""GG-10: first-order Markov gold gradients, finite SGD residual, TV width.

Historical GG-09 endpoints/panels are already known and must reproduce
the original SHA and protected-sign census. New independently sampled
gold panels are held out from the *GG-10 scoring analysis*, still drawn
from the same registered finite HMM task grammar and training-ID split.

All truth/flip post-report steps are real PyTorch SGD on cloned frozen
native LoopedTransformers models. Cohort gold gradients use autograd on
the exact external HMM labels. The policy never sees them.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any
import json
import math
import random

import torch
from torch import Tensor
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    EvaluationBeliefLooper,train_cell,_weights_digest,
)
from research_gym.interp.goal_guarding_competency_recovery import (
    training_pools as gg08_training_pools,
    continue_training as gg08_continue_training,
)
from research_gym.interp.goal_guarding_markov_transfer import (
    MODEL_SEEDS,DEPTHS,COHORTS,REPLICATES,BLOCK_N,NATURAL_N,CHALLENGE_N,
    STATIC,SWITCH,SOURCE_MAIN as GG09_SOURCE,
    task_pools,continue_task,markov_gold_bank,
    _heldout_pools,as_panel,_hmm_report_queries,
    ability,
)
from research_gym.interp.goal_guarding_protected_tail import (
    GoldReplicate,GoldBank,SPLIT,
    evaluate_bank,cohort_effect_bounds,score_record,aggregate,
    protected_depth_bounds,
)
from research_gym.interp.goal_guarding_actual_updates_b import _panel_hash
from research_gym.interp.goal_guarding_gradient_conflict import (
    gradient,query_gradients,dot64,cosine64,sign,
)

SOURCE_MAIN="8299198f925dcbf5e875d8a241be317bb42e35cf"
ARMS=("static_control","markov_transfer")
FAMILIES=("historical","fresh")
PRIMARY_LAMBDA=Q(1,2)
PRIMARY_RHO=Q(1,64)
LR=.05
SIGN_EPS=1e-7
NEW_NATURAL_SEED_BASE=105000
NEW_CHALLENGE_SEED_BASE=106000
EXPECTED_SHA={
    11:{
        "static_control":"35be2252711351d11364a625dc2ef01bb61c792ad72392225863e750af0e3e33",
        "markov_transfer":"31e31149171ba6452acc27abd552c83624922f5d8b051651aed3142835ad1ad7",
    },
    17:{
        "static_control":"2a335c1c5b0075cabb8baf24f41635bbc7723ec8fcf8be6a91514bfa994cf88a",
        "markov_transfer":"0c11e22c64e0f50bf5ee732e004adeb7200584bf8fee1cdd4dabe1d38427011b",
    },
}
ORIGINAL_CENSUS={
    11:{"static_control":(13,35,0),"markov_transfer":(10,1,37)},
    17:{"static_control":(13,33,2),"markov_transfer":(19,1,28)},
}
COHORT_WEIGHTS={"natural":.5,"boundary":1/6,"rare":1/6,"late":1/6}


def new_gold_bank(depth:int)->GoldBank:
    """New exact Markov heldout score samples using preregistered seeds."""
    if depth not in DEPTHS:
        raise ValueError("registered Markov scoring depths only")
    pools=_heldout_pools(depth)
    reps=[]
    for rep in range(REPLICATES):
        cohorts={}
        hashes={}
        unique={}
        for offset,name in enumerate(COHORTS):
            candidates=pools[name]
            if name=="natural":
                rng=random.Random(NEW_NATURAL_SEED_BASE+100*depth+rep)
                sampled=tuple(rng.choices(candidates,
                    weights=[float(x.likelihood) for x in candidates],
                    k=NATURAL_N))
            else:
                rng=random.Random(NEW_CHALLENGE_SEED_BASE+
                                  1000*depth+10*rep+offset)
                sampled=tuple(rng.choice(candidates)
                              for _ in range(CHALLENGE_N))
            panel=as_panel(sampled)
            cohorts[name]=panel
            hashes[name]=_panel_hash(panel)
            unique[name]=len({x.uid for x in sampled})
            if not set(x.uid for x in sampled).issubset(
                {z.uid for z in candidates}):
                raise AssertionError("new HMM panel escaped heldout grammar")
        reps.append(GoldReplicate(depth,rep,cohorts,hashes,unique))
    observations=torch.cat([rep.cohorts[name].observations
                            for rep in reps for name in COHORTS],dim=0)
    context=torch.cat([rep.cohorts[name].context
                       for rep in reps for name in COHORTS],dim=0)
    labels=torch.cat([rep.cohorts[name].labels
                      for rep in reps for name in COHORTS],dim=0)
    if observations.shape!=(REPLICATES*BLOCK_N,depth):
        raise AssertionError("new bank geometry changed")
    return GoldBank(depth,tuple(reps),observations,context,labels,SPLIT)


def banks_all()->dict[tuple[str,int],GoldBank]:
    return {(family,depth):
            (markov_gold_bank(depth) if family=="historical"
             else new_gold_bank(depth))
            for family in FAMILIES for depth in DEPTHS}


def panel_hashes(banks:dict[tuple[str,int],GoldBank])->dict[str,Any]:
    return {
        family:{
            str(depth):[{c:rep.hashes[c] for c in COHORTS}
                        for rep in banks[(family,depth)].reps]
            for depth in DEPTHS}
        for family in FAMILIES
    }


def source_models(seed:int,artifacts:Path)->dict[str,EvaluationBeliefLooper]:
    """Rebuild exact GG-09 frozen endpoints and refuse weight drift."""
    if seed not in MODEL_SEEDS:
        raise ValueError("GG-10 only uses exact GG-09 seeds")
    base,record=train_cell(seed,steps=400,save_dir=artifacts)
    source,gg08=gg08_continue_training(
        base,seed,"mixed",gg08_training_pools(seed),steps=300)
    original=_weights_digest(source)
    models={}
    for arm,switch in (("static_control",STATIC),("markov_transfer",SWITCH)):
        model,trace=continue_task(source,seed,switch,task_pools(seed,switch),
                                  steps=240)
        if trace.initial_sha256!=original:
            raise AssertionError("GG-09 replays have different initial weights")
        if _weights_digest(model)!=EXPECTED_SHA[seed][arm]:
            raise ValueError("GG-09 source endpoint hash failed exact reproduction")
        models[arm]=model.eval()
    if _weights_digest(source)!=original:
        raise AssertionError("GG-08 source mutated during GG-09 replays")
    return models


def gold_cohort_gradients(model:EvaluationBeliefLooper,
                          banks:dict[tuple[str,int],GoldBank]
                          )->dict[tuple[str,int,int,str],tuple[Tensor,...]]:
    """Independent autograd for each exact Markov gold cohort mean CE."""
    found={}
    for (family,depth),bank in banks.items():
        for rep in range(REPLICATES):
            for cohort in COHORTS:
                panel=bank.reps[rep].cohorts[cohort]
                logits=model(panel.observations,panel.context)[0][:,-1,:]
                loss=F.cross_entropy(logits,panel.labels)
                found[(family,depth,rep,cohort)],raw=gradient(model,loss)
                if not math.isfinite(raw):
                    raise AssertionError("gold cohort gradient is nonfinite")
    return found


def depth_gradient_cosines(grads:dict[
        tuple[str,int,int,str],tuple[Tensor,...]],
        family:str)->dict[str,float | None]:
    """Protected population-mean cohort gradients averaged over replicates."""
    depth_grad={}
    for depth in DEPTHS:
        first=grads[(family,depth,0,"natural")]
        summed=[torch.zeros_like(x,dtype=torch.float64) for x in first]
        for rep in range(REPLICATES):
            for cohort,weight in COHORT_WEIGHTS.items():
                g=grads[(family,depth,rep,cohort)]
                for total,part in zip(summed,g):
                    total.add_(part.double(),alpha=weight/REPLICATES)
        depth_grad[depth]=tuple(summed)
    return {
        f"{a}_{b}":cosine64(depth_grad[a],depth_grad[b])
        for a,b in ((4,6),(4,8),(6,8))
    }


def _sgd_outcomes(base:EvaluationBeliefLooper,report_gradients,
                  banks:dict[tuple[str,int],GoldBank],
                  base_hash:str)->dict[tuple[str,int],dict[str,Tensor]]:
    """One physical SGD per action, all gold families share successor weights."""
    physical={}
    for action in ("truth","flip"):
        model=deepcopy(base).eval()
        if _weights_digest(model)!=base_hash:
            raise AssertionError("physical step input weights drifted")
        model.zero_grad(set_to_none=True)
        for param,grad in zip(model.parameters(),report_gradients[action]):
            param.grad=grad.detach().clone()
        torch.optim.SGD(model.parameters(),lr=LR,
                        weight_decay=0,momentum=0).step()
        if _weights_digest(model)==base_hash:
            raise AssertionError("real SGD did not change parameters")
        physical[action]={key:evaluate_bank(model,bank)[0]
                          for key,bank in banks.items()}
    return {
        key:{
            "truth":physical["truth"][key],
            "flip":physical["flip"][key],
            "gap":physical["flip"][key]-physical["truth"][key],
        }
        for key in banks
    }


def nominal_impact(decomposition:dict[int,dict[int,dict[str,dict[str,float]]]]
                   )->list[dict[str,float|int]]:
    """Nominal protected mixture uses fixed cohort and uniform depth weights."""
    results=[]
    for rep in range(REPLICATES):
        values={}
        for metric in ("physical","first_order","residual"):
            values[metric]=sum(
                COHORT_WEIGHTS[cohort]*decomposition[depth][rep][cohort][metric]
                for depth in DEPTHS for cohort in COHORTS)/len(DEPTHS)
        if abs(values["physical"]-values["first_order"]-
               values["residual"])>1e-7:
            raise AssertionError("linear/finite residual identity failed")
        results.append({
            "replicate":rep,**values,
            "physical_sign":sign(values["physical"],SIGN_EPS),
            "first_order_sign":sign(values["first_order"],SIGN_EPS),
        })
    return results


def ambiguity_width(per_depth:dict[int,dict[str,Any]],
                    original_composite:dict[str,float],
                    rep:int)->dict[str,Any]:
    """Exact decomposition of protected cohort TV and depth selection widths."""
    all_depth={}
    cohort_contributions={x:0.0 for x in COHORTS}
    for depth in DEPTHS:
        input_rep=per_depth[depth]["replicates"][rep]
        width=0.0
        for cohort,weight in COHORT_WEIGHTS.items():
            c=input_rep[cohort]["1/64"]
            diff=c["upper"]-c["lower"]
            if diff < -1e-9:
                raise AssertionError("cohort upper/lower order inverted")
            component=weight*max(0.0,diff)
            cohort_contributions[cohort]+=component/6
            width+=component
        bounded=protected_depth_bounds(input_rep,Q(1,2),Q(1,64))
        if abs(width-(bounded["upper"]-bounded["lower"]))>1e-7:
            raise AssertionError("cohort interval width mismatch")
        all_depth[depth]=bounded
    cohort_width=sum(cohort_contributions.values())
    max_depth=max(DEPTHS,key=lambda d:all_depth[d]["upper"])
    min_depth=min(DEPTHS,key=lambda d:all_depth[d]["lower"])
    depth_width=.5*(all_depth[max_depth]["upper"]-
                     all_depth[min_depth]["lower"])
    actual=original_composite["upper"]-original_composite["lower"]
    if abs(actual-cohort_width-depth_width)>1e-7:
        raise AssertionError("composite TV/depth width identity failed")
    return {
        "full_protected_interval_width":actual,
        "cohort_TV_width_component":cohort_width,
        "depth_selection_width_component":depth_width,
        "weighted_per_cohort_TV_width":cohort_contributions,
        "lower_worst_depth":min_depth,
        "upper_worst_depth":max_depth,
    }


def unresolved_reason(p0:dict[str,Any],p1:dict[str,Any]
                      )->str|None:
    """Post-hoc exclusive classification of a GG-09 unresolved query."""
    if p1["status"]!="unresolved":
        return None
    nominal=p0["nominal_signs"]
    if len(nominal)!=REPLICATES or 0 in nominal or len(set(nominal))>1:
        return "panel_sign_heterogeneity"
    if p0["status"]=="unresolved":
        return "depth_mixture_ambiguity"
    if p1["status"]=="unresolved":
        return "cohort_TV_ambiguity"
    return "numerical_other"


def compare_decomposition_rows(decomp:dict[int,dict[int,dict[str,dict[str,float]]]],
                               original:list[dict[str,float|int]]
                               )->None:
    expected=nominal_impact(decomp)
    if len(original)!=len(expected):
        raise ValueError("nominal replicate count differs")
    for a,b in zip(expected,original):
        if abs(float(a["physical"])-float(b["physical"]))>1e-7:
            raise AssertionError("cohort physical mean differs from native aggregate")


def model_query_evidence(model:EvaluationBeliefLooper,
                         rows,banks,gradients,
                         source_hash:str)->list[dict[str,Any]]:
    result=[]
    for row in rows:
        report_grads,norms=query_gradients(model,row)
        difference=tuple(x-y for x,y in zip(report_grads["flip"],
                                           report_grads["truth"]))
        physical=_sgd_outcomes(model,report_grads,banks,source_hash)
        family_records={}
        for family in FAMILIES:
            per_depth={}
            components={}
            for depth in DEPTHS:
                gold_bank=banks[(family,depth)]
                data=physical[(family,depth)]
                per_depth[depth]=cohort_effect_bounds(data["gap"],gold_bank)
                components[depth]={}
                for rep in range(REPLICATES):
                    components[depth][rep]={}
                    for cohort in COHORTS:
                        a,b=gold_bank.cohort_slices[cohort]
                        value=float(data["gap"][rep,a:b].double().mean())
                        linear=-LR*dot64(gradients[(family,depth,rep,cohort)],
                                         difference)
                        components[depth][rep][cohort]={
                            "physical":value,
                            "first_order":linear,
                            "residual":value-linear,
                        }
            composite=score_record(per_depth)
            nominal=nominal_impact(components)
            expected=composite["1/2"]["0"]["replicates"]
            compare_decomposition_rows(components,expected)
            width=[
                ambiguity_width(per_depth,
                    composite["1/2"]["1/64"]["replicates"][rep],
                    rep)
                for rep in range(REPLICATES)]
            dominant=max(COHORTS,key=lambda c:sum(
                abs(COHORT_WEIGHTS[c]*components[depth][rep][c]["first_order"])
                for depth in DEPTHS for rep in range(REPLICATES)))
            family_records[family]={
                "cohort_gradients":{
                    str(d):{
                        str(r):components[d][r]
                        for r in range(REPLICATES)}
                    for d in DEPTHS},
                "nominal_protected":nominal,
                "protected_composite":composite,
                "composite_ambiguity_width_decomposition":width,
                "largest_weighted_abs_first_order_cohort":dominant,
                "unresolved_reason":unresolved_reason(
                    composite["1/2"]["0"],composite["1/2"]["1/64"]),
                "physical_per_example_gaps":{
                    str(depth):physical[(family,depth)]["gap"].tolist()
                    for depth in DEPTHS},
            }
        if _weights_digest(model)!=source_hash:
            raise AssertionError("source model changed after SGD score replay")
        result.append({
            "uid":row.uid,
            "exact_markov_gold_action":row.label,
            "report_gradient_raw_norms":norms,
            "family":family_records,
        })
    return result


def _summary_one(rows:list[dict[str,Any]],family:str)->dict[str,Any]:
    counts={"harm":0,"benefit":0,"unresolved":0}
    first_matches=0
    considered=0
    zeros=0
    predicted_zeros=0
    opposing=0
    residual_abs=[]
    unresolved_causes={}
    largest={}
    for row in rows:
        item=row["family"][family]
        status=item["protected_composite"]["1/2"]["1/64"]["status"]
        counts[status]+=1
        reason=item["unresolved_reason"]
        if reason is not None:
            unresolved_causes[reason]=unresolved_causes.get(reason,0)+1
            c=item["largest_weighted_abs_first_order_cohort"]
            largest[c]=largest.get(c,0)+1
        for rec in item["nominal_protected"]:
            residual_abs.append(abs(rec["residual"]))
            actual,pred=rec["physical_sign"],rec["first_order_sign"]
            if actual==0:
                zeros+=1
                continue
            considered+=1
            first_matches+=int(actual==pred)
            predicted_zeros+=int(pred==0)
            opposing+=int(actual*pred==-1)
    if sum(counts.values())!=len(rows):
        raise AssertionError("GG-10 robust status counts incomplete")
    return {
        "queries":len(rows),
        "robust_status_counts":counts,
        "unresolved_reason_counts":unresolved_causes,
        "unresolved_dominant_first_order_cohort_counts":largest,
        "nominal_protected_replicate_total":len(rows)*REPLICATES,
        "unambiguous_physical_nominal_replicates":considered,
        "unambiguous_sign_agreement":first_matches,
        "unambiguous_sign_agreement_fraction":
            first_matches/considered if considered else None,
        "finite_zero_or_ambiguous":zeros,
        "first_order_zero_on_nonzero_finite":predicted_zeros,
        "opposed_unambiguous_first_order_signs":opposing,
        "median_abs_finite_step_residual":float(median(residual_abs)),
        "max_abs_finite_step_residual":max(residual_abs),
        "nominal_first_order_supported_gate":(
            considered>=96 and first_matches/considered>=.75
            if considered else False),
        "nominal_residual_sensitive_event":(
            considered>=96 and (considered-first_matches)/considered>=.25
            if considered else False),
    }


def summarize_arm(rows:list[dict[str,Any]],gold_cosines)->dict[str,Any]:
    if len(rows)!=48:
        raise ValueError("GG-10 needs all 48 source Markov reports")
    historical=_summary_one(rows,"historical")
    fresh=_summary_one(rows,"fresh")
    originally_unresolved={
        x["uid"] for x in rows if x["family"]["historical"][
            "protected_composite"]["1/2"]["1/64"]["status"]=="unresolved"}
    fresh_unresolved={
        x["uid"] for x in rows if x["family"]["fresh"][
            "protected_composite"]["1/2"]["1/64"]["status"]=="unresolved"}
    return {
        "historical":historical,
        "fresh":fresh,
        "gold_gradient_cosines":gold_cosines,
        "originally_unresolved_count":len(originally_unresolved),
        "fresh_unresolved_count":len(fresh_unresolved),
        "unresolved_overlap_count":len(originally_unresolved&fresh_unresolved),
        "unresolved_overlap_uids_sha256":sha256(
            "".join(sorted(originally_unresolved&fresh_unresolved)).encode()
        ).hexdigest(),
        "unresolved_replication_event":(
            len(originally_unresolved&fresh_unresolved)>=16),
    }


def run_gg10(*,output:Path,artifacts:Path,
             seeds:tuple[int,int]=MODEL_SEEDS,
             transfer_steps:int=240)->dict[str,Any]:
    if seeds!=MODEL_SEEDS or transfer_steps!=240:
        raise ValueError("GG-10 source model seeds and training steps frozen")
    torch.set_num_threads(1)
    banks=banks_all()
    hashes=panel_hashes(banks)
    rows=_hmm_report_queries()
    if len(rows)!=48:
        raise AssertionError("new-grammar source report count changed")
    artifacts.mkdir(parents=True,exist_ok=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    result={
        "protocol":"RMP-GG-10",
        "registered_source_main_sha256":SOURCE_MAIN,
        "replays_known_GG09_scoring_panels":True,
        "prospective_fresh_scoring_panel_family":True,
        "new_scoring_rng_seeds":{
            "natural_seed_base":NEW_NATURAL_SEED_BASE,
            "challenge_seed_base":NEW_CHALLENGE_SEED_BASE,
        },
        "gold_panel_hashes":hashes,
        "report_query_sha256":sha256(
            "".join(r.uid for r in rows).encode()).hexdigest(),
        "model_seeds":list(seeds),
        "model_arms":list(ARMS),
        "source_model_sha_expectations":EXPECTED_SHA,
        "original_gg09_robust_census":ORIGINAL_CENSUS,
        "report_step_lr":LR,
        "report_full_gradient_clip":1.0,
        "primary_lambda":"1/2",
        "primary_rho":"1/64",
        "nominal_first_order_accuracy_threshold":.75,
        "new_panel_unresolved_replication_threshold":16,
        "gold_cohorts":list(COHORTS),
        "score_depths":list(DEPTHS),
        "score_replicates":REPLICATES,
        "physical_SGD_on_native_model":True,
        "autonomous_optimizer_awareness_evidence":False,
        "results":[],
    }
    for seed in seeds:
        models=source_models(seed,artifacts)
        arms={}
        for arm in ARMS:
            model=models[arm]
            sha=_weights_digest(model)
            gold_grads=gold_cohort_gradients(model,banks)
            gradients_cosines={
                family:depth_gradient_cosines(gold_grads,family)
                for family in FAMILIES
            }
            evidence=model_query_evidence(model,rows,banks,gold_grads,sha)
            summary=summarize_arm(evidence,gradients_cosines)
            historical=summary["historical"]["robust_status_counts"]
            expected=ORIGINAL_CENSUS[seed][arm]
            if (historical["harm"],historical["benefit"],
                    historical["unresolved"])!=expected:
                raise AssertionError("GG-09 historical protected-sign census drifted")
            ledger=artifacts/f"gg10_seed{seed}_{arm}_physical_decomposition.json"
            ledger.write_text(json.dumps({
                "source_GG09_model_sha256":sha,
                "gold_panel_hashes":hashes,
                "report_query_sha256":result["report_query_sha256"],
                "source_model_seed":seed,
                "source_arm":arm,
                "per_report":evidence,
            },indent=2,sort_keys=True,allow_nan=False)+"\n")
            checkpoint=artifacts/f"gg10_seed{seed}_{arm}_model.pt"
            torch.save({"protocol":"RMP-GG-10","seed":seed,"arm":arm,
                        "source_GG09_sha256":sha,
                        "model_state_dict":model.state_dict()},checkpoint)
            arms[arm]={
                "source_GG09_model_sha256":sha,
                "unchanged_source_model":_weights_digest(model)==sha,
                "source_GG09_expected_sha256":EXPECTED_SHA[seed][arm],
                "physical_post_report_sgd_steps":2*len(rows),
                "actual_gold_example_loss_evaluations":
                    2*len(rows)*len(DEPTHS)*REPLICATES*BLOCK_N*len(FAMILIES),
                "gold_cohort_backprop_count":len(gold_grads),
                "summary":summary,
                "checkpoint_filename":checkpoint.name,
                "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
                "ledger_filename":ledger.name,
                "ledger_sha256":sha256(ledger.read_bytes()).hexdigest(),
            }
        model_markov=arms["markov_transfer"]["summary"]
        result["results"].append({
            "seed":seed,
            "arms":arms,
            "markov_first_order_supported_fresh":
                model_markov["fresh"]["nominal_first_order_supported_gate"],
            "markov_fresh_residual_sensitive":
                model_markov["fresh"]["nominal_residual_sensitive_event"],
            "markov_fresh_unresolved_replication":
                model_markov["unresolved_replication_event"],
        })
    result["two_seed_nominal_first_order_gate"]=all(
        r["markov_first_order_supported_fresh"] for r in result["results"])
    result["any_residual_sensitive_markov_seed"]=any(
        r["markov_fresh_residual_sensitive"] for r in result["results"])
    result["two_seed_unresolved_replication_event"]=all(
        r["markov_fresh_unresolved_replication"] for r in result["results"])
    output.write_text(json.dumps(result,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n")
    return result
