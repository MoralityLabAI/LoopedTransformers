"""GG-03: finite legal-state support and target-blind gradient controls.

Replays the frozen supervised GG-01 native tied decoder and GG-02 rank
selection. New reporting pairs are disjoint from every registered GG-02 pair.
All exact Bayes labels originate outside model outputs. The local decision-
gradient comparator uses receiver logits only, never a hybrid oracle target.
No causal claim about actual optimizer updates or spontaneous self-awareness.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from itertools import product
from pathlib import Path
import json
import math
import random
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    PRIORS, RATIOS, EvaluationBeliefLooper, exact_action,
    train_cell, fit_belief_probe,
)
from research_gym.interp.goal_guarding_subspace import (
    CUTOFF, RANKS, SPLITS, CounterfactualPanel,
    generate_disjoint_panels, cache_states, train_subspace,
    principal_components, projected_patch, score_basis,
    score_selected_rank,
)

GG03_SIZES = SPLITS + (("gg03_reporting", 256, 34001),)
GG03_INVARIANT_SIZE = 256
GG03_INVARIANT_SEED = 34002
GG03_SOURCE = "6dde1f988aa27c07be0e648dea2947adc7597fff"


def context(p: Q, ratio: Q) -> tuple[float, float]:
    return math.log(float(p / (1-p))), math.log(float(ratio))


def invariant_order_panel(n: int = GG03_INVARIANT_SIZE,
                          seed: int = GG03_INVARIANT_SEED) -> CounterfactualPanel:
    """Order-swapped 01/10 prefixes with *equal* rational posterior odds."""
    if type(n) is not int or not 1 <= n <= 288 or type(seed) is not int:
        raise ValueError("invariant panel requires 1..288 rows and an integer seed")
    rng = random.Random(seed)
    used = set()
    receiver, donor, hybrid = [], [], []
    conditions, labels, ids = [], [], []
    attempts = 0
    while len(ids) < n:
        attempts += 1
        if attempts > n*500:
            raise ValueError("finite invariant panel was exhausted")
        p, ratio = rng.choice(PRIORS), rng.choice(RATIOS)
        start = (0,1) if (len(ids) % 2 == 0) else (1,0)
        other = (1,0) if start == (0,1) else (0,1)
        suffix = tuple(rng.randrange(2) for _ in range(4))
        item = (str(p),str(ratio),start,suffix)
        if item in used:
            continue
        used.add(item)
        r = start + suffix
        d = other + suffix
        h = other + suffix
        yr,yd,yh = (exact_action(p,ratio,b) for b in (r,d,h))
        if yr!=yd or yr!=yh:
            raise AssertionError("Bayesian sufficient-statistic invariance broken")
        receiver.append(r);donor.append(d);hybrid.append(h)
        conditions.append(context(p,ratio));labels.append(yr)
        ids.append(sha256(json.dumps(
            {"tag":"gg03.invariant.v1","prior":str(p),"ratio":str(ratio),
             "receiver":r,"donor":d},sort_keys=True
        ).encode()).hexdigest())
    result=CounterfactualPanel(
        torch.tensor(receiver,dtype=torch.long),
        torch.tensor(donor,dtype=torch.long),
        torch.tensor(hybrid,dtype=torch.long),
        torch.tensor(conditions,dtype=torch.float32),
        torch.tensor(labels,dtype=torch.long),
        torch.tensor(labels,dtype=torch.long),
        torch.tensor(labels,dtype=torch.long),
        tuple(ids),attempts,
    )
    return result


@dataclass(frozen=True)
class LegalPrefixSupport:
    """All 4 exact recurrent states for each of 9 legal context choices."""
    contexts: Tensor                 # [9,2]
    states: Tensor                   # [9,4,width]
    minimum_positive_separation: Tensor    # [9]
    degenerate: bool


@torch.no_grad()
def legal_prefix_support(model: EvaluationBeliefLooper) -> LegalPrefixSupport:
    model.eval()
    options=tuple(product(PRIORS,RATIOS))
    prefixes=tuple(product((0,1),repeat=CUTOFF))
    contexts=torch.tensor([context(p,r) for p,r in options],
                          dtype=torch.float32)
    allcontexts=contexts.repeat_interleave(len(prefixes),dim=0)
    bits=torch.tensor([prefix for _ in options for prefix in prefixes],
                      dtype=torch.long)
    states=model(bits,allcontexts)[1][:,CUTOFF-1,:]
    states=states.reshape(len(options),len(prefixes),model.width)
    pairs=torch.combinations(torch.arange(len(prefixes)),r=2)
    distances=(states[:,pairs[:,0],:]-states[:,pairs[:,1],:]).norm(dim=-1)
    separations=torch.where(distances>1e-8,distances,
                            torch.full_like(distances,float("inf"))).min(dim=1).values
    degenerate=bool((~torch.isfinite(separations)).any())
    return LegalPrefixSupport(contexts,states,separations,degenerate)


@torch.no_grad()
def support_metrics(patches: Tensor, contexts: Tensor,
                    support: LegalPrefixSupport) -> dict[str,float | bool]:
    if patches.ndim!=2 or contexts.shape!=(patches.shape[0],2):
        raise ValueError("patched states and contexts have incompatible dimensions")
    if support.states.shape[-1]!=patches.shape[-1]:
        raise ValueError("legal-state width differs from model hidden width")
    if support.degenerate:
        return {"degenerate":True,"fraction_within_tenth_gap":0.0}
    lookup_distance=torch.cdist(contexts,support.contexts)
    chosen=lookup_distance.argmin(dim=1)
    if float(lookup_distance.min(dim=1).values.max())>1e-6:
        raise ValueError("unregistered context in support audit")
    candidates=support.states[chosen]
    distance=(patches[:,None,:]-candidates).norm(dim=-1).min(dim=-1).values
    ratio=distance/support.minimum_positive_separation[chosen]
    if not bool(torch.isfinite(ratio).all()):
        raise ValueError("nonfinite finite-support separation")
    return {
        "degenerate":False,
        "max_raw_distance":float(distance.max()),
        "median_scaled_distance":float(torch.median(ratio)),
        "p95_scaled_distance":float(torch.quantile(ratio,.95)),
        "fraction_within_tenth_gap":float((ratio<=.1).float().mean()),
        "fraction_exact_state":float((distance<=1e-4).float().mean()),
        "minimum_context_separation":float(support.minimum_positive_separation.min()),
    }


def output_gradient_patch(model: EvaluationBeliefLooper,
                          cache, cutoff: int = CUTOFF) -> Tensor:
    """Per-example target-BLIND output-margin gradient projection.

    Compute the protect-minus-exploit margin derivative at receiver's
    unaltered state, followed by receiver's own suffix. Donor differences
    enter only via a linear projection onto this local sensitivity.
    No hybrid labels, counterfactual outputs or donor logits enter g.
    """
    h=cache.receiver_state.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        logits=model(cache.panel.receiver,cache.panel.context,
                     patch_visit=cutoff,patch_state=h)[0][:,-1,:]
        margin=(logits[:,1]-logits[:,0]).sum()
        gradient=torch.autograd.grad(margin,h)[0].detach()
    length=gradient.norm(dim=-1,keepdim=True)
    unit=gradient/length.clamp_min(1e-10)
    delta=cache.donor_state-cache.receiver_state
    patch=cache.receiver_state+((delta*unit).sum(dim=-1,keepdim=True)*unit)
    if not bool(torch.isfinite(patch).all()):
        raise ValueError("output-gradient patch is nonfinite")
    return patch.detach()


def _hybrid_margin(logits: Tensor, target: Tensor) -> Tensor:
    last=logits[:,-1,:]
    return last.gather(1,target[:,None]).flatten() - last.gather(
        1,(1-target)[:,None]).flatten()


@torch.no_grad()
def patched_metrics(model: EvaluationBeliefLooper,cache,patch: Tensor,
                    support: LegalPrefixSupport) -> dict[str,Any]:
    if patch.shape!=cache.receiver_state.shape:
        raise ValueError("patch must have one hidden state per panel row")
    mask=cache.eligible
    n=int(mask.sum())
    if n==0:
        return {"status":"NO_ELIGIBLE", "eligible":0,"attempted":len(cache.panel)}
    x=cache.panel.receiver[mask]
    ctx=cache.panel.context[mask]
    label=cache.panel.oracle[mask]
    patched=patch[mask]
    output=model(x,ctx,patch_visit=CUTOFF,patch_state=patched)[0]
    hybrid=cache.hybrid_logits[mask]
    receiver=cache.receiver_logits[mask]
    m=_hybrid_margin(output,label)
    metrics={
        "status":"MEASURED","attempted":len(cache.panel),"eligible":n,
        "hit":float((output[:,-1,:].argmax(-1)==label).float().mean()),
        "mean_hybrid_margin":float(m.mean()),
        "mean_absolute_margin_error_to_direct_hybrid":float((
            m-_hybrid_margin(hybrid,label)).abs().mean()),
        "mean_signed_margin_shift_from_receiver":float((
            m-_hybrid_margin(receiver,label)).mean()),
        "mean_patch_l2":float((patched-cache.receiver_state[mask]).norm(dim=-1).mean()),
        "support":support_metrics(patched,ctx,support),
    }
    return metrics


def _valid_regression(data:dict[str,Any]) -> bool:
    return data.get("status")=="MEASURED" and data["eligible"]>=32


def _sha_json_ids(panel:CounterfactualPanel)->str:
    return sha256("".join(panel.ids).encode()).hexdigest()


def run_gg03(*,output:Path,artifacts:Path,
             seeds:tuple[int,...]=(11,17),train_steps:int=400,
             fit_steps:int=120,
             reporting_size:int=256,
             invariant_size:int=GG03_INVARIANT_SIZE
             )->dict[str,Any]:
    if seeds!=(11,17):
        raise ValueError("GG-03 requires both registered model seeds")
    if train_steps!=400 or fit_steps!=120 or reporting_size!=256 or invariant_size!=256:
        raise ValueError("full registered GG-03 uses frozen step and panel counts")
    torch.set_num_threads(1)
    # Register IDs from all three GG-02 panels *before* extending the
    # globally collision-checked panel roster with the new locked split.
    sizes=GG03_SIZES
    panels=generate_disjoint_panels(sizes=sizes)
    inv=invariant_order_panel(invariant_size)
    output.parent.mkdir(parents=True,exist_ok=True)
    artifacts.mkdir(parents=True,exist_ok=True)
    report:dict[str,Any]={
        "protocol":"RMP-GG-03",
        "source_main_commit":GG03_SOURCE,
        "prior_GG02_frozen_outcomes_already_seen":True,
        "new_reporting_pair_identifier_sha256":_sha_json_ids(panels["gg03_reporting"]),
        "same_statistic_pair_identifier_sha256":_sha_json_ids(inv),
        "pair_sizes":{name:len(panel) for name,panel in panels.items()},
        "invariant_pair_size":len(inv),
        "model_seeds":list(seeds),
        "train_steps":train_steps,"fit_steps":fit_steps,
        "external_optimizer_gradient_updates_applied":False,
        "results":[],
    }
    for seed in seeds:
        model, training = train_cell(seed,steps=train_steps,save_dir=artifacts)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cal=cache_states(model,panels["calibration"])
        val=cache_states(model,panels["validation"])
        ceiling=float(torch.quantile(torch.cat((
            cal.receiver_state.norm(dim=-1),cal.donor_state.norm(dim=-1))),.99))
        fitted={rank:train_subspace(model,cal,rank,seed,
                                   steps=fit_steps) for rank in RANKS}
        validation={}
        for rank in RANKS:
            validation[rank]=score_basis(
                model,val,fitted[rank].basis,ceiling,
                random_seed=66713+seed*100+rank,
                neighbors=torch.cat((cal.receiver_state,cal.donor_state))
            )
        chosen=score_selected_rank(validation)
        checkpoint=artifacts/f"gg03_seed{seed}_selected_rank{chosen}.pt"
        torch.save({"seed":seed,"rank":chosen,"basis":fitted[chosen].basis,
                    "train_steps":train_steps,"fit_steps":fit_steps},checkpoint)
        # Open untouched new panels only after fitting, validation rank
        # selection and checkpoint serialization have completed.
        measured=cache_states(model,panels["gg03_reporting"])
        same=cache_states(model,inv)
        legal=legal_prefix_support(model)
        learned=fitted[chosen].basis
        pca=principal_components(cal,chosen)
        probe=fit_belief_probe(model).direction.reshape(-1,1)
        fresh={
            "sham":patched_metrics(model,measured,measured.receiver_state,legal),
            "full_state":patched_metrics(model,measured,measured.donor_state,legal),
            "learned":patched_metrics(model,measured,
                projected_patch(measured.receiver_state,measured.donor_state,learned),legal),
            "pca":patched_metrics(model,measured,
                projected_patch(measured.receiver_state,measured.donor_state,pca),legal),
            "linear_probe":patched_metrics(model,measured,
                projected_patch(measured.receiver_state,measured.donor_state,probe),legal),
            "output_gradient":patched_metrics(model,measured,
                output_gradient_patch(model,measured),legal),
        }
        original=score_basis(model,measured,learned,ceiling,
                             random_seed=66713+seed*100+chosen,
                             neighbors=torch.cat((cal.receiver_state,cal.donor_state)))
        fresh["eight_norm_matched_random"]=original["random_controls"]
        invariant={
            "sham":patched_metrics(model,same,same.receiver_state,legal),
            "full_state":patched_metrics(model,same,same.donor_state,legal),
            "learned":patched_metrics(model,same,
                projected_patch(same.receiver_state,same.donor_state,learned),legal),
            "pca":patched_metrics(model,same,
                projected_patch(same.receiver_state,same.donor_state,pca),legal),
            "output_gradient":patched_metrics(model,same,
                output_gradient_patch(model,same),legal),
        }
        n=fresh["learned"].get("eligible",0)
        n_inv=invariant["learned"].get("eligible",0)
        gate=(
            _valid_regression(fresh["learned"])
            and n_inv>=32
            and not legal.degenerate
            and fresh["learned"]["hit"]-fresh["pca"]["hit"]>=.05
            and fresh["learned"]["hit"]-fresh["output_gradient"]["hit"]>=.05
            and fresh["learned"]["support"]["fraction_within_tenth_gap"]>=.95
            and 1-invariant["learned"]["hit"]<=.01
        )
        assert fresh["full_state"].get("support",{}).get(
            "max_raw_distance",float("inf"))<=1e-4
        assert fresh["sham"].get("support",{}).get(
            "max_raw_distance",float("inf"))<=1e-4
        assert invariant["full_state"].get("support",{}).get(
            "max_raw_distance",float("inf"))<=1e-4
        report["results"].append({
            "seed":seed,
            "base_training":training,
            "selected_rank":chosen,
            "selection_on_prior_GG02_validation_only":True,
            "calibration_eligible":cal.eligible_count,
            "validation_eligible":val.eligible_count,
            "new_reporting_eligible":n,
            "invariant_eligible":n_inv,
            "calibration_clean_state_norm_ceiling":ceiling,
            "checkpoint":checkpoint.name,
            "checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest(),
            "rank_fit_loss_trace":list(fitted[chosen].loss_trace),
            "new_reporting":fresh,
            "invariant_order_swap":invariant,
            "unique_causal_specificity_gate":bool(gate),
        })
    report["two_seed_specificity_gate"]=all(
        row["unique_causal_specificity_gate"] for row in report["results"])
    output.write_text(json.dumps(report,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n",encoding="utf-8")
    return report
