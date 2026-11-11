"""GG-02: causal rank-k interchange on a *frozen* supervised Bayesian looper.

Fit orthogonal subspaces by differentiating the native tied-block
counterfactual continuation while all GG-01 model weights are frozen.
Exact-rational action targets and pair selection are supplied independently.
A distinct locked pair panel is read only after rank selection.

Research positive control only: no actual RL, hidden intent, or model
optimizer-self-awareness is measured by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as Q
from hashlib import sha256
from itertools import chain
from pathlib import Path
import json
import math
import random
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from research_gym.interp.goal_guarding_causal import (
    PRIORS, RATIOS, EvaluationBeliefLooper, exact_action, train_cell,
    fit_belief_probe,
)


CUTOFF = 2
RANKS = (2, 4, 8)
SPLITS = (("calibration", 256, 31001), ("validation", 128, 31002),
          ("locked", 256, 31003))


@dataclass(frozen=True)
class CounterfactualPanel:
    receiver: Tensor
    donor: Tensor
    hybrid: Tensor
    context: Tensor
    oracle: Tensor
    receiver_oracle: Tensor
    donor_oracle: Tensor
    ids: tuple[str, ...]
    attempts: int

    def __len__(self) -> int:
        return len(self.ids)


def _pair_id(p: Q, ratio: Q, receiver: tuple[int, ...],
             donor: tuple[int, ...]) -> str:
    canonical = {
        "domain": "rmp.gg02.pair.v1", "prior": str(p), "ratio": str(ratio),
        "receiver": list(receiver), "donor": list(donor),
    }
    return sha256(json.dumps(canonical, sort_keys=True,
                             separators=(",", ":")).encode()).hexdigest()


def generate_disjoint_panels(
    *, sizes: tuple[tuple[str,int,int], ...] = SPLITS,
    visits: int = 6, cutoff: int = CUTOFF,
) -> dict[str, CounterfactualPanel]:
    """Frozen PRNG panels, unique across splits, exact oracle exclusion."""
    if not 1 <= cutoff < visits <= 16:
        raise ValueError("cutoff and visits must satisfy 1 <= cutoff < visits <= 16")
    if any(type(n) is not int or n <= 0 or type(seed) is not int or
           not isinstance(name, str) or not name for name,n,seed in sizes):
        raise ValueError("panels require positive counts, integer seeds, and names")
    if len(set(name for name,_,_ in sizes)) != len(sizes):
        raise ValueError("panel names must be distinct")
    used: set[str] = set()
    panels: dict[str, CounterfactualPanel] = {}
    for name, count, seed in sizes:
        rng = random.Random(seed)
        receivers=[]; donors=[]; hybrids=[]; contexts=[]
        hybrids_y=[]; receivers_y=[]; donors_y=[]; identities=[]
        attempts=0
        while len(identities) < count and attempts < 2000 * count:
            attempts += 1
            p, ratio = rng.choice(PRIORS), rng.choice(RATIOS)
            receiver = tuple(rng.randrange(2) for _ in range(visits))
            donor = tuple(rng.randrange(2) for _ in range(visits))
            hybrid = donor[:cutoff] + receiver[cutoff:]
            y_h = exact_action(p, ratio, hybrid)
            y_r = exact_action(p, ratio, receiver)
            y_d = exact_action(p, ratio, donor)
            if y_h == y_r or y_h == y_d:
                continue
            ident = _pair_id(p, ratio, receiver, donor)
            if ident in used:
                continue
            used.add(ident)
            identities.append(ident)
            receivers.append(receiver)
            donors.append(donor)
            hybrids.append(hybrid)
            contexts.append((math.log(float(p/(1-p))),math.log(float(ratio))))
            hybrids_y.append(y_h)
            receivers_y.append(y_r)
            donors_y.append(y_d)
        if len(identities) != count:
            raise ValueError("oracle panel exhausted declared pair grammar")
        panels[name] = CounterfactualPanel(
            torch.tensor(receivers,dtype=torch.long),
            torch.tensor(donors,dtype=torch.long),
            torch.tensor(hybrids,dtype=torch.long),
            torch.tensor(contexts,dtype=torch.float32),
            torch.tensor(hybrids_y,dtype=torch.long),
            torch.tensor(receivers_y,dtype=torch.long),
            torch.tensor(donors_y,dtype=torch.long),
            tuple(identities),attempts,
        )
    return panels


@dataclass(frozen=True)
class CachedInterchange:
    panel: CounterfactualPanel
    receiver_state: Tensor
    donor_state: Tensor
    receiver_logits: Tensor
    hybrid_logits: Tensor
    eligible: Tensor

    @property
    def eligible_count(self) -> int:
        return int(self.eligible.sum())


@torch.no_grad()
def cache_states(model: EvaluationBeliefLooper,
                 panel: CounterfactualPanel, *, cutoff: int = CUTOFF
                 ) -> CachedInterchange:
    if type(cutoff) is not int or not 1 <= cutoff < panel.receiver.shape[1]:
        raise ValueError("cutoff must precede a continuation")
    model.eval()
    receiver_logits, receiver_states = model(panel.receiver,panel.context)
    _, donor_states = model(panel.donor,panel.context)
    hybrid_logits, _ = model(panel.hybrid,panel.context)
    eligible = ((receiver_logits[:,-1,:].argmax(-1)==panel.receiver_oracle)
                & (hybrid_logits[:,-1,:].argmax(-1)==panel.oracle))
    # The full donor-prefix state, transported into receiver's suffix,
    # is identical to a native run of the explicit hybrid.
    patched = model(panel.receiver,panel.context,patch_visit=cutoff,
                    patch_state=donor_states[:,cutoff-1,:])[0]
    if not torch.allclose(patched[:,cutoff-1:,:],
                          hybrid_logits[:,cutoff-1:,:],
                          rtol=3e-5,atol=3e-5):
        raise AssertionError("native complete-state continuation identity failed")
    sham = model(panel.receiver,panel.context,patch_visit=cutoff,
                 patch_state=receiver_states[:,cutoff-1,:])[0]
    if not torch.allclose(sham,receiver_logits,rtol=3e-5,atol=3e-5):
        raise AssertionError("receiver-state sham failed native identity")
    return CachedInterchange(
        panel,receiver_states[:,cutoff-1,:].detach(),
        donor_states[:,cutoff-1,:].detach(),
        receiver_logits.detach(),hybrid_logits.detach(),eligible,
    )


def orthogonal_basis(raw: Tensor) -> Tensor:
    if raw.ndim!=2 or raw.shape[0]<raw.shape[1]:
        raise ValueError("orthogonal basis needs [width,rank] with rank<=width")
    return torch.linalg.qr(raw,mode="reduced")[0]


def projected_patch(receiver: Tensor, donor: Tensor, basis: Tensor) -> Tensor:
    if (receiver.ndim!=2 or receiver.shape!=donor.shape or basis.ndim!=2
            or basis.shape[0]!=receiver.shape[1]):
        raise ValueError("patch dimensions disagree")
    # Orthonormal bases are constructed by QR, PCA, or normalized probe.
    delta=donor-receiver
    return receiver+(delta@basis)@basis.T


def principal_components(cache: CachedInterchange, rank: int) -> Tensor:
    delta=(cache.donor_state-cache.receiver_state).double()
    centered=delta-delta.mean(0,keepdim=True)
    eigval,eigvec=torch.linalg.eigh(centered.T@centered)
    return eigvec[:,-rank:].float().detach()


@dataclass(frozen=True)
class FittedSubspace:
    basis: Tensor
    rank: int
    seed: int
    loss_trace: tuple[tuple[int,float], ...]


def train_subspace(model: EvaluationBeliefLooper,
                   cache: CachedInterchange, rank: int, seed: int,
                   *, steps: int = 120, batch: int = 64) -> FittedSubspace:
    """Fit only the projection basis to counterfactual action CE.

    Uses all calibration pairs as registered, independent of which
    donor/receiver outputs were correct at the frozen pretrained checkpoint.
    """
    if type(rank) is not int or rank not in RANKS or rank > model.width:
        raise ValueError("rank must be one of the registered 2/4/8 and <= width")
    if type(steps) is not int or steps < 0 or type(batch) is not int or batch < 1:
        raise ValueError("invalid steps or batch")
    if not cache.panel.ids:
        raise ValueError("empty counterfactual calibration panel")
    model.eval()
    requires=[p.requires_grad for p in model.parameters()]
    for param in model.parameters():
        param.requires_grad_(False)
    generator=torch.Generator(device="cpu").manual_seed(41000+1000*seed+rank)
    raw=nn.Parameter(torch.randn(model.width,rank,generator=generator,
                                 dtype=torch.float32))
    optimizer=torch.optim.Adam([raw],lr=0.025,weight_decay=0.0)
    losses=[]
    try:
        for step in range(steps):
            choices=torch.randint(len(cache.panel),(batch,),generator=generator)
            receiver=cache.receiver_state[choices]
            donor=cache.donor_state[choices]
            basis=orthogonal_basis(raw)
            patched=projected_patch(receiver,donor,basis)
            logits=model(cache.panel.receiver[choices],
                         cache.panel.context[choices],
                         patch_visit=CUTOFF,patch_state=patched)[0][:,-1,:]
            loss=(F.cross_entropy(logits,cache.panel.oracle[choices])
                  +0.01*((patched-receiver)**2).sum(-1).mean())
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_([raw],1.0)
            optimizer.step()
            if step==0 or (step+1)%20==0 or step+1==steps:
                losses.append((step+1,float(loss.detach())))
        return FittedSubspace(orthogonal_basis(raw.detach()),rank,seed,
                              tuple(losses))
    finally:
        for param, required in zip(model.parameters(),requires):
            param.requires_grad_(required)


def _hybrid_margin(logits: Tensor, targets: Tensor) -> Tensor:
    final=logits[:,-1,:]
    return final.gather(1,targets[:,None]).flatten()-final.gather(
        1,(1-targets)[:,None]).flatten()


@torch.no_grad()
def score_basis(model: EvaluationBeliefLooper, cache: CachedInterchange,
                basis: Tensor, clean_norm_ceiling: float,
                *, random_seed: int, random_controls: int=8,
                neighbors: Tensor | None=None) -> dict[str,Any]:
    """Evaluate a frozen basis plus per-example norm-matched Gaussian controls."""
    if random_controls!=8:
        raise ValueError("GG-02 requires exactly eight random controls")
    eligible=cache.eligible
    n=cache.eligible_count
    if n==0:
        return {"eligible":0,"status":"NO_ELIGIBLE_PAIRS"}
    receiver=cache.receiver_state[eligible]
    donor=cache.donor_state[eligible]
    observations=cache.panel.receiver[eligible]
    context=cache.panel.context[eligible]
    targets=cache.panel.oracle[eligible]
    base_logits=cache.receiver_logits[eligible]
    target_logits=cache.hybrid_logits[eligible]
    patch=projected_patch(receiver,donor,basis)
    result=model(observations,context,patch_visit=CUTOFF,
                 patch_state=patch)[0]
    base_margin=_hybrid_margin(base_logits,targets)
    hybrid_margin=_hybrid_margin(target_logits,targets)
    def stats(logits):
        return {"hit":float((logits[:,-1,:].argmax(-1)==targets)
                            .float().mean()),
                "mean_margin":float(_hybrid_margin(logits,targets).mean())}
    norm_delta=(patch-receiver).norm(dim=-1)
    norm_ratio=patch.norm(dim=-1)/receiver.norm(dim=-1).clamp_min(1e-8)
    random_gen=torch.Generator(device="cpu").manual_seed(random_seed)
    random_stats=[]
    for _ in range(random_controls):
        direction=torch.randn(receiver.shape,generator=random_gen,
                              dtype=receiver.dtype,device=receiver.device)
        direction=direction/direction.norm(dim=1,keepdim=True).clamp_min(1e-8)
        random_patch=receiver+direction*norm_delta[:,None]
        logits=model(observations,context,patch_visit=CUTOFF,
                     patch_state=random_patch)[0]
        random_stats.append(stats(logits))
    nearest=None
    if neighbors is not None:
        safe=F.normalize(neighbors,dim=-1)
        nearest=float((F.normalize(patch,dim=-1)@safe.T).max(dim=-1).values.mean())
    here=stats(result)
    return {
        "status":"MEASURED","eligible":n,
        "attempted":len(cache.panel),"hit":here["hit"],
        "mean_margin":here["mean_margin"],
        "receiver_hit":stats(base_logits)["hit"],
        "hybrid_hit":stats(target_logits)["hit"],
        "margin_lift":float((_hybrid_margin(result,targets)-base_margin).mean()),
        "full_counterfactual_margin_lift":float((hybrid_margin-base_margin).mean()),
        "mean_patch_norm":float(norm_delta.mean()),
        "patch_to_receiver_norm_ratio_p99":float(torch.quantile(norm_ratio,.99)),
        "patch_to_receiver_norm_ratio_max":float(norm_ratio.max()),
        "clean_norm_99th":clean_norm_ceiling,
        "inside_clean_norm_ceiling":float((patch.norm(dim=-1)<=clean_norm_ceiling)
                                          .float().mean()),
        "nearest_clean_cosine":nearest,
        "random_controls":random_stats,
    }


@torch.no_grad()
def measure_simple_patch(model: EvaluationBeliefLooper,
                         cache: CachedInterchange,
                         basis: Tensor | None = None) -> dict[str,float]:
    mask=cache.eligible
    if not int(mask.sum()):
        return {"hit":float("nan"),"mean_margin":float("nan")}
    receiver=cache.receiver_state[mask]
    if basis is None:
        patch=receiver
    else:
        patch=projected_patch(receiver,cache.donor_state[mask],basis)
    logits=model(cache.panel.receiver[mask],cache.panel.context[mask],
                 patch_visit=CUTOFF,patch_state=patch)[0]
    return {
        "hit":float((logits[:,-1,:].argmax(-1)==cache.panel.oracle[mask])
                    .float().mean()),
        "mean_margin":float(_hybrid_margin(logits,cache.panel.oracle[mask]).mean()),
    }


def direction_probe_basis(model: EvaluationBeliefLooper) -> Tensor:
    """Frozen GG-01 probe direction; never trained on GG-02 intervention labels."""
    return fit_belief_probe(model).direction.detach().reshape(-1,1)


def score_selected_rank(metrics: dict[int, dict[str,Any]]) -> int:
    if set(metrics)!=set(RANKS):
        raise ValueError("validation must include all ranks")
    if any(metrics[k].get("status")!="MEASURED" for k in RANKS):
        raise ValueError("rank selection needs eligible validation evidence")
    return max(RANKS,key=lambda k:(metrics[k]["hit"],-k,
                                  metrics[k]["margin_lift"]))


def checkpoint_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def run_gg02(*,output:Path,checkpoints:Path,seeds:tuple[int,...]=(11,17),
             steps:int=400,fit_steps:int=120,
             sizes:tuple[tuple[str,int,int],...]=SPLITS) -> dict[str,Any]:
    if seeds!=(11,17):
        raise ValueError("registered GG-02 only accepts seed roster (11,17)")
    panels=generate_disjoint_panels(sizes=sizes)
    if set(panels)!={"calibration","validation","locked"}:
        raise ValueError("all registered dataset splits are required")
    torch.set_num_threads(1)
    checkpoints.mkdir(parents=True,exist_ok=True)
    report={"protocol":"RMP-GG-02","fit_steps":fit_steps,"train_steps":steps,
            "seeds":list(seeds),"split_sizes":{k:len(v) for k,v in panels.items()},
            "panel_identity_sha256":{k:sha256("".join(v.ids).encode()).hexdigest()
                                     for k,v in panels.items()},
            "GG01_source":"merged dfac2655512cf18be10cf1489973ffbcc0bc0b95",
            "frozen_supervised_positive_control":True,
            "spontaneous_goal_guarding_claim":False,"results":[]}
    for seed in seeds:
        model,training=train_cell(seed,steps=steps,save_dir=checkpoints)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        cal=cache_states(model,panels["calibration"])
        val=cache_states(model,panels["validation"])
        # The locked panel is deliberately not encoded until after rank selection.
        ceiling=float(torch.quantile(torch.cat((cal.receiver_state.norm(dim=-1),
                         cal.donor_state.norm(dim=-1))),.99))
        neighbors=torch.cat((cal.receiver_state,cal.donor_state),dim=0)
        baseline=direction_probe_basis(model)
        fitted={rank:train_subspace(model,cal,rank,seed,steps=fit_steps)
                for rank in RANKS}
        validation={}
        for rank in RANKS:
            validation[rank]=score_basis(
                model,val,fitted[rank].basis,ceiling,
                random_seed=66713+seed*100+rank,neighbors=neighbors)
        selected=score_selected_rank(validation)
        locked=cache_states(model,panels["locked"])
        locked_results={}
        for rank in RANKS:
            out=score_basis(model,locked,fitted[rank].basis,ceiling,
                      random_seed=66713+seed*100+rank,neighbors=neighbors)
            pca=measure_simple_patch(model,locked,principal_components(cal,rank))
            out["pca"]=pca
            out["loss_trace"]=list(fitted[rank].loss_trace)
            locked_results[str(rank)]=out
            subpath=checkpoints/f"gg02_seed{seed}_rank{rank}.pt"
            torch.save({"basis":fitted[rank].basis,
                        "rank":rank,"seed":seed,"train_loss_trace":fitted[rank].loss_trace},
                        subpath)
            out["basis_sha256"]=checkpoint_hash(subpath)
        selected_result=locked_results[str(selected)]
        sham=measure_simple_patch(model,locked)
        probe=measure_simple_patch(model,locked,baseline)
        all_random=[n["hit"] for n in selected_result["random_controls"]]
        gate=(
            selected_result["eligible"]>=32
            and selected_result["hit"]>=.50
            and selected_result["hit"]-sham["hit"]>=.10
            and all(selected_result["hit"]-x>=.05 for x in all_random)
            and selected_result["hit"]-selected_result["pca"]["hit"]>=.05
            and selected_result["inside_clean_norm_ceiling"]>=.95
        )
        report["results"].append({
            "seed":seed,"base_training":training,
            "calibration_eligible":cal.eligible_count,
            "validation_eligible":val.eligible_count,
            "locked_eligible":locked.eligible_count,
            "calibration_norm_ceiling":ceiling,
            "validation_rank_metrics":{str(k):v for k,v in validation.items()},
            "selected_rank_from_validation":selected,
            "locked_rank_metrics":locked_results,
            "locked_sham":sham,
            "locked_gg01_linear_probe":probe,
            "selected_rank_causal_gate":gate,
        })
    report["two_seed_causal_gate"]=all(s["selected_rank_causal_gate"]
                                       for s in report["results"])
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,sort_keys=True,
                                 allow_nan=False)+"\n",encoding="utf-8")
    return report
