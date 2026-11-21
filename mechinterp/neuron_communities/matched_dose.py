"""NEC-03D: physically dose-matched grafts into the same recurrent MLP sites.

The baseline preserves only native always-on hubs in selected GELU
memory-token visits. Native and control donors come from the same CLEAN
input/visit activations, then are added to that hubs-only trajectory.
This *clean-graft* endpoint differs from NEC-02's live-activation retention.
Matched dose is local ||W_down delta||_2, never posthoc output accuracy.
"""
from __future__ import annotations

import math

import torch

from mechinterp.neuron_communities.core import decode
from mechinterp.neuron_communities.source_sae import (
    FEATURES, L, WIDTH, physical_feature_delta,
)

MODES = ("hubs", "native", "sham", "sae", "sae_sham")
MAX_GAIN = 8.0
DOSE_TOLERANCE = 0.05
EPS = 1e-10


def by_visit(keys) -> dict[int, set[int]]:
    result = {}
    for key in keys:
        visit, neuron = decode(key)
        if visit not in range(L) or neuron not in range(WIDTH):
            raise ValueError("outside visit/MLP native coordinate")
        result.setdefault(visit,set()).add(neuron)
    return result


def masked(values, indices: set[int]) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] != WIDTH:
        raise ValueError("expected 64-D native GELU vector")
    gate = torch.zeros(WIDTH,dtype=values.dtype,device=values.device)
    if indices:
        gate[list(sorted(indices))]=1
    return values*gate[None,:]


def physical_dose(delta: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    """Norm of the direct contribution to the shared MLP residual branch."""
    if delta.ndim != 2 or delta.shape[1] != WIDTH:
        raise ValueError("wrong GELU delta")
    if down_weight.ndim != 2 or down_weight.shape[1] != WIDTH:
        raise ValueError("wrong MLP-down weight basis")
    return (delta @ down_weight.T).norm(dim=-1)


def match_dose(target: torch.Tensor, control: torch.Tensor,
               down_weight: torch.Tensor, *, max_gain: float = MAX_GAIN):
    """Per-sample local perturbation matching with explicit saturation.

    Controls are scaled using target physical dose, never using outputs
    or task labels. Any mismatch >5% remains visible in the receipt.
    """
    if target.shape != control.shape or not 0 < max_gain <= 100:
        raise ValueError("dose matching dimensions or cap invalid")
    t = physical_dose(target,down_weight)
    c = physical_dose(control,down_weight)
    gain = torch.where(t>EPS,t/c.clamp_min(EPS),torch.zeros_like(t))
    if not torch.isfinite(gain).all():
        raise AssertionError("nonfinite dose multiplier")
    gain=gain.clamp(min=0,max=max_gain)
    adjusted=control*gain[:,None]
    actual=physical_dose(adjusted,down_weight)
    rel=(actual-t).abs()/t.clamp_min(1e-6)
    good=(rel<=DOSE_TOLERANCE)
    info={"target":t.detach().tolist(),"raw_control":c.detach().tolist(),
          "realized":actual.detach().tolist(),"gain":gain.detach().tolist(),
          "relative_error":rel.detach().tolist(),
          "matched_under_5pct":good.detach().tolist(),
          "saturated":(gain >= max_gain-1e-6).detach().tolist(),
          "max_gain":max_gain,
          "dose_coordinate":"native 64D GELU -> frozen mlp_down 32D output L2"}
    return adjusted, info


@torch.no_grad()
def donor_roster(model, clean_acts: torch.Tensor, native_keys: list[str],
                 hubs: list[str], sham_keys: list[str],
                 sae, mean, scale, features, sae_sham):
    """Freeze all clean-donor vectors before evaluating any causal arm."""
    if clean_acts.ndim != 3 or clean_acts.shape[1:] != (L,WIDTH):
        raise ValueError("clean activations must be Bx12x64")
    select = by_visit(native_keys)
    hubs_by = by_visit(hubs)
    sham_by = by_visit(sham_keys)
    for visit, sel in select.items():
        if len(sham_by.get(visit,set())) != len(sel):
            raise ValueError("same-visit sham must match native unit count")
        if sel & hubs_by.get(visit,set()):
            raise ValueError("native community overlaps hubs")
        if sham_by.get(visit,set()) & hubs_by.get(visit,set()):
            raise ValueError("frozen sham overlaps hubs; matched recovery invalid")
        if len(features.get(visit,[])) != len(sel):
            raise ValueError("SAE selection cardinality differs from native")
        if len(sae_sham.get(visit,[])) != len(sel):
            raise ValueError("SAE random feature budget differs")
    if set(features) != set(select) or set(sae_sham) != set(select):
        raise ValueError("SAE feature selection visits differ from native")
    w = model.block.mlp_down.weight.detach()
    donors, doses, hub_masks = {}, {}, {}
    for visit, sel in sorted(select.items()):
        a = clean_acts[:,visit,:].detach()
        native = masked(a,sel)
        sham = masked(a,sham_by[visit])
        sae_delta = physical_feature_delta(sae,mean,scale,a,features[visit])
        sae_sham_delta = physical_feature_delta(
            sae,mean,scale,a,sae_sham[visit])
        sham_adj, sham_dose = match_dose(native,sham,w)
        sae_adj, sae_dose = match_dose(native,sae_delta,w)
        sae_rand_adj, sae_rand_dose = match_dose(native,sae_sham_delta,w)
        donors[visit]={
            "hubs":torch.zeros_like(native),
            "native":native,
            "sham":sham_adj,
            "sae":sae_adj,
            "sae_sham":sae_rand_adj}
        doses[visit]={"sham":sham_dose,
                      "sae":sae_dose,"sae_sham":sae_rand_dose,
                      "native_direct_dose":physical_dose(native,w).tolist()}
        hub_masks[visit]=masked(torch.ones_like(a),hubs_by.get(visit,set()))[0]
        for mode,delta in donors[visit].items():
            if not torch.isfinite(delta).all():
                raise AssertionError("nonfinite donor: "+mode)
    return donors,doses,hub_masks


@torch.no_grad()
def graft_forward(model, bits, donors, hub_masks, mode: str) -> torch.Tensor:
    if mode not in MODES:
        raise ValueError("unsupported native/SAE graft arm")
    if not donors:
        return model.logits(bits)
    count=0
    def change(module,args):
        nonlocal count
        visit=count
        count+=1
        if visit not in donors:
            return None
        a=args[0].clone()
        donor=donors[visit][mode]
        if a.shape[0]!=len(bits) or donor.shape!=(len(bits),WIDTH):
            raise AssertionError("wrong batch/visit for graft donor")
        a[:,1,:]=a[:,1,:]*hub_masks[visit][None,:]+donor
        return (a,)
    hook=model.block.mlp_down.register_forward_pre_hook(change)
    try:
        logits=model.logits(bits)
        if count!=L:
            raise AssertionError("recurrent shared hook call count differs")
        return logits
    finally:
        hook.remove()


def dose_summary(rows: list[dict], mode: str) -> dict:
    """Average across selected visit/sample units, not independent worlds."""
    if mode not in ("sham","sae","sae_sham"):
        raise ValueError("unknown control")
    t,a,g,s=[],[],[],[]
    for row in rows:
        for visit in row.values():
            vals=visit[mode]
            t.extend(vals["target"])
            a.extend(vals["relative_error"])
            g.extend(vals["matched_under_5pct"])
            s.extend(vals["saturated"])
    return {
        "site_sample_count":len(t),
        "mean_direct_target_dose":sum(t)/len(t) if t else None,
        "mean_relative_dose_mismatch":sum(a)/len(a) if a else None,
        "fraction_matched_within_5pct":sum(g)/len(g) if g else None,
        "fraction_gain_at_cap":sum(s)/len(s) if s else None,
        "max_gain":MAX_GAIN,
        "used_task_labels_for_scaling":False,
        "dose_failure_if_under_90pct_matched":
            (sum(g)/len(g)<0.90) if g else True,
    }
