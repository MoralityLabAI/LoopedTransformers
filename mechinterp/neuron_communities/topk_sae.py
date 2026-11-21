"""NEC-03F: truly TopK sparse autoencoder on PEF-07 native GELU neurons.

One frozen model, 64 physical GELU channels, 128 learned decoder features.
Training samples are the first 80 complete PEF-07 training contexts;
feature-attribution ranks are fitted on the other 80 training contexts.
The input-conditioned policy chooses only *active* codes, without labels.
"""
from __future__ import annotations

import hashlib

import torch
from torch import nn
import torch.nn.functional as F

from mechinterp.neuron_communities.source_sae import (
    FEATURES, WIDTH, L, TRAIN_CONTEXTS, DEV_CONTEXTS,
    collect_source, seal_array,
)
from research.pef07.pef07 import as_tensor, model_digest

TOP_K=16
TRAIN_STEPS=480
BATCH=256
SEED_OFFSET=50331


class TopKSAE(nn.Module):
    def __init__(self, features=FEATURES, k=TOP_K):
        super().__init__()
        if k<1 or k>features:
            raise ValueError("TopK must lie inside the feature dimension")
        self.encoder=nn.Linear(WIDTH,features)
        self.decoder=nn.Linear(features,WIDTH)
        self.features=features
        self.k=k

    def codes(self, normalized):
        if normalized.ndim!=2 or normalized.shape[1]!=WIDTH:
            raise ValueError("SAE needs native 64D GELU input")
        raw=F.relu(self.encoder(normalized))
        values,indices=torch.topk(raw,self.k,dim=-1)
        z=torch.zeros_like(raw).scatter(1,indices,values)
        return z

    def forward(self,normalized):
        codes=self.codes(normalized)
        return self.decoder(codes),codes


def fit_topk(model, train_rows, *, seed:int, steps:int=TRAIN_STEPS):
    if len(train_rows)!=16*TRAIN_CONTEXTS or steps<1:
        raise ValueError("require 80 source-training context quartets and positive steps")
    original=model_digest(model)
    data=collect_source(model,train_rows).reshape(-1,WIDTH)
    mean=data.mean(0)
    scale=data.std(0,unbiased=False).clamp_min(1e-3)
    samples=(data-mean)/scale
    torch.manual_seed(seed+SEED_OFFSET)
    sae=TopKSAE()
    opt=torch.optim.AdamW(sae.parameters(),lr=1e-3,weight_decay=0.0)
    rng=torch.Generator().manual_seed(seed+SEED_OFFSET+89)
    start=end=None
    for _ in range(steps):
        idx=torch.randint(len(samples),(BATCH,),generator=rng)
        recon,codes=sae(samples[idx])
        # Fixed 16-active-code capacity is the sparsity constraint.
        loss=(recon-samples[idx]).square().mean()
        if not torch.isfinite(loss):
            raise AssertionError("nonfinite TopK SAE training loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if start is None:
            start=float(loss.detach())
        end=float(loss.detach())
    sae.eval()
    with torch.no_grad():
        recon,codes=sae(samples)
        mse=float((recon-samples).square().mean())
        l0=float((codes>1e-7).float().sum(1).mean())
        nz_frac=float(((codes>1e-7).sum(1)==TOP_K).float().mean())
    if model_digest(model)!=original:
        raise AssertionError("TopK feature training mutated PEF model weights")
    receipt={
        "model_sha256":original,
        "basis":"native GELU pre-mlp_down memory token, 64D",
        "sae_features":FEATURES,
        "target_active_features_per_input":TOP_K,
        "observed_mean_active_features_per_input":l0,
        "fraction_inputs_with_exactly_k_positive_features":nz_frac,
        "train_contexts":TRAIN_CONTEXTS,
        "feature_selection_contexts":DEV_CONTEXTS,
        "heldout_contexts_seen":0,
        "train_examples":len(data),
        "optimizer_steps":steps,
        "loss_start":start,"loss_end":end,
        "train_normalized_reconstruction_mse":mse,
        "mean_sha256":seal_array(mean),
        "scale_sha256":seal_array(scale),
        "source_only":True,
    }
    return sae,receipt,mean,scale


def dev_ranks(model, sae, mean, scale, rows):
    if len(rows)!=16*DEV_CONTEXTS:
        raise ValueError("source-only dev ranking needs 80 complete contexts")
    original=model_digest(model)
    importance=torch.zeros((L,FEATURES),dtype=torch.float64)
    for offset in range(0,len(rows),16*4):
        bits=as_tensor(rows[offset:offset+16*4])
        hooked=[]
        def hook(module,args):
            hooked.append(args[0])
        handle=model.block.mlp_down.register_forward_pre_hook(hook)
        try:
            logits=model.logits(bits)
            choice=logits.argmax(1)
            margin=logits.gather(1,choice[:,None]).squeeze(1)-(
                logits.gather(1,(1-choice)[:,None]).squeeze(1))
            if len(hooked)!=L:
                raise AssertionError("missing recurring native GELU nodes")
            grad=torch.autograd.grad(margin.sum(),hooked)
        finally:
            handle.remove()
        decoder=sae.decoder.weight.detach().T*scale[None,:]
        with torch.no_grad():
            for visit,(act,g) in enumerate(zip(hooked,grad)):
                z=sae.codes((act[:,1,:].detach()-mean)/scale)
                contribution=z*(g[:,1,:].detach()@decoder.T)
                importance[visit]+=contribution.abs().double().sum(0)
    if model_digest(model)!=original:
        raise AssertionError("source-rank attribution mutated frozen model")
    averaged=importance/len(rows)
    ranks=[
        sorted(range(FEATURES),key=lambda j:(-float(averaged[visit,j]),j))
        for visit in range(L)]
    if any(sorted(x)!=list(range(FEATURES)) for x in ranks):
        raise AssertionError("TopK source policy rank is not a full permutation")
    return ranks,seal_array(averaged)


def active_feature_policy(sae,mean,scale,states,ranking,*,n:int):
    """Choose top and bottom n **active** features using a source-frozen rank.

    This is an input-conditioned label-blind routing policy: fixed K
    active features come from the frozen TopKSAE, and only their
    identity/activations are queried at evaluation time.
    """
    if states.ndim!=2 or states.shape[1]!=WIDTH:
        raise ValueError("expected [B,64] native GELU coordinates")
    if len(ranking)!=FEATURES or set(ranking)!=set(range(FEATURES)):
        raise ValueError("invalid frozen source ranking")
    if n<1 or n>TOP_K//2:
        raise ValueError("requested active policy budget unsupported")
    with torch.no_grad():
        z=sae.codes((states-mean)/scale)
        decoder=sae.decoder.weight.detach().T*scale[None,:]
        index={f:i for i,f in enumerate(ranking)}
        good=[]
        bad=[]
        all_active=[]
        a=torch.zeros_like(states)
        b=torch.zeros_like(states)
        for row in range(len(states)):
            active=[int(j) for j in torch.where(z[row]>1e-7)[0].tolist()]
            ordered=sorted(active,key=lambda j:index[j])
            first=ordered[:n]
            last=list(reversed(ordered[-n:]))
            if set(first)&set(last):
                raise AssertionError("not enough distinct active SAE features")
            if len(first)<n or len(last)<n:
                raise AssertionError("TopK sample had too few positive features")
            a[row]=z[row,first]@decoder[first]
            b[row]=z[row,last]@decoder[last]
            good.append(first)
            bad.append(last)
            all_active.append(active)
    return a,b,{"top_source_ranked_active":good,
                "bottom_source_ranked_active":bad,
                "all_active_features":all_active,
                "policy_kind":"source-ranked, input-conditioned, label-blind TopK",
                "selected_features_per_arm_per_visit":n}
