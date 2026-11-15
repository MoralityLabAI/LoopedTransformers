"""AF-13 differentiable asynchronous Horn soundness, monotonicity and deadline.

No label for an exact arrival time enters the objective. A recurrent model may
choose its own propagation order as long as it preserves given facts, keeps
unreachable atoms false, and eventually computes the entire least fixed point.
The soft one-step support constraint is diagnostic/regularizing rather than an
assertion that the model implements discrete Boolean closure at each call.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, State
from . import data


@dataclass
class AsyncTrace:
    logits:torch.Tensor             # [batch, recurrence step, physical atom, bool logit]
    codes:torch.Tensor              # [batch, recurrence step, physical atom, carried width]
    final:State


def trace_upstream(model: GraphLooper, batch:gym.GraphBatch, steps:int)->AsyncTrace:
    if steps not in (4,6,8,10,12):
        raise ValueError('unsupported AF13 recurrence horizon')
    a=model.atom_encoder(batch.atom_features)
    r=model.rule_encoder(batch.rule_features)
    logits=[];codes=[]
    for _ in range(steps):
        s=model._step(a,r,batch.prem_up,batch.head_up)
        a,r=s.a,s.r
        logits.append(model.subgoal_head(s.a));codes.append(s.az)
    return AsyncTrace(torch.stack(logits,dim=1),torch.stack(codes,dim=1),s)


def closure_bounds(c:gym.Case):
    """Known facts and least fixed point in *physical* atom coordinates."""
    initial=set(gym.initial_facts(c))
    reachable,_=gym.closure(initial,gym.logical_rules(c)[0])
    physical,_=data.physical_permutations(c.permutation)
    f=torch.zeros(gym.ATOMS,dtype=torch.bool)
    t=torch.zeros_like(f)
    for v in initial:f[physical[v]]=True
    for v in reachable:t[physical[v]]=True
    if not bool((f<=t).all()):raise AssertionError('facts outside closure')
    return f,t


def terminal_truth(cases:list[gym.Case])->torch.Tensor:
    return torch.stack([closure_bounds(c)[1].long() for c in cases])


def admissible_status(logits:torch.Tensor,cases:list[gym.Case]):
    """Hard metrics, independent of a prescribed Boolean wave clock."""
    if logits.ndim!=4 or logits.shape[0]!=len(cases) or logits.shape[2:]!=(gym.ATOMS,2):
        raise ValueError('AF13 logits/case roster mismatch')
    pred=logits.argmax(-1).bool()
    facts=torch.stack([closure_bounds(c)[0] for c in cases])[:,None,:]
    reached=torch.stack([closure_bounds(c)[1] for c in cases])[:,None,:]
    false_pos=(pred & ~reached)
    fact_loss=(facts & ~pred)
    decreases=(pred[:,:-1] & ~pred[:,1:])
    final=(pred[:,-1]==reached[:,0]).float().mean()
    return {
        'unsupported_true_count':int(false_pos.sum()),
        'missing_fact_count':int(fact_loss.sum()),
        'monotonicity_violation_count':int(decreases.sum()),
        'end_atom_accuracy':float(final),
        'fully_admissible_fraction':float((~false_pos.any(-1) & ~fact_loss.any(-1)).all(-1).float().mean()),
    }


def rule_support_loss(p:torch.Tensor,cases:list[gym.Case])->torch.Tensor:
    """Soft support by a previous-step premise, without an exact wave label.

    For each non-fact atom, require p_t(head) <= max_{r -> head}
    prod_{premise in r} p_(t-1)(premise). This penalizes premature
    unsupported arrivals; different valid rule orders are permitted.
    """
    if p.ndim!=3 or p.shape[0]!=len(cases) or p.shape[2]!=gym.ATOMS:
        raise ValueError('support probability shape mismatch')
    losses=[]
    for i,c in enumerate(cases):
        pm,_=data.physical_permutations(c.permutation)
        facts=gym.initial_facts(c)
        upstream=gym.logical_rules(c)[0]
        for node in range(gym.ATOMS):
            if node in facts:continue
            rules=[prem for prem,head in upstream if head==node]
            if rules:
                support=torch.stack([
                    p[i,:-1,list(pm[x] for x in prem)].prod(dim=-1)
                    for prem in rules
                ],dim=0).amax(dim=0)
            else:
                support=torch.zeros_like(p[i,:-1,pm[node]])
            losses.append(F.relu(p[i,1:,pm[node]]-support).square().mean())
    return torch.stack(losses).mean()


def asynchronous_loss(trace:AsyncTrace,cases:list[gym.Case],arm:str):
    """Stage-unclocked supervision with matched terminal-label baseline."""
    if arm not in ('terminal','relay_final','async_sound'):
        raise ValueError('unregistered AF13 arm')
    if trace.logits.ndim!=4 or trace.logits.shape[0]!=len(cases):
        raise ValueError('AF13 trace/case count mismatch')
    if arm=='terminal':
        return trace.logits.sum()*0., {'deadline_ce':0.,'fact_ce':0.,'outside_ce':0.,
                                      'monotonic_penalty':0.,'support_penalty':0.}
    truth=terminal_truth(cases).to(trace.logits.device)
    stage=trace.logits[:,-1] # final time only; NO exact Boolean wave labels
    ce=F.cross_entropy(stage.reshape(-1,2),truth.reshape(-1),reduction='none').reshape_as(truth)
    weights=torch.where(truth==1,2.,1.)
    deadline=(ce*weights).sum()/weights.sum()
    if arm=='relay_final':
        return .35*deadline, {'deadline_ce':float(deadline.detach()),'fact_ce':0.,'outside_ce':0.,
                              'monotonic_penalty':0.,'support_penalty':0.}
    p=F.softmax(trace.logits,dim=-1)[...,1]
    facts=torch.stack([closure_bounds(c)[0] for c in cases]).to(device=p.device)
    reached=truth.bool()
    never=reached.logical_not()
    # Fixed sets below are targets derived from the exact oracle, never features.
    factprob=p.masked_select(facts[:,None,:].expand_as(p))
    outside=p.masked_select(never[:,None,:].expand_as(p))
    fact_ce=-torch.log(factprob.clamp_min(1e-6)).mean() if factprob.numel() else p.sum()*0.
    outside_ce=-torch.log((1-outside).clamp_min(1e-6)).mean() if outside.numel() else p.sum()*0.
    monotone=F.relu(p[:,:-1]-p[:,1:]).square().mean()
    support=rule_support_loss(p,cases)
    total=.35*deadline+.08*fact_ce+.08*outside_ce+.10*monotone+.06*support
    return total,{'deadline_ce':float(deadline.detach()),'fact_ce':float(fact_ce.detach()),
                   'outside_ce':float(outside_ce.detach()),
                   'monotonic_penalty':float(monotone.detach()),
                   'support_penalty':float(support.detach())}
