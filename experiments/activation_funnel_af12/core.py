"""AF-12 synchronous Horn-closure supervision for the original TypedLoop.

Training targets come from independently recomputed finite Horn closure, never
from model-generated labels. Only facts and graph incidence reach the model.
A shared atom-status readout evaluates *each* atom after every upstream call.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, State
from . import data


@dataclass
class ClosureTrace:
    logits: torch.Tensor  # B, steps, physical atom, {false,true}
    codes: torch.Tensor   # B, steps, physical atom, bottleneck width
    final: State


def closure_layers(case: gym.Case, steps: int) -> torch.Tensor:
    """True synchronous least-Horn iteration after 1..steps waves.

    The physical permutation only determines index order; label derivation uses
    the original ordered logical Horn rules. No new physical graph properties
    appear in the oracle.
    """
    if not 1 <= steps <= 10:
        raise ValueError('unregistered count of upstream steps')
    up,_=gym.logical_rules(case)
    known=set(gym.initial_facts(case))
    atom_map,_=data.physical_permutations(case.permutation)
    labels=[]
    for _ in range(steps):
        additions={head for prem,head in up if set(prem).issubset(known)}
        known.update(additions)
        logical=torch.zeros(gym.ATOMS,dtype=torch.long)
        for node in known:
            logical[atom_map[node]]=1
        labels.append(logical)
    if steps>=case.depth:
        assert tuple(int(gym.SUBGOALS[j] in known) for j in (0,1))==gym.oracle(case)[0][:2]
    return torch.stack(labels)


def true_step_targets(cases,steps:int)->torch.Tensor:
    return torch.stack([closure_layers(c,steps) for c in cases])


def trace_upstream(model:GraphLooper,batch:gym.GraphBatch,steps:int)->ClosureTrace:
    """Run the unmodified TypedLoop, collecting actual carried atom codes.

    This is an extra forward pass included in all objective arms; all arms
    execute exactly the same sequence of original cell calls.
    """
    if steps not in (4,6,8,10):raise ValueError('unexpected upstream loop count')
    a=model.atom_encoder(batch.atom_features)
    r=model.rule_encoder(batch.rule_features)
    logits=[];codes=[]
    for _ in range(steps):
        s=model._step(a,r,batch.prem_up,batch.head_up)
        a,r=s.a,s.r
        logits.append(model.subgoal_head(s.a))
        codes.append(s.az)
    return ClosureTrace(torch.stack(logits,1),torch.stack(codes,1),s)


def relay_loss(trace:ClosureTrace,truth:torch.Tensor,arm:str):
    if trace.logits.shape[:-1]!=truth.shape or arm not in ('terminal','relay_final','closure_rollout'):
        raise ValueError('AF-12 relay-target or arm mismatch')
    if arm=='terminal':
        return trace.logits.sum()*0., {'auxiliary_CE':0.,'monotonic_penalty':0.}
    stage=trace.logits if arm=='closure_rollout' else trace.logits[:,-1:]
    gold=truth if arm=='closure_rollout' else truth[:,-1:]
    element=F.cross_entropy(stage.reshape(-1,2),gold.reshape(-1),reduction='none').reshape_as(gold)
    # Positive antecedent labels would otherwise be overwhelmed by padded/unreachable atoms.
    weighted=element*torch.where(gold==1,2.,1.)
    ce=weighted.sum()/torch.where(gold==1,2.,1.).sum()
    p=F.softmax(trace.logits,dim=-1)[...,1]
    mono=F.relu(p[:,:-1]-p[:,1:]).square().mean()
    extra=.35*ce+(.15*mono if arm=='closure_rollout' else 0.)
    return extra,{'auxiliary_CE':float(ce.detach()),'monotonic_penalty':float(mono.detach())}


def hypothesis_counts(logits:torch.Tensor,truth:torch.Tensor):
    if logits.ndim!=4 or logits.shape[:-1]!=truth.shape:
        raise ValueError('invalid closure-score arrays')
    pred=logits.argmax(-1)
    return {'atom_correct':int((pred==truth).sum()),'atom_n':int(truth.numel()),
            'final_atom_correct':int((pred[:,-1]==truth[:,-1]).sum()),
            'final_atom_n':int(truth[:,-1].numel()),
            'trajectory_all_correct':int((pred==truth).all(dim=-1).all(dim=-1).sum()),
            'examples':int(len(truth))}
