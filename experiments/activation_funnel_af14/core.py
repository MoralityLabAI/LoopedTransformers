"""AF-14: bounded-progress supervision on true Horn proof depths.

Only deadlines are supervised. A reachable atom may become true at any earlier
step. Exact goal and subgoal labels still come from the independent AF-05 Horn
oracle. No milestone, closure wave, or donor truth is included in model inputs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af13.core import (
    AsyncTrace, trace_upstream, asynchronous_loss,
)
from . import data

FAIRNESS_B = 2
MARGIN = 0.25
PROGRESS_WEIGHT = 0.35


def minimal_proof_depths(case: gym.Case) -> dict[int, int]:
    """Minimal *logical* Horn proof heights, from the frozen rule oracle.

    For a rule P->v its proof depth is 1+max(depth(p) for p in P).
    Iterate decreasing candidate heights to handle alternate proofs. Finite
    monotone Horn closure guarantees termination when a height is attainable.
    """
    heights = {p: 0 for p in gym.initial_facts(case)}
    rules = gym.logical_rules(case)[0]
    for _ in range(gym.ATOMS+1):
        improved = False
        for premise, head in rules:
            if all(p in heights for p in premise):
                candidate = 1 + max((heights[p] for p in premise), default=0)
                if head not in heights or candidate < heights[head]:
                    heights[head] = candidate
                    improved = True
        if not improved:
            break
    else:
        raise ValueError('minimal proof-depth relaxation failed to terminate')
    reached,_ = gym.closure(set(gym.initial_facts(case)), rules)
    if reached != set(heights):
        raise ValueError('minimal depth support differs from Horn closure')
    return heights


def progress_deadlines(case: gym.Case, bound: int=FAIRNESS_B) -> list[int]:
    """Per-*physical*-atom latest required 1-based recurrent step; -1 absent.

    A fact is required at the first neural call. Reachable nonfacts of minimal
    proof depth d have deadline b*d. These are upper bounds, never exact-wave
    arrival labels. Extra recurrent calls after the deadline test persistence.
    """
    if bound < 1:
        raise ValueError('positive bounded-fair progress period required')
    physical,_ = data.physical_permutations(case.permutation)
    result = [-1]*gym.ATOMS
    for logical,depth in minimal_proof_depths(case).items():
        result[physical[logical]] = max(1,bound*depth)
    return result



def closure_bounds(case: gym.Case):
    """AF-14 physical layout, independent of the frozen AF-13 map roster."""
    initial=set(gym.initial_facts(case))
    reached,_=gym.closure(initial,gym.logical_rules(case)[0])
    physical,_=data.physical_permutations(case.permutation)
    facts=torch.zeros(gym.ATOMS,dtype=torch.bool)
    truths=torch.zeros_like(facts)
    for v in initial:facts[physical[v]]=True
    for v in reached:truths[physical[v]]=True
    return facts,truths


def terminal_truth(cases: list[gym.Case]):
    return torch.stack([closure_bounds(c)[1].long() for c in cases])


def admissible_status(logits:torch.Tensor,cases:list[gym.Case]):
    """AF-13's identical hard-scoring rule with AF-14 layout binding."""
    if logits.ndim!=4 or logits.shape[0]!=len(cases) or logits.shape[2:]!=(gym.ATOMS,2):
        raise ValueError('AF14 status logits/case mismatch')
    pred=logits.argmax(-1).bool()
    facts=torch.stack([closure_bounds(c)[0] for c in cases])[:,None,:]
    reached=torch.stack([closure_bounds(c)[1] for c in cases])[:,None,:]
    false_pos=pred & ~reached
    fact_loss=facts & ~pred
    decreases=pred[:,:-1] & ~pred[:,1:]
    final=(pred[:,-1]==reached[:,0]).float().mean()
    return {'unsupported_true_count':int(false_pos.sum()),
      'missing_fact_count':int(fact_loss.sum()),
      'monotonicity_violation_count':int(decreases.sum()),
      'end_atom_accuracy':float(final),
      'fully_admissible_fraction':float((~false_pos.any(-1)&~fact_loss.any(-1)).all(-1).float().mean())}

def progress_counts(logits: torch.Tensor, cases: list[gym.Case]) -> list[dict[str,int]]:
    """Hard exact deadline correctness and all-postdeadline stability counts."""
    if logits.ndim!=4 or logits.shape[:1]!=(len(cases),) or logits.shape[2:]!=(gym.ATOMS,2):
        raise ValueError('AF14 logits/case shape mismatch')
    result=[]
    labels=logits.argmax(-1)
    length=logits.shape[1]
    for i,c in enumerate(cases):
        counts={'deadline_atom_count':0,'deadline_met_count':0,'deadline_stable_count':0}
        for node,deadline in enumerate(progress_deadlines(c)):
            if not 1<=deadline<=length: continue
            tail=labels[i,deadline-1:,node]
            counts['deadline_atom_count']+=1
            counts['deadline_met_count']+=int(tail[0].item()==1)
            counts['deadline_stable_count']+=int(bool((tail==1).all()))
        result.append(counts)
    return result


def bounded_progress_loss(trace: AsyncTrace, cases: list[gym.Case], arm: str):
    """Compute-matched additional semantically grounded closure-deadline loss.

    All arms receive the same source calls, cases, optimizer budget, donors,
    upstream horizon and downstream patch objective. The bounded arm adds
    margin penalties at and *after* each node's deadline, allowing arrival
    at any earlier time. There is no trajectory-valued exact-wave target.
    """
    if arm not in ('relay_final','async_sound','bounded_progress'):
        raise ValueError('AF14 unregistered arm')
    baseline='relay_final' if arm=='relay_final' else 'async_sound'
    parent,terms=asynchronous_loss(trace,cases,baseline)
    terms={**terms,'bounded_margin':0.,'deadline_atoms':0}
    if arm!='bounded_progress':return parent,terms
    step=trace.logits.shape[1]
    rows=[];eligible_count=0
    for i,case in enumerate(cases):
        margins=trace.logits[i,...,1]-trace.logits[i,...,0]
        per_case=[]
        for atom,deadline in enumerate(progress_deadlines(case)):
            if not 1<=deadline<=step: continue
            # Mean across the suffix, then across reachable atoms, then cases.
            per_case.append(F.softplus(MARGIN-margins[deadline-1:,atom]).mean())
            eligible_count+=1
        rows.append(torch.stack(per_case).mean() if per_case else margins.sum()*0.)
    value=torch.stack(rows).mean()
    terms.update({'bounded_margin':float(value.detach()),'deadline_atoms':eligible_count})
    return parent + PROGRESS_WEIGHT * value,terms
