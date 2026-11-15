"""AF-07: recurrence-resolved subgoal-locality interventions.

GraphLooper and its pinned original TypedLoop cell remain unchanged. Donor patches
operate on the *actual carried atom code* at the upstream/downstream boundary.
This module returns the semantic Q logits and carried Q codes after every
subsequent native step; it never reads target labels to build a state.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, State


@dataclass
class Trajectory:
    goal: torch.Tensor                   # [batch,2] final G logits
    q: torch.Tensor                      # [batch,2,2] final Q logits
    initial_q: torch.Tensor              # [batch,2,2] before stage 2
    q_steps: torch.Tensor                # [batch,steps,2,2]
    goal_steps: torch.Tensor             # [batch,steps,2]
    q_codes: torch.Tensor                # [batch,steps,2,code_width]
    prefix: State
    final: State


def trace_suffix(model: GraphLooper, batch: gym.GraphBatch,
                 start: State, steps: int = 3) -> Trajectory:
    if not 1 <= steps <= 7:
        raise ValueError('AF-07 trace steps outside registered 1..7')
    a, r = start.a, start.r
    initial_q, _ = model._read(a, batch.read_slots)
    qs, gs, codes = [], [], []
    for _ in range(steps):
        s = model._step(a, r, batch.prem_down, batch.head_down)
        q, goal = model._read(s.a, batch.read_slots)
        qs.append(q)
        gs.append(goal)
        codes.append(s.az.gather(1, batch.read_slots[:, :2].unsqueeze(-1).expand(-1,-1,model.code_width)))
        a, r = s.a, s.r
    return Trajectory(gs[-1], qs[-1], initial_q,torch.stack(qs,1),
                      torch.stack(gs,1),torch.stack(codes,1),start,s)


def patch_prefix(model: GraphLooper, recipient: gym.GraphBatch,
                 donor: gym.GraphBatch, selected: torch.Tensor,
                 addressed: torch.Tensor, r: State, d: State) -> State:
    n=len(recipient.targets)
    if selected.shape!=(n,) or addressed.shape!=(n,) or addressed.dtype!=torch.bool:
        raise ValueError('bad intervention selector')
    if not bool(((selected==0)|(selected==1)).all()):
        raise ValueError('invalid branch selector')
    for name in ('rule_features','prem_up','head_up','prem_down','head_down','read_slots'):
        if not torch.equal(getattr(recipient,name),getattr(donor,name)):
            raise ValueError('structural donor mismatch: '+name)
    selection=torch.where(addressed,selected,1-selected)
    slot=recipient.read_slots.gather(1,selection[:,None]).squeeze(1)
    mask=F.one_hot(slot,num_classes=gym.ATOMS).bool().unsqueeze(-1)
    az=torch.where(mask,d.az,r.az)
    rz=r.rz
    a,rule=model.codec.decode(az,rz)
    return State(a,rule,az,rz)


def paired_traces(model: GraphLooper, receiver: gym.GraphBatch,
                  donor: gym.GraphBatch, selected: torch.Tensor,
                  addressed: torch.Tensor, pre_loops: int,
                  post_loops: int = 3) -> tuple[Trajectory,Trajectory]:
    r=model.prefix(receiver,pre_loops)
    d=model.prefix(donor,pre_loops)
    clean=trace_suffix(model,receiver,r,post_loops)
    p=patch_prefix(model,receiver,donor,selected,addressed,r,d)
    patched=trace_suffix(model,receiver,p,post_loops)
    return clean,patched


def protect_index(selected: torch.Tensor,addressed: torch.Tensor) -> torch.Tensor:
    return torch.where(addressed,1-selected,selected)


def locality_components(clean: Trajectory, patched: Trajectory,
                        selected: torch.Tensor, addressed: torch.Tensor):
    """Penalize drift of the semantically untouched Q after every downstream step.

    A shared goal may change following an addressed patch. Only the other Q
    atom and its *probability* output are constrained. The clean trajectory
    is detached in this regularizer, while clean-label CE is trained elsewhere.
    """
    idx=protect_index(selected,addressed)
    if clean.q_steps.shape!=patched.q_steps.shape or idx.shape!=(len(clean.q),):
        raise ValueError('bad traced comparison')
    index=idx[:,None,None,None].expand(-1,clean.q_steps.shape[1],1,2)
    clean_logit=clean.q_steps.gather(2,index).squeeze(2)
    patch_logit=patched.q_steps.gather(2,index).squeeze(2)
    probability=F.kl_div(F.log_softmax(patch_logit,-1),
                         F.softmax(clean_logit.detach(),-1),reduction='batchmean')/clean.q_steps.shape[1]
    cidx=idx[:,None,None,None].expand(-1,clean.q_codes.shape[1],1,clean.q_codes.shape[-1])
    clean_code=clean.q_codes.gather(2,cidx).squeeze(2)
    patch_code=patched.q_codes.gather(2,cidx).squeeze(2)
    code=(patch_code-clean_code.detach()).square().mean()
    return probability, code


def step_status_loss(trace: Trajectory, target: torch.Tensor):
    if target.shape!=(len(trace.goal),3):
        raise ValueError('expected B,3 exact Horn truth')
    return torch.stack([F.cross_entropy(trace.q_steps[:,t].reshape(-1,2),
                                          target[:,:2].reshape(-1))
                        for t in range(trace.q_steps.shape[1])]).mean()


def trace_metrics(clean: Trajectory, patched: Trajectory,
                  truth: torch.Tensor, composed: torch.Tensor,
                  selected: torch.Tensor, addressed: torch.Tensor) -> dict:
    """Per-row outputs for independent numerical replay and metric aggregation."""
    pred=patched.q_steps.argmax(-1)
    protected=protect_index(selected,addressed)
    exact=pred.gather(2,protected[:,None,None].expand(-1,pred.shape[1],1)).squeeze(-1)
    d=(patched.q_codes-clean.q_codes).square().sum(-1).sqrt()
    other=d.gather(2,protected[:,None,None].expand(-1,d.shape[1],1)).squeeze(-1)
    expected=torch.where(addressed[:,None],composed,truth)
    stage=(pred==expected[:,:2].unsqueeze(1))
    return {'protected_q_correct_by_step':(exact==truth.gather(1,protected[:,None])).tolist(),
            'protected_code_l2_by_step':other.tolist(),
            'patched_q_correct_by_step':stage.tolist(),
            'patched_goal_by_step':patched.goal_steps.argmax(-1).tolist()}
