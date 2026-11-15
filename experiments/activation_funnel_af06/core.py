"""AF-06 mixed addressed / irrelevant subgoal code intervention.

The original TypedLoop and AF-05 GraphLooper remain unchanged. This wrapper
patches exactly one carried Q atom token per row, with a semantically valid
counterfactual target for addressed interventions and the recipient truth for
irrelevant controls. Every row performs the same recurrence calls.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05.model import GraphLooper, State, Output
from experiments.activation_funnel_af05 import gym


def ramp_mask(batch_size: int, step: int, horizon: int, permutation: torch.Tensor) -> torch.Tensor:
    """A deterministic, monotone count of addressed patches, 0 to B.

    The supplied permutation is generated from a separate fixed random stream
    and shared between continuation arms. This avoids mixing incompatible
    oracle labels for an already patched state.
    """
    if horizon < 2 or not 0 <= step < horizon or permutation.shape != (batch_size,):
        raise ValueError('invalid staged schedule')
    if sorted(permutation.tolist()) != list(range(batch_size)):
        raise ValueError('expected a row permutation')
    count = round(batch_size * step / (horizon - 1))
    return permutation < count


def mixed_patch(model: GraphLooper, recipient: gym.GraphBatch,
                donor: gym.GraphBatch, selected: torch.Tensor,
                addressed: torch.Tensor, pre_loops: int):
    """Return a differentiable model output plus the exact patched atom code.

    Caller must independently compute target via Horn closure, never student
    predictions. Each row chooses addressed Q_j or irrelevant Q_(1-j).
    """
    if addressed.shape != selected.shape or selected.shape != (len(recipient.targets),):
        raise ValueError('mixed patch selector shape mismatch')
    if addressed.dtype != torch.bool or not bool(((selected == 0) | (selected == 1)).all()):
        raise ValueError('bad patch selector values')
    for key in ('rule_features','prem_up','head_up','prem_down','head_down','read_slots'):
        if not torch.equal(getattr(recipient,key), getattr(donor,key)):
            raise ValueError('structurally invalid matched donor: '+key)
    r = model.prefix(recipient,pre_loops)
    d = model.prefix(donor,pre_loops)
    selection = torch.where(addressed,selected,1-selected)
    slot = recipient.read_slots.gather(1,selection[:,None]).squeeze(1)
    mask = F.one_hot(slot,num_classes=gym.ATOMS).bool().unsqueeze(-1)
    az = torch.where(mask,d.az,r.az)
    rz = r.rz
    a, rules = model.codec.decode(az,rz)
    goal,q,post = model.suffix(recipient,State(a,rules,az,rz))
    initial,_ = model._read(r.a,recipient.read_slots)
    return Output(goal,q,initial,State(a,rules,az,rz),post)


def mixed_targets(original: torch.Tensor, counterfactual: torch.Tensor,
                  addressed: torch.Tensor) -> torch.Tensor:
    if original.shape != counterfactual.shape or original.shape != (len(addressed),3):
        raise ValueError('counterfactual target shape mismatch')
    return torch.where(addressed[:,None],counterfactual,original)
