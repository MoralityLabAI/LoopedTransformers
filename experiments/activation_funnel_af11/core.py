"""AF-11: structural proof-state cut with supervised Boolean influence.

The exact finite difference for one swapped Q is
G(q_j^donor,q_other)-G(q_j^recipient)=(q_j^donor-q_j^recipient) I_j,
I_j=q_other for AND and I_j=1-q_other for OR.
This arithmetic is an oracle target, never a model input.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af10.core import paired as causal_paired


def influence(q:torch.Tensor,selected:torch.Tensor,gate:torch.Tensor):
    """Exact Boolean derivative of AND/OR at the untouched subgoal."""
    if q.ndim!=2 or q.shape[1]!=2 or selected.shape!=(len(q),) or gate.shape!=(len(q),):
        raise ValueError('bad Boolean influence input shapes')
    if not bool(((q==0)|(q==1)).all()) or not bool(((selected==0)|(selected==1)).all()) or not bool(((gate==0)|(gate==1)).all()):
        raise ValueError('non-Boolean Q, selected, or gate')
    other=q.gather(1,(1-selected)[:,None]).squeeze(1)
    return torch.where(gate==0,other,1-other)  # 0 AND, 1 OR


def signed_goal_difference(recipient:torch.Tensor,composed:torch.Tensor,
                           selected:torch.Tensor,gate:torch.Tensor):
    if recipient.shape!=composed.shape or recipient.ndim!=2 or recipient.shape[1]!=3:
        raise ValueError('expected B,3 truth arrays')
    dq=composed[:,:2].gather(1,selected[:,None]).squeeze(-1)-recipient[:,:2].gather(1,selected[:,None]).squeeze(-1)
    deriv=influence(recipient[:,:2],selected,gate)
    delta=composed[:,2]-recipient[:,2]
    if not bool((dq.abs()==1).all()) or not torch.equal(delta,dq*deriv):
        raise ValueError('Boolean influence target contradicts Horn oracle')
    return delta,deriv


def soft_signed_delta(clean_goal:torch.Tensor,patched_goal:torch.Tensor):
    if clean_goal.shape!=patched_goal.shape or clean_goal.shape[-1]!=2:
        raise ValueError('goal logits must have two classes')
    return F.softmax(patched_goal,dim=-1)[:,1]-F.softmax(clean_goal,dim=-1)[:,1]


def influence_loss(clean_goal,patched_goal,recipient_truth,hybrid_truth,
                   selected,gate):
    delta,deriv=signed_goal_difference(recipient_truth,hybrid_truth,selected,gate)
    score=soft_signed_delta(clean_goal,patched_goal)
    mse=(score-delta.to(score.dtype)).square().mean()
    # Explicit extra weight on whether the gate was sensitive to the intervention.
    active=deriv.bool()
    if bool(active.any()):
        align=(score[active]*delta[active].to(score.dtype))
        margin=F.softplus(.5-align).mean()
    else:
        margin=score.sum()*0.
    return mse,margin,{'informative':int(active.sum()),'mean_signed_delta_error':float((score-delta).abs().detach().mean())}


def replay(model,recipient,donor,selected,pre_loops,*,cut):
    # Cut/no-cut run exactly the same source prefixes and three suffixes.
    pair=causal_paired(model,recipient,donor,selected,pre_loops,3,cut=cut)
    if cut:
        # exact invariance of actual carried Q codes at every stage, not just final step
        expected=pair.patched.q_codes[:,0]
        if not torch.equal(pair.patched.q_codes,expected.unsqueeze(1).expand_as(pair.patched.q_codes)):
            raise AssertionError('causal-cut code changed across downstream recurrence')
    return pair
