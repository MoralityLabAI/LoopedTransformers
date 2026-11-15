"""AF-10: typed causal-cut (latched) downstream recurrence.

The original AF-05 GraphLooper and hash-pinned TypedLoop remain unmodified.
The proof-state interface holds both *actual carried* Q atom codes at their
upstream values throughout the downstream stage. This is a structural
noninterference guarantee, independent of training success or confidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from experiments.activation_funnel_af05 import gym
from experiments.activation_funnel_af05.model import GraphLooper, State
from experiments.activation_funnel_af07.core import Trajectory, patch_prefix, trace_suffix


def subgoal_mask(batch:gym.GraphBatch) -> torch.Tensor:
    """Boolean mask of public physical atom slots for Q0 and Q1."""
    p=batch.read_slots[:,:2]
    if p.ndim != 2 or p.shape[-1]!=2 or not bool((p[:,0]!=p[:,1]).all()):
        raise ValueError('invalid distinct subgoal slots')
    if not bool(((p>=0)&(p<gym.ATOMS)).all()):
        raise ValueError('out-of-range subgoal slot')
    return F.one_hot(p,num_classes=gym.ATOMS).bool().any(1).unsqueeze(-1)


def latch_suffix(model:GraphLooper,batch:gym.GraphBatch,start:State,steps:int=3) -> Trajectory:
    """Run ordinary native cell, then restore BOTH Q codes before every decode.

    Replacing the complete actual carried code at the boundary (rather than a
    hidden decoded tensor) creates an exactly testable causal cut. Gradient
    through the selected patched code still reaches downstream goal decisions.
    """
    if steps not in (3,5):raise ValueError('registered post loops are 3 and 5')
    if start.az.shape[-2:]!=(gym.ATOMS,model.code_width):
        raise ValueError('unexpected carried atom-code shape')
    mask=subgoal_mask(batch)
    a,r=start.a,start.r
    initial,_=model._read(a,batch.read_slots)
    qs,gs,codes=[],[],[]
    for _ in range(steps):
        proposal=model._step(a,r,batch.prem_down,batch.head_down)
        az=torch.where(mask,start.az,proposal.az)
        a,r=model.codec.decode(az,proposal.rz)
        state=State(a,r,az,proposal.rz)
        q,g=model._read(state.a,batch.read_slots)
        qs.append(q);gs.append(g)
        codes.append(state.az.gather(1,batch.read_slots[:,:2].unsqueeze(-1).expand(-1,-1,model.code_width)))
    return Trajectory(gs[-1],qs[-1],initial,torch.stack(qs,1),torch.stack(gs,1),
                      torch.stack(codes,1),start,state)


@dataclass
class CutPair:
    clean:Trajectory
    patched:Trajectory
    control:Trajectory
    prefix_clean:State
    prefix_donor:State
    selected:torch.Tensor


def paired(model:GraphLooper,recipient:gym.GraphBatch,donor:gym.GraphBatch,
           selected:torch.Tensor,pre_loops:int,post_loops:int=3,
           *,cut:bool=True) -> CutPair:
    """Same three suffixes and two prefixes for latch and ordinary controls."""
    r=model.prefix(recipient,pre_loops)
    d=model.prefix(donor,pre_loops)
    suffix=latch_suffix if cut else trace_suffix
    addressed=torch.ones(len(selected),dtype=torch.bool,device=selected.device)
    other=torch.zeros_like(addressed)
    p=patch_prefix(model,recipient,donor,selected,addressed,r,d)
    o=patch_prefix(model,recipient,donor,selected,other,r,d)
    return CutPair(suffix(model,recipient,r,post_loops),
                   suffix(model,recipient,p,post_loops),
                   suffix(model,recipient,o,post_loops),r,d,selected)


def exact_prefix_factorization(pair:CutPair, recipient_truth:torch.Tensor,
                               composed_truth:torch.Tensor):
    """Measure source Q, transported Q and downstream G with independent labels."""
    if recipient_truth.shape!=composed_truth.shape or recipient_truth.shape[-1]!=3:
        raise ValueError('mismatched original/composed Horn truth')
    n=len(recipient_truth)
    if pair.selected.shape!=(n,):raise ValueError('wrong selected roster')
    protect=1-pair.selected
    idx=protect[:,None,None,None].expand(n,pair.patched.q_codes.shape[1],1,
                                         pair.patched.q_codes.shape[-1])
    protected_patch=pair.patched.q_codes.gather(2,idx).squeeze(2)
    protected_clean=pair.clean.q_codes.gather(2,idx).squeeze(2)
    delta=(protected_patch-protected_clean).abs().amax(dim=-1)
    prefix_truth=recipient_truth[:,:2]
    pre_pred=pair.clean.initial_q.argmax(-1)
    cut_pred=pair.patched.q.argmax(-1)
    return {
      'prefix_joint_correct':(pre_pred==prefix_truth).all(-1),
      'prefix_protected_correct':pre_pred.gather(1,protect[:,None]).squeeze(1)==prefix_truth.gather(1,protect[:,None]).squeeze(1),
      'post_protected_correct':cut_pred.gather(1,protect[:,None]).squeeze(1)==prefix_truth.gather(1,protect[:,None]).squeeze(1),
      'post_composed_q_exact':(cut_pred==composed_truth[:,:2]).all(-1),
      'post_composed_goal_correct':pair.patched.goal.argmax(-1)==composed_truth[:,2],
      'protected_code_max_difference_by_step':delta,
    }


def cross_subgoal_jacobian(model:GraphLooper,batch:gym.GraphBatch,pre_loops:int,
                            selected:int=0)->torch.Tensor:
    """2x2 ambient Jacobian from selected Q code to *other* Q next-step code.

    It measures whether the original native typed cell contains a reverse
    information path through a shared downstream rule. Zero is structural for
    the latched comparison by construction, not claimed for the original cell.
    """
    if len(batch.targets)!=1 or selected not in (0,1):raise ValueError('single row and valid Q required')
    pre=model.prefix(batch,pre_loops)
    src=int(batch.read_slots[0,selected]);dst=int(batch.read_slots[0,1-selected])
    code=pre.az.detach().clone().requires_grad_()
    raw=model._step(model.codec.atom.decode(code),pre.r.detach(),batch.prem_down,batch.head_down)
    values=raw.az[0,dst]
    jac=[]
    for i in range(model.code_width):
        row=torch.autograd.grad(values[i],code,retain_graph=True)[0][0,src]
        jac.append(row)
    return torch.stack(jac,0)
