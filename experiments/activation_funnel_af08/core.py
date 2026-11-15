"""AF-08: qualified-source causal locality with status-conditional noncollapse.

All intervention labels are computed by the unchanged AF-05 Horn oracle. Q
codes and logits come from the original hash-pinned TypedLoop implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F

from experiments.activation_funnel_af07.core import (
    Trajectory, trace_suffix, patch_prefix, locality_components, step_status_loss,
    protect_index,
)


@dataclass
class Triple:
    clean: Trajectory
    patched: Trajectory
    control: Trajectory
    addressed: torch.Tensor


def triple_traces(model, receiver, donor, selected, addressed, pre_loops, post_loops=3):
    """Equal source forwards for each arm: two prefixes, three suffixes.

    `addressed=False` turns the training intervention into an irrelevant-subgoal
    control. The second suffix is always that control. Its semantic label stays
    the recipient truth. The clean, primary-patch and irrelevant-control suffixes are all executed.
    """
    r=model.prefix(receiver, pre_loops)
    d=model.prefix(donor, pre_loops)
    clean=trace_suffix(model, receiver, r, post_loops)
    p=patch_prefix(model,receiver,donor,selected,addressed,r,d)
    changed=trace_suffix(model,receiver,p,post_loops)
    control_state=patch_prefix(model,receiver,donor,selected,
                               torch.zeros_like(addressed),r,d)
    control=trace_suffix(model,receiver,control_state,post_loops)
    return Triple(clean,changed,control,addressed)


def class_separation(codes: torch.Tensor, truth: torch.Tensor, *, margin=0.45):
    """Label-conditional prototype separation on *actual carried codes*.

    Codes are B,T,2,W and truth is B,2. Opposing truth classes should be
    separated in each subgoal's representation at every stage. Loss is zero
    when a batch lacks both classes. The stage mean is over represented
    classes only, preventing a spurious penalty from empty categories.
    """
    if codes.ndim!=4 or codes.shape[2]!=2 or truth.shape!=(codes.shape[0],2):
        raise ValueError('noncollapse requires code B,T,2,W and Horn truth B,2')
    if not bool(((truth==0)|(truth==1)).all()):
        raise ValueError('invalid Boolean truth')
    if margin<=0:raise ValueError('positive separation margin required')
    terms=[]
    for j in range(2):
        neg=truth[:,j]==0
        pos=truth[:,j]==1
        if bool(neg.any()) and bool(pos.any()):
            a=codes[neg,:,j].mean(0)
            b=codes[pos,:,j].mean(0)
            distance=torch.linalg.vector_norm(a-b,dim=-1)
            terms.append(F.relu(margin-distance).square().mean())
    return torch.stack(terms).mean() if terms else codes.sum()*0.0


def selected_code_change(triple: Triple, selected, recipient_truth, hybrid_truth,
                         *, margin=0.35):
    """Detect status-changing addressed patches whose code collapsed.

    Only status-changing selected subgoals contribute. An unchanged donor
    truth cannot be used to manufacture a contrastive label. The predicted
    semantic state is supervised separately by the training CE.
    """
    if selected.shape!=(len(triple.clean.goal),) or hybrid_truth.shape!=recipient_truth.shape:
        raise ValueError('bad selected-code comparison')
    actual=recipient_truth.gather(1,selected[:,None]).squeeze(1)
    changed=hybrid_truth.gather(1,selected[:,None]).squeeze(1)
    used=triple.addressed & (actual!=changed)
    if not bool(used.any()):return triple.clean.q_codes.sum()*0.
    width=triple.clean.q_codes.shape[-1]
    pick=selected[:,None,None,None].expand(-1,triple.clean.q_codes.shape[1],1,width)
    base=triple.clean.q_codes.gather(2,pick).squeeze(2)
    swapped=triple.patched.q_codes.gather(2,pick).squeeze(2)
    dist=torch.linalg.vector_norm(swapped-base,dim=-1)
    return F.relu(margin-dist[used]).square().mean()


def qualified(metrics_by_split, minimum=.9):
    return all(metrics_by_split[split]['goal_accuracy']>=minimum and
               metrics_by_split[split]['subgoal_exact']>=minimum
               for split in ('train','dev'))


def code_separation_scores(codes: torch.Tensor, truth: torch.Tensor):
    """No-label-leak evaluation of within-batch code class centroid distances."""
    if codes.ndim!=4 or truth.shape!=(len(codes),2):raise ValueError('bad code score')
    out={}
    for j in range(2):
        neg=truth[:,j]==0;pos=truth[:,j]==1
        if bool(neg.any()) and bool(pos.any()):
            c0=codes[neg,:,j].mean(0)
            c1=codes[pos,:,j].mean(0)
            out[f'q{j}']=torch.linalg.vector_norm(c0-c1,dim=-1).tolist()
        else:out[f'q{j}']=None
    return out
