"""AF-09: exact signed semantic margin transport for the original GraphLooper.

For the actual protected-Q carried code z and intervened code z', the original
atom decoder and Q status readout are affine. Their true-class signed margin
therefore obeys |m_y(z')-m_y(z)| <= K_y ||z'-z||_2, with
K_y=||(W_R[y]-W_R[1-y]) W_D||_2. This gives a testable, sufficient
certificate for correct protected-state classification under intervention.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from experiments.activation_funnel_af07.core import (
    Trajectory, trace_suffix, patch_prefix, locality_components,
    step_status_loss,
)


@dataclass
class Triple:
    clean: Trajectory
    patched: Trajectory
    control: Trajectory
    addressed: torch.Tensor


def triple_traces(model, receiver, donor, selected, addressed, pre_loops, post_loops=3):
    """Two prefixes, then clean/addressed-or-other/irrelevant-control suffixes."""
    r=model.prefix(receiver,pre_loops)
    d=model.prefix(donor,pre_loops)
    clean=trace_suffix(model,receiver,r,post_loops)
    p=patch_prefix(model,receiver,donor,selected,addressed,r,d)
    changed=trace_suffix(model,receiver,p,post_loops)
    other=patch_prefix(model,receiver,donor,selected,
                       torch.zeros_like(addressed),r,d)
    control=trace_suffix(model,receiver,other,post_loops)
    return Triple(clean,changed,control,addressed)


def qualified(metrics_by_split,minimum=.90):
    return all(metrics_by_split[split]['goal_accuracy']>=minimum and
               metrics_by_split[split]['subgoal_exact']>=minimum
               for split in ('train','dev'))


def true_margin(logits:torch.Tensor, truth:torch.Tensor):
    """Compute exact two-class true-label logit advantage per row and step."""
    if logits.ndim!=3 or logits.shape[2]!=2 or truth.shape!=(len(logits),):
        raise ValueError('logit/truth shape mismatch')
    if not bool(((truth==0)|(truth==1)).all()):
        raise ValueError('non-Boolean ground truth')
    desired=logits.gather(2,truth[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
    other=logits.gather(2,(1-truth)[:,None,None].expand(-1,logits.shape[1],1)).squeeze(-1)
    return desired-other


def certificate_tensors(model,clean:Trajectory,patched:Trajectory,
                        selected:torch.Tensor,recipient_truth:torch.Tensor):
    """Return every term in the exact protected-Q affine transport inequality.

    Only the Q *other than* the selected addressed patch is protected. The
    recipient's original Horn truth supplies the signed class, never a neural
    prediction. Actual carried codes, rather than decoded working states,
    provide the displacement. Works with arbitrary downstream loop length.
    """
    if selected.shape!=(len(clean.q),) or recipient_truth.shape!=(len(clean.q),3):
        raise ValueError('wrong certificate selector or exact Horn truth')
    if clean.q_steps.shape!=patched.q_steps.shape or clean.q_codes.shape!=patched.q_codes.shape:
        raise ValueError('mismatched clean/patched recurrent traces')
    b,t,_,w=clean.q_codes.shape
    protected=1-selected
    if not bool(((selected==0)|(selected==1)).all()):
        raise ValueError('invalid selected subgoal')
    qidx=protected[:,None,None,None].expand(b,t,1,2)
    zidx=protected[:,None,None,None].expand(b,t,1,w)
    clean_logits=clean.q_steps.gather(2,qidx).squeeze(2)
    patch_logits=patched.q_steps.gather(2,qidx).squeeze(2)
    z=clean.q_codes.gather(2,zidx).squeeze(2)
    zp=patched.q_codes.gather(2,zidx).squeeze(2)
    y=recipient_truth.gather(1,protected[:,None]).squeeze(1)
    m_clean=true_margin(clean_logits,y)
    m_patch=true_margin(patch_logits,y)
    dz=torch.linalg.vector_norm(zp-z,dim=-1)
    readout=model.subgoal_head.weight  # 2 x 16
    decoder=model.codec.atom.decoder.weight  # 16 x code_width
    v=readout[y]-readout[1-y]
    sensitivity=torch.linalg.vector_norm(v @ decoder,dim=-1)
    bound=m_clean-sensitivity[:,None]*dz
    return {'protected':protected,'truth':y,'clean_code':z,'patched_code':zp,
            'clean_margin':m_clean,'patched_margin':m_patch,
            'displacement':dz,'sensitivity':sensitivity,'lower_bound':bound}


def semantic_margin_loss(model, clean, patched, selected, recipient_truth,
                         addressed, *, target_margin=.5):
    """Softplus avoids zero-gradient saturation of AF-08's prototype hinge.

    The true-label certificate term puts gradient on clean semantic margin,
    actual protected-code transport and affine readout sensitivity. A direct
    patched signed-margin term discourages a numerically stable wrong answer.
    Inactive addressed rows contribute no training signal.
    """
    terms=certificate_tensors(model,clean,patched,selected,recipient_truth)
    if addressed.shape!=(len(selected),) or addressed.dtype!=torch.bool:
        raise ValueError('addressed mask must be per-row Boolean')
    if target_margin<=0:raise ValueError('positive semantic target margin required')
    if not bool(addressed.any()):
        zero=terms['lower_bound'].sum()*0.
        return zero,zero,terms
    certified=F.softplus(target_margin-terms['lower_bound'][addressed]).mean()
    actual=F.softplus(target_margin-terms['patched_margin'][addressed]).mean()
    return certified,actual,terms


def certificate_rows(model,clean,patched,selected,recipient_truth,*,tol=1e-4):
    """Materialize testable certificate numbers without a neural-label oracle."""
    with torch.no_grad():
        x=certificate_tensors(model,clean,patched,selected,recipient_truth)
        a=x['clean_margin'];b=x['patched_margin']
        distance=x['sensitivity'][:,None]*x['displacement']
        if not bool((torch.abs(b-a)<=distance+tol).all()):
            raise ArithmeticError('affine semantic-margin certificate violated')
        certified=x['lower_bound']>tol
        correct=b>0
        if bool((certified & ~correct).any()):
            raise ArithmeticError('false positive protected-subgoal certificate')
        out=[]
        for i in range(len(selected)):
            out.append({'truth_protected':int(x['truth'][i]),
                        'clean_code':x['clean_code'][i].tolist(),
                        'patched_code':x['patched_code'][i].tolist(),
                        'clean_true_margin':a[i].tolist(),
                        'patched_true_margin':b[i].tolist(),
                        'code_distance':x['displacement'][i].tolist(),
                        'exact_sensitivity':float(x['sensitivity'][i]),
                        'certified_lower_bound':x['lower_bound'][i].tolist(),
                        'certificate_positive':certified[i].tolist(),
                        'patched_correct':correct[i].tolist(),
                        'clean_correct':(a[i]>0).tolist()})
        return out
