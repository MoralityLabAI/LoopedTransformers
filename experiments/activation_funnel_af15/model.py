"""Small TypedLoop rule policy with a host-owned Horn action gate."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from experiments.activation_funnel_af02.model import native_cell_class
from experiments.activation_funnel_af05 import gym as base
from . import gym

class TypedHornPolicy(nn.Module):
    def __init__(self, width=16, loops=2):
        super().__init__()
        self.width,self.loops=width,loops
        self.atom=nn.Linear(6,width)
        self.rule=nn.Linear(3,width)
        self.cell=native_cell_class()(width)
        self.rule_head=nn.Linear(width,1)
        self.stop_head=nn.Linear(width,1)

    def forward(self,batch,known):
        """Raw physical-rule logits and STOP; eligibility masking is host-side."""
        if known.ndim!=2 or known.shape[1]!=base.ATOMS:
            raise ValueError('known symbolic state must be physical B x atoms')
        af=torch.cat([batch.atom_features,known.to(batch.atom_features.dtype).unsqueeze(-1)],-1)
        a=self.atom(af);r=self.rule(batch.rule_features)
        prem=batch.prem_up+batch.prem_down;head=batch.head_up+batch.head_down
        for _ in range(self.loops):a,r=self.cell(a,r,prem,head)
        stop=a.gather(1,batch.read_slots[:,2:3,None].expand(-1,-1,self.width)).squeeze(1)
        return torch.cat([self.rule_head(r).squeeze(-1),self.stop_head(stop)],-1)


def observations(cases,states):
    if len(cases)!=len(states):raise ValueError('AF-15 states/cases mismatch')
    x=gym.stack_cases(cases)
    bits=torch.zeros(len(cases),base.ATOMS,dtype=torch.float32)
    gate=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    for i,(case,state) in enumerate(zip(cases,states)):
        atom,_=gym.physical_maps(case.permutation)
        for logical in state:bits[i,atom[logical]]=1
        for k in gym.enabled(case,state):gate[i,k]=True
        gate[i,gym.STOP]=True
    return x,bits,gate


def state_inputs(cases,states):
    x,bits,gate=observations(cases,states)
    positive=torch.zeros_like(gate)
    for i,(case,state) in enumerate(zip(cases,states)):
        for k in gym.optimal_actions(case,state):positive[i,k]=True
        if not bool((positive[i]&gate[i]).any()):raise ValueError('no optimal legal action')
    return x,bits,gate,positive


def candidate_logits(model,batch,bits,gate):
    return model(batch,bits).masked_fill(~gate,-1e4)


def objective(logits,gate,optimal,arm):
    if arm not in ('imitation','frontier'):
        raise ValueError('unknown AF-15 training arm')
    logits=logits.masked_fill(~gate,-1e4)
    objective=(-torch.logsumexp(logits.masked_fill(~optimal,-1e4),-1)
               +torch.logsumexp(logits,-1)).mean()
    if arm=='imitation':return objective,{'imitation':float(objective.detach()),'frontier':0.}
    terms=[]
    for row in range(len(logits)):
        pos=logits[row,optimal[row]]
        neg=logits[row,gate[row]&~optimal[row]]
        if neg.numel():
            terms.append(F.softplus(0.35-pos[:,None]+neg[None,:]).mean())
    extra=torch.stack(terms).mean() if terms else objective*0.
    return objective+.5*extra,{'imitation':float(objective.detach()),'frontier':float(extra.detach())}
