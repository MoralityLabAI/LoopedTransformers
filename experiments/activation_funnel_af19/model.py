"""AF-19 matched learned STOP readouts over local versus backward-rule ancestry.

Both arms have identical tensors, logits and dense arithmetic; the only difference
is the public graph mask controlling which rule tokens contribute to STOP,
distance and reachability. All action logits and host legality gates are shared.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from experiments.activation_funnel_af17.model import BellmanTypedPolicy
from experiments.activation_funnel_af05 import gym as base

MODES = ('local','ancestor')
LOOPS = (2,4)


def relevance_masks(batch):
    """Return direct goal-concluding rules and every transitive ancestor rule.

    Operates exclusively on public Horn incidence. No model-derived proof status,
    target or oracle fact is consulted. Both arms evaluate this same algorithm.
    """
    prem=batch.prem_up+batch.prem_down
    head=batch.head_up+batch.head_down
    if prem.ndim!=3 or head.shape!=prem.shape:
        raise ValueError('batched physical incidence must share [B,R,A]')
    batch_size,rule_count,atom_count=prem.shape
    if rule_count!=base.RULES or atom_count!=base.ATOMS:
        raise ValueError('unexpected atom/rule dimensions')
    goal=batch.read_slots[:,2]
    if goal.shape!=(batch_size,):raise ValueError('readout goal slots missing')
    root=torch.nn.functional.one_hot(goal,num_classes=atom_count).bool()
    nonempty=(head.sum(-1)>0)
    local=((head*root.unsqueeze(1)).sum(-1)>0) & nonempty
    active_atoms=root
    for _ in range(atom_count):
        active_rules=((head*active_atoms.unsqueeze(1)).sum(-1)>0) & nonempty
        upstream=((prem*active_rules.unsqueeze(-1)).sum(1)>0)
        active_atoms=active_atoms|upstream
    ancestor=((head*active_atoms.unsqueeze(1)).sum(-1)>0) & nonempty
    if not torch.all(ancestor | ~local):
        raise AssertionError('local goal rules must lie in goal ancestry')
    return local,ancestor


class ReadoutPolicy(BellmanTypedPolicy):
    def __init__(self, mode: str, loops: int=2):
        if mode not in MODES or loops not in LOOPS:
            raise ValueError('unregistered readout mode or loop count')
        super().__init__()
        self.mode=mode
        self.loops=loops
        self.summary_query=nn.Linear(self.width,self.width,bias=False)
        self.summary_key=nn.Linear(self.width,self.width,bias=False)
        self.summary_value=nn.Linear(self.width,self.width,bias=False)

    def summary(self, batch, goal, rules):
        local,ancestor=relevance_masks(batch)
        mask=local if self.mode=='local' else ancestor
        if not bool(mask.any(-1).all()):
            raise ValueError('goal must have at least one concluding rule')
        q=self.summary_query(goal)
        k=self.summary_key(rules)
        score=(k*q.unsqueeze(1)).sum(-1)/math.sqrt(self.width)
        weights=torch.softmax(score.masked_fill(~mask,-1e4),-1)
        return (weights.unsqueeze(-1)*self.summary_value(rules)).sum(1)

    def full(self,batch,known):
        if known.ndim!=2 or known.shape[1]!=base.ATOMS:
            raise ValueError('expected Bx15 host-known physical bits')
        af=torch.cat([batch.atom_features,known.to(batch.atom_features.dtype).unsqueeze(-1)],-1)
        atoms=self.atom(af);rules=self.rule(batch.rule_features)
        prem=batch.prem_up+batch.prem_down;head=batch.head_up+batch.head_down
        for _ in range(self.loops):
            atoms,rules=self.cell(atoms,rules,prem,head)
        goal=atoms.gather(1,batch.read_slots[:,2:3,None].expand(-1,-1,self.width)).squeeze(1)
        read=goal+self.summary(batch,goal,rules)
        reach=self.reach_head(read).squeeze(-1)
        raw=torch.cat([self.rule_head(rules).squeeze(-1),self.stop_head(read)-.75*reach[:,None]],-1)
        distance=self.distance_head(read).squeeze(-1)
        return raw,distance,reach

    def forward(self,batch,known):return self.full(batch,known)[0]
