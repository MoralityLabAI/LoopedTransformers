"""AF-17: native source-hash-pinned TypedLoop plus rule-cost and STOP heads.

The original TypedLoop code and AF-15 TypedHornPolicy source remain unchanged.
All training arms use the same parameters and action-inference implementation.
"""
from __future__ import annotations
import torch
from torch import nn
from experiments.activation_funnel_af15.model import TypedHornPolicy
from experiments.activation_funnel_af05 import gym as base

class BellmanTypedPolicy(TypedHornPolicy):
    def __init__(self):
        super().__init__(width=16,loops=2)
        self.distance_head=nn.Linear(16,1)
        self.reach_head=nn.Linear(16,1)

    def full(self,batch,known):
        if known.ndim!=2 or known.shape[1]!=base.ATOMS:
            raise ValueError('state must be 15 physical known bits')
        af=torch.cat([batch.atom_features,known.to(batch.atom_features.dtype).unsqueeze(-1)],-1)
        a=self.atom(af);r=self.rule(batch.rule_features)
        prem=batch.prem_up+batch.prem_down;head=batch.head_up+batch.head_down
        for _ in range(self.loops):a,r=self.cell(a,r,prem,head)
        goal=a.gather(1,batch.read_slots[:,2:3,None].expand(-1,-1,self.width)).squeeze(1)
        # Positive reach logit suppresses STOP in the *shared* action head.
        reach=self.reach_head(goal).squeeze(-1)
        logits=torch.cat([self.rule_head(r).squeeze(-1),
                          self.stop_head(goal)-.75*reach[:,None]],-1)
        distance=self.distance_head(goal).squeeze(-1)
        return logits,distance,reach

    def forward(self,batch,known):
        return self.full(batch,known)[0]
