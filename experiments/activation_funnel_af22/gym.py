"""AF-22 frozen Horn successor gym, with prospective physical layouts 82-86.

All model inputs contain public incidence, initial facts, and the current host
state. Exact BFS costs are labels and audit records only, never planner inputs.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af21 import gym as parent

STOP,INFINITY,CAP = parent.STOP,parent.INFINITY,16
SEEDS=(17,29,43,61,73,89,103,131)
SPLITS=('train','dev','held_layout','held_depth4','held_depth5')
LAYOUTS={'held_layout':(82,), 'held_depth4':(83,84), 'held_depth5':(85,86)}
HELD_VARIANTS=parent.HELD_VARIANTS

@dataclass(frozen=True)
class Case(parent.Case):
    @property
    def id(self):
        return (f'af22-{self.split}-{self.variant}-d{self.depth}-{self.gate}-'
                f'{"".join(map(str,self.facts))}-p{self.permutation}')


def physical_maps(index):
    if index not in range(82,87):return parent.physical_maps(index)
    rng=random.Random(222222+3191*index)
    atoms=list(range(base.ATOMS));rules=list(range(base.RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)

initial_facts=parent.initial_facts
logical_rules=parent.logical_rules
closure=parent.closure
optimal_remaining=parent.optimal_remaining

@lru_cache(maxsize=4096)
def encode(c):
    if not isinstance(c,Case):return parent.encode(c)
    apos,rpos=physical_maps(c.permutation)
    af=torch.zeros(base.ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(base.RULES,3,dtype=torch.float32)
    pu=torch.zeros(base.RULES,base.ATOMS);hu=pu.clone();pd=pu.clone();hd=pu.clone()
    facts=initial_facts(c)
    for atom in range(base.ATOMS):
        af[apos[atom]]=torch.tensor([float(atom in facts),float(atom==base.SUBGOALS[0]),
              float(atom==base.SUBGOALS[1]),float(atom==base.GOAL),1.])
    for i,(prem,head) in enumerate(logical_rules(c)):
        j=rpos[i];is_down=head==base.GOAL
        rf[j]=torch.tensor([len(prem)/2.,float(is_down),1.])
        pp,hh=(pd,hd) if is_down else (pu,hu)
        for atom in prem:pp[j,apos[atom]]=1.
        hh[j,apos[head]]=1.
    truth=closure(c)
    return base.GraphBatch(af,rf,pu,hu,pd,hd,
        torch.tensor([apos[n] for n in (*base.SUBGOALS,base.GOAL)]),
        torch.tensor([int(n in truth) for n in (*base.SUBGOALS,base.GOAL)],dtype=torch.long))


def case_hash(c):
    x=encode(c)
    return base.digest([getattr(x,f.name).tolist() for f in fields(base.GraphBatch) if f.name!='targets'])

@lru_cache(maxsize=12)
def cases_for(split):
    if split in ('train','dev'):return parent.cases_for(split)
    if split not in LAYOUTS:raise ValueError('AF22 unknown split')
    d=3 if split=='held_layout' else 4 if split=='held_depth4' else 5
    v=('bridged_shortcut',) if split=='held_layout' else HELD_VARIANTS
    return tuple(Case(split,variant,d,g,tuple(facts),layout)
        for variant,g,facts,layout in product(v,('and','or'),product((0,1),repeat=4),LAYOUTS[split]))

@lru_cache(maxsize=4096)
def physical_rules(c):
    if not isinstance(c,Case):return parent.physical_rules(c)
    _,rpos=physical_maps(c.permutation)
    return {rpos[i]:rule for i,rule in enumerate(logical_rules(c))}


def enabled(c,s):
    known=set(s)
    return tuple(sorted(k for k,(p,h) in physical_rules(c).items() if h not in known and set(p)<=known))


def sound_fire(c,s,a):
    known=frozenset(s)
    if a==STOP:return known,None
    if a not in enabled(c,known):raise ValueError('host rejected unsupported Horn action')
    p,h=physical_rules(c)[a]
    return known|{h},{'rule_slot':a,'premises':list(p),'head':h}


def optimal_actions(c,s):
    d=optimal_remaining(c,s)
    if d==0 or d>=INFINITY:return (STOP,)
    return tuple(a for a in enabled(c,s) if optimal_remaining(c,sound_fire(c,s,a)[0])==d-1)


def observations(cases,states,with_labels=False):
    if not cases or len(cases)!=len(states):raise ValueError('AF22 case/state length mismatch')
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
       for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    legal=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool);optimal=legal.clone()
    for i,(c,s) in enumerate(zip(cases,states)):
        apos,_=physical_maps(c.permutation)
        for a in s:bits[i,apos[a]]=1.
        for a in enabled(c,s):legal[i,a]=True
        legal[i,STOP]=True
        if with_labels:
            for a in optimal_actions(c,s):optimal[i,a]=True
    return (x,bits,legal,optimal) if with_labels else (x,bits,legal)


def manifest():
    from experiments.activation_funnel_af16 import gym as af16
    from experiments.activation_funnel_af17 import gym as af17
    from experiments.activation_funnel_af18 import gym as af18
    from experiments.activation_funnel_af19 import gym as af19
    from experiments.activation_funnel_af20 import gym as af20
    old=set()
    for module,splits in [(af16,('novel_layout','novel_depth4','novel_depth5'))]+[
      (m,('held_layout','held_depth4','held_depth5')) for m in (af17,af18,af19,af20,parent)]:
        for s in splits:
            old.update(module.case_hash(c) for c in module.cases_for(s))
    seen=set();rows={}
    for split in SPLITS:
        cases=cases_for(split)
        features=[case_hash(c) for c in cases]
        if len(set(features))!=len(features) or bool(seen.intersection(features)):
            raise ValueError('AF22 duplicate public features')
        if split.startswith('held') and bool(old.intersection(features)):
            raise ValueError('AF22 historical heldout leakage')
        seen.update(features)
        rows[split]={'count':len(cases),'features_sha256':base.digest(features),
          'case_ids_sha256':base.digest([c.id for c in cases]),
          'layouts':sorted({c.permutation for c in cases})}
    out={'experiment':'AF-22','logic':'AF21 cross-dependency Horn templates; novel physical layouts',
         'train_dev':'AF21 exact inherited 928/352 cases','splits':rows,
         'host_rule_gate':True,'no_historical_heldout_feature_overlap':True}
    out['manifest_sha256']=base.digest(out)
    return out
