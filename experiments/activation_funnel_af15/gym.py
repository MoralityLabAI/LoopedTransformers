"""AF-15: host-gated, interpretable Horn rule-selection gym.

The host accepts only a currently enabled rule, records its exact premises, and
monotonically adds its head. A small TypedLoop ranks legal rule fires. Both the
symbolic state and its proof trace are host-owned; the model receives initial
facts, public incidence/roles, and current facts, not oracle outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass,fields
from functools import lru_cache
from itertools import product
import hashlib,json,random
import torch
from experiments.activation_funnel_af05 import gym as base

FRESH=('fresh_depth3','fresh_comp','fresh_depth4','fresh_depth5')
LAYOUTS={'fresh_depth3':(36,), 'fresh_comp':(37,), 'fresh_depth4':(32,33), 'fresh_depth5':(34,35)}
DEPTH={'fresh_depth3':3,'fresh_comp':3,'fresh_depth4':4,'fresh_depth5':5}
STOP=base.RULES


def physical_maps(index):
    if index<8:return base.permutation_maps(index)
    if index not in range(32,38):raise ValueError('unregistered AF-15 physical layout')
    g=random.Random(8152026+index*2317)
    a=list(range(base.ATOMS));r=list(range(base.RULES));g.shuffle(a);g.shuffle(r)
    return tuple(a),tuple(r)


def encode(case):
    if case.permutation<8:return base.encode_case(case)
    up,down=base.logical_rules(case)
    apos,rpos=physical_maps(case.permutation)
    af=torch.zeros(base.ATOMS,5);rf=torch.zeros(base.RULES,3)
    pu=torch.zeros(base.RULES,base.ATOMS);hu=pu.clone();pd=pu.clone();hd=pu.clone()
    facts=base.initial_facts(case)
    for logical in range(base.ATOMS):
        idx=apos[logical]
        af[idx]=torch.tensor([float(logical in facts),float(logical==base.SUBGOALS[0]),
          float(logical==base.SUBGOALS[1]),float(logical==base.GOAL),1.])
    for k,(prem,head) in enumerate(up+down):
        j=rpos[k];rf[j]=torch.tensor([len(prem)/2.,float(k>=len(up)),1.])
        p=pu if k<len(up) else pd;h=hu if k<len(up) else hd
        for v in prem:p[j,apos[v]]=1.
        h[j,apos[head]]=1.
    return base.GraphBatch(af,rf,pu,hu,pd,hd,torch.tensor([apos[n] for n in (*base.SUBGOALS,base.GOAL)]),
                           torch.tensor(base.oracle(case)[0],dtype=torch.long))


def sample_hash(case):
    x=encode(case)
    return base.digest([getattr(x,f.name).tolist() for f in fields(base.GraphBatch) if f.name!='targets'])

@lru_cache(maxsize=6)
def cases_for(split):
    if split in ('train','dev'):return base.cases_for(split)
    if split not in FRESH:raise ValueError('unregistered AF-15 split')
    out=[]
    for layout in LAYOUTS[split]:
        for k0,k1,gate,facts in product(base.KINDS,base.KINDS,base.GATES,product((0,1),repeat=4)):
            special=(k0,k1,gate)==base.STRESS_COMPOSITION
            if split=='fresh_comp' and not special:continue
            out.append(base.Case(split,DEPTH[split],(k0,k1),gate,tuple(facts),layout))
    return tuple(out)


def full_rules(case):
    u,d=base.logical_rules(case)
    return u+d


def physical_rules(case):
    _,rpos=physical_maps(case.permutation)
    return {rpos[k]:rule for k,rule in enumerate(full_rules(case))}


def enabled(case,known):
    known=set(known)
    return tuple(sorted(i for i,(prem,head) in physical_rules(case).items()
                      if head not in known and all(p in known for p in prem)))


def sound_fire(case,known,action):
    """Only this host-side transition changes the symbolically certified state."""
    old=frozenset(known)
    if action==STOP:return old,None
    if action not in enabled(case,old):raise ValueError('inadmissible or unsupported Horn rule')
    prem,head=physical_rules(case)[action]
    return old|{head},{'rule_slot':action,'premises':list(prem),'head':head}


def final_closure(case):
    reached,_=base.closure(set(base.initial_facts(case)),full_rules(case))
    return reached


def _rules_key(case):
    return tuple((tuple(p),h) for p,h in full_rules(case))

@lru_cache(maxsize=200000)
def _shortest(rules,known):
    state=set(known)
    if base.GOAL in state:return 0
    choices=[h for prem,h in rules if h not in state and all(x in state for x in prem)]
    if not choices:return 10**6
    return 1+min(_shortest(rules,tuple(sorted(state|{h}))) for h in set(choices))


def optimal_remaining(case,known):
    return _shortest(_rules_key(case),tuple(sorted(known)))


def optimal_actions(case,known):
    """Exact minimum-action oracle for training labels and heldout calibration."""
    shortest=optimal_remaining(case,known)
    if shortest==0 or shortest>=10**6:return (STOP,)
    result=[]
    for i in enabled(case,known):
        next_state,_=sound_fire(case,known,i)
        if optimal_remaining(case,next_state)==shortest-1:result.append(i)
    if not result:raise AssertionError('goal reachable but no optimal enabled rule')
    return tuple(result)


def reference_episode(case):
    """Deterministic shortest-goal path; states are public symbolic states."""
    state=frozenset(base.initial_facts(case));examples=[]
    for _ in range(base.ATOMS+1):
        optimal=optimal_actions(case,state)
        examples.append((state,optimal))
        action=optimal[0]
        if action==STOP:break
        state,_=sound_fire(case,state,action)
    else:raise RuntimeError('Horn reference path failed to stop')
    return examples


def manifest():
    previous=set();seen=set();out={}
    for split in base.SPLITS:
        previous.update(sample_hash(c) for c in base.cases_for(split))
    for split in ('train','dev',*FRESH):
        cases=cases_for(split);hashes=[sample_hash(c) for c in cases]
        if len(hashes)!=len(set(hashes)) or (split in FRESH and any(x in previous or x in seen for x in hashes)):
            raise ValueError('overlapping AF-15 heldout input hashes')
        if split in FRESH:seen.update(hashes)
        out[split]={'count':len(cases),'layouts':list(LAYOUTS[split]) if split in FRESH else None,
                    'input_hashes_sha256':base.digest(hashes),'case_ids_sha256':base.digest([c.id for c in cases])}
    receipt={'experiment':'AF-15','scope':'new physical graph layouts; same Horn logic templates',
             'host_rule_gate':'enabled premises and unproved head only',
             'train_split':['train'],'dev_split':['dev'],'heldout':list(FRESH),'splits':out}
    receipt['manifest_sha256']=base.digest(receipt)
    return receipt


def stack_cases(cases):
    rows=[encode(c) for c in cases]
    if not rows:raise ValueError('empty AF-15 graph batch')
    return base.GraphBatch(*(torch.stack([getattr(r,f.name) for r in rows])
                       for f in fields(base.GraphBatch)))
