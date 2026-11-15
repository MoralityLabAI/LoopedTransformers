"""AF-16: novel Horn rule compositions with host-owned, sound rule firing.

AF-15 legacy train/dev sources are retained. New train/dev cases introduce
cross-support and shortcut structures. Held-out cases use cross-branch rule
premises, shared joins, and new goal shortcuts, under disjoint public layouts.
All labels are computed by a separate exact shortest-path oracle; the network
receives only public facts, incidence, rule-type metadata and the current set.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af15 import gym as af15

STOP=base.RULES
TRAIN_VARIANTS=('swap_support','choice_detour')
HELD_VARIANTS=('cross_requisite','cross_join','shortcut_fork')
SPLITS=('train','dev','novel_layout','novel_depth4','novel_depth5')
LAYOUTS={'train':(40,41),'dev':(42,),'novel_layout':(43,),
         'novel_depth4':(44,45),'novel_depth5':(46,47)}
DEPTHS={'train':(2,3),'dev':(2,3),'novel_layout':(3,),
        'novel_depth4':(4,),'novel_depth5':(5,)}
VARIANTS={'train':TRAIN_VARIANTS,'dev':TRAIN_VARIANTS,
          'novel_layout':TRAIN_VARIANTS,'novel_depth4':HELD_VARIANTS,
          'novel_depth5':HELD_VARIANTS}

@dataclass(frozen=True)
class Case:
    split:str
    variant:str
    depth:int
    gate:str
    facts:tuple[int,int,int,int]
    permutation:int
    @property
    def id(self):
        return f'{self.split}-{self.variant}-d{self.depth}-{self.gate}-{''.join(map(str,self.facts))}-p{self.permutation}'


def physical_maps(index):
    if 0<=index<8:return base.permutation_maps(index)
    if 32<=index<38:return af15.physical_maps(index)
    if index not in range(40,48):raise ValueError('AF-16 unregistered physical layout')
    rng=random.Random(161626+index*3119)
    a=list(range(base.ATOMS));r=list(range(base.RULES))
    rng.shuffle(a);rng.shuffle(r)
    return tuple(a),tuple(r)


def initial_facts(case):
    if not isinstance(case,Case):return base.initial_facts(case)
    return frozenset(k for k,bit in zip((0,1,7,8),case.facts) if bit)


def logical_rules(case):
    if not isinstance(case,Case):return af15.full_rules(case)
    up=[]
    ends=[]
    for start in (0,7):
        prev=start
        # Four internal relays support depths 2-5.
        for j in range(case.depth-1):
            nxt=start+2+j
            up.append(((prev,),nxt))
            prev=nxt
        ends.append(prev)
    a,b=ends
    q0,q1,g=(*base.SUBGOALS,base.GOAL)
    v=case.variant
    if v=='swap_support':
        up.extend([((a,8),q0),((b,1),q1)])
    elif v=='choice_detour':
        up.extend([((a,1),q0),((1,),q0),((b,8),q1)])
    elif v=='cross_requisite':
        up.extend([((a,1),q0),((b,q0),q1)])
    elif v=='cross_join':
        up.extend([((a,b),q0),((b,1),q1)])
    elif v=='shortcut_fork':
        up.extend([((a,1),q0),((b,8),q1)])
    else:raise ValueError('unknown AF-16 composition')
    down=[((q0,q1),g)] if case.gate=='and' else [((q0,),g),((q1,),g)]
    if v=='shortcut_fork':down.append(((a,8),g))
    assert len(up)+len(down)<=base.RULES,(v,len(up)+len(down))
    return tuple(up)+tuple(down)


def complete_closure(case):
    reached,_=base.closure(initial_facts(case),logical_rules(case))
    return reached


@lru_cache(maxsize=2500)
def encode(case):
    if not isinstance(case,Case):return af15.encode(case)
    rules=logical_rules(case);apos,rpos=physical_maps(case.permutation)
    af=torch.zeros(base.ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(base.RULES,3,dtype=torch.float32)
    pu=torch.zeros(base.RULES,base.ATOMS);hu=pu.clone();pd=pu.clone();hd=pu.clone()
    facts=initial_facts(case)
    for node in range(base.ATOMS):
        af[apos[node]]=torch.tensor([float(node in facts),float(node==base.SUBGOALS[0]),
             float(node==base.SUBGOALS[1]),float(node==base.GOAL),1.])
    for k,(prem,head) in enumerate(rules):
        j=rpos[k];downstream=head==base.GOAL
        rf[j]=torch.tensor([len(prem)/2.,float(downstream),1.])
        pm,hm=(pd,hd) if downstream else (pu,hu)
        for node in prem:pm[j,apos[node]]=1.
        hm[j,apos[head]]=1.
    truth=complete_closure(case)
    return base.GraphBatch(af,rf,pu,hu,pd,hd,
         torch.tensor([apos[n] for n in (*base.SUBGOALS,base.GOAL)]),
         torch.tensor([int(n in truth) for n in (*base.SUBGOALS,base.GOAL)],dtype=torch.long))


def case_hash(case):
    x=encode(case)
    return base.digest([getattr(x,f.name).tolist() for f in fields(base.GraphBatch) if f.name!='targets'])

@lru_cache(maxsize=8)
def cases_for(split):
    if split not in SPLITS:raise ValueError('unregistered AF-16 data split')
    rows=[]
    if split in ('train','dev'):
        rows.extend(af15.cases_for(split))
    for variant,depth,gate,facts,layout in product(VARIANTS[split],DEPTHS[split],
            ('and','or'),product((0,1),repeat=4),LAYOUTS[split]):
        rows.append(Case(split,variant,depth,gate,tuple(facts),layout))
    return tuple(rows)


@lru_cache(maxsize=2500)
def physical_rules(case):
    rules=logical_rules(case);_,rpos=physical_maps(case.permutation)
    return {rpos[i]:rule for i,rule in enumerate(rules)}


def enabled(case,known):
    known=set(known)
    return tuple(sorted(i for i,(prem,head) in physical_rules(case).items()
                        if head not in known and all(x in known for x in prem)))


def sound_fire(case,known,action):
    state=frozenset(known)
    if action==STOP:return state,None
    if action not in enabled(case,state):raise ValueError('inadmissible unsupported Horn inference')
    p,h=physical_rules(case)[action]
    return state|{h},{'rule_slot':action,'premises':list(p),'head':h}

@lru_cache(maxsize=400000)
def _shortest(rules,known):
    if base.GOAL in known:return 0
    state=set(known)
    successors={h for p,h in rules if h not in state and all(t in state for t in p)}
    if not successors:return 10**6
    return 1+min(_shortest(rules,tuple(sorted(state|{h}))) for h in successors)


def optimal_remaining(case,known):
    return _shortest(tuple(logical_rules(case)),tuple(sorted(known)))


def optimal_actions(case,known):
    count=optimal_remaining(case,known)
    if count==0 or count>=10**6:return (STOP,)
    actions=[]
    for i in enabled(case,known):
        after,_=sound_fire(case,known,i)
        if optimal_remaining(case,after)==count-1:actions.append(i)
    if not actions:raise RuntimeError('missing shortest Horn action')
    return tuple(actions)


def reference_episode(case):
    state=frozenset(initial_facts(case));out=[]
    for _ in range(base.ATOMS+2):
        optimum=optimal_actions(case,state);out.append((state,optimum))
        if optimum[0]==STOP:return out
        state,_=sound_fire(case,state,optimum[0])
    raise RuntimeError('reference Horn episode failed to terminate')


def observations(cases,states,with_labels=False):
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
            for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    gate=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    pos=gate.clone()
    for i,(case,state) in enumerate(zip(cases,states)):
        apos,_=physical_maps(case.permutation)
        for j in state:bits[i,apos[j]]=1
        for j in enabled(case,state):gate[i,j]=True
        gate[i,STOP]=True
        if with_labels:
            for j in optimal_actions(case,state):pos[i,j]=True
    return (x,bits,gate,pos) if with_labels else (x,bits,gate)


def manifest():
    seen=set();out={};old=set()
    for split in ('train','dev'):
        for c in af15.cases_for(split):old.add(case_hash(c))
    for split in SPLITS:
        cases=cases_for(split);hashes=[case_hash(c) for c in cases]
        if len(set(hashes))!=len(hashes):raise ValueError('within-split AF16 feature duplicates')
        if seen.intersection(hashes):raise ValueError('AF16 train-heldout feature hash leakage')
        seen.update(hashes)
        out[split]={'count':len(cases),'case_ids_sha256':base.digest([c.id for c in cases]),
                    'feature_hashes_sha256':base.digest(hashes),
                    'layouts':list(LAYOUTS[split]),'novel_structures':list(VARIANTS[split])}
    result={'experiment':'AF-16','scope':'heldout logical rule compositions and novel physical layouts',
         'legacy_source':'AF-15 train/dev only','splits':out,'host_sound_gate':True,
         'heldout_absent_from_training':True}
    result['manifest_sha256']=base.digest(result)
    return result
