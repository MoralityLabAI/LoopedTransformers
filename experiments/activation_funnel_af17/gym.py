"""AF-17: fixed Horn rule-cost corpus and genuinely fresh physical/logical heldouts.

Original AF-16 train and dev cases are unchanged. New cases compose branches with
contingent shortcuts, cross-branch joins, and alternating fallback rules.
All model features come from graph incidence, initial facts, current proved set.
The shortest-proof oracle is exclusively a training/evaluation label source.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af16 import gym as af16

STOP=base.RULES
INFINITY=10**6
CAP=10                         # explicit capped cost target, never actual infinity
SEEDS=(17,29,43,61)
SPLITS=('train','dev','held_layout','held_depth4','held_depth5')
HELD_VARIANTS=('serial_shortcut','dual_fallback','contingent_goal')
LAYOUTS={'held_layout':(52,), 'held_depth4':(53,54), 'held_depth5':(55,56)}

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
        return f'af17-{self.split}-{self.variant}-d{self.depth}-{self.gate}-{"".join(map(str,self.facts))}-p{self.permutation}'


def physical_maps(index):
    if index in range(52,57):
        rng=random.Random(171717 + 3191*index)
        atom=list(range(base.ATOMS));rule=list(range(base.RULES))
        rng.shuffle(atom);rng.shuffle(rule)
        return tuple(atom),tuple(rule)
    return af16.physical_maps(index)


def initial_facts(case):
    if not isinstance(case,Case):return af16.initial_facts(case)
    return frozenset(i for i,b in zip((0,1,7,8),case.facts) if b)


def logical_rules(case):
    if not isinstance(case,Case):return af16.logical_rules(case)
    up=[];ends=[]
    for branch in (0,1):
        start=7*branch
        prev=start
        for step in range(case.depth-1):
            nxt=start+2+step
            up.append(((prev,),nxt));prev=nxt
        ends.append(prev)
    a,b=ends;q0,q1=base.SUBGOALS;g=base.GOAL
    if case.variant=='serial_shortcut':
        up.extend([((a,1),q0),((b,q0),q1)])
        down=[((q0,q1),g)] if case.gate=='and' else [((q0,),g),((q1,),g)]
        down.append(((b,8),g))
    elif case.variant=='dual_fallback':
        up.extend([((a,b),q0),((b,8),q1),((q0,1),q1)])
        down=[((q0,q1),g)] if case.gate=='and' else [((q0,),g),((q1,),g)]
    elif case.variant=='contingent_goal':
        up.extend([((a,8),q0),((b,1),q1)])
        down=[((q0,q1),g)] if case.gate=='and' else [((q0,),g),((q1,),g)]
        down.extend([((a,1),g),((b,8),g)])
    else:raise ValueError('unknown heldout variant')
    rules=tuple(up+down)
    if len(rules)>base.RULES:raise ValueError('too many Horn rule slots')
    if any(len(p)==0 or len(p)>2 for p,_ in rules):raise ValueError('invalid arity')
    return rules


def closure(case):
    k,_=base.closure(initial_facts(case),logical_rules(case))
    return k


@lru_cache(maxsize=2000)
def encode(case):
    if not isinstance(case,Case):return af16.encode(case)
    rules=logical_rules(case);apos,rpos=physical_maps(case.permutation)
    af=torch.zeros(base.ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(base.RULES,3,dtype=torch.float32)
    pu=torch.zeros(base.RULES,base.ATOMS);hu=pu.clone();pd=pu.clone();hd=pu.clone()
    facts=initial_facts(case)
    for atom in range(base.ATOMS):
        af[apos[atom]]=torch.tensor([float(atom in facts),float(atom==base.SUBGOALS[0]),
            float(atom==base.SUBGOALS[1]),float(atom==base.GOAL),1.])
    for i,(prem,head) in enumerate(rules):
        j=rpos[i];down=head==base.GOAL
        rf[j]=torch.tensor([len(prem)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for node in prem:pp[j,apos[node]]=1.
        hh[j,apos[head]]=1.
    truth=closure(case)
    return base.GraphBatch(af,rf,pu,hu,pd,hd,
        torch.tensor([apos[n] for n in (*base.SUBGOALS,base.GOAL)]),
        torch.tensor([int(n in truth) for n in (*base.SUBGOALS,base.GOAL)],dtype=torch.long))


def case_hash(case):
    x=encode(case)
    return base.digest([getattr(x,f.name).tolist() for f in fields(base.GraphBatch) if f.name!='targets'])


@lru_cache(maxsize=10)
def cases_for(split):
    if split in ('train','dev'):return af16.cases_for(split)
    if split not in LAYOUTS:raise ValueError('unknown AF17 heldout split')
    depth=3 if split=='held_layout' else 4 if split=='held_depth4' else 5
    variants=('dual_fallback',) if split=='held_layout' else HELD_VARIANTS
    return tuple(Case(split,v,depth,g,tuple(f),p) for v,g,f,p in product(
        variants,('and','or'),product((0,1),repeat=4),LAYOUTS[split]))


@lru_cache(maxsize=2200)
def physical_rules(case):
    _,rpos=physical_maps(case.permutation)
    return {rpos[i]:rule for i,rule in enumerate(logical_rules(case))}


def enabled(case,known):
    s=set(known)
    return tuple(sorted(i for i,(prem,h) in physical_rules(case).items()
                        if h not in s and set(prem)<=s))


def sound_fire(case,known,action):
    state=frozenset(known)
    if action==STOP:return state,None
    if action not in enabled(case,state):raise ValueError('inadmissible Horn action')
    premises,head=physical_rules(case)[action]
    return state|{head},{'rule_slot':action,'premises':list(premises),'head':head}


@lru_cache(maxsize=700000)
def _min_steps(rules,known):
    if base.GOAL in known:return 0
    possible={h for prem,h in rules if h not in known and set(prem)<=set(known)}
    if not possible:return INFINITY
    return 1+min(_min_steps(rules,tuple(sorted(set(known)|{h}))) for h in possible)


def optimal_remaining(case,known):
    return _min_steps(tuple(logical_rules(case)),tuple(sorted(known)))


def optimal_actions(case,known):
    d=optimal_remaining(case,known)
    if d==0 or d>=INFINITY:return (STOP,)
    out=[]
    for i in enabled(case,known):
        nxt,_=sound_fire(case,known,i)
        if optimal_remaining(case,nxt)==d-1:out.append(i)
    if not out:raise ValueError('reachable state lacks an optimal rule')
    return tuple(out)


def action_costs(case,state):
    """Host-owned finite Bellman labels, capped only for numerical training.

    Costs are exact successor rule counts for reachable successors, capped to
    CAP for dead ends and early STOP. They are *not* model input features.
    """
    d=optimal_remaining(case,state)
    legal=enabled(case,state)
    costs=[float('nan')]*(base.RULES+1)
    for i in legal:
        after,_=sound_fire(case,state,i)
        rem=optimal_remaining(case,after)
        costs[i]=min(float(1+rem),float(CAP))
    costs[STOP]=0. if d==0 or d>=INFINITY else float(CAP)
    return costs


def reference_episode(case):
    state=frozenset(initial_facts(case));out=[]
    for _ in range(base.ATOMS+2):
        optimal=optimal_actions(case,state)
        out.append((state,optimal))
        if optimal[0]==STOP:return tuple(out)
        state,_=sound_fire(case,state,optimal[0])
    raise RuntimeError('nonterminating shortest-goal source trace')


def observations(cases,states,with_labels=False):
    if len(cases)!=len(states):raise ValueError('case and state count mismatch')
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    gate=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    opt=torch.zeros_like(gate)
    for i,(c,state) in enumerate(zip(cases,states)):
        apos,_=physical_maps(c.permutation)
        for node in state:bits[i,apos[node]]=1.
        for a in enabled(c,state):gate[i,a]=True
        gate[i,STOP]=True
        if with_labels:
            for a in optimal_actions(c,state):opt[i,a]=True
    if with_labels:return x,bits,gate,opt
    return x,bits,gate


def manifest():
    seen=set();out={}
    old_hold=set()
    for sp in ('novel_layout','novel_depth4','novel_depth5'):
        old_hold.update(af16.case_hash(c) for c in af16.cases_for(sp))
    for sp in SPLITS:
        rows=cases_for(sp);keys=[case_hash(c) for c in rows]
        if len(keys)!=len(set(keys)) or seen.intersection(keys):
            raise ValueError('AF17 split overlap')
        if sp.startswith('held') and old_hold.intersection(keys):
            raise ValueError('AF17 fresh holdout overlaps AF16 evaluated cases')
        seen.update(keys)
        out[sp]={'count':len(rows),'case_ids_sha256':base.digest([c.id for c in rows]),
            'feature_hashes_sha256':base.digest(keys),
            'layout_ids':sorted(set(c.permutation for c in rows))}
    rec={'experiment':'AF-17','parent_source':'AF-16 train/dev unchanged',
         'new_heldout_templates':list(HELD_VARIANTS),'splits':out,
         'host_owns_rule_legality':True,'all_heldout_feature_hashes_disjoint_from_AF16':True}
    rec['manifest_sha256']=base.digest(rec)
    return rec
