"""AF-18: controlled depth sweep on fresh layouts of AF-17 Horn compositions.

The network sees facts, incidence and the host's legal-action mask; exact
Bellman costs and reachability remain source-only targets. Training/dev cases
are precisely AF-17's original training/dev sets. Fresh test layouts 57--61
were not part of AF-16/17 tests.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import product
from collections import deque
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af16 import gym as af16
from experiments.activation_funnel_af17 import gym as parent

CAP=16
INFINITY=parent.INFINITY
STOP=base.RULES
SPLITS=('train','dev','held_layout','held_depth4','held_depth5')
LAYOUTS={'held_layout':(57,), 'held_depth4':(58,59), 'held_depth5':(60,61)}
HELD_VARIANTS=parent.HELD_VARIANTS

@dataclass(frozen=True)
class Case(parent.Case):
    @property
    def id(self):
        return f'af18-{self.split}-{self.variant}-d{self.depth}-{self.gate}-{''.join(map(str,self.facts))}-p{self.permutation}'


def physical_maps(index):
    if index not in range(57,62): return parent.physical_maps(index)
    rng=random.Random(181818+3191*index)
    atoms=list(range(base.ATOMS));rules=list(range(base.RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)

initial_facts=parent.initial_facts
logical_rules=parent.logical_rules
closure=parent.closure
optimal_remaining=parent.optimal_remaining


@lru_cache(maxsize=2000)
def encode(case):
    if not isinstance(case,Case):return parent.encode(case)
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
    if split in ('train','dev'):return parent.cases_for(split)
    if split not in LAYOUTS:raise ValueError('invalid AF18 split')
    depth=3 if split=='held_layout' else 4 if split=='held_depth4' else 5
    variants=('dual_fallback',) if split=='held_layout' else HELD_VARIANTS
    return tuple(Case(split,v,depth,g,tuple(f),p)
        for v,g,f,p in product(variants,('and','or'),product((0,1),repeat=4),LAYOUTS[split]))


@lru_cache(maxsize=2000)
def physical_rules(case):
    if not isinstance(case,Case):return parent.physical_rules(case)
    _,rpos=physical_maps(case.permutation)
    return {rpos[i]:r for i,r in enumerate(logical_rules(case))}


def enabled(case,known):
    state=set(known)
    return tuple(sorted(i for i,(p,h) in physical_rules(case).items() if h not in state and set(p)<=state))


def sound_fire(case,known,action):
    state=frozenset(known)
    if action==STOP:return state,None
    if action not in enabled(case,state):raise ValueError('inadmissible Horn action')
    p,h=physical_rules(case)[action]
    return state|{h},{'rule_slot':action,'premises':list(p),'head':h}


def optimal_actions(case,known):
    d=optimal_remaining(case,known)
    if d==0 or d>=INFINITY:return (STOP,)
    out=[]
    for i in enabled(case,known):
        nxt,_=sound_fire(case,known,i)
        if optimal_remaining(case,nxt)==d-1:out.append(i)
    if not out:raise RuntimeError('source shortest path unavailable')
    return tuple(out)


def action_costs(case,state,cap=CAP):
    if cap not in (10,16):raise ValueError('cap must be registered 10 or 16')
    d=optimal_remaining(case,state)
    costs=[float('nan')]*(base.RULES+1)
    for i in enabled(case,state):
        after,_=sound_fire(case,state,i)
        rem=optimal_remaining(case,after)
        costs[i]=float(min(cap,1+rem))
    costs[STOP]=0. if d==0 or d>=INFINITY else float(cap)
    return costs


def reference_episode(case):
    state=frozenset(initial_facts(case));result=[]
    for _ in range(base.ATOMS+2):
        choices=optimal_actions(case,state)
        result.append((state,choices))
        if choices[0]==STOP:return tuple(result)
        state,_=sound_fire(case,state,choices[0])
    raise RuntimeError('reference trace failed to terminate')


def observations(cases,states,with_labels=False):
    if len(cases)!=len(states) or not cases:raise ValueError('paired true-length nonempty cases required')
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    gate=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    opt=torch.zeros_like(gate)
    for i,(case,state) in enumerate(zip(cases,states)):
        apos,_=physical_maps(case.permutation)
        for j in state:bits[i,apos[j]]=1.
        for j in enabled(case,state):gate[i,j]=True
        gate[i,STOP]=True
        if with_labels:
            for j in optimal_actions(case,state):opt[i,j]=True
    if with_labels:return x,bits,gate,opt
    return x,bits,gate


def graph_distance(case,from_atom,to_atom=base.GOAL):
    """Undirected incidence graph distance, independent of network parameters."""
    adjacency={('a',i):set() for i in range(base.ATOMS)}
    for j,(p,h) in enumerate(logical_rules(case)):
        adjacency[('r',j)]=set()
        for atom in (*p,h):
            u=('a',atom);v=('r',j)
            adjacency[u].add(v);adjacency[v].add(u)
    start=('a',from_atom);goal=('a',to_atom);queue=deque([(start,0)]);seen={start}
    while queue:
        n,d=queue.popleft()
        if n==goal:return d
        for v in adjacency[n]:
            if v not in seen:seen.add(v);queue.append((v,d+1))
    return None


def probe_pairs():
    return tuple((Case('held_depth5','dual_fallback',depth,'and',(0,1,1,1),60),
                   Case('held_depth5','dual_fallback',depth,'and',(1,1,1,1),60))
                 for depth in (3,4,5))


def manifest():
    out={};seen=set();old=set()
    for sp in ('novel_layout','novel_depth4','novel_depth5'):
        old.update(af16.case_hash(c) for c in af16.cases_for(sp))
    for sp in ('held_layout','held_depth4','held_depth5'):
        old.update(parent.case_hash(c) for c in parent.cases_for(sp))
    for sp in SPLITS:
        cases=cases_for(sp);digests=[case_hash(c) for c in cases]
        if len(digests)!=len(set(digests)) or seen.intersection(digests):
            raise ValueError('AF18 duplicated public input')
        if sp.startswith('held') and old.intersection(digests):
            raise ValueError('AF18 heldout overlap with AF16/17')
        seen.update(digests)
        out[sp]={'cases':len(cases),'case_ids_sha256':base.digest([c.id for c in cases]),
            'feature_hashes_sha256':base.digest(digests),
            'layouts':sorted(set(c.permutation for c in cases))}
    result={'experiment':'AF-18','parent_train_dev':'AF17 byte-identical inputs',
        'heldout_new_layouts':list(range(57,62)), 'heldout_templates':'AF17 logical compositions reused',
        'source_only_optimal_costs':True,'host_rule_gate':True,'splits':out}
    result['manifest_sha256']=base.digest(result)
    return result
