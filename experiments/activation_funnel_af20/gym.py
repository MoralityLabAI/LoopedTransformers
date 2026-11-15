"""AF-20 goal-outcome counterfactual fact pairs and connected distractors.

Data labels use exact Horn reachability / minimum proof costs. Input pairs
share a *physical* rule graph and differ by one initial fact. A connected
irrelevant toggle has the same reachable goal and minimum proof cost, yet the
toggled atom lies on an undirected incidence path to the goal. This is a
conditional outcome-invariance contract, not a structural causal separation.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from itertools import combinations,product
from collections import defaultdict,deque
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af19 import gym as parent

STOP=base.RULES
INFINITY=parent.INFINITY
CAP=16
SEEDS=(17,29,43,61)
SPLITS=('train','dev','held_layout','held_depth4','held_depth5')
LAYOUTS={'held_layout':(67,), 'held_depth4':(68,69),'held_depth5':(70,71)}
HELD_VARIANTS=parent.HELD_VARIANTS
FACT_ATOMS=(0,1,7,8)

@dataclass(frozen=True)
class Case(parent.Case):
    @property
    def id(self):
        return f'af20-{self.split}-{self.variant}-d{self.depth}-{self.gate}-{"".join(map(str,self.facts))}-p{self.permutation}'


def physical_maps(index):
    if index not in range(67,72):return parent.physical_maps(index)
    rng=random.Random(202020+3191*index)
    atoms=list(range(base.ATOMS));rules=list(range(base.RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)

initial_facts=parent.initial_facts
logical_rules=parent.logical_rules
closure=parent.closure
optimal_remaining=parent.optimal_remaining

@lru_cache(maxsize=4096)
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
        j=rpos[i];down=(head==base.GOAL)
        rf[j]=torch.tensor([len(prem)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for atom in prem:pp[j,apos[atom]]=1.
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
    if split not in LAYOUTS:raise ValueError('AF-20 split unknown')
    depth=3 if split=='held_layout' else 4 if split=='held_depth4' else 5
    variants=('dual_fallback',) if split=='held_layout' else HELD_VARIANTS
    return tuple(Case(split,v,depth,g,tuple(f),layout)
        for v,g,f,layout in product(variants,('and','or'),product((0,1),repeat=4),LAYOUTS[split]))

@lru_cache(maxsize=4096)
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
    distance=optimal_remaining(case,known)
    if distance==0 or distance>=INFINITY:return (STOP,)
    actions=[]
    for i in enabled(case,known):
        nxt,_=sound_fire(case,known,i)
        if optimal_remaining(case,nxt)==distance-1:actions.append(i)
    if not actions:raise ValueError('reachable state lacks shortest rule')
    return tuple(actions)


def action_costs(case,known,cap=CAP):
    if cap!=CAP:raise ValueError('AF20 registered cap is 16')
    d=optimal_remaining(case,known)
    result=[float('nan')]*(base.RULES+1)
    for i in enabled(case,known):
        nxt,_=sound_fire(case,known,i)
        rem=optimal_remaining(case,nxt)
        result[i]=float(cap if rem>=INFINITY else min(cap,1+rem))
    result[STOP]=0. if d==0 or d>=INFINITY else float(cap)
    return result


def observations(cases,states,with_labels=False):
    if not cases or len(cases)!=len(states):raise ValueError('nonempty paired cases/states needed')
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    gate=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    optimal=torch.zeros_like(gate)
    for i,(c,s) in enumerate(zip(cases,states)):
        amap,_=physical_maps(c.permutation)
        for j in s:bits[i,amap[j]]=1.
        for j in enabled(c,s):gate[i,j]=True
        gate[i,STOP]=True
        if with_labels:
            for a in optimal_actions(c,s):optimal[i,a]=True
    return (x,bits,gate,optimal) if with_labels else (x,bits,gate)


def incidence_distance(case,atom,goal=base.GOAL):
    adjacency={('a',i):set() for i in range(base.ATOMS)}
    for i,(prem,head) in enumerate(logical_rules(case)):
        r=('r',i);adjacency[r]=set()
        for j in (*prem,head):adjacency[('a',j)].add(r);adjacency[r].add(('a',j))
    q=deque([(('a',atom),0)]);seen={('a',atom)}
    while q:
        node,dist=q.popleft()
        if node==('a',goal):return dist
        for nxt in adjacency[node]:
            if nxt not in seen:seen.add(nxt);q.append((nxt,dist+1))
    return None

@dataclass(frozen=True)
class Pair:
    left:int
    right:int
    kind:str
    changed_atom:int
    graph_distance:int
    reachable_left:bool
    reachable_right:bool
    minimum_left:int|None
    minimum_right:int|None

@lru_cache(maxsize=10)
def pairs_for(split):
    cases=cases_for(split);groups=defaultdict(list)
    for i,c in enumerate(cases):groups[(logical_rules(c),c.permutation)].append(i)
    result=[]
    for indices in groups.values():
        for i,j in combinations(indices,2):
            a,b=cases[i],cases[j]
            changed=[q for q in range(4) if a.facts[q]!=b.facts[q]]
            if len(changed)!=1:continue
            atom=FACT_ATOMS[changed[0]]
            distance=incidence_distance(a,atom)
            if distance is None:continue
            sa,sb=initial_facts(a),initial_facts(b)
            # Require a nontrivial STOP-vs-rule decision on both members.
            if not enabled(a,sa) or not enabled(b,sb):continue
            da=optimal_remaining(a,sa);db=optimal_remaining(b,sb)
            ra,rb=da<INFINITY,db<INFINITY
            if ra!=rb:kind='flip'
            elif ra and rb and da==db:kind='connected_invariant'
            else:continue
            result.append(Pair(i,j,kind,atom,distance,ra,rb,
                               da if ra else None,db if rb else None))
    result.sort(key=lambda p:(p.kind,p.left,p.right))
    if not any(p.kind=='flip' for p in result) or not any(p.kind=='connected_invariant' for p in result):
        raise ValueError('missing counterfactual pair class')
    return tuple(result)


def pair_receipt(split):
    cases=cases_for(split);pairs=pairs_for(split)
    rows=[{'left':cases[p.left].id,'right':cases[p.right].id,'kind':p.kind,
        'changed_atom':p.changed_atom,'incidence_distance':p.graph_distance,
        'reach':[int(p.reachable_left),int(p.reachable_right)],
        'cost':[p.minimum_left,p.minimum_right]} for p in pairs]
    return {'pairs':len(pairs),'kinds':{k:sum(p.kind==k for p in pairs)
       for k in ('flip','connected_invariant')},'digest':base.digest(rows),'rows':rows}


def manifest():
    seen=set();old=set();splits={};pairs={}
    for m in (parent,):
        for sp in ('held_layout','held_depth4','held_depth5'):
            old.update(m.case_hash(c) for c in m.cases_for(sp))
    from experiments.activation_funnel_af18 import gym as af18
    from experiments.activation_funnel_af17 import gym as af17
    from experiments.activation_funnel_af16 import gym as af16
    for m,parts in ((af18,('held_layout','held_depth4','held_depth5')),
                    (af17,('held_layout','held_depth4','held_depth5')),
                    (af16,('novel_layout','novel_depth4','novel_depth5'))):
        for sp in parts:old.update(m.case_hash(c) for c in m.cases_for(sp))
    for split in SPLITS:
        cs=cases_for(split);keys=[case_hash(c) for c in cs]
        if len(set(keys))!=len(keys) or seen.intersection(keys):raise ValueError('AF20 duplicated public inputs')
        if split.startswith('held') and old.intersection(keys):raise ValueError('AF20 heldout overlaps older experiment')
        seen.update(keys)
        splits[split]={'cases':len(cs),'case_ids_sha256':base.digest([c.id for c in cs]),
                      'feature_hash_sha256':base.digest(keys),
                      'layouts':sorted(set(c.permutation for c in cs))}
        pr=pair_receipt(split)
        pairs[split]={k:v for k,v in pr.items() if k!='rows'}
    ret={'experiment':'AF-20','source_train_dev':'AF19 unchanged',
         'heldout_new_layouts':list(range(67,72)),
         'pairs':pairs,'splits':splits,'host_rule_gate':True,
         'paired_targets':'independently recomputed finite Horn reachability and shortest proof cost'}
    ret['manifest_sha256']=base.digest(ret)
    return ret
