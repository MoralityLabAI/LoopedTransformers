"""AF-21: prospective composition-heldout Horn program plus AF-20 exact training roster.

Training and development cases and paired examples are inherited byte-for-byte
from AF-20. Newly composed programs use unique 72--76 physical permutations.
All source labels come from the host's exact finite shortest-proof oracle.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
from functools import lru_cache
from collections import defaultdict,deque
from itertools import product, combinations
import random
import torch
from experiments.activation_funnel_af05 import gym as base
from experiments.activation_funnel_af17 import gym as af17
from experiments.activation_funnel_af20 import gym as parent

STOP=parent.STOP
INFINITY=parent.INFINITY
CAP=16
SEEDS=(17,29,43,61,73,89,103,131)
SPLITS=('train','dev','held_layout','held_depth4','held_depth5')
LAYOUTS={'held_layout':(72,), 'held_depth4':(73,74), 'held_depth5':(75,76)}
HELD_VARIANTS=('bridged_shortcut','crossed_detour','double_dependency')
FACT_ATOMS=(0,1,7,8)

@dataclass(frozen=True)
class Case(parent.Case):
    @property
    def id(self):
        return (f'af21-{self.split}-{self.variant}-d{self.depth}-{self.gate}-'
                f'{"".join(map(str,self.facts))}-p{self.permutation}')


def physical_maps(index):
    if index not in range(72,77):
        return parent.physical_maps(index)
    rng=random.Random(212121+3191*index)
    atoms=list(range(base.ATOMS));rules=list(range(base.RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)

initial_facts=parent.initial_facts


def logical_rules(case):
    if not isinstance(case,Case):return parent.logical_rules(case)
    up=[];ends=[]
    for start in (0,7):
        previous=start
        for j in range(case.depth-1):
            nxt=start+2+j
            up.append(((previous,),nxt));previous=nxt
        ends.append(previous)
    a,b=ends;q0,q1=base.SUBGOALS;g=base.GOAL
    if case.variant=='bridged_shortcut':
        up.extend([((a,1),q0),((b,q0),q1),((b,8),q0)])
        shortcut=((b,1),g)
    elif case.variant=='crossed_detour':
        up.extend([((a,b),q0),((q0,8),q1),((b,1),q1)])
        shortcut=((a,1),g)
    elif case.variant=='double_dependency':
        up.extend([((a,1),q0),((b,q0),q1),((b,8),q1)])
        shortcut=((a,8),g)
    else:raise ValueError('unknown AF-21 heldout Horn composition')
    down=[((q0,q1),g)] if case.gate=='and' else [((q0,),g),((q1,),g)]
    down.append(shortcut)
    rules=tuple(up+down)
    if len(rules)>base.RULES or any(not p or len(p)>2 for p,_ in rules):
        raise ValueError('AF-21 graph exceeds registered rule slots or arity')
    return rules


def closure(case):
    state,_=base.closure(initial_facts(case),logical_rules(case))
    return state


def optimal_remaining(case,known):
    return af17._min_steps(tuple(logical_rules(case)),tuple(sorted(known)))


@lru_cache(maxsize=4096)
def encode(case):
    if not isinstance(case,Case):return parent.encode(case)
    apos,rpos=physical_maps(case.permutation)
    af=torch.zeros(base.ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(base.RULES,3,dtype=torch.float32)
    pu=torch.zeros(base.RULES,base.ATOMS);hu=pu.clone();pd=pu.clone();hd=pu.clone()
    facts=initial_facts(case)
    for atom in range(base.ATOMS):
        af[apos[atom]]=torch.tensor([float(atom in facts),float(atom==base.SUBGOALS[0]),
            float(atom==base.SUBGOALS[1]),float(atom==base.GOAL),1.])
    for i,(prem,head) in enumerate(logical_rules(case)):
        j=rpos[i];down=head==base.GOAL
        rf[j]=torch.tensor([len(prem)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for atom in prem:pp[j,apos[atom]]=1.
        hh[j,apos[head]]=1.
    truth=closure(case)
    return base.GraphBatch(af,rf,pu,hu,pd,hd,
        torch.tensor([apos[n] for n in (*base.SUBGOALS,base.GOAL)]),
        torch.tensor([int(n in truth) for n in (*base.SUBGOALS,base.GOAL)],dtype=torch.long))


def case_hash(c):
    x=encode(c)
    return base.digest([getattr(x,f.name).tolist() for f in fields(base.GraphBatch)
                        if f.name!='targets'])


@lru_cache(maxsize=12)
def cases_for(split):
    if split in ('train','dev'):return parent.cases_for(split)
    if split not in LAYOUTS:raise ValueError('unknown AF-21 split')
    depth=3 if split=='held_layout' else 4 if split=='held_depth4' else 5
    variants=('bridged_shortcut',) if split=='held_layout' else HELD_VARIANTS
    return tuple(Case(split,v,depth,gate,tuple(bits),layout)
                 for v,gate,bits,layout in product(variants,('and','or'),
                                                product((0,1),repeat=4),LAYOUTS[split]))


@lru_cache(maxsize=4096)
def physical_rules(c):
    if not isinstance(c,Case):return parent.physical_rules(c)
    _,rpos=physical_maps(c.permutation)
    return {rpos[i]:p for i,p in enumerate(logical_rules(c))}


def enabled(c,known):
    s=set(known)
    return tuple(sorted(i for i,(p,h) in physical_rules(c).items() if h not in s and set(p)<=s))


def sound_fire(c,known,action):
    s=frozenset(known)
    if action==STOP:return s,None
    if action not in enabled(c,s):raise ValueError('host rejected unsupported Horn rule')
    p,h=physical_rules(c)[action]
    return s|{h},{'rule_slot':action,'premises':list(p),'head':h}


def optimal_actions(c,known):
    d=optimal_remaining(c,known)
    if d==0 or d>=INFINITY:return (STOP,)
    out=[]
    for i in enabled(c,known):
        nxt,_=sound_fire(c,known,i)
        if optimal_remaining(c,nxt)==d-1:out.append(i)
    if not out:raise RuntimeError('reachable goal without optimal action')
    return tuple(out)


def action_costs(c,known,cap=CAP):
    if cap!=CAP:raise ValueError('AF21 cap fixed at 16')
    d=optimal_remaining(c,known)
    costs=[float('nan')]*(base.RULES+1)
    for i in enabled(c,known):
        nxt,_=sound_fire(c,known,i)
        rem=optimal_remaining(c,nxt)
        costs[i]=float(cap if rem>=INFINITY else min(cap,rem+1))
    costs[STOP]=0. if d==0 or d>=INFINITY else float(cap)
    return costs


def observations(cases,states,with_labels=False):
    if len(cases)!=len(states) or not cases:raise ValueError('paired cases/states required')
    x=base.GraphBatch(*(torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)))
    bits=torch.zeros(len(cases),base.ATOMS)
    legal=torch.zeros(len(cases),base.RULES+1,dtype=torch.bool)
    optimal=torch.zeros_like(legal)
    for i,(c,s) in enumerate(zip(cases,states)):
        amap,_=physical_maps(c.permutation)
        for node in s:bits[i,amap[node]]=1.
        for r in enabled(c,s):legal[i,r]=True
        legal[i,STOP]=True
        if with_labels:
            for r in optimal_actions(c,s):optimal[i,r]=True
    return (x,bits,legal,optimal) if with_labels else (x,bits,legal)


def incidence_distance(case,atom,goal=base.GOAL):
    adj={('a',k):set() for k in range(base.ATOMS)}
    for i,(prem,head) in enumerate(logical_rules(case)):
        r=('r',i);adj[r]=set()
        for node in (*prem,head):adj[('a',node)].add(r);adj[r].add(('a',node))
    q=deque([(('a',atom),0)]);seen={('a',atom)}
    while q:
        u,d=q.popleft()
        if u==('a',goal):return d
        for v in adj[u]:
            if v not in seen:seen.add(v);q.append((v,d+1))
    return None

Pair=parent.Pair


@lru_cache(maxsize=12)
def pairs_for(split):
    if split in ('train','dev'):return parent.pairs_for(split)
    cs=cases_for(split);groups=defaultdict(list)
    for i,c in enumerate(cs):groups[(logical_rules(c),c.permutation)].append(i)
    out=[]
    for indices in groups.values():
        for i,j in combinations(indices,2):
            c0,c1=cs[i],cs[j]
            changes=[k for k in range(4) if c0.facts[k]!=c1.facts[k]]
            if len(changes)!=1:continue
            atom=FACT_ATOMS[changes[0]]
            distance=incidence_distance(c0,atom)
            if distance is None:continue
            s0,s1=initial_facts(c0),initial_facts(c1)
            if not enabled(c0,s0) or not enabled(c1,s1):continue
            d0,d1=optimal_remaining(c0,s0),optimal_remaining(c1,s1)
            r0,r1=d0<INFINITY,d1<INFINITY
            if r0!=r1:kind='flip'
            elif r0 and r1 and d0==d1:kind='connected_invariant'
            else:continue
            out.append(Pair(i,j,kind,atom,distance,r0,r1,
                            d0 if r0 else None,d1 if r1 else None))
    out.sort(key=lambda x:(x.kind,x.left,x.right))
    if not any(p.kind=='flip' for p in out) or not any(p.kind=='connected_invariant' for p in out):
        raise ValueError('heldout lacks flip/invariant pair group')
    return tuple(out)


def pair_receipt(split):
    if split in ('train','dev'):return parent.pair_receipt(split)
    cs=cases_for(split)
    rows=[{'left':cs[p.left].id,'right':cs[p.right].id,'kind':p.kind,
           'changed_atom':p.changed_atom,'incidence_distance':p.graph_distance,
           'reach':[int(p.reachable_left),int(p.reachable_right)],
           'cost':[p.minimum_left,p.minimum_right]} for p in pairs_for(split)]
    return {'pairs':len(rows),'kinds':{k:sum(p.kind==k for p in pairs_for(split))
            for k in ('flip','connected_invariant')},'digest':base.digest(rows),'rows':rows}


def manifest():
    seen=set();older=set();splits={};pairs={}
    from experiments.activation_funnel_af16 import gym as af16
    from experiments.activation_funnel_af17 import gym as af17
    from experiments.activation_funnel_af18 import gym as af18
    from experiments.activation_funnel_af19 import gym as af19
    for mod,part in ((af16,('novel_layout','novel_depth4','novel_depth5')),
                     (af17,('held_layout','held_depth4','held_depth5')),
                     (af18,('held_layout','held_depth4','held_depth5')),
                     (af19,('held_layout','held_depth4','held_depth5')),
                     (parent,('held_layout','held_depth4','held_depth5'))):
        for sp in part:older.update(mod.case_hash(c) for c in mod.cases_for(sp))
    for sp in SPLITS:
        cs=cases_for(sp);hashes=[case_hash(c) for c in cs]
        if len(set(hashes))!=len(hashes) or seen.intersection(hashes):
            raise ValueError('duplicate graph features across AF21 splits')
        if sp.startswith('held') and older.intersection(hashes):
            raise ValueError('AF21 heldout overlaps prior scored graph')
        seen.update(hashes)
        splits[sp]={'cases':len(cs),'id_digest':base.digest([c.id for c in cs]),
                    'features_digest':base.digest(hashes),
                    'layouts':sorted(set(c.permutation for c in cs)),
                    'logical_variants':sorted(set(getattr(c,'variant','legacy') for c in cs))}
        rec=pair_receipt(sp)
        pairs[sp]={'flip':rec['kinds']['flip'],'connected_invariant':rec['kinds']['connected_invariant'],
                   'pair_digest':rec['digest']}
    out={'experiment':'AF-21','logic':'three new cross-dependency and shortcut combinations',
         'model_train_and_dev':'AF-20 exact 928/352 cases and 304/484 train paired records',
         'data_splits':splits,'paired_splits':pairs,
         'all_heldout_features_disjoint_from_AF16_to_AF20':True,
         'host_sound':True}
    out['manifest_digest']=base.digest(out)
    return out
