"""AF-29 source-only locally matched positive/negative Horn proof pairs.

Within each pair all root known facts, atom/rule physical permutations,
marked roles, directed local radius-two incidence neighborhoods around
roots/roles, and entire atom/rule in/out degree profiles are IDENTICAL.
Exactly two middle Horn rule premise edges are swapped, changing global
goal reachability while preserving all local signatures and the root rule
mask. No model weights or test outcomes enter pair construction.
"""
from __future__ import annotations

from collections import deque, defaultdict
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
import random

ATOMS, RULES, GOAL, STOP = 15, 14, 14, 14
SUBGOALS = (6, 13)
SEEDS = (17, 29, 43, 61, 73, 89, 103, 131)
SPLITS = ("train", "dev", "held_depth5", "held_depth6")
FAMILIES = {
    "train": ("train_linear", "train_branched"),
    "dev": ("dev_crossed", "dev_linear"),
    "held_depth5": ("held_gate", "held_shared"),
    "held_depth6": ("held_gate", "held_shared"),
}
LAYOUT_START = {"train":710, "dev":720, "held_depth5":730, "held_depth6":740}
DEPTHS = {
    ("train", "train_linear"):4,
    ("train", "train_branched"):5,
    ("dev", "dev_crossed"):5,
    ("dev", "dev_linear"):4,
    ("held_depth5", "held_gate"):5,
    ("held_depth5", "held_shared"):5,
    ("held_depth6", "held_gate"):6,
    ("held_depth6", "held_shared"):6,
}
POSITIVE_PAIRS_PER_FAMILY = 48
SOURCE_ROOTS = frozenset((0,1,2,3))


def digest(x):
    return hashlib.sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split: str
    family: str
    pair_index: int
    reachable: bool
    layout: int

    @property
    def id(self):
        return (f"af29-{self.split}-{self.family}"
                f"-p{self.pair_index:02d}-y{int(self.reachable)}-l{self.layout}")
    @property
    def pair_id(self):
        return f"af29-{self.split}-{self.family}-p{self.pair_index:02d}-l{self.layout}"
    @property
    def designed_positive(self):
        return self.reachable
    @property
    def depth(self):
        return DEPTHS[(self.split,self.family)]


@lru_cache(maxsize=4)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("unregistered matched-pair split")
    base=LAYOUT_START[split]
    return tuple(Case(split,family,pair_index,bool(label),
                      base+2*family_index+(pair_index%2))
                 for family_index,family in enumerate(FAMILIES[split])
                 for pair_index in range(POSITIVE_PAIRS_PER_FAMILY)
                 for label in (1,0))


def pair_mates(c):
    if c.split not in SPLITS:
        raise ValueError("unregistered pair source")
    return (Case(c.split,c.family,c.pair_index,True,c.layout),
            Case(c.split,c.family,c.pair_index,False,c.layout))


@lru_cache(maxsize=1024)
def initial_facts(c):
    return SOURCE_ROOTS


@lru_cache(maxsize=1024)
def logical_rules(c):
    # Every live proof follows a unique 4/5/6-firing path from the
    # shared four roots through productive intermediate fact 5 to goal.
    # The second GOAL path requires the unreachable atom 10 in both labels.
    # The negative pair swaps (5->productive,10->decoy) to
    # (10->productive,5->decoy), changing only two premise incidences.
    if (c.split,c.family) not in DEPTHS:
        raise ValueError("unknown split/family")
    d=c.depth
    if c.family=="train_linear":
        productive=[
            ((0,1),4),((4,2),5),((5,),6),((6,3),GOAL),
        ]
    elif c.family=="train_branched":
        productive=[
            ((0,1),4),((4,2),5),((5,),9),
            ((9,3),6),((6,),GOAL),
        ]
    elif c.family=="dev_crossed":
        productive=[
            ((0,3),4),((4,2),5),((5,),9),
            ((9,1),6),((6,),GOAL),
        ]
    elif c.family=="dev_linear":
        productive=[
            ((1,3),4),((4,2),5),((5,),6),((6,0),GOAL),
        ]
    elif c.family=="held_gate" and d==5:
        productive=[
            ((0,3),4),((4,2),5),((5,),9),
            ((9,),6),((6,1),GOAL),
        ]
    elif c.family=="held_shared" and d==5:
        productive=[
            ((1,3),4),((4,2),5),((5,),9),
            ((9,0),6),((6,),GOAL),
        ]
    elif c.family=="held_gate" and d==6:
        productive=[
            ((0,3),4),((4,2),5),((5,),9),
            ((9,),11),((11,),6),((6,1),GOAL),
        ]
    elif c.family=="held_shared" and d==6:
        productive=[
            ((1,3),4),((4,2),5),((5,),9),
            ((9,0),11),((11,),6),((6,),GOAL),
        ]
    else:
        raise ValueError("undeclared Horn topology")
    if len(productive)!=d:
        raise AssertionError("source shortest path count changed")
    productive=list(productive)
    # This productive middle clause always has a one-premise 5 input.
    position=2
    if productive[position][0]!=(5,):
        raise AssertionError("frozen productive source swap position changed")
    middle_decoy_input=(10,) if c.reachable else (5,)
    productive[position]=((5,) if c.reachable else (10,),
                          productive[position][1])
    # Create unrelated decoy-producing rule with same arity; swap its input.
    decoys=[
        (middle_decoy_input,7),
        ((7,),8),
        ((0,3),12),
        ((1,2),8),
        ((10,),13),
        ((13,),GOAL),
    ]
    rules=tuple((tuple(p),h) for p,h in (productive+decoys))
    if len(rules)>RULES or sum(h==GOAL for p,h in rules)!=2:
        raise AssertionError("out of source 14-rule/2-goal budget")
    if any(not p or len(p)>2 or h in p or h>=ATOMS or
           any(a>=ATOMS or a<0 for a in p) for p,h in rules):
        raise AssertionError("source Horn grammar malformed")
    return rules


@lru_cache(maxsize=1024)
def physical_maps(c):
    # Pair label is explicitly EXCLUDED from the source physical RNG seed.
    rng=random.Random(int(digest([
        "AF29_MIRRORED_PERM",c.split,c.family,c.pair_index,c.layout
    ])[:16],16))
    atoms=list(range(ATOMS))
    rules=list(range(RULES))
    rng.shuffle(atoms);rng.shuffle(rules)
    return tuple(atoms),tuple(rules)


@lru_cache(maxsize=1024)
def physical_rules(c):
    _,perm=physical_maps(c)
    return {perm[i]:(p,h) for i,(p,h) in enumerate(logical_rules(c))}


def enabled(c,known):
    known=frozenset(known)
    return tuple(sorted(
        slot for slot,(p,h) in physical_rules(c).items()
        if h not in known and all(a in known for a in p)
    ))


def sound_fire(c,known,action):
    state=frozenset(known)
    if action==STOP:
        return state,None
    if action not in enabled(c,state):
        raise ValueError("host rejected illegal Horn rule")
    prem,head=physical_rules(c)[action]
    return state|{head},{"rule_slot":action,
                         "premises":list(prem),"head":head}


def closure(c,known=None):
    current=set(initial_facts(c) if known is None else known)
    while True:
        future=current|{h for p,h in physical_rules(c).values()
                         if all(a in current for a in p)}
        if future==current:
            return frozenset(current)
        current=future


@lru_cache(maxsize=100000)
def shortest_steps(c,known=None):
    root=frozenset(initial_facts(c) if known is None else known)
    if GOAL in root:
        return 0
    seen={root};todo=deque([(root,0)])
    while todo:
        state,d=todo.popleft()
        for a in enabled(c,state):
            nxt,_=sound_fire(c,state,a)
            if GOAL in nxt:
                return d+1
            if nxt not in seen:
                seen.add(nxt);todo.append((nxt,d+1))
    return None


def independent_bfs(c,known=None):
    known=initial_facts(c) if known is None else known
    root=sum(1<<a for a in known)
    if root&(1<<GOAL):
        return 0
    masks=tuple((sum(1<<a for a in prem),1<<head)
                for prem,head in physical_rules(c).values())
    seen={root};todo=deque([(root,0)])
    while todo:
        bits,d=todo.popleft()
        for prem,head in masks:
            if prem&bits!=prem or head&bits:
                continue
            nxt=bits|head
            if nxt&(1<<GOAL):
                return d+1
            if nxt not in seen:
                seen.add(nxt);todo.append((nxt,d+1))
    return None


def optimal_actions(c,known):
    distance=shortest_steps(c,frozenset(known))
    if distance is None or distance==0:
        return (STOP,)
    return tuple(a for a in enabled(c,known)
                 if shortest_steps(c,sound_fire(c,known,a)[0])==distance-1)


def public_features(c):
    atoms,_=physical_maps(c)
    return {
        "atoms":ATOMS,"rules":RULES,
        "initial_physical":sorted(atoms[a] for a in initial_facts(c)),
        "goal_slot":atoms[GOAL],
        "subgoal_slots":[atoms[a] for a in SUBGOALS],
        "rules_physical":[
            [slot,sorted(atoms[a] for a in prem),atoms[head]]
            for slot,(prem,head) in sorted(physical_rules(c).items())
        ],
    }


def bipartite_incidence(c):
    """Source-full directed bipartite graph, IDs fixed within positive/negative pair."""
    atoms,_=physical_maps(c)
    directed=[]
    adjacency=defaultdict(set)
    for slot,(prem,head) in physical_rules(c).items():
        rule=("r",slot)
        for atom in prem:
            a=("a",atoms[atom])
            directed.append((a,rule,"premise"))
            adjacency[a].add(rule);adjacency[rule].add(a)
        h=("a",atoms[head])
        directed.append((rule,h,"head"))
        adjacency[h].add(rule);adjacency[rule].add(h)
    return tuple(directed),adjacency


def node_degrees(c):
    edges,_=bipartite_incidence(c)
    profile={}
    for source,dest,label in edges:
        profile.setdefault(source,[0,0])[1]+=1
        profile.setdefault(dest,[0,0])[0]+=1
    return tuple(sorted((kind,i,a,b)
                        for (kind,i),(a,b) in profile.items()))


def radius2_anchor_signature(c):
    """Identity-preserving directed radius-2 view of every root/marked atom."""
    edges,adj=bipartite_incidence(c)
    perm,_=physical_maps(c)
    anchors=tuple(("a",perm[a]) for a in (*sorted(SOURCE_ROOTS),*SUBGOALS,GOAL))
    views=[]
    for anchor in anchors:
        distance={anchor:0}
        todo=deque([anchor])
        while todo:
            cur=todo.popleft()
            if distance[cur]==2:
                continue
            for nbr in adj[cur]:
                if nbr not in distance:
                    distance[nbr]=distance[cur]+1
                    todo.append(nbr)
        local=frozenset(distance)
        # Rule direction, physical node IDs, and identity of anchor matter.
        lines=tuple(sorted((x,y,kind) for x,y,kind in edges
                           if x in local and y in local))
        views.append((anchor,tuple(sorted(local)),lines))
    return tuple(views)


def pair_structural_record(positive,negative):
    if not positive.reachable or negative.reachable or positive.pair_id!=negative.pair_id:
        raise ValueError("source pair did not preserve identity and labels")
    a=public_features(positive);b=public_features(negative)
    if (physical_maps(positive)!=physical_maps(negative) or
        a["initial_physical"]!=b["initial_physical"] or
        a["goal_slot"]!=b["goal_slot"] or
        a["subgoal_slots"]!=b["subgoal_slots"] or
        enabled(positive,SOURCE_ROOTS)!=enabled(negative,SOURCE_ROOTS)):
        raise AssertionError("source pair leaks through initial/marked roles or root mask")
    if node_degrees(positive)!=node_degrees(negative):
        raise AssertionError("source pair changes full directed degree sequence")
    sa=radius2_anchor_signature(positive)
    sb=radius2_anchor_signature(negative)
    if sa!=sb:
        raise AssertionError("source pair leaks through radius-two rooted view")
    ea={tuple((slot,tuple(pre),head))
        for slot,pre,head in a["rules_physical"]}
    eb={tuple((slot,tuple(pre),head))
        for slot,pre,head in b["rules_physical"]}
    common=ea&eb
    if len(ea-common)!=2 or len(eb-common)!=2:
        raise AssertionError("source positive/negative changes more than 2 rules")
    for pair in (positive,negative):
        if len(enabled(pair,SOURCE_ROOTS))<2:
            raise AssertionError("insufficient actionable root branching")
    pa,pb=shortest_steps(positive),shortest_steps(negative)
    if (pa!=positive.depth or pb is not None or
        independent_bfs(positive)!=pa or independent_bfs(negative)!=pb or
        (GOAL in closure(positive)) is not True or
        (GOAL in closure(negative)) is not False):
        raise AssertionError("source positive/negative goal reachability invalid")
    return {
        "pair_id":positive.pair_id,
        "positive_public_hash":digest(a),
        "negative_public_hash":digest(b),
        "local_radius2_signature_sha256":digest(sa),
        "directed_degree_sha256":digest(node_degrees(positive)),
        "positive_shortest":pa,
        "negative_shortest":pb,
        "root_enabled_physical":list(enabled(positive,SOURCE_ROOTS)),
        "positive_rule_differences":sorted(ea-common),
        "negative_rule_differences":sorted(eb-common),
    }


@lru_cache(maxsize=1024)
def encode(c):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if (base.ATOMS,base.RULES)!=(ATOMS,RULES):
        raise AssertionError("original AF23 frozen model input dimensions changed")
    atoms,_=physical_maps(c)
    af=torch.zeros(ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(RULES,3,dtype=torch.float32)
    pu=torch.zeros(RULES,ATOMS)
    hu=pu.clone();pd=pu.clone();hd=pu.clone()
    for logical in range(ATOMS):
        af[atoms[logical]]=torch.tensor([
            float(logical in SOURCE_ROOTS),
            float(logical==SUBGOALS[0]),
            float(logical==SUBGOALS[1]),
            float(logical==GOAL),1.,
        ])
    for slot,(prem,head) in physical_rules(c).items():
        is_goal=head==GOAL
        rf[slot]=torch.tensor([len(prem)/2.,float(is_goal),1.])
        pp,hh=(pd,hd) if is_goal else (pu,hu)
        for atom in prem:
            pp[slot,atoms[atom]]=1.
        hh[slot,atoms[head]]=1.
    # Critical: zero ALL legacy batch targets for every AF29 case.
    # Native original frozen ReadoutPolicy.full never uses .targets.
    # This prevents any inadvertent truth leakage into a future caller.
    return base.GraphBatch(
        af,rf,pu,hu,pd,hd,
        torch.tensor([atoms[x] for x in (*SUBGOALS,GOAL)]),
        torch.zeros(3,dtype=torch.long),
    )


def observations(cases,states,with_labels=False):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases)!=len(states):
        raise ValueError("AF29 original checkpoint observation roster invalid")
    batch=base.GraphBatch(*(
        torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)
    ))
    bits=torch.zeros(len(cases),ATOMS)
    legal=torch.zeros(len(cases),RULES+1,dtype=torch.bool)
    optimum=legal.clone()
    for i,(c,known) in enumerate(zip(cases,states)):
        perm,_=physical_maps(c)
        for atom in known:
            bits[i,perm[atom]]=1.
        for slot in enabled(c,known):
            legal[i,slot]=True
        legal[i,STOP]=True
        if with_labels:
            for slot in optimal_actions(c,known):
                optimum[i,slot]=True
    return (batch,bits,legal,optimum) if with_labels else (batch,bits,legal)


def training_states(c):
    states=[initial_facts(c)]
    current=states[0]
    for _ in range(2):
        legal=enabled(c,current)
        if not legal:
            break
        current,_=sound_fire(c,current,legal[0])
        if GOAL in current:
            break
        if current not in states:
            states.append(current)
    return tuple(states)


def manifest():
    """Source-only independently checked pair truth and local-structure gate."""
    from experiments.activation_funnel_af28 import gym as af28
    from experiments.activation_funnel_af27 import gym as af27
    old={digest(g.public_features(c)) for g in (af28,af27)
         for split in g.SPLITS for c in g.cases_for(split)}
    unique=set();topologies={}
    by_split={}
    for split in SPLITS:
        cases=cases_for(split)
        if (len(cases),sum(c.reachable for c in cases))!=(192,96):
            raise AssertionError("paired source split balance changed")
        results=[]
        for family in FAMILIES[split]:
            positive=[c for c in cases if c.family==family and c.reachable]
            negative=[c for c in cases if c.family==family and not c.reachable]
            if len(positive)!=48 or len(negative)!=48:
                raise AssertionError("family must include exactly 48 source pairs")
            template=positive[0]
            signature=digest(logical_rules(template))
            if signature in topologies:
                raise AssertionError("train/dev/heldout logical rule template reused")
            topologies[signature]=f"{split}/{family}"
            for pos,neg in zip(positive,negative):
                rec=pair_structural_record(pos,neg)
                if (rec["positive_public_hash"] in unique or
                    rec["negative_public_hash"] in unique or
                    rec["positive_public_hash"] in old or
                    rec["negative_public_hash"] in old):
                    raise AssertionError("duplicate or historical graph identity")
                unique.update((rec["positive_public_hash"],
                               rec["negative_public_hash"]))
                results.append(rec)
        if len(results)!=96:
            raise AssertionError("all source pairs must be independently registered")
        by_split[split]={
            "cases":192,"positive":96,"negative":96,
            "pairs":96,"families":list(FAMILIES[split]),
            "family_count":48,"layouts":sorted({c.layout for c in cases}),
            "source_pair_manifest_sha256":digest(results),
            "source_local_radius2_digest":digest([
                r["local_radius2_signature_sha256"] for r in results
            ]),
            "source_positive_public_graph_sha256":digest([
                r["positive_public_hash"] for r in results
            ]),
            "source_negative_public_graph_sha256":digest([
                r["negative_public_hash"] for r in results
            ]),
            "positive_shortest_histogram":{
                str(d):sum(r["positive_shortest"]==d for r in results)
                for d in sorted(set(r["positive_shortest"] for r in results))
            },
        }
    final={
        "experiment":"AF-29","source_only":True,
        "original_model_checkpoint_opened":False,
        "all_pairs_radius2_local_equal":True,
        "all_pairs_exact_per_node_degrees_equal":True,
        "all_pairs_same_initial_and_marked_role_positions":True,
        "all_pairs_goal_truth_disagrees":True,
        "model_visible_targets_all_zero":True,
        "cross_split_and_historical_graph_disjoint":True,
        "splits":by_split,
    }
    final["manifest_sha256"]=digest(final)
    return final


if __name__=="__main__":
    print(json.dumps(manifest(),indent=2,sort_keys=True))
