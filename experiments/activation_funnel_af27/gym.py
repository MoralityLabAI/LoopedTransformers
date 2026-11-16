"""AF-27 new, frozen, source-only Horn grammar transfer benchmark.

Both provable and unprovable starts contain exactly four known facts. Two
distinct Horn families use new untrained rule-dependency structures with
two alternative goal-concluding Horn rules and shortest positive proof length
of five or six physical rule firings. This module has no Torch dependency
until encode()/observations(), no neural checkpoints and no model selection.
"""
from __future__ import annotations
from collections import deque,Counter
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib,json,random

ATOMS,RULES,GOAL,STOP=15,14,14,14
SUBGOALS=(6,13)
SEEDS=(17,29,43,61,73,89,103,131)
SPLITS=("held_depth5","held_depth6")
FAMILIES=("forked_and","or_gate")
DEPTHS={"held_depth5":5,"held_depth6":6}
PER_FAMILY_POSITIVE=40


def digest(value):
    return hashlib.sha256(json.dumps(
        value,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split:str
    family:str
    variant:int
    layout:int

    @property
    def depth(self):
        return DEPTHS[self.split]

    @property
    def designed_positive(self):
        return self.variant<PER_FAMILY_POSITIVE

    @property
    def id(self):
        return (f"af27-{self.split}-{self.family}-"
                f"d{self.depth}-v{self.variant:02d}-l{self.layout}")


@lru_cache(maxsize=2)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("unknown AF27 heldout split")
    base=510 if split=="held_depth5" else 520
    return tuple(Case(split,fam,v,base+2*ix+v%2)
                 for ix,fam in enumerate(FAMILIES)
                 for v in range(96))


@lru_cache(maxsize=512)
def initial_facts(c):
    if c.designed_positive:
        return frozenset((0,1,2,3))
    # Critical 2 is absent; one decoy replaces it, keeping four facts
    # and making both possible goal-derivations unreachable.
    replacement=(7,8,10,11)[(c.variant-PER_FAMILY_POSITIVE)%4]
    return frozenset((0,1,3,replacement))


@lru_cache(maxsize=512)
def logical_rules(c):
    if c.family not in FAMILIES or c.split not in SPLITS:
        raise ValueError("unregistered AF27 Horn family")
    if c.family=="forked_and" and c.depth==5:
        rules=[
            ((0,1),4),((4,2),5),((5,),6),((5,3),13),
            ((6,13),GOAL),
            ((3,),7),((7,),8),((8,1),9),((9,),10),
            ((10,2),11),((11,2),GOAL),
            ((1,),12),
        ]
    elif c.family=="forked_and" and c.depth==6:
        rules=[
            ((0,1),4),((4,2),5),((5,),9),
            ((9,),6),((9,3),13),((6,13),GOAL),
            ((3,),7),((7,4),8),((8,1),10),((10,),11),
            ((11,2),12),((12,),GOAL),
        ]
    elif c.family=="or_gate" and c.depth==5:
        rules=[
            ((0,2),4),((4,1),5),((5,3),6),
            ((6,),13),((13,),GOAL),
            ((0,1),7),((7,2),8),((8,),9),
            ((9,),10),((10,),11),((11,2),GOAL),
            ((3,),12),
        ]
    else:
        rules=[
            ((0,2),4),((4,1),5),((5,3),6),
            ((6,),12),((12,),13),((13,),GOAL),
            ((0,1),7),((7,3),8),((8,),9),
            ((9,),10),((10,),11),((11,12),GOAL),
        ]
    rules=tuple((tuple(pre),head) for pre,head in rules)
    if len(rules)>RULES or sum(h==GOAL for pre,h in rules)!=2:
        raise AssertionError("AF27 must have at most 14 rules and two goal rules")
    if any(not pre or len(pre)>2 or h in pre or h>=ATOMS or
           any(x>=ATOMS for x in pre) for pre,h in rules):
        raise AssertionError("invalid positive Horn clause")
    return rules


@lru_cache(maxsize=512)
def physical_maps(c):
    rand=random.Random(int(digest([
        "AF27",c.split,c.family,c.variant,c.layout
    ])[:16],16))
    atom=list(range(ATOMS))
    rule=list(range(RULES))
    rand.shuffle(atom)
    rand.shuffle(rule)
    return tuple(atom),tuple(rule)


@lru_cache(maxsize=512)
def physical_rules(c):
    _,rules=physical_maps(c)
    return {rules[i]:item for i,item in enumerate(logical_rules(c))}


def enabled(c,known):
    state=frozenset(known)
    return tuple(sorted(slot for slot,(pre,head) in physical_rules(c).items()
                        if head not in state and all(p in state for p in pre)))


def sound_fire(c,known,slot):
    state=frozenset(known)
    if slot==STOP:
        return state,None
    if slot not in enabled(c,state):
        raise ValueError("physical Horn rule is not currently enabled")
    pre,head=physical_rules(c)[slot]
    return state|{head},{"rule_slot":slot,
                         "premises":list(pre),"head":head}


def closure(c,known=None):
    state=set(initial_facts(c) if known is None else known)
    while True:
        future=state|{head for pre,head in physical_rules(c).values()
                      if all(p in state for p in pre)}
        if future==state:
            return frozenset(state)
        state=future


@lru_cache(maxsize=100000)
def shortest_steps(c,known=None):
    root=frozenset(initial_facts(c) if known is None else known)
    if GOAL in root:
        return 0
    q=deque([(root,0)]);seen={root}
    while q:
        state,depth=q.popleft()
        for action in enabled(c,state):
            next_state,_=sound_fire(c,state,action)
            if GOAL in next_state:
                return depth+1
            if next_state not in seen:
                seen.add(next_state)
                q.append((next_state,depth+1))
    return None


def independent_bfs(c,known=None):
    root=frozenset(initial_facts(c) if known is None else known)
    start=sum(1<<x for x in root)
    if start & (1<<GOAL):
        return 0
    rules=tuple((sum(1<<p for p in pre),1<<head)
                for pre,head in physical_rules(c).values())
    q=deque([(start,0)]);seen={start}
    while q:
        bits,depth=q.popleft()
        for pre,head in rules:
            if pre&bits!=pre or head&bits:
                continue
            nxt=bits|head
            if nxt & (1<<GOAL):
                return depth+1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt,depth+1))
    return None


def optimal_actions(c,known):
    distance=shortest_steps(c,frozenset(known))
    if distance is None or distance==0:
        return (STOP,)
    return tuple(a for a in enabled(c,known)
                 if shortest_steps(c,sound_fire(c,known,a)[0])==distance-1)


def public_features(c):
    apos,_=physical_maps(c)
    return {
        "atoms":ATOMS,"rules":RULES,
        "initial_physical":sorted(apos[a] for a in initial_facts(c)),
        "goal_slot":apos[GOAL],
        "subgoal_slots":[apos[a] for a in SUBGOALS],
        "rules_physical":[
            [slot,sorted(apos[a] for a in pre),apos[head]]
            for slot,(pre,head) in sorted(physical_rules(c).items())
        ],
    }


@lru_cache(maxsize=512)
def encode(c):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if (base.ATOMS,base.RULES)!=(ATOMS,RULES):
        raise ValueError("original model input dimensionality has changed")
    apos,_=physical_maps(c)
    af=torch.zeros(ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(RULES,3,dtype=torch.float32)
    pu=torch.zeros(RULES,ATOMS)
    hu=pu.clone();pd=pu.clone();hd=pu.clone()
    roots=initial_facts(c)
    for atom in range(ATOMS):
        af[apos[atom]]=torch.tensor([
            float(atom in roots),float(atom==SUBGOALS[0]),
            float(atom==SUBGOALS[1]),float(atom==GOAL),1.,
        ])
    for slot,(pre,head) in physical_rules(c).items():
        down=head==GOAL
        rf[slot]=torch.tensor([len(pre)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for atom in pre:
            pp[slot,apos[atom]]=1.
        hh[slot,apos[head]]=1.
    derived=closure(c)
    return base.GraphBatch(
        af,rf,pu,hu,pd,hd,
        torch.tensor([apos[x] for x in (*SUBGOALS,GOAL)]),
        torch.tensor([int(x in derived) for x in (*SUBGOALS,GOAL)],
                     dtype=torch.long),
    )


def observations(cases,states,with_labels=False):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases)!=len(states):
        raise ValueError("invalid AF27 public graph/host state batch")
    x=base.GraphBatch(*(
        torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)
    ))
    bits=torch.zeros(len(cases),ATOMS)
    legal=torch.zeros(len(cases),RULES+1,dtype=torch.bool)
    optimal=legal.clone()
    for i,(c,known) in enumerate(zip(cases,states)):
        apos,_=physical_maps(c)
        for atom in known:
            bits[i,apos[atom]]=1.
        for slot in enabled(c,known):
            legal[i,slot]=True
        legal[i,STOP]=True
        if with_labels:
            for a in optimal_actions(c,known):
                optimal[i,a]=True
    return (x,bits,legal,optimal) if with_labels else (x,bits,legal)


def manifest():
    """Generate and independently validate new graphs WITHOUT model scoring.

    Symbolic baseline behavior is deliberately not a structural selection
    gate in AF27, unlike control-adversarial AF25.
    """
    from experiments.activation_funnel_af25 import gym as af25
    from experiments.activation_funnel_af24b import gym as af24b
    old={digest(af25.public_features(c))
         for split in af25.SPLITS for c in af25.cases_for(split)}
    old.update(digest(af24b.public_features(c))
               for split in af24b.SPLITS for c in af24b.cases_for(split))
    seen=set()
    result={}
    for split in SPLITS:
        cases=cases_for(split)
        pos=[c for c in cases if c.designed_positive]
        neg=[c for c in cases if not c.designed_positive]
        if (len(cases),len(pos),len(neg))!=(192,80,112):
            raise AssertionError("registered case balance changed")
        sigs=[digest(public_features(c)) for c in cases]
        if len(set(sigs))!=len(sigs) or set(sigs)&seen or set(sigs)&old:
            raise AssertionError("cross-original or within-current graph features overlap")
        seen.update(sigs)
        by_family={}
        for family in FAMILIES:
            cs=[c for c in cases if c.family==family]
            if (len(cs),sum(c.designed_positive for c in cs))!=(96,40):
                raise AssertionError("family roster changed")
            by_family[family]={
                "case_count":96,"positive_count":40,
                "goal_rule_arities":sorted(len(pre) for pre,h
                    in logical_rules(cs[0]) if h==GOAL),
                "root_legal_positive":Counter(),
                "root_suboptimal_positive":0,
            }
        for c in cases:
            root=initial_facts(c)
            if len(root)!=4:
                raise AssertionError("AF27 requires four initial known facts")
            d=shortest_steps(c)
            independent=independent_bfs(c)
            if d!=independent or (GOAL in closure(c))!=(d is not None):
                raise AssertionError("independent bitset Horn BFS differs from oracle")
            if (d is not None)!=c.designed_positive:
                raise AssertionError("truth balance does not match registration")
            if not c.designed_positive:
                continue
            if d!=c.depth:
                raise AssertionError("AF27 shortest proof has wrong depth")
            legal=enabled(c,root)
            distinct=len({sound_fire(c,root,a)[0] for a in legal})
            if len(legal)<2 or distinct<2:
                raise AssertionError("too little real root branching")
            opt=optimal_actions(c,root)
            record=by_family[c.family]
            record["root_legal_positive"][len(legal)]+=1
            record["root_suboptimal_positive"]+=int(len(opt)<len(legal))
        for family,record in by_family.items():
            if record["root_suboptimal_positive"]<20:
                raise AssertionError("insufficient cases with nonoptimal root action")
            record["root_legal_positive"]=dict(sorted(record["root_legal_positive"].items()))
        result[split]={
            "cases":192,"provable":80,"unprovable":112,
            "public_features_sha256":digest(sigs),
            "case_ids_sha256":digest([c.id for c in cases]),
            "layouts":sorted({c.layout for c in cases}),
            "family_statistics":by_family,
            "root_fact_count":4,
        }
    out={
        "experiment":"AF-27","source_only":True,
        "original_model_opened":False,
        "original_parent_public_features_overlap":False,
        "splits":result,
    }
    out["manifest_sha256"]=digest(out)
    return out


if __name__=="__main__":
    print(json.dumps(manifest(),indent=2,sort_keys=True))
