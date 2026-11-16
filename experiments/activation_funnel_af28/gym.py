"""AF-28 preregistered source-only four-way Horn grammar benchmark.

Train, development, and heldout split families have separate logical rule
incidence templates. Only source-owned public graphs and host-known facts are
model inputs. Exact BFS and closure are used solely in source manifest,
supervised train/dev labels and heldout scoring budgets, never at inference.
"""
from __future__ import annotations
from collections import deque,Counter
from dataclasses import dataclass,fields
from functools import lru_cache
import hashlib,json,random

ATOMS,RULES,GOAL,STOP=15,14,14,14
SUBGOALS=(6,13)
SEEDS=(17,29,43,61,73,89,103,131)
SPLITS=("train","dev","held_depth5","held_depth6")
TEMPLATES={
    "train":("train_shared_gate","train_dual_or"),
    "dev":("dev_crossed_and","dev_short_or"),
    "held_depth5":("held_forked_and","held_or_gate"),
    "held_depth6":("held_forked_and","held_or_gate"),
}
DEPTH={
    ("train","train_shared_gate"):4,
    ("train","train_dual_or"):5,
    ("dev","dev_crossed_and"):5,
    ("dev","dev_short_or"):4,
    ("held_depth5","held_forked_and"):5,
    ("held_depth5","held_or_gate"):5,
    ("held_depth6","held_forked_and"):6,
    ("held_depth6","held_or_gate"):6,
}
LAYOUT_BASE={"train":610,"dev":620,"held_depth5":630,"held_depth6":640}


def digest(x):
    return hashlib.sha256(json.dumps(
        x,sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()


@dataclass(frozen=True)
class Case:
    split:str
    family:str
    variant:int
    layout:int
    @property
    def id(self):
        return f"af28-{self.split}-{self.family}-v{self.variant:02d}-l{self.layout}"
    @property
    def depth(self):
        return DEPTH[(self.split,self.family)]
    @property
    def designed_positive(self):
        return self.variant<40


@lru_cache(maxsize=4)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("unregistered AF28 split")
    return tuple(
        Case(split,fam,v,LAYOUT_BASE[split]+2*i+v%2)
        for i,fam in enumerate(TEMPLATES[split])
        for v in range(96)
    )


@lru_cache(maxsize=1024)
def initial_facts(c):
    if c.designed_positive:
        return frozenset((0,1,2,3))
    replacement=(7,8,10,11)[(c.variant-40)%4] if (c.family=='held_forked_and' and c.depth==6) else (8,9,10,11)[(c.variant-40)%4]
    return frozenset((0,1,3,replacement))


@lru_cache(maxsize=1024)
def logical_rules(c):
    """Six source-registered productive/competing Horn motifs.

    All alternative goal routes explicitly require root fact 2 to ensure
    four-fact negative controls are unreachable. No two split templates are
    the same logical incidence graph.
    """
    kind=c.family;d=c.depth
    if kind=="train_shared_gate":
        rules=[
            ((0,1),4),((4,2),5),((5,3),6),((6,),GOAL),
            ((3,2),7),((7,),8),((8,),9),((9,),10),((10,2),GOAL),
            ((1,),11),((11,),12),
        ]
    elif kind=="train_dual_or":
        rules=[
            ((0,2),4),((4,3),5),((5,1),6),((6,),13),((13,),GOAL),
            ((1,3),7),((7,2),8),((8,),9),((9,),10),((10,),11),((11,2),GOAL),
        ]
    elif kind=="dev_crossed_and":
        rules=[
            ((0,3),4),((4,2),5),((5,),6),((5,1),13),((6,13),GOAL),
            ((2,3),7),((7,),8),((8,1),9),((9,),10),((10,),11),((11,2),GOAL),
        ]
    elif kind=="dev_short_or":
        rules=[
            ((0,2),4),((4,1),6),((6,3),13),((13,),GOAL),
            ((1,2),7),((7,3),8),((8,),9),((9,),10),((10,2),GOAL),
            ((3,),12),
        ]
    elif kind=="held_forked_and" and d==5:
        rules=[
            ((0,1),4),((4,2),5),((5,),6),((5,3),13),((6,13),GOAL),
            ((3,),7),((7,2),8),((8,),9),((9,),10),((10,),11),((11,2),GOAL),
        ]
    elif kind=="held_forked_and" and d==6:
        rules=[
            ((0,1),4),((4,2),5),((5,),9),((9,),6),((9,3),13),((6,13),GOAL),
            ((3,),7),((7,2),8),((8,),10),((10,),11),((11,),12),((12,2),GOAL),
        ]
    elif kind=="held_or_gate" and d==5:
        rules=[
            ((0,2),4),((4,1),5),((5,3),6),((6,),13),((13,),GOAL),
            ((1,3),7),((7,2),8),((8,),9),((9,),10),((10,),11),((11,2),GOAL),
        ]
    elif kind=="held_or_gate" and d==6:
        rules=[
            ((0,2),4),((4,1),5),((5,3),6),((6,),12),((12,),13),((13,),GOAL),
            ((1,3),7),((7,2),8),((8,),9),((9,),10),((10,),11),((11,2),GOAL),
        ]
    else:
        raise ValueError("wrong stage/template/depth")
    rules=tuple((tuple(p),h) for p,h in rules)
    if len(rules)>RULES or sum(h==GOAL for p,h in rules)!=2:
        raise AssertionError("source grammar must have two goal-concluding rules")
    if any(not p or len(p)>2 or h in p or h>=ATOMS or
           any(a>=ATOMS for a in p) for p,h in rules):
        raise AssertionError("unwell-formed positive Horn template")
    return rules


@lru_cache(maxsize=1024)
def physical_maps(c):
    rand=random.Random(int(digest(["AF28",c.id])[:16],16))
    atoms=list(range(ATOMS))
    rules=list(range(RULES))
    rand.shuffle(atoms);rand.shuffle(rules)
    return tuple(atoms),tuple(rules)


@lru_cache(maxsize=1024)
def physical_rules(c):
    _,perm=physical_maps(c)
    return {perm[i]:(prem,head)
            for i,(prem,head) in enumerate(logical_rules(c))}


def enabled(c,state):
    known=set(state)
    return tuple(sorted(
        slot for slot,(prem,head) in physical_rules(c).items()
        if head not in known and all(p in known for p in prem)
    ))


def sound_fire(c,state,slot):
    known=frozenset(state)
    if slot==STOP:
        return known,None
    if slot not in enabled(c,known):
        raise ValueError("host refused nonenabled Horn rule")
    prem,head=physical_rules(c)[slot]
    return known|{head},{"rule_slot":slot,"premises":list(prem),"head":head}


def closure(c,known=None):
    state=set(initial_facts(c) if known is None else known)
    while True:
        next_state=state|{h for prem,h in physical_rules(c).values()
                          if all(p in state for p in prem)}
        if next_state==state:
            return frozenset(state)
        state=next_state


@lru_cache(maxsize=100000)
def shortest_steps(c,known=None):
    root=frozenset(initial_facts(c) if known is None else known)
    if GOAL in root:return 0
    todo=deque([(root,0)])
    seen={root}
    while todo:
        state,d=todo.popleft()
        for slot in enabled(c,state):
            nxt,_=sound_fire(c,state,slot)
            if GOAL in nxt:return d+1
            if nxt not in seen:
                seen.add(nxt);todo.append((nxt,d+1))
    return None


def independent_bfs(c,known=None):
    start=sum(1<<p for p in (
        initial_facts(c) if known is None else known))
    if start&(1<<GOAL):return 0
    masks=tuple((sum(1<<p for p in prem),1<<head)
                for prem,head in physical_rules(c).values())
    seen={start};todo=deque([(start,0)])
    while todo:
        bits,d=todo.popleft()
        for pre,head in masks:
            if pre&bits!=pre or head&bits:
                continue
            nxt=bits|head
            if nxt&(1<<GOAL):return d+1
            if nxt not in seen:
                seen.add(nxt);todo.append((nxt,d+1))
    return None


def optimal_actions(c,known):
    dist=shortest_steps(c,frozenset(known))
    if dist is None or dist==0:
        return (STOP,)
    return tuple(a for a in enabled(c,known)
                 if shortest_steps(c,sound_fire(c,known,a)[0])==dist-1)


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


@lru_cache(maxsize=1024)
def encode(c):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if (base.ATOMS,base.RULES)!=(ATOMS,RULES):
        raise AssertionError("AF28 checkpoint network interface changed")
    apos,_=physical_maps(c)
    af=torch.zeros(ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(RULES,3,dtype=torch.float32)
    pu=torch.zeros(RULES,ATOMS)
    hu=pu.clone();pd=pu.clone();hd=pu.clone()
    roots=initial_facts(c)
    for a in range(ATOMS):
        af[apos[a]]=torch.tensor([
            float(a in roots),float(a==SUBGOALS[0]),
            float(a==SUBGOALS[1]),float(a==GOAL),1.,
        ])
    for slot,(prem,head) in physical_rules(c).items():
        down=head==GOAL
        rf[slot]=torch.tensor([len(prem)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for a in prem:pp[slot,apos[a]]=1.
        hh[slot,apos[head]]=1.
    derived=closure(c)
    return base.GraphBatch(
        af,rf,pu,hu,pd,hd,
        torch.tensor([apos[a] for a in (*SUBGOALS,GOAL)]),
        torch.tensor([int(a in derived) for a in (*SUBGOALS,GOAL)],
                     dtype=torch.long)
    )


def observations(cases,states,with_labels=False):
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases)!=len(states):
        raise ValueError("AF28 observation batch must be nonempty and paired")
    x=base.GraphBatch(*(
        torch.stack([getattr(encode(c),f.name) for c in cases])
        for f in fields(base.GraphBatch)
    ))
    bits=torch.zeros(len(cases),ATOMS)
    legal=torch.zeros(len(cases),RULES+1,dtype=torch.bool)
    optimal=legal.clone()
    for i,(c,known) in enumerate(zip(cases,states)):
        apos,_=physical_maps(c)
        for a in known:bits[i,apos[a]]=1.
        for slot in enabled(c,known):legal[i,slot]=True
        legal[i,STOP]=True
        if with_labels:
            for slot in optimal_actions(c,known):
                optimal[i,slot]=True
    return (x,bits,legal,optimal) if with_labels else (x,bits,legal)


def training_states(c):
    """Source-independent train/dev state sampling. No model outputs."""
    states=[initial_facts(c)]
    current=states[0]
    for _ in range(2):
        options=enabled(c,current)
        if not options:break
        current,_=sound_fire(c,current,options[0])
        if GOAL in current:break
        if current not in states:states.append(current)
    return tuple(states)


def manifest():
    """Pretraining structural receipt; no neural model features or outcomes."""
    old=set()
    from experiments.activation_funnel_af27 import gym as af27
    from experiments.activation_funnel_af25 import gym as af25
    for g in (af27,af25):
        for split in g.SPLITS:
            old.update(digest(g.public_features(c)) for c in g.cases_for(split))
    seen=set();rows={};topology={}
    for split in SPLITS:
        cases=cases_for(split)
        positives=[c for c in cases if c.designed_positive]
        negatives=[c for c in cases if not c.designed_positive]
        if (len(cases),len(positives),len(negatives))!=(192,80,112):
            raise AssertionError("source split balance changed")
        hashes=[digest(public_features(c)) for c in cases]
        if len(set(hashes))!=len(hashes) or set(hashes)&seen or set(hashes)&old:
            raise AssertionError("public graph reuse within/from earlier tests")
        seen.update(hashes)
        fam_stats={}
        for family in TEMPLATES[split]:
            subset=[c for c in cases if c.family==family]
            if (len(subset),sum(c.designed_positive for c in subset))!=(96,40):
                raise AssertionError("source family cohort balance changed")
            example=subset[0]
            signature=digest(logical_rules(example))
            if signature in topology:
                # It is permitted to repeat a *held* family in heldout depth6
                # only, with different rule and depth structure, so the
                # resulting exact incidence signature must still be unique.
                raise AssertionError("same logical grammar reused across source splits")
            topology[signature]=split+"/"+family
            fam_stats[family]={
                "logical_rules_sha256":signature,
                "shortest_positive_expected":example.depth,
                "goal_arity":sorted(len(pre) for pre,h in logical_rules(example)
                                    if h==GOAL),
                "positive_roots_with_suboptimal_actions":0,
                "positive_root_enabled":Counter(),
            }
        for c in cases:
            if len(initial_facts(c))!=4:
                raise AssertionError("exact four-root-fact requirement failed")
            if not enabled(c,initial_facts(c)):
                raise AssertionError(f"root STOP-versus-rule action is absent: {c.id}")
            d=shortest_steps(c)
            if d!=independent_bfs(c) or (GOAL in closure(c))!=(d is not None):
                raise AssertionError("independent bitset host shortest proof mismatch")
            if (d is not None)!=c.designed_positive:
                raise AssertionError(f"source truth balance mismatch: {c.id} expected_positive={c.designed_positive} bfs={d} roots={sorted(initial_facts(c))}")
            if not c.designed_positive:
                continue
            if d!=c.depth:
                raise AssertionError("minimum source proof length wrong")
            legal=enabled(c,initial_facts(c))
            if len(legal)<2 or len({sound_fire(c,initial_facts(c),a)[0]
                                   for a in legal})<2:
                raise AssertionError("source Horn graph has insufficient root branching")
            opt=optimal_actions(c,initial_facts(c))
            stat=fam_stats[c.family]
            stat["positive_root_enabled"][len(legal)]+=1
            stat["positive_roots_with_suboptimal_actions"]+=int(len(opt)<len(legal))
        for stat in fam_stats.values():
            stat["positive_root_enabled"]=dict(sorted(stat["positive_root_enabled"].items()))
        rows[split]={
            "cases":len(cases),"positive":len(positives),"negative":len(negatives),
            "layout_ids":sorted({c.layout for c in cases}),
            "public_graphs_sha256":digest(hashes),
            "source_case_ids_sha256":digest([c.id for c in cases]),
            "four_initial_facts":True,
            "source_templates":fam_stats,
        }
    out={
        "experiment":"AF-28","source_only":True,
        "model_checkpoint_opened":False,
        "grammar_disjoint_logical_incidence":True,
        "public_overlap_with_AF25_AF27":False,
        "split_receipts":rows,
    }
    out["manifest_sha256"]=digest(out)
    return out


if __name__=="__main__":
    print(json.dumps(manifest(),indent=2,sort_keys=True))
