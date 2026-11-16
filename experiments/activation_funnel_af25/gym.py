"""AF-25 source-only four-fact Horn adversary.

All cases have EXACTLY FOUR initial known atoms, on both truth labels.
Models retain the original AF-23 15-atom/14-rule public GraphBatch contract.
No original model checkpoint is opened by this file's manifest or tests.

The hard family has a 5/6-action shared-prefix goal proof and a longer
conjunctive goal alternative. Four (depth 5) or three (depth 6) independently
enabled near-goal decoy heads consume the shortest+1 proof slack under a
reverse-hop heuristic, while additive witness costs double-count shared
subproofs. Easy positive controls have an alternate productive shortcut.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, fields
from functools import lru_cache
import hashlib
import json
import random

ATOMS, RULES, GOAL, STOP = 15,14,14,14
SUBGOALS=(6,13)
SEEDS=(17,29,43,61,73,89,103,131)
SPLITS=("held_depth5","held_depth6")
DEPTHS={"held_depth5":5,"held_depth6":6}
FAMILIES=("decoy_and_shared_overcount","balanced_goal_shortcut")
POSITIVE={"decoy_and_shared_overcount":48,"balanced_goal_shortcut":32}
NEGATIVE={"decoy_and_shared_overcount":56,"balanced_goal_shortcut":56}
CHEAP=("ancestor_greedy","closure_direct","bfs_cap4","bfs_cap16")
DIAGNOSTIC=("bfs_cap64","fixed")


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
        return self.variant < POSITIVE[self.family]

    @property
    def id(self):
        return (f"af25-{self.split}-{self.family}-d{self.depth}"
                f"-v{self.variant:03d}-l{self.layout}")


@lru_cache(maxsize=2)
def cases_for(split):
    if split not in SPLITS:
        raise ValueError("unknown AF25 heldout split")
    base=410 if split=="held_depth5" else 420
    return tuple(
        Case(split,fam,variant,base+2*fidx+(variant%2))
        for fidx,fam in enumerate(FAMILIES)
        for variant in range(POSITIVE[fam]+NEGATIVE[fam])
    )


@lru_cache(maxsize=512)
def initial_facts(c):
    if c.designed_positive:
        return frozenset((0,1,2,3))
    # Four facts on both truth classes. Remove jointly needed root atom 2,
    # replace it with a single known decoy atom that cannot finish the goal.
    replacement=(7,8,10,11)[
        (c.variant-POSITIVE[c.family])%4
    ]
    return frozenset((0,1,3,replacement))


@lru_cache(maxsize=512)
def logical_rules(c):
    if c.family not in FAMILIES or c.split not in SPLITS:
        raise ValueError("unregistered AF25 Horn construction")
    if c.depth==5:
        good=[
            ((0,1),4),((4,2),5),((5,),6),((5,),13),((6,13),GOAL),
        ]
        decoy=[
            ((3,),7),((3,),8),((2,),9),((2,),10),
            ((7,8),11),((9,10),12),((11,12),GOAL),
        ]
    else:
        good=[
            ((0,1),4),((4,2),5),((5,),9),
            ((9,),6),((9,),13),((6,13),GOAL),
        ]
        decoy=[
            ((3,),7),((3,),8),((2,),10),
            ((7,8),11),((10,9),12),((11,12),GOAL),
        ]
    if c.family=="balanced_goal_shortcut":
        # The same four initial facts admit an alternate short proof:
        # root decoy 7 + shared prerequisite chain to subgoal 13 + goal.
        decoy[-1]=((7,13),GOAL)
    rules=tuple((tuple(pre),head) for pre,head in good+decoy)
    if len(rules)>RULES or sum(h==GOAL for _,h in rules)!=2:
        raise AssertionError("invalid AF25 two-goal structure")
    if any(not pre or len(pre)>2 or any(x>=ATOMS for x in pre)
           or h in pre or h>=ATOMS for pre,h in rules):
        raise AssertionError("ill-formed Horn rule")
    return rules


@lru_cache(maxsize=512)
def physical_maps(c):
    rng=random.Random(int(digest(
        ["AF25",c.split,c.family,c.variant,c.layout]
    )[:16],16))
    apos=list(range(ATOMS))
    rpos=list(range(RULES))
    rng.shuffle(apos)
    rng.shuffle(rpos)
    if c.family=="decoy_and_shared_overcount":
        # This is a FROZEN adversarial tie breaker, selected before scoring
        # neural models: choose the longer competing goal conclusion on ties.
        rs=logical_rules(c)
        long_index=max(i for i,(_,h) in enumerate(rs) if h==GOAL)
        shared_index=min(i for i,(_,h) in enumerate(rs) if h==GOAL)
        if rpos[long_index] > rpos[shared_index]:
            rpos[long_index],rpos[shared_index]=(
                rpos[shared_index],rpos[long_index]
            )
    return tuple(apos),tuple(rpos)


@lru_cache(maxsize=512)
def physical_rules(c):
    _,rpos=physical_maps(c)
    return {rpos[i]:(pre,head)
            for i,(pre,head) in enumerate(logical_rules(c))}


def enabled(c,known):
    known=set(known)
    return tuple(sorted(
        slot for slot,(pre,head) in physical_rules(c).items()
        if head not in known and all(p in known for p in pre)
    ))


def sound_fire(c,known,slot):
    state=frozenset(known)
    if slot==STOP:
        return state,None
    if slot not in enabled(c,state):
        raise ValueError("host rejected nonenabled physical rule")
    pre,head=physical_rules(c)[slot]
    return state|{head},{"rule_slot":slot,
                          "premises":list(pre),"head":head}


def closure(c,known=None):
    state=set(initial_facts(c) if known is None else known)
    while True:
        future=state|{
            h for pre,h in physical_rules(c).values()
            if all(p in state for p in pre)
        }
        if future==state:
            return frozenset(state)
        state=future


@lru_cache(maxsize=100000)
def shortest_steps(c,known=None):
    source=frozenset(initial_facts(c) if known is None else known)
    if GOAL in source:
        return 0
    queue=deque([(source,0)])
    visited={source}
    while queue:
        state,d=queue.popleft()
        for slot in enabled(c,state):
            next_state,_=sound_fire(c,state,slot)
            if GOAL in next_state:
                return d+1
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state,d+1))
    return None


def independent_bitset_bfs(c,known=None):
    """Independent physical rule-mask BFS without enabled() or sound_fire()."""
    root=initial_facts(c) if known is None else known
    start=sum(1<<atom for atom in root)
    if start & (1<<GOAL):
        return 0
    edges=tuple((
        sum(1<<p for p in pre),1<<head
    ) for pre,head in physical_rules(c).values())
    seen={start}
    todo=deque([(start,0)])
    while todo:
        state,d=todo.popleft()
        for p,h in edges:
            if p & state != p or h & state:
                continue
            nxt=state|h
            if nxt & (1<<GOAL):
                return d+1
            if nxt not in seen:
                seen.add(nxt)
                todo.append((nxt,d+1))
    return None


def optimal_actions(c,state):
    distance=shortest_steps(c,frozenset(state))
    if distance is None or distance==0:
        return (STOP,)
    return tuple(a for a in enabled(c,state)
                 if shortest_steps(c,sound_fire(c,state,a)[0])==distance-1)


def goal_ancestor_atoms(c):
    acc={GOAL}
    for _ in range(ATOMS+1):
        next_acc=acc|{
            x for pre,h in logical_rules(c) if h in acc for x in pre
        }
        if next_acc==acc:
            return frozenset(acc)
        acc=next_acc
    raise ValueError("unsettled ancestor closure")


def public_features(c):
    apos,_=physical_maps(c)
    return {
        "atoms":ATOMS,"rules":RULES,
        "initial_physical":sorted(apos[x] for x in initial_facts(c)),
        "goal_slot":apos[GOAL],
        "subgoal_slots":[apos[x] for x in SUBGOALS],
        "rules_physical":[
            [slot,sorted(apos[p] for p in pre),apos[head]]
            for slot,(pre,head) in sorted(physical_rules(c).items())
        ],
    }


@lru_cache(maxsize=512)
def encode(c):
    """Original AF23 GraphBatch, with no changed model dimensions."""
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if (base.ATOMS,base.RULES)!=(ATOMS,RULES):
        raise AssertionError("source model interface changed")
    apos,_=physical_maps(c)
    af=torch.zeros(ATOMS,5,dtype=torch.float32)
    rf=torch.zeros(RULES,3,dtype=torch.float32)
    pu=torch.zeros(RULES,ATOMS)
    hu=pu.clone()
    pd=pu.clone()
    hd=pu.clone()
    roots=initial_facts(c)
    for atom in range(ATOMS):
        af[apos[atom]]=torch.tensor([
            float(atom in roots),float(atom==SUBGOALS[0]),
            float(atom==SUBGOALS[1]),float(atom==GOAL),1.,
        ])
    for slot,(prem,head) in physical_rules(c).items():
        down=head==GOAL
        rf[slot]=torch.tensor([len(prem)/2.,float(down),1.])
        pp,hh=(pd,hd) if down else (pu,hu)
        for atom in prem:
            pp[slot,apos[atom]]=1.
        hh[slot,apos[head]]=1.
    truth=closure(c)
    return base.GraphBatch(
        af,rf,pu,hu,pd,hd,
        torch.tensor([apos[a] for a in (*SUBGOALS,GOAL)]),
        torch.tensor([int(a in truth) for a in (*SUBGOALS,GOAL)],
                     dtype=torch.long),
    )


def observations(cases,states,with_labels=False):
    """Model sees public physical incidence and legal state only."""
    import torch
    from experiments.activation_funnel_af05 import gym as base
    if not cases or len(cases)!=len(states):
        raise ValueError("nonempty equal-length public observations needed")
    x=base.GraphBatch(*(
        torch.stack([getattr(encode(c),field.name) for c in cases])
        for field in fields(base.GraphBatch)
    ))
    bits=torch.zeros(len(cases),ATOMS)
    legal=torch.zeros(len(cases),RULES+1,dtype=torch.bool)
    optimal=legal.clone()
    for i,(c,known) in enumerate(zip(cases,states)):
        apos,_=physical_maps(c)
        for atom in known:
            bits[i,apos[atom]]=1.
        for a in enabled(c,known):
            legal[i,a]=True
        legal[i,STOP]=True
        if with_labels:
            for a in optimal_actions(c,known):
                optimal[i,a]=True
    return (x,bits,legal,optimal) if with_labels else (x,bits,legal)


def symbolic_completion(c,arm,seed=17):
    """Source-only structural control scores, NEVER model evidence."""
    from experiments.activation_funnel_af24 import symbolic
    graph=symbolic.PublicHorn.from_public_features(public_features(c))
    budget=shortest_steps(c)
    state=graph.initial
    for step in range(budget+1 if budget is not None else 2):
        if graph.goal in state:
            break
        choice=symbolic.choose(graph,state,arm,seed,c.id,step)["action"]
        if choice is None:
            break
        state=graph.fire(state,choice)
    return graph.goal in state


def manifest():
    """Fail-closed structural control gate, evaluated before original checkpoints."""
    from experiments.activation_funnel_af23 import branching_gym as af23
    from experiments.activation_funnel_af24b import gym as af24b
    old={digest(af23.public_features(c))
         for split in af23.SPLITS for c in af23.cases_for(split)}
    old.update(digest(af24b.public_features(c))
               for split in af24b.SPLITS for c in af24b.cases_for(split))
    hashes_seen=set()
    result={}
    for split in SPLITS:
        cases=cases_for(split)
        positives=[c for c in cases if c.designed_positive]
        negatives=[c for c in cases if not c.designed_positive]
        if len(cases)!=192 or len(positives)!=80 or len(negatives)!=112:
            raise AssertionError("pre-registered AF25 roster count drift")
        public_hashes=[digest(public_features(c)) for c in cases]
        if (len(set(public_hashes)) != len(cases) or
                set(public_hashes)&hashes_seen or set(public_hashes)&old):
            raise AssertionError("AF25 public graph leakage or duplicates")
        hashes_seen.update(public_hashes)
        successes={arm:0 for arm in (*CHEAP,*DIAGNOSTIC)}
        hard_success={arm:0 for arm in CHEAP}
        structure={}
        for family in FAMILIES:
            cs=[c for c in cases if c.family==family]
            good=[c for c in cs if c.designed_positive]
            if (len(good),len(cs)-len(good)) != (
                    POSITIVE[family],NEGATIVE[family]):
                raise AssertionError("family roster was altered")
            structure[family]={"cases":len(cs),"provable":len(good),
                               "unprovable":len(cs)-len(good),
                               "root_enabled":Counter(),
                               "positive_rule_count":Counter(),
                               "goal_arity":sorted(
                                   len(pre) for pre,h in logical_rules(good[0])
                                   if h==GOAL
                               )}
        for c in cases:
            roots=initial_facts(c)
            if len(roots)!=4:
                raise AssertionError("AF25 start fact-count must equal AF23")
            exact=shortest_steps(c)
            separate=independent_bitset_bfs(c)
            if exact != separate or (GOAL in closure(c)) != (exact is not None):
                raise AssertionError("physical Horn oracle or closure mismatch")
            if (exact is not None)!=c.designed_positive:
                raise AssertionError("registered positive/negative truth changed")
            if not c.designed_positive:
                continue
            if exact!=c.depth:
                raise AssertionError("registered exact shortest length differs")
            legal=enabled(c,roots)
            distinct=len({sound_fire(c,roots,a)[0] for a in legal})
            if len(legal)<4 or distinct<4:
                raise AssertionError("root proof graph has too little branching")
            optimal=optimal_actions(c,roots)
            if len(optimal)/len(legal)>.75:
                raise AssertionError("root optimal action ratio too high")
            structure[c.family]["root_enabled"][len(legal)]+=1
            structure[c.family]["positive_rule_count"][len(logical_rules(c))]+=1
            if c.family=="decoy_and_shared_overcount":
                ancestors=goal_ancestor_atoms(c)
                bad=sum(
                    physical_rules(c)[a][1] in ancestors and a not in optimal
                    for a in legal
                )
                if bad<2:
                    raise AssertionError("too few misleading short-hop ancestors")
            for arm in successes:
                completed=int(symbolic_completion(c,arm))
                successes[arm]+=completed
                if c.family=="decoy_and_shared_overcount" and arm in hard_success:
                    hard_success[arm]+=completed
        if any(not 20 <= successes[arm] <= 60 for arm in CHEAP):
            raise AssertionError(
                f"registered inexpensive symbolic control structural gate failed: {successes}"
            )
        if any(hard_success[arm] > 24 for arm in CHEAP):
            raise AssertionError(
                f"hard family is too easy for cheap controls: {hard_success}"
            )
        if any(not symbolic_completion(c,"fixed") in (True,False)
               for c in positives):
            raise AssertionError("broken fixed control")
        result[split]={
            "cases":len(cases),"provable":80,"unprovable":112,
            "initial_facts_per_case":4,
            "public_feature_sha256":digest(public_hashes),
            "case_ids_sha256":digest([c.id for c in cases]),
            "layouts":sorted({c.layout for c in cases}),
            "family_structure":{
                f:{**r,"root_enabled":dict(sorted(r["root_enabled"].items())),
                   "positive_rule_count":dict(sorted(r["positive_rule_count"].items()))}
                for f,r in structure.items()
            },
            "source_only_positive_success":successes,
            "hard_family_positive_success":hard_success,
            "oracle_positive_success":80,
        }
    out={
        "experiment":"AF-25","source_only":True,
        "original_model_checkpoint_opened":False,
        "no_cross_parent_or_internal_public_feature_leakage":True,
        "splits":result,
        "structural_gate_pass":True,
    }
    out["manifest_sha256"]=digest(out)
    return out


if __name__=="__main__":
    print(json.dumps(manifest(),indent=2,sort_keys=True))
